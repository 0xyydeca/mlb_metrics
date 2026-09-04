"""Today's probable starting pitchers, and (for Automated Game Picks) final
game results, via the MLB Stats API.

`pybaseball` has no schedule/lineup/probable-pitcher support at all - this
is a genuinely new data source (statsapi.mlb.com, free/unauthenticated, via
the `MLB-StatsAPI` package). Confirmed starting lineups aren't available
here - they post only ~2-4 hours before first pitch, structurally too late
for this pipeline's 8am ET run regardless of data source - only probable
pitchers, which are typically settled days in advance.

This queries the lower-level `statsapi.get("schedule", ...)` call (the raw
MLB Stats API JSON) rather than the `statsapi.schedule()` convenience
wrapper. Confirmed via a live run of `scripts/debug_statsapi.py` (the
`Debug statsapi` GitHub Actions workflow, run 29878447354): the convenience
wrapper's `home_probable_pitcher`/`away_probable_pitcher` fields are name
strings only, with no numeric ID anywhere in that response - the raw
endpoint's `teams.{home,away}.probablePitcher.id` is genuinely required to
get the same ID space as Statcast's batter/pitcher columns. The field paths
parsed below (`teams.{home,away}.team.id`, `.probablePitcher.id`) match the
live response exactly. That same live response also confirmed
`status.detailedState` and `teams.{home,away}.score` as top-level fields on
every game (no extra hydrate needed) - but only for an in-progress game, not
a completed one, so `normalize_schedule_games`'s "Final" status handling is
a reasonable extrapolation, not itself live-confirmed. Verify it the same
way (a `Debug statsapi` run against a known past date) before trusting
Automated Game Picks' resolution in production.

`normalize_schedule`'s `game_datetime` field (each game's real start time,
for the dashboard's slate-filtering feature - see dfs_optimizer.py's
module docstring) is confirmed via a live run of `scripts/debug_statsapi.py`
(the `Debug statsapi` GitHub Actions workflow, run 30661418977, 2026-07-31):
the raw game dict's `gameDate` field ("2026-07-31T18:20:00Z") is a bare,
unconditional sibling key of `gamePk`/`teams`/`status` - no extra
`hydrate` parameter needed. Still defensively `.get()` rather than
subscripted (returns `None`, never raises, if a future response ever
omits it).
"""

import datetime
from zoneinfo import ZoneInfo

import pandas as pd

# The dashboard's audience is in Arizona, which does not observe DST (fixed
# UTC-7 year-round) - "today" must be Arizona's calendar day, not the
# running machine's own local time (UTC on GitHub Actions runners). A
# workflow_dispatch run late in the evening AZ time (e.g. 8:45pm AZ =
# 3:45am UTC, already past UTC midnight) would otherwise compute "today" as
# tomorrow by naive datetime.date.today(), publishing a day-ahead slate.
LOCAL_TIMEZONE = ZoneInfo("America/Phoenix")


def today_local() -> datetime.date:
    """Today's calendar date in Arizona time - see LOCAL_TIMEZONE above for
    why this isn't plain datetime.date.today()."""
    return datetime.datetime.now(LOCAL_TIMEZONE).date()


# Ported from docs/app.js's teamLogo() table (already validated against
# Statcast's own home_team/away_team abbreviations) and inverted. Joining on
# MLB's numeric team ID rather than trying to string-match abbreviations is
# deliberate - MLB Stats API and Baseball Savant are known to disagree on
# some codes (AZ/ARI, CWS/CHW, SD/SDP, SF/SFG, TB/TBR, KC/KCR, WSH/WSN).
TEAM_ID_TO_ABBREV = {
    109: "AZ", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CWS", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KC", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "ATH",
    143: "PHI", 134: "PIT", 135: "SD", 137: "SF", 136: "SEA",
    138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}


def normalize_schedule(raw: dict, fallback_date=None) -> pd.DataFrame:
    """Parse a raw `statsapi.get("schedule", ...)` response into one row per
    team per game: [date, team, opponent, probable_pitcher_key_mlbam,
    game_pk, game_datetime, is_home]. `probable_pitcher_key_mlbam` is null
    if not yet announced. `game_datetime` is the raw `gameDate` ISO-8601
    string as-is (NOT parsed into a pd.Timestamp - a tz-aware Timestamp
    round-trips through CSV as "2026-07-31 23:05:00+00:00", which isn't
    reliably parseable by JS `new Date()`, while the raw API string is) -
    see this module's docstring for why this field isn't yet live-confirmed
    like the others here, and null if absent, never raising. Games for a
    team whose numeric ID isn't in TEAM_ID_TO_ABBREV are skipped rather
    than guessed at.

    V1 simplification for dashboard/probable-pitcher slate views: a
    doubleheader only keeps a team's first game of the day, not both.
    Hitter matchup / feature / prediction paths must NOT use this shape -
    use `normalize_hitter_schedule` / `fetch_hitter_schedule`, which keep
    every game so game-one features are never attached to a game-two label.
    For a game-level (not per-team) view that also keeps every game, see
    `normalize_schedule_games`.
    """
    return _normalize_schedule_team_rows(raw, fallback_date=fallback_date, keep_doubleheaders=False)


def normalize_hitter_schedule(raw: dict, fallback_date=None) -> pd.DataFrame:
    """One row per team per game for hitter modeling - SAME columns as
    `normalize_schedule`, but BOTH halves of a doubleheader are kept.

    This is the schedule shape matchup / dfs_ml / predictions / hit-log
    assembly must use. Joining on `team` alone against a multi-game day
    is then intentional (one batter-game row per real contest), keyed by
    `game_pk`, rather than the old first-game-only simplification that
    silently dropped game two.
    """
    return _normalize_schedule_team_rows(raw, fallback_date=fallback_date, keep_doubleheaders=True)


def _normalize_schedule_team_rows(raw: dict, fallback_date=None, keep_doubleheaders: bool = False) -> pd.DataFrame:
    """Shared parser for the per-team schedule shapes. When
    `keep_doubleheaders` is False, the first game seen for each team wins
    (legacy normalize_schedule behavior)."""
    rows = []
    seen_teams = set()

    for date_entry in raw.get("dates", []):
        date = date_entry.get("date", fallback_date)
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_abbrev = TEAM_ID_TO_ABBREV.get((home.get("team") or {}).get("id"))
            away_abbrev = TEAM_ID_TO_ABBREV.get((away.get("team") or {}).get("id"))
            if home_abbrev is None or away_abbrev is None:
                continue

            for team_abbrev, opponent_abbrev, side, is_home in (
                (home_abbrev, away_abbrev, home, True),
                (away_abbrev, home_abbrev, away, False),
            ):
                if not keep_doubleheaders and team_abbrev in seen_teams:
                    continue
                seen_teams.add(team_abbrev)
                probable = side.get("probablePitcher") or {}
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "team": team_abbrev,
                        "opponent": opponent_abbrev,
                        "probable_pitcher_key_mlbam": probable.get("id"),
                        "game_pk": game.get("gamePk"),
                        "game_datetime": game.get("gameDate"),
                        "is_home": is_home,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "date", "team", "opponent", "probable_pitcher_key_mlbam", "game_pk", "game_datetime", "is_home",
        ],
    )


def normalize_schedule_games(raw: dict, fallback_date=None) -> pd.DataFrame:
    """Parse a raw `statsapi.get("schedule", ...)` response into one row per
    GAME (not per team, unlike normalize_schedule): [game_pk, date,
    home_team, away_team, home_probable_pitcher_key_mlbam,
    away_probable_pitcher_key_mlbam, status, home_score, away_score,
    game_datetime]. `status` is the raw `detailedState` string (e.g.
    "Scheduled", "In Progress", "Final") - callers should treat anything
    other than "Final" as not-yet-resolvable rather than special-casing
    every possible in-between state. `home_score`/`away_score` are null
    until the game starts posting a score. `game_datetime` is the raw
    `gameDate` ISO-8601 string (UTC) used for closing-line and odds
    matching.

    Unlike normalize_schedule, doubleheaders are NOT deduped - dropping game
    2 here would silently lose an entire game's pick/resolution, not just
    reuse a slightly-stale metric like the team-quality join can tolerate.
    Both games of a doubleheader will therefore get an identical predicted
    win probability (the team composite doesn't change between games) -
    a known first-pass simplification, not fixed here.
    """
    rows = []
    for date_entry in raw.get("dates", []):
        date = date_entry.get("date", fallback_date)
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_abbrev = TEAM_ID_TO_ABBREV.get((home.get("team") or {}).get("id"))
            away_abbrev = TEAM_ID_TO_ABBREV.get((away.get("team") or {}).get("id"))
            if home_abbrev is None or away_abbrev is None:
                continue

            home_probable = home.get("probablePitcher") or {}
            away_probable = away.get("probablePitcher") or {}
            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "date": pd.Timestamp(date),
                    "home_team": home_abbrev,
                    "away_team": away_abbrev,
                    "home_probable_pitcher_key_mlbam": home_probable.get("id"),
                    "away_probable_pitcher_key_mlbam": away_probable.get("id"),
                    "status": (game.get("status") or {}).get("detailedState"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "game_datetime": game.get("gameDate"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "game_pk", "date", "home_team", "away_team",
            "home_probable_pitcher_key_mlbam", "away_probable_pitcher_key_mlbam",
            "status", "home_score", "away_score", "game_datetime",
        ],
    )


def _fetch_raw_schedule(date, hydrate: str = "probablePitcher,team") -> dict:
    """Thin wrapper around the raw `statsapi.get("schedule", ...)` call,
    shared by every fetch function below so there's one place to monkeypatch
    in tests and one place documenting the data source."""
    import statsapi

    return statsapi.get("schedule", {"sportId": 1, "date": str(date), "hydrate": hydrate})


def fetch_probable_pitchers(date) -> pd.DataFrame:
    """Today's games and probable starting pitchers, one row per team
    (first game of a doubleheader only - see normalize_schedule). Use
    `fetch_hitter_schedule` for hitter matchup / feature paths that must
    keep both DH games."""
    return normalize_schedule(_fetch_raw_schedule(date), fallback_date=date)


def fetch_hitter_schedule(date) -> pd.DataFrame:
    """Today's per-team schedule for hitter modeling - one row per team per
    game, including both halves of a doubleheader (see
    normalize_hitter_schedule)."""
    return normalize_hitter_schedule(_fetch_raw_schedule(date), fallback_date=date)


def fetch_todays_games(date) -> pd.DataFrame:
    """Today's games, one row per game - see normalize_schedule_games."""
    return normalize_schedule_games(_fetch_raw_schedule(date), fallback_date=date)


def fetch_game_results(date) -> pd.DataFrame:
    """Games and their final scores for a PAST `date` - same endpoint and
    shape as fetch_todays_games, called backward instead of forward.
    `status`/`home_score`/`away_score` should be populated once each game
    is Final; anything still "Scheduled"/"In Progress"/postponed is not yet
    resolvable and callers should leave it pending rather than guess."""
    return normalize_schedule_games(_fetch_raw_schedule(date), fallback_date=date)


def build_probable_pitchers_table(schedule_df: pd.DataFrame, pave: pd.DataFrame) -> pd.DataFrame:
    """One row per team playing today (from `schedule_df`, see
    normalize_schedule/fetch_probable_pitchers), left-joined with that
    team's probable starter's pitching metrics from `pave` (see
    pitchers.assemble_pitchers). Left join, not inner - a team with an
    unannounced or unmatched starter still gets a row (with blank pitcher
    fields) so the day's full slate stays visible, rather than silently
    disappearing."""
    pitcher_cols = [c for c in ("Throws", "PAVE", "PAVE_PLUS", "Power_A_PLUS") if c in pave.columns]
    starters = pave[["key_mlbam", "name_first", "name_last"] + pitcher_cols]

    table = schedule_df[["date", "team", "opponent", "is_home", "probable_pitcher_key_mlbam"]].merge(
        starters, left_on="probable_pitcher_key_mlbam", right_on="key_mlbam", how="left"
    )
    table["pitcher_name"] = (
        table["name_first"].fillna("") + " " + table["name_last"].fillna("")
    ).str.strip()

    return table[["date", "team", "opponent", "is_home", "pitcher_name"] + pitcher_cols]
