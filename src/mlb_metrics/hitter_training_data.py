"""No-lookahead historical hitter-opportunity dataset: one row per
pregame candidate per (date, game_pk).

This is a training-table builder, not a live model. It does not change
pipeline.run(), predictions.select_picks, or any served artifact.

## Candidate universe (no lookahead)

For every historical team-game, candidates are constructed from Statcast
strictly before that game (morning snapshot: `game_date < game date`). A
batter is a candidate for that team-game iff:

1. their latest known team *before* the game is the team playing, and
2. they appeared for that team within the previous
   `config.HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES` of that team's games
   **or** within `config.HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS`
   calendar days.

The game's actual lineup is never consulted when deciding who was a
candidate. It is labels only (`Started`, `Appeared`, `Batting_Order`,
plate appearances, hits). Players who were candidates and did not appear
are kept as negative examples for an appearance model.

A future *live* confirmed-lineup snapshot may add a previously unseen
player to the candidate pool via `add_confirmed_lineup_candidates` (league
priors for features). Historical training never calls that path, and
must not silently use same-day outcomes to improve candidate coverage.

## Features vs labels

Pregame features reuse `dfs_ml.build_hitter_features` plus historical
lineup-consistency (`avg_batting_order`, `start_rate`) and recency
(`Last_Game_Date`, `Days_Rest` from history). No column is calculated
from the target game's own events. `Batting_Order` is a label (null when
the candidate did not start), never a feature.

## Same-date doubleheaders

Game identity is `game_pk`. The default morning snapshot uses
`game_date < date` for *every* contest that calendar date, so game-one
results do not leak into game-two features or into game-two's candidate
pool. A later snapshot (`prediction_timestamp_utc`) may include a
same-date earlier game only when that game carries a real
`game_completed_at` strictly before the timestamp. Completion times are
never inferred.
"""

from __future__ import annotations

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, helpers, lineup, matchup, pipeline

OPPORTUNITY_KEY_COLUMNS = ["date", "game_pk", "key_mlbam"]

CANDIDATE_SOURCE_PREGAME_LOOKBACK = "pregame_roster_lookback"
CANDIDATE_SOURCE_CONFIRMED_LINEUP = "confirmed_lineup"

GAME_COMPLETED_AT_COLUMN = "game_completed_at"

OPPORTUNITY_LABEL_COLUMNS = [
    "Started",
    "Appeared",
    "Batting_Order",
    "Plate_Appearances",
    "Official_At_Bats",
    "Got_Hit",
    "No_Game",
    "Hits",
]

# Historical batting-order mean / start rate / last appearance - from
# history only, never the target game's actual slot.
OPPORTUNITY_EXTRA_FEATURE_COLUMNS = [
    "avg_batting_order",
    "start_rate",
    "Last_Game_Date",
    "Days_Rest",
]

OPPORTUNITY_FEATURE_COLUMNS = list(dfs_ml.HITTER_FEATURE_COLUMNS) + OPPORTUNITY_EXTRA_FEATURE_COLUMNS

COVERAGE_COLUMNS = [
    "date",
    "game_pk",
    "team",
    "n_candidates",
    "n_actual_starters",
    "n_starters_in_candidates",
    "starter_coverage",
    "n_actual_appeared",
    "n_appeared_in_candidates",
    "appearance_coverage",
    "n_omitted_appeared",
    "n_omitted_no_pregame_history",
]


def morning_feature_as_of_timestamp(as_of_date) -> pd.Timestamp:
    """UTC midnight of the game date: nothing from that calendar date is in
    the morning snapshot's feature history."""
    ts = pd.Timestamp(as_of_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    else:
        ts = ts.tz_localize("UTC")
    return ts.normalize()


def _normalize_date(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=False)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed.dt.normalize()


def assert_unique_opportunity_keys(df: pd.DataFrame) -> None:
    """Hard fail if (date, game_pk, key_mlbam) is not unique."""
    if df.empty:
        return
    missing = [c for c in OPPORTUNITY_KEY_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(f"opportunity log missing key columns {missing}")
    dupes = df.duplicated(subset=OPPORTUNITY_KEY_COLUMNS, keep=False)
    if dupes.any():
        sample = df.loc[dupes, OPPORTUNITY_KEY_COLUMNS].head(10)
        raise AssertionError(
            "hitter opportunity log has duplicate (date, game_pk, key_mlbam) "
            f"rows; sample:\n{sample.to_string(index=False)}"
        )


def slice_history_before_game(
    persisted: pd.DataFrame,
    as_of_date,
    *,
    target_game_pk=None,
    prediction_timestamp_utc=None,
) -> pd.DataFrame:
    """Statcast rows strictly available when computing pregame features.

    Morning snapshot (`prediction_timestamp_utc` is None): `game_date <
    as_of_date`. Same-date doubleheader game one is excluded from game
    two, even if a `game_completed_at` column is present - completion
    times are ignored unless a caller opts into a later snapshot.

    Later snapshot: rows from an earlier same-date `game_pk` are included
    only when `game_completed_at` is a real timestamp strictly before
    `prediction_timestamp_utc`. Target-game rows are never included.
    Missing/unparseable completion times do not count as completed.
    """
    if persisted.empty:
        return persisted.copy()

    as_of = _normalize_date(as_of_date)
    dates = _date_series(persisted["game_date"])
    prior = persisted.loc[dates < as_of].copy()

    exclude_target = persisted["game_pk"].ne(target_game_pk) if target_game_pk is not None else True

    if prediction_timestamp_utc is None:
        if target_game_pk is None:
            return prior
        return prior.loc[prior["game_pk"].ne(target_game_pk)].copy()

    same_day = persisted.loc[(dates == as_of) & exclude_target].copy()
    if same_day.empty or GAME_COMPLETED_AT_COLUMN not in same_day.columns:
        return prior

    pred = pd.Timestamp(prediction_timestamp_utc)
    if pred.tzinfo is None:
        pred = pred.tz_localize("UTC")
    else:
        pred = pred.tz_convert("UTC")

    completed_at = pd.to_datetime(same_day[GAME_COMPLETED_AT_COLUMN], utc=True, errors="coerce")
    usable = same_day.loc[completed_at.notna() & (completed_at < pred)]
    if usable.empty:
        return prior
    return pd.concat([prior, usable], ignore_index=True)


def batter_team_appearances(statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter) with the batting team, from completed
    PAs only. Used for latest-team and lookback; never from a future game.
    """
    needed = [
        "game_date", "game_pk", "batter", "inning_topbot",
        "home_team", "away_team", "at_bat_number",
    ]
    empty = pd.DataFrame(columns=["game_date", "game_pk", "batter", "team", "at_bat_number"])
    if statcast.empty or any(c not in statcast.columns for c in needed):
        return empty

    completed = data.completed_events(statcast, needed)
    if completed.empty:
        return empty

    completed = completed.copy()
    completed["team"] = completed["away_team"].where(
        completed["inning_topbot"] == "Top", completed["home_team"]
    )
    completed["game_date"] = pd.to_datetime(completed["game_date"])
    first = (
        completed.sort_values(["game_date", "game_pk", "at_bat_number"])
        .drop_duplicates(["game_pk", "batter"], keep="first")
    )
    return first[["game_date", "game_pk", "batter", "team", "at_bat_number"]].reset_index(drop=True)


def latest_known_batter_team(appearances: pd.DataFrame) -> pd.DataFrame:
    """Latest team per batter by (game_date, game_pk, at_bat_number)."""
    if appearances.empty:
        return pd.DataFrame(columns=["key_mlbam", "team"])
    ordered = appearances.sort_values(["game_date", "game_pk", "at_bat_number"])
    latest = ordered.groupby("batter", as_index=False).tail(1)
    return latest[["batter", "team"]].rename(columns={"batter": "key_mlbam"}).reset_index(drop=True)


def eligible_lookback_batter_teams(
    appearances: pd.DataFrame,
    as_of_date,
    lookback_games: int = config.HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES,
    lookback_days: int = config.HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS,
) -> pd.DataFrame:
    """Unique (key_mlbam, team) that appeared for `team` within the lookback
    windows. Appearances are assumed to already be strictly before `as_of_date`.
    """
    empty = pd.DataFrame(columns=["key_mlbam", "team"])
    if appearances.empty:
        return empty

    as_of = _normalize_date(as_of_date)
    games = appearances[["team", "game_pk", "game_date"]].drop_duplicates()
    games = games.sort_values(["team", "game_date", "game_pk"], ascending=[True, False, False])
    games["recency_rank"] = games.groupby("team").cumcount()
    recent_games = games[games["recency_rank"] < lookback_games]
    by_games = appearances.merge(recent_games[["team", "game_pk"]], on=["team", "game_pk"], how="inner")

    day_cut = as_of - pd.Timedelta(days=lookback_days)
    appearance_dates = _date_series(appearances["game_date"])
    by_days = appearances.loc[appearance_dates >= day_cut]

    eligible = pd.concat(
        [by_games[["batter", "team"]], by_days[["batter", "team"]]],
        ignore_index=True,
    ).drop_duplicates()
    return eligible.rename(columns={"batter": "key_mlbam"}).reset_index(drop=True)


def build_pregame_candidate_universe(
    history: pd.DataFrame,
    team_schedule: pd.DataFrame,
    as_of_date,
    lookback_games: int = config.HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES,
    lookback_days: int = config.HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS,
) -> pd.DataFrame:
    """One row per (game_pk, team, key_mlbam) candidate from `history` only.

    `team_schedule` is the hitter schedule shape (one row per team per
    game_pk). The actual lineup is not an input.
    """
    columns = [
        "date", "game_pk", "team", "opponent", "is_home", "key_mlbam",
        "candidate_source", "candidate_lookback_games", "candidate_lookback_days",
    ]
    if history.empty or team_schedule.empty:
        return pd.DataFrame(columns=columns)

    appearances = batter_team_appearances(history)
    latest = latest_known_batter_team(appearances)
    eligible = eligible_lookback_batter_teams(
        appearances, as_of_date, lookback_games=lookback_games, lookback_days=lookback_days,
    )
    roster = latest.merge(eligible, on=["key_mlbam", "team"], how="inner")
    if roster.empty:
        return pd.DataFrame(columns=columns)

    schedule_columns = [c for c in ("date", "game_pk", "team", "opponent", "is_home") if c in team_schedule.columns]
    candidates = team_schedule[schedule_columns].merge(roster, on="team", how="inner")
    candidates = candidates.copy()
    candidates["candidate_source"] = CANDIDATE_SOURCE_PREGAME_LOOKBACK
    candidates["candidate_lookback_games"] = lookback_games
    candidates["candidate_lookback_days"] = lookback_days
    return candidates[columns].reset_index(drop=True)


def league_prior_feature_values(feature_frame: pd.DataFrame) -> pd.Series:
    """Median of available numeric opportunity features - the fill used when
    a live confirmed-lineup player has no pregame Statcast history of their
    own. Not a player-specific projection.
    """
    cols = [c for c in OPPORTUNITY_FEATURE_COLUMNS if c in feature_frame.columns]
    if not cols:
        return pd.Series(dtype=float)
    numeric = feature_frame[cols].select_dtypes(include="number")
    if numeric.empty:
        return pd.Series(dtype=float)
    return numeric.median(numeric_only=True)


def add_confirmed_lineup_candidates(
    historical_candidates: pd.DataFrame,
    confirmed_lineup: pd.DataFrame,
    league_priors: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """LIVE ONLY. Historical training never calls this.

    `confirmed_lineup` is a pregame snapshot: [game_pk, team, key_mlbam]
    (optional opponent/date/is_home). Players already in
    `historical_candidates` are left unchanged. Players on the confirmed
    lineup who were not in the lookback pool are appended with
    `candidate_source='confirmed_lineup'` and numeric features filled from
    `league_priors` (typically `league_prior_feature_values` of that slate's
    pregame WAVE/feature table).

    This must not be used to backfill historical coverage of rookies /
    call-ups after seeing who actually played.
    """
    required = {"game_pk", "team", "key_mlbam"}
    if confirmed_lineup.empty or not required.issubset(confirmed_lineup.columns):
        return historical_candidates.copy()

    existing_keys = historical_candidates[OPPORTUNITY_KEY_COLUMNS] if not historical_candidates.empty else pd.DataFrame(columns=OPPORTUNITY_KEY_COLUMNS)
    incoming = confirmed_lineup.copy()
    if "date" not in incoming.columns:
        if historical_candidates.empty or "date" not in historical_candidates.columns:
            raise ValueError("confirmed_lineup needs a date column when historical candidates are empty")
        date_by_game = historical_candidates[["game_pk", "date"]].drop_duplicates()
        incoming = incoming.merge(date_by_game, on="game_pk", how="left")

    incoming_keys = incoming[OPPORTUNITY_KEY_COLUMNS]
    merged = incoming_keys.merge(existing_keys, on=OPPORTUNITY_KEY_COLUMNS, how="left", indicator=True)
    new_keys = merged.loc[merged["_merge"] == "left_only", OPPORTUNITY_KEY_COLUMNS]
    if new_keys.empty:
        return historical_candidates.copy()

    extras = incoming.merge(new_keys, on=OPPORTUNITY_KEY_COLUMNS, how="inner").copy()
    extras["candidate_source"] = CANDIDATE_SOURCE_CONFIRMED_LINEUP
    if "candidate_lookback_games" not in extras.columns:
        extras["candidate_lookback_games"] = config.HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES
    if "candidate_lookback_days" not in extras.columns:
        extras["candidate_lookback_days"] = config.HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS

    priors = pd.Series(league_priors) if league_priors is not None else pd.Series(dtype=float)
    for col, value in priors.items():
        if col not in extras.columns:
            extras[col] = value
        else:
            extras[col] = extras[col].fillna(value)

    if historical_candidates.empty:
        return extras.reset_index(drop=True)
    return pd.concat([historical_candidates, extras], ignore_index=True)


def compute_game_hitter_labels(game_events: pd.DataFrame) -> pd.DataFrame:
    """Labels from the target game's own events only: one row per
    (game_pk, team, key_mlbam) who recorded a completed PA.

    `Batting_Order` is the reconstructed starting slot when they started
    (batting_order <= 9), else null. Pinch-hitters have Appeared=1,
    Started=0, Batting_Order null. Candidates who did not appear are not
    in this frame - callers left-join and fill DNP defaults.
    """
    empty = pd.DataFrame(
        columns=["game_pk", "team", "key_mlbam"] + OPPORTUNITY_LABEL_COLUMNS
    )
    if game_events.empty:
        return empty

    events = game_events.copy()
    if "pitch_number" not in events.columns:
        events["pitch_number"] = 1

    needed = [
        "game_date", "game_pk", "batter", "events", "inning_topbot",
        "home_team", "away_team", "at_bat_number",
    ]
    if any(c not in events.columns for c in needed):
        return empty

    completed = data.completed_events(events, needed)
    if completed.empty:
        return empty

    completed = completed.copy()
    completed["team"] = completed["away_team"].where(
        completed["inning_topbot"] == "Top", completed["home_team"]
    )
    completed["hit"] = helpers.is_hit(completed["events"])
    completed["official_ab"] = helpers.is_official_at_bat(completed["events"])

    agg = completed.groupby(["game_pk", "batter", "team"], as_index=False).agg(
        Plate_Appearances=("events", "count"),
        Official_At_Bats=("official_ab", "sum"),
        Hits=("hit", "sum"),
        Got_Hit=("hit", "max"),
    )

    with_id = data.assign_game_ids(events)
    order = data.assign_batting_order(with_id)
    id_map = (
        with_id.reset_index()[["game_id", "game_pk"]]
        .drop_duplicates()
    )
    order = order.merge(id_map, on="game_id", how="left")
    agg = agg.merge(
        order[["game_pk", "team", "batter", "batting_order"]],
        on=["game_pk", "team", "batter"],
        how="left",
    )

    agg["Started"] = (agg["batting_order"].fillna(99) <= lineup.STARTER_MAX_BATTING_ORDER).astype(int)
    agg["Batting_Order"] = agg["batting_order"].where(agg["Started"] == 1)
    agg["Appeared"] = 1
    agg["No_Game"] = 0
    agg["Got_Hit"] = agg["Got_Hit"].astype(int)
    agg["Hits"] = agg["Hits"].astype(int)
    agg["Official_At_Bats"] = agg["Official_At_Bats"].astype(int)
    agg["Plate_Appearances"] = agg["Plate_Appearances"].astype(int)
    return agg.rename(columns={"batter": "key_mlbam"})[
        ["game_pk", "team", "key_mlbam"] + OPPORTUNITY_LABEL_COLUMNS
    ].reset_index(drop=True)


def _dnp_label_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill labels for candidates who did not appear (negative examples)."""
    out = rows.copy()
    out["Appeared"] = out["Appeared"].fillna(0).astype(int)
    out["Started"] = out["Started"].fillna(0).astype(int)
    out["No_Game"] = out["No_Game"].fillna(1).astype(int)
    out.loc[out["Appeared"] == 0, "No_Game"] = 1
    out.loc[out["Appeared"] == 1, "No_Game"] = 0
    out["Plate_Appearances"] = out["Plate_Appearances"].fillna(0).astype(int)
    out["Official_At_Bats"] = out["Official_At_Bats"].fillna(0).astype(int)
    out["Hits"] = out["Hits"].fillna(0).astype(int)
    out["Got_Hit"] = out["Got_Hit"].fillna(0).astype(int)
    out.loc[out["Started"] == 0, "Batting_Order"] = pd.NA
    return out


def compute_candidate_coverage(
    candidates: pd.DataFrame,
    actual_labels: pd.DataFrame,
    history: pd.DataFrame,
    team_schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Per (date, game_pk, team) candidate-pool coverage vs actual starters
    and appearing hitters. Omitted rookies/call-ups are appearing hitters
    with *no* pregame Statcast history (the history slice already excludes
    the target game and, under a morning snapshot, same-date game one).
    """
    if team_schedule.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)

    history_batters = (
        set(history["batter"].dropna().unique()) if not history.empty and "batter" in history.columns else set()
    )
    cand_keys = (
        candidates[["game_pk", "team", "key_mlbam"]].drop_duplicates()
        if not candidates.empty
        else pd.DataFrame(columns=["game_pk", "team", "key_mlbam"])
    )
    actual = (
        actual_labels[["game_pk", "team", "key_mlbam", "Started", "Appeared"]]
        if not actual_labels.empty
        else pd.DataFrame(columns=["game_pk", "team", "key_mlbam", "Started", "Appeared"])
    )

    rows = []
    schedule = team_schedule[["date", "game_pk", "team"]].drop_duplicates()
    for rec in schedule.itertuples(index=False):
        game_pk, team = rec.game_pk, rec.team
        pool = set(
            cand_keys.loc[(cand_keys["game_pk"] == game_pk) & (cand_keys["team"] == team), "key_mlbam"]
        )
        appeared = actual.loc[(actual["game_pk"] == game_pk) & (actual["team"] == team)]
        starters = appeared.loc[appeared["Started"] == 1, "key_mlbam"]
        appearers = appeared.loc[appeared["Appeared"] == 1, "key_mlbam"]
        starter_ids = set(starters)
        appearer_ids = set(appearers)
        omitted = appearer_ids - pool
        omitted_no_history = {k for k in omitted if k not in history_batters}
        n_starters = len(starter_ids)
        n_appeared = len(appearer_ids)
        n_starters_in = len(starter_ids & pool)
        n_appeared_in = len(appearer_ids & pool)
        rows.append({
            "date": rec.date,
            "game_pk": game_pk,
            "team": team,
            "n_candidates": len(pool),
            "n_actual_starters": n_starters,
            "n_starters_in_candidates": n_starters_in,
            "starter_coverage": (n_starters_in / n_starters) if n_starters else pd.NA,
            "n_actual_appeared": n_appeared,
            "n_appeared_in_candidates": n_appeared_in,
            "appearance_coverage": (n_appeared_in / n_appeared) if n_appeared else pd.NA,
            "n_omitted_appeared": len(omitted),
            "n_omitted_no_pregame_history": len(omitted_no_history),
        })
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def _outputs_and_features(history: pd.DataFrame, todays_schedule: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    outputs = pipeline.compute_outputs(history)
    matchup_probability = matchup.compute_matchup_hit_probability(
        outputs["wave"], outputs["pave"], outputs["confidence"], todays_schedule,
    )
    features = dfs_ml.build_hitter_features(
        outputs["wave"], outputs["pave"], outputs["confidence"],
        todays_schedule, matchup_probability,
    )
    return outputs, features


def _attach_pregame_features(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    wave: pd.DataFrame,
    as_of_date,
    feature_as_of_timestamp,
) -> pd.DataFrame:
    # Candidates already carry schedule is_home; build_hitter_features also
    # emits is_home. Drop the schedule copy before the feature join so we
    # keep a single column instead of is_home_x / is_home_y.
    rows = candidates.drop(columns=[c for c in ("is_home",) if c in candidates.columns]).copy()
    feature_join_cols = ["key_mlbam", "game_pk"] + [
        c for c in dfs_ml.HITTER_FEATURE_COLUMNS if c in features.columns
    ]
    if features.empty:
        for col in dfs_ml.HITTER_FEATURE_COLUMNS:
            if col not in rows.columns:
                rows[col] = pd.NA
    else:
        rows = rows.merge(
            features[feature_join_cols].drop_duplicates(subset=["key_mlbam", "game_pk"]),
            on=["key_mlbam", "game_pk"],
            how="left",
        )

    extra = [c for c in ("key_mlbam", "avg_batting_order", "start_rate", "Last_Game_Date", "name_first", "name_last") if c in wave.columns]
    if extra:
        rows = rows.merge(wave[extra].drop_duplicates(subset=["key_mlbam"]), on="key_mlbam", how="left")
    else:
        for col in ("avg_batting_order", "start_rate", "Last_Game_Date"):
            if col not in rows.columns:
                rows[col] = pd.NA

    rows["as_of_date"] = _normalize_date(as_of_date)
    ts = pd.Timestamp(feature_as_of_timestamp)
    rows["feature_as_of_timestamp"] = ts.isoformat()

    last_game = pd.to_datetime(rows["Last_Game_Date"], errors="coerce")
    rows["Days_Rest"] = (rows["as_of_date"] - last_game.dt.normalize()).dt.days
    return rows


def _finalize_opportunity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    preferred = (
        [
            "date", "game_pk", "team", "opponent", "is_home", "key_mlbam",
            "name_first", "name_last",
            "candidate_source", "candidate_lookback_games", "candidate_lookback_days",
            "as_of_date", "feature_as_of_timestamp",
        ]
        + list(OPPORTUNITY_FEATURE_COLUMNS)
        + list(OPPORTUNITY_LABEL_COLUMNS)
    )
    ordered = []
    seen = set()
    for col in preferred:
        if col in rows.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    leftover = [c for c in rows.columns if c not in seen]
    rows = rows[ordered + leftover]
    rows = rows.sort_values(OPPORTUNITY_KEY_COLUMNS).reset_index(drop=True)
    assert_unique_opportunity_keys(rows)
    return rows


def assemble_hitter_opportunity_dataset(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int | None = None,
    *,
    lookback_games: int | None = None,
    lookback_days: int | None = None,
    prediction_timestamp_utc=None,
    persisted: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the opportunity log and per-team-game coverage tables.

    `prediction_timestamp_utc=None` is the morning snapshot (historical
    training default). `persisted` is an optional in-memory Statcast frame
    so tests need not round-trip parquet; production callers omit it.
    """
    lookback_games = config.HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES if lookback_games is None else lookback_games
    lookback_days = config.HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS if lookback_days is None else lookback_days
    season = season or config.SEASON_START.year

    if persisted is None:
        persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None or persisted.empty:
        return pd.DataFrame(), pd.DataFrame(columns=COVERAGE_COLUMNS)

    persisted = persisted.copy()
    persisted["game_date"] = pd.to_datetime(persisted["game_date"])

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    if team_schedule.empty:
        return pd.DataFrame(), pd.DataFrame(columns=COVERAGE_COLUMNS)

    team_schedule = team_schedule.copy()
    team_schedule["date"] = pd.to_datetime(team_schedule["date"])
    dates = sorted(team_schedule["date"].unique())
    if days:
        dates = dates[-days:]

    row_frames = []
    coverage_frames = []
    outputs_cache: dict[tuple, dict] = {}
    features_cache: dict[tuple, pd.DataFrame] = {}

    def _cached_outputs(history: pd.DataFrame, schedule_slice: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        history_pks = tuple(sorted(history["game_pk"].unique().tolist()))
        schedule_pks = tuple(sorted(schedule_slice["game_pk"].unique().tolist()))
        cache_key = (history_pks, schedule_pks)
        if cache_key not in outputs_cache:
            outputs, feats = _outputs_and_features(history, schedule_slice)
            outputs_cache[cache_key] = outputs
            features_cache[cache_key] = feats
        return outputs_cache[cache_key], features_cache[cache_key]

    for date in dates:
        todays_schedule = team_schedule[team_schedule["date"] == date]
        if todays_schedule.empty:
            continue
        day_events = persisted[_date_series(persisted["game_date"]) == _normalize_date(date)]
        actual_labels = compute_game_hitter_labels(day_events)

        # Group games that share the same pregame history (morning: all
        # games that date; later snapshot: per completed-game cut).
        history_groups: dict[tuple, list] = {}
        history_by_key: dict[tuple, pd.DataFrame] = {}
        for game_pk in todays_schedule["game_pk"].unique():
            history = slice_history_before_game(
                persisted,
                date,
                target_game_pk=game_pk,
                prediction_timestamp_utc=prediction_timestamp_utc,
            )
            if history.empty:
                continue
            hist_key = tuple(sorted(history["game_pk"].unique().tolist()))
            history_groups.setdefault(hist_key, []).append(game_pk)
            history_by_key[hist_key] = history

        date_rows = []
        date_coverage = []
        for hist_key, game_pks in history_groups.items():
            history = history_by_key[hist_key]
            schedule_slice = todays_schedule[todays_schedule["game_pk"].isin(game_pks)]
            outputs, features = _cached_outputs(history, schedule_slice)
            candidates = build_pregame_candidate_universe(
                history, schedule_slice, date,
                lookback_games=lookback_games, lookback_days=lookback_days,
            )
            as_of_ts = (
                pd.Timestamp(prediction_timestamp_utc)
                if prediction_timestamp_utc is not None
                else morning_feature_as_of_timestamp(date)
            )
            if candidates.empty:
                coverage = compute_candidate_coverage(
                    candidates, actual_labels, history, schedule_slice,
                )
                date_coverage.append(coverage)
                continue

            featured = _attach_pregame_features(
                candidates, features, outputs["wave"], date, as_of_ts,
            )
            label_cols = ["game_pk", "team", "key_mlbam"] + OPPORTUNITY_LABEL_COLUMNS
            if actual_labels.empty:
                labels = pd.DataFrame(columns=label_cols)
            else:
                labels = actual_labels[label_cols]
            featured = featured.merge(labels, on=["game_pk", "team", "key_mlbam"], how="left")
            featured = _dnp_label_defaults(featured)
            date_rows.append(featured)
            date_coverage.append(
                compute_candidate_coverage(candidates, actual_labels, history, schedule_slice)
            )

        if date_rows:
            row_frames.append(pd.concat(date_rows, ignore_index=True))
        if date_coverage:
            coverage_frames.append(pd.concat(date_coverage, ignore_index=True))

    rows = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame(columns=COVERAGE_COLUMNS)
    rows = _finalize_opportunity_rows(rows)
    if not coverage.empty:
        coverage = coverage.sort_values(["date", "game_pk", "team"]).reset_index(drop=True)
    return rows, coverage


def assemble_hitter_opportunity_log(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """One row per pregame candidate per game_pk. See
    `assemble_hitter_opportunity_dataset` for coverage.
    """
    rows, _coverage = assemble_hitter_opportunity_dataset(
        raw_dir, season, days, **kwargs,
    )
    return rows


def summarize_opportunity_log(rows: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    """Aggregate coverage rates and label distributions for the report script."""
    summary: dict = {
        "n_rows": int(len(rows)),
        "n_games": int(rows["game_pk"].nunique()) if not rows.empty and "game_pk" in rows.columns else 0,
        "n_dates": int(rows["date"].nunique()) if not rows.empty and "date" in rows.columns else 0,
    }
    if not coverage.empty:
        starter_den = coverage["n_actual_starters"].sum()
        appear_den = coverage["n_actual_appeared"].sum()
        summary["starter_coverage"] = (
            float(coverage["n_starters_in_candidates"].sum() / starter_den) if starter_den else None
        )
        summary["appearance_coverage"] = (
            float(coverage["n_appeared_in_candidates"].sum() / appear_den) if appear_den else None
        )
        summary["n_omitted_appeared"] = int(coverage["n_omitted_appeared"].sum())
        summary["n_omitted_no_pregame_history"] = int(coverage["n_omitted_no_pregame_history"].sum())
        summary["n_candidates_total"] = int(coverage["n_candidates"].sum())
    if not rows.empty:
        for col in ("Started", "Appeared", "Got_Hit", "No_Game"):
            if col in rows.columns:
                summary[f"{col}_rate"] = float(rows[col].mean())
                summary[f"{col}_count"] = int(rows[col].sum())
        if "Plate_Appearances" in rows.columns:
            summary["mean_plate_appearances"] = float(rows["Plate_Appearances"].mean())
        appeared = rows[rows["Appeared"] == 1] if "Appeared" in rows.columns else rows
        if not appeared.empty and "Got_Hit" in appeared.columns:
            summary["Got_Hit_rate_among_appeared"] = float(appeared["Got_Hit"].mean())
        if "candidate_source" in rows.columns:
            summary["candidate_sources"] = rows["candidate_source"].value_counts().to_dict()
    return summary
