import datetime

import pandas as pd
import pytest

from mlb_metrics import schedule


def test_today_local_uses_arizona_time_not_utc(monkeypatch):
    # 3:45am UTC on July 29 is still 8:45pm July 28 in Arizona (fixed
    # UTC-7, no DST) - the exact scenario a late-evening workflow_dispatch
    # run hit: naive datetime.date.today() on the GitHub Actions runner
    # (UTC) already reads July 29, publishing a day-ahead slate to an
    # Arizona audience for whom it's still July 28.
    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            utc_now = datetime.datetime(2026, 7, 29, 3, 45, tzinfo=datetime.timezone.utc)
            return utc_now.astimezone(tz) if tz else utc_now

    monkeypatch.setattr(schedule.datetime, "datetime", FakeDateTime)

    assert schedule.today_local() == datetime.date(2026, 7, 28)


def test_today_local_after_arizonas_own_midnight(monkeypatch):
    # 8:00am UTC = 1:00am Arizona - past Arizona's own midnight, so "today"
    # has genuinely rolled over there too.
    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            utc_now = datetime.datetime(2026, 7, 29, 8, 0, tzinfo=datetime.timezone.utc)
            return utc_now.astimezone(tz) if tz else utc_now

    monkeypatch.setattr(schedule.datetime, "datetime", FakeDateTime)

    assert schedule.today_local() == datetime.date(2026, 7, 29)


def test_team_id_to_abbrev_has_30_teams_and_matches_docs_app_js():
    # Spot-check against the table docs/app.js's teamLogo() ports from -
    # catches drift if either table is edited without the other.
    assert len(schedule.TEAM_ID_TO_ABBREV) == 30
    assert schedule.TEAM_ID_TO_ABBREV[133] == "ATH"
    assert schedule.TEAM_ID_TO_ABBREV[120] == "WSH"
    assert schedule.TEAM_ID_TO_ABBREV[147] == "NYY"


def _raw_schedule(games, game_pks=None, statuses=None, scores=None, game_datetimes=None):
    """games: list of (date, home_id, away_id, home_probable_id, away_probable_id).
    game_pks/statuses/scores/game_datetimes: optional parallel lists (same
    length as games), each entry a game_pk int / detailedState string /
    (home,away) score tuple / gameDate ISO string - used by the
    normalize_schedule_games tests (and, for game_datetimes, by
    normalize_schedule's own game_datetime tests)."""
    by_date = {}
    for i, (date, home_id, away_id, home_probable, away_probable) in enumerate(games):
        entry = by_date.setdefault(date, {"date": date, "games": []})
        game = {
            "teams": {
                "home": {"team": {"id": home_id}},
                "away": {"team": {"id": away_id}},
            }
        }
        if home_probable is not None:
            game["teams"]["home"]["probablePitcher"] = {"id": home_probable}
        if away_probable is not None:
            game["teams"]["away"]["probablePitcher"] = {"id": away_probable}
        if game_pks is not None:
            game["gamePk"] = game_pks[i]
        if statuses is not None:
            game["status"] = {"detailedState": statuses[i]}
        if scores is not None:
            home_score, away_score = scores[i]
            if home_score is not None:
                game["teams"]["home"]["score"] = home_score
            if away_score is not None:
                game["teams"]["away"]["score"] = away_score
        if game_datetimes is not None:
            game["gameDate"] = game_datetimes[i]
        entry["games"].append(game)
    return {"dates": list(by_date.values())}


def test_normalize_schedule_parses_probable_pitchers_and_opponents():
    # NYY (147) home vs BOS (111) away; NYY's probable is announced, BOS's isn't yet.
    raw = _raw_schedule([("2026-07-21", 147, 111, 592789, None)])

    result = schedule.normalize_schedule(raw).set_index("team")

    assert result.loc["NYY", "opponent"] == "BOS"
    assert result.loc["NYY", "probable_pitcher_key_mlbam"] == 592789
    assert result.loc["BOS", "opponent"] == "NYY"
    assert pd.isna(result.loc["BOS", "probable_pitcher_key_mlbam"])
    assert (result["date"] == pd.Timestamp("2026-07-21")).all()


def test_normalize_schedule_skips_unknown_team_ids():
    raw = _raw_schedule([("2026-07-21", 147, 999999, 592789, 111111)])

    result = schedule.normalize_schedule(raw)

    assert result.empty  # unknown away team id (999999) -> whole game skipped, not guessed at


def test_normalize_schedule_keeps_only_first_game_of_a_doubleheader():
    raw = _raw_schedule(
        [
            ("2026-07-21", 147, 111, 592789, 111111),
            ("2026-07-21", 147, 111, 605400, 222222),  # game 2, same two teams
        ]
    )

    result = schedule.normalize_schedule(raw)

    assert len(result[result["team"] == "NYY"]) == 1
    assert result[result["team"] == "NYY"].iloc[0]["probable_pitcher_key_mlbam"] == 592789


def test_normalize_hitter_schedule_keeps_both_games_of_a_doubleheader():
    raw = _raw_schedule(
        [
            ("2026-07-21", 147, 111, 592789, 111111),
            ("2026-07-21", 147, 111, 605400, 222222),
        ],
        game_pks=[824409, 824410],
    )

    result = schedule.normalize_hitter_schedule(raw)

    nyy = result[result["team"] == "NYY"].sort_values("game_pk")
    assert list(nyy["game_pk"]) == [824409, 824410]
    assert list(nyy["probable_pitcher_key_mlbam"]) == [592789, 605400]
    assert len(result[result["team"] == "BOS"]) == 2


def test_normalize_schedule_carries_game_pk_and_is_home():
    raw = _raw_schedule([("2026-07-21", 147, 111, 592789, None)], game_pks=[824409])

    result = schedule.normalize_schedule(raw).set_index("team")

    assert result.loc["NYY", "game_pk"] == 824409
    assert bool(result.loc["NYY", "is_home"]) is True
    assert result.loc["BOS", "game_pk"] == 824409
    assert bool(result.loc["BOS", "is_home"]) is False


def test_normalize_schedule_carries_game_datetime():
    raw = _raw_schedule(
        [("2026-07-21", 147, 111, 592789, None)],
        game_datetimes=["2026-07-21T23:05:00Z"],
    )

    result = schedule.normalize_schedule(raw).set_index("team")

    assert result.loc["NYY", "game_datetime"] == "2026-07-21T23:05:00Z"
    assert result.loc["BOS", "game_datetime"] == "2026-07-21T23:05:00Z"


def test_normalize_schedule_game_datetime_is_none_not_a_crash_when_absent():
    # gameDate isn't yet live-confirmed (see module docstring) - a raw
    # response missing it entirely must degrade to null, never raise.
    raw = _raw_schedule([("2026-07-21", 147, 111, 592789, None)])

    result = schedule.normalize_schedule(raw).set_index("team")

    assert pd.isna(result.loc["NYY", "game_datetime"])


def test_normalize_schedule_games_one_row_per_game_with_status_and_scores():
    raw = _raw_schedule(
        [("2026-07-21", 147, 111, 592789, 111111)],
        game_pks=[824409],
        statuses=["In Progress"],
        scores=[(2, 2)],
    )

    result = schedule.normalize_schedule_games(raw)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["game_pk"] == 824409
    assert row["home_team"] == "NYY"
    assert row["away_team"] == "BOS"
    assert row["home_probable_pitcher_key_mlbam"] == 592789
    assert row["away_probable_pitcher_key_mlbam"] == 111111
    assert row["status"] == "In Progress"
    assert row["home_score"] == 2
    assert row["away_score"] == 2


def test_normalize_schedule_games_final_game_scores():
    raw = _raw_schedule(
        [("2026-07-20", 147, 111, 592789, 111111)],
        game_pks=[824400],
        statuses=["Final"],
        scores=[(5, 3)],
    )

    result = schedule.normalize_schedule_games(raw)

    row = result.iloc[0]
    assert row["status"] == "Final"
    assert row["home_score"] == 5
    assert row["away_score"] == 3


def test_normalize_schedule_games_keeps_both_games_of_a_doubleheader():
    # Unlike normalize_schedule (team-per-row), normalize_schedule_games
    # must NOT dedupe - dropping game 2 would silently lose a whole game's
    # pick/resolution.
    raw = _raw_schedule(
        [
            ("2026-07-21", 147, 111, 592789, 111111),
            ("2026-07-21", 147, 111, 605400, 222222),
        ],
        game_pks=[824409, 824410],
        statuses=["Scheduled", "Scheduled"],
        scores=[(None, None), (None, None)],
    )

    result = schedule.normalize_schedule_games(raw)

    assert len(result) == 2
    assert set(result["game_pk"]) == {824409, 824410}
    assert (result["home_team"] == "NYY").all()
    assert (result["away_team"] == "BOS").all()


def test_normalize_schedule_games_skips_unknown_team_ids():
    raw = _raw_schedule(
        [("2026-07-21", 147, 999999, 592789, 111111)],
        game_pks=[824409],
        statuses=["Scheduled"],
        scores=[(None, None)],
    )

    result = schedule.normalize_schedule_games(raw)

    assert result.empty


def test_build_probable_pitchers_table_matches_announced_starter():
    schedule_df = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-22"), "team": "NYY", "opponent": "BOS",
         "is_home": True, "probable_pitcher_key_mlbam": 501},
    ])
    pave = pd.DataFrame([
        {"key_mlbam": 501, "name_first": "Gerrit", "name_last": "Cole",
         "PAVE": 0.25, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.75},
    ])

    result = schedule.build_probable_pitchers_table(schedule_df, pave)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team"] == "NYY"
    assert row["opponent"] == "BOS"
    assert row["pitcher_name"] == "Gerrit Cole"
    assert row["PAVE_PLUS"] == pytest.approx(0.8)
    assert row["Power_A_PLUS"] == pytest.approx(0.75)


def test_build_probable_pitchers_table_keeps_team_row_when_starter_unannounced():
    schedule_df = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-22"), "team": "NYY", "opponent": "BOS",
         "is_home": True, "probable_pitcher_key_mlbam": None},
    ])
    pave = pd.DataFrame(columns=["key_mlbam", "name_first", "name_last", "PAVE", "PAVE_PLUS", "Power_A_PLUS"])

    result = schedule.build_probable_pitchers_table(schedule_df, pave)

    assert len(result) == 1  # team still shows up, not silently dropped
    assert result.iloc[0]["pitcher_name"] == ""
    assert pd.isna(result.iloc[0]["PAVE_PLUS"])


def test_build_probable_pitchers_table_keeps_team_row_when_starter_not_in_pave():
    # Announced starter's key_mlbam isn't in pave.csv (e.g. a rookie call-up
    # with no persisted pitching history yet) - same graceful degradation.
    schedule_df = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-22"), "team": "NYY", "opponent": "BOS",
         "is_home": True, "probable_pitcher_key_mlbam": 999},
    ])
    pave = pd.DataFrame([
        {"key_mlbam": 501, "name_first": "Gerrit", "name_last": "Cole",
         "PAVE": 0.25, "PAVE_PLUS": 0.8, "Power_A_PLUS": 0.75},
    ])

    result = schedule.build_probable_pitchers_table(schedule_df, pave)

    assert len(result) == 1
    assert result.iloc[0]["pitcher_name"] == ""
