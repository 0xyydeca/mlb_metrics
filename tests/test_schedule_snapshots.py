"""As-of schedule / probable-starter snapshots and backtest modes."""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_metrics import config, game_picks_backtest, schedule_snapshots as ss


def _snap(
    *,
    game_pk,
    captured_at,
    home_p,
    away_p,
    target_date="2026-09-03",
    home="NYY",
    away="BOS",
    game_datetime="2026-09-03T23:10:00Z",
    status="Scheduled",
    source=ss.SOURCE_FIXTURE,
):
    row = {
        "captured_at_utc": captured_at,
        "prediction_target_date": pd.Timestamp(target_date),
        "game_pk": game_pk,
        "game_datetime": game_datetime,
        "home_team": home,
        "away_team": away,
        "home_probable_pitcher_key_mlbam": home_p,
        "away_probable_pitcher_key_mlbam": away_p,
        "starter_announcement_status": ss.starter_announcement_status(home_p, away_p),
        "game_status": status,
        "source": source,
    }
    row["snapshot_id"] = ss.make_snapshot_id(row)
    return row


def test_config_default_backtest_mode_is_as_of():
    assert config.SCHEDULE_BACKTEST_MODE_DEFAULT == "as_of_snapshot"
    assert "actual_starter" in config.SCHEDULE_BACKTEST_MODES
    assert "missing_snapshot" in config.SCHEDULE_BACKTEST_MODES


def test_no_snapshot_after_prediction_timestamp_is_used():
    snaps = pd.DataFrame([
        _snap(game_pk=100, captured_at="2026-09-03T12:00:00Z", home_p=101, away_p=201),
        _snap(game_pk=100, captured_at="2026-09-03T18:00:00Z", home_p=999, away_p=201),  # after pred
    ])
    selected = ss.select_as_of_snapshots(snaps, "2026-09-03T15:00:00Z")
    assert len(selected) == 1
    assert int(selected.iloc[0]["home_probable_pitcher_key_mlbam"]) == 101
    assert int(selected.iloc[0]["home_probable_pitcher_key_mlbam"]) != 999


def test_actual_starter_not_silently_substituted_when_as_of_missing():
    actual = pd.DataFrame([{
        "game_pk": 100,
        "date": pd.Timestamp("2026-09-03"),
        "home_team": "NYY",
        "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 101,
        "away_probable_pitcher_key_mlbam": 201,
        "status": "Final",
        "home_score": 5,
        "away_score": 2,
        "game_datetime": "2026-09-03T23:10:00Z",
    }])
    games, meta = ss.resolve_schedule_for_backtest(
        ss.BACKTEST_MODE_AS_OF,
        prediction_timestamp="2026-09-03T15:00:00Z",
        prediction_target_date="2026-09-03",
        schedule_snapshots=ss.empty_snapshot_frame(),
        actual_starter_games=actual,
    )
    assert games.empty
    assert meta["substituted_actual_starter"] is False
    assert meta["n_missing_snapshot"] == 1
    assert meta["production_equivalent"] is True
    ss.assert_no_actual_starter_substitution(meta)


def test_missing_snapshot_mode_is_explicitly_unavailable():
    games, meta = ss.resolve_schedule_for_backtest(ss.BACKTEST_MODE_MISSING)
    assert games.empty
    assert meta["reason"] == "explicitly_unavailable"
    assert meta["substituted_actual_starter"] is False


def test_doubleheader_starters_remain_distinct():
    snaps = pd.DataFrame([
        _snap(game_pk=100, captured_at="2026-09-03T12:00:00Z", home_p=101, away_p=201,
              game_datetime="2026-09-03T17:05:00Z"),
        _snap(game_pk=101, captured_at="2026-09-03T12:00:00Z", home_p=102, away_p=202,
              game_datetime="2026-09-03T23:10:00Z"),
    ])
    selected = ss.select_as_of_snapshots(snaps, "2026-09-03T15:00:00Z")
    by_pk = selected.set_index("game_pk")
    assert int(by_pk.loc[100, "home_probable_pitcher_key_mlbam"]) == 101
    assert int(by_pk.loc[101, "home_probable_pitcher_key_mlbam"]) == 102
    assert int(by_pk.loc[100, "away_probable_pitcher_key_mlbam"]) == 201
    assert int(by_pk.loc[101, "away_probable_pitcher_key_mlbam"]) == 202


def test_late_starter_change_appears_only_in_later_snapshots():
    snaps = pd.DataFrame([
        _snap(game_pk=100, captured_at="2026-09-03T12:00:00Z", home_p=101, away_p=201),
        _snap(game_pk=100, captured_at="2026-09-03T20:00:00Z", home_p=155, away_p=201),  # scratch/change
    ])
    morning = ss.select_as_of_snapshots(snaps, "2026-09-03T15:00:00Z")
    lineup_lock = ss.select_as_of_snapshots(snaps, "2026-09-03T20:30:00Z")
    assert int(morning.iloc[0]["home_probable_pitcher_key_mlbam"]) == 101
    assert int(lineup_lock.iloc[0]["home_probable_pitcher_key_mlbam"]) == 155


def test_append_only_persistence_keeps_prior_captures(tmp_path, monkeypatch):
    path = tmp_path / "schedule_snapshots.csv"
    monkeypatch.setattr(config, "SCHEDULE_SNAPSHOTS_PATH", str(path))
    first = ss.snapshots_from_schedule_games(
        pd.DataFrame([{
            "game_pk": 100, "date": pd.Timestamp("2026-09-03"),
            "home_team": "NYY", "away_team": "BOS",
            "home_probable_pitcher_key_mlbam": 101,
            "away_probable_pitcher_key_mlbam": 201,
            "status": "Scheduled", "game_datetime": "2026-09-03T23:10:00Z",
        }]),
        captured_at_utc="2026-09-03T12:00:00Z",
    )
    second = ss.snapshots_from_schedule_games(
        pd.DataFrame([{
            "game_pk": 100, "date": pd.Timestamp("2026-09-03"),
            "home_team": "NYY", "away_team": "BOS",
            "home_probable_pitcher_key_mlbam": 155,
            "away_probable_pitcher_key_mlbam": 201,
            "status": "Scheduled", "game_datetime": "2026-09-03T23:10:00Z",
        }]),
        captured_at_utc="2026-09-03T20:00:00Z",
    )
    ss.append_schedule_snapshots(first)
    ss.append_schedule_snapshots(second)
    loaded = ss.load_schedule_snapshots()
    assert len(loaded) == 2
    assert set(loaded["home_probable_pitcher_key_mlbam"].astype(int)) == {101, 155}


def test_starter_change_statistics():
    as_of = pd.DataFrame([
        {"game_pk": 1, "home_probable_pitcher_key_mlbam": 101, "away_probable_pitcher_key_mlbam": 201},
        {"game_pk": 2, "home_probable_pitcher_key_mlbam": 111, "away_probable_pitcher_key_mlbam": 211},
        {"game_pk": 3, "home_probable_pitcher_key_mlbam": pd.NA, "away_probable_pitcher_key_mlbam": 221},
    ])
    actual = pd.DataFrame([
        {"game_pk": 1, "home_probable_pitcher_key_mlbam": 101, "away_probable_pitcher_key_mlbam": 201},
        {"game_pk": 2, "home_probable_pitcher_key_mlbam": 999, "away_probable_pitcher_key_mlbam": 211},
        {"game_pk": 3, "home_probable_pitcher_key_mlbam": 121, "away_probable_pitcher_key_mlbam": 221},
    ])
    stats = ss.starter_change_statistics(as_of, actual)
    assert stats["n_compared_games"] == 3
    assert stats["probable_matched_actual"] == 1
    assert stats["starter_changed_before_game"] == 1
    assert stats["probable_starter_missing"] == 1
    assert stats["home_changed"] == 1


def test_starter_change_probability_effects_separate_modes():
    as_of = pd.DataFrame([
        {"game_pk": 1, "home_win_probability": 0.55, "Home_Won": 1},
        {"game_pk": 2, "home_win_probability": 0.60, "Home_Won": 0},
    ])
    actual = pd.DataFrame([
        {"game_pk": 1, "home_win_probability": 0.58},
        {"game_pk": 2, "home_win_probability": 0.45},
    ])
    flags = pd.DataFrame([
        {"game_pk": 1, "starter_changed": False},
        {"game_pk": 2, "starter_changed": True},
    ])
    effects = ss.starter_change_probability_effects(as_of, actual, change_flags=flags)
    assert effects["modes_merged"] is False
    assert effects["n_paired_games"] == 2
    assert effects["mean_abs_probability_delta"] == pytest.approx(0.09, abs=1e-9)
    assert effects["changed_subset_mean_abs_delta"] == pytest.approx(0.15, abs=1e-9)
    assert effects["matched_subset_mean_abs_delta"] == pytest.approx(0.03, abs=1e-9)
    assert effects["as_of_brier"] == effects["as_of_brier"]  # finite
    assert effects["actual_starter_brier"] == effects["actual_starter_brier"]


def test_evaluation_separates_two_backtest_modes():
    as_of = pd.DataFrame([
        {"game_pk": 1, "home_win_probability": 0.6, "Home_Won": 1, "schedule_backtest_mode": "as_of_snapshot"},
    ])
    actual = pd.DataFrame([
        {"game_pk": 1, "home_win_probability": 0.7, "Home_Won": 1, "schedule_backtest_mode": "actual_starter"},
        {"game_pk": 2, "home_win_probability": 0.4, "Home_Won": 0, "schedule_backtest_mode": "actual_starter"},
    ])
    report = game_picks_backtest.evaluate_schedule_backtest_modes_separately(as_of, actual)
    assert report["modes_merged"] is False
    assert report["as_of_snapshot"]["n_games"] == 1
    assert report["actual_starter"]["n_games"] == 2
    assert report["sample_report"]["modes_merged"] is False
    assert report["sample_report"]["production_equivalent_sample_sufficient"] is False


def test_evaluation_rejects_mixed_mode_frame():
    mixed = pd.DataFrame([
        {"game_pk": 1, "home_win_probability": 0.6, "Home_Won": 1, "schedule_backtest_mode": "as_of_snapshot"},
        {"game_pk": 2, "home_win_probability": 0.5, "Home_Won": 0, "schedule_backtest_mode": "actual_starter"},
    ])
    with pytest.raises(AssertionError, match="mixed modes"):
        game_picks_backtest.evaluate_schedule_backtest_modes_separately(mixed, None)


def test_production_equivalent_sample_report_does_not_merge():
    report = ss.production_equivalent_sample_report(10, actual_starter_n_games=500, min_games=100)
    assert report["production_equivalent_sample_sufficient"] is False
    assert report["modes_merged"] is False
    assert "do_not_merge" in report["label"]
    # Even with a large actual-starter diagnostic sample, sufficiency is as-of only.
    assert report["as_of_snapshot_n_games"] == 10


def test_as_of_resolve_uses_snapshot_starters_not_actual():
    snaps = pd.DataFrame([
        _snap(game_pk=100, captured_at="2026-09-03T12:00:00Z", home_p=101, away_p=201),
    ])
    actual = pd.DataFrame([{
        "game_pk": 100,
        "date": pd.Timestamp("2026-09-03"),
        "home_team": "NYY",
        "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 999,
        "away_probable_pitcher_key_mlbam": 888,
        "status": "Final",
        "home_score": 5,
        "away_score": 2,
    }])
    games, meta = ss.resolve_schedule_for_backtest(
        ss.BACKTEST_MODE_AS_OF,
        prediction_timestamp="2026-09-03T15:00:00Z",
        prediction_target_date="2026-09-03",
        schedule_snapshots=snaps,
        actual_starter_games=actual,
    )
    assert len(games) == 1
    assert int(games.iloc[0]["home_probable_pitcher_key_mlbam"]) == 101
    assert int(games.iloc[0]["away_probable_pitcher_key_mlbam"]) == 201
    assert games.iloc[0]["home_score"] == 5  # scores joined for labels only
    assert meta["schedule_backtest_mode"] == "as_of_snapshot"
    assert meta["substituted_actual_starter"] is False


def test_snapshots_from_schedule_games_sets_announcement_status():
    games = pd.DataFrame([{
        "game_pk": 1, "date": "2026-09-03", "home_team": "NYY", "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 101, "away_probable_pitcher_key_mlbam": pd.NA,
        "status": "Scheduled", "game_datetime": "2026-09-03T23:00:00Z",
    }])
    snaps = ss.snapshots_from_schedule_games(games, captured_at_utc="2026-09-03T12:00:00Z")
    assert snaps.iloc[0]["starter_announcement_status"] == ss.STARTER_STATUS_HOME_ONLY
