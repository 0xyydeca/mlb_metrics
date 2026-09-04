"""Fixture-based tests for confirmed-lineup snapshots and supersede rules.

Uses **normalized** snapshot fixtures (adapter output schema), not assumed
raw Stats API JSON — Stage A must confirm the live shape before raw fixtures
are committed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_metrics import config, game_predictions, lineup_snapshots, predictions


def _snap(
    *,
    game_pk,
    team,
    opponent,
    key_mlbam,
    batting_order,
    is_confirmed_starter=True,
    lineup_status="confirmed",
    fetched_at_utc="2026-09-03T18:00:00Z",
    game_datetime="2026-09-03T23:10:00Z",
    source="fixture",
    as_of_date="2026-09-03",
):
    return {
        "game_pk": game_pk,
        "team": team,
        "opponent": opponent,
        "key_mlbam": key_mlbam,
        "batting_order": batting_order,
        "is_confirmed_starter": is_confirmed_starter,
        "lineup_status": lineup_status,
        "fetched_at_utc": fetched_at_utc,
        "game_datetime": game_datetime,
        "source": source,
        "snapshot_id": "fixture",
        "as_of_date": as_of_date,
    }


def _pool_row(key_mlbam, game_pk, team="NYY", batting_order=3.0, p_appear=0.7):
    return {
        "key_mlbam": key_mlbam,
        "game_pk": game_pk,
        "team": team,
        "name_first": "A",
        "name_last": str(key_mlbam),
        "PA_L": 40,
        "PA_R": 40,
        "Approach": 0.5,
        "probability": 0.8,
        "Game_Hit_Probability": 0.8,
        "Matchup_Hit_Probability": 0.8,
        "avg_batting_order": batting_order,
        "start_rate": 0.9,
        "Last_Game_Date": pd.Timestamp("2026-09-01"),
        "P_Appear": p_appear,
        "WAVE": 0.3,
        "date": pd.Timestamp("2026-09-03"),
    }


# ---------------------------------------------------------------------------
# Snapshot adapter / apply
# ---------------------------------------------------------------------------


def test_unconfirmed_lineup_keeps_historical_appearance():
    pool = pd.DataFrame([_pool_row(1, 100)])
    snaps = lineup_snapshots.empty_snapshot_frame()
    out = lineup_snapshots.apply_confirmed_lineup_to_pool(pool, snaps)
    assert out.iloc[0]["P_Appear"] == pytest.approx(0.7)
    assert out.iloc[0]["lineup_status"] == "unconfirmed"


def test_confirmed_lineup_sets_scratch_risk_appearance_and_order():
    pool = pd.DataFrame([_pool_row(1, 100, batting_order=5.0, p_appear=0.55)])
    snaps = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=2),
    ]))
    out = lineup_snapshots.apply_confirmed_lineup_to_pool(pool, snaps)
    expected = 1.0 - config.LINEUP_CONFIRMED_SCRATCH_RISK
    assert out.iloc[0]["P_Appear"] == pytest.approx(expected)
    assert out.iloc[0]["batting_order"] == 2
    assert out.iloc[0]["avg_batting_order"] == 2
    assert out.iloc[0]["lineup_status"] == "confirmed"
    assert bool(out.iloc[0]["is_confirmed_starter"]) is True


def test_confirmed_player_missing_from_history_added_with_priors():
    pool = pd.DataFrame([_pool_row(1, 100)])
    snaps = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=1),
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=99, batting_order=9),
    ]))
    out = lineup_snapshots.apply_confirmed_lineup_to_pool(pool, snaps)
    assert 99 in set(out["key_mlbam"].astype(int))
    newbie = out[out["key_mlbam"] == 99].iloc[0]
    assert newbie["lineup_status"] == "confirmed"
    assert newbie["P_Appear"] == pytest.approx(1.0 - config.LINEUP_CONFIRMED_SCRATCH_RISK)


def test_late_scratch_zeros_appearance_and_marks_status():
    previous = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=3,
              fetched_at_utc="2026-09-03T17:00:00Z"),
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=2, batting_order=4,
              fetched_at_utc="2026-09-03T17:00:00Z"),
    ]))
    current = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=2, batting_order=4,
              fetched_at_utc="2026-09-03T18:30:00Z"),
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=3, batting_order=3,
              fetched_at_utc="2026-09-03T18:30:00Z"),
    ]))
    annotated = lineup_snapshots.annotate_scratches(previous, current)
    scratched = annotated[annotated["lineup_status"] == "scratched"]
    assert set(scratched["key_mlbam"].astype(int)) == {1}

    pool = pd.DataFrame([_pool_row(1, 100), _pool_row(2, 100), _pool_row(3, 100)])
    out = lineup_snapshots.apply_confirmed_lineup_to_pool(pool, annotated)
    row1 = out[out["key_mlbam"] == 1].iloc[0]
    assert row1["lineup_status"] == "scratched"
    assert row1["P_Appear"] == pytest.approx(0.0)
    row3 = out[out["key_mlbam"] == 3].iloc[0]
    assert row3["lineup_status"] == "confirmed"


def test_changed_batting_order_detected_and_applied():
    previous = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=1),
    ]))
    current = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=5,
              fetched_at_utc="2026-09-03T19:00:00Z"),
    ]))
    assert lineup_snapshots.changed_game_pks(previous, current) == {100}

    pool = pd.DataFrame([_pool_row(1, 100, batting_order=1.0)])
    out = lineup_snapshots.apply_confirmed_lineup_to_pool(pool, current)
    assert out.iloc[0]["batting_order"] == 5
    assert out.iloc[0]["avg_batting_order"] == 5


def test_doubleheader_lineups_are_independent_by_game_pk():
    snaps = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=101, team="NYY", opponent="BOS", key_mlbam=1, batting_order=1,
              game_datetime="2026-09-03T17:05:00Z"),
        _snap(game_pk=102, team="NYY", opponent="BOS", key_mlbam=1, batting_order=7,
              game_datetime="2026-09-03T23:10:00Z"),
    ]))
    pool = pd.DataFrame([_pool_row(1, 101), _pool_row(1, 102)])
    out = lineup_snapshots.apply_confirmed_lineup_to_pool(pool, snaps)
    g1 = out[out["game_pk"] == 101].iloc[0]
    g2 = out[out["game_pk"] == 102].iloc[0]
    assert g1["batting_order"] == 1
    assert g2["batting_order"] == 7


def test_idempotent_snapshot_persist_and_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LINEUP_SNAPSHOT_AUDIT_PATH", str(tmp_path / "audit.csv"))
    monkeypatch.setattr(config, "LINEUP_SNAPSHOT_LATEST_PATH", str(tmp_path / "latest.csv"))
    snaps = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=2),
    ]))
    fp = lineup_snapshots.snapshot_content_fingerprint(snaps)
    lineup_snapshots.persist_snapshots(snaps)
    lineup_snapshots.persist_snapshots(snaps)  # idempotent content
    audit = pd.read_csv(tmp_path / "audit.csv")
    latest = pd.read_csv(tmp_path / "latest.csv")
    assert lineup_snapshots.snapshot_content_fingerprint(latest) == fp
    # Exact duplicate rows collapsed on audit; content unchanged.
    assert len(latest) == 1
    assert len(audit) >= 1
    assert lineup_snapshots.changed_game_pks(snaps, snaps) == set()


def test_filter_snapshots_no_update_after_game_start():
    now = pd.Timestamp("2026-09-03T23:15:00Z")
    snaps = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=1,
              game_datetime="2026-09-03T23:10:00Z"),  # started
        _snap(game_pk=101, team="BOS", opponent="NYY", key_mlbam=2, batting_order=1,
              game_datetime="2026-09-04T01:10:00Z"),  # still upcoming
    ]))
    kept = lineup_snapshots.filter_snapshots_to_unstarted(snaps, now_utc=now)
    assert set(kept["game_pk"].astype(int)) == {101}
    assert lineup_snapshots.game_has_started("2026-09-03T23:10:00Z", now_utc=now) is True


def test_games_in_upcoming_window_exits_empty_outside_horizon():
    now = pd.Timestamp("2026-09-03T12:00:00Z")
    schedule_df = pd.DataFrame([
        {"game_pk": 1, "game_datetime": "2026-09-03T11:00:00Z"},  # already started
        {"game_pk": 2, "game_datetime": "2026-09-04T20:00:00Z"},  # beyond 6h
    ])
    upcoming = lineup_snapshots.games_in_upcoming_window(
        schedule_df, now_utc=now, window_hours=6.0,
    )
    assert upcoming.empty


def test_immutable_audit_preserves_history(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LINEUP_SNAPSHOT_AUDIT_PATH", str(tmp_path / "audit.csv"))
    monkeypatch.setattr(config, "LINEUP_SNAPSHOT_LATEST_PATH", str(tmp_path / "latest.csv"))
    first = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=1,
              fetched_at_utc="2026-09-03T17:00:00Z"),
    ]))
    second = lineup_snapshots.normalize_snapshot_frame(pd.DataFrame([
        _snap(game_pk=100, team="NYY", opponent="BOS", key_mlbam=1, batting_order=5,
              fetched_at_utc="2026-09-03T18:00:00Z"),
    ]))
    lineup_snapshots.persist_snapshots(first)
    lineup_snapshots.persist_snapshots(second)
    audit = pd.read_csv(tmp_path / "audit.csv")
    latest = pd.read_csv(tmp_path / "latest.csv")
    assert len(audit) == 2
    assert int(latest.iloc[0]["batting_order"]) == 5
    assert set(audit["batting_order"].astype(int)) == {1, 5}


# ---------------------------------------------------------------------------
# Prediction / game-pick supersede
# ---------------------------------------------------------------------------


def test_append_game_predictions_refreshes_unresolved_same_day(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    morning = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-03"), "game_pk": 100, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.55, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA, "prediction_snapshot_type": "morning",
    }])
    game_predictions.append_game_predictions(morning, log_path)

    lock = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-03"), "game_pk": 100, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "BOS", "predicted_probability": 0.60, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA, "prediction_snapshot_type": "lineup_lock",
    }])
    result = game_predictions.append_game_predictions(lock, log_path)
    assert len(result) == 1
    assert result.iloc[0]["predicted_winner"] == "BOS"
    assert result.iloc[0]["prediction_snapshot_type"] == "lineup_lock"
    audit = pd.read_csv(tmp_path / "game_predictions_audit.csv")
    assert len(audit) == 2


def test_append_game_predictions_preserves_resolved_history(tmp_path):
    log_path = str(tmp_path / "game_predictions.csv")
    resolved = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-03"), "game_pk": 100, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "NYY", "predicted_probability": 0.55, "metric": "GamePick_Win_Probability",
        "actual_winner": "NYY", "game_played": 1, "prediction_snapshot_type": "morning",
    }])
    other = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-03"), "game_pk": 101, "home_team": "LAD", "away_team": "SF",
        "predicted_winner": "LAD", "predicted_probability": 0.58, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA, "prediction_snapshot_type": "morning",
    }])
    game_predictions.append_game_predictions(pd.concat([resolved, other]), log_path)

    refresh = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-03"), "game_pk": 100, "home_team": "NYY", "away_team": "BOS",
        "predicted_winner": "BOS", "predicted_probability": 0.70, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA, "prediction_snapshot_type": "lineup_lock",
    }, {
        "date": pd.Timestamp("2026-09-03"), "game_pk": 101, "home_team": "LAD", "away_team": "SF",
        "predicted_winner": "SF", "predicted_probability": 0.61, "metric": "GamePick_Win_Probability",
        "actual_winner": pd.NA, "game_played": pd.NA, "prediction_snapshot_type": "lineup_lock",
    }])
    result = game_predictions.append_game_predictions(refresh, log_path).set_index("game_pk")
    assert result.loc[100, "actual_winner"] == "NYY"
    assert result.loc[100, "game_played"] == 1
    assert result.loc[100, "predicted_winner"] == "NYY"
    assert result.loc[101, "predicted_winner"] == "SF"


def test_append_predictions_supersedes_unresolved_game_when_other_game_resolved(tmp_path):
    log_path = str(tmp_path / "predictions.csv")
    morning = predictions.select_picks(
        pd.DataFrame([
            {**_pool_row(1, 100), "name_first": "A", "name_last": "One"},
            {**_pool_row(2, 101), "name_first": "B", "name_last": "Two"},
        ]),
        "2026-09-03",
        top_n=2,
        min_plate_appearances=30,
        prediction_snapshot_type="morning",
    )
    predictions.append_predictions(morning, log_path)

    resolved = pd.read_csv(log_path, parse_dates=["date"])
    # Resolve only game 100 (already started / completed).
    mask = resolved["game_pk"] == 100
    resolved.loc[mask, "at_bats"] = 4
    resolved.loc[mask, "actual_hit"] = 1
    resolved.to_csv(log_path, index=False)

    lock = predictions.select_picks(
        pd.DataFrame([
            {**_pool_row(3, 101), "name_first": "C", "name_last": "Three",
             "probability": 0.95, "Game_Hit_Probability": 0.95, "Matchup_Hit_Probability": 0.95},
        ]),
        "2026-09-03",
        top_n=1,
        min_plate_appearances=30,
        prediction_snapshot_type="lineup_lock",
    )
    combined = predictions.append_predictions(lock, log_path)
    g100 = combined[combined["game_pk"] == 100]
    g101 = combined[combined["game_pk"] == 101]
    assert len(g100) == 1
    assert g100.iloc[0]["actual_hit"] == 1
    assert set(g101["key_mlbam"].astype(int)) == {3}
    assert g101.iloc[0]["prediction_snapshot_type"] == "lineup_lock"


def test_fetch_lineup_snapshots_gated_until_schema_confirmed(monkeypatch):
    monkeypatch.setattr(config, "LINEUP_API_SCHEMA_CONFIRMED", False)
    empty = lineup_snapshots.fetch_lineup_snapshots("2026-09-03")
    assert empty.empty
