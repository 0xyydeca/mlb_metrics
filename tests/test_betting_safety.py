"""Betting-safety remediation: gates, snapshot hygiene, path isolation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, game_residual_model as grm, market_odds

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_predictions_dir_not_written_by_isolated_config(tmp_path, repo_predictions_dir):
    """Autouse fixture must redirect snapshot paths away from production."""
    assert Path(config.MARKET_ODDS_SNAPSHOTS_PATH).is_relative_to(tmp_path) or str(tmp_path) in config.MARKET_ODDS_SNAPSHOTS_PATH
    assert Path(config.SCHEDULE_SNAPSHOTS_PATH).is_relative_to(tmp_path) or str(tmp_path) in config.SCHEDULE_SNAPSHOTS_PATH

    before_odds = (
        (repo_predictions_dir / "market_odds_snapshots.csv").read_bytes()
        if (repo_predictions_dir / "market_odds_snapshots.csv").exists()
        else b""
    )
    before_sched = (
        (repo_predictions_dir / "schedule_snapshots.csv").read_bytes()
        if (repo_predictions_dir / "schedule_snapshots.csv").exists()
        else b""
    )

    # Write through config paths (isolated) — must not touch production bytes.
    market_odds.append_odds_snapshots(pd.DataFrame([{
        "provider_event_id": "x",
        "sportsbook": "DraftKings",
        "captured_at_utc": "2026-09-03T12:00:00Z",
        "game_datetime": "2026-09-03T23:00:00Z",
        "date": "2026-09-03",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_moneyline": -150,
        "away_moneyline": 130,
        "market_home_win_probability": 0.58,
        "source_status": "ok",
        "game_pk": 999,
        "match_method": "test",
        "snapshot_id": "odds_test_iso",
        "snapshot_role": "morning",
    }]))

    after_odds = (
        (repo_predictions_dir / "market_odds_snapshots.csv").read_bytes()
        if (repo_predictions_dir / "market_odds_snapshots.csv").exists()
        else b""
    )
    after_sched = (
        (repo_predictions_dir / "schedule_snapshots.csv").read_bytes()
        if (repo_predictions_dir / "schedule_snapshots.csv").exists()
        else b""
    )
    assert after_odds == before_odds
    assert after_sched == before_sched
    assert Path(config.MARKET_ODDS_SNAPSHOTS_PATH).exists()


def test_filter_rejects_unmatched_missing_pk_post_start_and_historical_morning():
    start = "2026-06-20T23:10:00Z"
    rows = pd.DataFrame([
        {
            "provider_event_id": "1", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-06-20T12:00:00Z", "game_datetime": start,
            "date": "2026-06-20", "home_team": "NYY", "away_team": "BOS",
            "home_moneyline": -150, "away_moneyline": 130,
            "market_home_win_probability": 0.58, "source_status": "ok",
            "game_pk": 100, "match_method": "teams_date",
            "snapshot_id": "ok1", "snapshot_role": "morning",
        },
        {
            "provider_event_id": "2", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-09-04T02:00:00Z", "game_datetime": start,
            "date": "2026-06-20", "home_team": "NYY", "away_team": "BOS",
            "home_moneyline": -150, "away_moneyline": 130,
            "market_home_win_probability": 0.58, "source_status": "unmatched",
            "game_pk": pd.NA, "match_method": pd.NA,
            "snapshot_id": "bad_unmatched", "snapshot_role": "morning",
        },
        {
            "provider_event_id": "3", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-06-20T23:30:00Z", "game_datetime": start,
            "date": "2026-06-20", "home_team": "NYY", "away_team": "BOS",
            "home_moneyline": -150, "away_moneyline": 130,
            "market_home_win_probability": 0.58, "source_status": "ok",
            "game_pk": 100, "match_method": "teams_date",
            "snapshot_id": "bad_post", "snapshot_role": "morning",
        },
        {
            "provider_event_id": "4", "sportsbook": "DraftKings",
            "captured_at_utc": "2026-09-04T02:00:00Z", "game_datetime": start,
            "date": "2026-06-20", "home_team": "DET", "away_team": "CWS",
            "home_moneyline": -130, "away_moneyline": 110,
            "market_home_win_probability": 0.54, "source_status": "ok",
            "game_pk": 200, "match_method": "teams_date",
            "snapshot_id": "bad_hist", "snapshot_role": "morning",
        },
    ])
    valid = market_odds.filter_valid_prediction_time_snapshots(rows, required_role="morning")
    assert list(valid["snapshot_id"]) == ["ok1"]


def test_coerce_role_never_relabels_postgame_fetch_as_morning():
    start = "2026-06-20T23:10:00Z"
    rows = pd.DataFrame([{
        "provider_event_id": "1", "sportsbook": "DraftKings",
        "captured_at_utc": "2026-09-04T02:00:00Z", "game_datetime": start,
        "date": "2026-06-20", "home_team": "NYY", "away_team": "BOS",
        "home_moneyline": -150, "away_moneyline": 130,
        "market_home_win_probability": 0.58, "source_status": "ok",
        "game_pk": 100, "match_method": "teams_date",
        "snapshot_id": "x", "snapshot_role": "raw",
    }])
    coerced = market_odds.coerce_requested_snapshot_role(rows, "morning")
    assert coerced.iloc[0]["snapshot_role"] == "raw"
    assert coerced.iloc[0]["source_status"] in {
        market_odds.SOURCE_HISTORICAL_POSTGAME,
        market_odds.SOURCE_POST_START,
    }


def test_failed_probability_gate_cannot_overwrite_production_artifact(tmp_path, monkeypatch):
    train = _load_script("train_game_residual_model", REPO_ROOT / "scripts" / "train_game_residual_model.py")
    prod = tmp_path / "game_residual_win_probability_model.joblib"
    monkeypatch.setattr(config, "GAME_RESIDUAL_MODEL_PATH", str(prod))

    # Pre-existing production-looking artifact must survive a failed gate.
    prod.write_text("EXISTING_ARTIFACT", encoding="utf-8")
    frame = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=5, freq="D").repeat(2),
        "game_pk": range(10),
        grm.HOME_WON_LABEL: [1, 0] * 5,
        grm.MARKET_AT_PRED_COL: [0.55] * 10,
        **{c: 0.0 for c in grm.RESIDUAL_FEATURE_COLUMNS},
    })
    # Minimal enrich columns used by fit
    for col in grm.RESIDUAL_FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0

    report = {
        "promotion_gate": {"passed": False, "checks": {"x": False}},
        "selected_configs": [{"logistic_C": 0.01}],
        "feature_columns": list(grm.RESIDUAL_FEATURE_COLUMNS),
        "n_outer_folds": 0,
        "n_games_evaluated": 0,
    }
    saved = train.maybe_save_residual_artifact(
        report, frame, prediction_snapshot_role="morning",
    )
    assert saved is None
    assert prod.read_text(encoding="utf-8") == "EXISTING_ARTIFACT"


def test_force_save_for_debug_refuses_production_path(tmp_path, monkeypatch):
    train = _load_script("train_game_residual_model", REPO_ROOT / "scripts" / "train_game_residual_model.py")
    prod = tmp_path / "models" / "game_residual_win_probability_model.joblib"
    prod.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "GAME_RESIDUAL_MODEL_PATH", str(prod))
    with pytest.raises(SystemExit, match="Refusing --force-save-for-debug"):
        train._assert_debug_path_not_production(str(prod))


def test_force_save_for_debug_writes_only_debug_path(tmp_path, monkeypatch):
    train = _load_script("train_game_residual_model", REPO_ROOT / "scripts" / "train_game_residual_model.py")
    prod = tmp_path / "models" / "game_residual_win_probability_model.joblib"
    debug = tmp_path / "debug" / "residual_DEBUG_ONLY.joblib"
    prod.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "GAME_RESIDUAL_MODEL_PATH", str(prod))
    prod.write_text("EXISTING", encoding="utf-8")

    frame = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=8, freq="D"),
        "game_pk": range(8),
        grm.HOME_WON_LABEL: [1, 0, 1, 0, 1, 0, 1, 0],
        grm.MARKET_AT_PRED_COL: [0.55] * 8,
        **{c: 0.1 for c in grm.RESIDUAL_FEATURE_COLUMNS},
    })
    report = {
        "promotion_gate": {"passed": False, "checks": {"x": False}},
        "selected_configs": [{"logistic_C": 0.01}],
        "feature_columns": list(grm.RESIDUAL_FEATURE_COLUMNS),
        "n_outer_folds": 0,
        "n_games_evaluated": 0,
    }
    saved = train.maybe_save_residual_artifact(
        report, frame, prediction_snapshot_role="morning",
        force_save_for_debug=str(debug),
    )
    assert saved == str(debug)
    assert debug.exists()
    assert prod.read_text(encoding="utf-8") == "EXISTING"


def test_recommend_bets_cli_cannot_bypass_disabled_betting_mode(monkeypatch):
    module = _load_script("recommend_bets", REPO_ROOT / "scripts" / "recommend_bets.py")
    monkeypatch.setattr(config, "BETTING_MODE", "disabled")
    with pytest.raises(SystemExit, match="READY FOR LIVE BETTING: NO"):
        module.assert_live_betting_cli_allowed()


def test_recommend_bets_cli_cannot_bypass_shadow_betting_mode(monkeypatch):
    module = _load_script("recommend_bets", REPO_ROOT / "scripts" / "recommend_bets.py")
    monkeypatch.setattr(config, "BETTING_MODE", "shadow")
    with pytest.raises(SystemExit, match="READY FOR LIVE BETTING: NO"):
        module.assert_live_betting_cli_allowed()


def test_recommend_bets_main_exits_before_printing_real_recommendations(monkeypatch, capsys):
    module = _load_script("recommend_bets", REPO_ROOT / "scripts" / "recommend_bets.py")
    monkeypatch.setattr(config, "BETTING_MODE", "disabled")
    monkeypatch.setattr(
        sys, "argv",
        ["recommend_bets.py", "--date", "2026-06-20"],
    )
    with pytest.raises(SystemExit):
        module.main()
    out = capsys.readouterr().out
    assert "Real bet recommendations" not in out


def test_config_modes_remain_shadow_disabled():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"


def test_production_snapshot_files_have_no_june_fixture_rows(repo_predictions_dir):
    odds = repo_predictions_dir / "market_odds_snapshots.csv"
    sched = repo_predictions_dir / "schedule_snapshots.csv"
    if odds.exists():
        df = pd.read_csv(odds)
        assert len(df) == 0 or not (df.get("source_status") == "unmatched").any()
        if "game_pk" in df.columns:
            assert df["game_pk"].notna().all() if len(df) else True
        # Specifically: no Sep-captured June rows labeled morning unmatched
        if len(df):
            assert not (
                (df["snapshot_role"] == "morning")
                & (df["source_status"] == "unmatched")
            ).any()
    if sched.exists():
        df = pd.read_csv(sched)
        if len(df) and "game_pk" in df.columns:
            assert not (df["game_pk"] == 100).any()
        if len(df) and "home_probable_pitcher_key_mlbam" in df.columns:
            assert not (
                (df["home_team"] == "NYY")
                & (df["away_team"] == "BOS")
                & (df["home_probable_pitcher_key_mlbam"] == 501)
            ).any()
