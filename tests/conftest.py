"""Shared pytest fixtures — keep tests out of production data/predictions."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlb_metrics import config


PRODUCTION_PREDICTION_PATH_ATTRS = (
    "MARKET_ODDS_SNAPSHOTS_PATH",
    "SCHEDULE_SNAPSHOTS_PATH",
    "GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH",
    "GAME_RESIDUAL_SHADOW_BETS_PATH",
    "MARKET_ODDS_EVENT_MAP_PATH",
    "STREAK_POLICY_SHADOW_DECISIONS_PATH",
    "LINEUP_LOCK_SHADOW_DECISIONS_PATH",
    "LINEUP_SNAPSHOT_AUDIT_PATH",
    "LINEUP_SNAPSHOT_LATEST_PATH",
    "PREDICTIONS_AUDIT_PATH",
    "GAME_PREDICTIONS_AUDIT_PATH",
    "HITTER_OPPORTUNITY_SHADOW_PREDICTIONS_PATH",
    "DFS_COMPLETE_SHADOW_PREDICTIONS_PATH",
    "NESTED_VALIDATION_REPORT_DIR",
    "GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH",
    "BETTING_PROMOTION_GATE_REPORT_PATH",
    "STREAK_POLICY_PROMOTION_GATE_REPORT_PATH",
)


@pytest.fixture(autouse=True)
def _isolate_production_prediction_paths(tmp_path, monkeypatch):
    """Redirect append-only prediction/report paths into tmp_path.

    Prevents tests and debug pipeline runs from contaminating the
    repository's ``data/predictions/`` and ``reports/model_validation/``
    production files with fixtures.
    """
    pred_dir = tmp_path / "isolated_predictions"
    report_dir = tmp_path / "isolated_reports" / "model_validation"
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "MARKET_ODDS_SNAPSHOTS_PATH": pred_dir / "market_odds_snapshots.csv",
        "SCHEDULE_SNAPSHOTS_PATH": pred_dir / "schedule_snapshots.csv",
        "GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH": pred_dir / "game_residual_shadow_predictions.csv",
        "GAME_RESIDUAL_SHADOW_BETS_PATH": pred_dir / "game_residual_shadow_bets.csv",
        "MARKET_ODDS_EVENT_MAP_PATH": pred_dir / "market_odds_event_map.csv",
        "STREAK_POLICY_SHADOW_DECISIONS_PATH": pred_dir / "streak_policy_shadow_decisions.csv",
        "LINEUP_LOCK_SHADOW_DECISIONS_PATH": pred_dir / "lineup_lock_runs.csv",
        "LINEUP_SNAPSHOT_AUDIT_PATH": pred_dir / "lineup_snapshots_audit.csv",
        "LINEUP_SNAPSHOT_LATEST_PATH": pred_dir / "lineup_snapshots_latest.csv",
        "PREDICTIONS_AUDIT_PATH": pred_dir / "predictions_audit.csv",
        "GAME_PREDICTIONS_AUDIT_PATH": pred_dir / "game_predictions_audit.csv",
        "HITTER_OPPORTUNITY_SHADOW_PREDICTIONS_PATH": pred_dir / "hitter_opportunity_shadow.csv",
        "DFS_COMPLETE_SHADOW_PREDICTIONS_PATH": pred_dir / "dfs_complete_shadow.csv",
        "NESTED_VALIDATION_REPORT_DIR": report_dir,
        "GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH": report_dir / "game_residual_nested.json",
        "BETTING_PROMOTION_GATE_REPORT_PATH": report_dir / "game_residual_betting_gate.json",
        "STREAK_POLICY_PROMOTION_GATE_REPORT_PATH": report_dir / "streak_policy_nested.json",
    }
    for attr, path in mapping.items():
        monkeypatch.setattr(config, attr, str(path))
    return mapping


@pytest.fixture
def repo_predictions_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "predictions"
