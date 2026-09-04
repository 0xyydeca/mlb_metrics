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
)


@pytest.fixture(autouse=True)
def _isolate_production_prediction_paths(tmp_path, monkeypatch):
    """Redirect append-only prediction snapshot/log paths into tmp_path.

    Prevents tests and debug pipeline runs from contaminating the
    repository's ``data/predictions/`` production files with fixtures.
    """
    pred_dir = tmp_path / "isolated_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "MARKET_ODDS_SNAPSHOTS_PATH": pred_dir / "market_odds_snapshots.csv",
        "SCHEDULE_SNAPSHOTS_PATH": pred_dir / "schedule_snapshots.csv",
        "GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH": pred_dir / "game_residual_shadow_predictions.csv",
        "GAME_RESIDUAL_SHADOW_BETS_PATH": pred_dir / "game_residual_shadow_bets.csv",
        "MARKET_ODDS_EVENT_MAP_PATH": pred_dir / "market_odds_event_map.csv",
    }
    for attr, path in mapping.items():
        monkeypatch.setattr(config, attr, str(path))
    return mapping


@pytest.fixture
def repo_predictions_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "predictions"
