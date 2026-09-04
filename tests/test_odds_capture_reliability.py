"""Tests for odds-capture reliability helpers and workflow wiring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mlb_metrics import market_odds

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "market_odds_capture.yml"
HEALTH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "market_odds_capture_health.yml"
HEALTH_SCRIPT = REPO_ROOT / "scripts" / "check_odds_capture_health.py"


def test_market_odds_capture_workflow_is_hardened_for_schedule_reliability():
    text = CAPTURE_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert "17,47 15-23 * * *" in text
    assert "cancel-in-progress: false" in text
    assert "requirements-odds-capture.txt" in text
    assert "requirements-dev.txt" not in text
    assert "git pull --rebase" in text
    assert "capture_market_odds.py" in text
    assert "timeout-minutes: 15" in text
    # Capture pipeline only — no residual training.
    assert "train_game_residual" not in text


def test_market_odds_capture_health_workflow_exists():
    text = HEALTH_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "check_odds_capture_health.py" in text
    assert "train_game_residual" not in text


def test_capture_health_ok_outside_window_even_if_stale(tmp_path):
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("check_odds_capture_health", HEALTH_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    snaps = market_odds.empty_snapshot_frame()
    # 12:00 UTC is outside 15-23 / 0-2 capture hours.
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    result = mod.evaluate_capture_health(
        max_age_minutes=90,
        snapshots=snaps,
        now_utc=now,
    )
    assert result["in_capture_window"] is False
    assert result["ok"] is True
    assert result["stale"] is False


def test_capture_health_stale_inside_window_without_recent_rows(tmp_path):
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("check_odds_capture_health", HEALTH_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    snaps = pd.DataFrame([{
        "provider_event_id": "e1",
        "sportsbook": "DraftKings",
        "captured_at_utc": "2026-09-04T14:00:00Z",  # 3+ hours before
        "game_datetime": "2026-09-04T23:00:00Z",
        "date": "2026-09-04",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_moneyline": -120,
        "away_moneyline": 100,
        "market_home_win_probability": 0.55,
        "source_status": "ok",
        "game_pk": 1,
        "match_method": "unique_matchup",
        "snapshot_id": "odds_old",
        "snapshot_role": "intraday",
    }])
    now = datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc)  # inside window
    result = mod.evaluate_capture_health(
        max_age_minutes=90,
        snapshots=snaps,
        now_utc=now,
    )
    assert result["in_capture_window"] is True
    assert result["stale"] is True
    assert result["ok"] is False


def test_capture_health_fresh_inside_window():
    import importlib.util
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("check_odds_capture_health", HEALTH_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    snaps = pd.DataFrame([{
        "provider_event_id": "e1",
        "sportsbook": "DraftKings",
        "captured_at_utc": "2026-09-04T17:45:00Z",
        "game_datetime": "2026-09-04T23:00:00Z",
        "date": "2026-09-04",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_moneyline": -120,
        "away_moneyline": 100,
        "market_home_win_probability": 0.55,
        "source_status": "ok",
        "game_pk": 1,
        "match_method": "unique_matchup",
        "snapshot_id": "odds_fresh",
        "snapshot_role": "intraday",
    }])
    now = datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc)
    result = mod.evaluate_capture_health(
        max_age_minutes=90,
        snapshots=snaps,
        now_utc=now,
    )
    assert result["stale"] is False
    assert result["ok"] is True
