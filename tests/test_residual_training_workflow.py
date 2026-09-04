"""Tests for residual training inventory / status reporting and workflow wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_metrics import config, game_residual_model as grm, market_odds

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "game_residual_training.yml"


def _snap(
    *,
    game_pk,
    role,
    status,
    captured,
    start,
    date="2026-09-03",
    snapshot_id="odds_x",
):
    return {
        "provider_event_id": f"e{game_pk}",
        "sportsbook": "DraftKings",
        "captured_at_utc": captured,
        "game_datetime": start,
        "date": pd.Timestamp(date),
        "home_team": "NYY",
        "away_team": "BOS",
        "home_moneyline": -130,
        "away_moneyline": 110,
        "market_home_win_probability": 0.55,
        "source_status": status,
        "game_pk": game_pk,
        "match_method": "unique_matchup",
        "snapshot_id": snapshot_id,
        "snapshot_role": role,
    }


def test_market_snapshot_inventory_counts_roles_and_valid_prediction_games():
    snaps = pd.DataFrame([
        _snap(
            game_pk=1, role=market_odds.SNAPSHOT_ROLE_MORNING, status=market_odds.SOURCE_OK,
            captured="2026-09-03T12:00:00Z", start="2026-09-03T23:00:00Z",
            snapshot_id="m1",
        ),
        _snap(
            game_pk=1, role=market_odds.SNAPSHOT_ROLE_LINEUP_LOCK, status=market_odds.SOURCE_OK,
            captured="2026-09-03T20:00:00Z", start="2026-09-03T23:00:00Z",
            snapshot_id="l1",
        ),
        _snap(
            game_pk=1, role=market_odds.SNAPSHOT_ROLE_INTRADAY, status=market_odds.SOURCE_POST_START,
            captured="2026-09-03T23:30:00Z", start="2026-09-03T23:00:00Z",
            snapshot_id="i1",
        ),
        _snap(
            game_pk=2, role=market_odds.SNAPSHOT_ROLE_INTRADAY, status=market_odds.SOURCE_OK,
            captured="2026-09-04T12:00:00Z", start="2026-09-04T23:00:00Z", date="2026-09-04",
            snapshot_id="i2",
        ),
    ])
    inv = market_odds.market_snapshot_inventory(snaps)
    assert inv["n_total"] == 4
    assert inv["n_source_ok"] == 3
    assert inv["n_source_post_start"] == 1
    assert inv["n_morning"] == 1
    assert inv["n_lineup_lock"] == 1
    assert inv["n_intraday"] == 2
    # game 1 morning+lock are valid prediction-time; intraday is not a pred role;
    # game 2 intraday SOURCE_OK is not in PREDICTION_TIME_ROLES.
    assert inv["n_valid_prediction_games"] == 1
    assert inv["n_valid_prediction_dates"] == 1
    assert inv["n_games_with_closing"] >= 1  # pre-start ok rows for game 1


def test_residual_training_status_labels_not_enough_data_without_reports(tmp_path):
    snaps = market_odds.empty_snapshot_frame()
    status = grm.residual_training_status(
        snapshots=snaps,
        nested_report_path=str(tmp_path / "missing_nested.json"),
        betting_report_path=str(tmp_path / "missing_betting.json"),
        model_path=str(tmp_path / "missing.joblib"),
    )
    assert status["valid_prediction_games"] == 0
    assert status["valid_dates"] == 0
    assert status["outer_folds_possible"] == 0
    assert status["residual_artifact"] == "no"
    assert status["probability_gate"] == "not_enough_data"
    assert status["betting_gate"] == "not_enough_data"
    assert status["game_prediction_mode"] == "shadow"
    assert status["betting_mode"] == "disabled"


def test_residual_training_status_reads_gate_pass_fail(tmp_path):
    nested = tmp_path / "nested.json"
    nested.write_text(
        '{"status":"ok","promotion_gate":{"passed":false,"checks":{"adequate_sample_size":false}},'
        '"betting_promotion_gate":{"passed":false,"checks":{"adequate_bets":false}}}',
        encoding="utf-8",
    )
    status = grm.residual_training_status(
        snapshots=market_odds.empty_snapshot_frame(),
        nested_report_path=str(nested),
        betting_report_path=str(nested),
        model_path=str(tmp_path / "missing.joblib"),
    )
    assert status["probability_gate"] == "fail"
    assert status["betting_gate"] == "fail"


def test_estimate_outer_folds_possible_requires_configured_history(monkeypatch):
    monkeypatch.setattr(config, "GAME_RESIDUAL_OUTER_MIN_TRAIN_DATES", 5)
    monkeypatch.setattr(config, "GAME_RESIDUAL_OUTER_TEST_BLOCK_DATES", 2)
    monkeypatch.setattr(config, "GAME_RESIDUAL_INNER_MIN_TRAIN_DATES", 3)
    monkeypatch.setattr(config, "GAME_RESIDUAL_INNER_TEST_BLOCK_DATES", 1)
    monkeypatch.setattr(config, "GAME_RESIDUAL_BETTING_FREEZE_DATES", 0)
    short = [f"2026-05-{i:02d}" for i in range(1, 6)]
    assert grm.estimate_outer_folds_possible(short) == 0
    long = [f"2026-05-{i:02d}" for i in range(1, 16)]
    assert grm.estimate_outer_folds_possible(long) >= 1


def test_game_residual_training_workflow_yaml():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert WORKFLOW_PATH.exists()
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert "0 14 * * 1" in text  # Monday 14:00 UTC
    assert "pip install -r requirements-dev.txt" in text
    assert "pytest" in text
    assert "check_betting_readiness.py --skip-pytest" in text
    assert "report_residual_training_status.py" in text
    assert "train_game_residual_model.py" in text
    assert "force-save" not in text
    assert "GAME_PREDICTION_MODE" in text
    assert "BETTING_MODE" in text
    assert "reports/model_validation/" in text
    assert "game_residual_win_probability_model.joblib" in text
    assert "git pull --rebase" in text
    # Informational readiness must not abort (exit 0 after capturing status).
    assert "exit 0" in text


def test_modes_remain_shadow_disabled():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"


def test_report_residual_training_status_script_runs(tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys

    odds = tmp_path / "odds.csv"
    market_odds.empty_snapshot_frame().to_csv(odds, index=False)
    monkeypatch.setattr(config, "MARKET_ODDS_SNAPSHOTS_PATH", str(odds))
    monkeypatch.setattr(config, "GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH", str(tmp_path / "n.json"))
    monkeypatch.setattr(config, "BETTING_PROMOTION_GATE_REPORT_PATH", str(tmp_path / "b.json"))
    monkeypatch.setattr(config, "GAME_RESIDUAL_MODEL_PATH", str(tmp_path / "m.joblib"))

    script = REPO_ROOT / "scripts" / "report_residual_training_status.py"
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("report_residual_training_status", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["report_residual_training_status.py", "--odds-snapshots", str(odds)])
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "MARKET SNAPSHOT INVENTORY:" in out
    assert "RESIDUAL TRAINING STATUS:" in out
    assert "residual_artifact=no" in out
