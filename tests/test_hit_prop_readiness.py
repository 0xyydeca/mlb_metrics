"""Formal hit-prop readiness evaluation fixtures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, hit_prop_forecast, hit_prop_readiness as readiness


def test_pre_score_audit_and_insufficient_verdict():
    report = readiness.build_readiness_report(allow_scoring=True)
    assert report["verdict"] == readiness.VERDICT_INSUFFICIENT
    assert report["validation_status"] == "insufficient_data"
    assert report["edge_claimed"] is False
    assert report["betting_mode_required"] == "disabled"
    assert config.BETTING_MODE == "disabled"
    assert report["scoring_attempted"] is False
    assert "structural_floor_or_outer_folds_not_met" in report["scoring_blocked_reasons"]
    assert report["pre_score_audit"]["status"] == "passed"
    assert report["structural_and_precision"]["structural"]["ready"] is False
    assert report["structural_and_precision"]["statistical_precision"]["ready"] is False
    assert report["software_tests_are_not_evidence_of_edge"] is True
    assert report["uncertainty"]["thresholds_lowered"] is False
    assert report["additional_observations_needed"]["additional_dates_to_structural_floor"] > 0
    assert report["next_permitted_checkpoint"]["next_registered_review_is_betting_launch"] is False


def test_structural_precision_separated():
    s = readiness.structural_and_precision_status(2, {})
    assert s["structural"]["ready"] is False
    assert s["statistical_precision"]["ready"] is False
    assert s["scoring_allowed"] is False
    s70 = readiness.structural_and_precision_status(70, {"sample_size_plan": {"approx_independent_days_for_0.01_logloss_style_target": 197}})
    # 70 may or may not yield 3 outer folds depending on fold builder; scoring_allowed tracks that.
    assert s70["structural"]["floor_dates"] == 70
    assert s70["statistical_precision"]["planning_power_independent_dates"] == 197
    assert s70["statistical_precision"]["ready"] is False


def test_probability_scoring_identical_opportunities_and_inconclusive_ci():
    rng = np.random.default_rng(1)
    rows = []
    for d in range(5):
        date = f"2026-06-{d+1:02d}"
        for i in range(20):
            mid = float(rng.uniform(0.4, 0.6))
            # Model nearly identical to market → inconclusive improvement CI.
            model = float(np.clip(mid + rng.normal(0, 0.01), 0.05, 0.95))
            y = int(rng.random() < mid)
            rows.append(
                {
                    "date": date,
                    "game_pk": 1000 + d,
                    "key_mlbam": 2000 + i,
                    "y_binary": y,
                    "market_mid_probability": mid,
                    "model_probability": model,
                }
            )
    frame = pd.DataFrame(rows)
    scored = readiness.score_probability_quality(frame)
    assert scored["status"] == "ok"
    assert hit_prop_forecast.CANDIDATE_MARKET_MID in scored["candidates"]
    assert "log_loss" in scored["candidates"][hit_prop_forecast.CANDIDATE_MARKET_MID]
    assert "brier" in scored["candidates"][hit_prop_forecast.CANDIDATE_MARKET_MID]
    assert scored["paired_bootstrap_model_minus_market_log_loss"]["n_blocks"] == 5
    # Near-identical model should not claim conclusive improvement.
    assert scored["conclusive_log_loss_improvement_vs_market"] is False
    assert scored["inconclusive_ci_treated_as"] == readiness.VERDICT_INSUFFICIENT


def test_map_verdicts():
    assert (
        readiness.map_gate_verdict({"verdict": "insufficient_evidence"}, {"status": "insufficient_data"})
        == readiness.VERDICT_INSUFFICIENT
    )
    assert (
        readiness.map_gate_verdict({"verdict": "edge_not_supported"}, {"status": "ok"})
        == readiness.VERDICT_UNSUPPORTED
    )
    assert (
        readiness.map_gate_verdict(
            {"verdict": "edge_supported"},
            {"status": "ok", "conclusive_log_loss_improvement_vs_market": False},
        )
        == readiness.VERDICT_INSUFFICIENT
    )
    assert (
        readiness.map_gate_verdict(
            {"verdict": "edge_supported"},
            {"status": "ok", "conclusive_log_loss_improvement_vs_market": True},
        )
        == readiness.VERDICT_SUPPORTED
    )


def test_write_readiness_report(tmp_path, monkeypatch):
    path = tmp_path / "readiness.json"
    monkeypatch.setattr(config, "HIT_PROP_READINESS_REPORT_PATH", str(path))
    report = readiness.build_readiness_report()
    out = readiness.write_readiness_report(report)
    assert out == str(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verdict"] == readiness.VERDICT_INSUFFICIENT
    assert loaded["report_hash"]


def test_modes_unchanged():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
