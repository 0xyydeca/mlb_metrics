"""Hit-prop forecast + paper evaluation fixtures (research-only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, hit_prop_forecast as forecast
from mlb_metrics import hit_prop_paper as paper
from mlb_metrics import hit_prop_research as research


def test_positive_ab_source_rejected_and_components_required():
    rejected = forecast.contract_yes_probability(
        p_qualify=1.0,
        p_hit_given_qualify=0.33,
        source="positive_ab_hit_rate",
    )
    assert rejected["contract_yes_probability"] is None
    assert rejected["status"] == "rejected_positive_ab_source"

    incomplete = forecast.baseball_proxy_components(game_hit_probability=0.62)
    assert incomplete["contract_yes_probability"] is None
    assert incomplete["status"] == "incomplete_components"

    ok = forecast.baseball_proxy_components(start_rate=0.9, game_hit_probability=0.5)
    assert ok["status"] == "ok"
    assert ok["contract_yes_probability"] == pytest.approx(0.45)


def test_missing_market_mid_stays_missing():
    assert forecast.market_mid_from_executable(None, 0.4) is None
    assert forecast.market_mid_from_executable(0.6, None) is None
    assert forecast.market_mid_from_executable(0.8, 0.1) is None  # crossed reconstruction
    mid = forecast.market_mid_from_executable(0.60, 0.44)
    assert mid == pytest.approx(0.58)


def test_walk_only_and_lfmp_labels_for_eval():
    rules = research.parse_contract_rules(
        "starting lineup and plate appearance; otherwise last fair market price"
    )
    walk = research.classify_contract_outcome(
        started=True, plate_appearances=1, at_bats=0, hits=0, game_status="Final", rules=rules
    )
    label = forecast.binary_label_from_settlement(walk)
    assert label["in_binary_eval"] is True
    assert label["y_binary"] == 0

    dnp = research.classify_contract_outcome(
        started=False, plate_appearances=0, at_bats=0, hits=0, game_status="Final", rules=rules
    )
    nonbinary = forecast.binary_label_from_settlement(dnp)
    assert nonbinary["in_binary_eval"] is False
    assert nonbinary["y_binary"] is None


def test_training_only_walk_forward_tuning():
    rng = np.random.default_rng(0)
    rows = []
    for i, date in enumerate(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]):
        for _ in range(40):
            mid = float(rng.uniform(0.35, 0.65))
            bb = float(np.clip(mid + rng.normal(0, 0.05), 0.05, 0.95))
            y = int(rng.random() < mid)
            rows.append(
                {
                    "date": date,
                    "y_binary": y,
                    "market_mid_probability": mid,
                    "baseball_contract_probability": bb,
                }
            )
    frame = pd.DataFrame(rows)
    result = forecast.chronological_train_calibrate_predict(frame, min_train_rows=30)
    assert result["status"] in {"ok", "insufficient_data"}
    assert result["tuning_scope"] == "chronological_training_folds_only"
    assert result["holdout_untouched"] is True
    if result["status"] == "ok":
        assert result["n_dates_scored"] >= 1
        assert "comparison" in result


def test_gates_report_insufficient_without_opening_nested(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HIT_PROP_FROZEN_POLICY_PATH", str(tmp_path / "frozen.json"))
    monkeypatch.setattr(config, "HIT_PROP_FORECAST_EXAMPLE_PATH", str(tmp_path / "example.json"))
    monkeypatch.setattr(
        config, "HIT_PROP_PAPER_EVALUATION_REPORT_PATH", str(tmp_path / "eval.json")
    )
    protocol_path = tmp_path / "protocol.json"
    monkeypatch.setattr(config, "HIT_PROP_PAPER_PROTOCOL_PATH", str(protocol_path))
    protocol = {
        "protocol_id": config.HIT_PROP_PAPER_PROTOCOL_ID,
        "protocol_version": "1",
        "registered_at_utc": config.HIT_PROP_PAPER_PROTOCOL_REGISTERED_UTC,
        "protocol_hash": "fixture",
        "candidates": [
            "same_time_market_mid_baseline",
            "contract_rule_adjusted_hitter_hit_model",
            "regularized_market_residual_prop_logistic",
        ],
        "sample_size_plan": {
            "structural_floor_independent_dates": config.HIT_PROP_MIN_ELIGIBLE_DATES,
            "approx_independent_days_for_0.01_logloss_style_target": 197,
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    report = paper.build_evaluation_report(open_nested_outcomes=False)
    assert report["nested_evaluation_outcomes"] == "not_opened"
    assert report["validation_status"] == "insufficient_data"
    assert report["verdict"] == "insufficient_evidence"
    assert report["edge_claimed"] is False
    assert report["september_25_is_paper_system_review_not_betting_launch"] is True
    assert report["additional_observations_needed"]["additional_dates_to_structural_floor"] > 0
    path = paper.write_evaluation_report(report)
    assert path.endswith("eval.json")
    loaded = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    assert loaded["validation_status"] == "insufficient_data"

    with pytest.raises(ValueError, match="remain closed"):
        paper.build_evaluation_report(open_nested_outcomes=True)


def test_forecast_settlement_example_and_paper_trade():
    example = paper.build_forecast_settlement_example()
    assert example["edge_claimed"] is False
    assert example["forecasts"]["rejected_positive_ab_source"]["status"] == "rejected_positive_ab_source"
    yes_trade = example["paper_trades"]["binary_yes_fill"]
    assert yes_trade["filled_qty"] == 1.0
    assert yes_trade["position"]["status"] == "settled"
    assert yes_trade["position"]["settlement_payout_per_contract"] == 1.0
    walk = example["paper_trades"]["walk_only_binary_no"]
    assert walk["position"]["settlement_payout_per_contract"] == 0.0
    dnp = example["paper_trades"]["dnp_lfmp_with_conservative_stress"]
    assert dnp["position"]["settlement_rule"] in {"venue_settlement_px", "last_fair_market_price"}
    assert example["paper_trades"]["skipped_unfilled"]["action"] == "pass"


def test_modes_unchanged():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    assert config.HIT_PROP_PAPER_PROTOCOL_ID.startswith("polymarket_us_mlb_hitter_hits")
