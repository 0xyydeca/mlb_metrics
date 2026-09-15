"""Paper ledger accounting and Polymarket research protocol tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlb_metrics import config, game_residual_model as grm, paper_ledger, polymarket_research


def test_modes_remain_disabled():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"


def test_hand_calculated_partial_fill_and_fees():
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    assert fee.taker_theta == pytest.approx(0.06)
    purchase = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.40, "size": 2.0}, {"price": 0.41, "size": 5.0}],
        requested_qty=3.0,
        fee=fee,
        display_price=0.10,
    )
    assert purchase["filled_qty"] == pytest.approx(3.0)
    assert purchase["acquisition_notional"] == pytest.approx(1.21)
    expected_fees = fee.taker_fee(0.40, 2.0) + fee.taker_fee(0.41, 1.0)
    assert purchase["fees_paid"] == pytest.approx(expected_fees)
    settled = paper_ledger.settle_position(
        filled_qty=3.0,
        acquisition_cost=purchase["acquisition_cost"],
        selected_team="NYY",
        winning_team="NYY",
    )
    assert settled["proceeds"] == pytest.approx(3.0)
    assert settled["net_pnl"] == pytest.approx(3.0 - purchase["acquisition_cost"])


def test_display_price_and_empty_book_do_not_fill():
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    purchase = paper_ledger.simulate_paper_purchase(
        asks=[],
        requested_qty=2.0,
        fee=fee,
        display_price=0.62,
    )
    assert purchase["status"] == "unfilled"
    assert purchase["filled_qty"] == 0.0


def test_canceled_settlement_is_not_auto_zero_loss():
    out = paper_ledger.settle_position(
        filled_qty=5.0,
        acquisition_cost=2.0,
        selected_team="LAD",
        winning_team=None,
        canceled=True,
    )
    assert out["status"] == "canceled"
    assert out["net_pnl"] is None
    assert out["open_exposure"] == pytest.approx(2.0)


def test_adverse_ticks_worsen_buy_price():
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    base = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.50, "size": 1.0}],
        requested_qty=1.0,
        fee=fee,
        adverse_ticks=0,
        tick_size=0.01,
    )
    worse = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.50, "size": 1.0}],
        requested_qty=1.0,
        fee=fee,
        adverse_ticks=5,
        tick_size=0.01,
    )
    assert worse["avg_fill_price"] == pytest.approx(base["avg_fill_price"] + 0.05)


def test_market_mid_excludes_crossed_book_and_converts_short_home():
    assert paper_ledger.market_mid_probability(best_bid=0.6, best_ask=0.55, home_is_long=True) is None
    mid = paper_ledger.market_mid_probability(best_bid=0.40, best_ask=0.42, home_is_long=False)
    assert mid == pytest.approx(1.0 - 0.41)


def test_decision_append_is_idempotent(tmp_path):
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    purchase = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.5, "size": 1.0}],
        requested_qty=1.0,
        fee=fee,
    )
    row = paper_ledger.build_decision_row(
        policy_version="t",
        venue_id="polymarket_us",
        market_id="m",
        market_slug="s",
        game_pk=9,
        side_team="NYY",
        is_long=True,
        decision_time_utc="2026-09-15T12:00:00Z",
        quote_receive_time_utc="2026-09-15T12:00:00Z",
        quote_request_time_utc="2026-09-15T12:00:00Z",
        fee=fee,
        action="buy",
        pass_reason=None,
        requested_qty=1.0,
        fill_assumption="walk_asks",
        purchase=purchase,
        model_name="m",
        model_probability=0.55,
        market_mid_probability_value=0.5,
        executable_buy=0.5,
        probability_source="test",
    )
    path = tmp_path / "decisions.csv"
    a = paper_ledger.append_decisions([row], path=str(path))
    b = paper_ledger.append_decisions([row], path=str(path))
    assert len(a) == 1
    assert len(b) == 1


def test_protocol_registers_before_outcomes_and_marks_exploratory(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    monkeypatch.setattr(config, "POLYMARKET_PAPER_PROTOCOL_PATH", str(protocol_path))
    doc = polymarket_research.register_protocol(path=str(protocol_path))
    assert doc["hypothesis"]
    assert "2026-09-14" in doc["exploratory_dates"]
    assert doc["pass_fail_criteria"]["betting_mode_if_fail"] == "disabled"
    assert doc["sample_size_plan"]["approx_independent_days"] >= 1
    doc2 = polymarket_research.register_protocol(path=str(protocol_path))
    assert doc2["protocol_hash"] == doc["protocol_hash"]


def test_market_only_fallback_excluded_from_independent_residual_scores():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "game_pk": 1,
                grm.HOME_WON_LABEL: 1,
                "market_mid_probability": 0.55,
                grm.HEURISTIC_COL: 0.60,
                grm.RESIDUAL_PROB_COL: 0.55,
                "probability_source": grm.PROBABILITY_SOURCE_MARKET_ONLY_FALLBACK,
            },
            {
                "date": "2026-09-02",
                "game_pk": 2,
                grm.HOME_WON_LABEL: 0,
                "market_mid_probability": 0.55,
                grm.HEURISTIC_COL: 0.40,
                grm.RESIDUAL_PROB_COL: 0.45,
                "probability_source": grm.PROBABILITY_SOURCE_MARKET_RESIDUAL,
            },
        ]
    )
    report = polymarket_research.compare_models_on_frame(frame)
    assert report["status"] == "ok"
    assert report["candidates"]["regularized_market_residual_logistic"]["n"] == 1


def test_gates_return_insufficient_without_history():
    protocol = polymarket_research.PaperResearchProtocol().to_dict()
    gates = polymarket_research.apply_gates(
        protocol=protocol,
        probability_report={"status": "insufficient_data"},
        strategy_report=None,
        n_eligible_dates=0,
        n_outer_folds_available=0,
    )
    assert gates["verdict"] == "insufficient_evidence"
    assert gates["validation_status"] == "insufficient_data"
    assert gates["betting_mode_required"] == "disabled"


def test_future_quote_cannot_change_earlier_decision_identity():
    early = paper_ledger.decision_id(
        policy_version="p",
        market_id="m",
        game_pk=1,
        decision_time_utc="2026-09-15T12:00:00Z",
        side_team="NYY",
        fill_assumption="walk_asks",
    )
    later = paper_ledger.decision_id(
        policy_version="p",
        market_id="m",
        game_pk=1,
        decision_time_utc="2026-09-15T12:00:00Z",
        side_team="NYY",
        fill_assumption="walk_asks",
    )
    assert early == later
    other = paper_ledger.decision_id(
        policy_version="p",
        market_id="m",
        game_pk=1,
        decision_time_utc="2026-09-15T18:00:00Z",
        side_team="NYY",
        fill_assumption="walk_asks",
    )
    assert other != early


def test_evaluation_report_round_trip(tmp_path):
    path = tmp_path / "eval.json"
    report = {
        "status": "insufficient_data",
        "validation_status": "insufficient_data",
        "verdict": "insufficient_evidence",
    }
    written = polymarket_research.write_evaluation_report(report, path=str(path))
    loaded = json.loads(Path(written).read_text(encoding="utf-8"))
    assert loaded["verdict"] == "insufficient_evidence"


def test_gitignore_whitelists_polymarket_research_reports():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "!reports/model_validation/polymarket_paper_protocol.json" in text
    assert "!reports/model_validation/polymarket_paper_evaluation.json" in text
    assert "!reports/model_validation/polymarket_frozen_policy.json" in text
