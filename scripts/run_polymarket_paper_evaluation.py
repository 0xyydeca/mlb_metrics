"""Register protocol, run Polymarket paper evaluation, write evidence report.

Does NOT place orders. Does NOT flip BETTING_MODE / GAME_PREDICTION_MODE.

Usage:
    PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import (
    config,
    game_residual_model as grm,
    paper_ledger,
    polymarket_research,
)


def _load_optional_frame(path: str):
    import pandas as pd

    if not path or not os.path.exists(path):
        return pd.DataFrame()
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build_fixture_ledger_examples() -> tuple[list[dict], list[dict], dict]:
    """Hand-calculated reconciliation examples used as acceptance fixtures."""
    fee = paper_ledger.fee_as_of("2026-09-15T12:00:00Z")
    asks = [
        {"price": 0.40, "size": 2.0},
        {"price": 0.41, "size": 5.0},
    ]
    purchase = paper_ledger.simulate_paper_purchase(
        asks=asks,
        requested_qty=3.0,
        fee=fee,
        delay_seconds=0,
        adverse_ticks=0,
        tick_size=0.01,
        display_price=0.39,
    )
    expected_notional = 1.21
    expected_fees = fee.taker_fee(0.40, 2.0) + fee.taker_fee(0.41, 1.0)
    expected_cost = expected_notional + expected_fees
    assert abs(purchase["acquisition_notional"] - expected_notional) < 1e-9
    assert abs(purchase["fees_paid"] - expected_fees) < 1e-9
    assert abs(purchase["acquisition_cost"] - expected_cost) < 1e-9
    assert purchase["filled_qty"] == 3.0

    decision = paper_ledger.build_decision_row(
        policy_version="fixture_reconcile_v1",
        venue_id="polymarket_us",
        market_id="fixture-m1",
        market_slug="fixture-slug",
        game_pk=1,
        side_team="NYY",
        is_long=True,
        decision_time_utc="2026-09-15T12:00:00Z",
        quote_receive_time_utc="2026-09-15T11:59:50Z",
        quote_request_time_utc="2026-09-15T11:59:49Z",
        fee=fee,
        action="buy",
        pass_reason=None,
        requested_qty=3.0,
        fill_assumption="walk_asks",
        purchase=purchase,
        model_name="fixture",
        model_probability=0.55,
        market_mid_probability_value=0.405,
        executable_buy=0.40,
        probability_source="fixture",
    )
    settled = paper_ledger.settle_position(
        filled_qty=purchase["filled_qty"],
        acquisition_cost=purchase["acquisition_cost"],
        selected_team="NYY",
        winning_team="NYY",
    )
    assert abs(settled["proceeds"] - purchase["filled_qty"]) < 1e-9
    assert abs(settled["net_pnl"] - (purchase["filled_qty"] - purchase["acquisition_cost"])) < 1e-9

    canceled = paper_ledger.settle_position(
        filled_qty=1.0,
        acquisition_cost=0.5,
        selected_team="BOS",
        winning_team=None,
        canceled=True,
    )
    assert canceled["status"] == "canceled"
    assert canceled["net_pnl"] is None

    empty = paper_ledger.simulate_paper_purchase(
        asks=[],
        requested_qty=1.0,
        fee=fee,
        display_price=0.55,
    )
    assert empty["filled_qty"] == 0.0
    assert empty["status"] == "unfilled"

    # Adverse ticks worsen price (buyer pays more).
    stressed = paper_ledger.simulate_paper_purchase(
        asks=[{"price": 0.50, "size": 1.0}],
        requested_qty=1.0,
        fee=fee,
        adverse_ticks=2,
        tick_size=0.01,
    )
    assert stressed["avg_fill_price"] == 0.52

    position_row = {
        "decision_id": decision["decision_id"],
        "acquisition_cost": purchase["acquisition_cost"],
        **settled,
        "settled_at_utc": "2026-09-16T06:00:00Z",
        "revision_of": None,
    }
    hand = {
        "example_a": {
            "filled_qty": purchase["filled_qty"],
            "notional": expected_notional,
            "fees": expected_fees,
            "acquisition_cost": expected_cost,
            "proceeds_if_win": purchase["filled_qty"],
            "net_if_win": purchase["filled_qty"] - expected_cost,
            "reconciled": True,
        },
        "example_b_canceled": canceled,
        "example_c_no_asks": empty,
        "example_d_adverse_ticks": {
            "avg_fill_price": stressed["avg_fill_price"],
            "adverse_ticks": 2,
        },
    }
    return [decision], [position_row], hand


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket paper evaluation (read-only)")
    parser.add_argument(
        "--shadow-predictions",
        default=config.GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH,
    )
    parser.add_argument("--report-path", default=config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH)
    args = parser.parse_args()

    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    protocol = polymarket_research.register_protocol()
    frozen = polymarket_research.write_frozen_policy()
    Path(config.POLYMARKET_PAPER_LEDGER_DIR).mkdir(parents=True, exist_ok=True)
    keep = Path(config.POLYMARKET_PAPER_LEDGER_DIR) / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")

    decisions_rows, position_rows, hand = _build_fixture_ledger_examples()
    research_decisions = os.path.join(config.POLYMARKET_PAPER_LEDGER_DIR, "fixture_decisions.csv")
    research_positions = os.path.join(config.POLYMARKET_PAPER_LEDGER_DIR, "fixture_positions.csv")
    for path in (research_decisions, research_positions):
        if os.path.exists(path):
            os.remove(path)
    decisions = paper_ledger.append_decisions(decisions_rows, path=research_decisions)
    positions = paper_ledger.append_positions(position_rows, path=research_positions)
    decisions2 = paper_ledger.append_decisions(decisions_rows, path=research_decisions)
    assert len(decisions2) == len(decisions)

    strategy_fixture = polymarket_research.ledger_performance_report(decisions, positions)
    strategy_fixture["hand_calculated"] = hand
    strategy_fixture["conservative_roi"] = paper_ledger.settled_roi(positions)

    shadow = _load_optional_frame(args.shadow_predictions)
    exploratory_sportsbook = None
    if not shadow.empty and grm.MARKET_AT_PRED_COL in shadow.columns:
        work = shadow.copy()
        work["market_mid_probability"] = work[grm.MARKET_AT_PRED_COL]
        label = None
        if grm.HOME_WON_LABEL in work.columns:
            label = grm.HOME_WON_LABEL
        elif "Home_Won" in work.columns:
            label = "Home_Won"
        if label is None:
            exploratory_sportsbook = {
                "status": "insufficient_data",
                "reason": "shadow_predictions_missing_labels",
                "n_rows": int(len(work)),
                "excluded_from_polymarket_gates": True,
            }
        else:
            exploratory_sportsbook = polymarket_research.compare_models_on_frame(
                work.rename(columns={label: grm.HOME_WON_LABEL}),
            )
            exploratory_sportsbook["cohort"] = "sportsbook_prior_shadow_diagnostic"
            exploratory_sportsbook["excluded_from_polymarket_gates"] = True

    probability_report = {
        "status": "insufficient_data",
        "reason": "no_polymarket_labeled_history",
        "n_games": 0,
        "n_dates": 0,
        "market_prior_note": (
            "Polymarket mid baseline not yet available for nested evaluation. "
            "Sportsbook residual shadow history is exploratory diagnostic only "
            "and cannot satisfy Polymarket gates."
        ),
    }

    n_quote_index_rows = 0
    quote_index = os.path.join(config.POLYMARKET_QUOTE_STORE_DIR, "quote_index.csv")
    if os.path.exists(quote_index):
        import pandas as pd

        n_quote_index_rows = int(len(pd.read_csv(quote_index)))
    n_mapped = 0
    if os.path.exists(config.POLYMARKET_CONTRACT_REGISTRY_PATH):
        import pandas as pd

        reg = pd.read_csv(config.POLYMARKET_CONTRACT_REGISTRY_PATH)
        n_mapped = int((reg.get("mapping_status") == "mapped").sum()) if not reg.empty else 0

    polymarket_eligible_dates = 0
    n_outer = polymarket_research.count_possible_outer_folds(polymarket_eligible_dates, protocol)

    gates = polymarket_research.apply_gates(
        protocol=protocol,
        probability_report=probability_report,
        strategy_report={
            "roi": {"roi": float("nan"), "n_settled": 0},
            "conservative_roi": {"roi": float("nan")},
            "note": "No Polymarket settled paper history yet; fixture ledger only.",
        },
        n_eligible_dates=polymarket_eligible_dates,
        n_outer_folds_available=n_outer,
    )

    report = {
        "generated_at_utc": polymarket_research.utc_now_iso(),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "policy_hash": frozen.get("policy_hash"),
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "verdict": gates["verdict"],
        "validation_status": gates["validation_status"],
        "gates": gates,
        "sample_size_plan": protocol.get("sample_size_plan"),
        "data_coverage": {
            "n_mapped_contracts": n_mapped,
            "n_quote_index_rows": n_quote_index_rows,
            "n_polymarket_labeled_eligible_dates": polymarket_eligible_dates,
            "exploratory_dates": list(protocol.get("exploratory_dates") or []),
            "note": (
                "Mapped contracts and quote rows are necessary but not sufficient. "
                "Nested evaluation requires labeled settled games with as-of quotes."
            ),
        },
        "probability_quality_polymarket": probability_report,
        "probability_quality_sportsbook_diagnostic_only": exploratory_sportsbook,
        "paper_ledger_fixture": strategy_fixture,
        "prospective": {
            "frozen_policy_path": config.POLYMARKET_FROZEN_POLICY_PATH,
            "checkpoints_game_days": list(config.POLYMARKET_PROSPECTIVE_CHECKPOINTS_GAME_DAYS),
            "future_results_manufactured": False,
            "additional_observations_required": {
                "min_eligible_dates_structural": protocol["fold_structure"][
                    "min_dates_structural_floor"
                ],
                "approx_independent_days_for_target_edge": (
                    protocol.get("sample_size_plan") or {}
                ).get("approx_independent_days"),
                "need": (
                    "Accumulate Polymarket books + official results for >= structural "
                    "floor complete outer blocks after freeze; re-run this script; "
                    "do not inspect the registered freeze tail during tuning."
                ),
            },
        },
        "limitations": [
            "Venue remains provisional Polymarket US until owner confirms.",
            "Sportsbook residual history cannot substitute for Polymarket priors.",
            "Fixture ledger proves accounting math, not edge.",
            "Betting stays disabled.",
        ],
    }
    path = polymarket_research.write_evaluation_report(report, path=args.report_path)
    print(f"Wrote protocol: {config.POLYMARKET_PAPER_PROTOCOL_PATH}")
    print(f"Wrote frozen policy: {config.POLYMARKET_FROZEN_POLICY_PATH}")
    print(f"Wrote evaluation report: {path}")
    print(f"VERDICT: {report['verdict']}")
    print(f"validation_status={report['validation_status']}")
    for reason in gates.get("gate_fail_reasons") or []:
        print(f"  gate: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
