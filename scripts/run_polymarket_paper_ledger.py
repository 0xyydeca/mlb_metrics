"""Run Polymarket paper ledger cycle + settlement (no orders).

Usage:
    PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
    PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --trace-only
    PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --settle-date 2026-09-14
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, game_baseball_snapshots, market_contracts, paper_pipeline, schedule


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket paper ledger (read-only)")
    parser.add_argument("--decision-time-utc", default=None)
    parser.add_argument("--delay-seconds", type=int, default=None)
    parser.add_argument("--adverse-ticks", type=int, default=None)
    parser.add_argument(
        "--allow-exploratory-fills",
        action="store_true",
        help="Allow hypothetical buys while evidence gates fail (labeled exploratory).",
    )
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--market-id", default=None)
    parser.add_argument("--settle-date", default=None, help="YYYY-MM-DD local schedule date")
    parser.add_argument("--decisions-path", default=config.POLYMARKET_PAPER_DECISIONS_PATH)
    parser.add_argument("--positions-path", default=config.POLYMARKET_PAPER_POSITIONS_PATH)
    parser.add_argument("--skip-cycle", action="store_true")
    args = parser.parse_args()

    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    trace = paper_pipeline.trace_observed_contract(market_id=args.market_id)
    path = paper_pipeline.write_trace_report(trace)
    print(f"Wrote contract trace: {path}")
    print(
        json.dumps(
            {
                k: trace.get(k)
                for k in (
                    "ok",
                    "market_id",
                    "game_pk",
                    "missing",
                    "missing_for_labeled_eval",
                    "paper_purchase",
                )
            },
            default=str,
        )
    )
    if args.trace_only:
        return 0 if trace.get("ok") else 1

    if not args.skip_cycle:
        baseball = None
        latest = config.GAME_BASEBALL_SNAPSHOT_LATEST_PATH
        if os.path.exists(latest):
            baseball = game_baseball_snapshots.normalize_game_snapshot_frame(pd.read_csv(latest))
        predictions = None
        for path in (
            "docs/data/game_picks_picks.csv",
            config.GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH,
        ):
            if os.path.exists(path):
                predictions = pd.read_csv(path)
                break
        summary = paper_pipeline.run_paper_decision_cycle(
            registry=market_contracts.load_registry(),
            baseball=baseball,
            predictions=predictions,
            decision_time_utc=args.decision_time_utc,
            delay_seconds=args.delay_seconds,
            adverse_ticks=args.adverse_ticks,
            allow_exploratory_fills=args.allow_exploratory_fills,
            decisions_path=args.decisions_path,
            positions_path=args.positions_path,
            write=True,
        )
        print(json.dumps(summary, indent=2, default=str))

    if args.settle_date:
        results = schedule.fetch_game_results(args.settle_date)
        settle = paper_pipeline.settle_open_positions(
            results=results,
            decisions_path=args.decisions_path,
            positions_path=args.positions_path,
        )
        print(json.dumps(settle, indent=2, default=str))

    print("BETTING_MODE=disabled; no automated orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
