"""Export Polymarket manual decision board for the docs dashboard.

Read-only. Does not place orders or change BETTING_MODE.

Usage:
    PYTHONPATH=src python scripts/export_polymarket_decision_board.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, decision_board, game_baseball_snapshots, market_contracts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Polymarket decision board (read-only)")
    parser.add_argument("--registry", default=config.POLYMARKET_CONTRACT_REGISTRY_PATH)
    parser.add_argument("--quote-dir", default=config.POLYMARKET_QUOTE_STORE_DIR)
    parser.add_argument("--board-path", default=config.POLYMARKET_DECISION_BOARD_PATH)
    parser.add_argument("--meta-path", default=config.POLYMARKET_DECISION_BOARD_META_PATH)
    parser.add_argument(
        "--predictions",
        default="docs/data/game_picks_picks.csv",
        help="Optional game picks / shadow predictions CSV for model probabilities.",
    )
    parser.add_argument(
        "--with-live-baseball",
        action="store_true",
        help="Fetch today's baseball snapshots (network).",
    )
    args = parser.parse_args()

    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    registry = market_contracts.load_registry(args.registry)
    quotes = None
    from mlb_metrics import quote_store

    quotes = quote_store.latest_quotes_by_market(args.quote_dir)

    baseball = None
    latest_bb = config.GAME_BASEBALL_SNAPSHOT_LATEST_PATH
    if args.with_live_baseball:
        from mlb_metrics import schedule

        baseball = game_baseball_snapshots.fetch_game_baseball_snapshots(schedule.today_local())
    elif os.path.exists(latest_bb):
        baseball = game_baseball_snapshots.normalize_game_snapshot_frame(pd.read_csv(latest_bb))

    predictions = None
    if args.predictions and os.path.exists(args.predictions):
        predictions = pd.read_csv(args.predictions)
    shadow = config.GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH
    if (predictions is None or predictions.empty) and os.path.exists(shadow):
        predictions = pd.read_csv(shadow)

    board, meta = decision_board.build_decision_board(
        registry=registry,
        quotes=quotes,
        baseball=baseball,
        predictions=predictions,
    )
    paths = decision_board.write_decision_board(
        board, meta, board_path=args.board_path, meta_path=args.meta_path
    )
    # Mirror evaluation report into docs/data for the static dashboard.
    eval_src = config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH
    eval_dst = os.path.join(os.path.dirname(args.board_path) or ".", "polymarket_paper_evaluation.json")
    if os.path.exists(eval_src):
        import shutil

        shutil.copyfile(eval_src, eval_dst)
        print(f"Copied evaluation snapshot to {eval_dst}")
    print(f"Wrote {paths['board_path']} rows={len(board)}")
    print(f"Wrote {paths['meta_path']}")
    print(
        f"evidence_verdict={meta['evidence_verdict']} "
        f"validation_status={meta['validation_status']} "
        f"paper_candidates={meta['n_paper_candidate']} "
        f"actionable={meta['n_actionable']}"
    )
    print("BETTING_MODE=disabled; no automated orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
