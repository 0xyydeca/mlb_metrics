"""Print market-odds inventory and residual training status.

Used by the Game Residual Training GitHub Action before/after
``train_game_residual_model.py``. Does not train, does not change modes,
and does not save artifacts.

Usage:
    python scripts/report_residual_training_status.py
    python scripts/report_residual_training_status.py --inventory-only
    python scripts/report_residual_training_status.py --status-only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import game_residual_model, market_odds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Print MARKET SNAPSHOT INVENTORY only.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print RESIDUAL TRAINING STATUS only.",
    )
    parser.add_argument(
        "--odds-snapshots",
        default=None,
        help="Override market odds snapshots CSV path.",
    )
    args = parser.parse_args()

    snaps = (
        market_odds.load_odds_snapshots(args.odds_snapshots)
        if args.odds_snapshots
        else market_odds.load_odds_snapshots()
    )
    inv = market_odds.market_snapshot_inventory(snaps)

    if not args.status_only:
        market_odds.print_market_snapshot_inventory(inv)

    if not args.inventory_only:
        if not args.status_only:
            print()
        status = game_residual_model.residual_training_status(snapshots=snaps)
        game_residual_model.print_residual_training_status(status)
        print(
            f"  (modes unchanged: GAME_PREDICTION_MODE={status['game_prediction_mode']!r} "
            f"BETTING_MODE={status['betting_mode']!r})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
