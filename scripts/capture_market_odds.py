"""Capture ESPN/DraftKings odds snapshots every ~30 minutes.

Independent of lineup-lock updates: ``run_lineup_lock_update.py`` exits when
lineup content is unchanged, so odds near first pitch were not guaranteed
to be captured. This script always fetches today's schedule + provider odds,
matches by ``game_pk`` with the safe matcher, and appends immutable snapshot
rows.

Roles:
- requested role is ``intraday`` (not morning) so post-start / historical
  captures are never relabeled as prediction-time morning priors
- ``coerce_requested_snapshot_role`` + ``mark_post_start_snapshots`` still
  classify post-start rows as ``post_start``

Usage:
    python scripts/capture_market_odds.py
    python scripts/capture_market_odds.py --date 2026-09-03
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, market_odds, schedule


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        default=None,
        help="MLB slate date (YYYY-MM-DD). Defaults to today in local schedule TZ.",
    )
    parser.add_argument(
        "--snapshots-path",
        default=config.MARKET_ODDS_SNAPSHOTS_PATH,
    )
    parser.add_argument(
        "--event-map-path",
        default=config.MARKET_ODDS_EVENT_MAP_PATH,
    )
    args = parser.parse_args()

    target = args.date if args.date is not None else schedule.today_local()
    print(f"Capturing market odds for {target} (role={market_odds.SNAPSHOT_ROLE_INTRADAY})")

    try:
        games = schedule.fetch_todays_games(target)
    except Exception as exc:
        print(f"ERROR: failed to fetch MLB schedule: {exc}")
        return 1

    if games is None or games.empty:
        print(f"No MLB games scheduled for {target}; nothing to capture.")
        return 0

    print(f"Schedule games: {len(games)}")
    matched = market_odds.fetch_and_persist_odds_snapshots(
        target,
        schedule_games=games,
        snapshot_role=market_odds.SNAPSHOT_ROLE_INTRADAY,
        snapshots_path=args.snapshots_path,
        event_map_path=args.event_map_path,
    )

    n_ok = int((matched["source_status"] == market_odds.SOURCE_OK).sum()) if not matched.empty else 0
    n_post = int((matched["source_status"] == market_odds.SOURCE_POST_START).sum()) if not matched.empty else 0
    n_amb = int((matched["source_status"] == market_odds.SOURCE_AMBIGUOUS_MATCH).sum()) if not matched.empty else 0
    n_un = int((matched["source_status"] == market_odds.SOURCE_UNMATCHED).sum()) if not matched.empty else 0
    print(
        f"Persisted {len(matched)} snapshot row(s): "
        f"ok={n_ok} post_start={n_post} ambiguous={n_amb} unmatched={n_un}"
    )
    print(f"Snapshots path: {args.snapshots_path}")
    print(f"Event map path: {args.event_map_path}")
    # Inventory from the on-disk log (includes prior captures).
    market_odds.print_market_snapshot_inventory(
        market_odds.market_snapshot_inventory(
            market_odds.load_odds_snapshots(args.snapshots_path)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
