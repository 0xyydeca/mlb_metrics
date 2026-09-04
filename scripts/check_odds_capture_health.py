"""Fail when market-odds capture goes stale during the MLB window.

Used by the Market Odds Capture Health GitHub Action. Does not train
models and does not change GAME_PREDICTION_MODE / BETTING_MODE.

Exit codes:
  0 — outside the capture window, or latest snapshot is fresh enough
  1 — inside the capture window and latest capture is too old / missing

Usage:
    python scripts/check_odds_capture_health.py
    python scripts/check_odds_capture_health.py --max-age-minutes 90
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, market_odds


# Same UTC hours as .github/workflows/market_odds_capture.yml
CAPTURE_UTC_HOURS = frozenset(list(range(15, 24)) + list(range(0, 3)))


def in_capture_window(now_utc: datetime | None = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return int(now.hour) in CAPTURE_UTC_HOURS


def latest_capture_age_minutes(
    snapshots: pd.DataFrame | None = None,
    *,
    now_utc: datetime | None = None,
) -> float | None:
    """Minutes since the newest captured_at_utc, or None if no timestamps."""
    frame = market_odds.normalize_snapshot_frame(
        snapshots if snapshots is not None else market_odds.load_odds_snapshots()
    )
    if frame.empty or "captured_at_utc" not in frame.columns:
        return None
    ts = pd.to_datetime(frame["captured_at_utc"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    latest = ts.max().to_pydatetime()
    return (now - latest).total_seconds() / 60.0


def evaluate_capture_health(
    *,
    max_age_minutes: float = 90.0,
    snapshots: pd.DataFrame | None = None,
    now_utc: datetime | None = None,
) -> dict:
    now = now_utc or datetime.now(timezone.utc)
    in_window = in_capture_window(now)
    age = latest_capture_age_minutes(snapshots, now_utc=now)
    stale = in_window and (age is None or age > float(max_age_minutes))
    return {
        "in_capture_window": in_window,
        "latest_age_minutes": age,
        "max_age_minutes": float(max_age_minutes),
        "stale": bool(stale),
        "ok": not stale,
        "snapshots_path": config.MARKET_ODDS_SNAPSHOTS_PATH,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=90.0,
        help="Fail inside the capture window when latest snapshot is older than this.",
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
    result = evaluate_capture_health(
        max_age_minutes=args.max_age_minutes,
        snapshots=snaps,
    )
    age = result["latest_age_minutes"]
    age_str = "none" if age is None else f"{age:.1f}"
    print("ODDS CAPTURE HEALTH:")
    print(f"  in_capture_window={result['in_capture_window']}")
    print(f"  latest_age_minutes={age_str}")
    print(f"  max_age_minutes={result['max_age_minutes']}")
    print(f"  stale={result['stale']}")
    print(f"  snapshots_path={result['snapshots_path']}")

    if result["stale"]:
        print(
            "::error::Market odds capture is stale during the MLB window. "
            "Check that Market Odds Capture scheduled runs are firing "
            "(fork schedules can lag; use workflow_dispatch as a backup)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
