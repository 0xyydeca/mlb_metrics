"""Capture Polymarket US MLB moneyline contracts + order books (read-only).

Does NOT place orders or change GAME_PREDICTION_MODE / BETTING_MODE.

Usage:
    python scripts/capture_polymarket.py
    python scripts/capture_polymarket.py --limit 50 --skip-books
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, market_contracts, quote_store, schedule_snapshots
from mlb_metrics.venues import get_venue_adapter
from mlb_metrics.venues.polymarket_us import executable_buy_price_for_team


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_schedule(path: str | None) -> pd.DataFrame:
    snaps = schedule_snapshots.load_schedule_snapshots(path)
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=["game_pk", "home_team", "away_team", "game_datetime"])
    cols = [c for c in ["game_pk", "home_team", "away_team", "game_datetime", "date"] if c in snaps.columns]
    out = snaps[cols].drop_duplicates(subset=["game_pk"], keep="last")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Polymarket US MLB moneylines (read-only)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--skip-books", action="store_true")
    parser.add_argument("--registry-path", default=config.POLYMARKET_CONTRACT_REGISTRY_PATH)
    parser.add_argument("--quote-dir", default=config.POLYMARKET_QUOTE_STORE_DIR)
    parser.add_argument("--schedule-snapshots", default=config.SCHEDULE_SNAPSHOTS_PATH)
    parser.add_argument("--coverage-report", default=config.POLYMARKET_COVERAGE_REPORT_PATH)
    parser.add_argument("--venue", default=config.POLYMARKET_VENUE_SELECTED)
    args = parser.parse_args()

    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    adapter = get_venue_adapter(args.venue)
    caps = adapter.capabilities()
    print(f"venue={adapter.venue_id} capabilities={caps}")
    if caps.order_placement:
        raise SystemExit("Refusing venue with order_placement=True in research capture")

    observed_at = _utc_now_iso()
    markets = adapter.list_mlb_moneyline_markets(limit=args.limit)
    print(f"Discovered {len(markets)} moneyline markets")

    schedule = _load_schedule(args.schedule_snapshots)
    print(f"Schedule games available for mapping: {len(schedule)}")
    existing = market_contracts.load_registry(args.registry_path)
    registry = market_contracts.upsert_registry(
        existing, markets, schedule, observed_at_utc=observed_at,
    )
    market_contracts.save_registry(registry, args.registry_path)
    coverage = market_contracts.coverage_summary(registry)
    print(
        "Registry coverage: "
        f"mapped={coverage['n_mapped']} unmatched={coverage['n_unmatched']} "
        f"ambiguous={coverage['n_ambiguous']} incomplete={coverage['n_incomplete']}"
    )

    quotes = []
    buy_samples = []
    if not args.skip_books:
        by_id = {m.market_id: m for m in markets}
        for market in markets:
            if market.closed:
                continue
            try:
                book = adapter.fetch_market_book(
                    market.market_slug,
                    market_id=market.market_id,
                    fee_coefficient=market.fee_coefficient,
                    tick_size=market.tick_size,
                    min_trade_qty=market.min_trade_qty,
                )
            except Exception as exc:  # noqa: BLE001 - capture continues; failures are logged
                print(f"book_failed market={market.market_slug}: {type(exc).__name__}: {exc}")
                continue
            quotes.append(book)
            if market.home_team:
                buy_samples.append(
                    executable_buy_price_for_team(market, book, market.home_team)
                    | {"market_id": market.market_id, "side": "home"}
                )
        persist = quote_store.append_quotes(quotes, store_dir=args.quote_dir)
        print(
            f"Quotes persisted: received={persist['n_received']} "
            f"written={persist['n_written']}"
        )
    health = quote_store.latest_quote_age_seconds(args.quote_dir, now_utc=observed_at)

    report = {
        "captured_at_utc": observed_at,
        "venue_id": adapter.venue_id,
        "n_markets": len(markets),
        "coverage": coverage,
        "quote_health": health,
        "n_books_captured": len(quotes),
        "sample_home_buys": buy_samples[:5],
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "notes": [
            "Read-only capture; no orders placed.",
            "Display prices are not fill quotes; books carry size.",
            "Provider gameId is not MLB game_pk.",
        ],
    }
    os.makedirs(os.path.dirname(args.coverage_report) or ".", exist_ok=True)
    with open(args.coverage_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote coverage report: {args.coverage_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
