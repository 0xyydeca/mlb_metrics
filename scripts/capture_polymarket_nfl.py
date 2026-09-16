"""Capture Polymarket US NFL moneylines (read-only; namespaced away from MLB).

Does NOT place orders or change GAME_PREDICTION_MODE / BETTING_MODE.
Does NOT write MLB registry/quote paths.

Usage:
    PYTHONPATH=src python scripts/capture_polymarket_nfl.py
    PYTHONPATH=src python scripts/capture_polymarket_nfl.py --limit 30 --skip-books
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, nfl_market_contracts, quote_store
from mlb_metrics.venues import get_venue_adapter


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Polymarket US NFL moneylines (read-only)")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip-books", action="store_true")
    parser.add_argument("--registry-path", default=config.POLYMARKET_NFL_CONTRACT_REGISTRY_PATH)
    parser.add_argument("--quote-dir", default=config.POLYMARKET_NFL_QUOTE_STORE_DIR)
    parser.add_argument("--coverage-report", default=config.POLYMARKET_NFL_COVERAGE_REPORT_PATH)
    parser.add_argument("--lookahead-days", type=int, default=7)
    args = parser.parse_args()

    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    adapter = get_venue_adapter(config.POLYMARKET_VENUE_SELECTED)
    caps = adapter.capabilities()
    print(f"venue={adapter.venue_id} sport=nfl capabilities={caps}")
    if caps.order_placement:
        raise SystemExit("Refusing venue with order_placement=True")

    started_at = _utc_now_iso()
    markets = adapter.list_moneyline_markets(league=config.POLYMARKET_NFL_LEAGUE_SLUG, limit=args.limit)
    print(f"Discovered {len(markets)} NFL moneyline markets")

    schedule_games = nfl_market_contracts.load_nfl_schedule_for_mapping(
        lookahead_days=args.lookahead_days
    )
    print(f"NFL schedule rows for mapping: {len(schedule_games)}")

    existing = nfl_market_contracts.load_registry(args.registry_path)
    registry = nfl_market_contracts.upsert_nfl_registry(
        existing, markets, schedule_games, observed_at_utc=started_at
    )
    nfl_market_contracts.save_registry(registry, args.registry_path)
    n_mapped = int((registry["mapping_status"] == "mapped").sum()) if not registry.empty else 0
    print(
        f"NFL registry: n={len(registry)} mapped={n_mapped} "
        f"path={args.registry_path}"
    )

    quotes = []
    book_failures = []
    if not args.skip_books:
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
                quotes.append(book)
            except Exception as exc:  # noqa: BLE001
                book_failures.append(
                    {
                        "market_id": market.market_id,
                        "error_type": type(exc).__name__,
                        "detail": str(exc)[:200],
                        "failure_class": "collector_or_http",
                    }
                )
        persist = quote_store.append_quotes(quotes, store_dir=args.quote_dir) if quotes else {}
        print(
            f"Quotes: received={len(quotes)} written={persist.get('n_written', 0)} "
            f"http_failures={len(book_failures)}"
        )

    report = {
        "captured_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "sport_id": "nfl",
        "venue_id": adapter.venue_id,
        "venue_label": "polymarket_us_provisional_research",
        "n_markets": len(markets),
        "n_schedule_rows": int(len(schedule_games)),
        "n_mapped": n_mapped,
        "n_books_captured": len(quotes),
        "n_book_http_failures": len(book_failures),
        "book_failure_samples": book_failures[:10],
        "registry_path": args.registry_path,
        "quote_dir": args.quote_dir,
        "mlb_paths_untouched": True,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "notes": [
            "Read-only NFL capture; no orders.",
            "sport_game_key uses nfl:{game_id}; MLB game_pk registry is not modified.",
            "Model development and real-money use remain gated.",
        ],
    }
    os.makedirs(os.path.dirname(args.coverage_report) or ".", exist_ok=True)
    with open(args.coverage_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
        f.write("\n")
    print(f"Wrote {args.coverage_report}")
    if book_failures and not quotes:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
