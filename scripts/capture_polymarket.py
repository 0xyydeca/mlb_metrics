"""Capture Polymarket US MLB moneyline contracts + order books (read-only).

Does NOT place orders or change GAME_PREDICTION_MODE / BETTING_MODE.

Usage:
    PYTHONPATH=src python scripts/capture_polymarket.py
    PYTHONPATH=src python scripts/capture_polymarket.py --limit 50 --skip-books
    PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, game_baseball_snapshots, market_contracts, quote_store, schedule
from mlb_metrics.venues import get_venue_adapter
from mlb_metrics.venues.polymarket_us import executable_buy_price_for_team


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Polymarket US MLB moneylines (read-only)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--skip-books", action="store_true")
    parser.add_argument("--registry-path", default=config.POLYMARKET_CONTRACT_REGISTRY_PATH)
    parser.add_argument("--quote-dir", default=config.POLYMARKET_QUOTE_STORE_DIR)
    parser.add_argument("--schedule-snapshots", default=config.SCHEDULE_SNAPSHOTS_PATH)
    parser.add_argument("--coverage-report", default=config.POLYMARKET_COVERAGE_REPORT_PATH)
    parser.add_argument("--venue", default=config.POLYMARKET_VENUE_SELECTED)
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=config.POLYMARKET_SCHEDULE_LOOKAHEAD_DAYS,
        help="Live StatsAPI schedule days beyond today_local() for mapping.",
    )
    parser.add_argument(
        "--snapshots-only",
        action="store_true",
        help="Disable live StatsAPI schedule fetch (tests / offline).",
    )
    parser.add_argument(
        "--with-baseball",
        action="store_true",
        help="Also fetch today's game baseball snapshots (starters/lineups).",
    )
    parser.add_argument(
        "--persist-baseball",
        action="store_true",
        help="Write baseball snapshots to configured prediction CSV paths (opt-in).",
    )
    args = parser.parse_args()

    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    adapter = get_venue_adapter(args.venue)
    caps = adapter.capabilities()
    print(f"venue={adapter.venue_id} capabilities={caps}")
    print(f"runtime_note={config.POLYMARKET_CAPTURE_RUNTIME_NOTE}")
    if caps.order_placement:
        raise SystemExit("Refusing venue with order_placement=True in research capture")

    started_at = _utc_now_iso()
    markets = adapter.list_mlb_moneyline_markets(limit=args.limit)
    print(f"Discovered {len(markets)} moneyline markets")

    schedule_games = market_contracts.load_mapping_schedule(
        schedule_snapshots_path=args.schedule_snapshots,
        lookahead_days=args.lookahead_days,
        prefer_live=not args.snapshots_only,
    )
    print(
        f"Schedule games available for mapping: {len(schedule_games)} "
        f"(live_preferred={not args.snapshots_only}, lookahead_days={args.lookahead_days})"
    )
    existing = market_contracts.load_registry(args.registry_path)
    registry = market_contracts.upsert_registry(
        existing, markets, schedule_games, observed_at_utc=started_at,
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
    book_failures: list[dict] = []
    book_liquidity: Counter = Counter()
    if not args.skip_books:
        for market in markets:
            if market.closed:
                book_liquidity["skipped_closed"] += 1
                continue
            try:
                book = adapter.fetch_market_book(
                    market.market_slug,
                    market_id=market.market_id,
                    fee_coefficient=market.fee_coefficient,
                    tick_size=market.tick_size,
                    min_trade_qty=market.min_trade_qty,
                )
            except Exception as exc:  # noqa: BLE001 - capture continues; failures recorded
                book_failures.append(
                    {
                        "market_id": market.market_id,
                        "market_slug": market.market_slug,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                        "failure_class": "collector_or_http",
                    }
                )
                print(f"book_failed market={market.market_slug}: {type(exc).__name__}: {exc}")
                continue
            quotes.append(book)
            if book.eligible:
                book_liquidity["eligible_book"] += 1
            elif book.eligibility_reason == "missing_ask":
                book_liquidity["empty_or_no_ask"] += 1
            else:
                book_liquidity[str(book.eligibility_reason or "ineligible")] += 1
            if market.home_team:
                buy_samples.append(
                    executable_buy_price_for_team(market, book, market.home_team)
                    | {"market_id": market.market_id, "side": "home"}
                )
        persist = quote_store.append_quotes(quotes, store_dir=args.quote_dir)
        print(
            f"Quotes persisted: received={persist['n_received']} "
            f"written={persist['n_written']} "
            f"http_failures={len(book_failures)}"
        )
    else:
        persist = {"n_received": 0, "n_written": 0, "partitions": []}

    finished_at = _utc_now_iso()
    health = quote_store.latest_quote_age_seconds(args.quote_dir, now_utc=finished_at)
    mapped_ids = (
        registry.loc[registry["mapping_status"] == "mapped", "market_id"].astype(str).tolist()
        if not registry.empty
        else [str(m.market_id) for m in markets if not m.closed]
    )
    per_contract = quote_store.per_contract_quote_health(
        args.quote_dir,
        market_ids=mapped_ids,
        now_utc=finished_at,
    )
    actionable = quote_store.actionable_suppressed(
        quote_health=health,
        collector_heartbeat_utc=finished_at if quotes or args.skip_books else None,
        now_utc=finished_at,
    )

    baseball = None
    if args.with_baseball:
        baseball = game_baseball_snapshots.fetch_game_baseball_snapshots(schedule.today_local())
        if args.persist_baseball:
            game_baseball_snapshots.persist_game_snapshots(baseball)
            print("Persisted baseball snapshots to configured prediction paths.")
        print(
            "Baseball snapshots: "
            f"n={len(baseball)} usable={int(baseball['usable_for_game_winner'].fillna(False).sum())}"
        )

    waterfall = game_baseball_snapshots.coverage_waterfall(
        schedule_games=schedule_games,
        registry=registry,
        quote_health_contracts=per_contract.get("contracts"),
        baseball_snapshots=baseball,
    )

    # Trace one mapped contract with a captured book when available.
    sample_trace = None
    mapped_reg = registry[registry["mapping_status"] == "mapped"]
    if not mapped_reg.empty and quotes:
        sample_row = mapped_reg.iloc[0]
        market = next((m for m in markets if str(m.market_id) == str(sample_row["market_id"])), None)
        book = next((q for q in quotes if str(q.market_id) == str(sample_row["market_id"])), None)
        q_health = next(
            (c for c in per_contract["contracts"] if str(c["market_id"]) == str(sample_row["market_id"])),
            None,
        )
        bb_row = None
        if baseball is not None and not baseball.empty and pd_notna(sample_row.get("game_pk")):
            hit = baseball[baseball["game_pk"] == int(sample_row["game_pk"])]
            if not hit.empty:
                bb_row = hit.iloc[0]
        sample_trace = game_baseball_snapshots.trace_contract_decision_inputs(
            registry_row=sample_row,
            quote_row=q_health,
            market=market,
            book=book,
            baseball_row=bb_row,
            fee_version=book.fee_version if book is not None else None,
        )

    report = {
        "captured_at_utc": started_at,
        "finished_at_utc": finished_at,
        "venue_id": adapter.venue_id,
        "venue_label": "polymarket_us_provisional_research",
        "n_markets": len(markets),
        "n_schedule_games": int(len(schedule_games)),
        "schedule_source": "snapshots_only" if args.snapshots_only else "live_plus_snapshots",
        "coverage": coverage,
        "quote_health": health,
        "per_contract_quote_health": {
            k: per_contract[k]
            for k in (
                "n_markets_checked",
                "n_fresh_eligible",
                "n_stale",
                "n_ineligible",
                "n_missing",
                "max_age_seconds",
            )
        },
        "n_books_captured": len(quotes),
        "n_books_written": int(persist.get("n_written") or 0),
        "n_book_http_failures": len(book_failures),
        "book_failure_samples": book_failures[:10],
        "book_liquidity_counts": dict(book_liquidity),
        "sample_home_buys": buy_samples[:5],
        "coverage_waterfall": waterfall,
        "actionable": actionable,
        "sample_contract_trace": sample_trace,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "notes": [
            "Read-only capture; no orders placed.",
            "Display prices are not fill quotes; books carry size.",
            "Provider gameId is not MLB game_pk.",
            "Mapping success is distinct from usable price coverage.",
            "Venue is provisional Polymarket US research until owner confirms venue.",
            "Price unit: USD share cost in [0,1]; qty unit: contracts.",
            config.POLYMARKET_CAPTURE_RUNTIME_NOTE,
        ],
    }
    os.makedirs(os.path.dirname(args.coverage_report) or ".", exist_ok=True)
    with open(args.coverage_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote coverage report: {args.coverage_report}")
    print(
        "Waterfall: "
        f"schedule={waterfall['n_schedule_games']} mapped={waterfall['n_mapped_contracts']} "
        f"usable_prices={waterfall['n_usable_prices']} "
        f"usable_baseball={waterfall['n_usable_baseball_snapshots']}"
    )
    if book_failures and not quotes:
        print(
            "LIMITATION: mapped contracts exist but no books were captured this run "
            "(collector/HTTP failures), not proof of missing market liquidity."
        )
        return 2
    return 0


def pd_notna(value) -> bool:
    try:
        import pandas as pd

        return value is not None and not (isinstance(value, float) and pd.isna(value)) and pd.notna(value)
    except Exception:
        return value is not None


if __name__ == "__main__":
    raise SystemExit(main())
