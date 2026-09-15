"""Audit Polymarket registry/quote coverage (read-only)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, market_contracts, quote_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Polymarket capture health")
    parser.add_argument("--registry-path", default=config.POLYMARKET_CONTRACT_REGISTRY_PATH)
    parser.add_argument("--quote-dir", default=config.POLYMARKET_QUOTE_STORE_DIR)
    parser.add_argument("--coverage-report", default=config.POLYMARKET_COVERAGE_REPORT_PATH)
    args = parser.parse_args()

    registry = market_contracts.load_registry(args.registry_path)
    coverage = market_contracts.coverage_summary(registry)
    health = quote_store.latest_quote_age_seconds(args.quote_dir)
    mapped_ids = (
        registry.loc[registry["mapping_status"] == "mapped", "market_id"].astype(str).tolist()
        if not registry.empty
        else []
    )
    per_contract = quote_store.per_contract_quote_health(
        args.quote_dir,
        market_ids=mapped_ids or None,
    )
    print("POLYMARKET AUDIT:")
    print(f"  venue_selected={config.POLYMARKET_VENUE_SELECTED} (provisional US research)")
    print(f"  n_contracts={coverage['n_contracts']}")
    print(f"  n_mapped={coverage['n_mapped']}")
    print(f"  n_unmatched={coverage['n_unmatched']}")
    print(f"  n_ambiguous={coverage['n_ambiguous']}")
    print(f"  mapped_rate={coverage['mapped_rate']}")
    print(f"  quote_index_rows={health['n_index_rows']}")
    print(f"  latest_quote={health['latest_receive_time_utc']}")
    print(f"  quote_age_seconds={health['age_seconds']}")
    print(f"  quote_stale={health['stale']}")
    print(
        f"  usable_fresh_books={per_contract['n_fresh_eligible']} "
        f"stale={per_contract['n_stale']} missing={per_contract['n_missing']}"
    )
    if os.path.exists(args.coverage_report):
        with open(args.coverage_report, encoding="utf-8") as f:
            report = json.load(f)
        print(f"  last_capture_at={report.get('captured_at_utc')}")
        print(f"  last_n_books_captured={report.get('n_books_captured')}")
        print(f"  last_n_book_http_failures={report.get('n_book_http_failures')}")
        if report.get("book_liquidity_counts"):
            print(f"  book_liquidity_counts={report.get('book_liquidity_counts')}")
        if report.get("coverage_waterfall"):
            print(f"  coverage_waterfall={report.get('coverage_waterfall')}")
    print(
        f"  modes: GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r} "
        f"BETTING_MODE={config.BETTING_MODE!r}"
    )
    if coverage["n_mapped"] > 0 and per_contract["n_fresh_eligible"] == 0:
        print(
            "  NOTE: mapping != usable price coverage "
            "(no fresh eligible books for mapped contracts)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
