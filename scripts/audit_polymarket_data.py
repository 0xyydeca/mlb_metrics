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
    print("POLYMARKET AUDIT:")
    print(f"  venue_selected={config.POLYMARKET_VENUE_SELECTED}")
    print(f"  n_contracts={coverage['n_contracts']}")
    print(f"  n_mapped={coverage['n_mapped']}")
    print(f"  n_unmatched={coverage['n_unmatched']}")
    print(f"  n_ambiguous={coverage['n_ambiguous']}")
    print(f"  mapped_rate={coverage['mapped_rate']}")
    print(f"  quote_index_rows={health['n_index_rows']}")
    print(f"  latest_quote={health['latest_receive_time_utc']}")
    print(f"  quote_age_seconds={health['age_seconds']}")
    print(f"  quote_stale={health['stale']}")
    if os.path.exists(args.coverage_report):
        with open(args.coverage_report, encoding="utf-8") as f:
            report = json.load(f)
        print(f"  last_capture_at={report.get('captured_at_utc')}")
    print(
        f"  modes: GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r} "
        f"BETTING_MODE={config.BETTING_MODE!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
