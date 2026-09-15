"""Health check for Polymarket capture + decision-input coverage (read-only)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, game_baseball_snapshots, market_contracts, quote_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket + baseball decision health check")
    parser.add_argument("--registry-path", default=config.POLYMARKET_CONTRACT_REGISTRY_PATH)
    parser.add_argument("--quote-dir", default=config.POLYMARKET_QUOTE_STORE_DIR)
    parser.add_argument("--coverage-report", default=config.POLYMARKET_COVERAGE_REPORT_PATH)
    parser.add_argument(
        "--baseball-latest",
        default=config.GAME_BASEBALL_SNAPSHOT_LATEST_PATH,
    )
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
    baseball = (
        game_baseball_snapshots.normalize_game_snapshot_frame(
            __import__("pandas").read_csv(args.baseball_latest)
        )
        if os.path.exists(args.baseball_latest)
        else game_baseball_snapshots.empty_game_snapshot_frame()
    )
    waterfall = game_baseball_snapshots.coverage_waterfall(
        schedule_games=__import__("pandas").DataFrame({"game_pk": []}),
        registry=registry,
        quote_health_contracts=per_contract.get("contracts"),
        baseball_snapshots=baseball,
    )
    # Prefer schedule count from coverage report when present.
    if os.path.exists(args.coverage_report):
        with open(args.coverage_report, encoding="utf-8") as f:
            report = json.load(f)
        waterfall["n_schedule_games"] = int(report.get("n_schedule_games") or 0)
        if report.get("coverage_waterfall"):
            waterfall = report["coverage_waterfall"]
    else:
        report = {}

    actionable = quote_store.actionable_suppressed(
        quote_health=health,
        collector_heartbeat_utc=report.get("finished_at_utc") or report.get("captured_at_utc"),
    )

    print("POLYMARKET / DECISION HEALTH:")
    print(f"  venue_selected={config.POLYMARKET_VENUE_SELECTED} (provisional US research)")
    print(f"  mapped_contracts={coverage['n_mapped']} / {coverage['n_contracts']}")
    print(f"  mapped_rate={coverage['n_mapped'] / coverage['n_contracts'] if coverage['n_contracts'] else 0:.3f}")
    print(
        f"  usable_prices={per_contract['n_fresh_eligible']} "
        f"stale={per_contract['n_stale']} missing={per_contract['n_missing']} "
        f"(distinct from mapping)"
    )
    print(
        f"  global_quote_age_seconds={health.get('age_seconds')} "
        f"stale={health.get('stale')} reason={health.get('reason')}"
    )
    print(
        f"  baseball_snapshots={len(baseball)} "
        f"usable={int(baseball['usable_for_game_winner'].fillna(False).sum()) if not baseball.empty else 0}"
    )
    print(f"  waterfall={waterfall}")
    print(f"  actionable={actionable}")
    print(
        f"  modes: GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r} "
        f"BETTING_MODE={config.BETTING_MODE!r}"
    )
    if coverage["n_mapped"] > 0 and per_contract["n_fresh_eligible"] == 0:
        print(
            "  LIMITATION: mapping succeeded but no fresh eligible books — "
            "do not treat mapped_rate as price coverage."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
