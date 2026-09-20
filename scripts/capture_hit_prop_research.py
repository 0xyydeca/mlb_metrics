"""Capture MLB 1+ hit prop research inventory (no orders, no model scoring).

Writes a unique JSON under --output-dir and optionally appends durable
research history under data/polymarket/research/hit_props/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_metrics import config, hit_prop_research, schedule
from mlb_metrics.venues.polymarket_us import PolymarketUSAdapter


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", type=dt.date.fromisoformat, default=schedule.today_local())
    parser.add_argument("--event-limit", type=int, default=config.HIT_PROP_RESEARCH_EVENT_LIMIT)
    parser.add_argument("--book-limit", type=int, default=config.HIT_PROP_RESEARCH_BOOK_LIMIT)
    parser.add_argument(
        "--persist-store",
        action="store_true",
        help="Append durable research registry/checkpoint under HIT_PROP_RESEARCH_STORE_DIR",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore capture checkpoint")
    args = parser.parse_args()
    if args.event_limit < 1 or args.book_limit < 0:
        parser.error("event-limit must be positive and book-limit nonnegative")

    report = hit_prop_research.capture(
        PolymarketUSAdapter(),
        date=args.date,
        event_limit=args.event_limit,
        book_limit=args.book_limit,
        persist_store=bool(args.persist_store),
        resume=not args.no_resume,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"hit-props-{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    # Collection status (research-only; never promotes).
    status = {
        "generated_at_utc": hit_prop_research.utc_now_iso(),
        "capture_id": report.get("capture_id"),
        "status": report.get("status"),
        "requested_local_date": report.get("requested_local_date"),
        "n_contracts": len(report.get("contracts") or []),
        "n_books_captured": report.get("n_books_captured"),
        "n_mapped_game_pk": report.get("n_mapped_game_pk"),
        "n_mapped_key_mlbam": report.get("n_mapped_key_mlbam"),
        "n_quarantined": report.get("n_quarantined"),
        "universe": report.get("universe"),
        "exclusions": report.get("exclusions"),
        "errors": report.get("errors"),
        "remaining_blockers": report.get("remaining_blockers"),
        "actionable": False,
        "edge_claimed": False,
        "research_only": True,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "report_path": str(path.resolve()),
    }
    status_path = Path(config.HIT_PROP_COLLECTION_STATUS_PATH)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    host = hit_prop_research.write_host_ops_report()
    example = hit_prop_research.write_contract_example(report)

    print(
        json.dumps(
            {
                "report": str(path.resolve()),
                "status": report["status"],
                "contracts": len(report["contracts"]),
                "books": report.get("n_books_captured", 0),
                "mapped_game_pk": report.get("n_mapped_game_pk", 0),
                "mapped_key_mlbam": report.get("n_mapped_key_mlbam", 0),
                "quarantined": report.get("n_quarantined", 0),
                "actionable": False,
                "host_ops": config.HIT_PROP_HOST_REPORT_PATH,
                "example": (
                    config.HIT_PROP_EXAMPLE_TRACE_PATH if example else None
                ),
                "host": host.get("host"),
            }
        )
    )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
