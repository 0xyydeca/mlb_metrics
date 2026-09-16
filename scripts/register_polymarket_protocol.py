"""Register Polymarket prospective paper protocol BEFORE inspecting outcomes.

Does NOT place orders. Does NOT open evaluation outcomes for promotion.
Does NOT flip BETTING_MODE / GAME_PREDICTION_MODE.

Usage:
    PYTHONPATH=src python scripts/register_polymarket_protocol.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, polymarket_research


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow", config.GAME_PREDICTION_MODE
    assert config.BETTING_MODE == "disabled", config.BETTING_MODE

    protocol = polymarket_research.register_protocol()
    frozen = polymarket_research.write_frozen_policy(protocol=protocol)
    capacity = protocol.get("remaining_season_capacity") or {}
    status = {
        "generated_at_utc": polymarket_research.utc_now_iso(),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("protocol_version"),
        "protocol_hash": protocol.get("protocol_hash"),
        "policy_hash": frozen.get("policy_hash"),
        "prospective_collection_start_local": config.POLYMARKET_PROSPECTIVE_COLLECTION_START_LOCAL,
        "exploratory_dates": list(protocol.get("exploratory_dates") or []),
        "checkpoints_game_days": list(config.POLYMARKET_PROSPECTIVE_CHECKPOINTS_GAME_DAYS),
        "remaining_season_capacity": capacity,
        "collection_host": protocol.get("collection_host"),
        "future_observations_claimed": False,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "notes": [
            "Protocol registered before opening new prospective evaluation outcomes.",
            "Remaining 2026 regular season cannot meet the structural date floor alone.",
            "Collection continues; insufficient_data is the preserved conclusion until floors are met.",
        ],
    }
    path = polymarket_research.write_collection_status(status)
    print(f"Wrote protocol: {config.POLYMARKET_PAPER_PROTOCOL_PATH}")
    print(f"  protocol_id={protocol.get('protocol_id')} version={protocol.get('protocol_version')}")
    print(f"  protocol_hash={protocol.get('protocol_hash')}")
    print(f"Wrote frozen policy: {config.POLYMARKET_FROZEN_POLICY_PATH}")
    print(f"  policy_hash={frozen.get('policy_hash')}")
    print(f"Wrote collection status: {path}")
    print(
        "remaining_regular_season_calendar_dates="
        f"{capacity.get('max_remaining_regular_season_calendar_dates')} "
        f"structural_floor={capacity.get('structural_floor_dates')} "
        f"conclusion={capacity.get('conclusion')}"
    )
    print("BETTING_MODE=disabled; no outcome inspection performed by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
