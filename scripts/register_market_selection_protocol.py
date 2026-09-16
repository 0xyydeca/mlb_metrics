"""Register market-selection protocol BEFORE examining comparative outcomes.

Does not place orders. Does not enable betting. Does not claim an edge.

Usage:
    PYTHONPATH=src python scripts/register_market_selection_protocol.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config


def _sha(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    protocol = {
        "protocol_id": "market_selection_next_sport_v1",
        "protocol_version": "1",
        "registered_at_utc": config.MARKET_SELECTION_PROTOCOL_REGISTERED_UTC,
        "objective": (
            "Select at most one next sport/market for research investment using "
            "reusable Polymarket infrastructure. Selection is feasibility-driven, "
            "not profitability-driven."
        ),
        "rules": {
            "at_most_one_selection": True,
            "no_selection_on_lucky_exploratory_backtest": True,
            "market_selection_is_tuning": True,
            "register_before_examining_comparative_outcomes": True,
            "reserve_untouched_evidence_for_selected_market": True,
            "missing_liquidity_is_not_zero_cost": True,
            "no_invented_scoring_weights": True,
            "no_data_purchases": True,
            "no_assumed_account_access": True,
            "model_development_gated_on_subsequent_validation": True,
            "real_money_gated": True,
        },
        "candidates_under_consideration": [
            {
                "id": "nfl_pregame_moneyline",
                "sport": "nfl",
                "market_type": "pregame_full_game_winner",
                "venue": "polymarket_us_provisional",
            },
            {
                "id": "nba_pregame_moneyline",
                "sport": "nba",
                "market_type": "pregame_full_game_winner",
                "venue": "polymarket_us_provisional",
            },
            {
                "id": "nhl_pregame_moneyline",
                "sport": "nhl",
                "market_type": "pregame_full_game_winner",
                "venue": "polymarket_us_provisional",
            },
        ],
        "feasibility_dimensions": [
            "contract_availability_and_settlement_complexity",
            "historical_prediction_time_data_and_labels",
            "executable_prices_spread_depth_fees_collection_cost",
            "opportunity_frequency_and_validation_time",
            "reusable_code_vs_sport_specific_work",
        ],
        "status_labels_allowed": ["feasible", "blocked", "uncertain"],
        "untouched_holdout_rule": (
            "After selection, register a sport-specific paper protocol before "
            "opening nested evaluation outcomes for that sport. Do not reuse "
            "MLB freeze tails or exploratory MLB dates as NFL/NBA evidence."
        ),
        "mlb_continues": True,
        "notes": [
            "Registered before writing the feasibility comparison outcomes.",
            "A no-expansion recommendation is allowed.",
            "Owner venue (US vs international) remains unconfirmed.",
        ],
    }
    protocol["protocol_hash"] = _sha(protocol)
    path = config.MARKET_SELECTION_PROTOCOL_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("protocol_hash") == protocol["protocol_hash"]:
            print(f"Protocol unchanged: {path}")
            print(f"protocol_hash={protocol['protocol_hash']}")
            return 0
        protocol["supersedes_protocol_hash"] = existing.get("protocol_hash")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {path}")
    print(f"protocol_hash={protocol['protocol_hash']}")
    print("No comparative evaluation outcomes inspected by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
