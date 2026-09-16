"""Register NFL Polymarket paper validation plan BEFORE nested evaluation outcomes.

Gates model development and real-money use. Does not place orders.
Does not inspect or claim prediction performance.

Usage:
    PYTHONPATH=src python scripts/register_nfl_paper_protocol.py
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
    assert os.path.exists(config.MARKET_SELECTION_PROTOCOL_PATH), (
        "Market-selection protocol must be registered first"
    )
    assert os.path.exists(config.MARKET_SELECTION_FEASIBILITY_PATH), (
        "Market-selection feasibility must be written before NFL validation plan"
    )
    with open(config.MARKET_SELECTION_FEASIBILITY_PATH, encoding="utf-8") as f:
        feasibility = json.load(f)
    selected = (feasibility.get("selection") or {}).get("selected_id")
    assert selected == "nfl_pregame_moneyline", (
        f"NFL validation plan only when NFL is selected; got {selected!r}"
    )

    protocol = {
        "protocol_id": config.POLYMARKET_NFL_PAPER_PROTOCOL_ID,
        "protocol_version": "1",
        "registered_at_utc": config.POLYMARKET_NFL_PAPER_PROTOCOL_REGISTERED_UTC,
        "market_id": "nfl_pregame_moneyline",
        "venue": "polymarket_us_provisional",
        "parent_selection_protocol_id": "market_selection_next_sport_v1",
        "parent_selection_protocol_hash": feasibility.get("protocol_hash"),
        "objective": (
            "Prospective paper evaluation of NFL pregame moneylines on provisional "
            "Polymarket US public data. Capture and mapping first; no nested outcome "
            "inspection until structural floors and frozen periods are assigned."
        ),
        "gates": {
            "model_development": "blocked_until_protocol_checkpoints_met",
            "real_money": "blocked",
            "automated_orders": "blocked",
            "nested_evaluation_outcomes": "not_opened",
            "mlb_evidence_reuse": "forbidden",
        },
        "identity": {
            "sport_game_key": "nfl:{game_id}",
            "native_game_id": "nflreadpy game_id",
            "mapping_required": True,
            "unmatched_markets_excluded_from_evaluation": True,
        },
        "cost_assumptions": {
            "fill_model": "walk_recorded_asks_only",
            "display_price_is_not_a_fill": True,
            "fee_formula": "theta * C * p * (1-p)",
            "manual_delay_seconds_grid": [0, 30, 60, 120],
            "adverse_ticks_grid": [0, 1, 2, 5],
            "conservative_delay_seconds": 60,
            "conservative_adverse_ticks": 2,
            "missing_liquidity_is_not_zero_cost": True,
            "canceled_not_auto_zero": True,
        },
        "structural_floors": {
            "min_eligible_dates": config.POLYMARKET_MIN_ELIGIBLE_DATES,
            "note": (
                "One NFL regular season (~2–3 independent game-days/week) is unlikely "
                "to reach the 70-date floor alone. Multi-season collection or an explicit "
                "NFL date-block redesign is required before promotion-scale claims."
            ),
        },
        "chronological_periods": {
            "timezone": "America/New_York",
            "prospective_collection_start_local": "2026-09-16",
            "exploratory_capture_only": True,
            "freeze_assignment": (
                "Assigned only once structural floor eligible dates exist. "
                "Do not open nested evaluation outcomes before freeze assignment."
            ),
            "untouched_evaluation": (
                "Final holdout eligible dates reserved after nested outer folds; "
                "not inspected during tuning. Separate from MLB freeze tails."
            ),
            "training_tuning": (
                "All non-exploratory eligible dates strictly before the frozen "
                "untouched evaluation tail; inner folds only for threshold selection."
            ),
        },
        "candidates_when_evaluation_opens": [
            "market_mid_baseline",
            "existing_nfl_game_prediction_prior_if_available",
        ],
        "evidence_collection_requirements": [
            "Continue public NFL moneyline capture under data/polymarket/nfl/.",
            "Persist mapping coverage; investigate unmatched markets without forcing joins.",
            "Record best bid/ask and eligibility; empty books remain liquidity failures.",
            "Settle only from official schedule results after kickoff; unknown ≠ zero.",
            "Register any comparative candidate experiment before inspecting its outcomes.",
            "Do not purchase data or assume account access beyond public gateway.",
        ],
        "checkpoints_before_opening_nested_outcomes": [
            "Mapped contracts with repeated pregame books across ≥ min_eligible_dates "
            "independent local dates (or explicit redesign of floor documented).",
            "Paper ledger delay/adverse-tick cost assumptions frozen.",
            "Untouched evaluation dates assigned and sealed.",
        ],
        "edge_claimed": False,
        "future_observations_claimed": False,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
    }
    protocol["protocol_hash"] = _sha(protocol)

    path = config.POLYMARKET_NFL_PAPER_PROTOCOL_PATH
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
    print("Nested evaluation outcomes not opened; edge_claimed=False.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
