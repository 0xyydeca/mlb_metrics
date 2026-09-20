"""Register hit-prop paper protocol + freeze paper-only policy (before nested outcomes)."""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, hit_prop_paper


def _sha(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_protocol() -> dict:
    sample = hit_prop_paper.sample_size_plan()
    protocol = {
        "protocol_id": config.HIT_PROP_PAPER_PROTOCOL_ID,
        "protocol_version": "1",
        "registered_at_utc": config.HIT_PROP_PAPER_PROTOCOL_REGISTERED_UTC,
        "market_id": "mlb_hitter_hits_1plus",
        "venue": "polymarket_us_provisional",
        "objective": (
            "Prospective paper evaluation of MLB 1+ hitter-hit props on provisional "
            "Polymarket US public data, using verified game_pk + key_mlbam identities "
            "and exact contract participation/settlement rules."
        ),
        "candidates": [
            "same_time_market_mid_baseline",
            "contract_rule_adjusted_hitter_hit_model",
            "regularized_market_residual_prop_logistic",
        ],
        "rules": {
            "register_before_examining_evaluation_outcomes": True,
            "do_not_convert_positive_ab_hit_rate_directly_to_contract_probability": True,
            "exclude_market_only_fallback_from_independent_model": True,
            "missing_historical_quotes_remain_missing": True,
            "fit_preprocessing_calibration_thresholds_in_training_folds_only": True,
            "game_winner_protocol_preserved": True,
            "automated_orders_absent": True,
            "real_money_gated": True,
            "september_25_is_paper_system_review_not_betting_launch": True,
        },
        "identity": {
            "game_pk_required": True,
            "key_mlbam_required": True,
            "quarantine_ambiguous_names": True,
            "quarantine_wrong_day": True,
            "quarantine_doubleheaders_without_unique_start_match": True,
            "quarantine_provider_id_conflicts": True,
            "quarantine_traded_or_wrong_team": True,
        },
        "contract_semantics": {
            "requires_starting_lineup": True,
            "requires_plate_appearance": True,
            "plate_appearance_not_equal_at_bat": True,
            "walk_only_pa_is_participation": True,
            "zero_hits_with_pa_settles_no": True,
            "non_participation_settles_last_fair_market_price": True,
            "unknown_outcomes_distinct_from_confirmed_zero_ab": True,
            "rules_version": "hit_prop_rules_v1",
        },
        "chronological_periods": {
            "timezone": "America/Phoenix",
            "prospective_collection_start_local": "2026-09-18",
            "exploratory_capture_dates_local": ["2026-09-18", "2026-09-19"],
            "freeze_assignment": (
                "Assigned only once structural floor eligible independent dates exist. "
                "Do not open nested evaluation outcomes before freeze assignment."
            ),
            "training_tuning": (
                "All non-exploratory eligible dates strictly before frozen untouched "
                "evaluation tail; inner folds only for calibration/thresholds."
            ),
            "untouched_evaluation": (
                "Final holdout eligible dates reserved after nested outer folds; "
                "not inspected during tuning. Separate from game-winner freeze tails."
            ),
        },
        "entry_timing": {
            "cutoff": "strictly_before_scheduled_start",
            "require_fresh_eligible_book": True,
            "quote_max_age_seconds": config.POLYMARKET_QUOTE_MAX_AGE_SECONDS,
            "entry_minutes_before_scheduled_start": config.HIT_PROP_ENTRY_MINUTES_BEFORE_START,
        },
        "cost_assumptions": {
            "fill_model": "walk_recorded_asks_only",
            "display_price_is_not_a_fill": True,
            "fee_formula": "theta * C * p * (1-p)",
            "manual_delay_seconds_grid": list(config.POLYMARKET_MANUAL_DELAY_SECONDS_GRID),
            "adverse_ticks_grid": list(config.POLYMARKET_ADVERSE_TICKS_GRID),
            "conservative_delay_seconds": config.POLYMARKET_CONSERVATIVE_DELAY_SECONDS,
            "conservative_adverse_ticks": config.POLYMARKET_CONSERVATIVE_ADVERSE_TICKS,
            "missing_liquidity_is_not_zero_cost": True,
        },
        "eligibility": {
            "mapped_game_pk_and_key_mlbam": True,
            "rules_hash_present": True,
            "quarantined_rows_excluded": True,
            "research_only_until_gates_pass": True,
        },
        "primary_metrics": [
            "log_loss_vs_same_time_market_mid",
            "brier_vs_same_time_market_mid",
            "paper_roi_after_fees_base_and_conservative_stress",
            "drawdown_and_liquidity_sensitivity",
        ],
        "pass_fail": {
            "validation_status_required": "validated_passed",
            "verdict_required_for_pilot_consideration": "edge_supported",
            "insufficient_data_is_valid_result": True,
            "do_not_lower_thresholds_to_force_pass": True,
        },
        "fold_structure": {
            "min_dates_structural_floor": config.HIT_PROP_MIN_ELIGIBLE_DATES,
            "min_outer_folds": config.HIT_PROP_MIN_OUTER_FOLDS,
            "note": "Aligned with game-winner structural floor; separate freeze tails.",
        },
        "sample_size_plan": {
            "structural_floor_independent_dates": config.HIT_PROP_MIN_ELIGIBLE_DATES,
            "approx_independent_days_for_0.01_logloss_style_target": sample[
                "approx_independent_days"
            ],
            "power": sample["power"],
            "alpha": sample["alpha"],
            "effect_logloss": sample["effect_logloss"],
            "assumed_day_sd": sample["assumed_day_sd"],
            "uses_final_holdout": False,
            "source": "development_planning_assumptions",
            "note": (
                "Players in the same game and games on the same date are correlated. "
                "Treat independent units as date blocks, not raw prop rows."
            ),
        },
        "evidence_collection_estimate_by_2026_09_25": {
            "calendar_days_available_through_review": 7,
            "expected_independent_dates_upper_bound": 7,
            "meets_structural_floor": False,
            "expected_report_status": "insufficient_data",
            "note": (
                "September 25 is a paper-system review, not a betting launch. "
                "Deliver reproducible workflow and an honest insufficient-evidence report."
            ),
        },
        "gates": {
            "model_development_for_production_serving": "blocked_until_protocol_checkpoints_met",
            "nested_evaluation_outcomes": "not_opened",
            "real_money": "blocked",
            "game_winner_protocol": "unchanged",
        },
        "edge_claimed": False,
        "future_observations_claimed": False,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
    }
    protocol["protocol_hash"] = _sha(protocol)
    return protocol


def main() -> int:
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    protocol = build_protocol()
    path = config.HIT_PROP_PAPER_PROTOCOL_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("protocol_hash") == protocol["protocol_hash"]:
            print(f"Protocol unchanged: {path}")
        else:
            protocol["supersedes_protocol_hash"] = existing.get("protocol_hash")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(protocol, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"Wrote {path}")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote {path}")

    frozen = hit_prop_paper.write_frozen_policy(protocol=protocol)
    print(f"Frozen policy: {config.HIT_PROP_FROZEN_POLICY_PATH}")
    print(f"protocol_hash={protocol['protocol_hash']}")
    print(f"policy_hash={frozen.get('policy_hash')}")
    print("Nested evaluation outcomes not opened; edge_claimed=False.")
    print("September 25 = paper-system review, not betting launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
