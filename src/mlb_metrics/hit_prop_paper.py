"""Hit-prop paper study: frozen policy, ledger accounting, insufficient-evidence eval.

Preserves the game-winner protocol untouched. Nested prop evaluation outcomes
stay closed until structural floors are met. September 25 is a paper-system
review, not a betting launch.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd

from mlb_metrics import config, hit_prop_forecast, hit_prop_research, paper_ledger, polymarket_research


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_protocol(path: str | None = None) -> dict[str, Any]:
    path = path or config.HIT_PROP_PAPER_PROTOCOL_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_size_plan(
    *,
    effect: float | None = None,
    sigma: float = 0.05,
) -> dict[str, Any]:
    """Defensible planning estimate using development assumptions (not holdout)."""
    effect = float(effect if effect is not None else config.HIT_PROP_INTENDED_LOGLOSS_IMPROVEMENT)
    plan = polymarket_research.sample_size_for_paired_mean(effect=effect, sigma=sigma)
    plan.update(
        {
            "structural_floor_independent_dates": int(config.HIT_PROP_MIN_ELIGIBLE_DATES),
            "min_outer_folds": int(config.HIT_PROP_MIN_OUTER_FOLDS),
            "unit_of_independence": "local_date_block",
            "correlation_note": (
                "Players in the same game and games on the same date are correlated; "
                "do not treat raw prop rows as independent Bernoulli trials."
            ),
            "assumptions_visible": True,
            "uses_final_holdout": False,
            "source": "development_planning_assumptions",
        }
    )
    return plan


def additional_observations_needed(
    *,
    n_eligible_dates_observed: int,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = protocol or (load_protocol() if os.path.exists(config.HIT_PROP_PAPER_PROTOCOL_PATH) else {})
    floor = int(
        (protocol.get("sample_size_plan") or {}).get("structural_floor_independent_dates")
        or config.HIT_PROP_MIN_ELIGIBLE_DATES
    )
    power_days = int(
        (protocol.get("sample_size_plan") or {}).get("approx_independent_days_for_0.01_logloss_style_target")
        or sample_size_plan().get("approx_independent_days")
        or 197
    )
    return {
        "n_eligible_dates_observed": int(n_eligible_dates_observed),
        "structural_floor_independent_dates": floor,
        "additional_dates_to_structural_floor": max(0, floor - int(n_eligible_dates_observed)),
        "planning_power_independent_dates": power_days,
        "additional_dates_to_planning_power_target": max(0, power_days - int(n_eligible_dates_observed)),
        "note": (
            "Structural floor is required before nested outer evaluation can complete. "
            "Planning power target is a separate development estimate, not a pass claim."
        ),
    }


def frozen_policy_document(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or load_protocol()
    doc = {
        "policy_version": config.HIT_PROP_PAPER_POLICY_VERSION,
        "evaluation_version": protocol.get("protocol_version"),
        "registered_at_utc": protocol.get("registered_at_utc"),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "venue_id": "polymarket_us",
        "venue_label": "provisional_polymarket_us_unconfirmed",
        "action": "paper_only",
        "betting_mode": "disabled",
        "market_id": "mlb_hitter_hits_1plus",
        "candidates": protocol.get("candidates"),
        "entry": {
            "cutoff": "strictly_before_scheduled_start",
            "entry_minutes_before_scheduled_start": config.HIT_PROP_ENTRY_MINUTES_BEFORE_START,
            "require_fresh_eligible_book": True,
            "quote_max_age_seconds": config.POLYMARKET_QUOTE_MAX_AGE_SECONDS,
            "require_mapped_game_pk_and_key_mlbam": True,
            "exclude_quarantined": True,
            "size_contracts": config.HIT_PROP_PAPER_ONE_SHARE,
            "exclude_market_only_fallback_from_independent_model": True,
            "do_not_use_positive_ab_hit_rate_as_contract_probability": True,
            "log_candidates_and_passes_before_outcomes": True,
        },
        "execution_stress_for_reporting": {
            "delay_seconds": config.POLYMARKET_CONSERVATIVE_DELAY_SECONDS,
            "adverse_ticks": config.POLYMARKET_CONSERVATIVE_ADVERSE_TICKS,
            "delay_grid": list(config.POLYMARKET_MANUAL_DELAY_SECONDS_GRID),
            "adverse_ticks_grid": list(config.POLYMARKET_ADVERSE_TICKS_GRID),
            "fill_model": "walk_recorded_asks_only",
            "fee_formula": "theta * C * p * (1-p)",
        },
        "settlement": {
            "binary_yes_no_when_qualifying_pa": True,
            "non_participation_uses_last_fair_market_price": True,
            "unknown_outcomes_remain_open": True,
            "missing_historical_quotes_remain_missing": True,
        },
        "prospective_checkpoints_game_days": list(config.HIT_PROP_PROSPECTIVE_CHECKPOINTS_GAME_DAYS),
        "game_winner_protocol_preserved": True,
        "september_25_is_paper_system_review_not_betting_launch": True,
        "evaluation_rule": (
            "Fit preprocessing/calibration/thresholds in chronological training folds only. "
            "Compare candidates on identical opportunities vs same-time market mid. "
            "Do not open nested evaluation outcomes until structural floors are met. "
            "insufficient_data is a valid result."
        ),
        "notes": [
            "Frozen for prospective paper observation of 1+ hit props.",
            "Does not enable live betting or automated orders.",
            "Artifact versions for baseball models are recorded at decision time when used.",
        ],
    }
    doc["policy_hash"] = _sha({k: v for k, v in doc.items() if k != "policy_hash"})
    return doc


def write_frozen_policy(
    path: str | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = path or config.HIT_PROP_FROZEN_POLICY_PATH
    doc = frozen_policy_document(protocol=protocol)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("policy_hash") == doc.get("policy_hash"):
            return existing
        doc["supersedes_policy_hash"] = existing.get("policy_hash")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return doc


def count_collection_dates(store_dir: str | None = None) -> dict[str, Any]:
    """Count research capture dates without opening nested evaluation labels."""
    store = hit_prop_research.research_store_dir(store_dir)
    path = os.path.join(store, "contracts.csv")
    if not os.path.exists(path):
        return {
            "n_rows": 0,
            "n_dates": 0,
            "dates": [],
            "n_mapped_eligible_rows": 0,
            "n_dates_with_executable_book": 0,
        }
    frame = pd.read_csv(path)
    dates = sorted({str(d) for d in frame.get("requested_local_date", pd.Series(dtype=str)).dropna().unique()})
    mapped = frame
    if {"game_mapping_status", "player_mapping_status"}.issubset(frame.columns):
        mapped = frame[
            (frame["game_mapping_status"] == hit_prop_research.MAPPING_MAPPED)
            & (frame["player_mapping_status"] == hit_prop_research.MAPPING_MAPPED)
        ]
    book_dates = []
    if "book_status" in frame.columns and "requested_local_date" in frame.columns:
        book_dates = sorted(
            {
                str(d)
                for d in frame.loc[frame["book_status"] == "captured", "requested_local_date"]
                .dropna()
                .unique()
            }
        )
    return {
        "n_rows": int(len(frame)),
        "n_dates": len(dates),
        "dates": dates,
        "n_mapped_eligible_rows": int(len(mapped)),
        "n_dates_with_executable_book": len(book_dates),
        "book_dates": book_dates,
        "nested_evaluation_outcomes_opened": False,
    }


def apply_prop_gates(
    *,
    protocol: dict[str, Any],
    probability_report: dict[str, Any],
    strategy_report: dict[str, Any] | None,
    n_eligible_dates: int,
    n_outer_folds_available: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    floor = int(config.HIT_PROP_MIN_ELIGIBLE_DATES)
    min_folds = int(config.HIT_PROP_MIN_OUTER_FOLDS)
    if n_eligible_dates < floor:
        reasons.append(f"insufficient_dates:{n_eligible_dates}<structural_floor:{floor}")
    if n_outer_folds_available < min_folds:
        reasons.append(f"insufficient_outer_folds:{n_outer_folds_available}<{min_folds}")
    if probability_report.get("status") != "ok":
        reasons.append(f"probability_status:{probability_report.get('status')}")

    residual = (probability_report.get("candidates") or {}).get(
        hit_prop_forecast.CANDIDATE_RESIDUAL
    )
    market = (probability_report.get("candidates") or {}).get(
        hit_prop_forecast.CANDIDATE_MARKET_MID
    )
    if residual and market and residual.get("log_loss") is not None and market.get("log_loss") is not None:
        if not (float(residual["log_loss"]) < float(market["log_loss"])):
            reasons.append("residual_log_loss_not_better_than_market")
    else:
        reasons.append("missing_residual_or_market_scores_on_nested_eval")

    strategy_ok = False
    if strategy_report:
        roi = (strategy_report.get("roi") or {}).get("roi")
        cons = (strategy_report.get("conservative_roi") or {}).get("roi")
        n_bets = (strategy_report.get("roi") or {}).get("n_settled", 0)
        if roi == roi and float(roi) > 0 and cons == cons and float(cons) > 0:
            if n_bets >= config.BETTING_PROMOTION_MIN_BETS:
                strategy_ok = True
            else:
                reasons.append(f"settled_bets_below_floor:{n_bets}")
        else:
            reasons.append("strategy_roi_not_positive_under_base_and_conservative")
    else:
        reasons.append("strategy_report_unavailable")

    if reasons:
        verdict = "insufficient_evidence"
        if n_eligible_dates >= floor and n_outer_folds_available >= min_folds:
            if any(
                r.startswith("residual_log_loss") or r.startswith("strategy_roi")
                for r in reasons
            ):
                verdict = "edge_not_supported"
        validation_status = (
            "insufficient_data" if verdict == "insufficient_evidence" else "validated_failed"
        )
    else:
        verdict = "edge_supported"
        validation_status = "validated_passed"
        if not strategy_ok:
            verdict = "insufficient_evidence"
            validation_status = "insufficient_data"

    return {
        "verdict": verdict,
        "validation_status": validation_status,
        "betting_mode_required": "disabled",
        "gate_fail_reasons": reasons,
        "strategy_ok": strategy_ok,
        "n_eligible_dates": n_eligible_dates,
        "n_outer_folds_available": n_outer_folds_available,
        "september_25_betting_launch": False,
    }


def ledger_paths(ledger_dir: str | None = None) -> dict[str, str]:
    root = ledger_dir or config.HIT_PROP_LEDGER_DIR
    return {
        "dir": root,
        "decisions": os.path.join(root, "decisions.csv"),
        "positions": os.path.join(root, "positions.csv"),
    }


def simulate_prop_paper_trade(
    *,
    yes_asks: list[dict[str, Any]],
    decision_time_utc: str,
    market_id: str,
    market_slug: str,
    game_pk: Any,
    key_mlbam: Any,
    player_name: str,
    model_name: str,
    model_probability: float | None,
    market_mid_probability: float | None,
    settlement: dict[str, Any] | None = None,
    lfmp_price: float | None = None,
    requested_qty: float | None = None,
    delay_seconds: int = 0,
    adverse_ticks: int = 0,
    policy_version: str | None = None,
) -> dict[str, Any]:
    """Size-limited fill + exact settlement path for props (including LFMP)."""
    fee = paper_ledger.fee_as_of(decision_time_utc)
    qty = float(requested_qty if requested_qty is not None else config.HIT_PROP_PAPER_ONE_SHARE)
    purchase = paper_ledger.simulate_paper_purchase(
        asks=yes_asks,
        requested_qty=qty,
        fee=fee,
        delay_seconds=delay_seconds,
        adverse_ticks=adverse_ticks,
        tick_size=0.01,
        display_price=None,
    )
    decision = paper_ledger.build_decision_row(
        policy_version=policy_version or config.HIT_PROP_PAPER_POLICY_VERSION,
        venue_id="polymarket_us",
        market_id=str(market_id),
        market_slug=str(market_slug),
        game_pk=game_pk,
        side_team=str(player_name or key_mlbam),
        is_long=True,
        decision_time_utc=decision_time_utc,
        quote_receive_time_utc=decision_time_utc,
        quote_request_time_utc=decision_time_utc,
        fee=fee,
        action="buy" if purchase["filled_qty"] > 0 else "pass",
        pass_reason=None if purchase["filled_qty"] > 0 else "unfilled_or_no_liquidity",
        requested_qty=qty,
        fill_assumption="walk_asks",
        purchase=purchase,
        model_name=model_name,
        model_probability=model_probability,
        market_mid_probability_value=market_mid_probability,
        executable_buy=(yes_asks[0]["price"] if yes_asks else None),
        probability_source="hit_prop_research",
    )
    decision["key_mlbam"] = key_mlbam
    decision["stat"] = "hits"
    decision["threshold"] = 1

    settled = None
    if purchase["filled_qty"] <= 0:
        settled = paper_ledger.settle_position(
            filled_qty=0.0,
            acquisition_cost=0.0,
            selected_team=str(player_name or key_mlbam),
            winning_team=None,
        )
    elif settlement is None:
        settled = paper_ledger.settle_position(
            filled_qty=purchase["filled_qty"],
            acquisition_cost=purchase["acquisition_cost"],
            selected_team=str(player_name or key_mlbam),
            winning_team=None,
        )
    elif settlement.get("settlement_class") in {
        "last_fair_market_price",
        "last_fair_market_price_or_unresolved",
    }:
        if lfmp_price is None:
            settled = {
                "status": "open",
                "settlement_payout_per_contract": None,
                "settlement_rule": "lfmp_price_missing_remains_open",
                "proceeds": None,
                "net_pnl": None,
                "open_exposure": float(purchase["acquisition_cost"]),
            }
        else:
            settled = paper_ledger.settle_position(
                filled_qty=purchase["filled_qty"],
                acquisition_cost=purchase["acquisition_cost"],
                selected_team=str(player_name or key_mlbam),
                winning_team=None,
                settlement_px=float(lfmp_price),
                settlement_rule="last_fair_market_price",
            )
    elif settlement.get("binary_yes") is True:
        settled = paper_ledger.settle_position(
            filled_qty=purchase["filled_qty"],
            acquisition_cost=purchase["acquisition_cost"],
            selected_team="YES",
            winning_team="YES",
            settlement_rule="qualifying_pa_binary_yes",
        )
    elif settlement.get("binary_yes") is False:
        settled = paper_ledger.settle_position(
            filled_qty=purchase["filled_qty"],
            acquisition_cost=purchase["acquisition_cost"],
            selected_team="YES",
            winning_team="NO",
            settlement_rule="qualifying_pa_binary_no",
        )
    else:
        settled = paper_ledger.settle_position(
            filled_qty=purchase["filled_qty"],
            acquisition_cost=purchase["acquisition_cost"],
            selected_team=str(player_name or key_mlbam),
            winning_team=None,
            settlement_rule="unknown_pending",
        )

    return {
        "decision": decision,
        "purchase": purchase,
        "settlement_input": settlement,
        "position": settled,
        "research_only": True,
        "actionable": False,
    }


def build_forecast_settlement_example() -> dict[str, Any]:
    """Reproducible hand-checked forecast→fill→settlement demo (not validation)."""
    rules = hit_prop_research.parse_contract_rules(
        "Must be in the starting lineup and record a plate appearance; otherwise last fair market price."
    )
    baseball = hit_prop_forecast.baseball_proxy_components(
        start_rate=0.85,
        game_hit_probability=0.62,
    )
    market_mid = hit_prop_forecast.market_mid_from_executable(0.60, 0.44)
    rejected_ab = hit_prop_forecast.contract_yes_probability(
        p_qualify=1.0,
        p_hit_given_qualify=0.3,
        source="positive_ab_hit_rate",
    )

    walk = hit_prop_research.classify_contract_outcome(
        started=True,
        plate_appearances=1,
        at_bats=0,
        hits=0,
        game_status="Final",
        rules=rules,
    )
    hit = hit_prop_research.classify_contract_outcome(
        started=True,
        plate_appearances=3,
        at_bats=3,
        hits=1,
        game_status="Final",
        rules=rules,
    )
    dnp = hit_prop_research.classify_contract_outcome(
        started=False,
        plate_appearances=0,
        at_bats=0,
        hits=0,
        game_status="Final",
        rules=rules,
    )

    asks = [{"price": 0.60, "size": 5.0}, {"price": 0.61, "size": 10.0}]
    trade_hit = simulate_prop_paper_trade(
        yes_asks=asks,
        decision_time_utc="2026-09-19T20:00:00Z",
        market_id="fixture-prop-1",
        market_slug="fixture-ausmar-gte1",
        game_pk=823976,
        key_mlbam=668885,
        player_name="Austin Martin",
        model_name=hit_prop_forecast.CANDIDATE_BASEBALL,
        model_probability=baseball.get("contract_yes_probability"),
        market_mid_probability=market_mid,
        settlement=hit,
        requested_qty=1.0,
        delay_seconds=0,
        adverse_ticks=0,
    )
    trade_walk = simulate_prop_paper_trade(
        yes_asks=asks,
        decision_time_utc="2026-09-19T20:00:00Z",
        market_id="fixture-prop-2",
        market_slug="fixture-walk-gte1",
        game_pk=823976,
        key_mlbam=668885,
        player_name="Austin Martin",
        model_name=hit_prop_forecast.CANDIDATE_BASEBALL,
        model_probability=baseball.get("contract_yes_probability"),
        market_mid_probability=market_mid,
        settlement=walk,
        requested_qty=1.0,
    )
    trade_dnp = simulate_prop_paper_trade(
        yes_asks=asks,
        decision_time_utc="2026-09-19T20:00:00Z",
        market_id="fixture-prop-3",
        market_slug="fixture-dnp-gte1",
        game_pk=823976,
        key_mlbam=668885,
        player_name="Austin Martin",
        model_name=hit_prop_forecast.CANDIDATE_BASEBALL,
        model_probability=baseball.get("contract_yes_probability"),
        market_mid_probability=market_mid,
        settlement=dnp,
        lfmp_price=0.55,
        requested_qty=1.0,
        delay_seconds=60,
        adverse_ticks=2,
    )
    skipped = simulate_prop_paper_trade(
        yes_asks=[],
        decision_time_utc="2026-09-19T20:00:00Z",
        market_id="fixture-prop-4",
        market_slug="fixture-noliquidity-gte1",
        game_pk=823976,
        key_mlbam=668885,
        player_name="Austin Martin",
        model_name=hit_prop_forecast.CANDIDATE_MARKET_MID,
        model_probability=market_mid,
        market_mid_probability=market_mid,
        settlement=None,
        requested_qty=1.0,
    )

    return {
        "generated_at_utc": hit_prop_research.utc_now_iso(),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "purpose": "reproducible_forecast_to_settlement_demo_not_validation",
        "rules_version": hit_prop_research.HIT_PROP_RULES_VERSION,
        "policy_version": config.HIT_PROP_PAPER_POLICY_VERSION,
        "forecasts": {
            "market_mid_from_executable": market_mid,
            "baseball_contract_proxy": baseball,
            "rejected_positive_ab_source": rejected_ab,
        },
        "settlement_classes": {
            "starter_with_hit": hit,
            "walk_only_pa_zero_hits": walk,
            "did_not_start_lfmp": dnp,
        },
        "paper_trades": {
            "binary_yes_fill": {
                "filled_qty": trade_hit["purchase"]["filled_qty"],
                "acquisition_cost": trade_hit["purchase"]["acquisition_cost"],
                "fees_paid": trade_hit["purchase"]["fees_paid"],
                "position": trade_hit["position"],
            },
            "walk_only_binary_no": {
                "filled_qty": trade_walk["purchase"]["filled_qty"],
                "acquisition_cost": trade_walk["purchase"]["acquisition_cost"],
                "position": trade_walk["position"],
            },
            "dnp_lfmp_with_conservative_stress": {
                "delay_seconds": 60,
                "adverse_ticks": 2,
                "filled_qty": trade_dnp["purchase"]["filled_qty"],
                "acquisition_cost": trade_dnp["purchase"]["acquisition_cost"],
                "position": trade_dnp["position"],
            },
            "skipped_unfilled": {
                "action": skipped["decision"]["action"],
                "position": skipped["position"],
            },
        },
        "notes": [
            "Training-only tuning is demonstrated in unit tests with chronological folds.",
            "This fixture does not open nested evaluation outcomes or claim an edge.",
        ],
    }


def write_forecast_settlement_example(path: str | None = None) -> dict[str, Any]:
    path = path or config.HIT_PROP_FORECAST_EXAMPLE_PATH
    payload = build_forecast_settlement_example()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return payload


def build_evaluation_report(
    *,
    protocol: dict[str, Any] | None = None,
    open_nested_outcomes: bool = False,
) -> dict[str, Any]:
    """Write pass/fail/insufficient report without opening nested outcomes by default."""
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    if open_nested_outcomes:
        raise ValueError(
            "Nested prop evaluation outcomes remain closed until structural floors "
            "and freeze assignment are met; do not open them from this entrypoint."
        )

    protocol = protocol or load_protocol()
    frozen = write_frozen_policy(protocol=protocol)
    collection = count_collection_dates()
    n_dates = int(collection.get("n_dates_with_executable_book") or 0)
    # Nested folds require structural floor; with thin history count is zero.
    n_folds = (
        polymarket_research.count_possible_outer_folds(n_dates, protocol=None)
        if n_dates >= config.HIT_PROP_MIN_ELIGIBLE_DATES
        else 0
    )
    probability_report = {
        "status": "insufficient_data",
        "reason": "nested_evaluation_outcomes_not_opened_insufficient_labeled_history",
        "n_eligible_dates": n_dates,
        "candidates": {},
        "note": (
            "Collection progress is counted separately from nested evaluation. "
            "Missing historical executable quotes remain missing."
        ),
    }
    strategy_report = None
    gates = apply_prop_gates(
        protocol=protocol,
        probability_report=probability_report,
        strategy_report=strategy_report,
        n_eligible_dates=n_dates,
        n_outer_folds_available=n_folds,
    )
    obs = additional_observations_needed(n_eligible_dates_observed=n_dates, protocol=protocol)
    example = write_forecast_settlement_example()

    report = {
        "generated_at_utc": hit_prop_research.utc_now_iso(),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "policy_version": frozen.get("policy_version"),
        "policy_hash": frozen.get("policy_hash"),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "future_observations_claimed": False,
        "september_25_is_paper_system_review_not_betting_launch": True,
        "nested_evaluation_outcomes": "not_opened",
        "game_winner_protocol_preserved": True,
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "collection_progress": collection,
        "probability_report": probability_report,
        "strategy_report": strategy_report,
        "gates": gates,
        "sample_size_plan": protocol.get("sample_size_plan") or sample_size_plan(),
        "additional_observations_needed": obs,
        "feature_availability_audit": {
            "hitter_pregame_features": (
                "WAVE/Game_Hit_Probability/matchup features from published hit log are "
                "pregame-reconstructible when built with as-of cutoff; target-game "
                "Started/Appeared/PA/AB/Hits/Got_Hit are leakage and never used as features."
            ),
            "venue_quotes": (
                "Only captured executable yes_buy/no_buy at decision-time receipts are "
                "eligible. Missing historical quotes stay missing; no fill from today "
                "or closing prices."
            ),
            "positive_ab_rates": "Forbidden as silent contract payout probabilities.",
        },
        "model_candidates": protocol.get("candidates"),
        "forecast_settlement_example_path": config.HIT_PROP_FORECAST_EXAMPLE_PATH,
        "forecast_settlement_example_summary": {
            "market_mid": (example.get("forecasts") or {}).get("market_mid_from_executable"),
            "baseball_contract_proxy": (example.get("forecasts") or {})
            .get("baseball_contract_proxy", {})
            .get("contract_yes_probability"),
            "rejected_positive_ab": (example.get("forecasts") or {})
            .get("rejected_positive_ab_source", {})
            .get("status"),
        },
        "validation_status": gates["validation_status"],
        "verdict": gates["verdict"],
    }
    return report


def write_evaluation_report(report: dict[str, Any], path: str | None = None) -> str:
    path = path or config.HIT_PROP_PAPER_EVALUATION_REPORT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cleaned = json.loads(json.dumps(report, default=str))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
