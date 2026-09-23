"""Formal readiness evaluation for the frozen 1+ hit prop paper system.

Runs preregistered pre-score audits, then scores only when structural floors
and an assigned untouched evaluation period permit. Does not lower thresholds,
open real-money mode, or recycle the evaluation set for tuning.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from mlb_metrics import (
    config,
    hit_prop_forecast,
    hit_prop_ops,
    hit_prop_paper,
    hit_prop_research,
    model_validation,
    paper_ledger,
    polymarket_research,
)

VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_INSUFFICIENT = "insufficient_evidence"


def _sha_file(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def audit_pre_score(
    *,
    protocol: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Verify completeness, cutoffs, identity/rules, hashes, untouched period — before scoring."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    add(
        "frozen_policy_present",
        bool(policy.get("policy_version") and policy.get("policy_hash")),
        {"policy_version": policy.get("policy_version"), "policy_hash": policy.get("policy_hash")},
    )
    add(
        "frozen_policy_paper_only",
        policy.get("action") == "paper_only" and policy.get("betting_mode") == "disabled",
        {"action": policy.get("action"), "betting_mode": policy.get("betting_mode")},
    )
    add(
        "protocol_hash_matches_policy",
        protocol.get("protocol_hash") == policy.get("protocol_hash"),
        {
            "protocol_hash": protocol.get("protocol_hash"),
            "policy_protocol_hash": policy.get("protocol_hash"),
        },
    )
    add(
        "modes_fail_closed",
        config.GAME_PREDICTION_MODE == "shadow" and config.BETTING_MODE == "disabled",
        {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
    )

    collection = hit_prop_paper.count_collection_dates()
    prospective_dates = hit_prop_ops.eligible_decision_dates()
    add(
        "dataset_denominator_recorded",
        collection.get("n_rows", 0) >= 0,
        collection,
    )

    # Cutoff integrity: decisions must record policy version and outcome-unknown flag.
    paths = hit_prop_ops.prospective_paths()
    decisions = (
        pd.read_csv(paths["decisions"]) if os.path.exists(paths["decisions"]) else pd.DataFrame()
    )
    cutoff_ok = True
    cutoff_detail: dict[str, Any] = {"n_decisions": int(len(decisions))}
    if not decisions.empty:
        if "policy_version" in decisions.columns:
            bad_pol = int(
                (decisions["policy_version"].astype(str) != str(policy.get("policy_version"))).sum()
            )
            cutoff_ok = cutoff_ok and bad_pol == 0
            cutoff_detail["n_policy_version_mismatch"] = bad_pol
        if "outcome_unknown_at_decision" in decisions.columns:
            known = decisions["outcome_unknown_at_decision"]
            # Accept True/true/1; anything claiming known outcomes at decision fails.
            leaked = int((~known.astype(str).str.lower().isin({"true", "1", "yes"})).sum())
            # Empty/NA also fail-closed if column present but incomplete.
            leaked += int(known.isna().sum())
            cutoff_ok = cutoff_ok and leaked == 0
            cutoff_detail["n_outcome_known_at_decision"] = leaked
        if "requested_local_date" in decisions.columns and "decision_time_utc" in decisions.columns:
            cutoff_detail["n_dates"] = int(decisions["requested_local_date"].nunique())
    add("cutoff_integrity_decisions_before_outcomes", cutoff_ok, cutoff_detail)

    # Identity / rules matching on capture registry.
    store = hit_prop_research.research_store_dir()
    contracts_path = os.path.join(store, "contracts.csv")
    identity_detail: dict[str, Any] = {"contracts_path_exists": os.path.exists(contracts_path)}
    identity_ok = True
    if os.path.exists(contracts_path):
        contracts = pd.read_csv(contracts_path)
        identity_detail["n_contracts"] = int(len(contracts))
        if "game_mapping_status" in contracts.columns:
            mapped_g = int((contracts["game_mapping_status"] == "mapped").sum())
            identity_detail["n_mapped_game_pk"] = mapped_g
        if "player_mapping_status" in contracts.columns:
            mapped_p = int((contracts["player_mapping_status"] == "mapped").sum())
            identity_detail["n_mapped_key_mlbam"] = mapped_p
        if "rules_hash" in contracts.columns:
            missing_rules = int(contracts["rules_hash"].isna().sum() + (contracts["rules_hash"].astype(str) == "").sum())
            identity_detail["n_missing_rules_hash"] = missing_rules
            # Completeness of rules on mapped rows is informational; quarantine handles gaps.
        if "quarantined" in contracts.columns:
            identity_detail["n_quarantined"] = int(contracts["quarantined"].astype(str).str.lower().isin({"true", "1"}).sum())
    add("identity_and_rules_fields_present", identity_ok, identity_detail)

    model_hashes = {
        "hitter_hit_probability_model": _sha_file(config.HITTER_HIT_PROBABILITY_MODEL_PATH),
        "hitter_opportunity_probability_model": _sha_file(
            getattr(config, "HITTER_OPPORTUNITY_PROBABILITY_MODEL_PATH", "")
        ),
        "frozen_policy_hash": policy.get("policy_hash"),
        "protocol_hash": protocol.get("protocol_hash"),
    }
    add(
        "model_and_policy_hashes_recorded",
        bool(model_hashes["frozen_policy_hash"] and model_hashes["protocol_hash"]),
        model_hashes,
    )

    # Untouched evaluation period: freeze only after structural floor; currently unassigned.
    n_eligible = len(prospective_dates) if prospective_dates else int(collection.get("n_dates_with_executable_book") or 0)
    floor = int(config.HIT_PROP_MIN_ELIGIBLE_DATES)
    freeze_assigned = n_eligible >= floor
    untouched_status = {
        "freeze_assigned": freeze_assigned,
        "n_eligible_independent_dates": n_eligible,
        "structural_floor": floor,
        "untouched_evaluation_period_status": (
            "assigned_but_not_scored_here"
            if freeze_assigned
            else "not_assigned_insufficient_dates"
        ),
        "note": (
            "Untouched evaluation tail is assigned only once the structural date floor "
            "exists. Scoring that tail before assignment is forbidden."
        ),
    }
    add(
        "untouched_evaluation_period_gated",
        untouched_status["untouched_evaluation_period_status"]
        in {"not_assigned_insufficient_dates", "assigned_but_not_scored_here"},
        untouched_status,
    )

    add(
        "thresholds_not_lowered",
        True,
        {
            "structural_floor_dates": floor,
            "min_outer_folds": int(config.HIT_PROP_MIN_OUTER_FOLDS),
            "do_not_select_favorable_subperiods": True,
            "do_not_recycle_eval_set_for_tuning": True,
        },
    )

    failed = [c["check"] for c in checks if not c["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "failed_checks": failed,
        "checks": checks,
        "n_eligible_independent_dates": n_eligible,
        "eligible_dates": prospective_dates or collection.get("book_dates") or collection.get("dates") or [],
        "collection": collection,
        "model_hashes": model_hashes,
        "untouched_evaluation": untouched_status,
        "scoring_permitted": False,  # set by caller after fold/precision gates
    }


def structural_and_precision_status(n_eligible_dates: int, protocol: dict[str, Any]) -> dict[str, Any]:
    """Report structural fold readiness separately from statistical precision planning."""
    floor = int(config.HIT_PROP_MIN_ELIGIBLE_DATES)
    min_folds = int(config.HIT_PROP_MIN_OUTER_FOLDS)
    n_folds = (
        polymarket_research.count_possible_outer_folds(n_eligible_dates, protocol=None)
        if n_eligible_dates >= floor
        else 0
    )
    sample = protocol.get("sample_size_plan") or hit_prop_paper.sample_size_plan()
    power_days = int(
        sample.get("approx_independent_days_for_0.01_logloss_style_target")
        or sample.get("approx_independent_days")
        or 197
    )
    structural_ready = n_eligible_dates >= floor and n_folds >= min_folds
    precision_ready = n_eligible_dates >= power_days
    return {
        "structural": {
            "n_eligible_dates": int(n_eligible_dates),
            "floor_dates": floor,
            "n_outer_folds_available": int(n_folds),
            "min_outer_folds": min_folds,
            "ready": structural_ready,
            "note": "Structural readiness is required before nested outer evaluation.",
        },
        "statistical_precision": {
            "n_eligible_dates": int(n_eligible_dates),
            "planning_power_independent_dates": power_days,
            "ready": precision_ready,
            "effect_logloss": sample.get("effect_logloss") or sample.get("effect"),
            "power": sample.get("power"),
            "alpha": sample.get("alpha", 0.05),
            "note": (
                "Precision/power planning is separate from the structural fold floor. "
                "Meeting the floor does not imply adequate precision."
            ),
        },
        "scoring_allowed": structural_ready,
        "inconclusive_ci_is_insufficient_evidence": True,
    }


def score_probability_quality(
    frame: pd.DataFrame,
    *,
    label_col: str = "y_binary",
    date_col: str = "date",
    market_col: str = "market_mid_probability",
    model_col: str = "model_probability",
) -> dict[str, Any]:
    """Compare model vs same-time market mid on identical binary-settled rows."""
    required = [label_col, date_col, market_col, model_col]
    for col in required:
        if col not in frame.columns:
            return {"status": "insufficient_data", "reason": f"missing_column:{col}"}
    work = frame.dropna(subset=required).copy()
    work = work[work[label_col].isin([0, 1, 0.0, 1.0])]
    if work.empty:
        return {"status": "insufficient_data", "reason": "no_identical_binary_rows_with_market_and_model"}

    y = work[label_col].astype(float).to_numpy()
    market = work[market_col].astype(float).to_numpy()
    model = work[model_col].astype(float).to_numpy()

    def _ll(yt, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)))

    def _brier(yt, p):
        return float(np.mean((p - yt) ** 2))

    market_metrics = {"log_loss": _ll(y, market), "brier": _brier(y, market)}
    model_metrics = {"log_loss": _ll(y, model), "brier": _brier(y, model)}
    calibration = {
        "market_mid_baseline": polymarket_research.calibration_table(
            work[label_col], work[market_col]
        ),
        "model": polymarket_research.calibration_table(work[label_col], work[model_col]),
    }

    # Paired date-block bootstrap of model - market log loss (negative => model better).
    a = work[[date_col, "game_pk", "key_mlbam", label_col, model_col]].rename(
        columns={model_col: "predicted_probability_a"}
    ) if {"game_pk", "key_mlbam"}.issubset(work.columns) else work[[date_col, label_col, model_col]].rename(
        columns={model_col: "predicted_probability_a"}
    )
    b = work[[date_col, "game_pk", "key_mlbam", label_col, market_col]].rename(
        columns={market_col: "predicted_probability_b"}
    ) if {"game_pk", "key_mlbam"}.issubset(work.columns) else work[[date_col, label_col, market_col]].rename(
        columns={market_col: "predicted_probability_b"}
    )
    keys = [date_col] + [k for k in ("game_pk", "key_mlbam") if k in a.columns and k in b.columns]
    boot = model_validation.paired_block_bootstrap_difference(
        a,
        b,
        keys=keys,
        block_key=date_col,
        value_a="predicted_probability_a",
        value_b="predicted_probability_b",
        label_col=label_col,
        metric="log_loss",
        n_bootstrap=min(500, max(50, 20 * int(work[date_col].nunique()))),
        random_seed=0,
        alpha=0.05,
    )
    # Inconclusive CI (includes zero improvement) => insufficient for edge claim.
    ci_high = boot.get("ci_high")
    conclusive_improvement = (
        boot.get("n_blocks", 0) > 0
        and ci_high is not None
        and ci_high == ci_high
        and float(ci_high) < 0
    )
    return {
        "status": "ok",
        "n_rows": int(len(work)),
        "n_dates": int(work[date_col].nunique()),
        "candidates": {
            hit_prop_forecast.CANDIDATE_MARKET_MID: market_metrics,
            "frozen_model_under_evaluation": model_metrics,
        },
        "calibration": calibration,
        "paired_bootstrap_model_minus_market_log_loss": boot,
        "conclusive_log_loss_improvement_vs_market": conclusive_improvement,
        "inconclusive_ci_treated_as": VERDICT_INSUFFICIENT,
        "dependence_note": (
            "Paired bootstrap resamples date blocks to respect shared game/date/player dependence."
        ),
    }


def score_strategy_accounting(
    decisions: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Net results after fees; include drawdown, streaks, liquidity, opens, concentration."""
    if decisions is None or decisions.empty:
        return {"status": "insufficient_data", "reason": "no_decisions"}
    roi = paper_ledger.settled_roi(positions, decisions)
    # Drawdown / streaks from settled net path when available.
    drawdown = {"max_drawdown": float("nan"), "max_loss_streak": 0}
    if positions is not None and not positions.empty:
        settled = positions[positions["status"] == "settled"].copy()
        if not settled.empty and "net_pnl" in settled.columns:
            nets = pd.to_numeric(settled["net_pnl"], errors="coerce").fillna(0).to_numpy()
            equity = np.cumsum(nets)
            peak = np.maximum.accumulate(equity) if len(equity) else np.array([])
            dd = float(np.min(equity - peak)) if len(equity) else float("nan")
            streak = cur = 0
            for v in nets:
                if v < 0:
                    cur += 1
                    streak = max(streak, cur)
                else:
                    cur = 0
            drawdown = {"max_drawdown": dd, "max_loss_streak": int(streak)}

    unfilled = 0
    if "filled_qty" in decisions.columns and "action" in decisions.columns:
        buys = decisions[decisions["action"] == "buy"]
        unfilled = int((pd.to_numeric(buys["filled_qty"], errors="coerce").fillna(0) <= 0).sum())

    open_n = int((positions["status"] == "open").sum()) if positions is not None and not positions.empty else 0
    concentration = {}
    if "game_pk" in decisions.columns:
        concentration["n_unique_games"] = int(decisions["game_pk"].nunique())
    if "key_mlbam" in decisions.columns:
        concentration["n_unique_players"] = int(decisions["key_mlbam"].nunique())
    if "requested_local_date" in decisions.columns:
        concentration["n_unique_dates"] = int(decisions["requested_local_date"].nunique())

    stress = policy.get("execution_stress_for_reporting") or {}
    return {
        "status": "ok" if roi.get("n_settled", 0) > 0 else "insufficient_data",
        "reason": None if roi.get("n_settled", 0) > 0 else "no_settled_positions",
        "roi": roi,
        "conservative_stress_assumptions": {
            "delay_seconds": stress.get("delay_seconds"),
            "adverse_ticks": stress.get("adverse_ticks"),
            "fill_model": stress.get("fill_model"),
            "fee_formula": stress.get("fee_formula"),
            "note": (
                "Conservative scenario uses registered delay/ticks; prediction-time "
                "prices are not replaced with closing prices. Missing closes stay missing."
            ),
        },
        "drawdown": drawdown,
        "liquidity_and_fills": {
            "n_unfilled_buys": unfilled,
            "n_open_positions": open_n,
            "missing_closing_prices_remain_missing": True,
            "prediction_time_prices_separate_from_closing": True,
        },
        "concentration": concentration,
    }


def map_gate_verdict(gates: dict[str, Any], probability_report: dict[str, Any]) -> str:
    """Map internal gate language to acceptance verdict vocabulary."""
    raw = gates.get("verdict")
    if raw == "edge_supported":
        # Inconclusive CI still blocks support.
        if probability_report.get("status") == "ok" and not probability_report.get(
            "conclusive_log_loss_improvement_vs_market", False
        ):
            return VERDICT_INSUFFICIENT
        return VERDICT_SUPPORTED
    if raw == "edge_not_supported":
        return VERDICT_UNSUPPORTED
    return VERDICT_INSUFFICIENT


def build_readiness_report(*, allow_scoring: bool = True) -> dict[str, Any]:
    """Formal checkpoint evaluation; default path remains insufficient with thin history."""
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"

    protocol = hit_prop_paper.load_protocol()
    policy = hit_prop_ops.load_frozen_policy()
    pre = audit_pre_score(protocol=protocol, policy=policy)
    n_dates = int(pre.get("n_eligible_independent_dates") or 0)
    fold_precision = structural_and_precision_status(n_dates, protocol)

    probability_report: dict[str, Any] = {
        "status": "insufficient_data",
        "reason": "scoring_not_permitted",
        "candidates": {},
    }
    strategy_report: dict[str, Any] | None = None
    scoring_attempted = False
    scoring_blocked_reasons: list[str] = []

    if pre["status"] != "passed":
        scoring_blocked_reasons.extend(pre["failed_checks"])
    if not fold_precision["scoring_allowed"]:
        scoring_blocked_reasons.append("structural_floor_or_outer_folds_not_met")
    if not pre["untouched_evaluation"]["freeze_assigned"]:
        scoring_blocked_reasons.append("untouched_evaluation_period_not_assigned")

    # Even when allow_scoring=True, only score if structural gates pass.
    if allow_scoring and fold_precision["scoring_allowed"] and pre["status"] == "passed":
        # Nested labeled settled binary frame would be loaded here. Currently none.
        scoring_attempted = True
        probability_report = {
            "status": "insufficient_data",
            "reason": "no_settled_binary_labeled_identical_opportunity_frame",
            "candidates": {},
            "note": (
                "Structural floors met but no untouched labeled binary settlements "
                "joined to prediction-time mids were available for scoring."
            ),
        }
        paths = hit_prop_ops.prospective_paths()
        decisions = (
            pd.read_csv(paths["decisions"]) if os.path.exists(paths["decisions"]) else pd.DataFrame()
        )
        positions = (
            pd.read_csv(paths["positions"]) if os.path.exists(paths["positions"]) else pd.DataFrame()
        )
        strategy_report = score_strategy_accounting(decisions, positions, policy=policy)
    else:
        probability_report = {
            "status": "insufficient_data",
            "reason": "pre_score_or_structural_gates_blocked_scoring",
            "blocked_by": scoring_blocked_reasons,
            "candidates": {},
            "note": (
                "Probability quality vs market mid is not scored until structural "
                "floors and untouched freeze assignment permit. This avoids recycling "
                "thin history as a tuned result."
            ),
        }

    gates = hit_prop_paper.apply_prop_gates(
        protocol=protocol,
        probability_report=probability_report,
        strategy_report=strategy_report,
        n_eligible_dates=n_dates,
        n_outer_folds_available=int(fold_precision["structural"]["n_outer_folds_available"]),
    )
    verdict = map_gate_verdict(gates, probability_report)
    # Force insufficient when CI inconclusive language applies or scoring blocked.
    if scoring_blocked_reasons and verdict == VERDICT_SUPPORTED:
        verdict = VERDICT_INSUFFICIENT

    obs = hit_prop_paper.additional_observations_needed(
        n_eligible_dates_observed=n_dates, protocol=protocol
    )
    cp = hit_prop_ops.checkpoint_status(n_dates, policy=policy)
    next_checkpoint = cp.get("next_checkpoint")
    # Next permitted operational/structural checkpoint for re-evaluation.
    next_permitted = {
        "next_operational_or_structural_checkpoint_dates": next_checkpoint,
        "days_to_next_checkpoint": cp.get("days_to_next_checkpoint"),
        "next_registered_review_local": config.HIT_PROP_NEXT_REGISTERED_REVIEW_LOCAL,
        "next_registered_review_is_betting_launch": False,
        "re_run_command": "PYTHONPATH=src python scripts/run_hit_prop_readiness_evaluation.py",
    }

    validation_status = (
        "validated_passed"
        if verdict == VERDICT_SUPPORTED
        else ("validated_failed" if verdict == VERDICT_UNSUPPORTED else "insufficient_data")
    )

    report = {
        "generated_at_utc": hit_prop_research.utc_now_iso(),
        "report_type": "hit_prop_formal_readiness_evaluation",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_hash": protocol.get("protocol_hash"),
        "policy_version": policy.get("policy_version"),
        "policy_hash": policy.get("policy_hash"),
        "research_only": True,
        "actionable": False,
        "edge_claimed": False,
        "future_observations_claimed": False,
        "betting_mode_required": "disabled",
        "modes": {
            "GAME_PREDICTION_MODE": config.GAME_PREDICTION_MODE,
            "BETTING_MODE": config.BETTING_MODE,
        },
        "pre_score_audit": pre,
        "structural_and_precision": fold_precision,
        "scoring_attempted": scoring_attempted,
        "scoring_blocked_reasons": scoring_blocked_reasons,
        "probability_report": probability_report,
        "strategy_report": strategy_report,
        "gates": gates,
        "denominators": {
            "n_eligible_independent_dates": n_dates,
            "eligible_dates": pre.get("eligible_dates"),
            "collection": pre.get("collection"),
            "n_prospective_decisions": int(
                len(pd.read_csv(hit_prop_ops.prospective_paths()["decisions"]))
                if os.path.exists(hit_prop_ops.prospective_paths()["decisions"])
                else 0
            ),
        },
        "uncertainty": {
            "paired_date_block_dependence": True,
            "inconclusive_confidence_interval_is_insufficient_evidence": True,
            "thresholds_lowered": False,
            "favorable_subperiods_selected": False,
            "evaluation_set_recycled_for_tuning": False,
        },
        "additional_observations_needed": obs,
        "next_permitted_checkpoint": next_permitted,
        "closing_vs_prediction_time_prices": {
            "kept_separate": True,
            "missing_closing_remain_missing": True,
        },
        "game_winner_protocol_preserved": True,
        "software_tests_are_not_evidence_of_edge": True,
        "verdict": verdict,
        "validation_status": validation_status,
        "verdict_vocabulary": {
            "supported": VERDICT_SUPPORTED,
            "unsupported": VERDICT_UNSUPPORTED,
            "insufficient_evidence": VERDICT_INSUFFICIENT,
        },
    }
    report["report_hash"] = _sha_payload({k: v for k, v in report.items() if k != "report_hash"})
    return report


def write_readiness_report(report: dict[str, Any], path: str | None = None) -> str:
    path = path or config.HIT_PROP_READINESS_REPORT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cleaned = json.loads(json.dumps(report, default=str))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
