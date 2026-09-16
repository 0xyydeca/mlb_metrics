"""Registered Polymarket paper evaluation protocol and evidence reports.

Register the hypothesis and criteria BEFORE opening new evaluation outcomes.
Dates inspected during foundation work are exploratory and excluded from
untouched validation / freeze claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mlb_metrics import config, game_residual_model as grm, model_validation, paper_ledger


PROTOCOL_VERSION = config.POLYMARKET_PAPER_PROTOCOL_VERSION


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class PaperResearchProtocol:
    """Immutable research registration for Polymarket game-winner paper eval."""

    protocol_id: str = config.POLYMARKET_PAPER_PROTOCOL_ID
    protocol_version: str = PROTOCOL_VERSION
    registered_at_utc: str = config.POLYMARKET_PAPER_PROTOCOL_REGISTERED_UTC
    venue_id: str = "polymarket_us"
    venue_label: str = "polymarket_us_provisional_research"
    market_type: str = "pregame_full_game_winner"
    hypothesis: str = (
        "Baseball information available before the decision cutoff improves "
        "same-time Polymarket book mid probabilities on log loss and Brier, "
        "and a predeclared entry policy has positive net paper ROI after "
        "fees under conservative delay/adverse-tick stress."
    )
    candidates: tuple[str, ...] = (
        "market_mid_baseline",
        "calibrated_heuristic_baseball",
        "regularized_market_residual_logistic",
    )
    primary_probability_metric: str = "log_loss"
    secondary_probability_metrics: tuple[str, ...] = ("brier", "auc")
    primary_return_metric: str = "settled_net_roi_after_fees"
    secondary_return_metrics: tuple[str, ...] = (
        "max_drawdown",
        "fill_rate",
        "skipped_opportunity_counts",
        "conservative_delay_adverse_roi",
    )
    uncertainty_methods: dict[str, Any] = field(
        default_factory=lambda: {
            "probability": "paired_date_block_bootstrap_on_log_loss_diff",
            "calibration": "equal_width_probability_bins",
            "returns": "settled_only_roi_with_open_exposure_reported_separately",
            "dependence": "games_sharing_a_date_are_one_block_not_independent_bets",
            "no_invented_confidence_score": True,
        }
    )
    intended_logloss_improvement: float = config.POLYMARKET_INTENDED_LOGLOSS_IMPROVEMENT
    cost_assumptions: dict[str, Any] = field(
        default_factory=lambda: {
            "fill_model": "walk_recorded_asks_only",
            "display_price_is_not_a_fill": True,
            "fee_formula": "theta * C * p * (1-p)",
            "fee_schedules": list(config.POLYMARKET_US_FEE_SCHEDULES),
            "default_size_contracts": config.POLYMARKET_PAPER_ONE_SHARE,
            "max_size_contracts": config.POLYMARKET_PAPER_MAX_CONTRACTS,
            "manual_delay_seconds_grid": list(config.POLYMARKET_MANUAL_DELAY_SECONDS_GRID),
            "adverse_ticks_grid": list(config.POLYMARKET_ADVERSE_TICKS_GRID),
            "conservative_delay_seconds": config.POLYMARKET_CONSERVATIVE_DELAY_SECONDS,
            "conservative_adverse_ticks": config.POLYMARKET_CONSERVATIVE_ADVERSE_TICKS,
            "settlement": "official_result_binary_unless_venue_px",
            "canceled_not_auto_zero": True,
        }
    )
    eligibility_rules: dict[str, Any] = field(
        default_factory=lambda: {
            "require_mapped_game_pk": True,
            "require_non_stale_eligible_book": True,
            "quote_max_age_seconds": config.POLYMARKET_QUOTE_MAX_AGE_SECONDS,
            "exclude_crossed_books_from_mid_baseline": True,
            "exclude_market_only_fallback_from_independent_model": True,
            "require_both_probable_pitchers_for_baseball_features": True,
            "exclude_postponed_canceled_suspended": True,
            "doubleheaders_are_separate_game_pk": True,
            "polymarket_prior_requires_revalidation": True,
            "sportsbook_history_is_not_polymarket_evidence": True,
            "log_every_eligible_candidate_and_pass_before_outcomes": True,
        }
    )
    decision_times: dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "nearest_pregame_recorded_quote_at_or_before_entry_window",
            "entry_minutes_before_scheduled_start": config.POLYMARKET_ENTRY_MINUTES_BEFORE_START,
            "cutoff": "strictly_before_scheduled_start",
            "no_post_start_features": True,
            "timestamp_decisions_before_outcomes": True,
        }
    )
    chronological_periods: dict[str, Any] = field(
        default_factory=lambda: {
            "timezone": "America/Phoenix",
            "exploratory_dates_local": list(config.POLYMARKET_EXPLORATORY_DATES),
            "prospective_collection_start_local": config.POLYMARKET_PROSPECTIVE_COLLECTION_START_LOCAL,
            "regular_season_end_local": config.POLYMARKET_2026_REGULAR_SEASON_END_LOCAL,
            "postseason_start_local": config.POLYMARKET_2026_POSTSEASON_START_LOCAL,
            "world_series_end_local": config.POLYMARKET_2026_WORLD_SERIES_END_LOCAL,
            "training_tuning": (
                "All non-exploratory eligible dates strictly before the frozen "
                "untouched evaluation tail; inner folds only for threshold/C selection."
            ),
            "untouched_evaluation": (
                f"Final {config.GAME_RESIDUAL_BETTING_FREEZE_DATES} eligible dates "
                "held out after nested outer folds; not inspected during tuning."
            ),
            "freeze_assignment": "Assigned only once structural floor eligible dates exist.",
        }
    )
    cohorts: dict[str, Any] = field(
        default_factory=lambda: {
            "regular_season": {
                "id": "mlb_2026_regular_season",
                "end_local": config.POLYMARKET_2026_REGULAR_SEASON_END_LOCAL,
                "primary_promotion_cohort": True,
            },
            "postseason": {
                "id": "mlb_2026_postseason",
                "start_local": config.POLYMARKET_2026_POSTSEASON_START_LOCAL,
                "end_local": config.POLYMARKET_2026_WORLD_SERIES_END_LOCAL,
                "primary_promotion_cohort": False,
                "do_not_transfer_regular_season_results": True,
            },
        }
    )
    fold_structure: dict[str, Any] = field(
        default_factory=lambda: {
            "outer_min_train_dates": config.GAME_RESIDUAL_OUTER_MIN_TRAIN_DATES,
            "outer_test_block_dates": config.GAME_RESIDUAL_OUTER_TEST_BLOCK_DATES,
            "inner_min_train_dates": config.GAME_RESIDUAL_INNER_MIN_TRAIN_DATES,
            "inner_test_block_dates": config.GAME_RESIDUAL_INNER_TEST_BLOCK_DATES,
            "freeze_dates": config.GAME_RESIDUAL_BETTING_FREEZE_DATES,
            "require_complete_test_blocks": True,
            "min_outer_folds": config.BETTING_PROMOTION_MIN_OUTER_FOLDS,
            "min_dates_structural_floor": config.GAME_RESIDUAL_MIN_DATES_FOR_COMPLETE_OUTER_FOLDS,
        }
    )
    evaluation_checkpoints: dict[str, Any] = field(
        default_factory=lambda: {
            "game_days": list(config.POLYMARKET_PROSPECTIVE_CHECKPOINTS_GAME_DAYS),
            "rule": (
                "At each checkpoint write a reproducible report from saved inputs. "
                "Do not promote or enable betting on interim checkpoints. "
                "Do not inspect the registered freeze tail during tuning."
            ),
            "checkpoint_7_14_are_progress_only": True,
            "promotion_requires_structural_floor_and_gates": True,
        }
    )
    exploratory_dates: tuple[str, ...] = config.POLYMARKET_EXPLORATORY_DATES
    tuning_budget: dict[str, Any] = field(
        default_factory=lambda: {
            "logistic_C_grid": list(config.GAME_RESIDUAL_LOGISTIC_C_GRID),
            "edge_threshold_grid": list(config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID),
            "nonlinear_challenger_deferred": True,
            "select_on_inner_folds_only": True,
            "final_freeze_untouched": True,
            "operational_fixes_do_not_require_new_evaluation_version": True,
            "policy_or_metric_changes_require_new_evaluation_version": True,
        }
    )
    pass_fail_criteria: dict[str, Any] = field(
        default_factory=lambda: {
            "probability_gate": {
                "require_complete_outer_folds": config.BETTING_PROMOTION_MIN_OUTER_FOLDS,
                "primary": "log_loss_point_better_than_market_mid",
                "paired_day_block_ci_excludes_zero_on_log_loss": True,
                "also_report_brier": True,
            },
            "strategy_gate": {
                "positive_roi_on_untouched_outer": True,
                "conservative_stress_roi_lower_bound_above_zero": True,
                "min_settled_bets_floor": config.BETTING_PROMOTION_MIN_BETS,
                "min_bet_dates_floor": config.BETTING_PROMOTION_MIN_BET_DATES,
                "floors_are_not_power_proof": True,
            },
            "insufficient_data_when_below_structural_floor": True,
            "betting_mode_if_fail": "disabled",
            "betting_mode_if_insufficient_data": "disabled",
        }
    )
    collection_host: dict[str, Any] = field(
        default_factory=lambda: {
            "host": config.POLYMARKET_COLLECTION_HOST,
            "runtime_note": config.POLYMARKET_COLLECTION_RUNTIME_NOTE,
            "workflow": ".github/workflows/polymarket_capture.yml",
            "paid_services_purchased": False,
            "missing_host": None,
        }
    )
    notes: tuple[str, ...] = (
        "Registered before opening new prospective evaluation outcomes for this protocol version.",
        "Exploratory dates inspected during data-foundation work are not untouched validation.",
        "Regular-season and postseason cohorts are evaluated separately; results do not transfer.",
        "The remaining 2026 regular season is expected to be below the structural date floor; "
        "insufficient_data remains a valid conclusion and collection continues past this season.",
        "Owner places bets manually; this module never places orders.",
        "Never claim future observations have already occurred.",
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["protocol_hash"] = _sha({k: v for k, v in d.items() if k != "protocol_hash"})
        return d


def sample_size_for_paired_mean(
    *,
    effect: float,
    sigma: float = 0.05,
    power: float = config.POLYMARKET_PRECISION_TARGET_POWER,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Approximate independent-day sample need for mean paired log-loss improvement.

    Uses normal two-sided z approximation. Day-block dependence is handled in
    evaluation via date bootstrap; this is a planning lower bound only.
    """
    effect = abs(float(effect))
    sigma = max(float(sigma), 1e-6)
    # z_0.975 ≈ 1.96, z_power(0.8) ≈ 0.84
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143 if abs(power - 0.8) < 1e-9 else 1.2815515655446004
    n = ((z_alpha + z_power) * sigma / effect) ** 2
    n_ceil = int(math.ceil(n))
    return {
        "effect_logloss": effect,
        "assumed_day_sd": sigma,
        "power": power,
        "alpha": alpha,
        "approx_independent_days": n_ceil,
        "structural_floor_dates": int(config.GAME_RESIDUAL_MIN_DATES_FOR_COMPLETE_OUTER_FOLDS),
        "note": (
            "Planning estimate only. Actual inference uses paired date-block "
            "bootstrap; shared-date games are not independent bets."
        ),
    }


def register_protocol(path: str | None = None) -> dict[str, Any]:
    """Write the immutable protocol registration (idempotent if unchanged)."""
    path = path or config.POLYMARKET_PAPER_PROTOCOL_PATH
    protocol = PaperResearchProtocol().to_dict()
    protocol["sample_size_plan"] = sample_size_for_paired_mean(
        effect=config.POLYMARKET_INTENDED_LOGLOSS_IMPROVEMENT
    )
    protocol["remaining_season_capacity"] = remaining_season_capacity()
    # Recompute hash including sample/capacity annexes.
    annex = {
        k: protocol[k]
        for k in protocol
        if k not in ("protocol_hash",)
    }
    protocol["protocol_hash"] = _sha(annex)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("protocol_hash") == protocol.get("protocol_hash"):
            return existing
        protocol["supersedes_protocol_hash"] = existing.get("protocol_hash")
        protocol["previous_registered_at_utc"] = existing.get("registered_at_utc")
        protocol["previous_protocol_id"] = existing.get("protocol_id")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return protocol


def remaining_season_capacity(
    *,
    as_of_local: str | None = None,
    collection_start_local: str | None = None,
) -> dict[str, Any]:
    """Estimate remaining regular-season game dates vs structural/power floors.

    Does not claim future observations have occurred. Calendar span is an
    upper bound on independent dates if every day has eligible games.
    """
    from datetime import date as date_cls

    as_of = date_cls.fromisoformat(as_of_local or config.POLYMARKET_PROSPECTIVE_COLLECTION_START_LOCAL)
    start = date_cls.fromisoformat(
        collection_start_local or config.POLYMARKET_PROSPECTIVE_COLLECTION_START_LOCAL
    )
    rs_end = date_cls.fromisoformat(config.POLYMARKET_2026_REGULAR_SEASON_END_LOCAL)
    ps_start = date_cls.fromisoformat(config.POLYMARKET_2026_POSTSEASON_START_LOCAL)
    ws_end = date_cls.fromisoformat(config.POLYMARKET_2026_WORLD_SERIES_END_LOCAL)
    first = max(as_of, start)
    remaining_rs = max(0, (rs_end - first).days + 1) if first <= rs_end else 0
    remaining_ps = max(0, (ws_end - max(first, ps_start)).days + 1) if first <= ws_end else 0
    floor = int(config.POLYMARKET_MIN_ELIGIBLE_DATES)
    power_days = sample_size_for_paired_mean(
        effect=config.POLYMARKET_INTENDED_LOGLOSS_IMPROVEMENT
    )["approx_independent_days"]
    return {
        "as_of_local": str(as_of),
        "collection_start_local": str(start),
        "regular_season_end_local": str(rs_end),
        "postseason_window_local": [str(ps_start), str(ws_end)],
        "max_remaining_regular_season_calendar_dates": remaining_rs,
        "max_remaining_postseason_calendar_dates": remaining_ps,
        "structural_floor_dates": floor,
        "approx_independent_days_for_target_edge": power_days,
        "remaining_regular_season_meets_structural_floor": remaining_rs >= floor,
        "remaining_regular_season_meets_power_plan": remaining_rs >= power_days,
        "conclusion": (
            "insufficient_for_structural_floor_this_regular_season"
            if remaining_rs < floor
            else "structural_floor_calendar_span_possible"
        ),
        "note": (
            "Calendar dates are an upper bound, not observed eligible days. "
            "Postseason is a separate cohort and does not fill the regular-season floor."
        ),
    }


def count_prospective_labeled_dates(
    decisions_path: str | None = None,
    *,
    exploratory_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Count distinct decision days in the paper ledger excluding exploratory dates.

    Labeled nested evaluation still requires joined official winners; this only
    measures collection progress of pre-outcome decision logs.
    """
    path = decisions_path or config.POLYMARKET_PAPER_DECISIONS_PATH
    exploratory = set(exploratory_dates or config.POLYMARKET_EXPLORATORY_DATES)
    if not os.path.exists(path):
        return {
            "n_decision_rows": 0,
            "n_distinct_decision_days": 0,
            "n_prospective_days": 0,
            "prospective_days": [],
            "n_buy": 0,
            "n_pass": 0,
            "reason": "decisions_missing",
        }
    frame = pd.read_csv(path)
    if frame.empty or "decision_time_utc" not in frame.columns:
        return {
            "n_decision_rows": int(len(frame)),
            "n_distinct_decision_days": 0,
            "n_prospective_days": 0,
            "prospective_days": [],
            "n_buy": 0,
            "n_pass": 0,
        }
    days = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.copy()
    frame["_day"] = days
    frame = frame[frame["_day"].notna()]
    all_days = sorted(set(frame["_day"].tolist()))
    prospective = [d for d in all_days if d not in exploratory]
    return {
        "n_decision_rows": int(len(frame)),
        "n_distinct_decision_days": len(all_days),
        "n_prospective_days": len(prospective),
        "prospective_days": prospective,
        "n_buy": int((frame["action"] == "buy").sum()) if "action" in frame.columns else 0,
        "n_pass": int((frame["action"] == "pass").sum()) if "action" in frame.columns else 0,
    }


def write_collection_status(status: dict[str, Any], path: str | None = None) -> str:
    path = path or config.POLYMARKET_COLLECTION_STATUS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def load_protocol(path: str | None = None) -> dict[str, Any]:
    path = path or config.POLYMARKET_PAPER_PROTOCOL_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def frozen_policy_document(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or register_protocol()
    doc = {
        "policy_version": f"{protocol['protocol_id']}_frozen_v1",
        "evaluation_version": protocol.get("protocol_version"),
        "registered_at_utc": protocol.get("registered_at_utc"),
        "protocol_hash": protocol.get("protocol_hash"),
        "venue_id": protocol.get("venue_id"),
        "action": "paper_only",
        "betting_mode": "disabled",
        "entry": {
            "require_fresh_eligible_book": True,
            "entry_minutes_before_scheduled_start": config.POLYMARKET_ENTRY_MINUTES_BEFORE_START,
            "size_contracts": config.POLYMARKET_PAPER_ONE_SHARE,
            "edge_vs_executable_ask_min": min(config.GAME_RESIDUAL_EDGE_THRESHOLD_GRID),
            "model": "regularized_market_residual_logistic_when_validated_else_pass",
            "exclude_market_only_fallback": True,
            "log_candidates_and_passes_before_outcomes": True,
        },
        "execution_stress_for_reporting": {
            "delay_seconds": config.POLYMARKET_CONSERVATIVE_DELAY_SECONDS,
            "adverse_ticks": config.POLYMARKET_CONSERVATIVE_ADVERSE_TICKS,
            "delay_grid": list(config.POLYMARKET_MANUAL_DELAY_SECONDS_GRID),
            "adverse_ticks_grid": list(config.POLYMARKET_ADVERSE_TICKS_GRID),
        },
        "cohorts": protocol.get("cohorts"),
        "prospective_checkpoints_game_days": list(
            config.POLYMARKET_PROSPECTIVE_CHECKPOINTS_GAME_DAYS
        ),
        "evaluation_rule": (
            "Log decisions before outcomes. Compare against same-time market mid "
            "on identical opportunities. Do not stop at the first profitable "
            "streak. Extend observation when inconclusive. Keep operational "
            "collector fixes separate from evaluation-version changes."
        ),
        "notes": [
            "Frozen for prospective paper observation.",
            "Does not enable live betting.",
            "Remaining 2026 regular season alone is below the structural date floor.",
        ],
    }
    doc["policy_hash"] = _sha(doc)
    return doc


def write_frozen_policy(path: str | None = None, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path or config.POLYMARKET_FROZEN_POLICY_PATH
    doc = frozen_policy_document(protocol=protocol)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("policy_hash") == doc.get("policy_hash"):
            return existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return doc

def calibration_table(
    y: pd.Series,
    p: pd.Series,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "p": pd.to_numeric(p, errors="coerce")})
    frame = frame.dropna()
    if frame.empty:
        return []
    frame["bin"] = pd.cut(frame["p"], bins=n_bins, labels=False, include_lowest=True)
    rows = []
    for b, part in frame.groupby("bin"):
        rows.append(
            {
                "bin": int(b) if pd.notna(b) else None,
                "n": int(len(part)),
                "mean_p": float(part["p"].mean()),
                "mean_y": float(part["y"].mean()),
            }
        )
    return rows


def compare_models_on_frame(
    frame: pd.DataFrame,
    *,
    label_col: str = grm.HOME_WON_LABEL,
    market_col: str = "market_mid_probability",
    heuristic_col: str = grm.HEURISTIC_COL,
    residual_col: str = grm.RESIDUAL_PROB_COL,
    probability_source_col: str = "probability_source",
) -> dict[str, Any]:
    """Score candidates on identical rows; drop market-only fallbacks from residual."""
    work = frame.copy()
    required = [label_col, market_col]
    for col in required:
        if col not in work.columns:
            return {"status": "insufficient_data", "reason": f"missing_column:{col}"}
    work = work.dropna(subset=[label_col, market_col])
    if work.empty:
        return {"status": "insufficient_data", "reason": "no_scored_rows", "n_games": 0}

    out: dict[str, Any] = {
        "status": "ok",
        "n_games": int(len(work)),
        "n_dates": int(work["date"].nunique()) if "date" in work.columns else None,
        "candidates": {},
        "calibration": {},
        "paired_vs_market": {},
    }
    market_metrics = grm.score_probabilities(work[label_col], work[market_col])
    out["candidates"]["market_mid_baseline"] = market_metrics
    out["calibration"]["market_mid_baseline"] = calibration_table(work[label_col], work[market_col])

    if heuristic_col in work.columns:
        h = work.dropna(subset=[heuristic_col])
        if not h.empty:
            out["candidates"]["calibrated_heuristic_baseball"] = grm.score_probabilities(
                h[label_col], h[heuristic_col]
            )
            out["paired_vs_market"]["calibrated_heuristic_baseball"] = grm.paired_diffs_vs_market(
                h[label_col], h[heuristic_col], h[market_col]
            )
            out["calibration"]["calibrated_heuristic_baseball"] = calibration_table(
                h[label_col], h[heuristic_col]
            )

    if residual_col in work.columns:
        r = work.dropna(subset=[residual_col])
        if probability_source_col in r.columns:
            r = r[r[probability_source_col] != grm.PROBABILITY_SOURCE_MARKET_ONLY_FALLBACK]
        if not r.empty:
            out["candidates"]["regularized_market_residual_logistic"] = grm.score_probabilities(
                r[label_col], r[residual_col]
            )
            out["paired_vs_market"]["regularized_market_residual_logistic"] = grm.paired_diffs_vs_market(
                r[label_col], r[residual_col], r[market_col]
            )
            out["calibration"]["regularized_market_residual_logistic"] = calibration_table(
                r[label_col], r[residual_col]
            )
            # Date-block bootstrap on log loss difference (model - market); negative is better.
            boot = model_validation.paired_block_bootstrap_difference(
                r.rename(columns={residual_col: "predicted_probability_a"}),
                r[[c for c in ("date", "game_pk", label_col, market_col) if c in r.columns]].rename(
                    columns={market_col: "predicted_probability_b"}
                ),
                keys=[c for c in ("date", "game_pk") if c in r.columns],
                block_key="date" if "date" in r.columns else "game_pk",
                value_a="predicted_probability_a",
                value_b="predicted_probability_b",
                label_col=label_col,
                metric="log_loss",
                n_bootstrap=min(500, config.MARKET_ODDS_BOOTSTRAP_SAMPLES),
                random_seed=config.MARKET_ODDS_BOOTSTRAP_SEED,
            )
            out["bootstrap_log_loss_model_minus_market"] = boot
    return out


def ledger_performance_report(
    decisions: pd.DataFrame,
    positions: pd.DataFrame,
) -> dict[str, Any]:
    merged = decisions.merge(positions, on="decision_id", how="left", suffixes=("", "_pos"))
    if "acquisition_cost" not in merged.columns and "acquisition_cost" in decisions.columns:
        pass
    settled = merged[merged.get("status") == "settled"] if "status" in merged.columns else merged.iloc[0:0]
    roi = paper_ledger.settled_roi(merged if "status" in merged.columns else positions)
    nets = []
    if not settled.empty and "net_pnl" in settled.columns:
        nets = [float(x) for x in pd.to_numeric(settled["net_pnl"], errors="coerce").fillna(0).tolist()]
    dd = paper_ledger.equity_drawdown(nets)
    by_date = {}
    if not settled.empty and "decision_time_utc" in settled.columns:
        settled = settled.copy()
        settled["_day"] = pd.to_datetime(settled["decision_time_utc"], utc=True, errors="coerce").dt.date
        for day, part in settled.groupby("_day"):
            by_date[str(day)] = float(pd.to_numeric(part["net_pnl"], errors="coerce").fillna(0).sum())
    return {
        "roi": roi,
        "drawdown": dd,
        "n_decisions": int(len(decisions)),
        "n_pass": int((decisions["action"] == "pass").sum()) if not decisions.empty else 0,
        "n_buy": int((decisions["action"] == "buy").sum()) if not decisions.empty else 0,
        "fill_rate": (
            float((decisions["filled_qty"] > 0).mean()) if not decisions.empty and "filled_qty" in decisions.columns else float("nan")
        ),
        "net_by_date": by_date,
        "concentration": {
            "n_settled_dates": len(by_date),
            "best_date_share": (
                float(max(by_date.values()) / sum(by_date.values()))
                if by_date and sum(by_date.values()) != 0
                else float("nan")
            ),
        },
    }


def apply_gates(
    *,
    protocol: dict[str, Any],
    probability_report: dict[str, Any],
    strategy_report: dict[str, Any] | None,
    n_eligible_dates: int,
    n_outer_folds_available: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    floor = int(protocol.get("fold_structure", {}).get("min_dates_structural_floor") or 70)
    min_folds = int(protocol.get("fold_structure", {}).get("min_outer_folds") or 3)
    if n_eligible_dates < floor:
        reasons.append(
            f"insufficient_dates:{n_eligible_dates}<structural_floor:{floor}"
        )
    if n_outer_folds_available < min_folds:
        reasons.append(
            f"insufficient_outer_folds:{n_outer_folds_available}<{min_folds}"
        )
    if probability_report.get("status") != "ok":
        reasons.append(f"probability_status:{probability_report.get('status')}")
    residual = (probability_report.get("candidates") or {}).get(
        "regularized_market_residual_logistic"
    )
    market = (probability_report.get("candidates") or {}).get("market_mid_baseline")
    if residual and market:
        if not (residual.get("log_loss", 1e9) < market.get("log_loss", 0)):
            reasons.append("residual_log_loss_not_better_than_market")
        boot = probability_report.get("bootstrap_log_loss_model_minus_market") or {}
        # model - market; improvement means CI high < 0
        if not (
            boot.get("n_blocks", 0) > 0
            and boot.get("ci_high") is not None
            and boot.get("ci_high") == boot.get("ci_high")
            and float(boot["ci_high"]) < 0
        ):
            reasons.append("paired_log_loss_ci_does_not_exclude_zero_improvement")
    else:
        reasons.append("missing_residual_or_market_scores")

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
        # Distinguish clear failure from insufficient data when we had enough dates
        # but metrics failed.
        if n_eligible_dates >= floor and n_outer_folds_available >= min_folds:
            if any(
                r.startswith("residual_log_loss") or r.startswith("paired_log_loss") or r.startswith("strategy_roi")
                for r in reasons
            ):
                verdict = "edge_not_supported"
        validation_status = "insufficient_data" if verdict == "insufficient_evidence" else "validated_failed"
    else:
        verdict = "edge_supported"
        validation_status = "validated_passed"
        if not strategy_ok:
            # Should not happen if reasons empty, but keep fail-closed.
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
    }


def write_evaluation_report(report: dict[str, Any], path: str | None = None) -> str:
    path = path or config.POLYMARKET_PAPER_EVALUATION_REPORT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cleaned = json.loads(json.dumps(report, default=str))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def count_possible_outer_folds(n_dates: int, protocol: dict[str, Any] | None = None) -> int:
    protocol = protocol or PaperResearchProtocol().to_dict()
    fs = protocol["fold_structure"]
    dates = list(range(int(n_dates)))
    nested, _freeze = model_validation.build_nested_folds(
        dates,
        outer_min_train_dates=int(fs["outer_min_train_dates"]),
        outer_test_block_dates=int(fs["outer_test_block_dates"]),
        inner_min_train_dates=int(fs["inner_min_train_dates"]),
        inner_test_block_dates=int(fs["inner_test_block_dates"]),
        freeze_dates=int(fs["freeze_dates"]),
        require_complete_test_blocks=bool(fs.get("require_complete_test_blocks", True)),
    )
    return len(nested)
