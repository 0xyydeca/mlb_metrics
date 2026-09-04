"""Betting readiness checker — fail-closed PASS/FAIL table.

Does not generate bets. Exits nonzero unless every requirement passes.
Under current defaults (GAME_PREDICTION_MODE=shadow, BETTING_MODE=disabled)
this script is expected to print READY FOR LIVE BETTING: NO.

Usage:
    python scripts/check_betting_readiness.py
    python scripts/check_betting_readiness.py --skip-pytest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, game_residual_model, market_odds, schedule_snapshots


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _load_json(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"_load_error": str(exc)}


def _fixture_contamination() -> Check:
    """Production snapshot files must not contain known test fixtures."""
    problems = []
    odds_path = os.path.join(REPO_ROOT, config.MARKET_ODDS_SNAPSHOTS_PATH)
    sched_path = os.path.join(REPO_ROOT, config.SCHEDULE_SNAPSHOTS_PATH)

    if os.path.exists(odds_path):
        odds = market_odds.normalize_snapshot_frame(pd.read_csv(odds_path))
        if not odds.empty:
            unmatched = (odds["source_status"] == market_odds.SOURCE_UNMATCHED).sum()
            missing_pk = odds["game_pk"].isna().sum()
            if unmatched or missing_pk:
                problems.append(
                    f"market_odds_snapshots: {len(odds)} rows "
                    f"(unmatched={unmatched}, missing_game_pk={missing_pk})"
                )
            # Historical postgame labeled morning
            bad_roles = market_odds.filter_valid_prediction_time_snapshots(odds)
            # contamination = any invalid prediction-time-looking rows that fail filter
            morningish = odds[odds["snapshot_role"].isin(list(market_odds.PREDICTION_TIME_ROLES))]
            if len(morningish) and len(bad_roles) < len(morningish):
                # some morning-tagged rows are invalid — already counted above if unmatched
                invalid_morning = len(morningish) - len(
                    market_odds.filter_valid_prediction_time_snapshots(
                        morningish, required_role=None,
                    )
                )
                if invalid_morning:
                    problems.append(
                        f"market_odds_snapshots: {invalid_morning} invalid prediction-time-tagged rows"
                    )

    if os.path.exists(sched_path):
        sched = pd.read_csv(sched_path)
        if not sched.empty:
            fixture = sched[
                (sched.get("game_pk") == 100)
                | (
                    (sched.get("home_team") == "NYY")
                    & (sched.get("away_team") == "BOS")
                    & (sched.get("home_probable_pitcher_key_mlbam") == 501)
                )
            ] if "game_pk" in sched.columns else sched.iloc[0:0]
            # safer explicit checks
            if "game_pk" in sched.columns and (sched["game_pk"] == 100).any():
                problems.append("schedule_snapshots: contains fixture game_pk=100")
            if (
                {"home_team", "away_team", "home_probable_pitcher_key_mlbam"}.issubset(sched.columns)
                and (
                    (sched["home_team"] == "NYY")
                    & (sched["away_team"] == "BOS")
                    & (sched["home_probable_pitcher_key_mlbam"] == 501)
                ).any()
            ):
                problems.append("schedule_snapshots: contains NYY/BOS pitcher 501/502 fixture")

    if problems:
        return Check("no_production_fixture_contamination", False, "; ".join(problems))
    return Check("no_production_fixture_contamination", True, "snapshot files empty or clean")


def _market_snapshot_checks() -> list[Check]:
    path = os.path.join(REPO_ROOT, config.MARKET_ODDS_SNAPSHOTS_PATH)
    checks = []
    if not os.path.exists(path):
        empty = market_odds.empty_snapshot_frame()
        checks.append(Check("valid_matched_prediction_time_market_snapshots", False, "file missing"))
        checks.append(Check("valid_closing_snapshots", False, "file missing"))
        checks.append(Check("no_post_start_snapshots_used_as_priors", True, "no file"))
        return checks

    snaps = market_odds.normalize_snapshot_frame(pd.read_csv(path))
    pred = market_odds.filter_valid_prediction_time_snapshots(snaps)
    checks.append(Check(
        "valid_matched_prediction_time_market_snapshots",
        len(pred) > 0,
        f"n_valid_prediction_time={len(pred)} / n_total={len(snaps)}",
    ))

    # Closing: at least one game with a valid pre-start closing selection.
    closing_ok = 0
    for gpk in snaps["game_pk"].dropna().unique():
        if market_odds.select_closing_snapshot(snaps, int(gpk)) is not None:
            closing_ok += 1
    checks.append(Check(
        "valid_closing_snapshots",
        closing_ok > 0,
        f"n_games_with_closing={closing_ok}",
    ))

    # Post-start rows must not appear in prediction-time filter.
    post = snaps[snaps["source_status"] == market_odds.SOURCE_POST_START] if not snaps.empty else snaps
    leaked = 0
    if not pred.empty and not post.empty:
        leaked = len(set(pred["snapshot_id"]) & set(post["snapshot_id"]))
    checks.append(Check(
        "no_post_start_snapshots_used_as_priors",
        leaked == 0,
        f"post_start_rows={len(post)} leaked_into_pred_filter={leaked}",
    ))
    return checks


def _report_and_gate_checks() -> list[Check]:
    residual_path = os.path.join(REPO_ROOT, config.GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH)
    betting_path = os.path.join(REPO_ROOT, config.BETTING_PROMOTION_GATE_REPORT_PATH)
    residual = _load_json(residual_path)
    betting = _load_json(betting_path) or residual

    checks = []
    checks.append(Check(
        "residual_nested_validation_report_exists",
        residual is not None and "_load_error" not in (residual or {}),
        residual_path if residual else "missing",
    ))

    pred_ok, pred_details = game_residual_model.evaluate_prediction_promotion_gate(residual)
    checks.append(Check(
        "probability_promotion_gate_passed",
        pred_ok,
        pred_details.get("reason", ""),
    ))

    bet_ok, bet_details = game_residual_model.evaluate_betting_promotion_gate(betting)
    checks.append(Check(
        "betting_promotion_gate_passed",
        bet_ok,
        bet_details.get("reason", ""),
    ))

    art = game_residual_model.load_residual_model(
        os.path.join(REPO_ROOT, config.GAME_RESIDUAL_MODEL_PATH)
    )
    art_exists = art is not None
    checks.append(Check(
        "residual_model_artifact_exists",
        art_exists,
        config.GAME_RESIDUAL_MODEL_PATH if art_exists else "missing",
    ))

    report_id = (betting or residual or {}).get("artifact_id") if isinstance(betting or residual, dict) else None
    # Also accept nested metadata
    if report_id is None and isinstance(residual, dict):
        report_id = (
            ((residual.get("promotion_gate") or {}).get("artifact_id"))
            or residual.get("artifact_id")
        )
    model_id = (art.metadata or {}).get("artifact_id") if art else None
    checks.append(Check(
        "residual_artifact_matches_report",
        bool(art_exists and model_id and report_id and model_id == report_id),
        f"model={model_id!r} report={report_id!r}",
    ))

    methods = (residual or {}).get("methods") or {}
    residual_m = methods.get(game_residual_model.METHOD_RESIDUAL_LOGISTIC) or {}
    n_games = int(
        residual_m.get("n_games")
        or (residual or {}).get("n_games_evaluated")
        or 0
    )
    n_folds = int((residual or {}).get("n_outer_folds") or 0)

    # Date-block counts live on nested bootstrap / ROI objects, not a top-level
    # residual_m["n_date_blocks"] field (which is often absent).
    hypo = residual_m.get("hypothetical_roi") or {}
    boot_brier = residual_m.get("paired_brier_bootstrap") or {}
    boot_ll = residual_m.get("paired_log_loss_bootstrap") or {}
    bet_gate = ((betting or residual or {}).get("betting_promotion_gate") or {})
    n_blocks = int(
        hypo.get("n_blocks")
        or boot_brier.get("n_blocks")
        or boot_ll.get("n_blocks")
        or bet_gate.get("n_blocks")
        or 0
    )

    checks.append(Check(
        "at_least_100_evaluated_games",
        n_games >= int(config.BETTING_PROMOTION_MIN_GAMES),
        f"n_games={n_games}",
    ))
    checks.append(Check(
        "at_least_3_untouched_outer_folds",
        n_folds >= int(config.BETTING_PROMOTION_MIN_OUTER_FOLDS),
        f"n_outer_folds={n_folds}",
    ))
    checks.append(Check(
        "at_least_10_independent_date_blocks",
        n_blocks >= int(config.BETTING_PROMOTION_MIN_DATE_BLOCKS),
        f"n_blocks={n_blocks} (from hypothetical_roi / paired bootstrap)",
    ))

    # Independent betting-volume policy floors (placeholders, not power calcs).
    n_bets = int(bet_gate.get("n_bets") or residual_m.get("n_bets") or 0)
    n_bet_dates = int(bet_gate.get("n_bet_dates") or 0)
    n_bet_weeks = int(bet_gate.get("n_bet_weeks") or 0)
    policy = bet_gate.get("policy_thresholds") or {
        "BETTING_PROMOTION_MIN_EVALUATED_GAMES": config.BETTING_PROMOTION_MIN_EVALUATED_GAMES,
        "BETTING_PROMOTION_MIN_BETS": config.BETTING_PROMOTION_MIN_BETS,
        "BETTING_PROMOTION_MIN_BET_DATES": config.BETTING_PROMOTION_MIN_BET_DATES,
        "BETTING_PROMOTION_MIN_BET_WEEKS": config.BETTING_PROMOTION_MIN_BET_WEEKS,
        "note": "policy thresholds, not statistically derived",
    }
    checks.append(Check(
        "policy_min_evaluated_games",
        n_games >= int(config.BETTING_PROMOTION_MIN_EVALUATED_GAMES),
        f"n_games={n_games} min={policy.get('BETTING_PROMOTION_MIN_EVALUATED_GAMES')} (policy)",
    ))
    checks.append(Check(
        "policy_min_bets",
        n_bets >= int(config.BETTING_PROMOTION_MIN_BETS),
        f"n_bets={n_bets} min={policy.get('BETTING_PROMOTION_MIN_BETS')} (policy)",
    ))
    checks.append(Check(
        "policy_min_bet_dates",
        n_bet_dates >= int(config.BETTING_PROMOTION_MIN_BET_DATES),
        f"n_bet_dates={n_bet_dates} min={policy.get('BETTING_PROMOTION_MIN_BET_DATES')} (policy)",
    ))
    checks.append(Check(
        "policy_min_bet_weeks",
        n_bet_weeks >= int(config.BETTING_PROMOTION_MIN_BET_WEEKS),
        f"n_bet_weeks={n_bet_weeks} min={policy.get('BETTING_PROMOTION_MIN_BET_WEEKS')} (policy)",
    ))

    brier_delta = residual_m.get("model_minus_market_brier")
    # Negative model_minus_market_brier means model better (lower Brier)
    # Confirm convention from residual tests: model_minus_market_brier
    improves_brier = (
        brier_delta is not None
        and brier_delta == brier_delta
        and float(brier_delta) < 0
    )
    # Some reports store paired improvement positively — also check gate checks.
    gate_checks = ((residual or {}).get("promotion_gate") or {}).get("checks") or {}
    if "positive_paired_brier_vs_market" in gate_checks:
        improves_brier = bool(gate_checks["positive_paired_brier_vs_market"])
    checks.append(Check(
        "residual_improves_paired_brier_vs_market",
        improves_brier,
        f"model_minus_market_brier={brier_delta}",
    ))

    ll_delta = residual_m.get("model_minus_market_log_loss")
    improves_ll = (
        ll_delta is not None
        and ll_delta == ll_delta
        and float(ll_delta) < 0
    )
    if "positive_paired_log_loss_vs_market" in gate_checks:
        improves_ll = bool(gate_checks["positive_paired_log_loss_vs_market"])
    checks.append(Check(
        "residual_improves_paired_log_loss_vs_market",
        improves_ll,
        f"model_minus_market_log_loss={ll_delta}",
    ))

    bet_gate = ((betting or residual or {}).get("betting_promotion_gate") or {})
    bet_checks = bet_gate.get("checks") or {}
    checks.append(Check(
        "positive_mean_closing_line_value",
        bool(bet_checks.get("positive_closing_line_value", False)),
        str(bet_checks.get("positive_closing_line_value")),
    ))
    checks.append(Check(
        "positive_hypothetical_roi",
        bool(bet_checks.get("positive_hypothetical_roi", False)),
        str(bet_checks.get("positive_hypothetical_roi")),
    ))
    checks.append(Check(
        "roi_bootstrap_lower_bound_above_threshold",
        bool(bet_checks.get("roi_ci_not_materially_negative", False)),
        str(bet_checks.get("roi_ci_not_materially_negative")),
    ))
    checks.append(Check(
        "performance_not_dominated_by_one_week",
        bool(bet_checks.get("no_single_week_dependence", False)),
        str(bet_checks.get("no_single_week_dependence")),
    ))

    checks.append(Check(
        "GAME_PREDICTION_MODE_is_live",
        config.GAME_PREDICTION_MODE == "live",
        f"GAME_PREDICTION_MODE={config.GAME_PREDICTION_MODE!r}",
    ))
    checks.append(Check(
        "BETTING_MODE_is_live",
        config.BETTING_MODE == "live",
        f"BETTING_MODE={config.BETTING_MODE!r}",
    ))

    # No fallback in mode resolvers under live config.
    g_mode, g_meta = game_residual_model.resolve_game_prediction_mode()
    b_mode, b_meta = game_residual_model.resolve_betting_mode()
    no_fallback = (
        g_mode == "live"
        and b_mode == "live"
        and not g_meta.get("fallback_used")
        and not b_meta.get("fallback_used")
    )
    checks.append(Check(
        "no_fallback_used",
        no_fallback,
        f"game_mode={g_mode}/{g_meta.get('fallback_reason')} "
        f"betting_mode={b_mode}/{b_meta.get('fallback_reason')}",
    ))
    return checks


def _run_pytest() -> Check:
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line"]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    return Check(
        "full_current_test_suite_passed",
        proc.returncode == 0,
        f"exit={proc.returncode} {tail[0][:120]}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip the full pytest suite (still reports FAIL for that check).",
    )
    args = parser.parse_args()

    checks: list[Check] = []
    if args.skip_pytest:
        checks.append(Check(
            "full_current_test_suite_passed",
            False,
            "skipped (--skip-pytest); treat as FAIL for readiness",
        ))
    else:
        print("Running full pytest suite (this may take a few minutes)...")
        checks.append(_run_pytest())

    checks.append(_fixture_contamination())
    checks.extend(_market_snapshot_checks())
    checks.extend(_report_and_gate_checks())

    print()
    print(f"{'CHECK':<55} {'RESULT':<6} DETAIL")
    print("-" * 100)
    for c in checks:
        result = "PASS" if c.passed else "FAIL"
        print(f"{c.name:<55} {result:<6} {c.detail}")

    all_pass = all(c.passed for c in checks)
    print()
    if all_pass:
        print("READY FOR LIVE BETTING: YES")
        return 0
    print("READY FOR LIVE BETTING: NO")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
