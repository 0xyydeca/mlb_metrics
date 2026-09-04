"""Backtest sit / one / two streak policy vs legacy fixed policies.

Reads a hitter opportunity log (or shadow predictions) with
Final_Hit_Probability / P_Appear, walks forward date-by-date, and compares:

- dp (streak_policy.choose_action)
- legacy_two (always top-2 by p_hit)
- legacy_threshold (sit unless p_hit clears DAILY_PICK_MIN_PROBABILITY)

Writes headline metrics + optional paired bootstrap JSON under
reports/model_validation/. Does not change live picks.

Usage:
    python scripts/backtest_streak_policy.py
    python scripts/backtest_streak_policy.py --opportunity-log data/predictions/hitter_opportunity_log.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, model_validation, streak_policy as sp


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if obj != obj else obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def _load_daily_slates(path: str, max_per_day: int = 20) -> list[tuple]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Build the opportunity log or pass --opportunity-log."
        )
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input must include a date column")
    df["date"] = pd.to_datetime(df["date"])
    # Prefer Final_Hit_Probability; fall back to Got_Hit labels only for
    # diagnostics — backtest sampling uses model probabilities when present.
    if "Final_Hit_Probability" not in df.columns:
        if "Game_Hit_Probability" in df.columns:
            df["Final_Hit_Probability"] = df["Game_Hit_Probability"]
        elif "predicted_probability" in df.columns:
            df["Final_Hit_Probability"] = df["predicted_probability"]
        else:
            raise ValueError("Need Final_Hit_Probability (or Game_Hit_Probability)")

    slates = []
    for date, day in df.groupby("date", sort=True):
        # Drop DNPs from the decision slate when Appeared is known-zero and
        # Final_Hit is null; keep probabilistic candidates.
        day = day.copy()
        if "No_Game" in day.columns:
            day = day[day["No_Game"].astype(float) != 1.0]
        cands = sp.candidates_from_frame(day)
        cands = sorted(cands, key=lambda c: c.resolved_outcomes().p_hit, reverse=True)[:max_per_day]
        if cands:
            slates.append((date, cands))
    return slates


def _result_summary(result: sp.BacktestResult) -> dict:
    return {
        "policy": result.policy_name,
        "utility": result.utility_name,
        "action_counts": result.action_counts,
        "coverage": result.coverage,
        "day_survival_rate": result.day_survival_rate,
        "reset_rate": result.reset_rate,
        "average_hits_added": result.average_hits_added,
        "final_streak": result.final_streak,
        "longest_streak": result.longest_streak,
        "probability_reaching_target": result.reach_target_rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--opportunity-log",
        default="data/predictions/hitter_opportunity_shadow_predictions.csv",
        help="CSV with date + Final_Hit_Probability (+ optional P_Appear)",
    )
    parser.add_argument("--utility", default=config.STREAK_POLICY_UTILITY, choices=list(sp.UTILITIES))
    parser.add_argument("--target", type=int, default=config.STREAK_POLICY_TARGET)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=config.NESTED_VALIDATION_RANDOM_SEED)
    parser.add_argument("--report-dir", default=config.NESTED_VALIDATION_REPORT_DIR)
    args = parser.parse_args()

    print("=== Streak policy backtest (sit / one / two) ===")
    print(f"Live mode: {config.STREAK_POLICY_MODE} (this script does not alter picks)")
    slates = _load_daily_slates(args.opportunity_log)
    print(f"Loaded {len(slates)} dated slates from {args.opportunity_log}")
    if not slates:
        print("No slates — exiting.")
        return

    solved = sp.solve_streak_policy_values(
        [list(s) for _, s in slates],
        horizon=min(40, len(slates)),
        utility_name=args.utility,
        target=args.target,
    )

    summaries = {}
    for policy in ("dp", "legacy_two", "legacy_threshold"):
        result = sp.run_policy_backtest(
            slates,
            policy=policy,
            utility_name=args.utility,
            target=args.target,
            solved=solved if policy == "dp" else None,
            n_bootstrap=min(100, args.bootstrap),
            seed=args.seed,
        )
        summaries[policy] = _result_summary(result)
        print(f"\n[{policy}]")
        for k, v in summaries[policy].items():
            if k in ("policy", "utility"):
                continue
            print(f"  {k}: {v}")

    bootstrap = sp.paired_bootstrap_policy_comparison(
        slates,
        utility_name=args.utility,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        metric="final_streak",
    )
    print("\nPaired bootstrap (dp - legacy_two) final_streak:")
    print(f"  mean_diff={bootstrap['mean_diff']:.4f} "
          f"CI=({bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f})")

    model_validation.ensure_reports_dir_readme(args.report_dir)
    report = {
        "status": "ok",
        "n_days": len(slates),
        "utility": args.utility,
        "target": args.target,
        "policies": summaries,
        "paired_bootstrap": bootstrap,
        "promotion_gate": {
            "untouched_outer_fold_improvement": False,
            "note": (
                "Set untouched_outer_fold_improvement=true only after nested "
                "outer-fold evidence beats legacy; live STREAK_POLICY_MODE "
                "stays shadow until then."
            ),
        },
        "shadow": True,
        "live_wiring": "none unless STREAK_POLICY_MODE=live and gate passes",
    }
    path = os.path.join(args.report_dir, config.STREAK_POLICY_REPORT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, sort_keys=True)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
