"""Scores backtest_pick_strategies_v2.py's output the same way
analyze_pick_strategy_backtest.py scores round 1 (unconditional top-
DAILY_PICK_MAX hit rate + Wilson CI, full season and first/second-half
split), plus pairwise Fisher exact tests against round 1's c1 and c2 so
the ablation/sweep candidates can be read against the two reference
points that matter: does this candidate close the gap to c1, or does it
stay down near c2?
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from scipy.stats import fisher_exact

from mlb_metrics import config, evaluation

STRATEGIES = {
    "backtest-c2a-no-lineup": "c2a: no lineup qualifier (PA+recency+teams)",
    "backtest-c2b-no-recency": "c2b: no recency qualifier (PA+lineup+teams)",
    "backtest-c1-rank-matchup-only": "c1-alt: rank by Matchup_Hit_Probability only",
    "backtest-c1-rank-combined-mean": "c1-alt: rank by combined_probability (additive)",
    "backtest-c1-pa15": "c1-alt: PA>=15 (looser volume bar)",
    "backtest-c1-pa50": "c1-alt: PA>=50 (stricter volume bar)",
}


def _score_slice(df: pd.DataFrame, model_version: str) -> dict:
    subset = df[df["model_version"] == model_version]
    top_picks = subset[subset["rank"] <= config.DAILY_PICK_MAX]
    resolved = evaluation.resolved_only(top_picks)
    n = len(resolved)
    successes = int(resolved["actual_hit"].sum()) if n else 0
    hit_rate = successes / n if n else float("nan")
    lo, hi = evaluation.wilson_confidence_interval(successes, n)

    day_success_rate = evaluation.top_k_hit_rate(subset, k=config.DAILY_PICK_MAX, require_all=True)
    longest = evaluation.longest_streak(df, metric="Game_Hit_Probability", max_picks=config.DAILY_PICK_MAX, model_version=model_version)

    return {"n": n, "successes": successes, "hit_rate": hit_rate, "wilson_ci_low": lo, "wilson_ci_high": hi,
            "day_success_rate": day_success_rate, "longest_streak": longest}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/backtest/pick_strategy_backtest_v2.csv")
    parser.add_argument("--reference", default="data/backtest/pick_strategy_backtest.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    ref = pd.read_csv(args.reference, parse_dates=["date"])
    dates = sorted(df["date"].unique())
    midpoint = dates[len(dates) // 2]
    first_half = df[df["date"] < midpoint]
    second_half = df[df["date"] >= midpoint]

    ref_c1 = _score_slice(ref, "backtest-c1-wave-vs-pave")
    ref_c2 = _score_slice(ref, "backtest-c2-current-heuristic")
    print(f"Reference (round 1): c1 hit_rate={ref_c1['hit_rate']:.3f} (n={ref_c1['n']}), "
          f"c2 hit_rate={ref_c2['hit_rate']:.3f} (n={ref_c2['n']})")
    print()

    rows = []
    for model_version, label in STRATEGIES.items():
        full = _score_slice(df, model_version)
        first = _score_slice(first_half, model_version)
        second = _score_slice(second_half, model_version)

        p_vs_c1 = fisher_exact([
            [full["successes"], full["n"] - full["successes"]],
            [ref_c1["successes"], ref_c1["n"] - ref_c1["successes"]],
        ])[1] if full["n"] and ref_c1["n"] else float("nan")
        p_vs_c2 = fisher_exact([
            [full["successes"], full["n"] - full["successes"]],
            [ref_c2["successes"], ref_c2["n"] - ref_c2["successes"]],
        ])[1] if full["n"] and ref_c2["n"] else float("nan")

        rows.append({
            "strategy": label, "window": "full_season", "n": full["n"], "hit_rate": full["hit_rate"],
            "ci_low": full["wilson_ci_low"], "ci_high": full["wilson_ci_high"],
            "day_success_rate": full["day_success_rate"], "longest_streak": full["longest_streak"],
            "p_vs_c1": p_vs_c1, "p_vs_c2": p_vs_c2,
        })
        rows.append({
            "strategy": label, "window": "first_half", "n": first["n"], "hit_rate": first["hit_rate"],
            "ci_low": first["wilson_ci_low"], "ci_high": first["wilson_ci_high"],
            "day_success_rate": first["day_success_rate"], "longest_streak": first["longest_streak"],
            "p_vs_c1": None, "p_vs_c2": None,
        })
        rows.append({
            "strategy": label, "window": "second_half", "n": second["n"], "hit_rate": second["hit_rate"],
            "ci_low": second["wilson_ci_low"], "ci_high": second["wilson_ci_high"],
            "day_success_rate": second["day_success_rate"], "longest_streak": second["longest_streak"],
            "p_vs_c1": None, "p_vs_c2": None,
        })

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))

    output_path = os.path.join(os.path.dirname(args.input), "pick_strategy_backtest_v2_summary.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
