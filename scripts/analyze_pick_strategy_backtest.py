"""Score scripts/backtest_pick_strategies.py's output: for each candidate
strategy, the PRIMARY comparison metric is per-pick hit rate (with a real
Wilson confidence interval and a binomial significance test against a
coin flip) among the top DAILY_PICK_MAX picks/day - unconditional, no
extra probability floor. Real Beat the Streak longest/current streak
(also computed unconditionally) is reported too, but as a SECONDARY/
illustrative stat only - a single extreme-value number that a lucky or
unlucky stretch can swing a lot, unsuitable as the primary comparison
target for a bias-resistant test (see this project's own config.py
precedent of preferring Wilson CIs over point-in-time streak records
wherever real statistical significance is at stake).

NOT gated on DAILY_PICK_MIN_PROBABILITY (0.77), unlike the live
dashboard's "recommended" grade: confirmed empirically against this
backtest's own output that the row-wise mean of predicted_probability/
probability/Matchup_Hit_Probability never once reaches 0.77 across all
2160 logged picks (max observed: 0.75) - that bar was calibrated
alongside the ML shortlist (Model_Hit_Probability), which is deliberately
excluded from every candidate here for leak-safety (see
backtest_pick_strategies.py's module docstring). Applying an
ML-calibrated bar to a no-ML backtest would silently zero out every
candidate's "recommended" pool rather than measure anything real.

The full backtested window is also split into first-half/second-half so
an apparent winner can be checked for holding up in BOTH halves
independently - not just riding one hot stretch of the season, which
would be a real form of confirmation bias even with a pre-registered
candidate set.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, evaluation

STRATEGIES = {
    "backtest-c0-legacy-approach": "c0: legacy Approach (no PAVE/matchup)",
    "backtest-c1-wave-vs-pave": "c1: WAVE vs PAVE (user's manual method)",
    "backtest-c2-current-heuristic": "c2: current live heuristic (no ML)",
}


def _score_slice(df: pd.DataFrame, model_version: str) -> dict:
    subset = df[df["model_version"] == model_version]
    top_picks = subset[subset["rank"] <= config.DAILY_PICK_MAX]
    resolved = evaluation.resolved_only(top_picks)
    n = len(resolved)
    successes = int(resolved["actual_hit"].sum()) if n else 0
    hit_rate = successes / n if n else float("nan")
    lo, hi = evaluation.wilson_confidence_interval(successes, n)
    p_value = evaluation.binomial_significance(successes, n, null_probability=0.5)

    day_success_rate = evaluation.top_k_hit_rate(subset, k=config.DAILY_PICK_MAX, require_all=True)
    brier = evaluation.brier_score(subset)

    # Unconditional (min_probability=0.0, the function's own default) -
    # every day's top DAILY_PICK_MAX picks count toward the streak, since
    # DAILY_PICK_MIN_PROBABILITY isn't a meaningful bar without the ML
    # shortlist (see module docstring).
    longest = evaluation.longest_streak(
        df, metric="Game_Hit_Probability", max_picks=config.DAILY_PICK_MAX, model_version=model_version,
    )
    current = evaluation.current_streak(
        df, metric="Game_Hit_Probability", max_picks=config.DAILY_PICK_MAX, model_version=model_version,
    )

    return {
        "n_top_picks": n,
        "hit_rate": hit_rate,
        "wilson_ci_low": lo,
        "wilson_ci_high": hi,
        "p_value_vs_coin_flip": p_value,
        "day_success_rate (all top-2 hit)": day_success_rate,
        "brier_score": brier,
        "longest_streak": longest,
        "current_streak_as_of_end": current,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/backtest/pick_strategy_backtest.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    dates = sorted(df["date"].unique())
    midpoint = dates[len(dates) // 2]
    first_half = df[df["date"] < midpoint]
    second_half = df[df["date"] >= midpoint]

    print(f"Full window: {dates[0].date()} to {dates[-1].date()} ({len(dates)} dates)")
    print(f"First half:  {dates[0].date()} to {(midpoint - pd.Timedelta(days=1)).date()}")
    print(f"Second half: {midpoint.date()} to {dates[-1].date()}")
    print()

    rows = []
    for model_version, label in STRATEGIES.items():
        full = _score_slice(df, model_version)
        first = _score_slice(first_half, model_version)
        second = _score_slice(second_half, model_version)
        rows.append({"strategy": label, "window": "full_season", **full})
        rows.append({"strategy": label, "window": "first_half", **first})
        rows.append({"strategy": label, "window": "second_half", **second})

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))

    output_path = os.path.join(os.path.dirname(args.input), "pick_strategy_backtest_summary.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
