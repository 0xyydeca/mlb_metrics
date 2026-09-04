"""Print candidate-pool coverage and label distributions for
data/predictions/hitter_opportunity_log.csv.

Does not train a model.

Usage:
    python scripts/report_hitter_opportunity_log.py
    python scripts/report_hitter_opportunity_log.py --log path/to/log.csv --coverage path/to/coverage.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import hitter_training_data


def _default_coverage_path(log_path: str) -> str:
    root, ext = os.path.splitext(log_path)
    return f"{root}_coverage{ext or '.csv'}"


def _fmt_rate(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def format_report(summary: dict) -> str:
    lines = [
        "Hitter opportunity log (data asset - not a live model)",
        f"  rows: {summary.get('n_rows', 0)}",
        f"  games: {summary.get('n_games', 0)}",
        f"  dates: {summary.get('n_dates', 0)}",
        "",
        "Candidate-pool coverage (vs actual game participants; actual lineup is labels only)",
        f"  starter coverage: {_fmt_rate(summary.get('starter_coverage'))}",
        f"  appearance coverage: {_fmt_rate(summary.get('appearance_coverage'))}",
        f"  omitted appearing hitters: {summary.get('n_omitted_appeared', 0)}",
        f"  omitted with no pregame history (rookies/call-ups): {summary.get('n_omitted_no_pregame_history', 0)}",
        "",
        "Label distributions (candidate rows, including DNPs)",
        f"  Started: {summary.get('Started_count', 0)} ({_fmt_rate(summary.get('Started_rate'))})",
        f"  Appeared: {summary.get('Appeared_count', 0)} ({_fmt_rate(summary.get('Appeared_rate'))})",
        f"  No_Game / void (did not appear): {summary.get('No_Game_count', 0)} ({_fmt_rate(summary.get('No_Game_rate'))})",
        f"  Got_Hit (all candidates): {summary.get('Got_Hit_count', 0)} ({_fmt_rate(summary.get('Got_Hit_rate'))})",
        f"  Got_Hit among appeared: {_fmt_rate(summary.get('Got_Hit_rate_among_appeared'))}",
    ]
    sources = summary.get("candidate_sources") or {}
    if sources:
        lines.append("  candidate_source: " + ", ".join(f"{k}={v}" for k, v in sources.items()))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="data/predictions/hitter_opportunity_log.csv")
    parser.add_argument("--coverage", default=None)
    args = parser.parse_args()
    coverage_path = args.coverage or _default_coverage_path(args.log)

    if not os.path.exists(args.log):
        print(f"No opportunity log at {args.log} - run scripts/build_hitter_opportunity_log.py first.")
        return

    rows = pd.read_csv(args.log, parse_dates=["date"])
    coverage = (
        pd.read_csv(coverage_path, parse_dates=["date"])
        if os.path.exists(coverage_path)
        else pd.DataFrame()
    )
    print(format_report(hitter_training_data.summarize_opportunity_log(rows, coverage)), end="")


if __name__ == "__main__":
    main()
