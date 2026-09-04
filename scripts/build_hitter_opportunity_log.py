"""Build/append data/predictions/hitter_opportunity_log.csv: one row per
pregame candidate hitter per game_pk, including players who did not
appear (negative examples for an appearance model).

Candidates are constructed from information strictly before each game
(see mlb_metrics.hitter_training_data). The actual lineup is labels
only. This is a data asset - it trains nothing and feeds nothing live.

Also writes a sibling coverage CSV (starter / appearance coverage and
omitted rookies/call-ups with no pregame history). Re-running is
idempotent: append + dedupe on (date, game_pk, key_mlbam), keep-last,
with an assertion that duplicates do not exist after the write.

Usage:
    python scripts/build_hitter_opportunity_log.py
    python scripts/build_hitter_opportunity_log.py --days 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, hitter_training_data

OPPORTUNITY_KEY_COLUMNS = hitter_training_data.OPPORTUNITY_KEY_COLUMNS


def _coverage_path(output: str) -> str:
    root, ext = os.path.splitext(output)
    return f"{root}_coverage{ext or '.csv'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--output", default="data/predictions/hitter_opportunity_log.csv")
    parser.add_argument(
        "--coverage-output",
        default=None,
        help="Per-team-game coverage CSV (default: <output stem>_coverage.csv)",
    )
    args = parser.parse_args()
    coverage_output = args.coverage_output or _coverage_path(args.output)

    new_rows, new_coverage = hitter_training_data.assemble_hitter_opportunity_dataset(
        args.raw_dir, args.season, args.days,
    )

    if new_rows.empty and not os.path.exists(args.output):
        print("No persisted Statcast history yet and no existing hitter_opportunity_log.csv - nothing to write.")
        return

    if os.path.exists(args.output):
        existing = pd.read_csv(args.output, parse_dates=["date"])
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=OPPORTUNITY_KEY_COLUMNS, keep="last")
    combined = combined.sort_values(OPPORTUNITY_KEY_COLUMNS).reset_index(drop=True)
    hitter_training_data.assert_unique_opportunity_keys(combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(combined)} rows, {len(new_rows)} recomputed this run).")

    if new_coverage.empty and not os.path.exists(coverage_output):
        return

    if os.path.exists(coverage_output) and not new_coverage.empty:
        existing_cov = pd.read_csv(coverage_output, parse_dates=["date"])
        combined_cov = pd.concat([existing_cov, new_coverage], ignore_index=True)
    elif os.path.exists(coverage_output):
        combined_cov = pd.read_csv(coverage_output, parse_dates=["date"])
    else:
        combined_cov = new_coverage

    if not combined_cov.empty:
        combined_cov = combined_cov.drop_duplicates(subset=["date", "game_pk", "team"], keep="last")
        combined_cov = combined_cov.sort_values(["date", "game_pk", "team"]).reset_index(drop=True)
        combined_cov.to_csv(coverage_output, index=False)
        print(f"Wrote {coverage_output} ({len(combined_cov)} team-games).")


if __name__ == "__main__":
    main()
