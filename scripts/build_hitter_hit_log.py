"""Build/append data/predictions/hitter_hit_log.csv: one row per hitter per
game_pk, every hitter with a game that date (not just the handful that ever
became an official Beat the Streak pick), carrying every
dfs_ml.HITTER_FEATURE_COLUMNS feature - starter_PAVE, Bullpen_PAVE, WAVE,
Game_Hit_Probability, etc. - computed strictly before that date, plus
Total_PA and a binary Got_Hit label for whether they actually got a hit
in THAT game.

Natural uniqueness: (date, game_pk, key_mlbam). Legacy CSV rows may lack
game_pk; migration fills NA and keeps the old (date, key_mlbam) dedupe
readable until those rows are recomputed.

This is a data asset for a future logistic regression on real hit
outcomes (see README) - it feeds nothing live today.

Run with no --days for a one-time full historical backfill; daily_update.yml
runs it with --days 10 so the daily incremental pass only recomputes recent
dates (each of which is idempotent against the existing file via the
dedupe below) instead of replaying the full season every day.

Usage:
    python scripts/build_hitter_hit_log.py
    python scripts/build_hitter_hit_log.py --days 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, dfs_backtest

HIT_LOG_KEY_COLUMNS = ["date", "game_pk", "key_mlbam"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--output", default="data/predictions/hitter_hit_log.csv")
    args = parser.parse_args()

    new_rows = dfs_backtest.assemble_hitter_hit_log(args.raw_dir, args.season, args.days)

    if new_rows.empty and not os.path.exists(args.output):
        print("No persisted Statcast history yet and no existing hitter_hit_log.csv - nothing to write.")
        return

    if os.path.exists(args.output):
        existing = pd.read_csv(args.output, parse_dates=["date"])
        if "game_pk" not in existing.columns:
            existing["game_pk"] = pd.NA
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    if "game_pk" not in combined.columns:
        combined["game_pk"] = pd.NA

    combined = combined.drop_duplicates(subset=HIT_LOG_KEY_COLUMNS, keep="last")
    combined = combined.sort_values(["date", "game_pk", "key_mlbam"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(combined)} rows, {len(new_rows)} recomputed this run).")


if __name__ == "__main__":
    main()
