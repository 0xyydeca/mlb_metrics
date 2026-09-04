"""Backfill historical Automated Game Picks from confidence.csv/pave.csv
git history plus already-persisted Statcast, and report how well the
model would have performed.

Default schedule mode is ``as_of_snapshot`` (production-equivalent probable
starters from append-only schedule snapshots). Pass
``--schedule-backtest-mode actual_starter`` for the labeled retrospective
diagnostic that uses Statcast actual starters. Modes are never merged.

Usage:
    python scripts/run_game_picks_backtest.py
    python scripts/run_game_picks_backtest.py --days 60
    python scripts/run_game_picks_backtest.py --schedule-backtest-mode actual_starter
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, game_evaluation, game_picks_backtest, game_predictions, pipeline


def main():
    parser = argparse.ArgumentParser(description="Backfill and score historical Automated Game Picks.")
    parser.add_argument("--predictions-log", default="data/predictions/game_predictions.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--docs-data-dir", default="docs/data")
    parser.add_argument("--repo-dir", default=".", help="Git checkout to replay confidence.csv/pave.csv history from.")
    parser.add_argument("--days", type=int, default=40, help="How many of the most recent daily commits to replay.")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument(
        "--schedule-backtest-mode",
        default=config.SCHEDULE_BACKTEST_MODE_DEFAULT,
        choices=list(config.SCHEDULE_BACKTEST_MODES),
        help="as_of_snapshot (default) vs actual_starter diagnostic.",
    )
    args = parser.parse_args()

    historical_picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=args.repo_dir, raw_dir=args.raw_dir, season=args.season, days=args.days,
        schedule_backtest_mode=args.schedule_backtest_mode,
    )
    print(
        f"Reconstructed {len(historical_picks)} historical game picks from the last "
        f"{args.days} day(s) (schedule_mode={args.schedule_backtest_mode})."
    )
    game_predictions.append_game_predictions(historical_picks, args.predictions_log)

    pipeline.write_game_picks_export(args.predictions_log, args.docs_data_dir)

    log = (
        pd.read_csv(args.predictions_log, parse_dates=["date"])
        if os.path.exists(args.predictions_log)
        else pd.DataFrame()
    )
    if log.empty:
        print("No game picks logged yet.")
        return

    picks, summary = game_evaluation.build_game_picks_export(log)
    print(f"\n{summary.loc[0, 'n_games_resolved']}/{len(log)} logged game picks resolved.")
    print(summary.to_string(index=False))
    print(f"\nWrote docs/data/game_picks_picks.csv and game_picks_summary.csv to {args.docs_data_dir}")


if __name__ == "__main__":
    main()
