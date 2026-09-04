"""Backtest the CURRENTLY LIVE hitter-pick and game-pick models and print an
accuracy/Brier-score report, without writing to data/predictions/ or
docs/data/.

Why this exists: predictions.csv/game_predictions.csv are strictly
append-only (a day's already-logged picks never get rewritten, only their
outcome fields fill in later - see those modules' docstrings), so a change
to the live selection logic can't be evaluated by looking at picks already
logged before the change shipped, and won't show up in NEW live picks until
tomorrow's run. Backtesting lets you evaluate a change immediately, by
replaying historical data through today's live code:

- Hitter picks: git_backtest.reconstruct_historical_picks replays every
  historical commit of docs/data/wave.csv through today's select_picks() -
  only the input data snapshot is historical, the selection logic is
  today's code, so this already reflects the live model.
- Game picks: game_picks_backtest.reconstruct_historical_game_picks_from_
  persisted goes a step further and recomputes confidence.csv/pave.csv
  fresh from persisted Statcast per replayed date too (not old git-committed
  snapshots), which matters because a signal like Power_A_PLUS may not
  exist in old commits at all.

Both reconstructions are resolved against real outcomes in-memory/in a
scratch temp file - nothing is written to the real predictions logs.

Usage:
    python scripts/evaluate_current_model.py
    python scripts/evaluate_current_model.py --hitter-days 40 --game-days 15
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, evaluation, game_evaluation, game_picks_backtest, git_backtest, predictions


def evaluate_hitter_picks(repo_dir: str, raw_dir: str, days: int | None) -> pd.DataFrame:
    picks = git_backtest.reconstruct_historical_picks(repo_dir=repo_dir, model_version=config.HITTER_MODEL_VERSION)
    if days:
        recent_dates = sorted(picks["date"].unique())[-days:]
        picks = picks[picks["date"].isin(recent_dates)]
    if picks.empty:
        return pd.DataFrame()

    persisted = data.load_persisted_statcast(raw_dir, config.SEASON_START.year)
    if persisted is None:
        print("  WARNING: no persisted Statcast data in --raw-dir; hitter picks can't be resolved.")
        return pd.DataFrame()
    completed = data.completed_events(persisted, ["game_date", "batter", "events"])

    with tempfile.TemporaryDirectory() as scratch_dir:
        log_path = os.path.join(scratch_dir, "predictions.csv")
        predictions.append_predictions(picks, log_path)
        resolved = predictions.resolve_predictions(log_path, completed)

    return evaluation.summarize(resolved, model_version=config.HITTER_MODEL_VERSION)


def evaluate_game_picks(raw_dir: str, days: int, schedule_backtest_mode: str | None = None) -> pd.DataFrame:
    picks = game_picks_backtest.reconstruct_historical_game_picks_from_persisted(
        raw_dir=raw_dir,
        days=days,
        model_version=config.GAME_PICK_MODEL_VERSION,
        schedule_backtest_mode=schedule_backtest_mode,
    )
    if picks.empty:
        return pd.DataFrame()

    _, summary = game_evaluation.build_game_picks_export(picks, model_version=config.GAME_PICK_MODEL_VERSION)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", default=".", help="Git checkout to replay docs/data/wave.csv history from.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--hitter-days", type=int, default=None,
        help="Limit the hitter-pick replay to the most recent N calendar dates (default: full history).",
    )
    parser.add_argument(
        "--game-days", type=int, default=20,
        help="Number of most recent game dates to replay for game picks. Recomputes the full pipeline "
        "per date, so keep this modest for interactive use.",
    )
    parser.add_argument(
        "--schedule-backtest-mode",
        default=config.SCHEDULE_BACKTEST_MODE_DEFAULT,
        choices=list(config.SCHEDULE_BACKTEST_MODES),
        help="as_of_snapshot (default) vs actual_starter diagnostic for game picks.",
    )
    args = parser.parse_args()

    print(f"Evaluating current hitter-pick model (model_version={config.HITTER_MODEL_VERSION})...")
    hitter_summary = evaluate_hitter_picks(args.repo_dir, args.raw_dir, args.hitter_days)
    if hitter_summary.empty:
        print("  No resolved hitter picks to evaluate.")
    else:
        print(hitter_summary.to_string(index=False))

    print(
        f"\nEvaluating current game-pick model (model_version={config.GAME_PICK_MODEL_VERSION}, "
        f"schedule_mode={args.schedule_backtest_mode})..."
    )
    game_summary = evaluate_game_picks(
        args.raw_dir, args.game_days, schedule_backtest_mode=args.schedule_backtest_mode,
    )
    if game_summary.empty:
        print("  No resolved game picks to evaluate.")
    else:
        print(game_summary.to_string(index=False))


if __name__ == "__main__":
    main()
