"""Train / validate the market-residual game-win challenger.

Assembles the game-pick training log, attaches prediction-time market
snapshots (never closing as features), runs nested rolling-origin
validation comparing:

  - market alone
  - current heuristic
  - heuristic + inner-fold calibration
  - regularized market-residual logistic
  - conservative nonlinear residual

Writes a machine-readable report under reports/model_validation/ and,
when the residual logistic beats market on nested outer folds under the
promotion criteria encoded in the report, saves a model artifact.

Does NOT enable live betting or change GAME_PREDICTION_MODE.

Usage:
    python scripts/train_game_residual_model.py
    python scripts/train_game_residual_model.py --season 2026 --days 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, game_picks_backtest, game_residual_model, market_odds


def _load_snapshots(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return market_odds.empty_snapshot_frame()
    return market_odds.normalize_snapshot_frame(pd.read_csv(path))


def main():
    parser = argparse.ArgumentParser(description="Train market-residual game-win model")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument(
        "--odds-snapshots",
        type=str,
        default=config.MARKET_ODDS_SNAPSHOTS_PATH,
        help="Append-only market odds snapshots (prediction-time + closing).",
    )
    parser.add_argument(
        "--prediction-snapshot-role",
        type=str,
        default="morning",
        choices=["morning", "lineup_lock", "opening"],
        help="Market role used as the residual prior (never closing).",
    )
    parser.add_argument(
        "--schedule-backtest-mode",
        default=config.SCHEDULE_BACKTEST_MODE_DEFAULT,
        choices=list(config.SCHEDULE_BACKTEST_MODES),
        help="as_of_snapshot (production-equivalent, default) vs actual_starter "
             "(retrospective diagnostic). Modes are never merged.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Override nested validation JSON path.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Write the report only; never save a model artifact.",
    )
    args = parser.parse_args()

    if args.prediction_snapshot_role == "closing":
        raise SystemExit("closing cannot be used as a prediction-time market prior")

    print("Assembling game pick log...")
    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=args.raw_dir,
        season=args.season,
        days=args.days,
        schedule_backtest_mode=args.schedule_backtest_mode,
    )
    if log.empty:
        print("No training rows; aborting.")
        return

    snapshots = _load_snapshots(args.odds_snapshots)
    print(
        f"Loaded {len(snapshots)} odds snapshots; attaching role={args.prediction_snapshot_role} "
        f"as prediction-time prior (closing evaluation-only)."
    )
    frame = game_residual_model.prepare_training_frame(
        log,
        market_snapshots=snapshots,
        prediction_snapshot_role=args.prediction_snapshot_role,
        closing_snapshots=snapshots,
    )
    print(f"Market-aligned training rows: {len(frame)}")
    if frame.empty:
        print(
            "No rows with prediction-time market probabilities. "
            "Persist morning/lineup_lock snapshots before training."
        )
        return

    print("Running nested residual validation...")
    report = game_residual_model.run_game_residual_nested_validation(frame)

    report_dir = config.NESTED_VALIDATION_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    report_path = args.report_path or os.path.join(
        report_dir, config.GAME_RESIDUAL_VALIDATION_REPORT_NAME,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote nested validation report: {report_path}")

    # Also mirror into the betting-gate path when the report carries one.
    betting_path = config.BETTING_PROMOTION_GATE_REPORT_PATH
    os.makedirs(os.path.dirname(betting_path) or ".", exist_ok=True)
    with open(betting_path, "w", encoding="utf-8") as f:
        json.dump({
            "betting_promotion_gate": report.get("betting_promotion_gate"),
            "methods": {
                "residual_logistic": (report.get("methods") or {}).get("residual_logistic"),
            },
            "source_report": report_path,
        }, f, indent=2, default=str)

    gate = report.get("promotion_gate") or {}
    print(f"Probability promotion gate passed: {gate.get('passed')}")
    print(f"Betting promotion gate passed: {(report.get('betting_promotion_gate') or {}).get('passed')}")

    if args.skip_save:
        print("--skip-save set; not writing model artifact.")
        return

    # Fit final residual logistic on all market-aligned rows with the
    # median selected C from outer folds (or strongest default).
    configs = report.get("selected_configs") or []
    if configs:
        C = float(pd.Series([c.get("logistic_C") for c in configs]).median())
    else:
        C = float(config.GAME_RESIDUAL_LOGISTIC_C_GRID[0])

    model = game_residual_model.MarketResidualLogistic(
        C=C,
        feature_columns=list(report.get("feature_columns") or game_residual_model.RESIDUAL_FEATURE_COLUMNS),
    )
    model.fit(frame, frame[game_residual_model.HOME_WON_LABEL], frame[game_residual_model.MARKET_AT_PRED_COL])
    artifact = game_residual_model.ResidualModelArtifact(
        family="logistic",
        model=model,
        feature_columns=list(model.feature_columns),
        metadata={
            "hyperparameters": {"C": C},
            "validation_summary": {
                "promotion_gate": gate,
                "n_outer_folds": report.get("n_outer_folds"),
                "n_games_evaluated": report.get("n_games_evaluated"),
            },
            "model_version": config.GAME_RESIDUAL_MODEL_VERSION,
            "training_data_start": str(frame["date"].min()),
            "training_data_cutoff": str(frame["date"].max()),
            "prediction_snapshot_role": args.prediction_snapshot_role,
        },
    )
    bundle = game_residual_model.save_residual_model(artifact)
    print(
        f"Saved residual model to {config.GAME_RESIDUAL_MODEL_PATH} "
        f"(artifact_id={bundle.get('artifact_id')}, C={C}). "
        f"GAME_PREDICTION_MODE remains '{config.GAME_PREDICTION_MODE}'."
    )


if __name__ == "__main__":
    main()
