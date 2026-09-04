"""Run nested rolling-origin validation and write a machine-readable report.

Does not change live prediction behavior. Does not dump enormous raw fold
predictions by default.

Usage:
    python scripts/run_nested_model_validation.py
    python scripts/run_nested_model_validation.py --freeze-dates 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, dfs_backtest, dfs_ml, model_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--min-plate-appearances", type=int, default=config.BACKTEST_MIN_PLATE_APPEARANCES)
    parser.add_argument("--freeze-dates", type=int, default=config.NESTED_VALIDATION_FREEZE_DATES)
    parser.add_argument("--report-dir", default=config.NESTED_VALIDATION_REPORT_DIR)
    parser.add_argument("--full-grids", action="store_true")
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    rows = dfs_backtest.assemble_hitter_hit_log(args.raw_dir, season=season, days=args.days)
    if rows.empty:
        print("No hit-log rows - nothing to validate.")
        return
    rows = rows[rows["Total_PA"] >= args.min_plate_appearances]
    if rows.empty:
        print("No rows clear the PA gate.")
        return

    for col in dfs_ml.HITTER_FEATURE_COLUMNS:
        if col not in rows.columns:
            rows[col] = 0.0

    candidates = model_validation.default_hitter_hit_candidates(
        list(dfs_ml.HITTER_FEATURE_COLUMNS), include_full_grids=args.full_grids,
    )
    nested_config = model_validation.NestedValidationConfig(
        freeze_dates=args.freeze_dates,
        report_dir=args.report_dir,
        bootstrap_samples=min(500, config.NESTED_VALIDATION_BOOTSTRAP_SAMPLES),
    )
    report = model_validation.run_nested_classifier_validation(
        rows, candidates, nested_config,
        champion_probability_col="Game_Hit_Probability",
    )
    model_validation.ensure_reports_dir_readme(args.report_dir)
    path = model_validation.write_validation_report(report, args.report_dir)
    print(f"status={report.get('status')} outer_folds={report.get('n_outer_folds')} report={path}")
    if report.get("freeze_dates"):
        print("Freeze dates (do not inspect during development):", report["freeze_dates"])
    agg = report.get("aggregate") or {}
    if agg:
        print(
            f"aggregate log_loss={agg.get('log_loss')} brier={agg.get('brier_score')} "
            f"roc_auc={agg.get('roc_auc')} coverage={agg.get('coverage')}"
        )


if __name__ == "__main__":
    main()
