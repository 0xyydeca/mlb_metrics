"""Train the shadow opportunity-aware hitter probability model.

Reads ``data/predictions/hitter_opportunity_log.csv`` (see
``scripts/build_hitter_opportunity_log.py``), runs nested rolling-origin
validation over appearance / expected-PA / conditional-hit components,
compares the direct conditional classifier to the PA-distribution
challenger on outer folds, writes a validation report + shadow
predictions, and optionally saves a model bundle.

Does **not** alter ``predictions.select_picks`` or any live pick path.

Usage:
    python scripts/train_hitter_probability_model.py
    python scripts/train_hitter_probability_model.py --full-grids
    python scripts/train_hitter_probability_model.py --skip-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, hitter_probability_model as hpm, model_validation


def _load_opportunity_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing opportunity log at {path}. "
            "Run scripts/build_hitter_opportunity_log.py first."
        )
    return pd.read_csv(path)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--opportunity-log",
        default="data/predictions/hitter_opportunity_log.csv",
        help="Path to hitter opportunity log CSV",
    )
    parser.add_argument(
        "--model-path",
        default=config.HITTER_OPPORTUNITY_PROBABILITY_MODEL_PATH,
        help="Where to save the shadow model bundle",
    )
    parser.add_argument(
        "--shadow-predictions",
        default=config.HITTER_OPPORTUNITY_SHADOW_PREDICTIONS_PATH,
        help="Outer-fold shadow predictions CSV",
    )
    parser.add_argument(
        "--report-dir",
        default=config.NESTED_VALIDATION_REPORT_DIR,
        help="Directory for nested validation JSON reports",
    )
    parser.add_argument(
        "--full-grids",
        action="store_true",
        help="Expand logit/GBM hyperparameter grids (slower)",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Run validation + shadow preds but do not write the model bundle",
    )
    parser.add_argument(
        "--outer-min-train-dates",
        type=int,
        default=config.NESTED_VALIDATION_OUTER_MIN_TRAIN_DATES,
    )
    parser.add_argument(
        "--outer-test-block-dates",
        type=int,
        default=config.NESTED_VALIDATION_OUTER_TEST_BLOCK_DATES,
    )
    parser.add_argument(
        "--inner-min-train-dates",
        type=int,
        default=config.NESTED_VALIDATION_INNER_MIN_TRAIN_DATES,
    )
    parser.add_argument(
        "--inner-test-block-dates",
        type=int,
        default=config.NESTED_VALIDATION_INNER_TEST_BLOCK_DATES,
    )
    parser.add_argument(
        "--freeze-dates",
        type=int,
        default=config.NESTED_VALIDATION_FREEZE_DATES,
    )
    args = parser.parse_args()

    print("=== Shadow opportunity hitter probability model ===")
    print("Live wiring: none (does not alter predictions.select_picks)")
    raw = _load_opportunity_log(args.opportunity_log)
    rows = hpm.prepare_opportunity_training_frame(raw)
    print(f"Loaded {len(raw)} log rows → {len(rows)} training rows "
          f"across {rows['date'].nunique()} dates "
          f"(appear rate={rows['Appeared'].mean():.3f})")

    nested_config = model_validation.NestedValidationConfig(
        outer_min_train_dates=args.outer_min_train_dates,
        outer_test_block_dates=args.outer_test_block_dates,
        inner_min_train_dates=args.inner_min_train_dates,
        inner_test_block_dates=args.inner_test_block_dates,
        freeze_dates=args.freeze_dates,
        report_dir=args.report_dir,
    )

    report = hpm.run_opportunity_nested_validation(
        rows,
        nested_config=nested_config,
        include_full_grids=args.full_grids,
    )
    shadow = report.pop("_shadow_predictions", pd.DataFrame())
    last_model = report.pop("_last_fitted_model", None)
    selected_specs = report.pop("_selected_specs", [])

    print(f"\nStatus: {report.get('status')}  outer_folds={report.get('n_outer_folds')}")
    agg = report.get("aggregate") or {}
    if agg:
        print("Aggregate outer metrics:")
        for k, v in agg.items():
            print(f"  {k}: {v}")
    ablation = report.get("ablation") or {}
    if ablation:
        print("\nAblation (inner selection):")
        print(f"  feature_set means: {ablation.get('mean_inner_log_loss_by_feature_set')}")
        print(f"  formulation means: {ablation.get('mean_inner_log_loss_by_formulation')}")
        print(f"  family means: {ablation.get('mean_inner_log_loss_by_family')}")
        print(f"  selected counts: {ablation.get('selected_counts')}")
    form_cmp = report.get("formulation_comparison") or {}
    if form_cmp:
        print("\nFormulation comparison (outer folds):")
        print(f"  {form_cmp.get('mean_outer_log_loss_by_formulation')}")
        print(f"  note: {form_cmp.get('note')}")

    model_validation.ensure_reports_dir_readme(args.report_dir)
    report_path = os.path.join(
        args.report_dir, config.HITTER_OPPORTUNITY_VALIDATION_REPORT_NAME,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, sort_keys=True)
    print(f"\nWrote validation report → {report_path}")

    if not shadow.empty:
        hpm.write_shadow_predictions(shadow, args.shadow_predictions)
        print(f"Wrote shadow predictions → {args.shadow_predictions} ({len(shadow)} rows)")
    else:
        print("No shadow prediction rows to write (insufficient outer folds).")

    if args.skip_save:
        print("Skipping model bundle save (--skip-save).")
        return

    spec = hpm.majority_selected_spec(selected_specs)
    if spec is None:
        print("No selected spec from nested validation; not saving a bundle.")
        return

    print(f"\nRefitting on active history with selected spec:\n  {spec.name}")
    fitted = hpm.refit_on_active_history(rows, spec, nested_config)
    train_start = str(rows["date"].min())
    train_cutoff = str(rows["date"].max())
    bundle = hpm.save_opportunity_model_bundle(
        fitted,
        args.model_path,
        validation_summary={
            "aggregate": report.get("aggregate"),
            "ablation": report.get("ablation"),
            "formulation_comparison": report.get("formulation_comparison"),
            "n_outer_folds": report.get("n_outer_folds"),
            "selected_spec": spec.name,
            "shadow": True,
            "live_wiring": "none",
        },
        training_data_start=train_start,
        training_data_cutoff=train_cutoff,
    )
    print(f"Saved shadow model bundle → {args.model_path}")
    print(f"  artifact_id={bundle.get('artifact_id')}")
    print(f"  formulation={fitted.formulation}")
    print(f"  n_features={len(fitted.feature_columns)}")
    # Keep last_model reference unused-warning free when folds exist
    _ = last_model


if __name__ == "__main__":
    main()
