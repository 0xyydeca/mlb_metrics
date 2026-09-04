"""Train / validate complete DraftKings-style DFS targets.

Assembles one row per (game_pk, player) with confirmed-starter / DNP zeros
and provenance-tagged official + reconstructed outcomes, then runs nested
time-aware validation against the complete label.

Legacy partial-target models (Actual_DK_Points_Modeled / FIP-proxy ER)
remain the live benchmark. Artifacts are only written as complete-target
challengers; live DFS overrides are never flipped solely because player
MAE improves by a trivial amount — see dfs_complete_targets promotion gate.

Usage:
    python scripts/train_dfs_complete_model.py
    python scripts/train_dfs_complete_model.py --season 2026 --days 60
    python scripts/train_dfs_complete_model.py --skip-save
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, dfs_complete_targets as dct, dfs_ml, ml_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train complete DFS target models")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument(
        "--report-path",
        type=str,
        default=config.DFS_COMPLETE_PROMOTION_GATE_REPORT_PATH,
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Write the nested validation report only; never save model artifacts.",
    )
    parser.add_argument(
        "--confirmed-lineups",
        type=str,
        default=None,
        help="Optional confirmed lineup CSV (game_pk, key_mlbam, is_confirmed_starter).",
    )
    parser.add_argument(
        "--salaries",
        type=str,
        default=None,
        help="Optional salary CSV with key_mlbam and Salary / Estimated_Salary.",
    )
    args = parser.parse_args()

    import pandas as pd

    confirmed = None
    if args.confirmed_lineups and os.path.exists(args.confirmed_lineups):
        confirmed = pd.read_csv(args.confirmed_lineups)
    salaries = None
    if args.salaries and os.path.exists(args.salaries):
        salaries = pd.read_csv(args.salaries)

    print("Assembling complete DK-style target rows...")
    tables = dct.assemble_complete_target_rows(
        raw_dir=args.raw_dir,
        season=args.season,
        days=args.days,
        confirmed_lineups=confirmed,
        salaries=salaries,
    )
    hitters = tables["hitters"]
    pitchers = tables["pitchers"]
    print(f"  hitters={len(hitters)} pitchers={len(pitchers)}")
    print(f"  legacy benchmark kept: {tables['legacy_benchmark']}")

    hitter_report = dct.run_complete_target_nested_validation(
        hitters,
        feature_columns=dfs_ml.HITTER_FEATURE_COLUMNS,
        role="hitter",
        feature_schedule_mode=tables.get("feature_schedule_mode"),
    )
    pitcher_report = dct.run_complete_target_nested_validation(
        pitchers,
        feature_columns=dfs_ml.PITCHER_FEATURE_COLUMNS,
        role="pitcher",
        feature_schedule_mode=tables.get("feature_schedule_mode"),
    )

    combined = {
        "status": (
            "ok"
            if hitter_report.get("status") == "ok" or pitcher_report.get("status") == "ok"
            else "insufficient_history"
        ),
        "target_construction": {
            "grain": "one row per (game_pk, player)",
            "includes_dnp_zeros": True,
            "legacy_partial_label": dct.LEGACY_LABEL,
            "complete_label": dct.COMPLETE_HITTER_LABEL,
            "fip_proxy_er_not_complete_truth": True,
        },
        "mode": {
            "configured": config.DFS_COMPLETE_TARGET_MODE,
            "note": "Live DFS still uses legacy partial targets until promotion gate passes.",
        },
        "hitters": hitter_report,
        "pitchers": pitcher_report,
        "promotion_gate": hitter_report.get("promotion_gate") or {
            "passed": False,
            "checks": {"legacy_partial_kept_as_benchmark": True},
        },
    }
    # Overall gate: require hitter gate when available.
    if hitter_report.get("status") == "ok":
        combined["promotion_gate"] = hitter_report["promotion_gate"]

    path = dct.write_complete_validation_report(combined, args.report_path)
    print(f"Wrote nested validation report -> {path}")
    print(f"  hitter status={hitter_report.get('status')} folds={hitter_report.get('n_outer_folds')}")
    print(f"  pitcher status={pitcher_report.get('status')} folds={pitcher_report.get('n_outer_folds')}")
    gate = combined.get("promotion_gate") or {}
    print(f"  promotion_gate.passed={gate.get('passed')} trivial_mae_only={gate.get('trivial_mae_only')}")

    if args.skip_save:
        print("Skipping model save (--skip-save).")
        return

    ok, details = dct.evaluate_complete_promotion_gate(combined)
    if not ok:
        print(f"NOT saving complete-target models ({details.get('reason')}).")
        print("Legacy partial-target models remain the live / benchmark path.")
        return

    # Fit final complete models on all assembled rows (challenger artifacts only).
    if not hitters.empty and dct.COMPLETE_HITTER_LABEL in hitters.columns:
        cfg = (hitter_report.get("selected_configs") or [{}])[-1]
        model = dct.fit_complete_target_model(
            hitters,
            dfs_ml.HITTER_FEATURE_COLUMNS,
            dct.COMPLETE_HITTER_LABEL,
            family=cfg.get("family", "ridge"),
            params=cfg.get("params"),
        )
        ml_models.save_model(model, config.DFS_COMPLETE_HITTER_MODEL_PATH)
        print(f"Saved complete hitter challenger -> {config.DFS_COMPLETE_HITTER_MODEL_PATH}")
        dct.apply_complete_shadow_predictions(
            hitters, model, dfs_ml.HITTER_FEATURE_COLUMNS,
        )
        print(f"Wrote shadow predictions -> {config.DFS_COMPLETE_SHADOW_PREDICTIONS_PATH}")

    if not pitchers.empty and dct.COMPLETE_PITCHER_LABEL in pitchers.columns:
        cfg = (pitcher_report.get("selected_configs") or [{}])[-1]
        model = dct.fit_complete_target_model(
            pitchers,
            dfs_ml.PITCHER_FEATURE_COLUMNS,
            dct.COMPLETE_PITCHER_LABEL,
            family=cfg.get("family", "ridge"),
            params=cfg.get("params"),
        )
        ml_models.save_model(model, config.DFS_COMPLETE_PITCHER_MODEL_PATH)
        print(f"Saved complete pitcher challenger -> {config.DFS_COMPLETE_PITCHER_MODEL_PATH}")


if __name__ == "__main__":
    main()
