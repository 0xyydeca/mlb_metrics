"""Train / validate the market-residual game-win challenger.

Assembles the game-pick training log, attaches prediction-time market
snapshots (never closing as features; never unmatched / post-start /
historically backfilled "morning" rows), runs nested rolling-origin
validation, and writes a machine-readable report.

The production artifact ``game_residual_win_probability_model.joblib`` is
saved ONLY when ``report["promotion_gate"]["passed"] is True``. A failed
gate never creates or overwrites that path.

Does NOT enable live betting or change GAME_PREDICTION_MODE / BETTING_MODE.

Usage:
    python scripts/train_game_residual_model.py
    python scripts/train_game_residual_model.py --season 2026 --days 120
    python scripts/train_game_residual_model.py --force-save-for-debug /tmp/debug_residual.joblib
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


def _production_model_path() -> str:
    return os.path.abspath(config.GAME_RESIDUAL_MODEL_PATH)


def _assert_debug_path_not_production(path: str) -> str:
    abs_path = os.path.abspath(path)
    prod = _production_model_path()
    if abs_path == prod:
        raise SystemExit(
            "Refusing --force-save-for-debug: path resolves to the production "
            f"artifact ({prod}). Pass a conspicuous non-production path."
        )
    # Also refuse saving inside data/models/ under the production filename.
    if os.path.basename(abs_path) == os.path.basename(prod) and os.path.dirname(abs_path) == os.path.dirname(prod):
        raise SystemExit(
            "Refusing --force-save-for-debug overwriting the production artifact path."
        )
    return abs_path


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _insufficient_report(*, reason: str, extras: dict | None = None) -> dict:
    report = {
        "status": "insufficient_data",
        "validation_status": "insufficient_data",
        "artifact_saved": False,
        "artifact_loaded": False,
        "n_outer_folds": 0,
        "methods": {},
        "promotion_gate": {"passed": False, "checks": {}},
        "betting_promotion_gate": {"passed": False, "checks": {}},
        "insufficient_reason": reason,
    }
    if extras:
        report.update(extras)
    return report


def maybe_save_residual_artifact(
    report: dict,
    frame: pd.DataFrame,
    *,
    prediction_snapshot_role: str,
    skip_save: bool = False,
    force_save_for_debug: str | None = None,
) -> str | None:
    """Save production artifact only when promotion_gate.passed is exactly True.

    Returns the path written, or None when nothing was saved.
    """
    gate = report.get("promotion_gate") or {}
    gate_passed = gate.get("passed") is True

    if skip_save and not force_save_for_debug:
        print("--skip-save set; not writing model artifact.")
        return None

    if not gate_passed and not force_save_for_debug:
        print("NOT SAVED — probability promotion gate failed.")
        return None

    configs = report.get("selected_configs") or []
    if configs:
        C = float(pd.Series([c.get("logistic_C") for c in configs]).median())
    else:
        C = float(config.GAME_RESIDUAL_LOGISTIC_C_GRID[0])

    model = game_residual_model.MarketResidualLogistic(
        C=C,
        feature_columns=list(
            report.get("feature_columns") or game_residual_model.RESIDUAL_FEATURE_COLUMNS
        ),
    )
    model.fit(
        frame,
        frame[game_residual_model.HOME_WON_LABEL],
        frame[game_residual_model.MARKET_AT_PRED_COL],
    )
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
            "prediction_snapshot_role": prediction_snapshot_role,
        },
    )

    if force_save_for_debug:
        debug_path = _assert_debug_path_not_production(force_save_for_debug)
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        bundle = game_residual_model.save_residual_model(artifact, path=debug_path)
        print(
            f"DEBUG-ONLY artifact written to {debug_path} "
            f"(artifact_id={bundle.get('artifact_id')}, gate_passed={gate_passed}). "
            f"Production path untouched."
        )
        if not gate_passed:
            print("NOT SAVED — probability promotion gate failed. (production path)")
        return debug_path

    # gate_passed is True here.
    bundle = game_residual_model.save_residual_model(artifact)
    print(
        f"Saved residual model to {config.GAME_RESIDUAL_MODEL_PATH} "
        f"(artifact_id={bundle.get('artifact_id')}, C={C}). "
        f"GAME_PREDICTION_MODE remains '{config.GAME_PREDICTION_MODE}'; "
        f"BETTING_MODE remains '{config.BETTING_MODE}'."
    )
    return config.GAME_RESIDUAL_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train market-residual game-win model")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days",
        type=int,
        default=config.GAME_RESIDUAL_TRAINING_HISTORY_DATES,
        help=(
            "Game-date horizon for assemble_game_pick_log "
            f"(default {config.GAME_RESIDUAL_TRAINING_HISTORY_DATES}; "
            f"structural fold minimum is "
            f"{config.GAME_RESIDUAL_MIN_DATES_FOR_COMPLETE_OUTER_FOLDS})."
        ),
    )
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
    parser.add_argument(
        "--force-save-for-debug",
        type=str,
        default=None,
        help="Write a debug-only artifact to this NON-PRODUCTION path. "
             "Never overwrites data/models/game_residual_win_probability_model.joblib.",
    )
    args = parser.parse_args()

    if args.prediction_snapshot_role == "closing":
        raise SystemExit("closing cannot be used as a prediction-time market prior")

    report_dir = config.NESTED_VALIDATION_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    report_path = args.report_path or os.path.join(
        report_dir, config.GAME_RESIDUAL_VALIDATION_REPORT_NAME,
    )
    betting_path = config.BETTING_PROMOTION_GATE_REPORT_PATH

    print("Assembling game pick log...")
    print(
        f"Training history horizon: days={args.days} "
        f"(min for complete outer folds="
        f"{config.GAME_RESIDUAL_MIN_DATES_FOR_COMPLETE_OUTER_FOLDS})"
    )
    log = game_picks_backtest.assemble_game_pick_log(
        raw_dir=args.raw_dir,
        season=args.season,
        days=args.days,
        schedule_backtest_mode=args.schedule_backtest_mode,
    )
    if log.empty:
        report = _insufficient_report(reason="empty_game_pick_log")
        _write_json(report_path, report)
        _write_json(betting_path, {
            "betting_promotion_gate": report.get("betting_promotion_gate"),
            "methods": {},
            "source_report": report_path,
            "artifact_id": None,
            "validation_status": "insufficient_data",
            "artifact_saved": False,
        })
        print(f"No training rows; wrote insufficient_data report: {report_path}")
        print(
            "RUN OUTCOME: validation_status=insufficient_data "
            "artifact_saved=False artifact_loaded=False"
        )
        return

    snapshots = _load_snapshots(args.odds_snapshots)
    valid_pred = market_odds.filter_valid_prediction_time_snapshots(
        snapshots, required_role=args.prediction_snapshot_role,
    )
    valid_dates = sorted({
        str(pd.Timestamp(d).date())
        for d in valid_pred.get("date", pd.Series(dtype="datetime64[ns]")).dropna().unique()
    }) if not valid_pred.empty else []
    outer_possible = game_residual_model.estimate_outer_folds_possible(valid_dates)
    print(
        f"Loaded {len(snapshots)} odds snapshots; "
        f"{len(valid_pred)} valid prediction-time ({args.prediction_snapshot_role}) rows "
        f"after rejecting unmatched/post-start/historical backfills."
    )
    print(
        f"Preflight: valid_prediction_dates={len(valid_dates)} "
        f"outer_folds_possible={outer_possible} "
        f"(need >= {config.BETTING_PROMOTION_MIN_OUTER_FOLDS} complete blocks)"
    )
    frame = game_residual_model.prepare_training_frame(
        log,
        market_snapshots=snapshots,
        prediction_snapshot_role=args.prediction_snapshot_role,
        closing_snapshots=snapshots,
    )
    print(f"Market-aligned training rows: {len(frame)}")
    if frame.empty:
        report = _insufficient_report(
            reason="no_prediction_time_market_rows",
            extras={
                "preflight_valid_prediction_dates": len(valid_dates),
                "preflight_outer_folds_possible": outer_possible,
            },
        )
        _write_json(report_path, report)
        _write_json(betting_path, {
            "betting_promotion_gate": report.get("betting_promotion_gate"),
            "methods": {},
            "source_report": report_path,
            "artifact_id": None,
            "validation_status": "insufficient_data",
            "artifact_saved": False,
        })
        print(
            "No rows with prediction-time market probabilities. "
            f"Wrote insufficient_data report: {report_path}"
        )
        print(
            "RUN OUTCOME: validation_status=insufficient_data "
            "artifact_saved=False artifact_loaded=False"
        )
        return

    print("Running nested residual validation...")
    report = game_residual_model.run_game_residual_nested_validation(frame)
    report.setdefault("preflight_valid_prediction_dates", len(valid_dates))
    report.setdefault("preflight_outer_folds_possible", outer_possible)
    report.setdefault("training_history_days", args.days)

    saved = maybe_save_residual_artifact(
        report,
        frame,
        prediction_snapshot_role=args.prediction_snapshot_role,
        skip_save=args.skip_save,
        force_save_for_debug=args.force_save_for_debug,
    )
    production_saved = bool(
        saved and (report.get("promotion_gate") or {}).get("passed") is True
        and saved == config.GAME_RESIDUAL_MODEL_PATH
    )
    report["artifact_saved"] = production_saved
    artifact_id = None
    if production_saved:
        art = game_residual_model.load_residual_model(saved)
        report["artifact_loaded"] = art is not None
        artifact_id = (art.metadata or {}).get("artifact_id") if art else None
    else:
        report["artifact_loaded"] = False

    _write_json(report_path, report)
    print(f"Wrote nested validation report: {report_path}")

    _write_json(betting_path, {
        "betting_promotion_gate": report.get("betting_promotion_gate"),
        "methods": {
            "residual_logistic": (report.get("methods") or {}).get("residual_logistic"),
        },
        "source_report": report_path,
        "artifact_id": artifact_id,
        "validation_status": report.get("validation_status"),
        "artifact_saved": production_saved,
        "artifact_loaded": report.get("artifact_loaded"),
    })

    gate = report.get("promotion_gate") or {}
    print(f"Probability promotion gate passed: {gate.get('passed')}")
    print(f"Betting promotion gate passed: {(report.get('betting_promotion_gate') or {}).get('passed')}")
    print(
        "RUN OUTCOME: "
        f"validation_status={report.get('validation_status')} "
        f"artifact_saved={report.get('artifact_saved')} "
        f"artifact_loaded={report.get('artifact_loaded')}"
    )


if __name__ == "__main__":
    main()
