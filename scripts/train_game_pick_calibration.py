"""Fit + report phase for a RECALIBRATION of the existing Automated Game
Picks heuristic (game_picks.compute_game_win_probabilities), not a
from-scratch replacement model - see scripts/train_game_pick_model.py for
that (already tried, already failed to beat the heuristic: see its own
README section).

Quant-analytics follow-up ("dig into calibration", 2026-08-24): real
production data showed the heuristic's own home_win_probability ratio has
a genuine (if modest) real edge over a coin flip (accuracy 0.547, n=459
resolved picks), but its own docstring explicitly admits it's NOT a
calibrated probability - and real data confirmed a concrete problem: its
spread (std 0.035 across 93 real market-compared games) is far narrower
than the real market's own spread on the exact same games (std 0.059,
max confidence 72.7% vs the model's own 63.7% ceiling). That narrowness
was the direct mechanical cause of a real bet-advice false-edge bug
(config.KELLY_MIN_EDGE raised 0.02 -> 0.05 as an immediate stopgap - see
README's "Real quant sanity-check" section) - on any game the market is
confident about, the model's comparatively muted probability for the
other side looks like value that isn't real.

This script fits ml_models.fit_probability_calibration (isotonic or
sigmoid/Platt - both are candidates, picked via real walk-forward CV, not
guessed) on top of the EXISTING heuristic's own raw home_win_probability,
against real resolved outcomes - a rescaling, not a new model. Save gate
(2026-08-24 policy - "as long as it beats our current model, save it"):
saved to config.GAME_PICK_CALIBRATION_MODEL_PATH ONLY if the calibrated
probability beats the raw (uncalibrated) heuristic - the current live
model for this signal - on an untouched final holdout; the naive
baseline is still reported for context but is no longer a required bar.
Reported honestly either way.

**Scope**: unlike train_game_pick_model.py's saved-but-unwired artifact,
THIS artifact IS wired live (game_picks.apply_calibration, called from
pipeline.run() and game_picks_backtest.reconstruct_historical_game_picks_from_persisted)
- loading gracefully falls back to the raw, uncalibrated heuristic when no
artifact exists yet (or hasn't cleared its own bar), same contract
dfs_ml.apply_ml_overrides already establishes for the hitter side.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py). Fully
offline - game_picks_backtest.assemble_game_pick_log reads real
starters/scores directly out of persisted Statcast, no live schedule/
network fetch anywhere in this script.

Usage:
    python scripts/train_game_pick_calibration.py
    python scripts/train_game_pick_calibration.py --season 2026
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, game_picks_backtest, ml_models


def _split_holdout(df, holdout_dates: int):
    dates = sorted(df["date"].unique())
    if len(dates) <= holdout_dates:
        return df.iloc[0:0], df, dates  # not enough dates for a real holdout
    cutoff_dates = set(dates[-holdout_dates:])
    holdout = df[df["date"].isin(cutoff_dates)]
    train_pool = df[~df["date"].isin(cutoff_dates)]
    return train_pool, holdout, dates


def _cv_log_loss(train_pool, method: str, min_train_dates: int, test_block_dates: int) -> float:
    """Mean walk-forward-CV log_loss for one candidate calibration method,
    fit fresh on each fold's own train slice (no lookahead) and scored on
    that fold's held-out block - the same no-lookahead discipline
    WalkForwardDateSplit enforces everywhere else in this project, applied
    manually here since fit_probability_calibration isn't a scikit-learn
    Estimator GridSearchCV can drive directly (no get_params/predict_proba).
    Returns NaN if no fold has both a non-empty train and test slice."""
    dates = train_pool["date"].reset_index(drop=True)
    raw = train_pool["home_win_probability"].reset_index(drop=True)
    y = train_pool["Home_Won"].reset_index(drop=True)

    splitter = ml_models.WalkForwardDateSplit(dates, min_train_dates, test_block_dates)
    fold_losses = []
    for train_idx, test_idx in splitter.split():
        fold_calibrator = ml_models.fit_probability_calibration(raw.iloc[train_idx], y.iloc[train_idx], method=method)
        fold_predicted = fold_calibrator.predict(raw.iloc[test_idx].to_numpy())
        fold_result = ml_models.evaluate_classifier_predictions(y.iloc[test_idx], fold_predicted)
        fold_losses.append(fold_result["log_loss"])

    if not fold_losses:
        return float("nan")
    return sum(fold_losses) / len(fold_losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Trim the game pick log to the most recent N dates instead of the full persisted "
             "history - each date recomputes the full pipeline, which is expensive. Full history "
             "is the default (None), matching train_game_pick_model.py's own --days flag.",
    )
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    print(f"Assembling game pick log (season={season}, days={args.days or 'all'})...")
    rows = game_picks_backtest.assemble_game_pick_log(args.raw_dir, season=season, days=args.days)

    if rows.empty:
        print("No game pick log rows assembled - nothing to fit.")
        return

    # A real, genuine NaN case (not a bug in this script): compute_game_win_probabilities
    # returns NaN for a game whose home/away composite rating is itself
    # NaN (e.g. a team missing from that date's confidence.csv snapshot -
    # clip() preserves NaN rather than flooring it). No real calibration
    # target to fit on for a row like that, so it's dropped and reported,
    # not silently zero-filled - unlike game_feature_matrix's raw X
    # features (which DO get NaN-filled to 0, a defensible neutral value
    # for a model feature), a fabricated 0.0/0.5 "probability" here would
    # actively corrupt the calibration curve.
    before = len(rows)
    rows = rows[rows["home_win_probability"].notna()].copy()
    if before - len(rows):
        print(f"Dropped {before - len(rows)} row(s) with a NaN raw home_win_probability (a real missing-composite-data date) - can't calibrate on those.")

    train_pool, holdout, dates = _split_holdout(rows, config.ML_FINAL_HOLDOUT_DATES)
    print(f"{len(dates)} distinct dates, {len(train_pool)} train-pool rows, {len(holdout)} holdout rows")
    if holdout.empty or train_pool.empty:
        print("Not enough distinct dates for a real holdout split - skipping.")
        return

    print("\n=== Walk-forward CV: isotonic vs sigmoid (Platt) ===")
    cv_results = {}
    for method in ("isotonic", "sigmoid"):
        cv_log_loss = _cv_log_loss(
            train_pool, method,
            config.GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_DATES, config.GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_DATES,
        )
        cv_results[method] = cv_log_loss
        print(f"  {method}: mean CV log_loss={cv_log_loss:.4f}")

    valid_results = {m: v for m, v in cv_results.items() if v == v}  # drop NaN
    if not valid_results:
        print("No CV fold had both a train and test slice - not enough history yet to validate. Not saved.")
        return
    best_method = min(valid_results, key=valid_results.get)
    print(f"  -> best by CV: {best_method}")

    print(f"\n=== Real final holdout ({len(holdout)} rows, {config.ML_FINAL_HOLDOUT_DATES} most recent dates) ===")
    calibrator = ml_models.fit_probability_calibration(
        train_pool["home_win_probability"], train_pool["Home_Won"], method=best_method
    )
    calibrated_proba = calibrator.predict(holdout["home_win_probability"].to_numpy())
    calibrated_result = ml_models.evaluate_classifier_predictions(holdout["Home_Won"], calibrated_proba)
    raw_result = ml_models.evaluate_classifier_predictions(
        holdout["Home_Won"], holdout["home_win_probability"].clip(0, 1)
    )

    print(
        f"  Calibrated ({best_method}): log_loss={calibrated_result['log_loss']:.4f}, "
        f"brier={calibrated_result['brier_score']:.4f}, roc_auc={calibrated_result['roc_auc']:.4f}, "
        f"accuracy={calibrated_result['accuracy']:.4f} (n={calibrated_result['n']})"
    )
    print(f"  Naive baseline:             log_loss={calibrated_result['baseline_log_loss']:.4f} (always predict base rate)")
    print(
        f"  Raw heuristic (uncalibrated): log_loss={raw_result['log_loss']:.4f}, "
        f"brier={raw_result['brier_score']:.4f}, roc_auc={raw_result['roc_auc']:.4f}, "
        f"accuracy={raw_result['accuracy']:.4f}"
    )
    print(f"  Spread (std): calibrated={calibrated_proba.std():.4f}, raw={holdout['home_win_probability'].std():.4f}")

    # Save gate (2026-08-24 policy: "our goal is to get more and more
    # accurate, so as long as it beats our current model, save it" - the
    # naive always-predict-base-rate baseline is still printed above for
    # context, but is no longer a REQUIRED bar to clear. "Our current
    # model" here is the raw, uncalibrated heuristic - what's actually
    # live via game_picks.apply_calibration today whenever no calibration
    # artifact exists yet (its own graceful-degradation fallback).
    beats_raw = calibrated_result["log_loss"] < raw_result["log_loss"]
    if beats_raw:
        final_calibrator = ml_models.fit_probability_calibration(
            rows["home_win_probability"], rows["Home_Won"], method=best_method
        )
        ml_models.save_model_bundle(
            final_calibrator,
            config.GAME_PICK_CALIBRATION_MODEL_PATH,
            model_type="game_pick_calibration",
            model_version=config.GAME_PICK_MODEL_VERSION,
            feature_columns=["home_win_probability"],
            training_data_start=train_pool["date"].min(),
            training_data_cutoff=train_pool["date"].max(),
            hyperparameters={},
            calibration_method=best_method,
            validation_summary={
                "holdout_log_loss": calibrated_result["log_loss"],
                "holdout_brier_score": calibrated_result["brier_score"],
                "raw_log_loss": raw_result["log_loss"],
                "holdout_n": calibrated_result["n"],
            },
        )
        print(
            f"\n  -> SAVED bundle to {config.GAME_PICK_CALIBRATION_MODEL_PATH} "
            f"(beats the raw heuristic on the real holdout - refit on the FULL log "
            f"before saving, same as train_game_pick_model.py's own pattern; wired live via "
            f"game_picks.apply_calibration)"
        )
    else:
        print(
            "\n  -> NOT saved (does not beat the raw heuristic on the real holdout - "
            "reported honestly). Any existing artifact at "
            f"{config.GAME_PICK_CALIBRATION_MODEL_PATH} is left untouched, not deleted - only a real "
            "improvement overwrites it."
        )


if __name__ == "__main__":
    main()
