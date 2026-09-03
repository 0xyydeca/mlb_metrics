"""Two deliverables from the game pick log (game_picks_backtest.assemble_game_pick_log,
see its docstring) - fit + report phase for a game-level win-probability
model, mirroring scripts/train_hitter_hit_model.py's exact structure and
scope on the team side:

1. **Significance report** (statsmodels.Logit): one univariate Logit per
   feature in game_picks.GAME_PICK_FEATURE_COLUMNS ("individually"), plus
   one multivariate Logit with all of them together ("combined"), fit on
   the full assembled log (no holdout split needed here, since statistical
   inference isn't a held-out-accuracy question). Features are standardized
   (z-scored) first so coefficient magnitudes are comparable across very
   different scales - p-values themselves are scale-invariant.

   Also tests CANDIDATE_FEATURE_COLUMNS - every bullpen-fatigue
   exploratory column game_picks_backtest.assemble_game_pick_log carries
   (game_picks_backtest.BULLPEN_FATIGUE_CANDIDATE_COLUMNS, reused
   directly as the single source of truth for this list): the original
   2-day recent-outs total ("what about bullpen rest/readiness",
   2026-08-25 - no real signal, see README), plus a real follow-up sweep
   ("I want to see if other applications of bullpen fatigue are
   significant... I don't care if they're cheap") - additional recency
   windows (config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS), distinct-relievers-
   used (workload BREADTH, not total), and back-to-back-relievers (which
   SPECIFIC arms are on zero rest). None is part of
   game_picks.GAME_PICK_FEATURE_COLUMNS yet; included in this same
   significance report so a real p-value decides
   whether either earns a permanent place in the model.

2. **Walk-forward-validated predictive model** (sklearn LogisticRegression +
   ml_models.py's WalkForwardDateSplit machinery): the SAME no-lookahead,
   nested-holdout methodology train_hitter_hit_model.py already uses. The
   selected model is evaluated ONCE on an untouched final holdout and must
   beat BOTH a naive baseline (always predict the base home-win rate) AND
   the existing home_win_probability heuristic column (carried through the
   log unchanged, from game_picks.compute_game_win_probabilities) before
   being saved to config.GAME_PICK_WIN_PROBABILITY_MODEL_PATH.

**Scope**: this fits and reports. The saved model artifact is NOT wired
into game_picks.compute_game_win_probabilities, game_predictions.select_game_picks,
or pipeline.run() - whether/how this ever nudges live game picks is a
separate, later, backtest-calibrated decision.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py). Fully
offline - assemble_game_pick_log reads real starters/scores directly out
of persisted Statcast, no live schedule/network fetch anywhere in this
script.

Usage:
    python scripts/train_game_pick_model.py
    python scripts/train_game_pick_model.py --season 2026
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from mlb_metrics import config, game_picks, game_picks_backtest, ml_models


def _split_holdout(df: pd.DataFrame, holdout_dates: int):
    dates = sorted(df["date"].unique())
    if len(dates) <= holdout_dates:
        return df.iloc[0:0], df, dates  # not enough dates for a real holdout
    cutoff_dates = set(dates[-holdout_dates:])
    holdout = df[df["date"].isin(cutoff_dates)]
    train_pool = df[~df["date"].isin(cutoff_dates)]
    return train_pool, holdout, dates


def _standardize(X: pd.DataFrame) -> pd.DataFrame:
    """z-score every column; drops any column with zero variance in this
    data (would otherwise divide by zero / be perfectly collinear with the
    intercept) and prints which, if any, were dropped."""
    std = X.std()
    constant_columns = std[std == 0].index.tolist()
    if constant_columns:
        print(f"  Excluding constant-in-this-data feature(s) from the significance report: {constant_columns}")
    X = X.drop(columns=constant_columns)
    return (X - X.mean()) / X.std()


def _fit_logit_report(y: pd.Series, X: pd.DataFrame, label: str):
    """Returns (table, result) on success, or None (never raises) if the
    fit fails - e.g. a near-perfectly-collinear multivariate design - so a
    single failed fit is reported and skipped rather than crashing the
    whole script."""
    try:
        result = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    except Exception as exc:
        print(f"  {label}: fit failed ({exc})")
        return None
    table = pd.DataFrame({
        "coef": result.params, "std_err": result.bse, "z": result.tvalues, "p_value": result.pvalues,
    }).drop(index="const")
    return table, result


# Exploratory candidates - every bullpen-fatigue column
# game_picks_backtest.assemble_game_pick_log carries (see that module's
# own BULLPEN_FATIGUE_CANDIDATE_COLUMNS comment for the full real
# history: the original 2-day recent-outs total found no signal
# (2026-08-25), so this reuses the SAME source-of-truth list rather than
# hardcoding a second copy, to test the broader sweep - "I want to see
# if other applications of bullpen fatigue are significant... I don't
# care if they're cheap"). None is part of
# game_picks.GAME_PICK_FEATURE_COLUMNS - tested here, alongside the live
# feature set, purely to decide whether any earns a permanent place in
# the model. Nothing here changes what the live model actually uses
# until a candidate clears a real bar and is deliberately added to
# GAME_PICK_FEATURE_COLUMNS in a follow-up change.
CANDIDATE_FEATURE_COLUMNS = game_picks_backtest.BULLPEN_FATIGUE_CANDIDATE_COLUMNS


def significance_report(rows: pd.DataFrame) -> None:
    print("\n=== Feature significance report (statsmodels.Logit, full log) ===")
    y = rows["Home_Won"]
    # .astype(float) only retags the dtype (every value is already
    # numeric) - same object-dtype guard train_hitter_hit_model.py needs,
    # in case an early-history date falls back to pd.NA for a not-yet-
    # computable column and upcasts the concatenated column to object.
    X = _standardize(game_picks.game_feature_matrix(rows).astype(float))

    # Same fillna(0)-for-not-yet-computable convention as
    # game_feature_matrix (a real 0 recent-outs is indistinguishable here
    # from "no relief appearance in the window yet" - an honest known
    # rough edge for an EXPLORATORY column, not worth a bespoke encoding
    # before we know whether this candidate has any real signal at all).
    candidates = _standardize(rows[CANDIDATE_FEATURE_COLUMNS].astype(float).fillna(0))
    X = pd.concat([X, candidates], axis=1)

    print(f"  n={len(rows)}, base home-win rate={y.mean():.4f}")
    print(f"  Candidate features under test (not yet in the live model): {CANDIDATE_FEATURE_COLUMNS}")

    print("\n  -- Individually (one univariate Logit per feature) --")
    univariate_rows = []
    for column in X.columns:
        fit = _fit_logit_report(y, X[[column]], column)
        if fit is None:
            continue
        table, _ = fit
        univariate_rows.append(table.loc[column])
    if univariate_rows:
        univariate_table = pd.DataFrame(univariate_rows)
        print(univariate_table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n  -- Combined (one multivariate Logit, all features together) --")
    fit = _fit_logit_report(y, X, "multivariate")
    if fit is not None:
        table, result = fit
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\n  pseudo R^2={result.prsquared:.4f}, log-likelihood={result.llf:.2f}, n={int(result.nobs)}")


def predictive_model(train_pool: pd.DataFrame, holdout: pd.DataFrame) -> None:
    print("\n=== Walk-forward-validated predictive model (sklearn LogisticRegression) ===")
    X_train = game_picks.game_feature_matrix(train_pool)
    y_train = train_pool["Home_Won"]
    dates_train = train_pool["date"]

    search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, LogisticRegression(max_iter=1000),
        {"C": config.GAME_PICK_LOGIT_C_GRID},
        config.GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_DATES, config.GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_DATES,
        scoring="neg_log_loss",
    )
    print(f"  Logistic Regression best_params={search.best_params_} cv_score={search.best_score_:.4f}")

    X_holdout = game_picks.game_feature_matrix(holdout)
    predicted_proba = search.predict_proba(X_holdout)[:, 1]
    result = ml_models.evaluate_classifier_predictions(holdout["Home_Won"], predicted_proba)
    heuristic_result = ml_models.evaluate_classifier_predictions(
        holdout["Home_Won"], holdout["home_win_probability"].clip(0, 1)
    )

    print(
        f"  Model (holdout):     log_loss={result['log_loss']:.4f}, brier={result['brier_score']:.4f}, "
        f"roc_auc={result['roc_auc']:.4f}, accuracy={result['accuracy']:.4f} (n={result['n']})"
    )
    print(f"  Naive baseline:      log_loss={result['baseline_log_loss']:.4f} (always predict base rate)")
    print(
        f"  home_win_probability heuristic: log_loss={heuristic_result['log_loss']:.4f}, "
        f"brier={heuristic_result['brier_score']:.4f}, roc_auc={heuristic_result['roc_auc']:.4f}"
    )

    # Save gate (2026-08-24 policy: "our goal is to get more and more
    # accurate, so as long as it beats our current model, save it" - the
    # naive always-predict-base-rate baseline is still printed above for
    # context, but is no longer a REQUIRED bar to clear. "Our current
    # model" here is the home_win_probability heuristic - the only thing
    # actually live for this signal today, since no artifact has ever
    # been saved to GAME_PICK_WIN_PROBABILITY_MODEL_PATH.
    beats_heuristic = result["log_loss"] < heuristic_result["log_loss"]
    if beats_heuristic:
        ml_models.save_model_bundle(
            search.best_estimator_,
            config.GAME_PICK_WIN_PROBABILITY_MODEL_PATH,
            model_type="game_pick_win_probability",
            model_version=config.GAME_PICK_MODEL_VERSION,
            feature_columns=list(game_picks.GAME_PICK_FEATURE_COLUMNS),
            training_data_start=train_pool["date"].min(),
            training_data_cutoff=train_pool["date"].max(),
            hyperparameters=dict(search.best_params_),
            calibration_method=None,
            validation_summary={
                "holdout_log_loss": result["log_loss"],
                "holdout_brier_score": result["brier_score"],
                "holdout_roc_auc": result["roc_auc"],
                "holdout_n": result["n"],
                "heuristic_log_loss": heuristic_result["log_loss"],
            },
        )
        print(
            f"  -> SAVED bundle to {config.GAME_PICK_WIN_PROBABILITY_MODEL_PATH} "
            f"(beats the home_win_probability heuristic - artifact only, NOT wired into live picks)"
        )
    else:
        print("  -> NOT saved (does not beat the home_win_probability heuristic - reported honestly)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Trim the game pick log to the most recent N dates instead of the full persisted "
             "history - each date recomputes the full pipeline, which is expensive. Full history "
             "is the default (None), matching train_hitter_hit_model.py's own --days flag.",
    )
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    print(f"Assembling game pick log (season={season}, days={args.days or 'all'})...")
    rows = game_picks_backtest.assemble_game_pick_log(args.raw_dir, season=season, days=args.days)

    if rows.empty:
        print("No game pick log rows assembled - nothing to fit.")
        return

    significance_report(rows)

    train_pool, holdout, dates = _split_holdout(rows, config.ML_FINAL_HOLDOUT_DATES)
    print(f"\n{len(dates)} distinct dates, {len(train_pool)} train-pool rows, {len(holdout)} holdout rows")
    if holdout.empty or train_pool.empty:
        print("Not enough distinct dates for a real holdout split - skipping the predictive model.")
        return

    predictive_model(train_pool, holdout)


if __name__ == "__main__":
    main()
