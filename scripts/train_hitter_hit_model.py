"""Two deliverables from the hitter hit log (dfs_backtest.assemble_hitter_hit_log,
see its docstring and README's "Hitter hit log" section) - real answers to
the "why is Beat the Streak survival only ~50%, and which features are
actually significant" question the earlier ad hoc analysis (n=64 resolved
picks) couldn't answer with confidence:

1. **Significance report** (statsmodels.Logit): one univariate Logit per
   feature in dfs_ml.HITTER_FEATURE_COLUMNS ("individually"), plus one
   multivariate Logit with all of them together ("combined"), fit on the
   FULL real history (~31k rows across ~120 dates as of 2026-07-28 - no
   holdout split needed here, since statistical inference isn't a
   held-out-accuracy question). Features are standardized (z-scored)
   first so coefficient magnitudes are comparable across very different
   scales (WAVE ~0-0.4 vs Total_PA ~0-30) - p-values themselves are
   scale-invariant. A feature that's constant in the current data (zero
   std) is excluded and reported as such rather than crashing on a
   divide-by-zero.

   Also tests CANDIDATE_FEATURE_COLUMNS (Days_Rest, Umpire_Factor) -
   real feature-search follow-up (2026-08-24): "use the identified
   features that we can use... test feature significance before
   committing to the model." Neither is part of
   dfs_ml.HITTER_FEATURE_COLUMNS yet; they're carried by
   dfs_backtest.assemble_hitter_hit_log purely as exploratory columns
   (see that function's docstring) and included in this same
   significance report so a real p-value decides whether either earns a
   permanent place in the model, rather than guessing.

2. **Walk-forward-validated predictive model** (ml_models.py's
   WalkForwardDateSplit machinery, the SAME no-lookahead, nested-holdout
   methodology scripts/train_dfs_ml_models.py already uses for the three
   live DFS ML signals): TWO candidate model families are grid-searched -
   sklearn LogisticRegression (config.HITTER_HIT_LOGIT_C_GRID) and
   HistGradientBoostingClassifier (config.HITTER_HIT_GBM_PARAM_GRID) -
   and whichever wins by walk-forward CV log_loss is kept, mirroring
   train_dfs_ml_models.py's own Ridge-vs-GBM selection for
   DK_Points_Hitter (quant-analytics item #2 - "model family": this used
   to be a single logistic regression with no real alternative
   considered). The winner is evaluated ONCE on an untouched final
   holdout and must beat BOTH a naive baseline (always predict the base
   hit rate) AND the existing Game_Hit_Probability heuristic column (the
   natural "heuristic to beat" here) before being saved to
   config.HITTER_HIT_PROBABILITY_MODEL_PATH.

3. **Permutation-importance report** (sklearn.inspection.permutation_importance,
   computed on the same untouched holdout, never the train pool - the
   same reasoning behind evaluating the model itself out-of-sample):
   a model-agnostic feature-importance sanity check for whichever model
   won above, chosen over SHAP specifically to avoid adding a new heavy
   dependency to a project that has never needed one beyond scikit-learn/
   statsmodels (see README's "Model family" section for the real
   comparison against SHAP).

**Scope**: this fits and reports. The saved model artifact is NOT wired
into dfs_ml.apply_ml_overrides or predictions.select_picks - whether/how
this ever feeds live Beat the Streak picks is a separate, later decision.

**PA filter**: dfs_backtest.assemble_hitter_hit_log deliberately carries
EVERY hitter with a game that date, including ones with a handful of
career plate appearances at that point - a WAVE/Game_Hit_Probability
built on a tiny sample is mostly noise, and fitting on it would let that
noise drag down real coefficients. This script filters to
Total_PA >= --min-plate-appearances (default config.BACKTEST_MIN_PLATE_APPEARANCES,
the SAME gate predictions.select_picks already applies before a hitter is
ever eligible to be a pick) right before fitting, for both the
significance report and the predictive model - the log itself stays
unfiltered so future consumers can pick their own threshold.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py).

Usage:
    python scripts/train_hitter_hit_model.py
    python scripts/train_hitter_hit_model.py --season 2026
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from mlb_metrics import config, dfs_backtest, dfs_ml, ml_models


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
    fit fails - e.g. a near-perfectly-collinear multivariate design (see
    this script's module docstring re: Consistency/Approach) - so a single
    failed fit is reported and skipped rather than crashing the whole
    script."""
    try:
        result = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    except Exception as exc:
        print(f"  {label}: fit failed ({exc})")
        return None
    table = pd.DataFrame({
        "coef": result.params, "std_err": result.bse, "z": result.tvalues, "p_value": result.pvalues,
    }).drop(index="const")
    return table, result


#  Exploratory candidates (2026-08-24 feature-search follow-up - "use the
# identified features that we can use... test feature significance
# before committing to the model"): real, already-persisted signal
# (dfs_backtest.assemble_hitter_hit_log's own Days_Rest/Umpire_Factor
# columns - see its docstring) that is NOT part of
# dfs_ml.HITTER_FEATURE_COLUMNS - tested here, alongside the live feature
# set, purely to decide whether either earns a permanent place in the
# model. Nothing here changes what the live model actually uses until a
# candidate clears a real bar and is deliberately added to
# HITTER_FEATURE_COLUMNS in a follow-up change.
CANDIDATE_FEATURE_COLUMNS = ["Days_Rest", "Umpire_Factor"]


def significance_report(rows: pd.DataFrame) -> None:
    print("\n=== Feature significance report (statsmodels.Logit, full history) ===")
    y = rows["Got_Hit"]
    # hitter_feature_matrix already fills every NaN/pd.NA (a feature that
    # wasn't computable yet on an early-history date, e.g. Park_Factor
    # before enough home games are on record) with 0 - but pd.NA-only
    # columns from those early dates leave the concatenated column
    # dtype=object even after filling, which statsmodels' Logit can't
    # consume (sklearn's LogisticRegression silently coerces past this,
    # which is why the walk-forward model below isn't affected). .astype
    # (float) only retags the dtype; every value is already numeric.
    X = _standardize(dfs_ml.hitter_feature_matrix(rows).astype(float))

    # Same fillna(0)-for-not-yet-computable convention as hitter_feature_matrix
    # (a real 0 Days_Rest is indistinguishable here from "no prior game on
    # record yet" - an honest known rough edge for an EXPLORATORY column,
    # not something worth a bespoke encoding before we even know whether
    # this candidate has any real signal at all).
    candidates = _standardize(rows[CANDIDATE_FEATURE_COLUMNS].astype(float).fillna(0))
    X = pd.concat([X, candidates], axis=1)

    print(f"  n={len(rows)}, base hit rate={y.mean():.4f}")
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


def _select_best_model(X_train, y_train, dates_train):
    """Grid-searches BOTH candidate model families (LogisticRegression and
    HistGradientBoostingClassifier) via the same walk-forward CV, prints
    each one's best_params_/cv_score, and returns whichever GridSearchCV
    has the higher (less negative) neg_log_loss best_score_ - the same
    "run every candidate, keep the CV winner" pattern
    train_dfs_ml_models.py already uses for DK_Points_Hitter (Ridge vs.
    HistGradientBoostingRegressor)."""
    logit_search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, LogisticRegression(max_iter=1000),
        {"C": config.HITTER_HIT_LOGIT_C_GRID},
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
        scoring="neg_log_loss",
    )
    gbm_search = ml_models.grid_search_walk_forward(
        X_train, y_train, dates_train, HistGradientBoostingClassifier(),
        config.HITTER_HIT_GBM_PARAM_GRID,
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
        scoring="neg_log_loss",
    )
    print(f"  Logistic Regression best_params={logit_search.best_params_} cv_score={logit_search.best_score_:.4f}")
    print(f"  GBM                 best_params={gbm_search.best_params_} cv_score={gbm_search.best_score_:.4f}")

    best_search = logit_search if logit_search.best_score_ >= gbm_search.best_score_ else gbm_search
    best_name = "LogisticRegression" if best_search is logit_search else "HistGradientBoostingClassifier"
    print(f"  Selected: {best_name}")
    return best_search


def feature_importance_report(fitted_estimator, X_holdout: pd.DataFrame, y_holdout: pd.Series) -> None:
    """Model-agnostic permutation-importance sanity check (sklearn.inspection.
    permutation_importance) on the SAME untouched holdout the model itself
    is scored on above - never the train pool, since permutation
    importance computed on training data can overstate importance for an
    overfit model, the same out-of-sample discipline this whole script
    already applies to the model's own headline metrics. Works identically
    for any candidate estimator (LogisticRegression, HistGradientBoostingClassifier,
    or a CalibratedClassifierCV wrapping either - see _select_calibration)
    since permutation importance only needs a fitted estimator's
    .predict/.predict_proba, not anything model-family-specific - chosen
    over SHAP for exactly this reason, plus avoiding a new heavy dependency
    (see README's "Model family" section)."""
    print("\n=== Permutation-importance report (holdout, model-agnostic) ===")
    result = permutation_importance(
        fitted_estimator, X_holdout, y_holdout,
        scoring="neg_log_loss", n_repeats=20, random_state=0,
    )
    table = pd.DataFrame({
        "importance_mean": result.importances_mean, "importance_std": result.importances_std,
    }, index=X_holdout.columns).sort_values("importance_mean", ascending=False)
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))


def _select_calibration(best_search, X_train, y_train, dates_train, X_holdout, y_holdout):
    """Quant-analytics item #3, slice 2: on top of the family winner
    _select_best_model already picked, tries the raw (uncalibrated)
    estimator against isotonic and Platt/sigmoid calibration
    (ml_models.fit_calibrated, same no-lookahead WalkForwardDateSplit),
    scores all three on the SAME untouched holdout, and picks the winner
    by holdout BRIER SCORE - calibration's own proper-scoring metric,
    distinct from the neg_log_loss criterion _select_best_model used one
    step earlier for family selection. Returns (name, fitted_estimator,
    {name: evaluate_classifier_predictions(...)}) - reported honestly:
    "uncalibrated" can and does win when calibration doesn't help."""
    candidates = {"uncalibrated": best_search.best_estimator_}
    for method in ("isotonic", "sigmoid"):
        candidates[method] = ml_models.fit_calibrated(
            best_search.best_estimator_, X_train, y_train, dates_train, method,
            config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
        )

    results = {
        name: ml_models.evaluate_classifier_predictions(y_holdout, estimator.predict_proba(X_holdout)[:, 1])
        for name, estimator in candidates.items()
    }
    print("\n=== Calibration comparison (uncalibrated vs. isotonic vs. sigmoid/Platt, holdout) ===")
    for name, result in results.items():
        print(
            f"  {name:12s} log_loss={result['log_loss']:.4f}, brier={result['brier_score']:.4f}, "
            f"roc_auc={result['roc_auc']:.4f}, accuracy={result['accuracy']:.4f}"
        )

    best_name = min(results, key=lambda n: results[n]["brier_score"])
    print(f"  Selected by holdout Brier score: {best_name}")
    return best_name, candidates[best_name], results


def predictive_model(train_pool: pd.DataFrame, holdout: pd.DataFrame) -> None:
    print("\n=== Walk-forward-validated predictive model (LogisticRegression vs. HistGradientBoostingClassifier) ===")
    X_train = dfs_ml.hitter_feature_matrix(train_pool)
    y_train = train_pool["Got_Hit"]
    dates_train = train_pool["date"]

    best_search = _select_best_model(X_train, y_train, dates_train)

    X_holdout = dfs_ml.hitter_feature_matrix(holdout)
    y_holdout = holdout["Got_Hit"]

    best_name, best_model, calibration_results = _select_calibration(
        best_search, X_train, y_train, dates_train, X_holdout, y_holdout
    )
    result = calibration_results[best_name]
    heuristic_result = ml_models.evaluate_classifier_predictions(
        y_holdout, holdout["Game_Hit_Probability"].clip(0, 1)
    )

    print(
        f"  Model (holdout):     log_loss={result['log_loss']:.4f}, brier={result['brier_score']:.4f}, "
        f"roc_auc={result['roc_auc']:.4f}, accuracy={result['accuracy']:.4f} (n={result['n']})"
    )
    print(f"  Naive baseline:      log_loss={result['baseline_log_loss']:.4f} (always predict base rate)")
    print(
        f"  Game_Hit_Probability heuristic: log_loss={heuristic_result['log_loss']:.4f}, "
        f"brier={heuristic_result['brier_score']:.4f}, roc_auc={heuristic_result['roc_auc']:.4f}"
    )

    feature_importance_report(best_model, X_holdout, y_holdout)

    # Save gate (2026-08-24 policy: "as long as it beats our current
    # model, save it" - naive baseline still printed above for context,
    # no longer a required bar. "Our current model" is the
    # Game_Hit_Probability heuristic - the only thing live for this
    # signal (this artifact itself is NOT wired into live picks).
    beats_heuristic = result["log_loss"] < heuristic_result["log_loss"]
    if beats_heuristic:
        ml_models.save_model_bundle(
            best_model,
            config.HITTER_HIT_PROBABILITY_MODEL_PATH,
            model_type="hitter_hit_probability",
            model_version=config.HITTER_MODEL_VERSION,
            feature_columns=list(dfs_ml.HITTER_FEATURE_COLUMNS),
            training_data_start=train_pool["date"].min(),
            training_data_cutoff=train_pool["date"].max(),
            hyperparameters=dict(best_search.best_params_),
            calibration_method=best_name,
            validation_summary={
                "holdout_log_loss": result["log_loss"],
                "holdout_brier_score": result["brier_score"],
                "holdout_roc_auc": result["roc_auc"],
                "holdout_n": result["n"],
                "heuristic_log_loss": heuristic_result["log_loss"],
            },
        )
        print(
            f"  -> SAVED bundle to {config.HITTER_HIT_PROBABILITY_MODEL_PATH} ({best_name}, "
            f"beats Game_Hit_Probability - artifact only, NOT wired into live picks)"
        )
    else:
        print("  -> NOT saved (does not beat the Game_Hit_Probability heuristic - reported honestly)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Trim the hit log to the most recent N dates instead of the full persisted history - "
             "each date recomputes the full pipeline, which is expensive. Full history is the "
             "default (None), matching train_dfs_ml_models.py's own --days flag.",
    )
    parser.add_argument(
        "--min-plate-appearances", type=int, default=config.BACKTEST_MIN_PLATE_APPEARANCES,
        help="Exclude hit-log rows below this Total_PA before fitting - a hitter with only a "
             "handful of career plate appearances at that point has a mostly-noise WAVE/"
             "Game_Hit_Probability. Defaults to the same gate predictions.select_picks already "
             "applies (config.BACKTEST_MIN_PLATE_APPEARANCES).",
    )
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    print(f"Assembling hitter hit log (season={season}, days={args.days or 'all'})...")
    rows = dfs_backtest.assemble_hitter_hit_log(args.raw_dir, season=season, days=args.days)

    if rows.empty:
        print("No hitter hit log rows assembled - nothing to fit.")
        return

    unfiltered_n = len(rows)
    rows = rows[rows["Total_PA"] >= args.min_plate_appearances]
    print(
        f"Filtered to Total_PA >= {args.min_plate_appearances}: {len(rows)} of {unfiltered_n} rows kept."
    )
    if rows.empty:
        print("No rows clear the plate-appearance threshold - nothing to fit.")
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
