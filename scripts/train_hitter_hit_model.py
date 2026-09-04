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

2. **Nested rolling-origin validated predictive model**
   (mlb_metrics.model_validation): outer folds for honest reporting;
   inner folds select model family / hyperparameters / calibration /
   one-vs-two-pick policy. Aggregate outer-fold metrics replace
   repeatedly inspecting the same final 20-date holdout. An optional
   freeze period (config.NESTED_VALIDATION_FREEZE_DATES) is never used for
   selection - do not peek at it during development. The winner may be
   saved to config.HITTER_HIT_PROBABILITY_MODEL_PATH when nested-outer
   log_loss beats Game_Hit_Probability.

**Scope**: this fits and reports. The saved model artifact's live wiring
is unchanged by this script (see dfs_ml / predictions). Nested validation
does not alter inference code paths.

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

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_metrics import config, dfs_backtest, dfs_ml, ml_models, model_validation


def _split_holdout(df: pd.DataFrame, holdout_dates: int):
    """Legacy helper retained for scripts that still peek at a final block.

    Prefer model_validation.freeze_tail_dates / nested folds for new work.
    """
    dates = sorted(df["date"].unique())
    if len(dates) <= holdout_dates:
        return df.iloc[0:0], df, dates
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


def predictive_model(rows: pd.DataFrame) -> None:
    """Nested rolling-origin validation (outer/inner), then optional save.

    Model family, hyperparameters, calibration, and one-vs-two-pick policy
    are selected on **inner** folds only. Each outer test block is scored
    once after selection. Aggregate outer-fold metrics replace a single
    final 20-date holdout as the research score. Live serving is unchanged.
    """
    print("\n=== Nested rolling-origin validation (family / calib / policy on inner folds) ===")
    feature_columns = list(dfs_ml.HITTER_FEATURE_COLUMNS)
    # Need numeric feature columns present on the frame.
    for col in feature_columns:
        if col not in rows.columns:
            rows = rows.copy()
            rows[col] = 0.0

    candidates = model_validation.default_hitter_hit_candidates(
        feature_columns, include_full_grids=False,
    )
    nested_config = model_validation.NestedValidationConfig(
        outer_min_train_dates=config.NESTED_VALIDATION_OUTER_MIN_TRAIN_DATES,
        outer_test_block_dates=config.NESTED_VALIDATION_OUTER_TEST_BLOCK_DATES,
        inner_min_train_dates=config.NESTED_VALIDATION_INNER_MIN_TRAIN_DATES,
        inner_test_block_dates=config.NESTED_VALIDATION_INNER_TEST_BLOCK_DATES,
        freeze_dates=config.NESTED_VALIDATION_FREEZE_DATES,
        random_seed=config.NESTED_VALIDATION_RANDOM_SEED,
        bootstrap_samples=min(200, config.NESTED_VALIDATION_BOOTSTRAP_SAMPLES),
        report_dir=config.NESTED_VALIDATION_REPORT_DIR,
    )
    report = model_validation.run_nested_classifier_validation(
        rows,
        candidates,
        nested_config,
        date_col="date",
        label_col="Got_Hit",
        at_bats_col="Total_PA",
        key_cols=("key_mlbam", "game_pk"),
        champion_probability_col="Game_Hit_Probability",
    )
    model_validation.ensure_reports_dir_readme(nested_config.report_dir)
    report_path = model_validation.write_validation_report(
        report, nested_config.report_dir, filename="hitter_hit_nested_validation.json",
    )
    print(f"  Wrote {report_path} (status={report.get('status')})")
    if report.get("freeze_dates"):
        print(f"  Freeze dates (do not inspect during development): {report['freeze_dates']}")

    aggregate = report.get("aggregate") or {}
    if report.get("status") != "ok" or not report.get("outer_folds"):
        print("  Insufficient history for nested folds - skipping model save.")
        return

    print(
        f"  Aggregate outer folds (n={aggregate.get('n_outer_folds')}): "
        f"log_loss={aggregate.get('log_loss')}, brier={aggregate.get('brier_score')}, "
        f"roc_auc={aggregate.get('roc_auc')}, base_rate={aggregate.get('base_rate')}"
    )
    print(
        f"  Policy: top_one_advance={aggregate.get('top_one_advance_rate')}, "
        f"two_pick_survival={aggregate.get('two_pick_survival_rate')}, "
        f"coverage={aggregate.get('coverage')}, no_game_rate={aggregate.get('no_game_rate')}"
    )
    if report.get("bootstrap_comparisons"):
        boot = report["bootstrap_comparisons"][0]
        print(
            f"  Paired bootstrap challenger-champion {boot.get('metric')} diff="
            f"{boot.get('point_difference')} CI=[{boot.get('ci_low')}, {boot.get('ci_high')}]"
        )

    # Majority-vote selected candidate across outer folds, then refit on
    # all non-freeze dates for the optional artifact (still not live-wired
    # beyond whatever already loads HITTER_HIT_PROBABILITY_MODEL_PATH).
    selected_names = [f.get("selected_candidate") for f in report["outer_folds"] if f.get("selected_candidate")]
    if not selected_names:
        print("  No outer-fold selections - skipping save.")
        return
    winner_name = max(set(selected_names), key=selected_names.count)
    winner = next((c for c in candidates if c.name == winner_name), None)
    if winner is None:
        print(f"  Could not resolve winning candidate {winner_name!r} - skipping save.")
        return
    print(f"  Majority-vote selected configuration: {winner_name}")

    active_dates, _freeze = model_validation.freeze_tail_dates(
        rows["date"], nested_config.freeze_dates,
    )
    train_pool = rows[rows["date"].isin(active_dates)].copy()
    feats = [c for c in winner.feature_columns if c in train_pool.columns]
    y_train = train_pool["Got_Hit"].astype(float)
    cal_min, cal_block = nested_config.calibration_split_sizes()

    # Heuristic baseline on the same outer-test date blocks.
    heuristic_losses = []
    for fold in report["outer_folds"]:
        start, end = fold["test_start"], fold["test_end"]
        mask = (rows["date"].astype(str) >= start) & (rows["date"].astype(str) <= end)
        block = rows.loc[mask]
        if block.empty or "Game_Hit_Probability" not in block.columns:
            continue
        heur = ml_models.evaluate_classifier_predictions(
            block["Got_Hit"], block["Game_Hit_Probability"].clip(0, 1),
        )
        heuristic_losses.append(heur["log_loss"])
    heuristic_log_loss = float(np.nanmean(heuristic_losses)) if heuristic_losses else float("nan")
    model_log_loss = aggregate.get("log_loss", float("nan"))
    print(f"  Nested-outer model log_loss={model_log_loss}, heuristic log_loss={heuristic_log_loss}")

    beats_heuristic = (
        model_log_loss == model_log_loss
        and heuristic_log_loss == heuristic_log_loss
        and model_log_loss < heuristic_log_loss
    )
    if beats_heuristic:
        # Live predict_hitter_hit_probability expects raw (unz-scored)
        # HITTER_FEATURE_COLUMNS - refit without the nested-selection
        # StandardizePreprocessor so serving stays unchanged.
        raw_X = train_pool[feats].astype(float).fillna(0)
        raw_fitted = model_validation.fit_candidate_estimator(
            winner, raw_X, y_train, train_pool["date"],
            min_train_dates=cal_min, test_block_dates=cal_block,
        )
        ml_models.save_model_bundle(
            raw_fitted,
            config.HITTER_HIT_PROBABILITY_MODEL_PATH,
            model_type="hitter_hit_probability",
            model_version=config.HITTER_MODEL_VERSION,
            feature_columns=list(feats),
            training_data_start=train_pool["date"].min(),
            training_data_cutoff=train_pool["date"].max(),
            hyperparameters=dict(winner.hyperparameters),
            calibration_method=winner.calibration_method,
            validation_summary={
                "nested_outer_log_loss": model_log_loss,
                "nested_outer_brier_score": aggregate.get("brier_score"),
                "nested_outer_roc_auc": aggregate.get("roc_auc"),
                "nested_outer_n_folds": aggregate.get("n_outer_folds"),
                "heuristic_log_loss": heuristic_log_loss,
                "selected_candidate": winner_name,
                "validation_report": report_path,
            },
        )
        print(
            f"  -> SAVED bundle to {config.HITTER_HIT_PROBABILITY_MODEL_PATH} "
            f"(beats Game_Hit_Probability on nested outer folds - artifact only)"
        )
    else:
        print("  -> NOT saved (does not beat the Game_Hit_Probability heuristic on nested outer folds)")


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
    predictive_model(rows)


if __name__ == "__main__":
    main()
