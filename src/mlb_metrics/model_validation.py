"""Nested rolling-origin validation for date-based baseball models.

Replaces the research pattern of repeatedly inspecting the same recent-date
holdout for model family, calibration, feature additions, shortlist
behavior, and thresholds. Outer folds provide honest out-of-sample
scores; inner folds (strictly inside each outer training block) select
configuration; the outer test block is evaluated once after selection.

## Freeze period (optional, default off)

`NestedValidationConfig.freeze_dates` reserves the most-recent N distinct
dates AFTER all fold construction. Those dates are never used for:

- outer or inner training
- outer or inner testing
- model / calibration / feature / threshold / shortlist / pick-policy
  selection

They exist only as an optional final untouched check after research has
stopped. **Do not repeatedly inspect the freeze period during development**
- treating it as another tuning dial reintroduces the single-holdout
leakage this module exists to prevent.

## Live behavior

This module does not change prediction serving, model loading, or
pipeline.run(). Training scripts may call it for research reports and
artifact save gates; inference paths do not import it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

from mlb_metrics import config, evaluation, ml_models


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DateFold:
    """One rolling-origin fold with explicit date boundaries."""

    fold_id: int
    train_dates: tuple
    test_dates: tuple

    @property
    def train_start(self):
        return self.train_dates[0] if self.train_dates else None

    @property
    def train_end(self):
        return self.train_dates[-1] if self.train_dates else None

    @property
    def test_start(self):
        return self.test_dates[0] if self.test_dates else None

    @property
    def test_end(self):
        return self.test_dates[-1] if self.test_dates else None

    def assert_no_overlap(self) -> None:
        overlap = set(self.train_dates) & set(self.test_dates)
        if overlap:
            raise AssertionError(f"Fold {self.fold_id} has train/test date overlap: {sorted(overlap)[:5]}")


@dataclass(frozen=True)
class NestedFold:
    """Outer fold plus the inner folds built only on its training dates."""

    outer: DateFold
    inner_folds: tuple[DateFold, ...]


def unique_sorted_dates(dates: Iterable) -> list:
    return sorted(pd.Series(list(dates)).dropna().unique())


def freeze_tail_dates(all_dates: Sequence, freeze_dates: int) -> tuple[list, list]:
    """Split sorted dates into (active_for_nested_cv, freeze_tail).

    The freeze tail is the most recent `freeze_dates` distinct dates and is
    excluded from every nested fold. When freeze_dates=0, freeze_tail is [].
    """
    dates = unique_sorted_dates(all_dates)
    freeze_dates = max(0, int(freeze_dates))
    if freeze_dates <= 0:
        return dates, []
    if freeze_dates >= len(dates):
        return [], dates
    return dates[:-freeze_dates], dates[-freeze_dates:]


def build_rolling_origin_folds(
    dates: Sequence,
    min_train_dates: int,
    test_block_dates: int,
) -> list[DateFold]:
    """Outer- or inner-style rolling-origin folds over `dates`.

    Fold k trains on the earliest dates strictly before its test block and
    tests on the next `test_block_dates` dates. Returns [] when history is
    insufficient for even one fold (fewer than min_train_dates + 1 dates,
    or empty test blocks).
    """
    unique_dates = unique_sorted_dates(dates)
    min_train_dates = int(min_train_dates)
    test_block_dates = int(test_block_dates)
    if min_train_dates < 1 or test_block_dates < 1:
        return []
    if len(unique_dates) < min_train_dates + 1:
        return []

    folds: list[DateFold] = []
    cursor = min_train_dates
    fold_id = 0
    while cursor < len(unique_dates):
        train = tuple(unique_dates[:cursor])
        test = tuple(unique_dates[cursor:cursor + test_block_dates])
        if train and test:
            fold = DateFold(fold_id=fold_id, train_dates=train, test_dates=test)
            fold.assert_no_overlap()
            # Chronology: every train date precedes every test date.
            if max(train) >= min(test):
                raise AssertionError(
                    f"Fold {fold_id} is not chronological: max(train)={max(train)} "
                    f">= min(test)={min(test)}"
                )
            folds.append(fold)
            fold_id += 1
        cursor += test_block_dates
    return folds


def build_nested_folds(
    dates: Sequence,
    *,
    outer_min_train_dates: int,
    outer_test_block_dates: int,
    inner_min_train_dates: int,
    inner_test_block_dates: int,
    freeze_dates: int = 0,
) -> tuple[list[NestedFold], list]:
    """Build outer folds on active dates; inner folds on each outer train.

    Returns (nested_folds, freeze_tail_dates). Insufficient history yields
    an empty nested_folds list (never raises).
    """
    active, freeze_tail = freeze_tail_dates(dates, freeze_dates)
    outer_folds = build_rolling_origin_folds(
        active, outer_min_train_dates, outer_test_block_dates,
    )
    nested: list[NestedFold] = []
    for outer in outer_folds:
        inner = build_rolling_origin_folds(
            outer.train_dates, inner_min_train_dates, inner_test_block_dates,
        )
        # An outer fold with no usable inner folds cannot select a config
        # honestly - skip it rather than silently falling back to a single
        # holdout peek at the outer test block.
        if not inner:
            continue
        for inner_fold in inner:
            inner_fold.assert_no_overlap()
            if set(inner_fold.test_dates) & set(outer.test_dates):
                raise AssertionError("Inner test dates leaked into outer test dates")
            if not set(inner_fold.train_dates).issubset(set(outer.train_dates)):
                raise AssertionError("Inner train dates escaped the outer train block")
            if not set(inner_fold.test_dates).issubset(set(outer.train_dates)):
                raise AssertionError("Inner test dates escaped the outer train block")
        nested.append(NestedFold(outer=outer, inner_folds=tuple(inner)))
    return nested, freeze_tail


# ---------------------------------------------------------------------------
# Preprocessing (fit on training fold only)
# ---------------------------------------------------------------------------


class StandardizePreprocessor:
    """Column-wise z-score fit exclusively on a training matrix.

    Columns with zero training variance are dropped (not filled with a
    fabricated scale). Transform on unseen data uses the training mean/std
    only - never refit on test rows.
    """

    def __init__(self):
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> StandardizePreprocessor:
        X = pd.DataFrame(X).astype(float)
        std = X.std(axis=0)
        keep = std[std > 0].index.tolist()
        self.columns_ = keep
        if not keep:
            self.mean_ = pd.Series(dtype=float)
            self.std_ = pd.Series(dtype=float)
            return self
        subset = X[keep]
        self.mean_ = subset.mean(axis=0)
        self.std_ = subset.std(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardizePreprocessor.transform called before fit")
        X = pd.DataFrame(X).astype(float)
        if not self.columns_:
            return pd.DataFrame(index=X.index)
        subset = X.reindex(columns=self.columns_)
        return (subset - self.mean_) / self.std_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class ImputeStandardizePreprocessor:
    """Train-only median imputation + missingness indicators + z-score.

    Fit exclusively on the training fold:
    - per-column training median (league-prior style constant fill)
    - a binary ``{col}__missing`` indicator for every column that had any
      training missingness (indicators stay raw 0/1; not z-scored)
    - then standardize the imputed source columns with positive training variance

    Test rows never update medians, means, or stds.
    """

    def __init__(self):
        self.median_: pd.Series | None = None
        self.missing_indicator_cols_: list[str] = []
        self.source_columns_: list[str] = []
        self.scaler_ = StandardizePreprocessor()

    def fit(self, X: pd.DataFrame) -> ImputeStandardizePreprocessor:
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        self.source_columns_ = list(X.columns)
        self.median_ = X.median(axis=0, skipna=True).fillna(0.0)
        # Columns with any train missingness get an indicator.
        self.missing_indicator_cols_ = [
            c for c in self.source_columns_ if X[c].isna().any()
        ]
        filled = X.fillna(self.median_)
        self.scaler_.fit(filled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None:
            raise RuntimeError("ImputeStandardizePreprocessor.transform called before fit")
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        X = X.reindex(columns=self.source_columns_)
        filled = X.fillna(self.median_)
        scaled = self.scaler_.transform(filled)
        for c in self.missing_indicator_cols_:
            scaled[f"{c}__missing"] = X[c].isna().astype(float).to_numpy()
        return scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def assert_preprocessor_fit_only_on_train(preprocessor: StandardizePreprocessor, X_train: pd.DataFrame) -> None:
    """Test helper: refitting on the same train frame reproduces state."""
    other = StandardizePreprocessor().fit(X_train)
    assert preprocessor.columns_ == other.columns_
    if preprocessor.columns_:
        pd.testing.assert_series_equal(preprocessor.mean_, other.mean_)
        pd.testing.assert_series_equal(preprocessor.std_, other.std_)


def assert_imputer_fit_only_on_train(
    preprocessor: ImputeStandardizePreprocessor,
    X_train: pd.DataFrame,
) -> None:
    """Test helper: refitting on the same train frame reproduces imputer state."""
    other = ImputeStandardizePreprocessor().fit(X_train)
    assert preprocessor.source_columns_ == other.source_columns_
    assert preprocessor.missing_indicator_cols_ == other.missing_indicator_cols_
    pd.testing.assert_series_equal(preprocessor.median_, other.median_)
    X = pd.DataFrame(X_train).apply(pd.to_numeric, errors="coerce").reindex(
        columns=other.source_columns_
    )
    assert_preprocessor_fit_only_on_train(preprocessor.scaler_, X.fillna(other.median_))


def _impute_frame_for_assert(
    X_train: pd.DataFrame,
    fitted: ImputeStandardizePreprocessor,
) -> pd.DataFrame:
    X = pd.DataFrame(X_train).apply(pd.to_numeric, errors="coerce").reindex(columns=fitted.source_columns_)
    return X.fillna(fitted.median_)


# ---------------------------------------------------------------------------
# Candidates and scoring
# ---------------------------------------------------------------------------


@dataclass
class ModelCandidate:
    """One fully-specified configuration eligible for inner-fold selection.

    All of these fields may differ across candidates; only inner-fold
    scores decide the winner for an outer training block.
    """

    name: str
    estimator: BaseEstimator
    feature_columns: list[str]
    calibration_method: str | None = None  # None | "isotonic" | "sigmoid"
    threshold: float = 0.5
    shortlist_size: int | None = None
    n_picks: int = 1
    hyperparameters: dict = field(default_factory=dict)


def expand_candidate_grid(
    *,
    families: Sequence[dict],
    feature_subsets: Sequence[Sequence[str]],
    calibration_methods: Sequence[str | None] = (None,),
    thresholds: Sequence[float] = (0.5,),
    shortlist_sizes: Sequence[int | None] = (None,),
    n_picks_options: Sequence[int] = (1,),
) -> list[ModelCandidate]:
    """Cartesian product of family/param grids with policy/feature knobs.

    Each `families` entry: {"name": str, "estimator": estimator, "param_grid": dict}.
    """
    candidates: list[ModelCandidate] = []
    for family in families:
        base = family["estimator"]
        grid = family.get("param_grid") or {}
        param_list = list(ParameterGrid(grid)) if grid else [{}]
        for params in param_list:
            est = clone(base)
            if params:
                est.set_params(**params)
            for features in feature_subsets:
                for calib in calibration_methods:
                    for threshold in thresholds:
                        for shortlist in shortlist_sizes:
                            for n_picks in n_picks_options:
                                label = (
                                    f"{family['name']}|params={params}|feats={len(features)}|"
                                    f"calib={calib}|thr={threshold}|shortlist={shortlist}|picks={n_picks}"
                                )
                                candidates.append(ModelCandidate(
                                    name=label,
                                    estimator=est,
                                    feature_columns=list(features),
                                    calibration_method=calib,
                                    threshold=float(threshold),
                                    shortlist_size=shortlist,
                                    n_picks=int(n_picks),
                                    hyperparameters=dict(params),
                                ))
    return candidates


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def calibration_intercept_slope(actual: pd.Series, predicted_proba: pd.Series) -> dict:
    """Fit actual ~ logit(predicted): intercept and slope of the calibration line.

    Ideal calibration is intercept≈0, slope≈1. Returns NaNs when the fit is
    impossible (single-class labels, empty input).
    """
    y = pd.Series(actual).astype(float).reset_index(drop=True)
    p = pd.Series(predicted_proba).astype(float).reset_index(drop=True)
    if len(y) == 0 or y.nunique() < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(_logit(p.to_numpy()).reshape(-1, 1), y)
        return {
            "calibration_intercept": float(model.intercept_[0]),
            "calibration_slope": float(model.coef_[0][0]),
        }
    except Exception:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}


def classifier_probability_metrics(actual: pd.Series, predicted_proba: pd.Series) -> dict:
    """Row-level probability metrics plus calibration intercept/slope."""
    base = ml_models.evaluate_classifier_predictions(actual, predicted_proba)
    y = pd.Series(actual).astype(float)
    base["base_rate"] = float(y.mean()) if len(y) else float("nan")
    base.update(calibration_intercept_slope(y, predicted_proba))
    return base


def fit_candidate_estimator(
    candidate: ModelCandidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series,
    *,
    min_train_dates: int,
    test_block_dates: int,
) -> Any:
    """Fit base estimator (and optional walk-forward calibration) on train only."""
    estimator = clone(candidate.estimator)
    if candidate.calibration_method in (None, "uncalibrated"):
        estimator.fit(X_train, y_train)
        return estimator
    return ml_models.fit_calibrated(
        estimator, X_train, y_train, dates_train,
        method=candidate.calibration_method,
        min_train_dates=min_train_dates,
        test_block_dates=test_block_dates,
    )


def predict_proba_positive(estimator, X: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Estimator predict_proba must return a two-column matrix")
    return proba[:, 1]


def rows_for_dates(df: pd.DataFrame, dates: Sequence, date_col: str = "date") -> pd.DataFrame:
    date_set = set(dates)
    return df[df[date_col].isin(date_set)].copy()


def build_ranked_picks(
    scored: pd.DataFrame,
    *,
    date_col: str = "date",
    score_col: str = "predicted_probability",
    label_col: str = "Got_Hit",
    at_bats_col: str | None = "Total_PA",
    shortlist_size: int | None = None,
    n_picks: int = 1,
    key_cols: Sequence[str] = ("key_mlbam", "game_pk"),
) -> pd.DataFrame:
    """Top-n_picks per date from model scores (optional shortlist gate).

    When `shortlist_size` is set, only the top shortlist_size rows by score
    per date remain eligible before the final top-n_picks cut - a stand-in
    for legacy shortlist-then-rank policy search without calling live
    select_picks.
    """
    if scored.empty:
        return pd.DataFrame(columns=["date", "rank", "predicted_probability", "actual_hit", "at_bats", *key_cols])

    frames = []
    for date, group in scored.groupby(date_col, sort=True):
        g = group.sort_values(score_col, ascending=False)
        if shortlist_size is not None:
            g = g.head(int(shortlist_size))
        top = g.head(int(n_picks)).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top["date"] = date
        top["predicted_probability"] = top[score_col]
        top["actual_hit"] = top[label_col]
        if at_bats_col and at_bats_col in top.columns:
            top["at_bats"] = top[at_bats_col]
        else:
            top["at_bats"] = 1
        keep = ["date", "rank", "predicted_probability", "actual_hit", "at_bats"] + [
            c for c in key_cols if c in top.columns
        ]
        frames.append(top[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def policy_metrics_from_picks(picks: pd.DataFrame, n_candidate_dates: int, n_picks: int) -> dict:
    """Coverage / no-game / top-one / two-pick survival metrics."""
    k = max(1, int(n_picks))
    metrics = evaluation.selection_strategy_metrics(
        picks, k=k, n_candidate_dates=n_candidate_dates,
    )
    out = {
        "coverage": metrics.get("coverage_rate", float("nan")),
        "no_game_rate": metrics.get("no_game_rate", float("nan")),
        "top_one_hit_rate": metrics.get("top_1_advance_rate", float("nan")),
        "top_one_advance_rate": metrics.get("top_1_advance_rate", float("nan")),
        "n_candidate_dates": metrics.get("n_candidate_dates", n_candidate_dates),
    }
    if k >= 2:
        out["two_pick_survival_rate"] = metrics.get("top_2_survival_rate", float("nan"))
        out["two_pick_reset_rate"] = metrics.get("top_2_reset_rate", float("nan"))
    else:
        out["two_pick_survival_rate"] = float("nan")
        out["two_pick_reset_rate"] = float("nan")
    return out


def default_hitter_hit_candidates(
    feature_columns: Sequence[str] | None = None,
    *,
    include_full_grids: bool = False,
) -> list[ModelCandidate]:
    """Candidate list for hitter-hit nested validation.

    Default is a focused grid suitable for CI / routine training. Pass
    `include_full_grids=True` to expand the production logit/GBM grids ×
    calibration × one-vs-two-pick policy.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    if feature_columns is None:
        from mlb_metrics import dfs_ml
        feature_columns = list(dfs_ml.HITTER_FEATURE_COLUMNS)
    features = list(feature_columns)

    if include_full_grids:
        families = [
            {
                "name": "LogisticRegression",
                "estimator": LogisticRegression(max_iter=1000),
                "param_grid": {"C": list(config.HITTER_HIT_LOGIT_C_GRID)},
            },
            {
                "name": "HistGradientBoostingClassifier",
                "estimator": HistGradientBoostingClassifier(
                    random_state=config.NESTED_VALIDATION_RANDOM_SEED,
                ),
                "param_grid": dict(config.HITTER_HIT_GBM_PARAM_GRID),
            },
        ]
        calibration_methods: Sequence[str | None] = (None, "isotonic", "sigmoid")
        n_picks_options: Sequence[int] = (1, 2)
    else:
        families = [
            {
                "name": "LogisticRegression",
                "estimator": LogisticRegression(max_iter=1000),
                "param_grid": {"C": [0.1, 1.0]},
            },
            {
                "name": "HistGradientBoostingClassifier",
                "estimator": HistGradientBoostingClassifier(
                    max_depth=2, learning_rate=0.1, max_iter=50,
                    min_samples_leaf=50, random_state=config.NESTED_VALIDATION_RANDOM_SEED,
                ),
                "param_grid": {},
            },
        ]
        calibration_methods = (None, "sigmoid")
        n_picks_options = (1, 2)

    return expand_candidate_grid(
        families=families,
        feature_subsets=[features],
        calibration_methods=calibration_methods,
        thresholds=(0.5,),
        shortlist_sizes=(None,),
        n_picks_options=n_picks_options,
    )


# ---------------------------------------------------------------------------
# Nested validation runner
# ---------------------------------------------------------------------------


@dataclass
class NestedValidationConfig:
    outer_min_train_dates: int = config.NESTED_VALIDATION_OUTER_MIN_TRAIN_DATES
    outer_test_block_dates: int = config.NESTED_VALIDATION_OUTER_TEST_BLOCK_DATES
    inner_min_train_dates: int = config.NESTED_VALIDATION_INNER_MIN_TRAIN_DATES
    inner_test_block_dates: int = config.NESTED_VALIDATION_INNER_TEST_BLOCK_DATES
    freeze_dates: int = config.NESTED_VALIDATION_FREEZE_DATES
    random_seed: int = config.NESTED_VALIDATION_RANDOM_SEED
    bootstrap_samples: int = config.NESTED_VALIDATION_BOOTSTRAP_SAMPLES
    report_dir: str = config.NESTED_VALIDATION_REPORT_DIR
    primary_score: str = "log_loss"
    higher_is_better: bool = False
    calibration_min_train_dates: int | None = None
    calibration_test_block_dates: int | None = None

    def calibration_split_sizes(self) -> tuple[int, int]:
        min_train = self.calibration_min_train_dates or max(5, self.inner_min_train_dates // 2)
        block = self.calibration_test_block_dates or max(1, self.inner_test_block_dates)
        return int(min_train), int(block)


def _score_is_better(score: float, best: float, higher_is_better: bool) -> bool:
    if score != score:
        return False
    if best != best:
        return True
    return score > best if higher_is_better else score < best


def score_candidate_on_inner_folds(
    df: pd.DataFrame,
    candidate: ModelCandidate,
    nested: NestedFold,
    nested_config: NestedValidationConfig,
    *,
    date_col: str = "date",
    label_col: str = "Got_Hit",
) -> dict:
    """Mean primary score across inner folds (selection criterion only)."""
    scores = []
    for inner in nested.inner_folds:
        train = rows_for_dates(df, inner.train_dates, date_col)
        test = rows_for_dates(df, inner.test_dates, date_col)
        feats = [c for c in candidate.feature_columns if c in train.columns]
        if not feats or train.empty or test.empty:
            continue
        X_train_raw = train[feats]
        X_test_raw = test[feats]
        y_train = train[label_col].astype(float)
        y_test = test[label_col].astype(float)
        pre = StandardizePreprocessor().fit(X_train_raw)
        X_train = pre.transform(X_train_raw)
        X_test = pre.transform(X_test_raw)
        if X_train.shape[1] == 0:
            continue
        cal_min, cal_block = nested_config.calibration_split_sizes()
        try:
            estimator = fit_candidate_estimator(
                candidate, X_train, y_train, train[date_col],
                min_train_dates=cal_min, test_block_dates=cal_block,
            )
            proba = predict_proba_positive(estimator, X_test)
        except Exception:
            continue
        metrics = classifier_probability_metrics(y_test, proba)
        scores.append(metrics.get(nested_config.primary_score, float("nan")))
    if not scores:
        return {"mean_primary_score": float("nan"), "n_inner_folds_scored": 0}
    arr = np.asarray(scores, dtype=float)
    return {
        "mean_primary_score": float(np.nanmean(arr)),
        "n_inner_folds_scored": int(np.sum(~np.isnan(arr))),
    }


def select_candidate_on_inner_folds(
    df: pd.DataFrame,
    candidates: Sequence[ModelCandidate],
    nested: NestedFold,
    nested_config: NestedValidationConfig,
    **kwargs,
) -> tuple[ModelCandidate | None, list[dict]]:
    """Pick the single best candidate using inner folds only."""
    scored = []
    best = None
    best_score = float("nan")
    for candidate in candidates:
        result = score_candidate_on_inner_folds(df, candidate, nested, nested_config, **kwargs)
        result = {**result, "candidate": candidate.name}
        scored.append(result)
        if _score_is_better(result["mean_primary_score"], best_score, nested_config.higher_is_better):
            best = candidate
            best_score = result["mean_primary_score"]
    return best, scored


def evaluate_selected_on_outer_test(
    df: pd.DataFrame,
    candidate: ModelCandidate,
    outer: DateFold,
    nested_config: NestedValidationConfig,
    *,
    date_col: str = "date",
    label_col: str = "Got_Hit",
    at_bats_col: str | None = "Total_PA",
    key_cols: Sequence[str] = ("key_mlbam", "game_pk"),
) -> dict:
    """Fit on outer train (preprocess fit on train only); score outer test once."""
    train = rows_for_dates(df, outer.train_dates, date_col)
    test = rows_for_dates(df, outer.test_dates, date_col)
    feats = [c for c in candidate.feature_columns if c in train.columns]
    report = {
        "fold_id": outer.fold_id,
        "train_start": str(outer.train_start),
        "train_end": str(outer.train_end),
        "test_start": str(outer.test_start),
        "test_end": str(outer.test_end),
        "n_train_dates": len(outer.train_dates),
        "n_test_dates": len(outer.test_dates),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "selected_candidate": candidate.name,
        "selected_hyperparameters": dict(candidate.hyperparameters),
        "selected_calibration_method": candidate.calibration_method,
        "selected_threshold": candidate.threshold,
        "selected_shortlist_size": candidate.shortlist_size,
        "selected_n_picks": candidate.n_picks,
        "feature_columns": list(feats),
    }
    if not feats or train.empty or test.empty:
        report["error"] = "insufficient_rows_or_features"
        return report

    pre = StandardizePreprocessor().fit(train[feats])
    X_train = pre.transform(train[feats])
    X_test = pre.transform(test[feats])
    y_train = train[label_col].astype(float)
    y_test = test[label_col].astype(float)
    cal_min, cal_block = nested_config.calibration_split_sizes()
    estimator = fit_candidate_estimator(
        candidate, X_train, y_train, train[date_col],
        min_train_dates=cal_min, test_block_dates=cal_block,
    )
    proba = predict_proba_positive(estimator, X_test)
    metrics = classifier_probability_metrics(y_test, proba)
    report.update(metrics)

    scored_test = test.copy()
    scored_test["predicted_probability"] = proba
    picks = build_ranked_picks(
        scored_test,
        date_col=date_col,
        score_col="predicted_probability",
        label_col=label_col,
        at_bats_col=at_bats_col,
        shortlist_size=candidate.shortlist_size,
        n_picks=candidate.n_picks,
        key_cols=key_cols,
    )
    report.update(policy_metrics_from_picks(picks, n_candidate_dates=len(outer.test_dates), n_picks=candidate.n_picks))
    report["_outer_predictions"] = scored_test[
        [c for c in [date_col, *key_cols, label_col] if c in scored_test.columns]
        + ["predicted_probability"]
    ]
    report["_outer_picks"] = picks
    return report


def _mean_ignore_nan(values: Iterable) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def aggregate_outer_fold_reports(fold_reports: Sequence[dict]) -> dict:
    """Mean metrics across outer folds (selection must not use this to peek)."""
    keys = [
        "log_loss", "brier_score", "roc_auc", "base_rate",
        "calibration_intercept", "calibration_slope",
        "coverage", "no_game_rate",
        "top_one_hit_rate", "top_one_advance_rate",
        "two_pick_survival_rate", "two_pick_reset_rate",
        "n_train_rows", "n_test_rows",
    ]
    aggregate = {"n_outer_folds": len(fold_reports)}
    for key in keys:
        aggregate[key] = _mean_ignore_nan(r.get(key, float("nan")) for r in fold_reports)
    aggregate["total_test_rows"] = int(sum(r.get("n_test_rows", 0) or 0 for r in fold_reports))
    aggregate["total_train_rows"] = int(sum(r.get("n_train_rows", 0) or 0 for r in fold_reports))
    return aggregate


def paired_block_bootstrap_difference(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    keys: Sequence[str],
    block_key: str = "date",
    value_a: str = "predicted_probability_a",
    value_b: str = "predicted_probability_b",
    label_col: str = "Got_Hit",
    metric: str = "log_loss",
    n_bootstrap: int = 1000,
    random_seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired date/game/player-aligned bootstrap of metric(challenger) - metric(champion)."""
    a = frame_a.copy()
    b = frame_b.copy()
    rename_a = {c: f"{c}_a" for c in a.columns if c not in keys and c != label_col}
    rename_b = {c: f"{c}_b" for c in b.columns if c not in keys and c != label_col}
    if label_col in a.columns and label_col in b.columns:
        b = b.drop(columns=[label_col])
    aligned = a.rename(columns=rename_a).merge(b.rename(columns=rename_b), on=list(keys), how="inner")
    if aligned.empty:
        return {
            "metric": metric,
            "point_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": n_bootstrap,
            "n_blocks": 0,
            "n_aligned_rows": 0,
            "resampled_unit": block_key,
        }

    if value_a not in aligned.columns:
        cand = [c for c in aligned.columns if c.endswith("_a") and "prob" in c]
        value_a = cand[0] if cand else value_a
    if value_b not in aligned.columns:
        cand = [c for c in aligned.columns if c.endswith("_b") and "prob" in c]
        value_b = cand[0] if cand else value_b

    def _metric(frame: pd.DataFrame, proba_col: str) -> float:
        return float(classifier_probability_metrics(frame[label_col], frame[proba_col]).get(metric, float("nan")))

    point = _metric(aligned, value_a) - _metric(aligned, value_b)
    blocks = unique_sorted_dates(aligned[block_key])
    rng = np.random.default_rng(random_seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    block_arr = np.asarray(blocks, dtype=object)
    n_blocks = len(block_arr)
    for i in range(n_bootstrap):
        sampled = block_arr[rng.integers(0, n_blocks, size=n_blocks)]
        parts = [aligned[aligned[block_key] == blk] for blk in sampled]
        boot = pd.concat(parts, ignore_index=True) if parts else aligned.iloc[0:0]
        diffs[i] = float("nan") if boot.empty else _metric(boot, value_a) - _metric(boot, value_b)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(finite, alpha / 2))
        ci_high = float(np.quantile(finite, 1 - alpha / 2))
    return {
        "metric": metric,
        "point_difference": float(point) if point == point else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "n_blocks": n_blocks,
        "n_aligned_rows": int(len(aligned)),
        "resampled_unit": block_key,
        "aligned_keys": list(keys),
    }


def run_nested_classifier_validation(
    df: pd.DataFrame,
    candidates: Sequence[ModelCandidate],
    nested_config: NestedValidationConfig | None = None,
    *,
    date_col: str = "date",
    label_col: str = "Got_Hit",
    at_bats_col: str | None = "Total_PA",
    key_cols: Sequence[str] = ("key_mlbam", "game_pk"),
    champion_probability_col: str | None = None,
    write_predictions: bool = False,
) -> dict:
    """Full nested rolling-origin run → machine-readable report dict."""
    nested_config = nested_config or NestedValidationConfig()
    if df.empty or date_col not in df.columns:
        return {
            "status": "insufficient_history",
            "n_outer_folds": 0,
            "freeze_dates": [],
            "freeze_warning": "Freeze dates must not be inspected repeatedly during development.",
            "outer_folds": [],
            "aggregate": {},
        }

    dates = unique_sorted_dates(df[date_col])
    nested_folds, freeze_tail = build_nested_folds(
        dates,
        outer_min_train_dates=nested_config.outer_min_train_dates,
        outer_test_block_dates=nested_config.outer_test_block_dates,
        inner_min_train_dates=nested_config.inner_min_train_dates,
        inner_test_block_dates=nested_config.inner_test_block_dates,
        freeze_dates=nested_config.freeze_dates,
    )
    report: dict[str, Any] = {
        "status": "ok" if nested_folds else "insufficient_history",
        "config": asdict(nested_config),
        "n_input_dates": len(dates),
        "n_active_dates": len(dates) - len(freeze_tail),
        "freeze_dates": [str(d) for d in freeze_tail],
        "freeze_warning": (
            "Optional freeze period: excluded from every nested train/test fold "
            "and from every selection decision. Do not repeatedly inspect these "
            "dates during model or policy development."
        ),
        "random_seed": nested_config.random_seed,
        "outer_folds": [],
        "inner_selection_details": [],
        "bootstrap_comparisons": [],
    }
    if not nested_folds:
        report["aggregate"] = {}
        report["n_outer_folds"] = 0
        return report

    fold_reports = []
    challenger_parts = []
    champion_parts = []

    for nested in nested_folds:
        selected, inner_scores = select_candidate_on_inner_folds(
            df, candidates, nested, nested_config, date_col=date_col, label_col=label_col,
        )
        report["inner_selection_details"].append({
            "outer_fold_id": nested.outer.fold_id,
            "inner_scores": inner_scores,
            "selected": None if selected is None else selected.name,
        })
        if selected is None:
            continue
        outer_report = evaluate_selected_on_outer_test(
            df, selected, nested.outer, nested_config,
            date_col=date_col, label_col=label_col,
            at_bats_col=at_bats_col, key_cols=key_cols,
        )
        preds = outer_report.pop("_outer_predictions", None)
        picks = outer_report.pop("_outer_picks", None)
        if write_predictions:
            outer_report["n_prediction_rows"] = 0 if preds is None else int(len(preds))
            outer_report["n_pick_rows"] = 0 if picks is None else int(len(picks))
        fold_reports.append(outer_report)

        if preds is not None and champion_probability_col and champion_probability_col in df.columns:
            champ = rows_for_dates(df, nested.outer.test_dates, date_col)
            merge_keys = [c for c in [date_col, *key_cols] if c in preds.columns and c in champ.columns]
            if merge_keys:
                challenger_parts.append(preds.rename(columns={"predicted_probability": "predicted_probability_a"}))
                champion_parts.append(
                    champ[merge_keys + [label_col, champion_probability_col]].rename(
                        columns={champion_probability_col: "predicted_probability_b"}
                    )
                )

    report["outer_folds"] = fold_reports
    report["aggregate"] = aggregate_outer_fold_reports(fold_reports)
    report["n_outer_folds"] = len(fold_reports)

    if challenger_parts and champion_parts:
        chall = pd.concat(challenger_parts, ignore_index=True)
        champ = pd.concat(champion_parts, ignore_index=True)
        align_keys = [c for c in [date_col, *key_cols] if c in chall.columns and c in champ.columns]
        report["bootstrap_comparisons"].append(
            paired_block_bootstrap_difference(
                chall, champ,
                keys=align_keys,
                block_key=date_col,
                value_a="predicted_probability_a",
                value_b="predicted_probability_b",
                label_col=label_col,
                metric=nested_config.primary_score,
                n_bootstrap=nested_config.bootstrap_samples,
                random_seed=nested_config.random_seed,
            )
        )
    return report


def write_validation_report(
    report: dict,
    report_dir: str | None = None,
    filename: str = "nested_validation_report.json",
) -> str:
    """Write a machine-readable JSON report. Does not dump raw fold predictions."""
    report_dir = report_dir or config.NESTED_VALIDATION_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    cleaned = json.loads(json.dumps(report, default=str))
    path = os.path.join(report_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def ensure_reports_dir_readme(report_dir: str | None = None) -> Path:
    report_dir_path = Path(report_dir or config.NESTED_VALIDATION_REPORT_DIR)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    readme = report_dir_path / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Model validation reports\n\n"
            "Machine-readable nested rolling-origin validation output "
            "(`nested_validation_report.json`). Generated locally / in CI — "
            "not enormous raw fold-prediction dumps. Optional freeze dates "
            "listed in each report must not be repeatedly inspected during "
            "development.\n",
            encoding="utf-8",
        )
    return report_dir_path
