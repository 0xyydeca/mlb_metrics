"""Shadow hitter probability model that explicitly models opportunity.

Three components:

1. Appearance: ``P(Appeared)`` on all candidate rows (incl. DNPs).
2. Expected opportunity: ``E[Plate_Appearances | Appeared]`` on Appeared==1.
3. Conditional hit: ``P(Got_Hit | Appeared, pregame)`` on Appeared==1.

Primary final score (direct formulation)::

    Final_Hit_Probability = P_Appear * P_Hit_Given_Appearance

Challenger (PA-distribution formulation)::

    P(hit | appear) = sum_n P(PA=n | appear) * [1 - (1 - p_hit_per_PA)^n]

Predicted expected PA may enter the conditional hit model via out-of-fold
predictions on training folds only. Actual target-game plate appearances
are never used as features at prediction time.

Feature selection is reduced vs expanded (composites like
``Game_Hit_Probability`` / ``Consistency`` / ``Approach`` /
``Matchup_Hit_Probability`` are expanded-only). Inner nested folds choose
family, hypers, calibration, feature set, and formulation; outer folds
score once. Preprocessing uses sklearn Pipeline / ColumnTransformer with
domain priors fit only on training folds.

Wired into production via ``config.HITTER_SELECTION_MODE``:

- ``legacy`` / ``shadow``: official picks unchanged; shadow logs challenger
  diagnostics (components, shadow_rank, artifact id).
- ``live``: Final_Hit_Probability is the sole ranking / threshold / log /
  Brier / dashboard score (requires promotion gate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_metrics import config, ml_models, model_validation

# ---------------------------------------------------------------------------
# Labels / leakage guards
# ---------------------------------------------------------------------------

APPEARANCE_LABEL = "Appeared"
EXPECTED_PA_LABEL = "Plate_Appearances"
CONDITIONAL_HIT_LABEL = "Got_Hit"
PREDICTED_EXPECTED_PA = "Predicted_Expected_PA"
P_HIT_PER_PA_LABEL = "Hit_Rate_Per_PA"

LEAKAGE_COLUMNS = frozenset({
    "Started", "Appeared", "Batting_Order", "Plate_Appearances",
    "Official_At_Bats", "Hits", "Got_Hit", "No_Game",
    P_HIT_PER_PA_LABEL,
})

# Composite / derived scores that must not ride along with all raw
# ingredients by default. Expanded candidate set may include them.
COMPOSITE_FEATURE_COLUMNS = [
    "probability",
    "Game_Hit_Probability",
    "Consistency",
    "Approach",
    "Matchup_Hit_Probability",
]

# Reduced: distinct opportunity / skill concepts without stacking composites
# on top of their ingredients.
REDUCED_FEATURE_COLUMNS = [
    # empirical-Bayes batter hit skill (WAVE already shrunk upstream)
    "WAVE", "WAVE_L", "WAVE_R", "PA_L", "PA_R",
    # contact quality
    "Exit_Velo", "Barrel_Rate", "xBA", "xwOBA",
    # whiff / chase
    "Whiff_Rate", "Chase_Rate",
    # handedness / pitch-family matchup ingredients
    "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE",
    "starter_fastball_rate", "starter_breaking_rate", "starter_offspeed_rate",
    # starter / bullpen / park / home-away
    "starter_PAVE", "Bullpen_PAVE", "Park_Factor", "is_home",
    # team offensive environment proxy
    "Expected_Bases",
    # historical opportunity / lineup consistency / recency
    "avg_batting_order", "start_rate", "Days_Rest",
]

EXPANDED_EXTRA_COLUMNS = [
    "Expected_BB", "Expected_HBP", "Expected_RBI",
    "WAVE_Home", "WAVE_Away",
] + COMPOSITE_FEATURE_COLUMNS

EXPANDED_FEATURE_COLUMNS = list(dict.fromkeys(
    REDUCED_FEATURE_COLUMNS + EXPANDED_EXTRA_COLUMNS
))

FEATURE_SET_COLUMNS = {
    "reduced": REDUCED_FEATURE_COLUMNS,
    "expanded": EXPANDED_FEATURE_COLUMNS,
}

# Domain imputation groups (fit on training fold only).
NEUTRAL_MULTIPLIER_COLUMNS = ["Park_Factor"]  # missing → ~1.0, not 0
RATE_PRIOR_COLUMNS = [
    "WAVE", "WAVE_L", "WAVE_R", "WAVE_Home", "WAVE_Away",
    "Barrel_Rate", "xBA", "xwOBA",
    "Whiff_Rate", "Chase_Rate",
    "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE",
    "starter_fastball_rate", "starter_breaking_rate", "starter_offspeed_rate",
    "start_rate", "probability", "Game_Hit_Probability",
    "Matchup_Hit_Probability", "Consistency", "Approach",
]
# Missing order ≠ batting 1st; missing starter ≠ elite zero-rate pitcher.
MISSINGNESS_INDICATOR_COLUMNS = [
    "avg_batting_order",
    "starter_PAVE",
    "Bullpen_PAVE",
    "Days_Rest",
    "Exit_Velo",
    "Park_Factor",
]

MODEL_TYPE = "hitter_opportunity_probability"
SHADOW_MODEL_VERSION = "opportunity-v1"


# ---------------------------------------------------------------------------
# Domain preprocessing (fit on train folds only)
# ---------------------------------------------------------------------------


def _as_float_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


class DomainImputer(BaseEstimator, TransformerMixin):
    """Impute with training-fold priors; never blanket-fill every feature with 0.

    - Normalized multipliers (e.g. Park_Factor): neutral prior near 1.0
    - Rate features: training-fold median (league prior proxy)
    - Continuous levels (starter_PAVE, Exit_Velo, …): training-fold median
    - avg_batting_order: training-fold median (not 1.0)
    Missingness indicators are attached separately.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        neutral_columns: Sequence[str] | None = None,
        rate_columns: Sequence[str] | None = None,
        neutral_prior: float = 1.0,
    ):
        # Store constructor args unchanged (sklearn clone contract).
        self.columns = columns
        self.neutral_columns = neutral_columns
        self.rate_columns = rate_columns
        self.neutral_prior = neutral_prior

    def _neutral_cols(self) -> list[str]:
        if self.neutral_columns is not None:
            return list(self.neutral_columns)
        return list(NEUTRAL_MULTIPLIER_COLUMNS)

    def fit(self, X, y=None):
        frame = _as_float_frame(X)
        declared = None if self.columns is None else list(self.columns)
        if declared is not None and all(c in frame.columns for c in declared):
            self.columns_ = declared
        elif declared is not None and len(declared) == frame.shape[1]:
            # ColumnTransformer numpy slice: preserve declared names by position
            frame = pd.DataFrame(frame.to_numpy(), columns=declared, index=frame.index)
            self.columns_ = declared
        else:
            self.columns_ = [str(c) for c in frame.columns]
        neutral = set(self._neutral_cols())
        prior = float(self.neutral_prior)
        self.imputation_values_ = {}
        for col in self.columns_:
            series = pd.to_numeric(frame[col], errors="coerce")
            if col in neutral:
                self.imputation_values_[col] = prior
            else:
                median = float(series.median()) if series.notna().any() else 0.0
                if median != median:
                    median = prior if col in neutral else 0.0
                self.imputation_values_[col] = median
        return self

    def transform(self, X):
        frame = _as_float_frame(X)
        if not all(c in frame.columns for c in self.columns_):
            frame = pd.DataFrame(
                np.asarray(frame, dtype=float),
                columns=list(self.columns_),
                index=getattr(X, "index", None),
            )
        out = pd.DataFrame(index=frame.index)
        for col in self.columns_:
            series = pd.to_numeric(frame[col], errors="coerce")
            out[col] = series.fillna(self.imputation_values_[col]).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class MissingnessIndicators(BaseEstimator, TransformerMixin):
    """Binary indicators for selected columns (1 = missing / null)."""

    def __init__(self, columns: Sequence[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        frame = _as_float_frame(X)
        declared = None if self.columns is None else list(self.columns)
        if declared is not None and all(c in frame.columns for c in declared):
            self.columns_ = declared
        elif declared is not None and len(declared) == frame.shape[1]:
            frame = pd.DataFrame(frame.to_numpy(), columns=declared, index=frame.index)
            self.columns_ = declared
        else:
            wanted = declared if declared is not None else [str(c) for c in frame.columns]
            self.columns_ = [c for c in wanted if c in frame.columns]
        self.feature_names_ = [f"{c}__missing" for c in self.columns_]
        return self

    def transform(self, X):
        frame = _as_float_frame(X)
        if self.columns_ and not all(c in frame.columns for c in self.columns_):
            frame = pd.DataFrame(
                np.asarray(frame, dtype=float),
                columns=list(self.columns_),
                index=getattr(X, "index", None),
            )
        out = pd.DataFrame(index=frame.index)
        for col, name in zip(self.columns_, self.feature_names_):
            series = pd.to_numeric(frame[col], errors="coerce")
            out[name] = series.isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


class _EnsureColumns(BaseEstimator, TransformerMixin):
    """Materialize a fixed column order (missing → NaN) before ColumnTransformer."""

    def __init__(self, columns: Sequence[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _ensure_feature_frame(X, list(self.columns or []))


def build_opportunity_preprocessor(
    feature_columns: Sequence[str],
    *,
    scale: bool = True,
    indicator_columns: Sequence[str] | None = None,
) -> Pipeline:
    """Pipeline: ensure columns → ColumnTransformer(impute + missingness) → optional scale.

    All imputation statistics are fit only when the Pipeline is fit on a
    training fold. No blanket zero-fill of every feature.
    """
    cols = list(feature_columns)
    indicators = (
        list(indicator_columns)
        if indicator_columns is not None
        else [c for c in MISSINGNESS_INDICATOR_COLUMNS if c in cols]
    )
    transformers = [("impute", DomainImputer(columns=cols), cols)]
    if indicators:
        transformers.append(
            ("missing", MissingnessIndicators(columns=indicators), indicators),
        )
    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    try:
        transformer.set_output(transform="pandas")
    except Exception:
        pass
    steps: list[tuple[str, Any]] = [
        ("ensure", _EnsureColumns(columns=cols)),
        ("columns", transformer),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _ensure_feature_frame(X, columns: Sequence[str]) -> pd.DataFrame:
    frame = _as_float_frame(X)
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            out[col] = np.nan
    return out


def assert_no_leakage_features(feature_columns: Sequence[str]) -> None:
    bad = [c for c in feature_columns if c in LEAKAGE_COLUMNS]
    if bad:
        raise ValueError(f"Leakage features requested for model input: {bad}")


def available_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = FEATURE_SET_COLUMNS[feature_set]
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Estimator builders
# ---------------------------------------------------------------------------


def _logit_classifier(**params) -> LogisticRegression:
    defaults = {"max_iter": 1000, "solver": "lbfgs"}
    defaults.update(params)
    return LogisticRegression(**defaults)


def _hgbm_classifier(**params) -> HistGradientBoostingClassifier:
    defaults = {
        "max_depth": 2,
        "learning_rate": 0.1,
        "max_iter": 80,
        "min_samples_leaf": 50,
        "random_state": config.NESTED_VALIDATION_RANDOM_SEED,
    }
    defaults.update(params)
    return HistGradientBoostingClassifier(**defaults)


def _ridge_regressor(**params) -> Ridge:
    defaults = {"alpha": 1.0}
    defaults.update(params)
    return Ridge(**defaults)


def _hgbm_regressor(**params) -> HistGradientBoostingRegressor:
    defaults = {
        "max_depth": 2,
        "learning_rate": 0.1,
        "max_iter": 80,
        "min_samples_leaf": 50,
        "random_state": config.NESTED_VALIDATION_RANDOM_SEED,
    }
    defaults.update(params)
    return HistGradientBoostingRegressor(**defaults)


def build_classifier_pipeline(
    feature_columns: Sequence[str],
    family: str,
    params: dict | None = None,
    *,
    calibration: str | None = None,
    scale_linear: bool = True,
) -> Pipeline:
    """Classifier Pipeline: domain preprocess → family (+ optional calibration)."""
    assert_no_leakage_features(feature_columns)
    params = dict(params or {})
    scale = scale_linear and family == "logit"
    preprocess = build_opportunity_preprocessor(feature_columns, scale=scale)
    if family == "logit":
        clf = _logit_classifier(**params)
    elif family == "hgbm":
        clf = _hgbm_classifier(**params)
    else:
        raise ValueError(f"Unknown classifier family: {family}")

    pipe = Pipeline([("preprocess", preprocess), ("model", clf)])
    if calibration in ("isotonic", "sigmoid"):
        # CalibratedClassifierCV wraps the full pipeline so preprocess stays
        # inside each calibration fold (inner validation only at call sites).
        return Pipeline([
            ("calibrated", CalibratedClassifierCV(pipe, method=calibration, cv=3)),
        ])
    return pipe


def build_regressor_pipeline(
    feature_columns: Sequence[str],
    family: str,
    params: dict | None = None,
    *,
    scale_linear: bool = True,
) -> Pipeline:
    assert_no_leakage_features(feature_columns)
    params = dict(params or {})
    scale = scale_linear and family == "ridge"
    preprocess = build_opportunity_preprocessor(feature_columns, scale=scale)
    if family == "ridge":
        reg = _ridge_regressor(**params)
    elif family == "hgbm":
        reg = _hgbm_regressor(**params)
    else:
        raise ValueError(f"Unknown regressor family: {family}")
    return Pipeline([("preprocess", preprocess), ("model", reg)])


# ---------------------------------------------------------------------------
# PA-distribution challenger math
# ---------------------------------------------------------------------------


def truncated_poisson_pmf(lam: np.ndarray, max_n: int = 8) -> np.ndarray:
    """P(PA=n | appear) for n=1..max_n via truncated Poisson (condition on ≥1).

    Shape: (len(lam), max_n) with columns corresponding to n=1..max_n.
    """
    lam = np.asarray(lam, dtype=float).reshape(-1)
    lam = np.clip(lam, 1e-6, None)
    # Unnormalized P(N=k) for k=0..max_n, then condition on N>=1 and truncate.
    ks = np.arange(0, max_n + 1)
    # log pmf: -λ + k log λ - log(k!)
    log_fact = np.cumsum(np.log(np.maximum(ks, 1)))
    log_fact[0] = 0.0
    log_pmf = -lam[:, None] + ks[None, :] * np.log(lam)[:, None] - log_fact[None, :]
    pmf = np.exp(log_pmf - log_pmf.max(axis=1, keepdims=True))
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    # Drop n=0 mass and renormalize over 1..max_n
    pmf_pos = pmf[:, 1:]
    denom = pmf_pos.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return pmf_pos / denom


def game_hit_prob_from_pa_distribution(
    expected_pa: np.ndarray,
    p_hit_per_pa: np.ndarray,
    *,
    max_n: int = 8,
) -> np.ndarray:
    """Σ_n P(PA=n|appear) × [1 − (1 − p)^n]."""
    p = np.clip(np.asarray(p_hit_per_pa, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    lam = np.asarray(expected_pa, dtype=float).reshape(-1)
    pmf = truncated_poisson_pmf(lam, max_n=max_n)  # (N, max_n) for n=1..max_n
    ns = np.arange(1, max_n + 1, dtype=float)
    term = 1.0 - np.power(1.0 - p[:, None], ns[None, :])
    return (pmf * term).sum(axis=1)


# ---------------------------------------------------------------------------
# OOF predicted expected PA (never actual target PA)
# ---------------------------------------------------------------------------


def out_of_fold_expected_pa(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    family: str,
    params: dict | None = None,
    *,
    date_col: str = "date",
    min_train_dates: int = 10,
    test_block_dates: int = 5,
) -> pd.Series:
    """Rolling-origin OOF ``Predicted_Expected_PA`` for Appeared==1 rows.

    Rows that never appear in a test block (early dates) receive NaN; callers
    should impute those with an in-fold prior (train median) before use.
    Actual ``Plate_Appearances`` are never written into features.
    """
    assert_no_leakage_features(feature_columns)
    appeared = df[df[APPEARANCE_LABEL].astype(float) == 1.0].copy()
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    if appeared.empty or date_col not in appeared.columns:
        return preds

    dates = model_validation.unique_sorted_dates(appeared[date_col])
    folds = model_validation.build_rolling_origin_folds(
        dates,
        min_train_dates=min_train_dates,
        test_block_dates=test_block_dates,
    )
    feats = [c for c in feature_columns if c in appeared.columns]
    for fold in folds:
        train = model_validation.rows_for_dates(appeared, fold.train_dates, date_col)
        test = model_validation.rows_for_dates(appeared, fold.test_dates, date_col)
        if train.empty or test.empty or not feats:
            continue
        pipe = build_regressor_pipeline(feats, family, params)
        y = train[EXPECTED_PA_LABEL].astype(float).clip(lower=0.0)
        pipe.fit(train[feats], y)
        hat = pipe.predict(test[feats])
        preds.loc[test.index] = np.clip(hat, 0.0, None)
    return preds


def attach_predicted_expected_pa(
    df: pd.DataFrame,
    predicted: pd.Series,
    *,
    fill_value: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out[PREDICTED_EXPECTED_PA] = predicted.reindex(out.index).astype(float)
    if fill_value is not None:
        out[PREDICTED_EXPECTED_PA] = out[PREDICTED_EXPECTED_PA].fillna(fill_value)
    return out


# ---------------------------------------------------------------------------
# Composite opportunity model
# ---------------------------------------------------------------------------


@dataclass
class OpportunityModelSpec:
    """One selectable configuration for the shadow opportunity model."""

    name: str
    feature_set: str = "reduced"
    appearance_family: str = "logit"
    appearance_params: dict = field(default_factory=dict)
    appearance_calibration: str | None = None
    expected_pa_family: str = "ridge"
    expected_pa_params: dict = field(default_factory=dict)
    conditional_family: str = "logit"
    conditional_params: dict = field(default_factory=dict)
    conditional_calibration: str | None = None
    formulation: str = "direct"  # or "pa_distribution"
    use_predicted_pa_feature: bool = True
    max_pa_n: int = 8


@dataclass
class FittedOpportunityModel:
    """Serializable multi-component estimator for shadow predictions."""

    spec: OpportunityModelSpec
    feature_columns: list
    appearance_pipeline: Any
    expected_pa_pipeline: Any
    conditional_pipeline: Any  # direct hit classifier OR p_hit_per_PA regressor
    train_expected_pa_prior: float
    formulation: str

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = [c for c in self.feature_columns if c != PREDICTED_EXPECTED_PA]
        X = _ensure_feature_frame(df, feats)
        p_appear = model_validation.predict_proba_positive(self.appearance_pipeline, X)
        expected_pa = np.clip(self.expected_pa_pipeline.predict(X), 0.0, None)

        frame = X.copy()
        if self.spec.use_predicted_pa_feature:
            frame[PREDICTED_EXPECTED_PA] = expected_pa
            cond_cols = list(self.feature_columns)
            if PREDICTED_EXPECTED_PA not in cond_cols:
                cond_cols = cond_cols + [PREDICTED_EXPECTED_PA]
        else:
            cond_cols = [c for c in self.feature_columns if c != PREDICTED_EXPECTED_PA]
        X_cond = _ensure_feature_frame(frame, cond_cols)

        if self.formulation == "pa_distribution":
            # conditional_pipeline predicts p_hit_per_PA in [0,1]
            raw = self.conditional_pipeline.predict(X_cond)
            p_per_pa = np.clip(raw, 1e-6, 1 - 1e-6)
            p_hit_given = game_hit_prob_from_pa_distribution(
                expected_pa, p_per_pa, max_n=self.spec.max_pa_n,
            )
        else:
            p_hit_given = model_validation.predict_proba_positive(
                self.conditional_pipeline, X_cond,
            )
            p_per_pa = np.full_like(p_hit_given, np.nan, dtype=float)

        final = np.clip(p_appear * p_hit_given, 0.0, 1.0)
        return pd.DataFrame({
            "P_Appear": p_appear,
            "Expected_PA_hat": expected_pa,
            "P_Hit_Given_Appearance": p_hit_given,
            "P_Hit_Per_PA": p_per_pa,
            "Final_Hit_Probability": final,
        }, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_components(df)["Final_Hit_Probability"].to_numpy(dtype=float)


def _hit_rate_per_pa(df: pd.DataFrame) -> pd.Series:
    pa = pd.to_numeric(df[EXPECTED_PA_LABEL], errors="coerce").astype(float)
    hits = pd.to_numeric(df["Hits"], errors="coerce").astype(float)
    rate = hits / pa.replace(0, np.nan)
    return rate.clip(0.0, 1.0)


def fit_opportunity_model(
    train_df: pd.DataFrame,
    spec: OpportunityModelSpec,
    *,
    date_col: str = "date",
    oof_min_train_dates: int = 10,
    oof_test_block_dates: int = 5,
) -> FittedOpportunityModel:
    """Fit all three components on ``train_df`` (labels required)."""
    feature_cols = available_feature_columns(train_df, spec.feature_set)
    if not feature_cols:
        raise ValueError(f"No features available for feature_set={spec.feature_set}")
    assert_no_leakage_features(feature_cols)

    appear_y = train_df[APPEARANCE_LABEL].astype(float)
    appear_pipe = build_classifier_pipeline(
        feature_cols, spec.appearance_family, spec.appearance_params,
        calibration=spec.appearance_calibration,
    )
    appear_pipe.fit(_ensure_feature_frame(train_df, feature_cols), appear_y)

    appeared = train_df[appear_y == 1.0]
    if appeared.empty:
        raise ValueError("No Appeared==1 rows to train opportunity / hit components")

    pa_pipe = build_regressor_pipeline(
        feature_cols, spec.expected_pa_family, spec.expected_pa_params,
    )
    pa_y = appeared[EXPECTED_PA_LABEL].astype(float).clip(lower=0.0)
    pa_pipe.fit(_ensure_feature_frame(appeared, feature_cols), pa_y)
    pa_prior = float(pa_y.median()) if pa_y.notna().any() else 3.5

    cond_feature_cols = list(feature_cols)
    train_for_cond = appeared.copy()
    if spec.use_predicted_pa_feature:
        oof = out_of_fold_expected_pa(
            train_df, feature_cols, spec.expected_pa_family, spec.expected_pa_params,
            date_col=date_col,
            min_train_dates=oof_min_train_dates,
            test_block_dates=oof_test_block_dates,
        )
        train_for_cond = attach_predicted_expected_pa(
            train_for_cond, oof.reindex(train_for_cond.index), fill_value=pa_prior,
        )
        if PREDICTED_EXPECTED_PA not in cond_feature_cols:
            cond_feature_cols = cond_feature_cols + [PREDICTED_EXPECTED_PA]

    if spec.formulation == "pa_distribution":
        # Regress empirical Hits/PA; never expose raw PA as a feature.
        rate_y = _hit_rate_per_pa(train_for_cond).fillna(0.25)
        # Use ridge/hgbm regressor; map appearance family naming
        rate_family = "ridge" if spec.conditional_family == "logit" else "hgbm"
        cond_pipe = build_regressor_pipeline(
            cond_feature_cols, rate_family, spec.conditional_params,
        )
        cond_pipe.fit(_ensure_feature_frame(train_for_cond, cond_feature_cols), rate_y)
    else:
        hit_y = train_for_cond[CONDITIONAL_HIT_LABEL].astype(float)
        cond_pipe = build_classifier_pipeline(
            cond_feature_cols, spec.conditional_family, spec.conditional_params,
            calibration=spec.conditional_calibration,
        )
        cond_pipe.fit(_ensure_feature_frame(train_for_cond, cond_feature_cols), hit_y)

    return FittedOpportunityModel(
        spec=spec,
        feature_columns=cond_feature_cols,
        appearance_pipeline=appear_pipe,
        expected_pa_pipeline=pa_pipe,
        conditional_pipeline=cond_pipe,
        train_expected_pa_prior=pa_prior,
        formulation=spec.formulation,
    )


# ---------------------------------------------------------------------------
# Candidate grid + nested validation
# ---------------------------------------------------------------------------


def default_opportunity_specs(*, include_full_grids: bool = False) -> list[OpportunityModelSpec]:
    """Focused (CI) or fuller grids for inner selection."""
    specs: list[OpportunityModelSpec] = []
    feature_sets = ("reduced", "expanded")
    formulations = ("direct", "pa_distribution")

    if include_full_grids:
        logit_cs = list(config.HITTER_OPPORTUNITY_LOGIT_C_GRID)
        hgbm_grid = [
            {"max_depth": d, "learning_rate": lr, "max_iter": it, "min_samples_leaf": leaf}
            for d in config.HITTER_OPPORTUNITY_GBM_PARAM_GRID["max_depth"]
            for lr in config.HITTER_OPPORTUNITY_GBM_PARAM_GRID["learning_rate"]
            for it in config.HITTER_OPPORTUNITY_GBM_PARAM_GRID["max_iter"]
            for leaf in config.HITTER_OPPORTUNITY_GBM_PARAM_GRID["min_samples_leaf"]
        ]
        calibrations = (None, "sigmoid")
        families = [("logit", logit_cs), ("hgbm", hgbm_grid)]
    else:
        logit_cs = [0.1, 1.0]
        hgbm_grid = [{"max_depth": 2, "learning_rate": 0.1, "max_iter": 50, "min_samples_leaf": 50}]
        calibrations = (None, "sigmoid")
        families = [("logit", logit_cs), ("hgbm", hgbm_grid)]

    for feature_set in feature_sets:
        for formulation in formulations:
            for family, param_list in families:
                if family == "logit":
                    param_iter = [{"C": c} for c in param_list]
                else:
                    param_iter = list(param_list)
                for params in param_iter:
                    for cal in calibrations:
                        if formulation == "pa_distribution" and cal is not None:
                            # rate regressor — skip classifier calibration
                            continue
                        name = (
                            f"{formulation}|{feature_set}|{family}|"
                            f"params={params}|cal={cal}"
                        )
                        specs.append(OpportunityModelSpec(
                            name=name,
                            feature_set=feature_set,
                            appearance_family=family,
                            appearance_params=dict(params),
                            appearance_calibration=cal if formulation == "direct" else None,
                            expected_pa_family="ridge" if family == "logit" else "hgbm",
                            expected_pa_params=(
                                {"alpha": 1.0 / max(params.get("C", 1.0), 1e-6)}
                                if family == "logit"
                                else dict(params)
                            ),
                            conditional_family=family,
                            conditional_params=dict(params),
                            conditional_calibration=cal if formulation == "direct" else None,
                            formulation=formulation,
                            use_predicted_pa_feature=True,
                        ))
    return specs


def _final_probability_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    return model_validation.classifier_probability_metrics(actual, predicted)


def score_opportunity_spec_on_inner_folds(
    df: pd.DataFrame,
    spec: OpportunityModelSpec,
    nested: model_validation.NestedFold,
    nested_config: model_validation.NestedValidationConfig,
    *,
    date_col: str = "date",
) -> dict:
    scores = []
    for inner in nested.inner_folds:
        train = model_validation.rows_for_dates(df, inner.train_dates, date_col)
        test = model_validation.rows_for_dates(df, inner.test_dates, date_col)
        if train.empty or test.empty:
            continue
        try:
            model = fit_opportunity_model(
                train, spec, date_col=date_col,
                oof_min_train_dates=max(5, nested_config.inner_min_train_dates // 2),
                oof_test_block_dates=max(1, nested_config.inner_test_block_dates),
            )
            pred = model.predict_proba(test)
            y = test[CONDITIONAL_HIT_LABEL].astype(float)
            metrics = _final_probability_metrics(y, pd.Series(pred, index=test.index))
            scores.append(metrics.get(nested_config.primary_score, float("nan")))
        except Exception:
            continue
    mean_score = model_validation._mean_ignore_nan(scores)
    return {
        "name": spec.name,
        "mean_primary_score": mean_score,
        "n_inner_folds_scored": len(scores),
        "formulation": spec.formulation,
        "feature_set": spec.feature_set,
    }


def select_opportunity_spec_on_inner_folds(
    df: pd.DataFrame,
    specs: Sequence[OpportunityModelSpec],
    nested: model_validation.NestedFold,
    nested_config: model_validation.NestedValidationConfig,
    *,
    date_col: str = "date",
) -> tuple[OpportunityModelSpec | None, list[dict]]:
    scored = [
        score_opportunity_spec_on_inner_folds(df, spec, nested, nested_config, date_col=date_col)
        for spec in specs
    ]
    best = None
    best_score = float("nan")
    for row, spec in zip(scored, specs):
        score = row["mean_primary_score"]
        if model_validation._score_is_better(score, best_score, nested_config.higher_is_better):
            best_score = score
            best = spec
    return best, scored


def evaluate_opportunity_on_outer_test(
    df: pd.DataFrame,
    spec: OpportunityModelSpec,
    outer: model_validation.DateFold,
    nested_config: model_validation.NestedValidationConfig,
    *,
    date_col: str = "date",
) -> dict:
    train = model_validation.rows_for_dates(df, outer.train_dates, date_col)
    test = model_validation.rows_for_dates(df, outer.test_dates, date_col)
    model = fit_opportunity_model(
        train, spec, date_col=date_col,
        oof_min_train_dates=max(5, nested_config.inner_min_train_dates // 2),
        oof_test_block_dates=max(1, nested_config.inner_test_block_dates),
    )
    components = model.predict_components(test)
    y = test[CONDITIONAL_HIT_LABEL].astype(float)
    metrics = _final_probability_metrics(y, components["Final_Hit_Probability"])
    # Appearance / PA auxiliary metrics
    appear_y = test[APPEARANCE_LABEL].astype(float)
    appear_metrics = _final_probability_metrics(appear_y, components["P_Appear"])
    appeared_mask = appear_y == 1.0
    pa_mae = float("nan")
    if appeared_mask.any():
        pa_true = test.loc[appeared_mask, EXPECTED_PA_LABEL].astype(float)
        pa_hat = components.loc[appeared_mask, "Expected_PA_hat"]
        pa_mae = float(np.mean(np.abs(pa_true - pa_hat)))

    pred_frame = test[["date", "game_pk", "key_mlbam"]].copy() if set(["date", "game_pk", "key_mlbam"]).issubset(test.columns) else test.copy()
    for col in components.columns:
        pred_frame[col] = components[col].to_numpy()
    pred_frame[CONDITIONAL_HIT_LABEL] = y.to_numpy()
    pred_frame[APPEARANCE_LABEL] = appear_y.to_numpy()

    return {
        "fold_id": outer.fold_id,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "selected_spec": spec.name,
        "formulation": spec.formulation,
        "feature_set": spec.feature_set,
        "metrics": metrics,
        "appearance_metrics": appear_metrics,
        "expected_pa_mae_on_appeared": pa_mae,
        "_outer_predictions": pred_frame,
        "_fitted_model": model,
    }


def run_opportunity_nested_validation(
    df: pd.DataFrame,
    specs: Sequence[OpportunityModelSpec] | None = None,
    nested_config: model_validation.NestedValidationConfig | None = None,
    *,
    date_col: str = "date",
    include_full_grids: bool = False,
) -> dict:
    """Nested rolling-origin validation for the opportunity shadow model."""
    nested_config = nested_config or model_validation.NestedValidationConfig()
    specs = list(specs) if specs is not None else default_opportunity_specs(
        include_full_grids=include_full_grids,
    )
    if df.empty or date_col not in df.columns:
        return {
            "status": "insufficient_history",
            "n_outer_folds": 0,
            "outer_folds": [],
            "aggregate": {},
            "ablation": {},
            "formulation_comparison": {},
        }

    dates = model_validation.unique_sorted_dates(df[date_col])
    nested_folds, freeze_tail = model_validation.build_nested_folds(
        dates,
        outer_min_train_dates=nested_config.outer_min_train_dates,
        outer_test_block_dates=nested_config.outer_test_block_dates,
        inner_min_train_dates=nested_config.inner_min_train_dates,
        inner_test_block_dates=nested_config.inner_test_block_dates,
        freeze_dates=nested_config.freeze_dates,
    )
    report: dict[str, Any] = {
        "status": "ok" if nested_folds else "insufficient_history",
        "model_type": MODEL_TYPE,
        "config": asdict(nested_config),
        "n_input_dates": len(dates),
        "n_active_dates": len(dates) - len(freeze_tail),
        "freeze_dates": [str(d) for d in freeze_tail],
        "freeze_warning": (
            "Optional freeze period: excluded from every nested train/test fold "
            "and from every selection decision. Do not repeatedly inspect these "
            "dates during model or policy development."
        ),
        "n_specs": len(specs),
        "outer_folds": [],
        "inner_selection_details": [],
        "shadow": True,
        "live_wiring": "none — does not alter predictions.select_picks",
    }
    if not nested_folds:
        report["n_outer_folds"] = 0
        report["aggregate"] = {}
        report["ablation"] = {}
        report["formulation_comparison"] = {}
        return report

    fold_reports = []
    all_preds = []
    last_model = None
    selected_specs = []

    for nested in nested_folds:
        selected, inner_scores = select_opportunity_spec_on_inner_folds(
            df, specs, nested, nested_config, date_col=date_col,
        )
        report["inner_selection_details"].append({
            "outer_fold_id": nested.outer.fold_id,
            "inner_scores": inner_scores,
            "selected": None if selected is None else selected.name,
        })
        if selected is None:
            continue
        outer_report = evaluate_opportunity_on_outer_test(
            df, selected, nested.outer, nested_config, date_col=date_col,
        )
        preds = outer_report.pop("_outer_predictions", None)
        model = outer_report.pop("_fitted_model", None)
        if preds is not None:
            preds = preds.copy()
            preds["outer_fold_id"] = nested.outer.fold_id
            all_preds.append(preds)
        if model is not None:
            last_model = model
        selected_specs.append(selected)
        fold_reports.append(outer_report)
        report["outer_folds"].append(outer_report)

    report["n_outer_folds"] = len(fold_reports)
    report["aggregate"] = _aggregate_opportunity_folds(fold_reports)
    report["ablation"] = _ablation_summary(report["inner_selection_details"])
    report["formulation_comparison"] = _formulation_outer_comparison(fold_reports)
    report["_shadow_predictions"] = (
        pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    )
    report["_last_fitted_model"] = last_model
    report["_selected_specs"] = selected_specs
    return report


def _aggregate_opportunity_folds(fold_reports: Sequence[dict]) -> dict:
    if not fold_reports:
        return {}
    keys = ["log_loss", "brier", "roc_auc"]
    out = {}
    for key in keys:
        vals = [fr.get("metrics", {}).get(key) for fr in fold_reports]
        out[f"mean_{key}"] = model_validation._mean_ignore_nan(vals)
    appear_ll = [fr.get("appearance_metrics", {}).get("log_loss") for fr in fold_reports]
    out["mean_appearance_log_loss"] = model_validation._mean_ignore_nan(appear_ll)
    pa_mae = [fr.get("expected_pa_mae_on_appeared") for fr in fold_reports]
    out["mean_expected_pa_mae"] = model_validation._mean_ignore_nan(pa_mae)
    return out


def _ablation_summary(inner_selection_details: Sequence[dict]) -> dict:
    """How often reduced vs expanded / families win on inner folds."""
    counts = {
        "feature_set": {"reduced": 0, "expanded": 0},
        "formulation": {"direct": 0, "pa_distribution": 0},
        "family_token": {},
    }
    for detail in inner_selection_details:
        selected = detail.get("selected") or ""
        if "|reduced|" in selected:
            counts["feature_set"]["reduced"] += 1
        elif "|expanded|" in selected:
            counts["feature_set"]["expanded"] += 1
        if selected.startswith("direct|"):
            counts["formulation"]["direct"] += 1
        elif selected.startswith("pa_distribution|"):
            counts["formulation"]["pa_distribution"] += 1
        parts = selected.split("|")
        if len(parts) >= 3:
            fam = parts[2]
            counts["family_token"][fam] = counts["family_token"].get(fam, 0) + 1

        # Mean inner scores by feature_set / formulation for ablation table
    by_feature = {"reduced": [], "expanded": []}
    by_form = {"direct": [], "pa_distribution": []}
    by_family = {}
    for detail in inner_selection_details:
        for row in detail.get("inner_scores") or []:
            fs = row.get("feature_set")
            if fs in by_feature and row.get("mean_primary_score") == row.get("mean_primary_score"):
                by_feature[fs].append(row["mean_primary_score"])
            form = row.get("formulation")
            if form in by_form and row.get("mean_primary_score") == row.get("mean_primary_score"):
                by_form[form].append(row["mean_primary_score"])
            name = row.get("name") or ""
            parts = name.split("|")
            if len(parts) >= 3:
                fam = parts[2]
                by_family.setdefault(fam, []).append(row["mean_primary_score"])

    return {
        "selected_counts": counts,
        "mean_inner_log_loss_by_feature_set": {
            k: model_validation._mean_ignore_nan(v) for k, v in by_feature.items()
        },
        "mean_inner_log_loss_by_formulation": {
            k: model_validation._mean_ignore_nan(v) for k, v in by_form.items()
        },
        "mean_inner_log_loss_by_family": {
            k: model_validation._mean_ignore_nan(v) for k, v in by_family.items()
        },
        "note": (
            "Ablation uses inner-fold selection scores only; outer folds are "
            "not used to choose feature set or formulation."
        ),
    }


def _formulation_outer_comparison(fold_reports: Sequence[dict]) -> dict:
    """Outer-fold metrics grouped by the formulation that won inner selection."""
    by_form: dict[str, list] = {"direct": [], "pa_distribution": []}
    for fr in fold_reports:
        form = fr.get("formulation")
        if form in by_form:
            by_form[form].append(fr.get("metrics", {}).get("log_loss"))
    return {
        "n_outer_folds_by_formulation": {k: len(v) for k, v in by_form.items()},
        "mean_outer_log_loss_by_formulation": {
            k: model_validation._mean_ignore_nan(v) for k, v in by_form.items()
        },
        "note": (
            "Compares direct conditional classifier vs PA-distribution challenger "
            "using nested outer folds (formulation selected inside each outer "
            "train block). Not a single final-holdout bake-off."
        ),
    }


def majority_selected_spec(selected_specs: Sequence[OpportunityModelSpec]) -> OpportunityModelSpec | None:
    if not selected_specs:
        return None
    # Prefer the most recent outer fold's selection (latest data regime).
    return selected_specs[-1]


def refit_on_active_history(
    df: pd.DataFrame,
    spec: OpportunityModelSpec,
    nested_config: model_validation.NestedValidationConfig,
    *,
    date_col: str = "date",
) -> FittedOpportunityModel:
    dates = model_validation.unique_sorted_dates(df[date_col])
    active, _freeze = model_validation.freeze_tail_dates(dates, nested_config.freeze_dates)
    train = model_validation.rows_for_dates(df, active, date_col)
    return fit_opportunity_model(
        train, spec, date_col=date_col,
        oof_min_train_dates=max(5, nested_config.inner_min_train_dates // 2),
        oof_test_block_dates=max(1, nested_config.inner_test_block_dates),
    )


def save_opportunity_model_bundle(
    model: FittedOpportunityModel,
    path: str,
    *,
    validation_summary: dict | None = None,
    training_data_start=None,
    training_data_cutoff=None,
) -> dict:
    """Save multi-component shadow model via ``ml_models.save_model_bundle``."""
    payload = {
        "fitted": model,
        "spec": asdict(model.spec),
        "formulation": model.formulation,
        "feature_columns": list(model.feature_columns),
        "train_expected_pa_prior": model.train_expected_pa_prior,
        "components": {
            "appearance": "P(Appeared)",
            "expected_pa": "E[PA | Appeared]",
            "conditional_hit": (
                "P(Got_Hit | Appeared)" if model.formulation == "direct"
                else "p_hit_per_PA + Poisson PA distribution"
            ),
            "final": "P_Appear × P_Hit_Given_Appearance",
        },
        "shadow": True,
        "live_wiring": "none",
    }
    return ml_models.save_model_bundle(
        payload,
        path,
        model_type=MODEL_TYPE,
        model_version=SHADOW_MODEL_VERSION,
        feature_columns=model.feature_columns,
        training_data_start=training_data_start,
        training_data_cutoff=training_data_cutoff,
        hyperparameters=asdict(model.spec),
        calibration_method=(
            model.spec.conditional_calibration or model.spec.appearance_calibration
        ),
        validation_summary=validation_summary or {},
    )


def load_opportunity_model(path: str) -> FittedOpportunityModel | None:
    bundle = ml_models.load_model_bundle(path)
    if bundle is None:
        return None
    est = bundle.get("estimator")
    if isinstance(est, FittedOpportunityModel):
        return est
    if isinstance(est, dict) and isinstance(est.get("fitted"), FittedOpportunityModel):
        return est["fitted"]
    return None


def write_shadow_predictions(predictions: pd.DataFrame, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    predictions.to_csv(path, index=False)


def prepare_opportunity_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop no-game rows for training; keep DNPs (Appeared==0) as negatives."""
    out = df.copy()
    if "No_Game" in out.columns:
        out = out[out["No_Game"].astype(float) != 1.0]
    required = [APPEARANCE_LABEL, EXPECTED_PA_LABEL, CONDITIONAL_HIT_LABEL, "Hits", "date"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Opportunity log missing required columns: {missing}")
    # Coerce labels
    out[APPEARANCE_LABEL] = out[APPEARANCE_LABEL].astype(float)
    out[CONDITIONAL_HIT_LABEL] = out[CONDITIONAL_HIT_LABEL].astype(float).fillna(0.0)
    out[EXPECTED_PA_LABEL] = pd.to_numeric(out[EXPECTED_PA_LABEL], errors="coerce").fillna(0.0)
    out["Hits"] = pd.to_numeric(out["Hits"], errors="coerce").fillna(0.0)
    # DNP: force zeros
    dnp = out[APPEARANCE_LABEL] != 1.0
    out.loc[dnp, CONDITIONAL_HIT_LABEL] = 0.0
    out.loc[dnp, EXPECTED_PA_LABEL] = 0.0
    out.loc[dnp, "Hits"] = 0.0
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Serving + promotion gate (champion / challenger integration)
# ---------------------------------------------------------------------------

FINAL_HIT_PROBABILITY = "Final_Hit_Probability"
COMPONENT_PROBABILITY_COLUMNS = [
    "P_Appear",
    "Expected_PA_hat",
    "P_Hit_Given_Appearance",
    FINAL_HIT_PROBABILITY,
]


def predict_final_hit_probability(
    candidates: pd.DataFrame,
    *,
    model_path: str | None = None,
    as_of_date=None,
) -> tuple[pd.DataFrame, dict]:
    """Attach opportunity-model components + Final_Hit_Probability.

    Returns ``(frame, status)`` where ``frame`` has candidate rows plus
    component probabilities. Never raises on a missing artifact - returns
    NA component columns and ``status["loaded"]=False``.
    """
    path = model_path or config.HITTER_OPPORTUNITY_PROBABILITY_MODEL_PATH
    status = ml_models.inspect_model_path(path)
    empty = candidates.copy() if not candidates.empty else pd.DataFrame()
    for col in COMPONENT_PROBABILITY_COLUMNS:
        if col not in empty.columns:
            empty[col] = pd.NA
    if candidates.empty:
        status = dict(status)
        status["fallback_reason"] = status.get("fallback_reason") or "empty_candidates"
        return empty, status

    model = load_opportunity_model(path)
    if model is None:
        status = dict(status)
        status["loaded"] = False
        status["fallback_reason"] = status.get("fallback_reason") or "missing_or_corrupt_artifact"
        return empty, status

    frame = candidates.copy()
    if as_of_date is not None and "Days_Rest" not in frame.columns and "Last_Game_Date" in frame.columns:
        days = (pd.Timestamp(as_of_date) - pd.to_datetime(frame["Last_Game_Date"])).dt.days
        frame["Days_Rest"] = days.astype(float)

    try:
        components = model.predict_components(frame)
    except Exception as exc:
        status = dict(status)
        status["loaded"] = False
        status["fallback_reason"] = f"predict_failed:{type(exc).__name__}"
        return empty, status

    out = frame.copy()
    for col in COMPONENT_PROBABILITY_COLUMNS:
        out[col] = components[col].to_numpy()
    status = dict(status)
    status["loaded"] = True
    status["fallback_reason"] = None
    status["model_type"] = MODEL_TYPE
    return out, status


def attach_shadow_ranks(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add ``shadow_rank`` by Final_Hit_Probability (1 = best)."""
    out = candidates.copy()
    if FINAL_HIT_PROBABILITY not in out.columns or out[FINAL_HIT_PROBABILITY].isna().all():
        out["shadow_rank"] = pd.NA
        return out
    order = out[FINAL_HIT_PROBABILITY].astype(float).rank(method="first", ascending=False)
    out["shadow_rank"] = order.astype("Int64")
    return out


def evaluate_promotion_gate(report: dict | None) -> tuple[bool, dict]:
    """Fail-closed gate for promoting Final_Hit_Probability to live.

    Requires a nested-validation report with an explicit ``promotion_gate``
    block showing lower log loss / Brier than legacy, non-inferior coverage
    and no-game rate, improved or non-inferior top-one advance, improved
    expected streak utility, and no material high-tail calibration
    degradation — all from untouched outer folds. Missing evidence → fail.
    """
    details: dict[str, Any] = {
        "passed": False,
        "reason": "missing_report",
        "checks": {},
    }
    if not report or not isinstance(report, dict):
        return False, details

    gate = report.get("promotion_gate")
    if not isinstance(gate, dict):
        details["reason"] = "missing_promotion_gate_block"
        return False, details

    required_bools = [
        "lower_log_loss_than_legacy",
        "lower_brier_than_legacy",
        "non_inferior_coverage",
        "no_worse_no_game_rate",
        "improved_or_non_inferior_top_one_advance",
        "improved_expected_streak_utility",
        "no_material_high_tail_calibration_degradation",
        "based_on_untouched_outer_folds",
    ]
    checks = {}
    for key in required_bools:
        val = gate.get(key)
        checks[key] = bool(val) if isinstance(val, (bool, np.bool_)) else False
    details["checks"] = checks
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        details["reason"] = f"gate_checks_failed:{','.join(failed)}"
        details["passed"] = False
        return False, details

    details["reason"] = "passed"
    details["passed"] = True
    return True, details


def promotion_gate_satisfied(report_path: str | None = None) -> tuple[bool, dict]:
    """Load the committed nested report and evaluate the promotion gate."""
    import json
    import os

    path = report_path or config.HITTER_PROMOTION_GATE_REPORT_PATH
    if not path or not os.path.exists(path):
        return False, {"passed": False, "reason": "missing_report_file", "path": path}
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        return False, {
            "passed": False,
            "reason": f"report_unreadable:{type(exc).__name__}",
            "path": path,
        }
    ok, details = evaluate_promotion_gate(report)
    details["path"] = path
    return ok, details


def resolve_hitter_selection_mode(
    configured: str | None = None,
    *,
    report_path: str | None = None,
    force_live: bool = False,
) -> tuple[str, dict]:
    """Resolve effective selection mode.

    ``live`` only sticks when the promotion gate passes (or ``force_live``
    for explicit test overrides). A configured ``live`` without the gate
    falls back to ``shadow`` with a recorded reason.
    """
    mode = (configured if configured is not None else config.HITTER_SELECTION_MODE) or "shadow"
    if mode not in config.HITTER_SELECTION_MODES:
        return "shadow", {
            "configured": mode,
            "effective": "shadow",
            "fallback_used": True,
            "fallback_reason": f"invalid_mode:{mode}",
            "promotion_gate": {},
        }
    if mode != "live":
        return mode, {
            "configured": mode,
            "effective": mode,
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": {},
        }

    if force_live:
        return "live", {
            "configured": "live",
            "effective": "live",
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": {"forced": True},
        }

    ok, gate_details = promotion_gate_satisfied(report_path)
    if ok:
        return "live", {
            "configured": "live",
            "effective": "live",
            "fallback_used": False,
            "fallback_reason": None,
            "promotion_gate": gate_details,
        }
    return "shadow", {
        "configured": "live",
        "effective": "shadow",
        "fallback_used": True,
        "fallback_reason": f"promotion_gate_failed:{gate_details.get('reason')}",
        "promotion_gate": gate_details,
    }


def effective_hitter_model_version(selection_mode: str | None = None) -> str:
    mode = selection_mode or config.HITTER_SELECTION_MODE
    if mode == "live":
        return config.HITTER_MODEL_VERSION_LIVE
    return config.HITTER_MODEL_VERSION
