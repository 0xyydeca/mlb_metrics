"""Shared machine-learning infrastructure for the weak-signal follow-up
(DFS hitter/pitcher projections, Age Curves HR9) - written once here rather
than reimplemented per signal, the same way matchup.py's log5/clip-and-blend
helpers are shared between the hitter and team-level models.

The one thing every one of these training jobs needs and sklearn doesn't
provide out of the box: a walk-forward, date-respecting cross-validation
split. sklearn's default K-fold (and GridSearchCV's default CV) shuffles
rows randomly, which would let near-identical rows from the same date (or
adjacent dates - a batter's form barely changes day to day) land in both
the train and test fold, leaking information a live model would never
actually have. Every other backtest in this project
(dfs_backtest.backtest_dfs_projections, game_picks_backtest, age_curve's
backtest_projection_accuracy) enforces "only data strictly before the test
date" - WalkForwardDateSplit is that same discipline, wired into sklearn's
CV protocol so GridSearchCV respects it automatically.

For **nested** rolling-origin research (outer folds for honest reporting,
inner folds for family / hyperparameter / calibration / feature / policy
selection, optional untouched freeze period), see
`mlb_metrics.model_validation` - that module replaces repeatedly
inspecting the same final holdout block. Live prediction loaders in this
file are unchanged.
"""

import os
import subprocess
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV


class WalkForwardDateSplit:
    """sklearn-compatible CV splitter (get_n_splits/split). Blocks
    consecutive TEST dates together into folds, growing the training window
    forward in time - fold 1 trains on the earliest `min_train_dates` dates
    and tests on the next `test_block_dates`; fold 2 trains on everything up
    through fold 1's test block and tests on the following
    `test_block_dates`; and so on. Never puts a later date in train while an
    earlier date is in test.

    `dates` must be a sequence aligned POSITIONALLY with whatever X is
    passed to `.split()` - GridSearchCV's `split(X, y, groups)` call doesn't
    carry dates itself, so this stores dates at construction time and
    assumes the caller passes X/y to grid_search_walk_forward in the same
    row order `dates` was built from.
    """

    def __init__(self, dates, min_train_dates: int, test_block_dates: int):
        self.dates = pd.Series(dates).reset_index(drop=True)
        self.min_train_dates = min_train_dates
        self.test_block_dates = test_block_dates

    def _unique_sorted_dates(self):
        return sorted(self.dates.unique())

    def split(self, X=None, y=None, groups=None):
        unique_dates = self._unique_sorted_dates()
        cursor = self.min_train_dates
        while cursor < len(unique_dates):
            train_dates = set(unique_dates[:cursor])
            test_dates = set(unique_dates[cursor:cursor + self.test_block_dates])
            train_idx = np.flatnonzero(self.dates.isin(train_dates).to_numpy())
            test_idx = np.flatnonzero(self.dates.isin(test_dates).to_numpy())
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
            cursor += self.test_block_dates

    def get_n_splits(self, X=None, y=None, groups=None):
        unique_dates = self._unique_sorted_dates()
        n = 0
        cursor = self.min_train_dates
        while cursor < len(unique_dates):
            n += 1
            cursor += self.test_block_dates
        return n


def grid_search_walk_forward(
    X, y, dates, estimator, param_grid, min_train_dates, test_block_dates,
    scoring: str = "neg_mean_absolute_error",
) -> GridSearchCV:
    """Fits a GridSearchCV using WalkForwardDateSplit as the CV splitter -
    the one place a non-default (date-respecting) CV needs to be wired in,
    so every signal's training script gets no-lookahead grid search without
    hand-rolling the split logic per signal. Returns the fitted
    GridSearchCV (`.best_estimator_`/`.best_params_`/`.cv_results_` are all
    available on the result, same as any sklearn GridSearchCV)."""
    splitter = WalkForwardDateSplit(dates, min_train_dates, test_block_dates)
    search = GridSearchCV(estimator, param_grid, cv=splitter, scoring=scoring)
    search.fit(X, y)
    return search


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAE, correlation, naive-baseline MAE (always guess actual's own
    mean), and n - the exact metric set scripts/backtest_dfs_rankings.py's
    _report and scripts/backtest_age_curve.py's run_one_metric already use,
    factored out once so a model-vs-heuristic comparison is apples-to-apples
    against the identical formula, not a re-derived one."""
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    n = len(actual)
    if n == 0:
        return {"mae": float("nan"), "baseline_mae": float("nan"), "correlation": float("nan"), "n": 0}

    mae = (actual - predicted).abs().mean()
    baseline_mae = (actual - actual.mean()).abs().mean()
    correlation = actual.corr(predicted) if n > 1 else float("nan")
    return {"mae": mae, "baseline_mae": baseline_mae, "correlation": correlation, "n": n}


def evaluate_classifier_predictions(actual: pd.Series, predicted_proba: pd.Series) -> dict:
    """accuracy (@0.5 threshold), log_loss, brier_score, roc_auc,
    baseline_log_loss (always predict actual's own base rate), and n - the
    classification analog of evaluate_predictions, for a binary label and a
    predicted probability in [0, 1]. roc_auc is NaN (not raised) when
    `actual` is single-class, the same "can't compute, don't crash" stance
    evaluate_predictions takes on an empty input."""
    actual = pd.Series(actual).reset_index(drop=True)
    predicted_proba = pd.Series(predicted_proba).reset_index(drop=True)
    n = len(actual)
    if n == 0:
        return {
            "accuracy": float("nan"), "log_loss": float("nan"), "brier_score": float("nan"),
            "roc_auc": float("nan"), "baseline_log_loss": float("nan"), "n": 0,
        }

    base_rate = actual.mean()
    predicted_class = (predicted_proba >= 0.5).astype(int)
    baseline_proba = pd.Series(base_rate, index=actual.index)

    return {
        "accuracy": accuracy_score(actual, predicted_class),
        "log_loss": log_loss(actual, predicted_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(actual, predicted_proba),
        "roc_auc": roc_auc_score(actual, predicted_proba) if actual.nunique() > 1 else float("nan"),
        "baseline_log_loss": log_loss(actual, baseline_proba, labels=[0, 1]),
        "n": n,
    }


def fit_calibrated(fitted_estimator, X, y, dates, method: str, min_train_dates: int, test_block_dates: int) -> CalibratedClassifierCV:
    """Quant-analytics item #3, slice 2 ("uncertainty quantification" -
    isotonic/Platt calibration on top of the raw model output): wraps a
    classifier in sklearn's CalibratedClassifierCV(method=...), refit via
    the SAME no-lookahead WalkForwardDateSplit every other walk-forward fit
    in this project uses - calibration is fit only on data strictly before
    each internal fold's own test block, never leaking a later date into
    an earlier fold's calibration, the identical discipline
    grid_search_walk_forward already enforces for hyperparameter search.

    `fitted_estimator` only supplies the hyperparameters, via
    sklearn.base.clone (returns a fresh UNFIT clone with the same params,
    never mutates the original) - CalibratedClassifierCV needs to fit its
    own base estimator once per internal fold, it can't reuse an
    already-fitted one without the deprecated cv="prefit" path. `method`
    is "isotonic" or "sigmoid" (Platt scaling) - the caller decides which,
    this function doesn't pick one a priori."""
    splitter = WalkForwardDateSplit(dates, min_train_dates, test_block_dates)
    calibrated = CalibratedClassifierCV(clone(fitted_estimator), method=method, cv=splitter)
    calibrated.fit(X, y)
    return calibrated


class _SigmoidCalibrator:
    """Platt scaling for an ALREADY-COMPUTED probability estimate (not a
    full classifier - see fit_calibrated above for that case): fits
    sklearn's own LogisticRegression on the logit-transformed raw
    probability as the sole feature, so `.predict()` reproduces the
    textbook Platt formula sigmoid(a*logit(p) + b) via sklearn's own
    fitted coefficients rather than a hand-derived one. Wrapped in a class
    (not a bare function) so fit_probability_calibration can return this
    and IsotonicRegression interchangeably - callers only ever call
    `.predict(raw_probability)`, never branch on which method was used."""

    def __init__(self, logistic_regression: LogisticRegression):
        self._model = logistic_regression

    def predict(self, raw_probability):
        raw_probability = np.asarray(raw_probability, dtype=float)
        logit = _logit(raw_probability)
        return self._model.predict_proba(logit.reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """log(p / (1-p)), clipped away from the real 0/1 boundary first - an
    exact 0 or 1 probability would otherwise divide by zero / take log(0)."""
    clipped = np.clip(p, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def fit_probability_calibration(raw_probability: pd.Series, actual: pd.Series, method: str = "isotonic"):
    """Fits a monotonic recalibration mapping an ALREADY-COMPUTED
    probability estimate (e.g. a heuristic ratio, not a full classifier's
    raw score - see fit_calibrated above for that case) to a properly
    calibrated one, against real 0/1 outcomes. Quant-analytics follow-up:
    "dig into calibration" - the real, honest fix for a probability
    estimate whose spread doesn't match its own real outcome rate (too
    compressed toward 0.5, too spread out, or non-monotonically off in
    places), without assuming a specific parametric shape a priori.

    `method="isotonic"` (default): sklearn.isotonic.IsotonicRegression
    (out_of_bounds="clip") - a flexible, monotonic, non-parametric fit.
    Can correct compression, over-spread, or a non-monotonic miscalibration
    all at once, but has more effective degrees of freedom, so it can
    overfit a small/noisy sample more readily than the alternative below.

    `method="sigmoid"`: Platt scaling (_SigmoidCalibrator above) - a
    single 2-parameter logistic curve on the logit-transformed input.
    Lower-variance and more robust with a small sample, but assumes the
    miscalibration really is shaped like a single sigmoid correction
    (e.g. can't fix a non-monotonic wiggle isotonic could).

    Returns a fitted object exposing `.predict(raw_probability_array) ->
    calibrated_probability_array`, the same interface regardless of
    `method`, so callers (and game_picks.apply_calibration) never need to
    branch on which one was actually used. Callers are responsible for
    picking `method` via real walk-forward validation (see
    scripts/train_game_pick_calibration.py) - this function doesn't guess
    which is better for a given dataset on its own."""
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw_probability, actual)
        return model
    if method == "sigmoid":
        logit = _logit(np.asarray(raw_probability, dtype=float))
        logistic_regression = LogisticRegression()
        logistic_regression.fit(logit.reshape(-1, 1), actual)
        return _SigmoidCalibrator(logistic_regression)
    raise ValueError(f"Unknown calibration method {method!r} - expected 'isotonic' or 'sigmoid'.")


def save_model(model, path: str) -> None:
    """joblib.dump, creating parent directories as needed - matches this
    project's existing pattern of committing binary data artifacts to git
    (data/raw/*.parquet, data/raw/lahman/*.parquet).

    Prefer `save_model_bundle` for newly trained live models so prediction
    rows can carry reproducible provenance. This plain dump remains for
    older training scripts/tests and for callers that intentionally store
    an estimator without metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Returns the estimator (never raises) on any load failure - missing
    file, corrupt file, a scikit-learn version mismatch - so callers can
    use the same "fall back to the heuristic" pattern pipeline.run()
    already uses for a failed schedule fetch, rather than crashing the
    daily build.

    Backward compatible with both legacy plain joblib estimators and the
    newer model-bundle format (`save_model_bundle`): bundles are unwrapped
    to their `.estimator` so existing `.predict` / `.predict_proba`
    callers need no changes."""
    payload = _safe_joblib_load(path)
    if payload is None:
        return None
    if _is_bundle_payload(payload):
        return payload.get("estimator")
    return payload


# ---------------------------------------------------------------------------
# Model-bundle format (backward compatible with plain joblib estimators)
# ---------------------------------------------------------------------------

MODEL_BUNDLE_FORMAT = "mlb_metrics_model_bundle_v1"

MODEL_BUNDLE_REQUIRED_FIELDS = (
    "estimator",
    "model_type",
    "artifact_id",
    "trained_at_utc",
    "training_data_start",
    "training_data_cutoff",
    "feature_columns",
    "feature_schema_hash",
    "hyperparameters",
    "calibration_method",
    "training_code_sha",
    "validation_summary",
    "model_version",
)


def feature_schema_hash(feature_columns) -> str:
    """Stable, reproducible hash of an ordered feature-column schema.
    Column order matters (train/serve parity); names are taken as-is."""
    columns = [str(c) for c in list(feature_columns)]
    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_code_sha() -> str:
    """Best-effort code SHA for provenance stamps.

    Order: `GITHUB_SHA` (Actions) → `git rev-parse HEAD` (local checkout)
    → `"unknown"`. Never raises - a missing `.git`, a failed subprocess,
    or a non-git working tree all degrade to `"unknown"`."""
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return str(env_sha).strip() or "unknown"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sha = (result.stdout or "").strip()
            if sha:
                return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def compute_artifact_id(
    *,
    model_type: str,
    model_version: str,
    feature_schema_hash_value: str,
    training_data_cutoff,
    hyperparameters: dict | None = None,
    calibration_method: str | None = None,
    training_code_sha: str | None = None,
) -> str:
    """Stable artifact id derived only from reproducible metadata (not
    wall-clock trained_at). Same inputs → same id across re-saves."""
    canonical = {
        "model_type": model_type,
        "model_version": model_version,
        "feature_schema_hash": feature_schema_hash_value,
        "training_data_cutoff": None if training_data_cutoff is None else str(training_data_cutoff),
        "hyperparameters": hyperparameters or {},
        "calibration_method": calibration_method,
        "training_code_sha": training_code_sha or "unknown",
    }
    payload = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _safe_joblib_load(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _is_bundle_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("format") == MODEL_BUNDLE_FORMAT


def build_model_bundle(
    estimator,
    *,
    model_type: str,
    model_version: str,
    feature_columns,
    training_data_start=None,
    training_data_cutoff=None,
    hyperparameters: dict | None = None,
    calibration_method: str | None = None,
    validation_summary: dict | None = None,
    trained_at_utc: str | None = None,
    training_code_sha: str | None = None,
    artifact_id: str | None = None,
) -> dict:
    """Build a serializable model-bundle dict (not yet written to disk)."""
    columns = list(feature_columns)
    schema_hash = feature_schema_hash(columns)
    code_sha = training_code_sha if training_code_sha is not None else resolve_code_sha()
    artifact = artifact_id or compute_artifact_id(
        model_type=model_type,
        model_version=model_version,
        feature_schema_hash_value=schema_hash,
        training_data_cutoff=training_data_cutoff,
        hyperparameters=hyperparameters,
        calibration_method=calibration_method,
        training_code_sha=code_sha,
    )
    return {
        "format": MODEL_BUNDLE_FORMAT,
        "estimator": estimator,
        "model_type": model_type,
        "artifact_id": artifact,
        "trained_at_utc": trained_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_data_start": None if training_data_start is None else str(training_data_start),
        "training_data_cutoff": None if training_data_cutoff is None else str(training_data_cutoff),
        "feature_columns": columns,
        "feature_schema_hash": schema_hash,
        "hyperparameters": hyperparameters or {},
        "calibration_method": calibration_method,
        "training_code_sha": code_sha,
        "validation_summary": validation_summary or {},
        "model_version": model_version,
    }


def save_model_bundle(estimator, path: str, **bundle_kwargs) -> dict:
    """Save a newly trained model as a provenance-bearing bundle and
    return the bundle dict that was written."""
    bundle = build_model_bundle(estimator, **bundle_kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_model_bundle(path: str) -> dict | None:
    """Load a model bundle (or wrap a legacy plain estimator). Returns
    None when the file is missing/corrupt - never raises.

    Legacy plain estimators are returned as a bundle-shaped dict with
    `artifact_id="legacy"`, null training metadata, and the estimator in
    place - so callers can branch on provenance without a second code path
    for "was this a bundle." """
    payload = _safe_joblib_load(path)
    if payload is None:
        return None
    if _is_bundle_payload(payload):
        return payload
    # Legacy plain estimator - readable provenance defaults, no guessing
    # at training windows or feature schemas that were never recorded.
    return {
        "format": MODEL_BUNDLE_FORMAT,
        "estimator": payload,
        "model_type": "legacy_plain_estimator",
        "artifact_id": "legacy",
        "trained_at_utc": None,
        "training_data_start": None,
        "training_data_cutoff": None,
        "feature_columns": [],
        "feature_schema_hash": None,
        "hyperparameters": {},
        "calibration_method": None,
        "training_code_sha": "unknown",
        "validation_summary": {},
        "model_version": "legacy",
        "is_legacy_plain_estimator": True,
    }


def inspect_model_path(path: str) -> dict:
    """Status dict for a model path - always returns, never raises.

    Distinguishes a missing/corrupt artifact (fallback_used=True with an
    explicit reason) from a successfully loaded estimator/bundle so
    prediction logs can record heuristic fallbacks that would otherwise
    look identical to a normal no-model day."""
    empty = {
        "loaded": False,
        "is_bundle": False,
        "is_legacy_plain_estimator": False,
        "fallback_used": True,
        "fallback_reason": "missing_artifact",
        "estimator": None,
        "artifact_id": None,
        "training_data_cutoff": None,
        "feature_schema_hash": None,
        "model_version": None,
        "model_type": None,
        "calibration_method": None,
        "training_code_sha": None,
        "feature_columns": [],
        "validation_summary": {},
    }
    if not path or not os.path.exists(path):
        return empty

    try:
        payload = joblib.load(path)
    except Exception:
        empty["fallback_reason"] = "load_error"
        return empty

    if _is_bundle_payload(payload):
        return {
            "loaded": payload.get("estimator") is not None,
            "is_bundle": True,
            "is_legacy_plain_estimator": False,
            "fallback_used": payload.get("estimator") is None,
            "fallback_reason": "empty_estimator" if payload.get("estimator") is None else None,
            "estimator": payload.get("estimator"),
            "artifact_id": payload.get("artifact_id"),
            "training_data_cutoff": payload.get("training_data_cutoff"),
            "feature_schema_hash": payload.get("feature_schema_hash"),
            "model_version": payload.get("model_version"),
            "model_type": payload.get("model_type"),
            "calibration_method": payload.get("calibration_method"),
            "training_code_sha": payload.get("training_code_sha"),
            "feature_columns": list(payload.get("feature_columns") or []),
            "validation_summary": payload.get("validation_summary") or {},
        }

    return {
        "loaded": True,
        "is_bundle": False,
        "is_legacy_plain_estimator": True,
        "fallback_used": False,
        "fallback_reason": None,
        "estimator": payload,
        "artifact_id": "legacy",
        "training_data_cutoff": None,
        "feature_schema_hash": None,
        "model_version": "legacy",
        "model_type": "legacy_plain_estimator",
        "calibration_method": None,
        "training_code_sha": "unknown",
        "feature_columns": [],
        "validation_summary": {},
    }


def provenance_fields_from_model_status(status: dict | None) -> dict:
    """Subset of inspect_model_path output stamped onto prediction rows."""
    status = status or {}
    if not status.get("loaded"):
        return {
            "model_artifact_id": pd.NA,
            "training_data_cutoff": pd.NA,
            "feature_schema_hash": pd.NA,
            "fallback_used": True,
            "fallback_reason": status.get("fallback_reason") or "missing_artifact",
        }
    return {
        "model_artifact_id": status.get("artifact_id"),
        "training_data_cutoff": status.get("training_data_cutoff"),
        "feature_schema_hash": status.get("feature_schema_hash"),
        "fallback_used": bool(status.get("fallback_used")),
        "fallback_reason": status.get("fallback_reason"),
    }
