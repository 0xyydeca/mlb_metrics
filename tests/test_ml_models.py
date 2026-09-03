import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge

from mlb_metrics import ml_models


class _OverconfidentClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a real LogisticRegression but deliberately exaggerates its
    predicted probabilities away from 0.5 (raised to a power > 1 in
    log-odds space) - a synthetic stand-in for a genuinely overconfident
    classifier (e.g. an under-regularized gradient-boosted model), so
    test_fit_calibrated_improves_brier_score_for_an_overconfident_classifier
    below can assert a REAL, deterministic before/after calibration
    improvement rather than relying on incidental real-world overfitting
    behavior. Implements get_params/set_params (required by
    sklearn.base.clone, which ml_models.fit_calibrated uses)."""

    def __init__(self, exaggeration: float = 4.0):
        self.exaggeration = exaggeration

    def fit(self, X, y):
        self.base_ = LogisticRegression().fit(X, y)
        self.classes_ = self.base_.classes_
        return self

    def predict_proba(self, X):
        p = np.clip(self.base_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        exaggerated = 1 / (1 + np.exp(-logit * self.exaggeration))
        return np.column_stack([1 - exaggerated, exaggerated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_walk_forward_date_split_never_puts_future_in_train():
    dates = pd.Series([f"2026-01-{d:02d}" for d in range(1, 21) for _ in range(3)])  # 20 dates, 3 rows each
    splitter = ml_models.WalkForwardDateSplit(dates, min_train_dates=10, test_block_dates=5)

    folds = list(splitter.split())
    assert len(folds) == splitter.get_n_splits()
    assert len(folds) == 2  # cursor 10 -> test [10:15]; cursor 15 -> test [15:20]; cursor 20 stops

    unique_dates = sorted(dates.unique())
    for train_idx, test_idx in folds:
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        assert train_dates.isdisjoint(test_dates)
        max_train_date = max(train_dates)
        min_test_date = min(test_dates)
        assert unique_dates.index(max_train_date) < unique_dates.index(min_test_date)

    # First fold trains on the first 10 dates only.
    first_train_dates = set(dates.iloc[folds[0][0]])
    assert first_train_dates == set(unique_dates[:10])
    first_test_dates = set(dates.iloc[folds[0][1]])
    assert first_test_dates == set(unique_dates[10:15])


def test_walk_forward_date_split_too_few_dates_yields_no_folds():
    dates = pd.Series([f"2026-01-{d:02d}" for d in range(1, 6)])  # 5 dates
    splitter = ml_models.WalkForwardDateSplit(dates, min_train_dates=10, test_block_dates=5)

    assert list(splitter.split()) == []
    assert splitter.get_n_splits() == 0


def test_grid_search_walk_forward_selects_best_alpha_and_respects_split():
    rng = np.random.RandomState(0)
    n_dates = 20
    rows_per_date = 20
    dates = np.repeat([f"2026-01-{d:02d}" for d in range(1, n_dates + 1)], rows_per_date)
    x = rng.normal(size=len(dates))
    y = 3 * x + rng.normal(scale=0.01, size=len(dates))  # near-noiseless linear signal
    X = pd.DataFrame({"x": x})

    search = ml_models.grid_search_walk_forward(
        X, y, dates, Ridge(), {"alpha": [0.001, 1000.0]},
        min_train_dates=10, test_block_dates=5,
    )
    # A near-noiseless linear relationship should strongly prefer the
    # unregularized (tiny-alpha) fit over one crushed toward 0 by alpha=1000.
    assert search.best_params_["alpha"] == 0.001
    assert search.n_splits_ == ml_models.WalkForwardDateSplit(dates, 10, 5).get_n_splits()


def test_fit_calibrated_uses_the_walk_forward_splitter_not_a_default_cv():
    rng = np.random.RandomState(0)
    n_dates, rows_per_date = 30, 10
    dates = np.repeat([f"2026-01-{d:02d}" for d in range(1, n_dates + 1)], rows_per_date)
    x = rng.normal(size=len(dates))
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    X = pd.DataFrame({"x": x})

    base = LogisticRegression().fit(X, y)
    calibrated = ml_models.fit_calibrated(base, X, pd.Series(y), dates, "sigmoid", min_train_dates=10, test_block_dates=5)

    # A real, catchable regression: if fit_calibrated silently fell back to
    # a default CV (e.g. passing an int instead of the WalkForwardDateSplit
    # instance) instead of actually wiring in the date-respecting splitter,
    # the number of internally fit calibrators would no longer match
    # WalkForwardDateSplit's own fold count for these exact dates/params.
    expected_folds = ml_models.WalkForwardDateSplit(dates, 10, 5).get_n_splits()
    assert len(calibrated.calibrated_classifiers_) == expected_folds


def test_fit_calibrated_improves_brier_score_for_an_overconfident_classifier():
    rng = np.random.RandomState(0)
    n_dates, rows_per_date = 30, 60
    dates = np.repeat([f"2026-01-{d:02d}" for d in range(1, n_dates + 1)], rows_per_date)
    x = rng.normal(size=len(dates))
    true_p = 1 / (1 + np.exp(-x))
    y = rng.binomial(1, true_p)
    X = pd.DataFrame({"x": x})

    overconfident = _OverconfidentClassifier(exaggeration=4.0).fit(X, y)
    raw_proba = overconfident.predict_proba(X)[:, 1]
    raw_brier_vs_truth = np.mean((raw_proba - true_p) ** 2)

    calibrated = ml_models.fit_calibrated(
        overconfident, X, pd.Series(y), dates, "isotonic", min_train_dates=10, test_block_dates=5
    )
    calibrated_proba = calibrated.predict_proba(X)[:, 1]
    calibrated_brier_vs_truth = np.mean((calibrated_proba - true_p) ** 2)

    # Calibration is scored against known GROUND-TRUTH probabilities here
    # (not just observed 0/1 outcomes) - a real, deterministic confirmation
    # that isotonic recalibration measurably corrects the deliberately
    # exaggerated raw scores toward the true generating probability, not
    # just "the numbers moved somewhere."
    assert calibrated_brier_vs_truth < raw_brier_vs_truth


def test_evaluate_predictions_exact_arithmetic():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    predicted = pd.Series([1.0, 2.0, 3.0, 6.0])  # last prediction off by 2

    result = ml_models.evaluate_predictions(actual, predicted)

    assert result["n"] == 4
    assert result["mae"] == pytest.approx(0.5)  # (0+0+0+2)/4
    assert result["baseline_mae"] == pytest.approx(1.0)  # mean=2.5, |1-2.5|+|2-2.5|+|3-2.5|+|4-2.5| = 1+0.5+0.5+1.5=3.5/4
    assert result["correlation"] == pytest.approx(actual.corr(predicted))


def test_evaluate_predictions_empty_returns_nan_not_crash():
    result = ml_models.evaluate_predictions(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert result["n"] == 0
    assert pd.isna(result["mae"])
    assert pd.isna(result["baseline_mae"])
    assert pd.isna(result["correlation"])


def test_evaluate_classifier_predictions_exact_arithmetic():
    import math

    actual = pd.Series([0, 0, 1, 1])
    predicted_proba = pd.Series([0.1, 0.4, 0.6, 0.9])  # every prediction on the correct side of 0.5

    result = ml_models.evaluate_classifier_predictions(actual, predicted_proba)

    assert result["n"] == 4
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)  # perfect rank separation

    expected_log_loss = -sum([
        math.log(1 - 0.1), math.log(1 - 0.4), math.log(0.6), math.log(0.9),
    ]) / 4
    assert result["log_loss"] == pytest.approx(expected_log_loss)

    expected_brier = ((0.1 - 0) ** 2 + (0.4 - 0) ** 2 + (0.6 - 1) ** 2 + (0.9 - 1) ** 2) / 4
    assert result["brier_score"] == pytest.approx(expected_brier)

    # base rate = 0.5 -> always predicting 0.5 gives -log(0.5) per row regardless of actual
    assert result["baseline_log_loss"] == pytest.approx(-math.log(0.5))


def test_evaluate_classifier_predictions_single_class_auc_is_nan_not_raise():
    actual = pd.Series([1, 1, 1])
    predicted_proba = pd.Series([0.6, 0.7, 0.8])

    result = ml_models.evaluate_classifier_predictions(actual, predicted_proba)

    assert result["n"] == 3
    assert pd.isna(result["roc_auc"])
    assert result["accuracy"] == pytest.approx(1.0)


def test_evaluate_classifier_predictions_empty_returns_nan_not_crash():
    result = ml_models.evaluate_classifier_predictions(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert result["n"] == 0
    assert pd.isna(result["accuracy"])
    assert pd.isna(result["log_loss"])
    assert pd.isna(result["brier_score"])
    assert pd.isna(result["roc_auc"])
    assert pd.isna(result["baseline_log_loss"])


def test_save_and_load_model_round_trips(tmp_path):
    model = Ridge(alpha=5.0).fit([[1], [2], [3]], [2, 4, 6])
    path = str(tmp_path / "nested" / "model.joblib")

    ml_models.save_model(model, path)
    loaded = ml_models.load_model(path)

    assert loaded.alpha == 5.0
    assert loaded.predict([[4]])[0] == pytest.approx(model.predict([[4]])[0])


def test_load_model_unwraps_bundle_to_estimator(tmp_path):
    model = Ridge(alpha=2.0).fit([[1], [2]], [2, 4])
    path = str(tmp_path / "bundle.joblib")
    ml_models.save_model_bundle(
        model, path,
        model_type="test", model_version="v1",
        feature_columns=["x"], training_data_start="2026-01-01", training_data_cutoff="2026-06-01",
        hyperparameters={"alpha": 2.0}, calibration_method=None,
        validation_summary={"mae": 0.1}, training_code_sha="abc123",
    )

    loaded = ml_models.load_model(path)
    assert loaded.alpha == 2.0
    assert loaded.predict([[3]])[0] == pytest.approx(model.predict([[3]])[0])


def test_load_legacy_plain_estimator_still_works(tmp_path):
    path = str(tmp_path / "legacy.joblib")
    plain = Ridge(alpha=1.5).fit([[1], [2]], [1, 2])
    joblib = __import__("joblib")
    joblib.dump(plain, path)

    loaded = ml_models.load_model(path)
    assert loaded.alpha == 1.5

    bundle = ml_models.load_model_bundle(path)
    assert bundle["artifact_id"] == "legacy"
    assert bundle["is_legacy_plain_estimator"] is True
    assert bundle["estimator"].alpha == 1.5


def test_save_and_load_model_bundle_round_trips_metadata(tmp_path):
    model = Ridge(alpha=3.0).fit([[1], [2]], [2, 4])
    path = str(tmp_path / "bundle.joblib")
    saved = ml_models.save_model_bundle(
        model, path,
        model_type="dfs_hitter",
        model_version="v1",
        feature_columns=["a", "b"],
        training_data_start="2026-03-01",
        training_data_cutoff="2026-07-01",
        hyperparameters={"alpha": 3.0},
        calibration_method="none",
        validation_summary={"n": 10},
        training_code_sha="deadbeef",
    )

    loaded = ml_models.load_model_bundle(path)
    assert loaded["format"] == ml_models.MODEL_BUNDLE_FORMAT
    assert loaded["model_type"] == "dfs_hitter"
    assert loaded["model_version"] == "v1"
    assert loaded["feature_columns"] == ["a", "b"]
    assert loaded["feature_schema_hash"] == ml_models.feature_schema_hash(["a", "b"])
    assert loaded["training_data_cutoff"] == "2026-07-01"
    assert loaded["training_code_sha"] == "deadbeef"
    assert loaded["artifact_id"] == saved["artifact_id"]
    assert loaded["estimator"].alpha == 3.0


def test_feature_schema_hash_is_deterministic_and_order_sensitive():
    a = ml_models.feature_schema_hash(["WAVE", "PA_L", "PA_R"])
    b = ml_models.feature_schema_hash(["WAVE", "PA_L", "PA_R"])
    c = ml_models.feature_schema_hash(["PA_L", "WAVE", "PA_R"])
    assert a == b
    assert a != c
    assert len(a) == 64


def test_compute_artifact_id_is_stable_for_same_metadata():
    kwargs = dict(
        model_type="hitter_hit_probability",
        model_version="v4",
        feature_schema_hash_value=ml_models.feature_schema_hash(["a", "b"]),
        training_data_cutoff="2026-07-01",
        hyperparameters={"C": 1.0},
        calibration_method="isotonic",
        training_code_sha="abc",
    )
    assert ml_models.compute_artifact_id(**kwargs) == ml_models.compute_artifact_id(**kwargs)


def test_inspect_model_path_missing_and_corrupt(tmp_path):
    missing = ml_models.inspect_model_path(str(tmp_path / "nope.joblib"))
    assert missing["loaded"] is False
    assert missing["fallback_used"] is True
    assert missing["fallback_reason"] == "missing_artifact"

    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_text("not a real joblib file")
    bad = ml_models.inspect_model_path(str(corrupt))
    assert bad["loaded"] is False
    assert bad["fallback_reason"] == "load_error"


def test_resolve_code_sha_uses_github_sha_env(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "actionsha123")
    assert ml_models.resolve_code_sha() == "actionsha123"


def test_resolve_code_sha_unknown_when_git_unavailable(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def raise_os_error(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(ml_models.subprocess, "run", raise_os_error)
    assert ml_models.resolve_code_sha() == "unknown"


def test_load_model_missing_file_returns_none(tmp_path):
    assert ml_models.load_model(str(tmp_path / "does_not_exist.joblib")) is None


def test_load_model_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_text("not a real joblib file")
    assert ml_models.load_model(str(path)) is None


def _compressed_probability_dataset(seed: int = 0, n: int = 2000, compression: float = 0.3):
    """A real, deterministic stand-in for the real bug this was built to
    fix (quant-analytics "dig into calibration"): a raw probability
    compressed toward 0.5 relative to the TRUE generating probability
    (compression=0.3 means the raw score only expresses 30% of the real
    log-odds spread) - same shape as game_picks.compute_game_win_probability's
    real, measured under-spread (std 0.035 vs the real market's 0.059 on
    the same games). Returns (raw_probability, actual_0_1, true_probability)."""
    rng = np.random.RandomState(seed)
    true_logit = rng.normal(scale=1.5, size=n)
    true_p = 1 / (1 + np.exp(-true_logit))
    actual = rng.binomial(1, true_p)
    raw_p = 1 / (1 + np.exp(-compression * true_logit))
    return pd.Series(raw_p), pd.Series(actual), true_p


def test_fit_probability_calibration_isotonic_corrects_a_compressed_probability():
    raw_p, actual, true_p = _compressed_probability_dataset()
    raw_brier_vs_truth = np.mean((raw_p.to_numpy() - true_p) ** 2)

    calibrator = ml_models.fit_probability_calibration(raw_p, actual, method="isotonic")
    calibrated_p = calibrator.predict(raw_p.to_numpy())
    calibrated_brier_vs_truth = np.mean((calibrated_p - true_p) ** 2)

    # Same "score against known ground truth, not just observed outcomes"
    # discipline as test_fit_calibrated_improves_brier_score_for_an_overconfident_classifier.
    assert calibrated_brier_vs_truth < raw_brier_vs_truth
    # The real point of this fix: calibration should widen the compressed
    # spread back out toward the true one, not just shift it around.
    assert calibrated_p.std() > raw_p.std()


def test_fit_probability_calibration_sigmoid_corrects_a_compressed_probability():
    raw_p, actual, true_p = _compressed_probability_dataset()
    raw_brier_vs_truth = np.mean((raw_p.to_numpy() - true_p) ** 2)

    calibrator = ml_models.fit_probability_calibration(raw_p, actual, method="sigmoid")
    calibrated_p = calibrator.predict(raw_p.to_numpy())
    calibrated_brier_vs_truth = np.mean((calibrated_p - true_p) ** 2)

    assert calibrated_brier_vs_truth < raw_brier_vs_truth
    assert calibrated_p.std() > raw_p.std()


def test_fit_probability_calibration_unknown_method_raises():
    raw_p, actual, _ = _compressed_probability_dataset(n=50)
    with pytest.raises(ValueError):
        ml_models.fit_probability_calibration(raw_p, actual, method="not-a-real-method")


def test_fit_probability_calibration_isotonic_predict_is_monotonic():
    raw_p, actual, _ = _compressed_probability_dataset()
    calibrator = ml_models.fit_probability_calibration(raw_p, actual, method="isotonic")
    grid = np.linspace(0.01, 0.99, 50)
    calibrated = calibrator.predict(grid)
    # Isotonic regression is monotonic by construction - a real, structural
    # property, not just a numeric coincidence of this dataset.
    assert (np.diff(calibrated) >= 0).all()
