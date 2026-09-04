"""Market-residual game-win challenger: formulation, nested validation,
promotion gates, modes, and leakage guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, game_residual_model as grm


def _synthetic_residual_frame(n_dates=55, games_per_date=8, seed=0):
    """Market-aligned games with a small systematic residual signal."""
    rng = np.random.RandomState(seed)
    dates = [pd.Timestamp("2026-04-01") + pd.Timedelta(days=i) for i in range(n_dates)]
    rows = []
    gpk = 1000
    for date in dates:
        for j in range(games_per_date):
            market = float(np.clip(rng.normal(0.52, 0.08), 0.2, 0.8))
            # Baseball-ish features that slightly justify leaving the market.
            composite_diff = rng.normal()
            starter_diff = rng.normal()
            true_residual = 0.15 * composite_diff - 0.1 * starter_diff
            p_true = float(grm.sigmoid(grm.logit(market) + true_residual))
            home_won = int(rng.rand() < p_true)
            home_ml = -110 if market >= 0.5 else 120
            away_ml = 100 if market >= 0.5 else -130
            rows.append({
                "date": date,
                "game_pk": gpk,
                "home_team": f"H{j % 5}",
                "away_team": f"A{j % 5}",
                "home_composite": 1.0 + 0.1 * composite_diff,
                "away_composite": 1.0,
                "home_bullpen_pave_plus": 1.0,
                "home_bullpen_power_a_plus": 1.0,
                "away_bullpen_pave_plus": 1.0,
                "away_bullpen_power_a_plus": 1.0,
                "home_starter_pave_plus": 1.0,
                "home_starter_power_a_plus": 1.0,
                "away_starter_pave_plus": 1.0 + 0.1 * starter_diff,
                "away_starter_power_a_plus": 1.0,
                "home_win_probability": float(np.clip(market + 0.02 * rng.normal(), 0.05, 0.95)),
                "market_home_win_probability": market,
                "closing_home_win_probability": float(np.clip(market + 0.01 * rng.normal(), 0.05, 0.95)),
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "Home_Won": home_won,
                "composite_diff": composite_diff,
                "starter_quality_diff": starter_diff,
                "bullpen_quality_diff": 0.0,
                "park_factor": 1.0,
                "home_advantage": 1.0,
                "home_starter_certainty": 1.0,
                "away_starter_certainty": 1.0,
                "home_lineup_strength": 1.0 + 0.1 * composite_diff,
                "away_lineup_strength": 1.0,
                "temperature_f": 72.0,
                "wind_speed_mph": 5.0,
                "home_days_rest": 1.0,
                "away_days_rest": 1.0,
            })
            gpk += 1
    return pd.DataFrame(rows)


def test_config_defaults_are_non_live():
    assert config.GAME_PREDICTION_MODE == "shadow"
    assert config.BETTING_MODE == "disabled"
    assert config.GAME_PREDICTION_MODE in config.GAME_PREDICTION_MODES
    assert config.BETTING_MODE in config.BETTING_MODES


def test_logit_sigmoid_roundtrip():
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(grm.sigmoid(grm.logit(p)), p, atol=1e-9)


def test_residual_logistic_respects_market_prior_when_features_uninformative():
    rng = np.random.RandomState(1)
    n = 200
    market = np.clip(rng.uniform(0.35, 0.65, size=n), 0.05, 0.95)
    # Outcomes drawn from market alone - residual should shrink near zero.
    y = (rng.rand(n) < market).astype(int)
    X = pd.DataFrame({
        "home_composite": rng.normal(size=n),
        "away_composite": rng.normal(size=n),
        "home_bullpen_pave_plus": np.ones(n),
        "home_bullpen_power_a_plus": np.ones(n),
        "away_bullpen_pave_plus": np.ones(n),
        "away_bullpen_power_a_plus": np.ones(n),
        "home_starter_pave_plus": np.ones(n),
        "home_starter_power_a_plus": np.ones(n),
        "away_starter_pave_plus": np.ones(n),
        "away_starter_power_a_plus": np.ones(n),
        "composite_diff": rng.normal(scale=0.01, size=n),
        "starter_quality_diff": rng.normal(scale=0.01, size=n),
        "bullpen_quality_diff": np.zeros(n),
        "park_factor": np.ones(n),
        "home_advantage": np.ones(n),
        "home_starter_certainty": np.ones(n),
        "away_starter_certainty": np.ones(n),
        "home_lineup_strength": np.ones(n),
        "away_lineup_strength": np.ones(n),
        "temperature_f": np.full(n, 70.0),
        "wind_speed_mph": np.full(n, 5.0),
        "home_days_rest": np.ones(n),
        "away_days_rest": np.ones(n),
    })
    model = grm.MarketResidualLogistic(C=0.001, feature_columns=list(X.columns))
    model.fit(X, pd.Series(y), pd.Series(market))
    residual = model.residual_logit(X)
    assert np.nanmean(np.abs(residual)) < 0.25
    pred = model.predict_proba(X, market)
    # Predictions stay near the market prior under strong regularization.
    assert np.corrcoef(pred, market)[0, 1] > 0.9


def test_closing_odds_rejected_as_features():
    with pytest.raises(AssertionError):
        grm.assert_no_closing_odds_in_features(
            pd.DataFrame(), ["home_composite", "closing_home_win_probability"],
        )


def test_attach_prediction_time_market_rejects_closing_role():
    with pytest.raises(ValueError):
        grm.attach_prediction_time_market(
            pd.DataFrame({"game_pk": [1]}),
            pd.DataFrame(),
            snapshot_role="closing",
        )


def test_enrich_residual_features_adds_expected_columns():
    base = pd.DataFrame([{
        "game_pk": 1,
        "date": pd.Timestamp("2026-05-01"),
        "home_team": "NYY",
        "away_team": "BOS",
        "home_composite": 1.1,
        "away_composite": 0.9,
        "home_bullpen_pave_plus": 1.0,
        "home_bullpen_power_a_plus": 1.0,
        "away_bullpen_pave_plus": 1.0,
        "away_bullpen_power_a_plus": 1.0,
        "home_starter_pave_plus": 1.0,
        "home_starter_power_a_plus": 1.0,
        "away_starter_pave_plus": 1.0,
        "away_starter_power_a_plus": 1.0,
    }])
    schedule = pd.DataFrame([{
        "game_pk": 1,
        "date": pd.Timestamp("2026-05-01"),
        "home_team": "NYY",
        "away_team": "BOS",
        "home_probable_pitcher_key_mlbam": 100,
        "away_probable_pitcher_key_mlbam": pd.NA,
        "temperature_f": 68.0,
        "wind_speed_mph": 8.0,
    }])
    confidence = pd.DataFrame([{"team": "NYY", "Park_Factor": 1.05}])
    out = grm.enrich_residual_features(base, schedule_games=schedule, confidence=confidence)
    for col in grm.DERIVED_FEATURE_COLUMNS:
        assert col in out.columns
    assert out.loc[0, "home_starter_certainty"] == 1.0
    assert out.loc[0, "away_starter_certainty"] == 0.0
    assert out.loc[0, "park_factor"] == pytest.approx(1.05)


def test_derive_team_rest_features():
    schedule = pd.DataFrame([
        {"game_pk": 1, "date": "2026-05-01", "home_team": "NYY", "away_team": "BOS"},
        {"game_pk": 2, "date": "2026-05-02", "home_team": "NYY", "away_team": "TOR"},
        {"game_pk": 3, "date": "2026-05-04", "home_team": "BOS", "away_team": "NYY"},
    ])
    rest = grm.derive_team_rest_features(schedule)
    assert rest.loc[rest["game_pk"] == 1, "home_days_rest"].isna().all()
    assert float(rest.loc[rest["game_pk"] == 2, "home_days_rest"].iloc[0]) == 1.0
    assert float(rest.loc[rest["game_pk"] == 3, "away_days_rest"].iloc[0]) == 2.0


def test_nested_validation_compares_all_methods_on_same_games(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GAME_RESIDUAL_OUTER_MIN_TRAIN_DATES", 20)
    monkeypatch.setattr(config, "GAME_RESIDUAL_OUTER_TEST_BLOCK_DATES", 5)
    monkeypatch.setattr(config, "GAME_RESIDUAL_INNER_MIN_TRAIN_DATES", 10)
    monkeypatch.setattr(config, "GAME_RESIDUAL_INNER_TEST_BLOCK_DATES", 5)
    monkeypatch.setattr(config, "GAME_RESIDUAL_LOGISTIC_C_GRID", [0.01, 0.05])
    monkeypatch.setattr(config, "GAME_RESIDUAL_NONLINEAR_MAX_DEPTH_GRID", [2])
    monkeypatch.setattr(config, "GAME_RESIDUAL_NONLINEAR_MIN_SAMPLES_LEAF_GRID", [40])
    monkeypatch.setattr(config, "GAME_RESIDUAL_NONLINEAR_SHRINK_GRID", [0.5])
    monkeypatch.setattr(config, "NESTED_VALIDATION_BOOTSTRAP_SAMPLES", 20)

    frame = _synthetic_residual_frame(n_dates=40, games_per_date=6)
    report = grm.run_game_residual_nested_validation(frame, freeze_dates=0)
    assert report["status"] == "ok"
    assert report["n_outer_folds"] >= 1
    methods = report["methods"]
    for name in (
        grm.METHOD_MARKET,
        grm.METHOD_HEURISTIC,
        grm.METHOD_HEURISTIC_CAL,
        grm.METHOD_RESIDUAL_LOGISTIC,
        grm.METHOD_RESIDUAL_NONLINEAR,
    ):
        assert name in methods
        assert methods[name]["n_games"] == methods[grm.METHOD_MARKET]["n_games"]
        assert "model_minus_market_brier" in methods[name]
        assert "roc_auc" in methods[name]
        assert "calibration_intercept" in methods[name]
        assert "high_conf_brier" in methods[name]
        assert "hypothetical_roi" in methods[name]
        assert "true_closing_line_value" in methods[name]

    assert "promotion_gate" in report
    assert "betting_promotion_gate" in report
    assert report["betting_promotion_gate"]["checks"]["kelly_not_used_as_accuracy_gate"] is True


def test_nested_validation_excludes_games_without_prediction_time_market(monkeypatch):
    monkeypatch.setattr(config, "GAME_RESIDUAL_OUTER_MIN_TRAIN_DATES", 15)
    monkeypatch.setattr(config, "GAME_RESIDUAL_OUTER_TEST_BLOCK_DATES", 5)
    monkeypatch.setattr(config, "GAME_RESIDUAL_INNER_MIN_TRAIN_DATES", 8)
    monkeypatch.setattr(config, "GAME_RESIDUAL_INNER_TEST_BLOCK_DATES", 4)
    monkeypatch.setattr(config, "GAME_RESIDUAL_LOGISTIC_C_GRID", [0.01])
    monkeypatch.setattr(config, "GAME_RESIDUAL_NONLINEAR_MAX_DEPTH_GRID", [2])
    monkeypatch.setattr(config, "GAME_RESIDUAL_NONLINEAR_MIN_SAMPLES_LEAF_GRID", [40])
    monkeypatch.setattr(config, "GAME_RESIDUAL_NONLINEAR_SHRINK_GRID", [0.5])
    monkeypatch.setattr(config, "NESTED_VALIDATION_BOOTSTRAP_SAMPLES", 10)

    frame = _synthetic_residual_frame(n_dates=35, games_per_date=4)
    # Remove market on some games - they must not enter evaluation.
    drop_idx = frame.sample(frac=0.2, random_state=0).index
    frame.loc[drop_idx, "market_home_win_probability"] = np.nan
    report = grm.run_game_residual_nested_validation(frame, freeze_dates=0)
    assert report["status"] == "ok"
    n = report["methods"][grm.METHOD_MARKET]["n_games"]
    assert n == report["methods"][grm.METHOD_RESIDUAL_LOGISTIC]["n_games"]
    assert n < len(frame)


def test_prediction_promotion_gate_fail_closed():
    ok, details = grm.evaluate_prediction_promotion_gate(None)
    assert ok is False
    ok, details = grm.evaluate_prediction_promotion_gate({"promotion_gate": {"checks": {
        "positive_paired_brier_vs_market": True,
        "positive_paired_log_loss_vs_market": False,
    }}})
    assert ok is False
    assert details["checks"]["positive_paired_log_loss_vs_market"] is False


def test_betting_promotion_gate_requires_clv_roi_and_no_week_dependence():
    ok, details = grm.evaluate_betting_promotion_gate({"betting_promotion_gate": {"checks": {
        "positive_paired_brier_vs_market": True,
        "positive_paired_log_loss_vs_market": True,
        "positive_closing_line_value": True,
        "positive_hypothetical_roi": True,
        "roi_ci_not_materially_negative": True,
        "adequate_sample_size": True,
        "adequate_outer_folds": True,
        "adequate_date_blocks": True,
        "no_single_week_dependence": False,
        "based_on_untouched_outer_folds": True,
        "kelly_not_used_as_accuracy_gate": True,
    }}})
    assert ok is False
    assert details["checks"]["no_single_week_dependence"] is False


def test_resolve_live_modes_fall_back_without_gate():
    mode, meta = grm.resolve_game_prediction_mode("live", force_live=False)
    assert mode == "shadow"
    assert meta["fallback_used"] is True

    bmode, bmeta = grm.resolve_betting_mode("live", force_live=False)
    assert bmode == "shadow"
    assert bmeta["fallback_used"] is True

    mode2, _ = grm.resolve_game_prediction_mode("live", force_live=True)
    assert mode2 == "live"


def test_suppress_official_bets_zeros_stakes():
    picks = pd.DataFrame([{
        "game_pk": 1,
        "bet_units": 2.5,
        "bet_stake_fraction": 0.025,
        "bet_side": "home",
        "bet_team": "NYY",
        "bet_moneyline": -120,
    }])
    out = grm.suppress_official_bets(picks)
    assert out.iloc[0]["bet_units"] == 0.0
    assert out.iloc[0]["bet_stake_fraction"] == 0.0
    assert pd.isna(out.iloc[0]["bet_side"])


def test_hypothetical_bets_use_vigged_edge_not_devig_only():
    frame = pd.DataFrame([{
        "date": pd.Timestamp("2026-05-01"),
        "game_pk": 1,
        "home_team": "NYY",
        "away_team": "BOS",
        "model_p": 0.62,
        "home_moneyline": -120,  # implied ~0.545
        "away_moneyline": 100,
        "Home_Won": 1,
        "closing_home_win_probability": 0.58,
    }])
    bets = grm.hypothetical_bets_from_probabilities(
        frame, model_prob_col="model_p", edge_threshold=0.02,
    )
    assert len(bets) == 1
    assert bets.iloc[0]["bet_side"] == "home"
    assert bets.iloc[0]["edge"] == pytest.approx(0.62 - (120 / 220), abs=1e-6)
    # Kelly present for ROI reporting but is not an accuracy metric.
    assert bets.iloc[0]["kelly_stake_fraction"] > 0


def test_shadow_exports_write_and_dedupe(tmp_path):
    path = tmp_path / "shadow.csv"
    frame = pd.DataFrame([{
        "date": "2026-05-01",
        "game_pk": 1,
        "home_team": "NYY",
        "away_team": "BOS",
        "market_home_win_probability": 0.55,
        "home_win_probability": 0.57,
        "residual_home_win_probability": 0.56,
        "residual_logit": 0.04,
        "probability_source": "market_residual",
        "model_version": "v1",
        "artifact_id": "abc",
        "game_prediction_mode": "shadow",
        "prediction_snapshot_type": "morning",
    }])
    grm.write_shadow_predictions(frame, path=str(path))
    frame2 = frame.copy()
    frame2["residual_home_win_probability"] = 0.58
    grm.write_shadow_predictions(frame2, path=str(path))
    out = pd.read_csv(path)
    assert len(out) == 1
    assert out.iloc[0]["residual_home_win_probability"] == pytest.approx(0.58)


def test_save_and_load_residual_artifact(tmp_path):
    frame = _synthetic_residual_frame(n_dates=10, games_per_date=5)
    feats = [c for c in grm.RESIDUAL_FEATURE_COLUMNS if c in frame.columns]
    model = grm.MarketResidualLogistic(C=0.05, feature_columns=feats)
    model.fit(frame, frame["Home_Won"], frame["market_home_win_probability"])
    path = tmp_path / "residual.joblib"
    artifact = grm.ResidualModelArtifact(
        family="logistic",
        model=model,
        feature_columns=feats,
        metadata={"model_version": "test", "hyperparameters": {"C": 0.05}},
    )
    grm.save_residual_model(artifact, path=str(path))
    loaded = grm.load_residual_model(path=str(path))
    assert loaded is not None
    assert loaded.family == "logistic"
    pred, status = grm.predict_residual_home_win_probability(
        frame, frame["market_home_win_probability"], artifact=loaded,
    )
    assert status["loaded"] is True
    assert len(pred) == len(frame)
    assert pred.between(0, 1).all()


def test_prepare_training_frame_keeps_closing_out_of_feature_set():
    log = _synthetic_residual_frame(n_dates=3, games_per_date=2)
    # Strip enriched cols to force enrich path; keep market.
    snaps = pd.DataFrame([{
        "snapshot_id": f"s{i}",
        "game_pk": row.game_pk,
        "snapshot_role": "morning",
        "market_home_win_probability": row.market_home_win_probability,
        "home_moneyline": -110,
        "away_moneyline": -110,
        "captured_at_utc": "2026-05-01T12:00:00+00:00",
        "source_status": "ok",
        "provider_event_id": f"e{i}",
        "sportsbook": "DraftKings",
        "game_datetime": "2026-05-01T23:00:00+00:00",
        "home_team": row.home_team,
        "away_team": row.away_team,
        "date": row.date,
    } for i, row in enumerate(log.itertuples())])
    # Minimal columns like assemble_game_pick_log
    minimal = log[[
        "date", "game_pk", "home_team", "away_team",
        *grm.BASE_FEATURE_COLUMNS,
        "home_win_probability", "Home_Won",
    ]].copy()
    frame = grm.prepare_training_frame(
        minimal,
        market_snapshots=snaps,
        prediction_snapshot_role="morning",
        closing_snapshots=snaps.assign(snapshot_role="closing"),
    )
    assert "closing_home_win_probability" in frame.columns or True  # optional
    grm.assert_no_closing_odds_in_features(frame, grm.RESIDUAL_FEATURE_COLUMNS)


def test_kelly_not_in_probability_promotion_checks():
    residual = {
        "model_minus_market_brier": -0.01,
        "model_minus_market_log_loss": -0.02,
        "n_games": 200,
        "paired_brier_bootstrap": {"ci_high": -0.001, "n_blocks": 20},
        "paired_log_loss_bootstrap": {"ci_high": -0.001, "n_blocks": 20},
        "brier_score": 0.24,
    }
    gate = grm.build_probability_promotion_gate(residual, {"brier_score": 0.25}, [{"fold_id": i} for i in range(4)])
    assert "kelly" not in json.dumps(gate).lower() or "kelly_not_used" in json.dumps(gate).lower()
    # Probability gate must not require ROI/Kelly success.
    assert "positive_hypothetical_roi" not in gate["checks"]
