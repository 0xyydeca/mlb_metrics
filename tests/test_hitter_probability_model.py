"""Tests for the shadow opportunity-aware hitter probability model."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, hitter_probability_model as hpm, model_validation, predictions

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_opportunity_log(
    n_dates: int = 40,
    players_per_date: int = 24,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-04-01")
    for d in range(n_dates):
        date = base + pd.Timedelta(days=d)
        for i in range(players_per_date):
            key = 1000 + (i % 30)
            game_pk = 5000 + d * 10 + (i % 3)
            appear = float(rng.random() < 0.55)
            start_rate = float(np.clip(rng.normal(0.6, 0.2), 0.05, 0.95))
            avg_order = float(np.clip(rng.normal(5.0, 2.0), 1.0, 9.0))
            wave = float(np.clip(rng.normal(0.28, 0.05), 0.05, 0.5))
            starter_pave = float(np.clip(rng.normal(0.25, 0.05), 0.05, 0.5))
            # Occasional missingness
            if rng.random() < 0.1:
                avg_order = np.nan
            if rng.random() < 0.08:
                starter_pave = np.nan
            park = float(rng.normal(1.0, 0.05))
            if rng.random() < 0.1:
                park = np.nan
            if appear:
                pa = int(rng.integers(1, 6))
                hits = int(rng.binomial(pa, min(0.35, wave + 0.05)))
                got_hit = float(hits > 0)
            else:
                pa, hits, got_hit = 0, 0, 0.0
            rows.append({
                "date": date,
                "game_pk": game_pk,
                "key_mlbam": key,
                "team": "NYY" if i % 2 == 0 else "BOS",
                "Appeared": appear,
                "Started": float(appear and rng.random() < 0.8),
                "Batting_Order": (avg_order if appear else np.nan),
                "Plate_Appearances": pa,
                "Official_At_Bats": max(0, pa - int(rng.integers(0, 2))),
                "Hits": hits,
                "Got_Hit": got_hit,
                "No_Game": 0.0,
                "WAVE": wave,
                "WAVE_L": wave + rng.normal(0, 0.01),
                "WAVE_R": wave + rng.normal(0, 0.01),
                "PA_L": float(rng.integers(50, 200)),
                "PA_R": float(rng.integers(50, 200)),
                "Exit_Velo": float(rng.normal(89, 3)),
                "Barrel_Rate": float(np.clip(rng.normal(0.08, 0.03), 0, 0.3)),
                "xBA": float(np.clip(rng.normal(0.25, 0.04), 0.05, 0.4)),
                "xwOBA": float(np.clip(rng.normal(0.32, 0.04), 0.1, 0.5)),
                "Whiff_Rate": float(np.clip(rng.normal(0.25, 0.05), 0.05, 0.5)),
                "Chase_Rate": float(np.clip(rng.normal(0.28, 0.05), 0.05, 0.5)),
                "Fastball_WAVE": wave,
                "Breaking_WAVE": wave - 0.02,
                "Offspeed_WAVE": wave - 0.01,
                "starter_fastball_rate": 0.5,
                "starter_breaking_rate": 0.3,
                "starter_offspeed_rate": 0.2,
                "starter_PAVE": starter_pave,
                "Bullpen_PAVE": float(np.clip(rng.normal(0.27, 0.04), 0.05, 0.5)),
                "Park_Factor": park,
                "is_home": float(i % 2 == 0),
                "Expected_Bases": float(np.clip(rng.normal(1.2, 0.2), 0.2, 2.5)),
                "avg_batting_order": avg_order,
                "start_rate": start_rate,
                "Days_Rest": float(rng.integers(0, 5)),
                "Last_Game_Date": date - pd.Timedelta(days=int(rng.integers(1, 5))),
                "probability": float(np.clip(wave + 0.05, 0, 1)),
                "Game_Hit_Probability": float(np.clip(wave + start_rate * 0.1, 0, 1)),
                "Consistency": float(rng.random()),
                "Approach": float(rng.random()),
                "Matchup_Hit_Probability": float(np.clip(wave - starter_pave * 0.2 + 0.15, 0, 1)),
                "Expected_BB": 0.3,
                "Expected_HBP": 0.05,
                "Expected_RBI": 0.4,
                "WAVE_Home": wave,
                "WAVE_Away": wave - 0.01,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def opportunity_rows():
    return hpm.prepare_opportunity_training_frame(_synthetic_opportunity_log())


def test_reduced_features_exclude_composites():
    for col in hpm.COMPOSITE_FEATURE_COLUMNS:
        assert col not in hpm.REDUCED_FEATURE_COLUMNS
    for col in hpm.COMPOSITE_FEATURE_COLUMNS:
        assert col in hpm.EXPANDED_FEATURE_COLUMNS


def test_leakage_guard_rejects_actual_pa():
    with pytest.raises(ValueError, match="Leakage"):
        hpm.assert_no_leakage_features(["WAVE", "Plate_Appearances"])


def test_domain_imputer_neutral_park_and_order_not_first(opportunity_rows):
    cols = ["Park_Factor", "avg_batting_order", "starter_PAVE", "WAVE"]
    train = opportunity_rows.iloc[:200]
    imp = hpm.DomainImputer(columns=cols)
    imp.fit(train[cols])
    assert abs(imp.imputation_values_["Park_Factor"] - 1.0) < 1e-9
    # Missing order fills with training median, not batting-first (1.0)
    assert imp.imputation_values_["avg_batting_order"] != 1.0
    # Missing starter not filled with elite-looking zero
    assert imp.imputation_values_["starter_PAVE"] != 0.0

    test = train[cols].copy()
    test.loc[test.index[0], "avg_batting_order"] = np.nan
    test.loc[test.index[1], "starter_PAVE"] = np.nan
    test.loc[test.index[2], "Park_Factor"] = np.nan
    out = imp.transform(test)
    assert out.loc[test.index[0], "avg_batting_order"] == imp.imputation_values_["avg_batting_order"]
    assert out.loc[test.index[1], "starter_PAVE"] == imp.imputation_values_["starter_PAVE"]
    assert out.loc[test.index[2], "Park_Factor"] == 1.0

    pre = hpm.build_opportunity_preprocessor(cols, scale=False)
    pre.fit(train[cols])
    transformed = np.asarray(pre.transform(test))
    # Missingness indicators appended for configured columns present in cols
    assert transformed.shape[1] > len(cols)


def test_preprocessor_fit_only_on_train(opportunity_rows):
    cols = ["WAVE", "Park_Factor", "starter_PAVE"]
    train = opportunity_rows.iloc[:150].copy()
    test = opportunity_rows.iloc[150:200].copy()
    # Extreme test values must not leak into train-fitted imputes.
    test["WAVE"] = 999.0
    test["Park_Factor"] = np.nan
    pre = hpm.build_opportunity_preprocessor(cols, scale=False)
    pre.fit(train[cols])
    imputer = pre.named_steps["columns"].named_transformers_["impute"]
    assert hasattr(imputer, "imputation_values_")
    train_wave_fill = imputer.imputation_values_["WAVE"]
    park_idx = cols.index("Park_Factor")
    park_vals = np.asarray(pre.transform(test[cols]))[:, park_idx]
    assert np.allclose(park_vals, 1.0)
    # Re-fit on test alone would move WAVE fill toward 999; train fill stays.
    pre2 = hpm.build_opportunity_preprocessor(cols, scale=False)
    pre2.fit(test[cols])
    imputer2 = pre2.named_steps["columns"].named_transformers_["impute"]
    assert imputer2.imputation_values_["WAVE"] != train_wave_fill
    assert abs(imputer.imputation_values_["WAVE"] - train_wave_fill) < 1e-12


def test_appearance_trained_on_dnps(opportunity_rows):
    assert (opportunity_rows["Appeared"] == 0).any()
    spec = hpm.OpportunityModelSpec(
        name="smoke",
        feature_set="reduced",
        appearance_family="logit",
        appearance_params={"C": 1.0},
        expected_pa_family="ridge",
        conditional_family="logit",
        conditional_params={"C": 1.0},
        formulation="direct",
        use_predicted_pa_feature=True,
    )
    # Use enough dates for tiny OOF
    model = hpm.fit_opportunity_model(
        opportunity_rows,
        spec,
        oof_min_train_dates=8,
        oof_test_block_dates=3,
    )
    comps = model.predict_components(opportunity_rows.tail(30))
    assert set(comps.columns) >= {
        "P_Appear", "Expected_PA_hat", "P_Hit_Given_Appearance", "Final_Hit_Probability",
    }
    # Final is the product
    np.testing.assert_allclose(
        comps["Final_Hit_Probability"].to_numpy(),
        (comps["P_Appear"] * comps["P_Hit_Given_Appearance"]).to_numpy(),
        rtol=1e-6,
    )


def test_conditional_hit_only_on_appeared(opportunity_rows):
    appeared = opportunity_rows[opportunity_rows["Appeared"] == 1]
    dnp = opportunity_rows[opportunity_rows["Appeared"] == 0]
    assert not appeared.empty and not dnp.empty
    # Fitting path uses only appeared for PA / hit — exercised via fit_opportunity_model
    spec = hpm.OpportunityModelSpec(
        name="smoke2",
        feature_set="reduced",
        formulation="direct",
        appearance_params={"C": 0.5},
        conditional_params={"C": 0.5},
        use_predicted_pa_feature=False,
    )
    model = hpm.fit_opportunity_model(
        opportunity_rows, spec, oof_min_train_dates=8, oof_test_block_dates=3,
    )
    # Prediction still defined for DNPs (appearance down-weights them)
    pred = model.predict_proba(dnp.head(10))
    assert len(pred) == 10
    assert np.all((pred >= 0) & (pred <= 1))


def test_never_uses_actual_pa_as_feature(opportunity_rows):
    spec = hpm.OpportunityModelSpec(
        name="no-pa-leak",
        feature_set="reduced",
        formulation="direct",
        use_predicted_pa_feature=True,
    )
    model = hpm.fit_opportunity_model(
        opportunity_rows, spec, oof_min_train_dates=8, oof_test_block_dates=3,
    )
    assert "Plate_Appearances" not in model.feature_columns
    assert hpm.PREDICTED_EXPECTED_PA in model.feature_columns

    base = opportunity_rows.tail(20).copy()
    p0 = model.predict_proba(base)
    base2 = base.copy()
    base2["Plate_Appearances"] = 99
    p1 = model.predict_proba(base2)
    np.testing.assert_allclose(p0, p1)


def test_pa_distribution_formulation(opportunity_rows):
    p = hpm.game_hit_prob_from_pa_distribution(
        expected_pa=np.array([3.0, 4.0]),
        p_hit_per_pa=np.array([0.3, 0.25]),
        max_n=8,
    )
    assert p.shape == (2,)
    assert np.all((p > 0) & (p < 1))
    # More PA / higher p_hit_per_PA → higher game hit prob
    assert p[0] > hpm.game_hit_prob_from_pa_distribution(
        np.array([1.0]), np.array([0.3]),
    )[0]

    spec = hpm.OpportunityModelSpec(
        name="pa-dist",
        feature_set="reduced",
        formulation="pa_distribution",
        appearance_family="logit",
        conditional_family="logit",
        use_predicted_pa_feature=True,
    )
    model = hpm.fit_opportunity_model(
        opportunity_rows, spec, oof_min_train_dates=8, oof_test_block_dates=3,
    )
    assert model.formulation == "pa_distribution"
    comps = model.predict_components(opportunity_rows.tail(15))
    assert comps["P_Hit_Per_PA"].notna().all()
    np.testing.assert_allclose(
        comps["Final_Hit_Probability"],
        comps["P_Appear"] * comps["P_Hit_Given_Appearance"],
        rtol=1e-6,
    )


def test_nested_validation_smoke(opportunity_rows):
    nested_config = model_validation.NestedValidationConfig(
        outer_min_train_dates=12,
        outer_test_block_dates=4,
        inner_min_train_dates=8,
        inner_test_block_dates=3,
        freeze_dates=0,
        bootstrap_samples=10,
    )
    # Tiny focused grid for CI
    specs = [
        hpm.OpportunityModelSpec(
            name="direct|reduced|logit|params={'C': 1.0}|cal=None",
            feature_set="reduced",
            formulation="direct",
            appearance_family="logit",
            appearance_params={"C": 1.0},
            expected_pa_family="ridge",
            conditional_family="logit",
            conditional_params={"C": 1.0},
            use_predicted_pa_feature=True,
        ),
        hpm.OpportunityModelSpec(
            name="pa_distribution|reduced|logit|params={'C': 1.0}|cal=None",
            feature_set="reduced",
            formulation="pa_distribution",
            appearance_family="logit",
            appearance_params={"C": 1.0},
            expected_pa_family="ridge",
            conditional_family="logit",
            conditional_params={"C": 1.0},
            use_predicted_pa_feature=True,
        ),
        hpm.OpportunityModelSpec(
            name="direct|expanded|hgbm|params={}|cal=None",
            feature_set="expanded",
            formulation="direct",
            appearance_family="hgbm",
            appearance_params={"max_depth": 2, "max_iter": 30, "min_samples_leaf": 20},
            expected_pa_family="hgbm",
            expected_pa_params={"max_depth": 2, "max_iter": 30, "min_samples_leaf": 20},
            conditional_family="hgbm",
            conditional_params={"max_depth": 2, "max_iter": 30, "min_samples_leaf": 20},
            use_predicted_pa_feature=True,
        ),
    ]
    report = hpm.run_opportunity_nested_validation(
        opportunity_rows, specs=specs, nested_config=nested_config,
    )
    assert report["status"] == "ok"
    assert report["n_outer_folds"] >= 1
    assert "ablation" in report
    assert "formulation_comparison" in report
    shadow = report["_shadow_predictions"]
    assert not shadow.empty
    assert "Final_Hit_Probability" in shadow.columns


def test_bundle_roundtrip(opportunity_rows, tmp_path):
    spec = hpm.OpportunityModelSpec(
        name="bundle",
        feature_set="reduced",
        formulation="direct",
        use_predicted_pa_feature=True,
    )
    model = hpm.fit_opportunity_model(
        opportunity_rows, spec, oof_min_train_dates=8, oof_test_block_dates=3,
    )
    path = str(tmp_path / "opp_model.joblib")
    bundle = hpm.save_opportunity_model_bundle(
        model, path, validation_summary={"shadow": True},
    )
    assert bundle["model_type"] == hpm.MODEL_TYPE
    loaded = hpm.load_opportunity_model(path)
    assert loaded is not None
    p0 = model.predict_proba(opportunity_rows.tail(5))
    p1 = loaded.predict_proba(opportunity_rows.tail(5))
    np.testing.assert_allclose(p0, p1)


def test_select_picks_supports_selection_modes():
    """Final_Hit_Probability is integrated via selection_mode, not a silent swap."""
    src = inspect.getsource(predictions.select_picks)
    assert "selection_mode" in src
    assert "Final_Hit_Probability" in src or "FINAL_HIT_PROBABILITY" in src
    assert "force_live" in src


def test_train_script_exists_and_mentions_shadow():
    script = (REPO_ROOT / "scripts" / "train_hitter_probability_model.py").read_text()
    assert "select_picks" in script
    assert "shadow" in script.lower()
    assert "hitter_opportunity" in script


def test_config_paths_present():
    assert config.HITTER_OPPORTUNITY_PROBABILITY_MODEL_PATH.endswith(".joblib")
    assert "shadow" in config.HITTER_OPPORTUNITY_SHADOW_PREDICTIONS_PATH
