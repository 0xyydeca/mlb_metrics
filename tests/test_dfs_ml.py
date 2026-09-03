import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlb_metrics import config, dfs, dfs_ml, ml_models


def _wave_row(key_mlbam, team, **overrides):
    row = {
        "key_mlbam": key_mlbam, "team": team,
        "WAVE": 0.28, "WAVE_L": 0.30, "WAVE_R": 0.27, "PA_L": 20, "PA_R": 40,
        "probability": 0.65, "Game_Hit_Probability": 0.70, "Consistency": 0.05,
        "Approach": 0.45, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
        "Fastball_WAVE": 0.30, "Breaking_WAVE": 0.20, "Offspeed_WAVE": 0.15,
        "Whiff_Rate": 0.22, "Chase_Rate": 0.28,
        "WAVE_Home": 0.32, "WAVE_Away": 0.24,
    }
    row.update(overrides)
    return row


def test_build_hitter_features_joins_matchup_ingredients():
    # is_home=True means the game's venue is BOS's own park (see
    # matchup.py's identical venue_team logic) - Park_Factor must be
    # looked up under "BOS", not the opponent "NYY".
    wave = pd.DataFrame([_wave_row(1, "BOS")])
    pave = pd.DataFrame([{
        "key_mlbam": 99, "PAVE": 0.24,
        "Fastball_Rate": 0.60, "Breaking_Rate": 0.25, "Offspeed_Rate": 0.15,
    }])
    confidence = pd.DataFrame([
        {"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95},
        {"team": "BOS", "Bullpen_PAVE": 0.25, "Park_Factor": 1.05},
    ])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}])

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability).set_index("key_mlbam")

    assert result.loc[1, "starter_PAVE"] == pytest.approx(0.24)
    assert result.loc[1, "Bullpen_PAVE"] == pytest.approx(0.26)
    assert result.loc[1, "Park_Factor"] == pytest.approx(1.05)
    assert result.loc[1, "is_home"] == True  # noqa: E712
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.72)
    assert result.loc[1, "Whiff_Rate"] == pytest.approx(0.22)
    assert result.loc[1, "Chase_Rate"] == pytest.approx(0.28)
    assert result.loc[1, "Expected_Bases"] == pytest.approx(1.5)
    assert result.loc[1, "Fastball_WAVE"] == pytest.approx(0.30)
    assert result.loc[1, "WAVE_Home"] == pytest.approx(0.32)
    assert result.loc[1, "WAVE_Away"] == pytest.approx(0.24)
    assert result.loc[1, "starter_fastball_rate"] == pytest.approx(0.60)
    assert result.loc[1, "starter_breaking_rate"] == pytest.approx(0.25)
    assert result.loc[1, "starter_offspeed_rate"] == pytest.approx(0.15)
    assert list(result.reset_index().columns) == ["key_mlbam", "game_pk"] + dfs_ml.HITTER_FEATURE_COLUMNS
    assert pd.isna(result.loc[1, "game_pk"])


def test_build_hitter_features_falls_back_to_null_when_pitch_family_columns_missing():
    # An older wave.csv/pave.csv snapshot predating pitch-family signals
    # has neither the batter's own Fastball_WAVE/etc. nor the starter's
    # own Fastball_Rate/etc. - degrades to null (not a KeyError), same
    # convention WAVE_L/WAVE_R's own fallback test establishes.
    wave = pd.DataFrame([{
        "key_mlbam": 1, "team": "BOS", "WAVE": 0.28, "WAVE_L": 0.30, "WAVE_R": 0.27,
        "PA_L": 20, "PA_R": 40, "probability": 0.65, "Game_Hit_Probability": 0.70,
        "Consistency": 0.05, "Approach": 0.45, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
    }])
    pave = pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}])
    confidence = pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": False, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}])

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability).set_index("key_mlbam")

    assert pd.isna(result.loc[1, "Fastball_WAVE"])
    assert pd.isna(result.loc[1, "starter_fastball_rate"])
    assert pd.isna(result.loc[1, "Whiff_Rate"])
    assert pd.isna(result.loc[1, "WAVE_Home"])
    assert pd.isna(result.loc[1, "WAVE_Away"])


def test_build_hitter_features_falls_back_to_wave_when_wave_l_r_missing():
    # Regression test: an old wave.csv snapshot predating platoon-awareness
    # has no WAVE_L/WAVE_R columns at all - build_hitter_features must fall
    # back to WAVE rather than KeyError, same as matchup.py's _platoon_wave.
    wave = pd.DataFrame([{
        "key_mlbam": 1, "team": "BOS", "WAVE": 0.28, "PA_L": 20, "PA_R": 40,
        "probability": 0.65, "Game_Hit_Probability": 0.70, "Consistency": 0.05,
        "Approach": 0.45, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": 0.25, "xwOBA": 0.32,
    }])
    pave = pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}])
    confidence = pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": False, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}])

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability).set_index("key_mlbam")

    assert result.loc[1, "WAVE_L"] == pytest.approx(0.28)
    assert result.loc[1, "WAVE_R"] == pytest.approx(0.28)


def test_build_hitter_features_excludes_batter_with_no_resolvable_matchup():
    wave = pd.DataFrame([_wave_row(1, "BOS")])
    pave = pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}])
    confidence = pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}])
    matchup_probability = pd.DataFrame(columns=["key_mlbam", "Matchup_Hit_Probability"])  # no match for batter 1

    result = dfs_ml.build_hitter_features(wave, pave, confidence, schedule_df, matchup_probability)

    assert result.empty


def test_hitter_feature_matrix_fills_missing_and_coerces_is_home():
    features = pd.DataFrame([{
        "key_mlbam": 1, "WAVE": 0.28, "WAVE_L": 0.30, "WAVE_R": 0.27, "PA_L": 20, "PA_R": 40,
        "probability": 0.65, "Game_Hit_Probability": 0.70, "Consistency": 0.05, "Approach": 0.45,
        "Expected_Bases": 1.5, "Exit_Velo": 90.0, "Barrel_Rate": 0.08, "xBA": pd.NA, "xwOBA": 0.32,
        "starter_PAVE": pd.NA, "Bullpen_PAVE": 0.26, "Park_Factor": pd.NA,
        "is_home": True, "Matchup_Hit_Probability": 0.72,
        "Fastball_WAVE": pd.NA, "starter_fastball_rate": pd.NA,
        "Whiff_Rate": pd.NA,
    }])

    X = dfs_ml.hitter_feature_matrix(features)

    assert list(X.columns) == dfs_ml.HITTER_FEATURE_COLUMNS
    assert X.loc[0, "starter_PAVE"] == 0
    assert X.loc[0, "Park_Factor"] == 0
    assert X.loc[0, "xBA"] == 0
    assert X.loc[0, "is_home"] == 1.0
    assert X.loc[0, "Fastball_WAVE"] == 0
    assert X.loc[0, "starter_fastball_rate"] == 0
    assert X.loc[0, "Whiff_Rate"] == 0


def test_pitcher_feature_matrix_selects_columns_and_fills_missing():
    features = pd.DataFrame([{
        "key_mlbam": 99, "starts": 5, "K9": 9.0, "BB9": 3.0, "HR9": 1.2,
        "IP_per_start": 6.0, "Expected_IP": 6.0, "Expected_K": 6.0,
        "Expected_H_Allowed": pd.NA, "FIP_Windowed": 4.1, "extra_col": "ignored",
    }])

    X = dfs_ml.pitcher_feature_matrix(features)

    assert list(X.columns) == dfs_ml.PITCHER_FEATURE_COLUMNS
    assert X.loc[0, "Expected_H_Allowed"] == 0


class _ConstantModel:
    """Minimal stand-in for a fitted sklearn estimator - always predicts a
    fixed value, so override arithmetic is exactly checkable."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return pd.Series([self.value] * len(X)).to_numpy()


def _hitter_output_row(key_mlbam):
    return {
        "key_mlbam": key_mlbam, "name_first": "Test", "name_last": "Hitter", "team": "BOS",
        "opponent": "NYY", "is_home": True, "game_pk": 824651, "game_datetime": "2026-07-31T18:20:00Z",
        "PA_L": 20, "PA_R": 40, "Expected_Bases": 1.5,
        "Expected_BB": 0.3, "Expected_HBP": 0.1, "Expected_RBI": 0.4,
        "Game_Hit_Probability": 0.70, "Matchup_Hit_Probability": 0.72, "Matchup_Ratio": 1.03,
        "Adjusted_Expected_Bases": 1.545, "DK_Points_Hitter_HitType": 3.09, "DK_Points_Hitter": 4.17,
    }


def test_apply_ml_overrides_no_model_leaves_heuristic_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DFS_HITTER_MODEL_PATH", str(tmp_path / "missing_hitter.joblib"))
    monkeypatch.setattr(config, "DFS_PITCHER_H_ALLOWED_MODEL_PATH", str(tmp_path / "missing_h.joblib"))
    monkeypatch.setattr(config, "DFS_PITCHER_BB_MODEL_PATH", str(tmp_path / "missing_bb.joblib"))

    hitters_df = pd.DataFrame([_hitter_output_row(1)])
    hitter_features = pd.DataFrame([dict(dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}]),
        pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}]),
    ).iloc[0])])
    pitchers_df = pd.DataFrame([{
        "key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "team": "NYY", "opponent": "BOS",
        "is_home": False, "starts": 5, "K9": 9.0, "BB9": 3.0, "HR9": 1.2, "IP_per_start": 6.0,
        "Expected_IP": 6.0, "Expected_K": 6.0, "Expected_BB": 2.0, "Expected_H_Allowed": 8.0,
        "FIP_Windowed": 4.1, "Expected_ER": 2.7, "DK_Points_Pitcher": 15.0,
    }])

    out_hitters, out_pitchers = dfs_ml.apply_ml_overrides(hitters_df, hitter_features, pitchers_df)

    assert out_hitters.loc[0, "DK_Points_Hitter"] == pytest.approx(4.17)
    assert out_hitters.loc[0, "DK_Points_Hitter_Source"] == "heuristic"
    # Regression guard: merging hitter_features into hitters_df used to
    # silently rename shared columns (PA_L, Expected_Bases, ...) to _x/_y
    # suffixes, so HITTER_DFS_COLUMNS' own names vanished from the output
    # entirely - every dashboard cell showed blank/undefined. Assert the
    # exact expected column set and that the untouched fields kept their
    # real values, not NaN from a botched merge.
    assert set(out_hitters.columns) == set(dfs.HITTER_DFS_COLUMNS) | {"DK_Points_Hitter_Source"}
    assert not any(col.endswith(("_x", "_y")) for col in out_hitters.columns)
    assert out_hitters.loc[0, "PA_L"] == 20
    assert out_hitters.loc[0, "Expected_Bases"] == pytest.approx(1.5)
    assert out_hitters.loc[0, "Game_Hit_Probability"] == pytest.approx(0.70)
    assert out_pitchers.loc[0, "Expected_H_Allowed"] == pytest.approx(8.0)
    assert out_pitchers.loc[0, "Expected_BB"] == pytest.approx(2.0)
    assert out_pitchers.loc[0, "Expected_H_Allowed_Source"] == "heuristic"
    assert out_pitchers.loc[0, "Expected_BB_Source"] == "heuristic"
    assert out_pitchers.loc[0, "DK_Points_Pitcher"] == pytest.approx(15.0)


def test_apply_ml_overrides_with_model_overrides_and_recomputes_pitcher_total(tmp_path, monkeypatch):
    hitter_model_path = str(tmp_path / "hitter.joblib")
    h_allowed_path = str(tmp_path / "h_allowed.joblib")
    bb_path = str(tmp_path / "bb.joblib")
    ml_models.save_model(_ConstantModel(6.5), hitter_model_path)
    ml_models.save_model(_ConstantModel(7.0), h_allowed_path)
    ml_models.save_model(_ConstantModel(1.5), bb_path)
    monkeypatch.setattr(config, "DFS_HITTER_MODEL_PATH", hitter_model_path)
    monkeypatch.setattr(config, "DFS_PITCHER_H_ALLOWED_MODEL_PATH", h_allowed_path)
    monkeypatch.setattr(config, "DFS_PITCHER_BB_MODEL_PATH", bb_path)

    hitters_df = pd.DataFrame([_hitter_output_row(1)])
    hitter_features = dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}]),
        pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}]),
    )
    pitchers_df = pd.DataFrame([{
        "key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "team": "NYY", "opponent": "BOS",
        "is_home": False, "starts": 5, "K9": 9.0, "BB9": 3.0, "HR9": 1.2, "IP_per_start": 6.0,
        "Expected_IP": 6.0, "Expected_K": 6.0, "Expected_BB": 2.0, "Expected_H_Allowed": 8.0,
        "FIP_Windowed": 4.1, "Expected_ER": 2.7, "DK_Points_Pitcher": 15.0,
    }])

    out_hitters, out_pitchers = dfs_ml.apply_ml_overrides(hitters_df, hitter_features, pitchers_df)

    assert out_hitters.loc[0, "DK_Points_Hitter"] == pytest.approx(6.5)
    assert out_hitters.loc[0, "DK_Points_Hitter_Source"] == "model"
    # Feature-only columns (WAVE, starter_PAVE, ...) must not leak into the
    # returned output frame, and the columns HITTER_DFS_COLUMNS shares with
    # HITTER_FEATURE_COLUMNS (PA_L, Expected_Bases, ...) must survive
    # intact under their real names, not get silently suffixed away by a
    # botched merge.
    feature_only_columns = set(dfs_ml.HITTER_FEATURE_COLUMNS) - set(dfs.HITTER_DFS_COLUMNS)
    assert not feature_only_columns & set(out_hitters.columns)
    assert set(out_hitters.columns) == set(dfs.HITTER_DFS_COLUMNS) | {"DK_Points_Hitter_Source"}
    assert out_hitters.loc[0, "PA_L"] == 20
    assert out_hitters.loc[0, "Expected_Bases"] == pytest.approx(1.5)

    assert out_pitchers.loc[0, "Expected_H_Allowed"] == pytest.approx(7.0)
    assert out_pitchers.loc[0, "Expected_BB"] == pytest.approx(1.5)
    assert out_pitchers.loc[0, "Expected_H_Allowed_Source"] == "model"
    assert out_pitchers.loc[0, "Expected_BB_Source"] == "model"

    expected_total = (
        6.0 * config.DFS_DK_PITCHER_IP_POINTS
        + 6.0 * config.DFS_DK_PITCHER_K_POINTS
        + 1.5 * config.DFS_DK_PITCHER_BB_POINTS
        + 7.0 * config.DFS_DK_PITCHER_H_POINTS
        + 2.7 * config.DFS_DK_PITCHER_ER_POINTS
    )
    assert out_pitchers.loc[0, "DK_Points_Pitcher"] == pytest.approx(expected_total)


def test_apply_ml_overrides_clips_negative_predictions_to_zero(tmp_path, monkeypatch):
    hitter_model_path = str(tmp_path / "hitter.joblib")
    ml_models.save_model(_ConstantModel(-3.0), hitter_model_path)
    monkeypatch.setattr(config, "DFS_HITTER_MODEL_PATH", hitter_model_path)
    monkeypatch.setattr(config, "DFS_PITCHER_H_ALLOWED_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(config, "DFS_PITCHER_BB_MODEL_PATH", str(tmp_path / "missing2.joblib"))

    hitters_df = pd.DataFrame([_hitter_output_row(1)])
    hitter_features = dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}]),
        pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}]),
    )
    pitchers_df = pd.DataFrame(columns=["key_mlbam", "Expected_IP", "Expected_K", "Expected_BB", "Expected_H_Allowed", "Expected_ER", "DK_Points_Pitcher"])

    out_hitters, _ = dfs_ml.apply_ml_overrides(hitters_df, hitter_features, pitchers_df)

    assert out_hitters.loc[0, "DK_Points_Hitter"] == 0


class _ConstantProbaModel:
    """Minimal stand-in for a fitted sklearn classifier - always predicts a
    fixed positive-class probability, mirroring _ConstantModel above but for
    predict_proba's [n_samples, 2] shape."""

    def __init__(self, positive_proba):
        self.positive_proba = positive_proba

    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([1 - np.full(len(X), self.positive_proba), np.full(len(X), self.positive_proba)])


def test_predict_hitter_hit_probability_no_model_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HITTER_HIT_PROBABILITY_MODEL_PATH", str(tmp_path / "missing.joblib"))

    hitter_features = dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99}]),
        pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.72}]),
    )

    result = dfs_ml.predict_hitter_hit_probability(hitter_features)

    assert result.empty
    assert list(result.columns) == ["key_mlbam", "game_pk", "Model_Hit_Probability"]


def test_predict_hitter_hit_probability_with_model_returns_probability(tmp_path, monkeypatch):
    model_path = str(tmp_path / "hit_probability.joblib")
    ml_models.save_model(_ConstantProbaModel(0.63), model_path)
    monkeypatch.setattr(config, "HITTER_HIT_PROBABILITY_MODEL_PATH", model_path)

    hitter_features = dfs_ml.build_hitter_features(
        pd.DataFrame([_wave_row(1, "BOS")]),
        pd.DataFrame([{"key_mlbam": 99, "PAVE": 0.24}]),
        pd.DataFrame([{"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 1.05}]),
        pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99, "game_pk": 824651}]),
        pd.DataFrame([{"key_mlbam": 1, "game_pk": 824651, "Matchup_Hit_Probability": 0.72}]),
    )

    result = dfs_ml.predict_hitter_hit_probability(hitter_features).set_index("key_mlbam")

    assert result.loc[1, "Model_Hit_Probability"] == pytest.approx(0.63)
    assert result.loc[1, "game_pk"] == 824651


def test_build_hitter_features_doubleheader_keeps_both_games_with_correct_starter():
    wave = pd.DataFrame([_wave_row(1, "BOS")])
    pave = pd.DataFrame([
        {"key_mlbam": 99, "PAVE": 0.20, "Fastball_Rate": 0.60, "Breaking_Rate": 0.25, "Offspeed_Rate": 0.15},
        {"key_mlbam": 88, "PAVE": 0.40, "Fastball_Rate": 0.50, "Breaking_Rate": 0.30, "Offspeed_Rate": 0.20},
    ])
    confidence = pd.DataFrame([
        {"team": "NYY", "Bullpen_PAVE": 0.26, "Park_Factor": 0.95},
        {"team": "BOS", "Bullpen_PAVE": 0.25, "Park_Factor": 1.05},
    ])
    schedule_df = pd.DataFrame([
        {"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 99, "game_pk": 101},
        {"team": "BOS", "opponent": "NYY", "is_home": True, "probable_pitcher_key_mlbam": 88, "game_pk": 102},
    ])
    matchup_probability = pd.DataFrame([
        {"key_mlbam": 1, "game_pk": 101, "Matchup_Hit_Probability": 0.70},
        {"key_mlbam": 1, "game_pk": 102, "Matchup_Hit_Probability": 0.55},
    ])

    result = dfs_ml.build_hitter_features(
        wave, pave, confidence, schedule_df, matchup_probability
    ).set_index("game_pk")

    assert list(result.index) == [101, 102]
    assert (result["key_mlbam"] == 1).all()
    assert result.loc[101, "starter_PAVE"] == pytest.approx(0.20)
    assert result.loc[102, "starter_PAVE"] == pytest.approx(0.40)
    assert result.loc[101, "Matchup_Hit_Probability"] == pytest.approx(0.70)
    assert result.loc[102, "Matchup_Hit_Probability"] == pytest.approx(0.55)
