import pandas as pd
import pytest

from mlb_metrics import config, dfs


def test_compute_matchup_adjustment_basic_ratio():
    matchup_prob = pd.Series([0.84, 0.35])
    game_prob = pd.Series([0.70, 0.70])

    result = dfs.compute_matchup_adjustment(matchup_prob, game_prob)

    assert result.tolist() == pytest.approx([0.84 / 0.70, 0.35 / 0.70])


def test_compute_matchup_adjustment_clips_extreme_ratio():
    # matchup_prob/game_prob would be 5.0 without clipping.
    matchup_prob = pd.Series([0.50])
    game_prob = pd.Series([0.10])

    result = dfs.compute_matchup_adjustment(matchup_prob, game_prob)

    lo, hi = config.DFS_MATCHUP_RATIO_CLIP
    assert result.iloc[0] == pytest.approx(hi)


def test_compute_matchup_adjustment_floors_denominator_before_dividing():
    # game_prob is below DFS_MATCHUP_RATIO_MIN_DENOM - must be floored, not
    # divided by directly (which would blow the ratio up even further).
    matchup_prob = pd.Series([0.10])
    game_prob = pd.Series([0.001])

    result = dfs.compute_matchup_adjustment(matchup_prob, game_prob)

    lo, hi = config.DFS_MATCHUP_RATIO_CLIP
    assert result.iloc[0] == pytest.approx(hi)  # still clipped, but via the floored denom


def _wave_row(key_mlbam, team, pa_l, pa_r, expected_bases, game_hit_probability, expected_bb=0.0, expected_hbp=0.0, expected_rbi=0.0):
    return {
        "key_mlbam": key_mlbam, "name_first": "Test", "name_last": "Hitter", "team": team,
        "PA_L": pa_l, "PA_R": pa_r, "Expected_Bases": expected_bases, "Game_Hit_Probability": game_hit_probability,
        "Expected_BB": expected_bb, "Expected_HBP": expected_hbp, "Expected_RBI": expected_rbi,
    }


def test_compute_hitter_dk_points_exact_arithmetic():
    wave = pd.DataFrame([_wave_row(1, "BOS", 20, 20, 1.5, 0.70, expected_bb=0.4, expected_hbp=0.1, expected_rbi=0.3)])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.84}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True}])

    result = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df).set_index("key_mlbam")

    expected_ratio = 0.84 / 0.70
    expected_adjusted = 1.5 * expected_ratio
    expected_hit_type_points = expected_adjusted * config.DFS_DK_POINTS_PER_TOTAL_BASE
    expected_points = (
        expected_hit_type_points
        + 0.4 * config.DFS_DK_HITTER_BB_POINTS
        + 0.1 * config.DFS_DK_HITTER_HBP_POINTS
        + 0.3 * config.DFS_DK_HITTER_RBI_POINTS
    )

    assert result.loc[1, "opponent"] == "NYY"
    assert result.loc[1, "Matchup_Ratio"] == pytest.approx(expected_ratio)
    assert result.loc[1, "Adjusted_Expected_Bases"] == pytest.approx(expected_adjusted)
    assert result.loc[1, "DK_Points_Hitter_HitType"] == pytest.approx(expected_hit_type_points)
    assert result.loc[1, "DK_Points_Hitter"] == pytest.approx(expected_points)


def test_compute_hitter_dk_points_excludes_below_min_plate_appearances():
    wave = pd.DataFrame([_wave_row(1, "BOS", 5, 5, 1.5, 0.70)])  # 10 total, below default min
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.84}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True}])

    result = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df, min_plate_appearances=30)

    assert result.empty


def test_compute_hitter_dk_points_carries_game_pk_and_game_datetime():
    wave = pd.DataFrame([_wave_row(1, "BOS", 20, 20, 1.5, 0.70)])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.84}])
    schedule_df = pd.DataFrame([{
        "team": "BOS", "opponent": "NYY", "is_home": True,
        "game_pk": 824651, "game_datetime": "2026-07-31T18:20:00Z",
    }])

    result = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df).set_index("key_mlbam")

    assert result.loc[1, "game_pk"] == 824651
    assert result.loc[1, "game_datetime"] == "2026-07-31T18:20:00Z"


def test_compute_hitter_dk_points_missing_game_pk_column_is_na_not_a_crash():
    # A schedule_df without game_pk/game_datetime at all (older fixtures,
    # historical backtest replays) must degrade to NA, not KeyError on the
    # HITTER_DFS_COLUMNS selection.
    wave = pd.DataFrame([_wave_row(1, "BOS", 20, 20, 1.5, 0.70)])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.84}])
    schedule_df = pd.DataFrame([{"team": "BOS", "opponent": "NYY", "is_home": True}])

    result = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df).set_index("key_mlbam")

    assert pd.isna(result.loc[1, "game_pk"])
    assert pd.isna(result.loc[1, "game_datetime"])


def test_compute_hitter_dk_points_doubleheader_does_not_cartesian_duplicate():
    wave = pd.DataFrame([_wave_row(1, "BOS", 20, 20, 1.5, 0.70)])
    matchup_probability = pd.DataFrame([
        {"key_mlbam": 1, "game_pk": 101, "Matchup_Hit_Probability": 0.84},
        {"key_mlbam": 1, "game_pk": 102, "Matchup_Hit_Probability": 0.60},
    ])
    schedule_df = pd.DataFrame([
        {"team": "BOS", "opponent": "NYY", "is_home": True, "game_pk": 101},
        {"team": "BOS", "opponent": "NYY", "is_home": True, "game_pk": 102},
    ])

    result = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df)

    assert len(result) == 2
    assert set(result["game_pk"]) == {101, 102}
    by_pk = result.set_index("game_pk")
    assert by_pk.loc[101, "Matchup_Hit_Probability"] == pytest.approx(0.84)
    assert by_pk.loc[102, "Matchup_Hit_Probability"] == pytest.approx(0.60)


def test_compute_hitter_dk_points_excludes_team_with_no_game_today():
    wave = pd.DataFrame([_wave_row(1, "BOS", 20, 20, 1.5, 0.70)])
    matchup_probability = pd.DataFrame([{"key_mlbam": 1, "Matchup_Hit_Probability": 0.84}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "TB", "is_home": True}])  # BOS has no game

    result = dfs.compute_hitter_dk_points(wave, matchup_probability, schedule_df)

    assert result.empty


def _pitcher_form_row(key_mlbam, starts, ip, k9, bb9, hr9, ip_per_start):
    return {"key_mlbam": key_mlbam, "starts": starts, "IP": ip, "K9": k9, "BB9": bb9, "HR9": hr9, "IP_per_start": ip_per_start}


def test_compute_pitcher_dk_points_exact_arithmetic():
    pave = pd.DataFrame([{"key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "PAVE": 0.24}])
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=5, ip=30, k9=9.0, bb9=3.0, hr9=1.2, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": 99}])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df).set_index("key_mlbam")

    ip = 6.0
    expected_k = 9.0 * ip / 9
    expected_bb = 3.0 * ip / 9
    expected_bf = ip * config.DFS_BATTERS_FACED_PER_INNING
    expected_h = 0.24 * expected_bf
    expected_fip = (13 * 1.2 + 3 * 3.0 - 2 * 9.0) / 9 + config.FIP_CONSTANT
    expected_er = expected_fip * ip / 9
    expected_points = (
        ip * config.DFS_DK_PITCHER_IP_POINTS
        + expected_k * config.DFS_DK_PITCHER_K_POINTS
        + expected_bb * config.DFS_DK_PITCHER_BB_POINTS
        + expected_h * config.DFS_DK_PITCHER_H_POINTS
        + expected_er * config.DFS_DK_PITCHER_ER_POINTS
    )

    assert result.loc[99, "Expected_IP"] == pytest.approx(ip)
    assert result.loc[99, "Expected_K"] == pytest.approx(expected_k)
    assert result.loc[99, "Expected_BB"] == pytest.approx(expected_bb)
    assert result.loc[99, "Expected_H_Allowed"] == pytest.approx(expected_h)
    assert result.loc[99, "FIP_Windowed"] == pytest.approx(expected_fip)
    assert result.loc[99, "Expected_ER"] == pytest.approx(expected_er)
    assert result.loc[99, "DK_Points_Pitcher"] == pytest.approx(expected_points)


def test_compute_pitcher_dk_points_negative_windowed_fip_clipped_to_zero_er():
    # HR9=0, BB9=0, K9 very high -> FIP_Windowed goes deeply negative before
    # the additive constant corrects it -> Expected_ER must floor at 0, not
    # go negative (a pitcher can't allow negative earned runs).
    pave = pd.DataFrame([{"key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "PAVE": 0.20}])
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=5, ip=30, k9=15.0, bb9=0.0, hr9=0.0, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": 99}])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df).set_index("key_mlbam")

    raw_fip = (13 * 0 + 3 * 0 - 2 * 15.0) / 9 + config.FIP_CONSTANT
    assert raw_fip < 0  # confirms this fixture actually exercises the clip
    assert result.loc[99, "Expected_ER"] == 0


def test_compute_pitcher_dk_points_carries_game_pk_and_game_datetime():
    pave = pd.DataFrame([{"key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "PAVE": 0.24}])
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=5, ip=30, k9=9.0, bb9=3.0, hr9=1.2, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{
        "team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": 99,
        "game_pk": 824651, "game_datetime": "2026-07-31T18:20:00Z",
    }])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df).set_index("key_mlbam")

    assert result.loc[99, "game_pk"] == 824651
    assert result.loc[99, "game_datetime"] == "2026-07-31T18:20:00Z"


def test_compute_pitcher_dk_points_missing_game_pk_column_is_na_not_a_crash():
    pave = pd.DataFrame([{"key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "PAVE": 0.24}])
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=5, ip=30, k9=9.0, bb9=3.0, hr9=1.2, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": 99}])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df).set_index("key_mlbam")

    assert pd.isna(result.loc[99, "game_pk"])
    assert pd.isna(result.loc[99, "game_datetime"])


def test_compute_pitcher_dk_points_excludes_below_min_starts():
    pave = pd.DataFrame([{"key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "PAVE": 0.24}])
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=2, ip=12, k9=9.0, bb9=3.0, hr9=1.2, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": 99}])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df, min_starts=3)

    assert result.empty


def test_compute_pitcher_dk_points_excludes_unannounced_starter():
    pave = pd.DataFrame([{"key_mlbam": 99, "name_first": "Test", "name_last": "Pitcher", "PAVE": 0.24}])
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=5, ip=30, k9=9.0, bb9=3.0, hr9=1.2, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": pd.NA}])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df)

    assert result.empty


def test_compute_pitcher_dk_points_missing_pave_falls_back_to_league_average():
    pave = pd.DataFrame(columns=["key_mlbam", "name_first", "name_last", "PAVE"])  # no row for this pitcher
    pitcher_form_df = pd.DataFrame([_pitcher_form_row(99, starts=5, ip=30, k9=9.0, bb9=3.0, hr9=1.2, ip_per_start=6.0)])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "is_home": False, "probable_pitcher_key_mlbam": 99}])

    result = dfs.compute_pitcher_dk_points(pave, pitcher_form_df, schedule_df).set_index("key_mlbam")

    expected_bf = 6.0 * config.DFS_BATTERS_FACED_PER_INNING
    expected_h = config.MATCHUP_LEAGUE_PAVE_FALLBACK * expected_bf
    assert result.loc[99, "Expected_H_Allowed"] == pytest.approx(expected_h)
