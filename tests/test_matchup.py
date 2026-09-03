import pandas as pd
import pytest

from mlb_metrics import matchup


def test_clip_and_blend_pitching_quality_exact_arithmetic():
    result = matchup.clip_and_blend_pitching_quality(pd.Series([0.9]), pd.Series([1.1]))
    # .6*.9 + .4*1.1 = .98 - unchanged, still used by game_picks.py's
    # team-level model (see clip_and_blend_pitching_pave below for the
    # raw-PAVE version compute_matchup_hit_probability uses instead).
    assert result.iloc[0] == pytest.approx(0.98)


def test_clip_and_blend_pitching_quality_clips_before_blending():
    result = matchup.clip_and_blend_pitching_quality(pd.Series([5.0]), pd.Series([0.1]))
    # Clipped to (0.5, 1.75) BEFORE blending: .6*1.75 + .4*.5 = 1.25.
    assert result.iloc[0] == pytest.approx(1.25)


def test_clip_and_blend_pitching_quality_missing_values_default_neutral():
    result = matchup.clip_and_blend_pitching_quality(pd.Series([None]), pd.Series([1.2]))
    # starter component neutral (1.0): .6*1.0 + .4*1.2 = 1.08
    assert result.iloc[0] == pytest.approx(1.08)


def test_clip_and_blend_pitching_pave_exact_arithmetic():
    result = matchup.clip_and_blend_pitching_pave(pd.Series([0.27]), pd.Series([0.297]), league_pave=0.27)
    # .6*.27 + .4*.297 = .2808 - both within the clip range, so unclipped.
    assert result.iloc[0] == pytest.approx(0.2808)


def test_clip_and_blend_pitching_pave_clips_before_blending():
    # league_pave=0.27 -> clip range is (0.135, 0.4725). Raw values (1.5,
    # 0.02) are both extreme outliers relative to that.
    result = matchup.clip_and_blend_pitching_pave(pd.Series([1.5]), pd.Series([0.02]), league_pave=0.27)
    # Clipped to (0.135, 0.4725) BEFORE blending: .6*.4725 + .4*.135 = .3375
    # (blending the raw values first and clipping only the final result
    # would give a different number - this test would catch that bug.)
    assert result.iloc[0] == pytest.approx(0.3375)


def test_clip_and_blend_pitching_pave_missing_values_default_to_league_pave():
    result = matchup.clip_and_blend_pitching_pave(pd.Series([None]), pd.Series([0.297]), league_pave=0.27)
    # starter component neutral (league_pave): .6*.27 + .4*.297 = .2808
    assert result.iloc[0] == pytest.approx(0.2808)


def _wave(rows):
    """rows: list of (key_mlbam, team, wave)."""
    return pd.DataFrame([{"key_mlbam": k, "team": t, "WAVE": w} for k, t, w in rows])


def test_compute_matchup_hit_probability_exact_arithmetic():
    wave = _wave([(1, "NYY", 0.30)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0}])  # opposing probable starter
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # league_pave = 0.27/1.0 = 0.27. opponent_rate = .6*.27 + .4*.297 = .2808.
    # matchup_ab_rate = log5(.30, .2808, .27) = .311487965... .
    # Matchup_Hit_Probability = 1 - (1-.311487965)**3.5 = .729173987...
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.7291739871326228)


def test_pave_is_clipped_before_blending_not_after():
    wave = _wave([(1, "NYY", 0.30)])
    # Two pitchers in the pool: the probable starter (an extreme-outlier
    # small-sample PAVE) and a second qualified pitcher near league-average,
    # so league_pave reflects a realistic baseline the outlier gets clipped
    # against rather than defining itself.
    pave = pd.DataFrame([
        {"key_mlbam": 999, "PAVE": 1.5, "PAVE_PLUS": 1.5 / 0.27},  # extreme outlier
        {"key_mlbam": 1000, "PAVE": 0.27, "PAVE_PLUS": 1.0},  # league-average anchor
    ])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.02}])  # extreme outlier
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # league_pave = mean(1.5/(1.5/.27), .27/1.0) = mean(.27, .27) = .27.
    # Clip range (0.135, 0.4725): starter (1.5) clips down to .4725, bullpen
    # (0.02) clips up to .135. opponent_rate = .6*.4725 + .4*.135 = .3375.
    # matchup_ab_rate = log5(.30, .3375, .27) = .371186441...
    # Matchup_Hit_Probability = 1 - (1-.371186441)**3.5 = .802836444...
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.802836443768993)


def test_missing_probable_starter_uses_league_average_neutral():
    wave = _wave([(1, "NYY", 0.50)])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE", "PAVE_PLUS"])  # no one announced yet
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": None}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # Empty pave -> league_pave falls back to config.MATCHUP_LEAGUE_PAVE_FALLBACK
    # (0.245). starter component neutral (league_pave itself):
    # opponent_rate = .6*.245 + .4*.297 = .2658.
    # matchup_ab_rate = log5(.50, .2658, .245) = .513528...
    # Matchup_Hit_Probability = 1 - (1-.513528...)**3.5
    from mlb_metrics import config

    league_pave = config.MATCHUP_LEAGUE_PAVE_FALLBACK
    opponent_rate = 0.6 * league_pave + 0.4 * 0.297
    numerator = 0.50 * opponent_rate / league_pave
    denominator = numerator + (1 - 0.50) * (1 - opponent_rate) / (1 - league_pave)
    matchup_ab_rate = numerator / denominator
    expected = 1 - (1 - matchup_ab_rate) ** config.WAVE_TRIALS_PER_GAME
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(expected)


def test_batter_with_no_game_today_is_excluded():
    wave = _wave([(1, "NYY", 0.30), (2, "SEA", 0.28)])
    pave = pd.DataFrame(columns=["key_mlbam", "PAVE", "PAVE_PLUS"])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.27}])
    # Only NYY has a game today; SEA does not appear in the schedule at all.
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": None}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df)

    assert list(result["key_mlbam"]) == [1]


def test_compute_matchup_hit_probability_doubleheader_uses_each_games_starter():
    wave = _wave([(1, "NYY", 0.30)])
    pave = pd.DataFrame([
        {"key_mlbam": 999, "PAVE": 0.20, "PAVE_PLUS": 0.20 / 0.27},
        {"key_mlbam": 1000, "PAVE": 0.40, "PAVE_PLUS": 0.40 / 0.27},
        {"key_mlbam": 1001, "PAVE": 0.27, "PAVE_PLUS": 1.0},  # league-average anchor
    ])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame([
        {"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "game_pk": 101, "is_home": True},
        {"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 1000, "game_pk": 102, "is_home": True},
    ])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("game_pk")

    assert list(result.index) == [101, 102]
    assert (result["key_mlbam"] == 1).all()
    # Soft starter (0.20) -> lower hit probability than hard starter (0.40).
    assert result.loc[101, "Matchup_Hit_Probability"] < result.loc[102, "Matchup_Hit_Probability"]


def _wave_with_platoon(rows):
    """rows: list of (key_mlbam, team, WAVE, WAVE_L, WAVE_R)."""
    return pd.DataFrame(
        [{"key_mlbam": k, "team": t, "WAVE": w, "WAVE_L": wl, "WAVE_R": wr} for k, t, w, wl, wr in rows]
    )


def _expected_probability(batter_rate, opponent_rate, league_pave, park_multiplier=1.0):
    from mlb_metrics import config

    numerator = batter_rate * opponent_rate / league_pave
    denominator = numerator + (1 - batter_rate) * (1 - opponent_rate) / (1 - league_pave)
    matchup_ab_rate = (numerator / denominator) * park_multiplier
    return 1 - (1 - matchup_ab_rate) ** config.WAVE_TRIALS_PER_GAME


def test_platoon_uses_wave_l_against_a_lefty_starter():
    wave = _wave_with_platoon([(1, "NYY", 0.30, 0.50, 0.20)])  # much better vs LHP than blended WAVE
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0, "Throws": "L"}])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": True}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    expected = _expected_probability(0.50, 0.6 * 0.27 + 0.4 * 0.297, league_pave=0.27)
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(expected)


def test_platoon_uses_wave_r_against_a_righty_starter():
    wave = _wave_with_platoon([(1, "NYY", 0.30, 0.50, 0.20)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0, "Throws": "R"}])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": True}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    expected = _expected_probability(0.20, 0.6 * 0.27 + 0.4 * 0.297, league_pave=0.27)
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(expected)


def test_platoon_falls_back_to_blended_wave_when_starter_throws_unknown():
    wave = _wave_with_platoon([(1, "NYY", 0.30, 0.50, 0.20)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0}])  # no Throws column at all
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": True}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    # Falls back to blended WAVE (0.30) - same number as the original
    # (pre-platoon) exact-arithmetic test.
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.7291739871326228)


def test_park_factor_applies_home_teams_park_when_batter_is_home():
    wave = _wave([(1, "NYY", 0.30)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0}])
    confidence = pd.DataFrame([
        {"team": "BOS", "Bullpen_PAVE": 0.297, "Park_Factor": 1.20},  # not today's venue - irrelevant
        {"team": "NYY", "Bullpen_PAVE": 0.27, "Park_Factor": 1.10},  # NYY is home today -> this park applies
    ])
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": True}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    expected = _expected_probability(0.30, 0.6 * 0.27 + 0.4 * 0.297, league_pave=0.27, park_multiplier=1.10)
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(expected)


def test_park_factor_uses_opponents_park_when_batter_is_away():
    wave = _wave([(1, "NYY", 0.30)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0}])
    confidence = pd.DataFrame([
        {"team": "BOS", "Bullpen_PAVE": 0.297, "Park_Factor": 0.90},  # BOS is home today -> this park applies
        {"team": "NYY", "Bullpen_PAVE": 0.27, "Park_Factor": 1.20},  # NYY's own park - irrelevant, NYY is away
    ])
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": False}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    expected = _expected_probability(0.30, 0.6 * 0.27 + 0.4 * 0.297, league_pave=0.27, park_multiplier=0.90)
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(expected)


def test_park_factor_is_clipped_before_applying():
    wave = _wave([(1, "NYY", 0.30)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0}])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297, "Park_Factor": 2.0}])  # extreme outlier
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": False}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    from mlb_metrics import config

    expected = _expected_probability(
        0.30, 0.6 * 0.27 + 0.4 * 0.297, league_pave=0.27, park_multiplier=config.MATCHUP_PARK_FACTOR_CLIP[1]
    )
    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(expected)


def test_park_factor_missing_column_is_a_no_op():
    wave = _wave([(1, "NYY", 0.30)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0}])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])  # no Park_Factor column at all
    schedule_df = pd.DataFrame(
        [{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999, "is_home": True}]
    )

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    assert result.loc[1, "Matchup_Hit_Probability"] == pytest.approx(0.7291739871326228)


def test_final_probability_stays_within_zero_one():
    wave = _wave([(1, "NYY", 0.99)])
    pave = pd.DataFrame([{"key_mlbam": 999, "PAVE": 0.10, "PAVE_PLUS": 0.10 / 0.27}])  # a very weak pitcher
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.10}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")

    assert 0.0 <= result.loc[1, "Matchup_Hit_Probability"] <= 1.0


def test_league_arsenal_mix_recovers_real_mean_from_pave():
    pave = pd.DataFrame([
        {"Fastball_Rate": 0.6, "Breaking_Rate": 0.3, "Offspeed_Rate": 0.1},
        {"Fastball_Rate": 0.4, "Breaking_Rate": 0.3, "Offspeed_Rate": 0.3},
    ])

    result = matchup._league_arsenal_mix(pave)

    assert result == pytest.approx({"fastball": 0.5, "breaking": 0.3, "offspeed": 0.2})


def test_league_arsenal_mix_falls_back_when_columns_missing():
    from mlb_metrics import config

    pave = pd.DataFrame([{"key_mlbam": 1, "PAVE": 0.27}])  # no arsenal columns at all

    result = matchup._league_arsenal_mix(pave)

    assert result == config.MATCHUP_LEAGUE_ARSENAL_FALLBACK


def test_pitch_arsenal_multiplier_zero_batter_data_stays_float_and_neutral():
    # Real bug found via a live retrain (GitHub Actions): the original
    # implementation divided by `vs_league.replace(0, pd.NA)`, which
    # silently upcast the Series to `object` dtype whenever it hit a real
    # zero - every op downstream then ran through Python's own operators
    # instead of numpy's, and a later `(1 - matchup_ab_rate) ** 3.5` on a
    # real row produced a genuine Python `complex` value that crashed
    # `.clip(0, 1)` with a TypeError. A batter with zero real data in
    # every pitch family (vs_league == vs_starter == 0, a real 0/0) is
    # exactly the case that used to trigger the replace() - must resolve
    # to a real float64 1.0 (neutral), never an object-dtype NA.
    matchup_df = pd.DataFrame([{
        "Fastball_WAVE": 0.0, "Breaking_WAVE": 0.0, "Offspeed_WAVE": 0.0,
        "starter_fastball_rate": 0.6, "starter_breaking_rate": 0.3, "starter_offspeed_rate": 0.1,
    }])
    league_mix = {"fastball": 0.55, "breaking": 0.30, "offspeed": 0.15}

    result = matchup._pitch_arsenal_multiplier(matchup_df, league_mix)

    assert result.dtype == float
    assert result.iloc[0] == pytest.approx(1.0)


def test_pitch_arsenal_multiplier_is_a_noop_when_batter_columns_missing():
    matchup_df = pd.DataFrame([
        {"starter_fastball_rate": 0.7, "starter_breaking_rate": 0.2, "starter_offspeed_rate": 0.1}
    ])
    league_mix = {"fastball": 0.55, "breaking": 0.30, "offspeed": 0.15}

    result = matchup._pitch_arsenal_multiplier(matchup_df, league_mix)

    assert (result == 1.0).all()


def test_pitch_arsenal_multiplier_is_a_noop_when_starter_columns_missing():
    matchup_df = pd.DataFrame([
        {"Fastball_WAVE": 0.35, "Breaking_WAVE": 0.25, "Offspeed_WAVE": 0.30}
    ])
    league_mix = {"fastball": 0.55, "breaking": 0.30, "offspeed": 0.15}

    result = matchup._pitch_arsenal_multiplier(matchup_df, league_mix)

    assert (result == 1.0).all()


def test_pitch_arsenal_multiplier_exact_ratio_at_full_weight(monkeypatch):
    from mlb_metrics import config

    monkeypatch.setattr(config, "MATCHUP_PITCH_ARSENAL_WEIGHT", 1.0)
    matchup_df = pd.DataFrame([{
        "Fastball_WAVE": 0.40, "Breaking_WAVE": 0.20, "Offspeed_WAVE": 0.10,
        "starter_fastball_rate": 0.80, "starter_breaking_rate": 0.10, "starter_offspeed_rate": 0.10,
    }])
    league_mix = {"fastball": 0.55, "breaking": 0.30, "offspeed": 0.15}

    result = matchup._pitch_arsenal_multiplier(matchup_df, league_mix)

    # vs_starter = .40*.80 + .20*.10 + .10*.10 = .35
    # vs_league  = .40*.55 + .20*.30 + .10*.15 = .295
    # ratio = .35/.295 = 1.186440678 -> clipped to (0.85, 1.15) -> 1.15
    assert result.iloc[0] == pytest.approx(config.MATCHUP_PITCH_ARSENAL_CLIP[1])


def test_pitch_arsenal_multiplier_ships_at_zero_weight_by_default():
    # Same batter/starter data as the full-weight test above, but at the
    # real shipped default (0.0) - must be an exact no-op regardless of
    # how skewed the ratio would otherwise be.
    matchup_df = pd.DataFrame([{
        "Fastball_WAVE": 0.40, "Breaking_WAVE": 0.20, "Offspeed_WAVE": 0.10,
        "starter_fastball_rate": 0.80, "starter_breaking_rate": 0.10, "starter_offspeed_rate": 0.10,
    }])
    league_mix = {"fastball": 0.55, "breaking": 0.30, "offspeed": 0.15}

    result = matchup._pitch_arsenal_multiplier(matchup_df, league_mix)

    assert result.iloc[0] == pytest.approx(1.0)


def test_compute_matchup_hit_probability_applies_pitch_arsenal_at_nonzero_weight(monkeypatch):
    from mlb_metrics import config

    monkeypatch.setattr(config, "MATCHUP_PITCH_ARSENAL_WEIGHT", 1.0)
    wave = pd.DataFrame([{
        "key_mlbam": 1, "team": "NYY", "WAVE": 0.30,
        "Fastball_WAVE": 0.40, "Breaking_WAVE": 0.20, "Offspeed_WAVE": 0.10,
    }])
    # Two pitchers in the pool so the league-average mix (used as the
    # neutral baseline) genuinely differs from the probable starter's own
    # skewed-fastball mix - a real, nonzero multiplier, not a trivial
    # single-row no-op.
    pave = pd.DataFrame([
        {"key_mlbam": 999, "PAVE": 0.27, "PAVE_PLUS": 1.0,
         "Fastball_Rate": 0.80, "Breaking_Rate": 0.10, "Offspeed_Rate": 0.10},
        {"key_mlbam": 1000, "PAVE": 0.27, "PAVE_PLUS": 1.0,
         "Fastball_Rate": 0.30, "Breaking_Rate": 0.50, "Offspeed_Rate": 0.20},
    ])
    confidence = pd.DataFrame([{"team": "BOS", "Bullpen_PAVE": 0.297}])
    schedule_df = pd.DataFrame([{"team": "NYY", "opponent": "BOS", "probable_pitcher_key_mlbam": 999}])

    result = matchup.compute_matchup_hit_probability(wave, pave, confidence, schedule_df).set_index("key_mlbam")
    baseline = matchup.compute_matchup_hit_probability(
        wave.drop(columns=["Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE"]), pave, confidence, schedule_df
    ).set_index("key_mlbam")

    # league_mix = mean of the two pitchers = fastball .55/breaking .30/offspeed .15.
    # vs_starter = .40*.80 + .20*.10 + .10*.10 = .35
    # vs_league  = .40*.55 + .20*.30 + .10*.15 = .295
    # ratio = .35/.295 = 1.186440678 -> clipped to (0.85, 1.15) -> 1.15,
    # applied on top of (not instead of) the platoon/park-adjusted rate the
    # baseline call already produces.
    assert result.loc[1, "Matchup_Hit_Probability"] != pytest.approx(baseline.loc[1, "Matchup_Hit_Probability"])
