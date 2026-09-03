"""ML-based replacements for dfs.py's three weak-signal projections
(real backtest numbers, config.py's DFS section):
  DK_Points_Hitter:    correlation -0.004 (essentially zero)
  Expected_H_Allowed:  MAE worse than the naive baseline
  Expected_BB:         weak correlation (0.182)

Walk-forward, grid-searched (ml_models.py) Ridge/gradient-boosting models,
trained on dfs_backtest.assemble_ml_training_rows's full-history feature
table - the SAME no-lookahead discipline (each row's features come only
from data strictly before that row's game date) the existing heuristic
backtest already uses.

## Hitter features: raw ingredients, not the flagged-bad ratio

dfs.compute_matchup_adjustment's ratio (Matchup_Hit_Probability / a
batter's own Game_Hit_Probability, applied multiplicatively to
Expected_Bases) was flagged in dfs.py's own docstring as the single
highest-risk modeling choice in that module, and the real backtest
confirmed it: essentially zero correlation. build_hitter_features exposes
the RAW ingredients that ratio was built from (the batter's own form, plus
matchup.py's own internal log5-blend inputs - the opposing starter's PAVE,
their bullpen's PAVE, the venue's Park_Factor, is_home) instead of only
the already-blended Matchup_Hit_Probability, so a model can learn its own
(possibly additive, possibly nonlinear) combination rather than
inheriting the multiplicative assumption. Also includes Expected_BB/
Expected_HBP/Expected_RBI (hitters.compute_extended_dk_rates) now that
dfs.compute_hitter_dk_points scores those categories too - the SAME
raw-ingredients philosophy applies: these are fed directly, not run
through the flagged ratio.

**Retraining note**: HITTER_FEATURE_COLUMNS widened when BB/HBP/RBI
scoring shipped, again to add real Statcast batted-ball-quality signals
(Exit_Velo/Barrel_Rate/xBA/xwOBA, hitters.compute_quality_of_contact),
again to add pitch-type-family signals (Fastball_WAVE/Breaking_WAVE/
Offspeed_WAVE - hitters.compute_pitch_family_rates - and today's probable
starter's own starter_fastball_rate/starter_breaking_rate/
starter_offspeed_rate - pitchers.compute_pitch_arsenal), and again to add
real per-pitch plate-discipline signals (Whiff_Rate/Chase_Rate -
hitters.compute_plate_discipline) - any model artifact trained on an
OLDER (narrower) schema is schema-incompatible and MUST be retired
(deleted, not left in place) whenever this list changes, since BOTH
apply_ml_overrides's predict() call AND
predict_hitter_hit_probability's predict_proba() call are NOT wrapped in
a try/except - a stale artifact would hard-crash
scripts/build_dfs_rankings.py/live daily picks rather than gracefully
falling back. Every prior modeling attempt in this project used only
outcome-RATE signals (WAVE/PAVE-style hit/walk/RBI rates) - never a real
measure of contact QUALITY (how hard/well the ball was actually hit,
independent of whether that particular ball found a fielder), never a
pitch-type-specific matchup signal (does this batter's own fastball/
breaking/offspeed split suggest a better/worse day against THIS starter's
own real pitch mix), and never a real per-pitch plate-discipline signal
(how often the batter swings and misses entirely, or chases a pitch
genuinely outside the strike zone) - even though the raw Statcast columns
needed for all three (launch_speed/launch_angle/
estimated_ba_using_speedangle/estimated_woba_using_speedangle/
launch_speed_angle for the first, pitch_type for the second, description/
zone for the third) were sitting unused in the already-persisted raw data
the whole time.

## Pitcher features: dfs.py's own engineered columns

Unlike hitters, dfs.compute_pitcher_dk_points's engineered columns
(K9/BB9/HR9/IP_per_start/Expected_IP/Expected_K/Expected_H_Allowed/
FIP_Windowed) aren't suspected of a bad transform the way the hitter
ratio was - Expected_H_Allowed's problem (real backtest: MAE worse than
baseline despite positive correlation) looks like a systematic scaling
bias (PAVE * Expected_IP * DFS_BATTERS_FACED_PER_INNING), the kind of
thing a regression CAN absorb from the same inputs. So pitcher features
reuse dfs.py's already-engineered columns directly (this also keeps
train-time and live-serving feature assembly identical by construction,
since both call dfs.compute_pitcher_dk_points).

## Fallback discipline

apply_ml_overrides only overrides a signal when a trained model artifact
exists (ml_models.load_model returns non-None) - a missing/corrupt/
not-yet-trained artifact silently keeps the existing heuristic column
untouched, the same "missing optional input degrades to a documented
fallback, never a crash" pattern pipeline.run() already uses for a failed
schedule fetch. Each signal is independently gated - a hitter model
failure never affects the pitcher columns and vice versa.
"""

import pandas as pd

from mlb_metrics import config, dfs, ml_models

HITTER_FEATURE_COLUMNS = [
    "WAVE", "WAVE_L", "WAVE_R", "PA_L", "PA_R", "probability",
    "Game_Hit_Probability", "Consistency", "Approach", "Expected_Bases",
    "Expected_BB", "Expected_HBP", "Expected_RBI",
    "Exit_Velo", "Barrel_Rate", "xBA", "xwOBA",
    "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE",
    "Whiff_Rate", "Chase_Rate",
    "WAVE_Home", "WAVE_Away",
    "starter_PAVE", "Bullpen_PAVE", "Park_Factor", "is_home",
    "starter_fastball_rate", "starter_breaking_rate", "starter_offspeed_rate",
    "Matchup_Hit_Probability",
]

PITCHER_FEATURE_COLUMNS = [
    "starts", "K9", "BB9", "HR9", "IP_per_start", "Expected_IP",
    "Expected_K", "Expected_H_Allowed", "FIP_Windowed",
]


def build_hitter_features(
    wave: pd.DataFrame,
    pave: pd.DataFrame,
    confidence: pd.DataFrame,
    schedule_df: pd.DataFrame,
    matchup_probability: pd.DataFrame,
) -> pd.DataFrame:
    """One row per hitter-game with a resolvable Matchup_Hit_Probability -
    the same qualifying join dfs.compute_hitter_dk_points uses - carrying
    key_mlbam, game_pk, plus every column in HITTER_FEATURE_COLUMNS.

    Joins schedule on `team` and matchup on (`key_mlbam`, `game_pk`) when
    `game_pk` is present on both sides, so a doubleheader never fans out
    into a cartesian product of game-one features × game-two labels.
    Callers must pass the hitter schedule shape (one row per team per
    game, both DH halves kept). Used identically at training-table
    assembly time (dfs_backtest.assemble_ml_training_rows /
    assemble_hitter_hit_log) and at live prediction time
    (apply_ml_overrides / predict_hitter_hit_probability)."""
    schedule_columns = [
        c for c in ("team", "opponent", "probable_pitcher_key_mlbam", "is_home", "game_pk")
        if c in schedule_df.columns
    ]
    df = wave.merge(schedule_df[schedule_columns], on="team", how="inner")
    if "game_pk" not in df.columns:
        df["game_pk"] = pd.NA

    # Falls back to the hand-blended overall WAVE when WAVE_L/WAVE_R aren't
    # present at all - an old wave.csv snapshot predating platoon-awareness,
    # same known scenario matchup.py's _platoon_wave already guards against.
    for column in ("WAVE_L", "WAVE_R"):
        if column not in df.columns:
            df[column] = df["WAVE"]

    # A batter's own pitch-family rates (an even older wave.csv snapshot
    # predating hitters.compute_pitch_family_rates) degrade to null rather
    # than a crash - hitter_feature_matrix's blanket fillna(0) then reads
    # this exactly like "no family data", the same convention every other
    # optional feature column here already uses.
    for column in ("Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE"):
        if column not in df.columns:
            df[column] = pd.NA

    # A batter's own home/road split (hitters.compute_home_road_split, an
    # even older wave.csv snapshot predating it) degrades to null the same
    # way - same convention as the pitch-family columns just above.
    for column in ("WAVE_Home", "WAVE_Away"):
        if column not in df.columns:
            df[column] = pd.NA

    # A batter's own Whiff_Rate/Chase_Rate (hitters.compute_plate_discipline,
    # an OPTIONAL assemble_hitters input - absent whenever the caller didn't
    # pass all_pitches) degrade to null the same way - never a KeyError just
    # because the live pipeline's own optional wiring wasn't exercised.
    for column in ("Whiff_Rate", "Chase_Rate"):
        if column not in df.columns:
            df[column] = pd.NA

    pave_columns = [
        c for c in ("key_mlbam", "PAVE", "Fastball_Rate", "Breaking_Rate", "Offspeed_Rate") if c in pave.columns
    ]
    starter = pave[pave_columns].rename(
        columns={
            "key_mlbam": "probable_pitcher_key_mlbam", "PAVE": "starter_PAVE",
            "Fastball_Rate": "starter_fastball_rate", "Breaking_Rate": "starter_breaking_rate",
            "Offspeed_Rate": "starter_offspeed_rate",
        }
    )
    df = df.merge(starter, on="probable_pitcher_key_mlbam", how="left")
    for column in ("starter_fastball_rate", "starter_breaking_rate", "starter_offspeed_rate"):
        if column not in df.columns:
            df[column] = pd.NA

    if "Bullpen_PAVE" in confidence.columns:
        bullpen = confidence[["team", "Bullpen_PAVE"]].rename(columns={"team": "opponent"})
        df = df.merge(bullpen, on="opponent", how="left")
    else:
        df["Bullpen_PAVE"] = pd.NA

    if "Park_Factor" in confidence.columns and "is_home" in df.columns:
        df["venue_team"] = df["team"].where(df["is_home"], df["opponent"])
        park = confidence[["team", "Park_Factor"]].rename(columns={"team": "venue_team"})
        df = df.merge(park, on="venue_team", how="left")
    else:
        df["Park_Factor"] = pd.NA

    matchup_cols = ["key_mlbam", "Matchup_Hit_Probability"]
    if "game_pk" in matchup_probability.columns:
        matchup_cols = ["key_mlbam", "game_pk", "Matchup_Hit_Probability"]
        df = df.merge(matchup_probability[matchup_cols], on=["key_mlbam", "game_pk"], how="inner")
    else:
        df = df.merge(matchup_probability[matchup_cols], on="key_mlbam", how="inner")
    return df[["key_mlbam", "game_pk"] + HITTER_FEATURE_COLUMNS]


def hitter_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """The numeric X matrix a hitter model trains/predicts on - is_home
    coerced to float (0/1), everything else NaN-filled to 0 (an
    unannounced/unmatched opposing starter's starter_PAVE, e.g.) rather
    than dropped, so a row is never silently excluded from prediction just
    because one matchup ingredient is missing."""
    X = features_df.reindex(columns=HITTER_FEATURE_COLUMNS).copy()
    X["is_home"] = X["is_home"].fillna(False).astype(float)
    return X.fillna(0)


def pitcher_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """The numeric X matrix a pitcher model trains/predicts on."""
    return features_df.reindex(columns=PITCHER_FEATURE_COLUMNS).copy().fillna(0)


def apply_ml_overrides(
    hitters_df: pd.DataFrame,
    hitter_features: pd.DataFrame,
    pitchers_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overrides DK_Points_Hitter/Expected_H_Allowed/Expected_BB with a
    trained model's prediction wherever a model artifact
    (config.DFS_*_MODEL_PATH) is present and loadable - otherwise leaves
    the existing heuristic column completely untouched. Adds
    `DK_Points_Hitter_Source`/`Expected_H_Allowed_Source`/
    `Expected_BB_Source` columns ("model" or "heuristic") so the served
    output visibly shows which path produced each number, rather than
    hiding the fallback. Recomputes DK_Points_Pitcher from any overridden
    pitcher components, so the combined total stays internally consistent
    with whichever Expected_H_Allowed/Expected_BB actually got used.

    `hitters_df` is dfs.compute_hitter_dk_points's own output
    (HITTER_DFS_COLUMNS); `hitter_features` is build_hitter_features's
    output for the same slate (carries key_mlbam + HITTER_FEATURE_COLUMNS,
    including the raw matchup ingredients HITTER_DFS_COLUMNS doesn't
    expose). The two share several column names (PA_L, Expected_Bases,
    Game_Hit_Probability, is_home, Matchup_Hit_Probability), so this
    predicts from `hitter_features` directly (never merges its columns
    into `hitters_df`, which would otherwise silently rename the shared
    columns to `_x`/`_y` suffixes - a real bug caught after shipping,
    since HITTER_DFS_COLUMNS' own PA_L/Expected_Bases/etc. would vanish
    from the output CSV and the dashboard would show every cell as blank)
    and merges back only the one new prediction column, keyed on
    key_mlbam. `pitchers_df` is dfs.compute_pitcher_dk_points's own output
    (PITCHER_DFS_COLUMNS), which already carries every column in
    PITCHER_FEATURE_COLUMNS directly under its own names, so no merge is
    needed on the pitcher side at all."""
    hitters_df = hitters_df.copy()
    hitters_df["DK_Points_Hitter_Source"] = "heuristic"
    hitter_model = ml_models.load_model(config.DFS_HITTER_MODEL_PATH)
    if hitter_model is not None and not hitters_df.empty and not hitter_features.empty:
        X = hitter_feature_matrix(hitter_features)
        predicted = hitter_model.predict(X).clip(min=0)
        merge_keys = ["key_mlbam"]
        # Only join on game_pk when both sides have real (non-null) values -
        # otherwise NA game_pk columns (legacy fixtures / dtype object vs
        # int64) would either cartesian-fail or coerce-error on the merge.
        if (
            "game_pk" in hitter_features.columns
            and "game_pk" in hitters_df.columns
            and hitter_features["game_pk"].notna().any()
            and hitters_df["game_pk"].notna().any()
        ):
            merge_keys = ["key_mlbam", "game_pk"]
        prediction_lookup = hitter_features[merge_keys].copy()
        prediction_lookup["_ml_dk_points_hitter"] = predicted
        hitters_df = hitters_df.merge(prediction_lookup, on=merge_keys, how="left")
        has_prediction = hitters_df["_ml_dk_points_hitter"].notna()
        hitters_df.loc[has_prediction, "DK_Points_Hitter"] = hitters_df.loc[has_prediction, "_ml_dk_points_hitter"]
        hitters_df.loc[has_prediction, "DK_Points_Hitter_Source"] = "model"
        hitters_df = hitters_df.drop(columns="_ml_dk_points_hitter")
    hitters_df = hitters_df.sort_values("DK_Points_Hitter", ascending=False)

    pitchers_df = pitchers_df.copy()
    pitchers_df["Expected_H_Allowed_Source"] = "heuristic"
    pitchers_df["Expected_BB_Source"] = "heuristic"
    if not pitchers_df.empty:
        X = pitcher_feature_matrix(pitchers_df)

        h_model = ml_models.load_model(config.DFS_PITCHER_H_ALLOWED_MODEL_PATH)
        if h_model is not None:
            pitchers_df["Expected_H_Allowed"] = h_model.predict(X).clip(min=0)
            pitchers_df["Expected_H_Allowed_Source"] = "model"

        bb_model = ml_models.load_model(config.DFS_PITCHER_BB_MODEL_PATH)
        if bb_model is not None:
            pitchers_df["Expected_BB"] = bb_model.predict(X).clip(min=0)
            pitchers_df["Expected_BB_Source"] = "model"

        if h_model is not None or bb_model is not None:
            pitchers_df["DK_Points_Pitcher"] = (
                pitchers_df["Expected_IP"] * config.DFS_DK_PITCHER_IP_POINTS
                + pitchers_df["Expected_K"] * config.DFS_DK_PITCHER_K_POINTS
                + pitchers_df["Expected_BB"] * config.DFS_DK_PITCHER_BB_POINTS
                + pitchers_df["Expected_H_Allowed"] * config.DFS_DK_PITCHER_H_POINTS
                + pitchers_df["Expected_ER"] * config.DFS_DK_PITCHER_ER_POINTS
            )
    pitchers_df = pitchers_df.sort_values("DK_Points_Pitcher", ascending=False)

    return hitters_df, pitchers_df


def predict_hitter_hit_probability(hitter_features: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, game_pk, Model_Hit_Probability] - the trained
    hit-probability classifier's (config.HITTER_HIT_PROBABILITY_MODEL_PATH,
    scripts/train_hitter_hit_model.py) predicted probability for each row
    in `hitter_features` (build_hitter_features's own output - same
    train/serve feature parity apply_ml_overrides relies on). Returns an
    EMPTY frame (never raises) if the model artifact is missing/corrupt/a
    version mismatch - same ml_models.load_model fallback discipline as
    apply_ml_overrides, so a caller can print/skip rather than crash. This
    model is a classifier, so `.predict_proba(X)[:, 1]` (probability of the
    positive class), not apply_ml_overrides's regression-style `.predict()`."""
    model = ml_models.load_model(config.HITTER_HIT_PROBABILITY_MODEL_PATH)
    if model is None or hitter_features.empty:
        return pd.DataFrame(columns=["key_mlbam", "game_pk", "Model_Hit_Probability"])

    X = hitter_feature_matrix(hitter_features)
    predicted = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "key_mlbam": hitter_features["key_mlbam"].values,
        "Model_Hit_Probability": predicted,
    })
    if "game_pk" in hitter_features.columns:
        out["game_pk"] = hitter_features["game_pk"].values
    else:
        out["game_pk"] = pd.NA
    return out[["key_mlbam", "game_pk", "Model_Hit_Probability"]]
