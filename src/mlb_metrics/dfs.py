"""Estimated DraftKings Classic MLB fantasy points for today's hitters and
probable starting pitchers - a ranked list of good plays for
docs/dfs.html. This module itself has no notion of salary or roster slots
- that's dfs_optimizer.py's job (see its module docstring), built on top
of this module's DK_Points_Hitter/DK_Points_Pitcher output. IMPORTANT:
the optimizer's salaries are a MODELED estimate, not real DraftKings
prices - DraftKings has no public salary API - see
estimated_salary.py's module docstring before assuming otherwise.

DraftKings' scoring rules (config.DFS_DK_*) were confirmed live via web
search (not from memory) - see config.py's DFS section for sources.

## Hitters

DK pays non-linearly for hit type (a double isn't 2x a single's value:
5 != 2*3), but this project only computes a linear Expected_Bases signal
(hitters.compute_wtb) - no per-player 1B/2B/3B/HR rate breakdown exists.
DK_Points_Hitter approximates the non-linear scoring with a single
calibrated "DK points per expected total base" constant
(config.DFS_DK_POINTS_PER_TOTAL_BASE, computed from real Lahman 2015-2025
hit-type shares - see its docstring), applied to Expected_Bases after
scaling it for TODAY's specific matchup via compute_matchup_adjustment - a
ratio of Matchup_Hit_Probability (today's actual opposing pitcher,
platoon+park adjusted) to the batter's own blended Game_Hit_Probability.
That ratio is a real heuristic, not a rigorous transform: it's derived
from hit-PROBABILITY signals but applied multiplicatively to a
TOTAL-BASES signal - the two are correlated but not the same thing (a
good matchup for making contact isn't necessarily proportionally as good
for extra-base power). This was flagged as the single highest-risk
modeling choice here, and the real backtest (scripts/backtest_dfs_rankings.py)
confirmed it: essentially zero correlation (-0.004) against actual
next-game hit-type points - reported honestly at the time, not hidden.

**Update: this heuristic has since been replaced for live output.**
dfs_ml.py trains a gradient-boosting model (scripts/train_dfs_ml_models.py)
on the RAW ingredients this ratio was built from (the batter's own
WAVE/Game_Hit_Probability/Consistency/Approach, the opposing starter's and
bullpen's PAVE, Park_Factor, is_home) instead of the multiplicative ratio
itself, validated on a real held-out backtest (config.py's ML section has
the full numbers: correlation improved from -0.004 to 0.145, MAE beats
both the naive baseline and this module's own heuristic). When a
validated model artifact exists (config.DFS_HITTER_MODEL_PATH),
dfs_ml.apply_ml_overrides swaps it in for DK_Points_Hitter automatically;
compute_matchup_adjustment/compute_hitter_dk_points below are the fallback
path, used whenever no validated artifact is present - they are NOT
removed, and remain what a fresh install without a trained model serves.

**Update: BB/HBP/RBI now included too.** The original v1 excluded five of
DraftKings' nine hitter scoring categories (BB, HBP, runs, RBI, SB) - a
gap the real data made obvious once compared against real elite hitters
(a high-OBP power bat's real DK average is substantially inflated by
walks/runs/RBI, categories this model couldn't see at all, which
compressed exactly the players who should separate from replacement
level). `hitters.compute_extended_dk_rates` now adds recency-windowed
Expected_BB/Expected_HBP/Expected_RBI (same windowed-blend machinery as
Expected_Bases, reusing config.WHOPS_WTB_WINDOWS - see that function's
docstring), each scored at its own DK point value
(config.DFS_DK_HITTER_BB_POINTS/HBP_POINTS/RBI_POINTS) and added directly
to DK_Points_Hitter - NOT run through compute_matchup_adjustment's ratio,
which was already flagged as this module's highest-risk choice and
confirmed not to hold up; compounding it onto three more categories would
just spread that same risk further. `DK_Points_Hitter_HitType` keeps the
old hit-only subtotal alongside the new combined total, for continuity
with the original backtest numbers.

Expected_RBI is itself an approximation (helpers.estimate_rbi: a
completed plate appearance's own bat_score/post_bat_score delta) with two
known, accepted limitations - see that function's docstring (can miss a
run scored on an earlier pitch of the same PA; can't distinguish a
legitimate RBI from an error-scored or GIDP-third-out run, neither of
which is a real RBI).

**Still explicitly excluded from v1**: runs scored (crediting the RUN to
the batter who eventually scored needs tracking a specific runner across
multiple subsequent plate appearances within the same game/inning via
on_1b/on_2b/on_3b - real engineering work with real edge-case risk from
wild pitches/passed balls/caught-stealing/errors moving runners in ways
not obviously visible in this project's row-per-pitch data shape,
deferred as a documented v1.1 follow-up, not silently dropped) and stolen
bases (Statcast's `events` column has no structured stolen-base event
type at all - steals only appear in the free-text `des` description
field, e.g. "Jeremy Peña steals (1) 2nd base", attributed to whoever was
AT BAT during the steal, not the runner - crediting the correct player
would require parsing a name out of free text and matching it back to a
player id, a real departure from this project's established numeric-ID-
join convention (schedule.py's/roster_positions.py's explicit rationale
for avoiding name-matching) - explicitly NOT planned, not just deferred).

## Pitchers

Needs one new signal this project didn't have: pitcher_form.py's
recency-windowed K9/BB9/HR9/IP-per-start (mirroring pitchers.compute_pave's
windowing pattern applied to different underlying stats). DK_Points_Pitcher
combines:
- Expected_IP (= pitcher_form's blended IP_per_start) at
  config.DFS_DK_PITCHER_IP_POINTS/inning.
- Expected_K = K9 * Expected_IP / 9, at config.DFS_DK_PITCHER_K_POINTS/K.
- Expected_BB = BB9 * Expected_IP / 9, at config.DFS_DK_PITCHER_BB_POINTS/BB.
- Expected_H_Allowed = PAVE (pitchers.compute_pave's AB-level hit rate,
  already computed for the live pipeline) * an estimated batters-faced
  count (Expected_IP * config.DFS_BATTERS_FACED_PER_INNING, computed from
  this project's own persisted Statcast - see that constant's docstring),
  at config.DFS_DK_PITCHER_H_POINTS/hit.
- Expected_ER, estimated via FIP rather than a real earned-run signal -
  this project has a consistent, repeated house principle against
  modeling ERA/earned runs directly (defense/sequencing-dependent, not
  just the pitcher's own stuff - the same reasoning pitchers.py's
  Power_A_PLUS and age_curve.py's pitcher metrics both already use FIP
  over ERA for). FIP_Windowed is computed from K9/BB9/HR9 the same
  formula age_curve.py/traditional_stats.py use from raw counts (the
  count-based and rate-based forms are algebraically identical - IP
  cancels out), reusing config.FIP_CONSTANT. Expected_ER =
  FIP_Windowed * Expected_IP / 9, clipped at 0 (a small-sample-windowed
  FIP can go slightly negative before the additive constant fully
  corrects it). Deliberately excludes HBP allowed (pitcher_form.py
  doesn't track it - a documented simplification, consistent with
  age_curve.py's own pitcher FIP already being HBP-inclusive only where
  Lahman happens to carry it).

**Explicitly excluded from v1**: Win (needs a win-probability estimate -
too dependent on the pitcher's own team's offense that day to reasonably
approximate), and the rare discrete bonuses (complete game, complete game
shutout, no-hitter). Relief pitchers are entirely out of scope - IP per
appearance is only a meaningful "per start" number for a starter, not a
bullpen arm's variable-length outing.

**Update: Expected_H_Allowed and Expected_BB have since been replaced for
live output.** The original backtest found Expected_H_Allowed's MAE was
actually WORSE than the naive baseline (a likely systematic scaling bias
in the PAVE * batters-faced calculation above) and Expected_BB only weakly
correlated. scripts/train_dfs_ml_models.py trains a Ridge regression on
the SAME inputs (K9/BB9/HR9/IP_per_start/Expected_IP/Expected_K/
Expected_H_Allowed/FIP_Windowed) for each component separately, validated
on a real held-out backtest - both now beat their naive baseline AND this
module's own heuristic (config.py's ML section has the full numbers).
dfs_ml.apply_ml_overrides swaps in either/both wherever a validated
artifact exists (config.DFS_PITCHER_H_ALLOWED_MODEL_PATH/
DFS_PITCHER_BB_MODEL_PATH), recomputing DK_Points_Pitcher from whichever
components were actually used; compute_pitcher_dk_points below remains the
fallback path when no validated artifact is present.

## Qualifiers

Hitters need config.BACKTEST_MIN_PLATE_APPEARANCES (the same guard
predictions.select_picks uses) and a game today. Pitchers need
config.DFS_PITCHER_MIN_STARTS recorded starts (a 1-2 start sample makes
HR9 especially pure noise) and to be a team's announced probable starter
today - no neutral fallback for an unannounced pitcher, unlike
matchup.py's hitter-side blends, since there's no reasonable "neutral
pitcher" to rank.
"""

import pandas as pd

from mlb_metrics import config

HITTER_DFS_COLUMNS = [
    "key_mlbam", "name_first", "name_last", "team", "opponent", "is_home",
    "game_pk", "game_datetime",
    "PA_L", "PA_R", "Expected_Bases", "Expected_BB", "Expected_HBP", "Expected_RBI",
    "Game_Hit_Probability", "Matchup_Hit_Probability", "Matchup_Ratio",
    "Adjusted_Expected_Bases", "DK_Points_Hitter_HitType", "DK_Points_Hitter",
]

PITCHER_DFS_COLUMNS = [
    "key_mlbam", "name_first", "name_last", "team", "opponent", "is_home",
    "game_pk", "game_datetime",
    "starts", "K9", "BB9", "HR9", "IP_per_start", "Expected_IP",
    "Expected_K", "Expected_BB", "Expected_H_Allowed", "FIP_Windowed",
    "Expected_ER", "DK_Points_Pitcher",
]


def compute_matchup_adjustment(matchup_hit_probability: pd.Series, game_hit_probability: pd.Series) -> pd.Series:
    """Ratio of today's matchup-specific hit probability to the batter's
    own blended baseline - >1 means today's matchup is more favorable than
    their average, <1 less favorable. `game_hit_probability` is floored at
    config.DFS_MATCHUP_RATIO_MIN_DENOM before dividing (a batter with an
    almost-zero blended probability would otherwise blow the ratio up
    arbitrarily on noise), and the final ratio is clipped to
    config.DFS_MATCHUP_RATIO_CLIP - clipping the INPUT denominator and the
    OUTPUT ratio separately guards against two different failure modes,
    same clip-before-and-after-blending spirit as
    matchup.clip_and_blend_pitching_pave."""
    denom = game_hit_probability.clip(lower=config.DFS_MATCHUP_RATIO_MIN_DENOM)
    ratio = matchup_hit_probability / denom
    lo, hi = config.DFS_MATCHUP_RATIO_CLIP
    return ratio.clip(lo, hi)


def compute_hitter_dk_points(
    wave: pd.DataFrame,
    matchup_probability: pd.DataFrame,
    schedule_df: pd.DataFrame,
    min_plate_appearances: int = config.BACKTEST_MIN_PLATE_APPEARANCES,
) -> pd.DataFrame:
    """One row per hitter clearing `min_plate_appearances` (PA_L + PA_R,
    same guard predictions.select_picks uses) whose team has a game in
    `schedule_df` today AND who has a resolvable Matchup_Hit_Probability
    (matchup.compute_matchup_hit_probability's own output - already
    restricted to today's games) - a batter with no game today, or not
    found in `matchup_probability`, is excluded, not defaulted. Sorted by
    DK_Points_Hitter descending."""
    qualified = wave[(wave["PA_L"] + wave["PA_R"]) >= min_plate_appearances].copy()

    schedule_columns = [
        c for c in ("team", "opponent", "is_home", "game_pk", "game_datetime") if c in schedule_df.columns
    ]
    scheduled = qualified.merge(schedule_df[schedule_columns], on="team", how="inner")
    # Join matchup on (key_mlbam, game_pk) whenever both sides carry game_pk
    # so a doubleheader never fans into a cartesian product of
    # schedule-games × matchup-games for the same batter.
    matchup_cols = ["key_mlbam", "Matchup_Hit_Probability"]
    matchup_keys = ["key_mlbam"]
    if "game_pk" in matchup_probability.columns and "game_pk" in scheduled.columns:
        matchup_cols = ["key_mlbam", "game_pk", "Matchup_Hit_Probability"]
        matchup_keys = ["key_mlbam", "game_pk"]
    scheduled = scheduled.merge(matchup_probability[matchup_cols], on=matchup_keys, how="inner")

    scheduled["Matchup_Ratio"] = compute_matchup_adjustment(
        scheduled["Matchup_Hit_Probability"], scheduled["Game_Hit_Probability"]
    )
    scheduled["Adjusted_Expected_Bases"] = scheduled["Expected_Bases"] * scheduled["Matchup_Ratio"]
    scheduled["DK_Points_Hitter_HitType"] = scheduled["Adjusted_Expected_Bases"] * config.DFS_DK_POINTS_PER_TOTAL_BASE
    scheduled["DK_Points_Hitter"] = (
        scheduled["DK_Points_Hitter_HitType"]
        + scheduled["Expected_BB"] * config.DFS_DK_HITTER_BB_POINTS
        + scheduled["Expected_HBP"] * config.DFS_DK_HITTER_HBP_POINTS
        + scheduled["Expected_RBI"] * config.DFS_DK_HITTER_RBI_POINTS
    )

    # game_pk/game_datetime feed the dashboard's slate-filtering optimizer
    # (dfs_optimizer.py) - a schedule_df that lacks them (older fixtures,
    # historical backtest replays) degrades to NA rather than a KeyError on
    # the HITTER_DFS_COLUMNS selection below.
    for column in ("game_pk", "game_datetime"):
        if column not in scheduled.columns:
            scheduled[column] = pd.NA

    return scheduled[HITTER_DFS_COLUMNS].sort_values("DK_Points_Hitter", ascending=False)


def compute_pitcher_dk_points(
    pave: pd.DataFrame,
    pitcher_form: pd.DataFrame,
    schedule_df: pd.DataFrame,
    min_starts: int = config.DFS_PITCHER_MIN_STARTS,
) -> pd.DataFrame:
    """One row per team's announced probable starter today
    (schedule_df's probable_pitcher_key_mlbam) who also clears
    `min_starts` in `pitcher_form`. No neutral fallback for an
    unannounced/unmatched/under-min-starts pitcher - simply excluded, not
    given a league-average placeholder. Relief pitchers are out of scope
    entirely (not just filtered by min_starts) - `pitcher_form` only ever
    carries starts, and only a team's probable STARTER is looked up here.
    Sorted by DK_Points_Pitcher descending."""
    starters = schedule_df.dropna(subset=["probable_pitcher_key_mlbam"]).copy()
    starters["key_mlbam"] = starters["probable_pitcher_key_mlbam"].astype(int)

    merged = starters.merge(pitcher_form, on="key_mlbam", how="inner")
    merged = merged[merged["starts"] >= min_starts].copy()

    pave_columns = [c for c in ("key_mlbam", "name_first", "name_last", "PAVE") if c in pave.columns]
    merged = merged.merge(pave[pave_columns], on="key_mlbam", how="left")
    merged["PAVE"] = merged["PAVE"].fillna(config.MATCHUP_LEAGUE_PAVE_FALLBACK)

    merged["Expected_IP"] = merged["IP_per_start"]
    merged["Expected_K"] = merged["K9"] * merged["Expected_IP"] / 9
    merged["Expected_BB"] = merged["BB9"] * merged["Expected_IP"] / 9
    expected_batters_faced = merged["Expected_IP"] * config.DFS_BATTERS_FACED_PER_INNING
    merged["Expected_H_Allowed"] = merged["PAVE"] * expected_batters_faced

    merged["FIP_Windowed"] = (13 * merged["HR9"] + 3 * merged["BB9"] - 2 * merged["K9"]) / 9 + config.FIP_CONSTANT
    merged["Expected_ER"] = (merged["FIP_Windowed"] * merged["Expected_IP"] / 9).clip(lower=0)

    merged["DK_Points_Pitcher"] = (
        merged["Expected_IP"] * config.DFS_DK_PITCHER_IP_POINTS
        + merged["Expected_K"] * config.DFS_DK_PITCHER_K_POINTS
        + merged["Expected_BB"] * config.DFS_DK_PITCHER_BB_POINTS
        + merged["Expected_H_Allowed"] * config.DFS_DK_PITCHER_H_POINTS
        + merged["Expected_ER"] * config.DFS_DK_PITCHER_ER_POINTS
    )

    # See compute_hitter_dk_points's identical guard - a schedule_df
    # missing game_pk/game_datetime degrades to NA, not a KeyError.
    for column in ("game_pk", "game_datetime"):
        if column not in merged.columns:
            merged[column] = pd.NA

    return merged[PITCHER_DFS_COLUMNS].sort_values("DK_Points_Pitcher", ascending=False)
