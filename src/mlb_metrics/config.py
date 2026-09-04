"""Named constants for the window lengths and blend weights used across the
metrics pipeline. Every value here was previously a hardcoded literal spread
across scripts/wave.py; consolidating them here is what lets Phase B (the
backtesting framework) grid-search alternative weightings without editing
metric logic.

Each *_WINDOWS list is a sequence of (days_back, weight) pairs, applied to a
per-window rate/stat via a straight weighted sum. days_back=None means "full
season to date". The order and values below reproduce the original script's
formulas exactly; note that windows/weights are NOT consistent between
metrics (e.g. WAVE weights recent data highest, PAVE weights it lowest) -
this reflects the original hand-tuning, not a design decision made here.
"""

from datetime import date

# 2026 MLB season boundaries used for the full-season Statcast pull.
SEASON_START = date(2026, 3, 25)
SEASON_END = date(2026, 10, 10)

# WAVE (hitter batting-average-based hit probability): full/81d/30d/10d.
WAVE_WINDOWS = [
    (None, 0.150),
    (81, 0.250),
    (30, 0.275),
    (10, 0.325),
]
# Trials-per-game used to convert an at-bat hit rate into a game hit
# probability: probability = 1 - (1 - rate) ** WAVE_TRIALS_PER_GAME.
WAVE_TRIALS_PER_GAME = 3.5

# Quant-analytics item #3, slice 1 ("uncertainty quantification" -
# Bayesian shrinkage for small-sample hitters): real-unit (at-bats)
# pseudo-observation strength for helpers.shrink_rate, applied to each
# per-window at-bat hit rate inside hitters.compute_wave BEFORE blending.
# Earned via a real full-season backtest (scripts/backtest_shrinkage.py,
# dispatched via .github/workflows/debug_backtest_shrinkage.yml against
# the full persisted 2026 season, n=33,035 PA-gated hitter-dates): swept
# {0, 25, 50}, strength=50 beat strength=0 (unshrunk) on the PA-gated
# population's log_loss (0.6798 vs. 0.6908) - see README's "Bayesian
# shrinkage for small-sample hitters" section for the full numbers,
# including the full (unfiltered) population where the unshrunk log_loss
# blows up to 0.9468 on the small-sample rows this constant targets.
WAVE_SHRINKAGE_STRENGTH = 50.0

# PAVE (pitcher hits-allowed rate, converted from a per-PA rate to a
# per-AB rate by excluding walks/HBP only - NOT strikeouts, which are
# real at-bats; see pitchers.py's module docstring for the real bug this
# used to have): full/30d/81d/15d.
PAVE_WINDOWS = [
    (None, 0.300),
    (30, 0.265),
    (81, 0.230),
    (15, 0.205),
]
# A pitcher must have thrown at least this fraction of the max at-bats faced
# by any pitcher in the pool to count toward the PAVE_PLUS league-average baseline.
PAVE_QUALIFIED_AB_FRACTION = 0.75
# Plate appearances assumed per start, used for Expected_Hits/Bases/HRs.
PAVE_PA_PER_START = 22

# Game_Hit_Probability (rate of games with >=1 hit, not at-bat rate): full/81d/30d/10d.
GAME_HIT_PROB_WINDOWS = [
    (None, 0.175),
    (81, 0.225),
    (30, 0.275),
    (10, 0.325),
]

# Same shrinkage treatment as WAVE_SHRINKAGE_STRENGTH above, but in
# real-unit GAMES (not at-bats) - Game_Hit_Probability's own sample size
# is a game count, typically much smaller than WAVE's at-bat count for
# the same player, so this is deliberately its own separate constant
# rather than reusing WAVE_SHRINKAGE_STRENGTH's value. Earned via the
# same real full-season backtest as WAVE_SHRINKAGE_STRENGTH above
# (n=33,035 PA-gated hitter-dates): swept {0, 25, 50}, strength=25 beat
# strength=0 (unshrunk) on the PA-gated population's log_loss (0.6752
# vs. 0.6983) - see README's "Bayesian shrinkage for small-sample
# hitters" section for the full numbers. See
# hitters.compute_game_hit_probability and scripts/backtest_shrinkage.py.
GAME_HIT_PROB_SHRINKAGE_STRENGTH = 25.0

# hitters.compute_current_hit_streaks: a batter whose most recent game is
# more than this many days before the latest game_date in the data is
# excluded from the "current streak" leaderboard - generous enough for a
# team's occasional single off-day, tight enough to exclude anyone not
# currently playing.
HIT_STREAK_RECENT_DAYS = 5

# WHOPS/RC and WTB share the same window scheme: full/7d/30d/15d.
WHOPS_WTB_WINDOWS = [
    (None, 0.175),
    (7, 0.225),
    (30, 0.275),
    (15, 0.325),
]
# Trials-per-game used for WTB's Expected_Bases (same role as WAVE_TRIALS_PER_GAME).
WTB_TRIALS_PER_GAME = 3.5

# --- Team-level metrics ---

# strength / current_strength / pyth_strength blend: 10g/30g/81g/full.
TEAM_STRENGTH_WINDOWS = [
    (10, 0.5),
    (30, 0.3),
    (81, 0.1),
    (None, 0.1),
]

# offensive_edge (bases scored/allowed per game) blend: 10g/30g/81g/full.
OFFENSIVE_EDGE_WINDOWS = [
    (10, 0.3),
    (30, 0.3),
    (81, 0.2),
    (None, 0.2),
]

# suppression_resistance (1 - pct of games scoring under 3 runs) blend: 162g/81g/30g/10g.
SUPPRESSION_WINDOWS = [
    (162, 0.2),
    (81, 0.2),
    (30, 0.3),
    (10, 0.3),
]

# Pythagorean win-expectation exponent (custom-tuned, not the classic 2).
PYTHAGOREAN_EXPONENT = 1.83

# Every z-scored team metric (Strength, SOS, current_strength, pyth_strength,
# pyth_SOS, Confidence, pyth_Confidence, offensive_edge, suppression_resistance)
# is renormalized as 1 + (z_score * NORMALIZATION_Z_SCALE).
NORMALIZATION_Z_SCALE = 0.15

# Confidence = Strength + SOS * CONFIDENCE_SOS_WEIGHT (and same for pyth_*).
CONFIDENCE_SOS_WEIGHT = 0.3

# --- Backtesting (Phase B) ---

# Minimum combined full-season plate appearances (PA_L + PA_R) for a batter
# to be eligible as a daily "pick" - without this, a batter with a handful
# of at-bats and a lucky hit can show a probability of 1.0 and dominate the
# top of the rankings on pure sample-size noise. This matches a filter the
# original script computed but never actually applied to its output
# (`WAVE[(WAVE['l_at_bat'] + WAVE['r_at_bat']) > 30]`, wave.py:129).
BACKTEST_MIN_PLATE_APPEARANCES = 30

# Default number of top-ranked picks to log/evaluate per day.
BACKTEST_TOP_N = 5

# A hitter only qualifies as a pick if BOTH probability (the binomial-model
# estimate from at-bat-level WAVE) AND Game_Hit_Probability (the directly
# observed rate of games with >=1 hit) clear this bar. Either one alone is
# misleading: high Game_Hit_Probability with low probability is a batter who
# barely gets a hit most games (a lot of 1-for-4/5s); high probability with
# low Game_Hit_Probability is boom-or-bust (multi-hit games mixed with
# 0-fors) - neither is the "reliable to get a hit today" signal the pick is
# supposed to represent. Empirically validated via a 42-day git-history
# replay backtest (see git_backtest.py): requiring both >=0.7 and ranking the
# qualified pool by Approach (Game_Hit_Probability * probability) raised the
# resolved hit rate of the picks that actually clear DAILY_PICK_MIN_PROBABILITY
# from 55% to 65% and cut the Brier score from 0.284 to 0.260, without
# reducing pick coverage (still >=1 pick on all 42 backtested days).
HITTER_MIN_PROBABILITY = 0.7

# REMOVED (v4, HITTER_MODEL_VERSION) - a probability-threshold gate on
# Model_Hit_Probability, used when the model directly ranked the whole
# qualified pool (v3). v4 replaced that with a rank-based shortlist
# (see HITTER_MODEL_SHORTLIST_SIZE below) instead of a probability
# threshold, which needs no calibrated bar to derive/backtest at all - so
# this constant (which had never gotten past an unvalidated 0.5 placeholder,
# since the real backtest that was meant to derive it was blocked by this
# project's dev sandbox's network policy) is gone rather than migrated.

# The model's role in predictions.select_picks is a BROAD QUALITY FILTER,
# not the final ranker: on any day Model_Hit_Probability is available, the
# already-qualified pool is narrowed to its top HITTER_MODEL_SHORTLIST_SIZE
# candidates by Model_Hit_Probability BEFORE the heuristic (Matchup_Approach/
# Approach, via rank_metric) ranks among survivors and picks the final
# top_n. This is a deliberate reversal of v3 (HITTER_MODEL_VERSION), which
# let the model rank the WHOLE pool directly, gated only by the now-removed
# HITTER_MIN_MODEL_PROBABILITY - real live feedback after v3 shipped: a day
# it surfaced Freddie Freeman as the lone recommended pick and dropped
# Jeremy Pena, even though the user explicitly values hitters like Pena/
# Freeman "because of their place in the lineup" - a signal Approach/
# Matchup_Approach implicitly captures via the avg_batting_order/start_rate
# qualifiers (see LINEUP_TOP_HALF_MAX_SLOT/LINEUP_MIN_START_RATE) that
# Model_Hit_Probability doesn't see directly (not one of
# dfs_ml.HITTER_FEATURE_COLUMNS). Keeping the model as a broad shortlist
# gate still kills pure hot-streak outliers (the model's original purpose,
# see HITTER_MODEL_VERSION's v3 paragraph) while handing the final call
# back to the heuristic signal the model doesn't capture.
#
# 10 is an explicit user-specified value, not backtest-derived like almost
# everything else in this file - it should still be validated via a real
# run of scripts/backtest_selection_rule.py (comparing this shortlist
# design against the plain Matchup_Approach heuristic) before being trusted
# or retuned, reported honestly either way. That real run is currently
# blocked in the interactive dev sandbox this was built in (github.com
# egress policy blocks the fetch pipeline.compute_outputs/
# data.get_name_register needs for player name lookups) - dispatch
# .github/workflows/debug_backtest_selection_rule.yml on a real GitHub
# Actions runner post-merge, same deferred-validation path v3 itself never
# completed either.
HITTER_MODEL_SHORTLIST_SIZE = 10

# predictions.select_picks excludes a hitter whose most recent completed
# game (hitters.compute_last_game_dates's Last_Game_Date) is more than this
# many days before the pick date - a career-long PA total and season-long
# WAVE/Game_Hit_Probability rate stay high even if a hitter has been hurt/
# benched for a week, so without this an inactive player could still be
# pick-eligible. A dedicated constant, not reused from
# HIT_STREAK_RECENT_DAYS (the Hit Streaks dashboard leaderboard's own
# recency window) - independently tunable even though both start at 5.
HITTER_MAX_DAYS_SINCE_LAST_GAME = 5

# Stamped onto every hitter pick logged (predictions.select_picks) so
# evaluation.py/the dashboard can segment stats by which selection-logic
# version actually produced a given row, instead of silently blending old
# and new logic into one number - this is what let the qualifier/ranking
# change above (and any future recalibration) actually show up in "did it
# work" stats, rather than being diluted by history logged under old logic
# forever. Bump this string whenever select_picks' qualifier or ranking
# logic meaningfully changes.
#
# v3: pipeline.run() gained a new top rank_metric tier, Model_Hit_Probability
# (the validated logistic regression - see dfs_ml.py's module docstring),
# used whenever the model artifact loads AND schedule/matchup data is
# available; falls back to v2's Matchup_Approach, then Approach, exactly
# as before. Fixes a real bug: Approach/Matchup_Approach are both heavily
# recency-weighted toward a batter's last 10-30 days (GAME_HIT_PROB_WINDOWS/
# WAVE_WINDOWS), so a currently-hot batter dominated the ranking regardless
# of today's actual matchup - Matchup_Hit_Probability's multiplicative
# adjustment wasn't a big enough swing to overturn that. Model_Hit_Probability
# treats matchup ingredients (starter_PAVE, Bullpen_PAVE, Park_Factor,
# platoon-adjusted WAVE) as independent learned features instead. Also
# replaces (not adds to) the probability gate on days this tier is active -
# see (removed in v4) HITTER_MIN_MODEL_PROBABILITY.
#
# v4: reverses v3's "model ranks the whole pool" design. Real feedback
# after v3 shipped live: on a day the model surfaced Freddie Freeman as the
# LONE recommended pick and dropped Jeremy Pena, even though the user
# explicitly values hitters like Pena/Freeman "because of their place in
# the lineup" - the model doesn't see lineup-order/everyday-player signals
# directly, while Approach/Matchup_Approach implicitly do via the
# avg_batting_order/start_rate qualifiers. v4 keeps Model_Hit_Probability
# as a BROAD quality gate (top HITTER_MODEL_SHORTLIST_SIZE candidates by
# model score) but hands the FINAL narrowing back to Matchup_Approach/
# Approach, so a pure hot-streak outlier the model doesn't rate can't win,
# but a legitimate lineup-order signal the model doesn't capture can still
# decide among the model's own shortlisted favorites. Also drops the
# min_model_probability threshold entirely (see HITTER_MODEL_SHORTLIST_SIZE)
# - a rank-based cutoff needs no probability bar to calibrate.
HITTER_MODEL_VERSION = "v4-model-shortlist"

# Champion/challenger integration for a single authoritative hitter
# probability (`Final_Hit_Probability`). Modes:
#   legacy - production selection unchanged (v4 shortlist + Matchup_Approach)
#   shadow - production picks unchanged; log Final_Hit_Probability + components
#           + shadow_rank without changing official picks
#   live   - Final_Hit_Probability is the only quantity for ranking, thresholding,
#            logging, Brier/log-loss, dashboard display, and streak grading
#
# Default is shadow unless a committed nested-validation report clearly
# satisfies the promotion gate (see hitter_probability_model.promotion_gate_satisfied).
# Do not flip this to live without that gate.
HITTER_SELECTION_MODES = ("legacy", "shadow", "live")
HITTER_SELECTION_MODE = "shadow"
HITTER_MODEL_VERSION_LIVE = "v5-final-hit-probability"
HITTER_PROMOTION_GATE_REPORT_PATH = (
    "reports/model_validation/hitter_opportunity_probability_nested.json"
)

# Beat the Streak Tracker (dashboard): a batter is only "recommended" if
# evaluation._combined_probability (the mean of whichever of
# predicted_probability/Game_Hit_Probability, probability, and
# Matchup_Hit_Probability are available for that pick - see that function's
# docstring) clears this bar - on a day with no good matchups, that's zero
# picks; on a strong day, up to DAILY_PICK_MAX picks.
#
# Previously this gated on predicted_probability (Game_Hit_Probability)
# alone at 0.80 - a single-signal bar that ignored `probability` and
# `Matchup_Hit_Probability` entirely, and produced 0-pick days whenever GHP
# landed just under 0.80 even with a strong matchup (e.g. a real 0.79 GHP/
# strong-matchup day that should have surfaced a pick, didn't). Blending in
# the other two signals is the fix, but also means the same nominal
# threshold is no longer directly comparable to before - a mean of two or
# three probabilities runs lower than GHP alone whenever `probability`
# trails GHP (common - see JOINT_PROBABILITY_GATE_COLUMNS's docstring),
# so keeping 0.80 with the new blend would have been STRICTER, not looser
# (empirically: 3/42 zero-pick days at 0.80 in the backtest below, vs 1/42
# under the old GHP-only gate).
#
# 0.77 is empirically validated via a 42-day git-history replay backtest
# (git_backtest.reconstruct_historical_picks, resolved against persisted
# Statcast - Matchup_Hit_Probability is NaN throughout that replay window,
# since it's never persisted to git history, so this validates the
# GHP+probability blend specifically): the highest threshold that still
# achieves full day coverage (0/42 zero-pick days) while matching the best
# resolved hit rate/Brier score plateau seen across the whole 0.70-0.77
# range (62.5% hit rate, 0.2894 Brier, n=56 resolved picks) - thresholds
# above 0.77 both start producing zero-pick days again AND stop improving
# (sometimes worsening) hit rate/Brier on this sample. Revisit once more
# live picks accumulate carrying real (non-NaN) Matchup_Hit_Probability.
#
# That revisit already happened, informally: once live runs started
# carrying real (non-NaN) Matchup_Hit_Probability most days, the blended
# mean runs lower on an ordinary day than the NaN-heavy 42-day replay ever
# exercised - real live data hit a 5-day-straight zero-pick stretch
# (2026-08-11 through 2026-08-15) where the top-ranked candidate's real
# combined probability landed at 0.71-0.77, just under this bar every
# single day, despite real qualified candidates existing all five days.
# Rather than re-chase a moving threshold with another point estimate, this
# constant's MEANING changed instead of its value (see
# evaluation.graded_daily_picks): it's no longer "the bar a day needs to
# clear to show anything" - the dashboard now always shows its real top
# DAILY_PICK_MAX candidates, each individually graded "recommended" (still
# means "clears this exact bar") or "speculative" (doesn't, shown for
# visibility, doesn't count toward the tracked streak - see
# evaluation._recommended_picks/streak_progression, both unchanged). This
# constant's own validated value/meaning as a streak-counting bar is
# untouched; only "what happens below it" changed from "nothing shown" to
# "shown, honestly labeled."
DAILY_PICK_MAX = 2
DAILY_PICK_MIN_PROBABILITY = 0.77

# Streak-action decision layer (mlb_metrics.streak_policy): sit / one / two
# based on Final_Hit_Probability outcome triples + DP expected utility.
# Default shadow - does not change official picks until a promotion gate
# shows untouched outer-fold improvement vs legacy fixed two-pick /
# threshold policies.
STREAK_POLICY_MODES = ("off", "shadow", "live")
STREAK_POLICY_MODE = "shadow"
STREAK_POLICY_UTILITY = "expected_final_streak"
STREAK_POLICY_TARGET = 57
STREAK_POLICY_MAX_CANDIDATES = 12
STREAK_POLICY_DEFAULT_APPEAR = 0.85
# Conservative dependence penalties (additive). Default 0 until earned
# inside nested validation - never a claimed learned correlation.
STREAK_POLICY_DEPENDENCE_SAME_GAME = 0.0
STREAK_POLICY_DEPENDENCE_SAME_TEAM = 0.0
STREAK_POLICY_DEPENDENCE_SHARED_PARK = 0.0
STREAK_POLICY_DEPENDENCE_OPPOSING_SAME_GAME = 0.0
STREAK_POLICY_SHADOW_DECISIONS_PATH = (
    "data/predictions/streak_policy_shadow_decisions.csv"
)
STREAK_POLICY_PROMOTION_GATE_REPORT_PATH = (
    "reports/model_validation/streak_policy_nested.json"
)
STREAK_POLICY_REPORT_NAME = "streak_policy_backtest.json"

# Quant-analytics item #4, slice 2 ("decision theory for the actual game
# structure" - correlation from being in the same game): predictions.
# select_picks's #2-pick diversification tie-break. When the #1 and #2
# ranked candidates share a real game_pk, and a candidate from a
# DIFFERENT game ranks within this margin of the original #2's own
# rank_metric value, the different-game candidate is preferred instead -
# same-game correlation (shared weather/park/pitching-matchup quality)
# can only ever raise the chance both picks miss together, never lower
# it, so diversifying when it's nearly free is a real, robust
# improvement whose direction doesn't depend on knowing the exact
# correlation magnitude. 0.0 (the live default) is the exact null
# hypothesis - today's unmodified single-column ranking, bit-for-bit -
# until a real backtest earns a nonzero value (see
# scripts/backtest_same_game_diversification.py and README's "Same-game
# diversification" section for the real go/no-go numbers once run).
SAME_GAME_DIVERSIFICATION_MARGIN = 0.0

# --- Lineup awareness ---
#
# Backtesting found ~30% of logged top-5 picks had zero at-bats on the day
# they were picked - the batter wasn't even in the lineup. These qualifiers
# (see lineup.py, predictions.select_picks) use each batter's *historical*
# batting-order slot - derived from Statcast data already persisted, via
# data.assign_batting_order - as a proxy for "reliably plays, and plays high
# enough in the order to get real at-bats", without needing same-day lineup
# confirmation (which isn't available until a few hours before first pitch -
# see schedule.py).

# Rolling window (team games, not days) used to compute batting-order
# consistency. A game-count window, unlike the day-count *_WINDOWS lists
# above, since roster usage doesn't follow a calendar cadence.
LINEUP_WINDOW_GAMES = 20

# A batter's average batting-order slot over the window must be strictly
# above this to count as "top half" - the 9-man order's true median is 5,
# so 4.5 means strictly better than a 50/50 split, matching "more at-bats
# than the bottom of the order" precisely (a mean of exactly 5.0 is not
# "top half").
LINEUP_TOP_HALF_MAX_SLOT = 4.5

# Fraction of the team's games in the window (or fewer, early in the season)
# a batter must have actually started in to count as a regular, not a bench
# player on a hot week or a recent call-up riding an unsustainable streak.
# A rate, not an absolute game count, so it doesn't hard-disqualify every
# batter before a team has played LINEUP_WINDOW_GAMES games.
LINEUP_MIN_START_RATE = 0.6

# Historical hitter-opportunity training rows (hitter_training_data.py): a
# batter is a pregame candidate for a team-game iff their latest known team
# strictly before the game is that team AND they appeared for that team in
# at least one of (a) the team's previous N games or (b) any game for that
# team within D calendar days. Either window is sufficient (OR). The actual
# lineup is never used to decide historical candidates - it is labels only.
# Live confirmed-lineup snapshots may add previously unseen players via
# league priors (add_confirmed_lineup_candidates); that path is not used in
# historical training and must not be used to "fix" candidate coverage.
HITTER_OPPORTUNITY_LOOKBACK_TEAM_GAMES = 10
HITTER_OPPORTUNITY_LOOKBACK_CALENDAR_DAYS = 14

# Confirmed-lineup snapshots (mlb_metrics.lineup_snapshots). Stage A
# (scripts/debug_statsapi_lineups.py) must confirm the live Stats API
# field paths before flipping LINEUP_API_SCHEMA_CONFIRMED to True.
LINEUP_API_SCHEMA_CONFIRMED = False
LINEUP_CONFIRMED_SCRATCH_RISK = 0.02  # P(scratch | confirmed starter) before first pitch
LINEUP_SNAPSHOT_AUDIT_PATH = "data/predictions/lineup_snapshots_audit.csv"
LINEUP_SNAPSHOT_LATEST_PATH = "data/predictions/lineup_snapshots_latest.csv"
LINEUP_LOCK_WINDOW_HOURS = 6.0  # only recompute unstarted games starting within this window
LINEUP_LOCK_SHADOW_DECISIONS_PATH = "data/predictions/lineup_lock_runs.csv"

# As-of schedule / probable-starter snapshots (mlb_metrics.schedule_snapshots).
# Append-only captures of the slate the live pipeline actually saw, so
# game-model backtests can use production-equivalent information rather
# than silently substituting the actual starter.
SCHEDULE_SNAPSHOTS_PATH = "data/predictions/schedule_snapshots.csv"
SCHEDULE_BACKTEST_MODES = ("as_of_snapshot", "actual_starter", "missing_snapshot")
# Primary backtest mode: production-equivalent as-of snapshots.
SCHEDULE_BACKTEST_MODE_DEFAULT = "as_of_snapshot"
# Morning prediction hour (UTC) used when a backtest has a target date but
# no recorded pipeline capture time. Arizona ~8am is UTC 15:00 year-round.
SCHEDULE_BACKTEST_MORNING_HOUR_UTC = 15
# Honest floor before treating as-of evaluation as an adequate
# production-equivalent sample. Below this, label insufficient and do not
# merge actual-starter diagnostic rows into the as-of set.
SCHEDULE_AS_OF_MIN_PRODUCTION_GAMES = 100

# Immutable audit copies of every published prediction batch (morning +
# lineup_lock). The published predictions.csv / game_predictions.csv files
# keep only the latest valid pregame row per key.
PREDICTIONS_AUDIT_PATH = "data/predictions/predictions_audit.csv"
GAME_PREDICTIONS_AUDIT_PATH = "data/predictions/game_predictions_audit.csv"

# --- Matchup awareness (Part B) ---
#
# Blends a batter's own AB-level hit rate (WAVE) with the specific pitching
# they're projected to face today: their team's opponent's probable starter
# (most of their at-bats) and that opponent's bullpen (the rest), via a log5
# (odds-ratio) combination against the league-average PAVE - the standard
# sabermetric technique for combining two rate stats measured against a
# shared baseline (see matchup.py). Uses raw PAVE/Bullpen_PAVE (an actual
# batting-average-against-scale rate), not PAVE_PLUS/Bullpen_PAVE_PLUS (a
# league-normalized ratio) - log5 needs real rates on the same 0-1 scale as
# WAVE, not a ratio centered on 1.0. game_picks.py's team-level model is
# unaffected and keeps using PAVE_PLUS/clip_and_blend_pitching_quality (a
# different, still-appropriate use for its already-normalized composites).

# Assumed share of a batter's at-bats against the opposing starter vs. the
# opposing bullpen (roughly 2-3 of a 3-5 AB game). Sums to 1.0 so the blend
# has no systematic bias vs. the league-average PAVE it's measured against.
MATCHUP_STARTER_AB_SHARE = 0.6
MATCHUP_BULLPEN_AB_SHARE = 0.4

# PAVE_PLUS/Bullpen_PAVE_PLUS are clipped to this range before blending (not
# just clipping the final probability) - assemble_pitchers has no upper
# bound on individual PAVE_PLUS, so a small-sample outlier (an opener's short
# outing, early season) could otherwise multiply a good Game_Hit_Probability
# past 1.0 and erase real distinctions at the ceiling clip. Used by
# game_picks.py's clip_and_blend_pitching_quality (team-level model).
MATCHUP_PAVE_PLUS_CLIP = (0.5, 1.75)

# Same outlier protection as MATCHUP_PAVE_PLUS_CLIP, but expressed as a
# multiplier of that day's league-average PAVE (since raw PAVE isn't
# pre-normalized to a mean of 1.0 the way PAVE_PLUS is) - a starter/bullpen
# PAVE is clipped to [lo*league_pave, hi*league_pave] before blending. Used
# by matchup.py's clip_and_blend_pitching_pave (hitter-level model).
MATCHUP_PAVE_CLIP_MULTIPLIER = (0.5, 1.75)

# Fallback league-average PAVE (roughly historical MLB batting-average-against)
# used only when the day's `pave` table can't yield a real value (e.g. no
# qualified pitchers yet, or a test fixture) - matchup.py's log5 blend needs
# some league baseline to divide by and this should never bind in practice
# once the season is underway.
MATCHUP_LEAGUE_PAVE_FALLBACK = 0.245

# Whether a batter faces a lefty or righty starter today changes which of
# their platoon-split rates (WAVE_L/WAVE_R - see hitters.compute_wave) is
# the relevant one; blending both together (the old behavior) understates a
# real platoon-driven matchup either way. matchup.py picks WAVE_L/WAVE_R
# based on the probable starter's own Throws (pitchers.compute_pitcher_throws)
# and falls back to the handedness-blended WAVE when Throws is unknown
# (unannounced starter, not found in pave.csv) - same missing-data
# degradation pattern used throughout this module. Always applied (no
# separate weight knob) when the data is available - see
# MATCHUP_PARK_FACTOR_WEIGHT's docstring for the backtest that validated
# turning it on, which validated platoon and park together.

# Statcast's own venue proxy: each team's home games are treated as being
# at one park (see teams.compute_park_factors) - Park_Factor is the
# combined runs/game at that venue relative to the across-all-parks
# average, clipped to this range before blending into the matchup AB rate
# so a small-sample outlier park doesn't swing a probability further than
# any real park effect would (MLB parks realistically span roughly this
# range - Coors-like hitter's parks near the top, extreme pitcher's parks
# near the bottom).
MATCHUP_PARK_FACTOR_CLIP = (0.85, 1.15)

# How much of the (clipped) park effect gets applied to the matchup AB
# rate: effective_multiplier = 1 + MATCHUP_PARK_FACTOR_WEIGHT * (Park_Factor - 1).
# 0.0 is a no-op (fully off); 1.0 applies the park effect in full.
#
# Empirically validated via a 15-date persisted-Statcast backtest (recomputes
# wave/pave/confidence fresh per date via pipeline.compute_outputs, same
# technique game_picks_backtest.py's persisted variant uses - platoon/park
# columns don't exist in old git-committed snapshots, so a git-history
# replay can't see them), comparing four variants on the full top-N
# qualified-candidate pool (n=43-52 resolved picks - the more reliable
# sample; the further-filtered "recommended" rank<=2 subset only had
# n=11-18, too small to trust on its own and noisier in the opposite
# direction on this window):
#   baseline (neither):        58% hit rate, 0.286 Brier
#   platoon only:               62% hit rate, 0.261 Brier
#   park only:                  57% hit rate, 0.293 Brier (alone, roughly a wash)
#   both (this default):        65% hit rate, 0.244 Brier - best of all four
# Platoon and park together beat either alone or neither on the larger
# sample, so both ship on by default; revisit (especially park alone, whose
# solo read was weak here) once more dates accumulate.
MATCHUP_PARK_FACTOR_WEIGHT = 1.0

# Pitch-type-specific platoon matchup: does this batter's own real
# fastball/breaking/offspeed hit-rate split (hitters.compute_pitch_family_rates)
# suggest a better/worse-than-average day against THIS starter's own real
# pitch-mix (pitchers.compute_pitch_arsenal), independent of the
# handedness platoon adjustment above (a batter can have a real fastball/
# breaking-ball split that a same-handed vs. opposite-handed platoon split
# doesn't capture at all). Same clip-then-blend shape as
# MATCHUP_PARK_FACTOR_CLIP - one small-sample outlier pitcher's mix (a
# reliever with a handful of tracked pitches) shouldn't swing a
# probability further than a real arsenal skew would.
MATCHUP_PITCH_ARSENAL_CLIP = (0.85, 1.15)

# Same role as PITCHER_MATCHUP_WEIGHT_GRID - swept by a real backtest
# before choosing MATCHUP_PITCH_ARSENAL_WEIGHT below. 0.0 is the built-in
# null hypothesis (multiplier == 1.0 exactly, not an approximation of it).
MATCHUP_PITCH_ARSENAL_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

# Real league-average pitch-family usage mix, used as the neutral baseline
# a batter's own family rates are compared against (the same role
# MATCHUP_LEAGUE_PAVE_FALLBACK plays for the handedness-platoon blend) -
# computed once directly from the real persisted data/raw/statcast_2026.parquet
# (helpers.pitch_type_family over every real classified pitch, 2026-08-19:
# 307,172 fastball / 167,553 breaking / 81,627 offspeed of 556,352
# classified pitches) rather than assumed; matchup._league_arsenal_mix
# recovers the real mean directly from whatever `pave` it's given at
# runtime and only falls back to this constant when that can't yield one
# (empty pave, or a test fixture missing the Fastball_Rate/Breaking_Rate/
# Offspeed_Rate columns).
MATCHUP_LEAGUE_ARSENAL_FALLBACK = {"fastball": 0.5521, "breaking": 0.3012, "offspeed": 0.1467}

# Ships at 0.0 (informational-only, exactly reproduces the pre-arsenal
# matchup rate) until a real backtest earns a nonzero default - same
# "ship conservatively" precedent as PITCHER_MATCHUP_OFFENSE_WEIGHT, not
# MATCHUP_PARK_FACTOR_WEIGHT's (that one earned its 1.0 default from a
# real backtest BEFORE shipping; this one hasn't been backtested yet).
# Re-run once a real matchup-weight backtest exists and update this
# comment with the real numbers, honestly, either direction.
MATCHUP_PITCH_ARSENAL_WEIGHT = 0.0

# --- Automated Game Picks ---
#
# Predicts a winner for each of today's games from team-level metrics (not
# hitter picks) - see game_picks.py. Unvalidated first pass, same as Matchup
# awareness above: logged and tracked before ever being trusted.

# Equal-weighted blend of the four team-level signals the user asked for by
# name into one offensive composite rating. All four inputs are already
# z-normalized to mean 1.0 (NORMALIZATION_Z_SCALE), so a straight weighted
# average needs no further rescaling. suppression_resistance is deliberately
# counted both directly and inside true_power (which already averages it
# with offensive_edge) - the user asked for both signals by name.
GAME_PICK_COMPOSITE_WEIGHTS = [
    ("pyth_Strength", 0.25),
    ("pyth_Confidence", 0.25),
    ("suppression_resistance", 0.25),
    ("true_power", 0.25),
]

# A game is only "picked" if the favored side's win probability clears this
# bar - a day can surface 0 or more picks depending on how much separation
# the model sees, not a forced pick every game. Much lower than
# DAILY_PICK_MIN_PROBABILITY (0.77): single-game MLB win probabilities are
# compressed near 50/50 even for real favorites, so reusing the hitter-pick
# bar would produce picks on almost no days. First-pass default, meant to be
# recalibrated once real data accumulates.
GAME_PICK_MIN_PROBABILITY = 0.58

# Floors each team's matchup-adjusted rating before computing
# home_rating / (home_rating + away_rating), purely as a degenerate-input
# guard - composites center around 1.0, so this shouldn't bind in practice.
GAME_PICK_RATING_FLOOR = 0.05

# How much Power_A_PLUS (total-bases-allowed rate against, a run-prevention/
# ERA-like signal - see pitchers.py's module docstring) contributes to the
# opposing-pitching-quality multiplier, blended against PAVE_PLUS (hit-rate
# against): final = (1-w)*PAVE_PLUS_quality + w*Power_A_PLUS_quality.
# Empirically validated via a 30+ day persisted-Statcast backtest (see
# game_picks.py's module docstring): an equal 0.5/0.5 blend beat pure
# PAVE_PLUS (w=0.0) on both accuracy (56.3% vs 54.3%) and Brier score
# (0.2493 vs 0.2517), and beat other weights tried (0.25, 1.0 i.e.
# Power_A_PLUS alone). Revisit once more resolved games accumulate.
GAME_PICK_SUSCEPTIBILITY_WEIGHT = 0.5

# Same purpose as HITTER_MODEL_VERSION above, for game_predictions.py -
# bump whenever compute_game_win_probabilities'/select_game_picks' logic
# meaningfully changes (e.g. GAME_PICK_SUSCEPTIBILITY_WEIGHT's introduction).
GAME_PICK_MODEL_VERSION = "v1"

# --- Age Curves (exploratory, separate page - not part of the daily pick pipeline) ---
#
# Given a current player's age and season stat line (traditional_stats.py,
# not WAVE/PAVE - Lahman's historical seasons have no Statcast-derived
# signal to compare against), finds the K nearest historical same-position
# seasons at the same age (age_curve.py) and projects next season from what
# those comparables actually did next. No era/park adjustment - see
# age_curve.py's module docstring.

# A separate curve/projection/comparable-list per metric, not one blended
# number - e.g. power (SLG) typically peaks earlier and declines faster
# than plate discipline (OBP), contact rate (AVG) tends to be more stable -
# so a single composite would hide that. Same "metric" convention already
# used by predictions.csv/game_predictions.csv elsewhere in this project (a
# metric name plus generic value columns).
AGE_CURVE_HITTER_METRICS = ["AVG", "OBP", "SLG", "OPS"]

# Deliberately no ERA - same reasoning as pitchers.py's Power_A_PLUS (ties
# to sequencing/defense/inherited runners, not just the pitcher's own
# stuff). K9/BB9/HR9 are the three "own stuff" component rates (mirroring
# AVG/OBP/SLG's role on the hitter side); FIP is the composite of those
# same three components into one number (mirroring OPS), deliberately
# defense-independent by construction - the exact ERA-scale gap this
# project has avoided elsewhere for the live pick models, but appropriate
# here since this page is explicitly about historical career-arc
# comparison, not a prediction signal feeding a pick.
AGE_CURVE_PITCHER_METRICS = ["K9", "BB9", "HR9", "FIP"]

# FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + this constant. The constant is a
# pure additive shift (traditionally computed per-season so lgFIP==lgERA)
# that puts FIP on the same familiar ~3-4 scale as ERA. For age_curve.py it
# does NOT affect comparable-search distances or projections at all (both
# are computed as differences, and an additive constant cancels out of any
# difference) - purely cosmetic there, not a modeling simplification. A
# single fixed value is used here instead of a real per-season constant
# (which would need runs-allowed/ERA data this project deliberately doesn't
# use elsewhere). Also reused by dfs.py's Expected_ER estimate (see its
# docstring) - shared across both consumers rather than duplicated, since
# it's the same constant either way. Not "AGE_CURVE_*"-prefixed despite
# originating there, since it's no longer Age-Curves-only.
FIP_CONSTANT = 3.10

# Minimum at-bats for a hitter season (current or historical) to be
# eligible at all - a tiny-sample season can otherwise show an OPS of 0 or
# 3.000+ on pure noise, same reasoning as BACKTEST_MIN_PLATE_APPEARANCES above.
AGE_CURVE_MIN_AB = 200

# Minimum innings pitched for a pitcher season to be eligible - same
# small-sample-noise reasoning as AGE_CURVE_MIN_AB. Modest (a swingman/
# spot-starter's workload, not a qualified-ERA-title threshold), since this
# is an exploratory comparison tool, not a selection gate.
AGE_CURVE_MIN_IP = 50

# How many nearest same-age comparables (by whichever metric is in use) to
# use per projection.
AGE_CURVE_K_NEIGHBORS = 25

# Comparables are drawn from ages within this many years of the current
# player's age (0 = exact age match only). A small window trades strict
# age-comparability for a larger, less noisy comparable pool.
AGE_CURVE_AGE_WINDOW = 1

# Empirically validated (scripts/backtest_age_curve.py) against 500 real
# sampled player-seasons from 2010-2019 (32,158 qualified historical
# hitter-seasons total, 1871-2025, from Lahman's actual database via
# Lahman_Raw/ - not synthetic data), using only comparable data available
# at or before each test season's own year (no lookahead). Every metric
# beat the naive "always guess the sample's mean" baseline, with a real
# (not noise-level) positive correlation between projected and actual
# next-season value:
#   AVG: MAE 0.0245 vs. 0.0254 baseline, correlation 0.352 (n=355 scored)
#   OBP: MAE 0.0274 vs. 0.0284 baseline, correlation 0.429 (n=355 scored)
#   SLG: MAE 0.0533 vs. 0.0582 baseline, correlation 0.473 (n=355 scored)
#   OPS: MAE 0.0756 vs. 0.0789 baseline, correlation 0.446 (n=355 scored)
# The remaining 145/500 sampled seasons had no comparable with a resolvable
# next season and couldn't be scored (see project_next_season's docstring -
# reported, not hidden). SLG/OPS show the strongest signal; AVG/OBP beat
# the baseline only modestly, consistent with contact rate/plate discipline
# being harder to project from a same-age same-value comparable pool alone
# than power is. Re-run this backtest (and update these numbers) after any
# change to AGE_CURVE_K_NEIGHBORS/AGE_CURVE_AGE_WINDOW/AGE_CURVE_MIN_AB.

# Same methodology, pitcher metrics: 500 real sampled pitcher-seasons from
# 2010-2019 (28,127 qualified historical pitcher-seasons total, 1871-2025):
#   K9:  MAE 1.1476 vs. 1.5878 baseline, correlation 0.746 (n=296 scored)
#   BB9: MAE 0.6277 vs. 0.7776 baseline, correlation 0.616 (n=296 scored)
#   HR9: MAE 0.3312 vs. 0.3264 baseline, correlation 0.321 (n=296 scored)
#   FIP: MAE 0.5907 vs. 0.6606 baseline, correlation 0.440 (n=296 scored)
# K9/BB9/FIP all clearly beat their naive baselines. HR9 is reported
# honestly as a wash, not hidden or rounded away: its MAE is essentially
# tied with (very slightly worse than) just guessing the sample mean,
# despite a real positive correlation - year-to-year home-run rate is
# notoriously volatile (batted-ball luck, park effects, defense) even
# though it correlates with itself somewhat. Treat HR9 projections as a
# weak signal. Re-run this backtest (and update these numbers) after any
# change to AGE_CURVE_K_NEIGHBORS/AGE_CURVE_AGE_WINDOW/AGE_CURVE_MIN_IP.

# --- DFS Player Rankings (docs/dfs.html, dfs.py) ---
#
# Estimated DraftKings Classic MLB fantasy points for today's hitters and
# probable starters - a ranked list of good plays. See dfs.py's module
# docstring for the full methodology and v1 limitations. A separate
# "Optimal Lineup" section further below builds a salary-cap-and-position
# optimizer on top of these projections - see that section's own comment
# for why its salaries are a modeled estimate, not real DraftKings prices.

# DraftKings Classic MLB scoring, confirmed live (not from memory) via
# https://dknetwork.draftkings.com/2020/05/29/beginner-mlb-dfs-scoring/ and
# https://www.draftkings.com/help/rules/2/59 - hitters.
DFS_DK_HITTER_SINGLE_POINTS = 3
DFS_DK_HITTER_DOUBLE_POINTS = 5
DFS_DK_HITTER_TRIPLE_POINTS = 8
DFS_DK_HITTER_HR_POINTS = 10
# DFS_DK_HITTER_BB_POINTS/HBP_POINTS/RBI_POINTS are now used live in
# dfs.compute_hitter_dk_points (hitters.compute_extended_dk_rates supplies
# the Expected_BB/Expected_HBP/Expected_RBI these multiply against - see
# dfs.py's module docstring for why runs/SB are still excluded).
# DFS_DK_HITTER_RUN_POINTS/SB_POINTS remain unused (no runs-scored-by-
# batter or stolen-base signal exists - see dfs.py's module docstring for
# exactly why each is deferred/not-planned) - kept so dfs_backtest.py can
# still report the FULL real DK score for a historical day alongside the
# modeled subset, honestly, instead of silently comparing against a
# partial actual.
DFS_DK_HITTER_RUN_POINTS = 2
DFS_DK_HITTER_RBI_POINTS = 2
DFS_DK_HITTER_BB_POINTS = 2
DFS_DK_HITTER_HBP_POINTS = 2
DFS_DK_HITTER_SB_POINTS = 5
# DraftKings removed the caught-stealing penalty from its current ruleset
# (confirmed live via web search, July 2026) - 0, not a negative value.
DFS_DK_HITTER_CS_POINTS = 0

# DraftKings Classic MLB scoring - pitchers. Same sources as above.
DFS_DK_PITCHER_IP_POINTS = 2.25  # per inning (0.75 per out)
DFS_DK_PITCHER_K_POINTS = 2
DFS_DK_PITCHER_BB_POINTS = -0.6
DFS_DK_PITCHER_H_POINTS = -0.6
DFS_DK_PITCHER_HBP_POINTS = -0.6
DFS_DK_PITCHER_ER_POINTS = -2
# Out of scope for v1 (see dfs.py's module docstring: Win needs a
# win-probability estimate this project doesn't build; CG/CGSO/no-hitter
# are rare discrete events with no signal here) - kept, unused, purely as a
# visible reminder of what real DK pitcher scoring includes that this
# ranking does NOT model, so a reader doesn't mistake DK_Points_Pitcher for
# the full real score.
DFS_DK_PITCHER_WIN_POINTS = 4
DFS_DK_PITCHER_CG_POINTS = 2.5
DFS_DK_PITCHER_CGSO_POINTS = 2.5
DFS_DK_PITCHER_NO_HITTER_POINTS = 5

# DK's hitter scoring is non-linear in total bases (a double isn't 2x a
# single's value: 5 != 2*3), but this project only computes a linear
# Expected_Bases signal (hitters.compute_wtb) - no per-player 1B/2B/3B/HR
# rate breakdown exists. This is a single calibrated "DK points per
# expected total base" approximating that non-linear scoring: computed
# from REAL MLB-wide hit-type shares, Lahman batting 2015-2025 (not a
# guessed league average) -
#   singles 272,744 / doubles 84,058 / triples 7,831 / HR 59,419 of
#   424,052 total hits -> shares 64.32% / 19.82% / 1.85% / 14.01%
#   TB/hit  = .6432*1 + .1982*2 + .0185*3 + .1401*4 = 1.6555
#   pts/hit = .6432*3 + .1982*5 + .0185*8 + .1401*10 = 4.4696
#   pts/TB  = 4.4696 / 1.6555 = 2.6998
DFS_DK_POINTS_PER_TOTAL_BASE = 2.6998

# Ratio of Matchup_Hit_Probability to the batter's own blended
# Game_Hit_Probability, used to scale Expected_Bases for today's specific
# matchup (see dfs.compute_matchup_adjustment). Game_Hit_Probability is
# floored at this value before dividing, so a batter with almost no
# recorded games doesn't blow the ratio up arbitrarily large on noise.
DFS_MATCHUP_RATIO_MIN_DENOM = 0.05

# Same outlier protection as MATCHUP_PAVE_PLUS_CLIP, reused for
# consistency: a 2x swing either direction off a real matchup edge is
# already large; wider than that is almost certainly small-sample noise,
# not real signal. Unvalidated at this exact value - revisit once
# scripts/backtest_dfs_rankings.py has real numbers.
DFS_MATCHUP_RATIO_CLIP = (0.5, 1.75)

# pitcher_form.compute_pitcher_dfs_form's window scheme - deliberately a
# SEPARATE constant from PAVE_WINDOWS (not reused), even though it starts
# from the same weighting, so tuning one never silently perturbs the other
# (PAVE_WINDOWS also feeds matchup.py's log5 blend and game_picks.py, both
# already backtested against the current weights). Windows are wider than
# PAVE_WINDOWS's: PAVE blends AT-BAT-level data (hundreds of rows/window
# even in a short window); this blends START-level data (~1 start per 5
# days for a rotation pitcher), so PAVE's 15-day window would leave as few
# as 2-3 real starts in the tightest bucket. First-pass, unvalidated -
# revisit once the backtest script has real MAE/correlation numbers across
# alternative weightings.
DFS_PITCHER_WINDOWS = [
    (None, 0.30),
    (60, 0.25),
    (30, 0.25),
    (15, 0.20),
]

# Bullpen fatigue/readiness (2026-08-24 feature-search follow-up):
# pitchers.compute_bullpen_recent_workload's lookback window, in real
# calendar days strictly before the target date. A single fixed recency
# CUTOFF, not a config.PAVE_WINDOWS-style multi-window blend - this is a
# recent WORKLOAD total (a fatigue proxy), not a rate that needs
# small-sample smoothing across windows the way a hit-rate signal does.
# 2 days is a first-pass, unvalidated choice (real bullpen usage/rest
# patterns commonly discussed in baseball are 1-3 days) - revisit once a
# real significance/backtest result exists.
BULLPEN_FATIGUE_RECENT_DAYS = 2

# Real dispatched result (2026-08-25, GitHub Actions run 32792241148,
# n=1,963 games): compute_bullpen_recent_workload at the 2-day window
# above showed no significant signal (home p=0.4727, away p=0.8985
# univariate). Explicit follow-up - "I want to see if other applications
# of bullpen fatigue are significant... I don't care if they're cheap" -
# sweeps additional window lengths as separate candidates (see
# scripts/train_game_pick_model.py's CANDIDATE_FEATURE_COLUMNS) rather
# than assuming the 2-day cutoff was the right one.
BULLPEN_FATIGUE_CANDIDATE_WINDOWS = [1, 3, 5]

# A pitcher needs at least this many recorded starts (pitcher_form's
# unweighted full-season `starts` count) to be ranked at all - a 1-2 start
# sample is enough for HR9 especially to be pure small-sample noise (one
# blowup start can swing it to an absurd extreme). Modest, not a
# qualified-ERA-title bar - a rookie's second MLB start should still
# surface, same reasoning as AGE_CURVE_MIN_IP being deliberately modest.
DFS_PITCHER_MIN_STARTS = 3

# Sabermetric rule of thumb translating Expected_IP into an estimated
# batters-faced count, used only to scale PAVE (an AB-level rate) into
# Expected_H_Allowed = PAVE * Expected_IP * this constant, since neither
# PAVE nor pitcher_form.py carries a real per-appearance batters-faced
# count. Computed from this project's OWN persisted 2026 Statcast (not a
# guessed league average): 113,334 real completed at-bat events over
# 78,432 real outs recorded (26,144 IP) = 4.335 batters faced per inning.
DFS_BATTERS_FACED_PER_INNING = 4.335

# Empirically validated (scripts/backtest_dfs_rankings.py) against 15 real
# game dates in July 2026, recomputed fresh from persisted Statcast with no
# lookahead (the same discipline as game_picks_backtest.py/
# backtest_age_curve.py) - see dfs_backtest.py's module docstring for
# exactly what "actual" means and its honesty limits.
#
# Hitters (DK_Points_Hitter, hit-type scoring only - HISTORICAL/
# PRE-EXPANSION, kept for the record, not current): MAE 3.7719 vs. 3.7496
# naive-baseline MAE, correlation -0.004 (n=4,156 scored). This was
# reported honestly as NOT a working signal for single-game DFS
# hit-scoring - the projection was indistinguishable from noise at the
# single-game level. The single highest-risk design choice here (see
# dfs.py's module docstring) is compute_matchup_adjustment's ratio,
# derived from hit-PROBABILITY signals but applied to a TOTAL-BASES
# signal - this backtest was exactly the check that heuristic needed, and
# it did not hold up.
#
# Hitters (DK_Points_Hitter, CURRENT - hit-type + BB/HBP/RBI, see dfs.py's
# module docstring for the widened scope): MAE 4.7569 vs. 4.7457
# naive-baseline MAE, correlation 0.009 (n=5,430 scored, 20 most recent
# game dates as of 2026-07-26). Widening the scored categories did NOT fix
# the underlying weak-signal problem - correlation is still essentially
# zero. This is the expected result, not a surprise: the flawed
# multiplicative ratio flagged above still drives DK_Points_Hitter_HitType
# (most of the point total), and the new BB/HBP/RBI terms are additive on
# top of it, not a structural fix. Do not treat the heuristic
# DK_Points_Hitter as validated. The live default is the ML model
# (dfs_ml.apply_ml_overrides, see the "Machine Learning" section below),
# which was specifically designed to bypass this ratio - re-run
# scripts/train_dfs_ml_models.py after this widening (see that section's
# retraining note) rather than relying on the heuristic here.
#
# Pitchers: a real, if modest, positive signal - better than hitters, but
# still weak on some components:
#   Expected_IP:         MAE 1.0257 vs. 1.0332 baseline, correlation 0.351 (n=360)
#   Expected_K:           MAE 1.8971 vs. 1.9922 baseline, correlation 0.393 (n=360)
#   Expected_BB:           MAE 1.0570 vs. 1.0380 baseline, correlation 0.182 (n=360) - weak
#   Expected_H_Allowed:     MAE 2.6545 vs. 1.7860 baseline, correlation 0.223 (n=360) - MAE
#     actually WORSE than the naive baseline despite positive correlation,
#     suggesting DFS_BATTERS_FACED_PER_INNING/PAVE scaling is systematically
#     off, not just noisy - a second flagged follow-up.
#   DK_Points_Pitcher (combined): MAE 6.9409 vs. 7.1970 baseline,
#     correlation 0.306 (n=360) - IP/K carry the real signal here.
# Re-run this backtest (and update these numbers) after any change to
# DFS_PITCHER_WINDOWS/DFS_BATTERS_FACED_PER_INNING/DFS_MATCHUP_RATIO_CLIP.

# --- Machine Learning (weak-signal follow-up: ml_models.py, dfs_ml.py, age_curve_ml.py) ---
#
# The heuristic backtest numbers directly above (DK_Points_Hitter corr
# -0.004; Expected_H_Allowed MAE worse than baseline; Expected_BB corr
# 0.182) and Age Curves' HR9 "wash" (see below) motivated a real attempt at
# walk-forward-validated ML models (scripts/train_dfs_ml_models.py,
# scripts/train_age_curve_hr9_model.py) rather than tuning the existing
# heuristics further. See ml_models.py's module docstring for the shared
# walk-forward CV mechanism and dfs_ml.py/age_curve_ml.py for what each
# model is trained on.

# WalkForwardDateSplit parameters for the two DFS pitcher-side models
# (Expected_H_Allowed, Expected_BB) - larger blocks than the hitter side
# since pitcher-start rows are far sparser per date (~2 starters/team,
# ~13-26 rows/date vs. thousands of hitter-day rows).
ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER = 30
ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER = 10
ML_WALK_FORWARD_MIN_TRAIN_DATES_PITCHER = 40
ML_WALK_FORWARD_TEST_BLOCK_DATES_PITCHER = 15

# Dates reserved as a final holdout the grid search never sees at all -
# model selection (grid search + CV) only ever uses earlier dates; the
# before/after numbers reported in this file's comments and in the README
# come exclusively from predicting this untouched block, refit on
# everything before it. Matches the size of the original 15-20 date
# heuristic-only backtest sample, so it's a fair head-to-head comparison.
#
# Prefer nested rolling-origin validation (model_validation.py) for new
# research: outer/inner folds replace repeatedly inspecting this same
# block for family/calibration/features/policy. When a freeze period is
# reserved via NESTED_VALIDATION_FREEZE_DATES, that tail is also untouched
# and must not be repeatedly inspected during development.
ML_FINAL_HOLDOUT_DATES = 20

# Nested rolling-origin validation (mlb_metrics.model_validation). Outer
# folds grow training forward through the season; inner folds select
# model family / hyperparameters / calibration / features / threshold /
# shortlist / one-vs-two-pick policy. Outer test is scored once after
# inner selection. Defaults reuse the hitter walk-forward block sizes.
NESTED_VALIDATION_OUTER_MIN_TRAIN_DATES = ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER
NESTED_VALIDATION_OUTER_TEST_BLOCK_DATES = ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER
NESTED_VALIDATION_INNER_MIN_TRAIN_DATES = 20
NESTED_VALIDATION_INNER_TEST_BLOCK_DATES = 5
# Optional untouched freeze tail AFTER all nested fold construction.
# Default 0 (no freeze). When > 0, those most-recent dates are excluded
# from every train/test fold and from every selection decision - do not
# peek at them while iterating on models or policy.
NESTED_VALIDATION_FREEZE_DATES = 0
NESTED_VALIDATION_REPORT_DIR = "reports/model_validation"
NESTED_VALIDATION_BOOTSTRAP_SAMPLES = 1000
NESTED_VALIDATION_RANDOM_SEED = 0

DFS_HITTER_MODEL_PATH = "data/models/dfs_hitter_model.joblib"
DFS_PITCHER_H_ALLOWED_MODEL_PATH = "data/models/dfs_pitcher_h_allowed_model.joblib"
DFS_PITCHER_BB_MODEL_PATH = "data/models/dfs_pitcher_bb_model.joblib"

# Complete DraftKings-style target construction (mlb_metrics.dfs_complete_targets).
# Evaluates against full official/reconstructed game outcomes, not the
# partial Actual_DK_Points_Modeled subset. Legacy partial-target models
# remain the live/heuristic benchmark until the promotion gate passes.
DFS_COMPLETE_TARGET_MODE = "shadow"  # legacy | shadow | live
DFS_COMPLETE_TARGET_MODES = ("legacy", "shadow", "live")
DFS_COMPLETE_HITTER_MODEL_PATH = "data/models/dfs_complete_hitter_model.joblib"
DFS_COMPLETE_PITCHER_MODEL_PATH = "data/models/dfs_complete_pitcher_model.joblib"
DFS_COMPLETE_SHADOW_PREDICTIONS_PATH = (
    "data/predictions/dfs_complete_shadow_predictions.csv"
)
DFS_COMPLETE_PROMOTION_GATE_REPORT_PATH = (
    "reports/model_validation/dfs_complete_targets_nested.json"
)
DFS_COMPLETE_VALIDATION_REPORT_NAME = "dfs_complete_targets_nested.json"
# Promotion requires meaningful lineup-level / ranking improvement - not a
# trivial player-MAE tick alone.
DFS_COMPLETE_MIN_LINEUP_SCORE_IMPROVEMENT = 0.5  # mean abs lineup error reduction
DFS_COMPLETE_MIN_RANK_CORR_IMPROVEMENT = 0.02
DFS_COMPLETE_MIN_TOP_DECILE_RECALL_IMPROVEMENT = 0.02
DFS_COMPLETE_TRIVIAL_MAE_EPSILON = 0.05  # MAE-only wins below this do not promote
DFS_COMPLETE_OUTER_MIN_TRAIN_DATES = 30
DFS_COMPLETE_OUTER_TEST_BLOCK_DATES = 10
DFS_COMPLETE_INNER_MIN_TRAIN_DATES = 15
DFS_COMPLETE_INNER_TEST_BLOCK_DATES = 5
DFS_COMPLETE_RIDGE_ALPHA_GRID = [0.3, 1, 3, 10, 30]
DFS_COMPLETE_GBM_PARAM_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200],
    "min_samples_leaf": [50, 200],
}
AGE_CURVE_HR9_MODEL_PATH = "data/models/age_curve_hr9_model.joblib"
HITTER_HIT_PROBABILITY_MODEL_PATH = "data/models/hitter_hit_probability_model.joblib"

# Shadow opportunity-aware hitter probability model
# (mlb_metrics.hitter_probability_model / scripts/train_hitter_probability_model.py).
# Explicitly models P(Appear), E[PA|Appear], and P(Got_Hit|Appear).
# Wired into the pipeline only via HITTER_SELECTION_MODE (shadow logs
# diagnostics; live uses Final_Hit_Probability as the sole authoritative
# score). Never silently replaces legacy selection without the promotion gate.
HITTER_OPPORTUNITY_PROBABILITY_MODEL_PATH = (
    "data/models/hitter_opportunity_probability_model.joblib"
)
HITTER_OPPORTUNITY_SHADOW_PREDICTIONS_PATH = (
    "data/predictions/hitter_opportunity_shadow_predictions.csv"
)
HITTER_OPPORTUNITY_VALIDATION_REPORT_NAME = "hitter_opportunity_probability_nested.json"
HITTER_OPPORTUNITY_LOGIT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
HITTER_OPPORTUNITY_GBM_PARAM_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.1],
    "max_iter": [100, 200],
    "min_samples_leaf": [50, 200],
}

# Deliberately narrow grids (1-2 hyperparameters, <10 combinations) given
# the modest walk-forward fold count on the DFS side (~116 dates -> ~6-8
# blocked CV folds) - a wide sweep over that few folds would just fit CV
# noise, not find a real best setting. Ridge is the primary candidate for
# both pitcher signals (small distinct-pitcher pool, ~755 across full
# history); a heavily depth-capped gradient-boosting grid is included as a
# secondary candidate only for the hitter signal, where the distinct-batter
# pool (~604) and row count are both larger.
DFS_HITTER_RIDGE_ALPHA_GRID = [0.1, 1, 3, 10, 30, 100]
DFS_HITTER_GBM_PARAM_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.1],
    "max_iter": [100, 200],
    "min_samples_leaf": [50, 200],
}
DFS_PITCHER_RIDGE_ALPHA_GRID = [0.1, 0.3, 1, 3, 10, 30, 100]

# LogisticRegression's C (inverse regularization strength) grid for the
# hitter hit-probability model (scripts/train_hitter_hit_model.py) - same
# narrow-grid caution as above, same walk-forward date-grain as the DFS
# hitter model (ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER/_TEST_BLOCK_DATES_HITTER
# above, reused rather than duplicated).
HITTER_HIT_LOGIT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

# HistGradientBoostingClassifier's second-candidate grid for the same
# hitter hit-probability model - literally DFS_HITTER_GBM_PARAM_GRID's
# values (same order-of-magnitude row/fold count on this signal), not a
# freshly invented grid. train_hitter_hit_model.py runs this alongside
# LogisticRegression above and keeps whichever wins by walk-forward CV
# score, mirroring train_dfs_ml_models.py's own Ridge-vs-GBM selection for
# DK_Points_Hitter (quant-analytics item #2 - "model family").
HITTER_HIT_GBM_PARAM_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.1],
    "max_iter": [100, 200],
    "min_samples_leaf": [50, 200],
}

# Game-pick win-probability model (scripts/train_game_pick_model.py) - fit +
# report phase only, mirrors the hitter-side model above but not wired into
# live picks. One row per game per date (~15 games/day in a full slate, not
# thousands of hitter-rows/date), so both the train/test block sizing below
# is smaller than the hitter side's - closer to the DFS pitcher-side
# constants than the hitter ones. ML_FINAL_HOLDOUT_DATES above is reused
# directly for the holdout split, no separate constant.
GAME_PICK_WIN_PROBABILITY_MODEL_PATH = "data/models/game_pick_win_probability_model.joblib"
GAME_PICK_LOGIT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_DATES = 40
GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_DATES = 15

# Quant-analytics follow-up ("dig into calibration", 2026-08-24):
# scripts/train_game_pick_calibration.py's saved recalibration of
# game_picks.compute_game_win_probabilities' raw home_win_probability
# ratio - a DIFFERENT artifact from GAME_PICK_WIN_PROBABILITY_MODEL_PATH
# above (that one is a from-scratch replacement model that already failed
# to beat the heuristic; this one instead RESCALES the heuristic's own
# output against real outcomes, isotonic or Platt/sigmoid - see
# ml_models.fit_probability_calibration). Reuses the same walk-forward
# constants and ML_FINAL_HOLDOUT_DATES above for direct comparability
# with that earlier attempt, not new ones.
GAME_PICK_CALIBRATION_MODEL_PATH = "data/models/game_pick_calibration_model.joblib"

# Market-residual game-win challenger (mlb_metrics.game_residual_model):
# logit(P_final) = logit(P_market_at_prediction_time) + residual(features).
# Modes mirror hitter selection: legacy keeps the calibrated heuristic;
# shadow logs residual probs without changing official picks; live requires
# the promotion gate. Default shadow - do not flip to live without the gate.
GAME_PREDICTION_MODES = ("legacy", "shadow", "live")
GAME_PREDICTION_MODE = "shadow"
GAME_RESIDUAL_MODEL_VERSION = "v1-market-residual"
GAME_RESIDUAL_MODEL_VERSION_LIVE = "v2-market-residual-live"
GAME_RESIDUAL_MODEL_PATH = "data/models/game_residual_win_probability_model.joblib"
GAME_RESIDUAL_SHADOW_PREDICTIONS_PATH = (
    "data/predictions/game_residual_shadow_predictions.csv"
)
GAME_RESIDUAL_SHADOW_BETS_PATH = "data/predictions/game_residual_shadow_bets.csv"
GAME_RESIDUAL_PROMOTION_GATE_REPORT_PATH = (
    "reports/model_validation/game_residual_nested.json"
)
GAME_RESIDUAL_VALIDATION_REPORT_NAME = "game_residual_nested.json"
# Strong regularization toward a near-zero residual (small C / large alpha).
GAME_RESIDUAL_LOGISTIC_C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]
GAME_RESIDUAL_LOGIT_RESIDUAL_CAP = 0.75  # clip residual logits (~+/-17pp near 0.5)
GAME_RESIDUAL_NONLINEAR_MAX_DEPTH_GRID = [2, 3]
GAME_RESIDUAL_NONLINEAR_MIN_SAMPLES_LEAF_GRID = [40, 80]
GAME_RESIDUAL_NONLINEAR_SHRINK_GRID = [0.25, 0.5, 0.75]
GAME_RESIDUAL_EDGE_THRESHOLD_GRID = [0.01, 0.02, 0.03, 0.04, 0.05]
# Betting advice is a separate promotion surface from probability accuracy.
# disabled: no stakes written to the official game-predictions log
# shadow: hypothetical bets only (shadow CSV); official bet_units stay 0
# live: requires the betting promotion gate (never the default)
BETTING_MODES = ("disabled", "shadow", "live")
BETTING_MODE = "disabled"
BETTING_PROMOTION_GATE_REPORT_PATH = (
    "reports/model_validation/game_residual_betting_gate.json"
)
BETTING_PROMOTION_MIN_OUTER_FOLDS = 3
BETTING_PROMOTION_MIN_GAMES = 100
BETTING_PROMOTION_MIN_DATE_BLOCKS = 10
BETTING_PROMOTION_MAX_SINGLE_WEEK_PROFIT_SHARE = 0.45
BETTING_PROMOTION_MATERIAL_NEGATIVE_ROI = -0.02
# Nested CV sizing for the residual challenger (game-level rows are sparse).
GAME_RESIDUAL_OUTER_MIN_TRAIN_DATES = 30
GAME_RESIDUAL_OUTER_TEST_BLOCK_DATES = 10
GAME_RESIDUAL_INNER_MIN_TRAIN_DATES = 15
GAME_RESIDUAL_INNER_TEST_BLOCK_DATES = 5

# Market benchmark (ESPN odds, mlb_metrics.market_odds) - quant-analytics
# item #6, slice 2. "DraftKings" is the only provider seen in every real
# pickcenter row slice 1's confirmation dispatch found (GitHub Actions run
# 32516808493, 2026-08-21); market_odds._parse_pickcenter_row falls back
# to the first available row rather than dropping a game outright if this
# is ever absent for a real game. MARKET_ODDS_BACKFILL_DAYS_BACK bounds
# scripts/backfill_market_odds.py's default lookback to the real depth
# that same dispatch actually confirmed ESPN still serves odds for
# (checked 2026-08-16, 5 days before the dispatch) - depth beyond that is
# unconfirmed, so it isn't the default, though a caller may still pass a
# larger --days-back and let dates beyond what ESPN retains simply come
# back empty rather than being blocked outright.
MARKET_ODDS_PREFERRED_PROVIDER = "DraftKings"
MARKET_ODDS_BACKFILL_DAYS_BACK = 5
# Append-only ESPN/odds snapshot log. Closing line = latest valid snapshot
# strictly before game start - never the morning-only capture.
MARKET_ODDS_SNAPSHOTS_PATH = "data/predictions/market_odds_snapshots.csv"
MARKET_ODDS_EVENT_MAP_PATH = "data/predictions/market_odds_event_map.csv"
MARKET_ODDS_MATCH_TIME_TOLERANCE_MINUTES = 45
MARKET_ODDS_BOOTSTRAP_SAMPLES = 1000
MARKET_ODDS_BOOTSTRAP_SEED = 0

# Kelly-criterion bet sizing (kelly.py, scripts/recommend_bets.py) - a
# follow-up to the market benchmark above: turns "model probability
# disagrees with the market" into an actual recommended stake. Single
# straight bets only, sized against the REAL vigged price (see
# recommend_bets.py's own comment on why this differs from
# market_home_win_probability above).
KELLY_FRACTION_MULTIPLIER = 0.5  # half-Kelly - full Kelly is only growth-optimal if the probability estimate is exactly right; this project's own probabilities carry real estimation error (see the Wilson CIs throughout this project), and full Kelly's downside variance is severe when the estimate is even slightly off. Half-Kelly is the standard practitioner default.
# ALWAYS applied, layered on top of whichever probability sizes the bet
# (2026-08-26, direct follow-up to the "units risked" fix above - "the
# short-priced-favorite blowup risk"): game_predictions.advise_bets sizes
# off a real, per-bet pessimistic probability (game_picks.apply_kelly_uncertainty,
# grounded in each team's own real season-to-date Wilson CI - see
# teams.compute_team_win_rate_ci) whenever that data is available, but this
# multiplier is NOT replaced by 1.0 in that case anymore - a team's season-
# long win-rate CI can stay genuinely tight (a real, well-established
# record) even though THIS SPECIFIC GAME still carries real matchup-level
# uncertainty the season-level CI can't see, so a confident raw model
# probability run through the CI-based pessimism can still imply a real
# double-digit-percent-of-bankroll full-Kelly stake on a heavy favorite
# (see the README's own worked Dodgers/Rockies example). The CI-based
# pessimism and this flat multiplier are two independent, stacked layers
# of conservatism - one doesn't substitute for the other.
KELLY_MIN_EDGE = 0.05  # minimum (model probability - real vigged market-implied probability) before recommending any stake at all - a buffer against noise in the model's own probability estimate, not a number backed by a formal calculation.
KELLY_DAILY_UNIT_CAP = 5  # hard portfolio-level cap, in units (see UNIT_SIZE_FRACTION below), on the TOTAL stake advised across all of a single day's bets combined - a deliberate user-set risk limit (2026-08-25), not derived from data. game_predictions.advise_bets scales ALL of a date's stakes down proportionally (never selectively drops any one bet) whenever their sum would otherwise exceed this, so no single day's combined advice can ever risk more than this many units regardless of how many real edges clear on that date.
KELLY_MAX_SINGLE_BET_UNIT_CAP = 2  # hard cap, in units, on any ONE game's advised stake regardless of what Kelly computes - a deliberate user-set risk limit (2026-08-26, "the short-priced-favorite blowup risk" follow-up), not derived from data. Directly bounds Kelly's own amplification of a thin probability margin into a large stake for short-priced favorites (stake sensitivity to the sizing probability is proportional to 1 + 1/b, which is large when the net odds b are small) - a risk the CI-based pessimistic probability and kelly_fraction_multiplier can reduce but not fully eliminate on their own. Applied in game_predictions.advise_bets BEFORE the daily cap (which then still applies across whatever survives).
# Real follow-up (2026-08-26, direct response to "maybe we scaled it down
# too much... this is now taking out games where the line is genuinely
# appealing (some underdogs who have a chance)"): real production data
# (2026-08-26) showed EVERY team's real season-to-date Wilson win-rate CI
# (95% - the statsmodels/helpers.wilson_ci default) sitting at roughly an
# 8-point half-width even at 132-133 games played (nearly a full season) -
# not a fluke, just the real sqrt(n) floor of a ~130-game binomial
# proportion's precision. Combined (root-sum-square) across two teams,
# that is consistently ~11-12 points of haircut on every single game, all
# season, regardless of how much real data accumulates - larger than most
# real edges this project has ever advised on (see KELLY_MIN_EDGE's own
# history above), so it was quietly zeroing out real, honest edges,
# concretely: a real 2026-08-26 game (NYM +143, model 51.9% vs. market-
# implied 41.2%, a genuine 10.7-point edge) got fully erased by an 11.7-
# point haircut. See KELLY_MAX_SINGLE_BET_UNIT_CAP above for the other
# half of this fix - the per-bet cap is what actually bounds the specific
# short-priced-favorite risk that motivated widening this CI in the first
# place, so this constant no longer has to do that job alone.
KELLY_UNCERTAINTY_CI_ALPHA = 0.32  # ~68% CI (roughly 1 standard error) instead of the usual 95% (alpha=0.05) - a real, standard, NAMED statistical convention (not a hand-picked number), used ONLY for teams.compute_team_win_rate_ci's bet-sizing input (helpers.wilson_ci's own default alpha=0.05 is untouched everywhere else - every other real CI this project reports, e.g. win_rate_on_advised_bets_ci_low/high, beat_closing_line_rate_ci_low/high, stays a genuine 95% CI for honest reporting).
# Raised from 0.02 after real production data (2026-08-24) showed 8 of 10
# real games clearing the old 2% bar, with de-vigged model/market gaps as
# large as 12 percentage points - implausible as genuine value against a
# real, liquid MLB moneyline market. The likely cause: this project's own
# model probabilities cluster much closer to 50/50 than real sportsbook
# lines do (the model is comparatively low-spread/conservative), so on
# any game the market is confident about, the model's comparatively muted
# probability for the live underdog looks like "value" that isn't real -
# it's the model under-informing itself relative to the market, not a
# market inefficiency. This project's own beat_closing_line_rate (0.357,
# n=14) is not yet statistically distinguishable from a coin flip
# (evaluation.binomial_significance p=0.42 as of 2026-08-24) - there is
# currently no proven evidence this model forecasts games better than the
# market at all. 0.05 is a stopgap that filters out most of that noise,
# NOT a fix for the underlying calibration/skill gap - see the
# "Real quant sanity-check" README section for the fuller writeup and the
# real follow-up options (raise this further, and/or gate bet-advice on
# beat_closing_line_rate actually clearing statistical significance
# before advising anything at all).
KELLY_MIN_GAMES_FOR_CONFIDENCE = 100  # a conservative, round floor for scripts/recommend_bets.py's printed confidence banner - NOT a formal power-calculation result. Below this many real n_beat_closing_line_compared games (see game_evaluation.py), the script still shows the real computed edge/stake numbers but prints an explicit "not yet statistically validated" warning rather than hiding them - real numbers, honestly labeled, not a silent gate.
UNIT_SIZE_FRACTION = 0.01  # what "1 unit" means, as a fraction of bankroll - the standard sports-betting convention (bettors report/track performance in bankroll-agnostic "units risked/won" rather than dollars, since bankroll size varies per person and shouldn't be required to compare or log a strategy's real results). 1% is a common real convention, not a formally derived number. game_predictions.csv logs bet_units = kelly_stake_fraction / UNIT_SIZE_FRACTION - a real bettor still converts units to their own real dollar stake at bet time via scripts/recommend_bets.py's own optional --bankroll flag, which this constant doesn't touch.

# Age Curves HR9's year-blocked CV (age_curve_ml.YearBlockedSplit) trains
# only on seasons strictly before AGE_CURVE_HR9_TEST_YEAR_START (a
# conservative, no-lookahead choice for a single GLOBAL regression model -
# stricter than age_curve.backtest_projection_accuracy's own per-row "at or
# before this test season's year" cutoff, which lets a later test row see
# comparables from just-earlier years in the SAME held-out decade; a single
# regression trained once can't cheaply replicate that per-row refiltering,
# so this holds out the whole 2010-2019 decade rather than only each row's
# own future). That decade of pre-2010 training data spans 1871-2009 (~139
# years), which comfortably supports more CV folds than the DFS side's
# ~116 dates - the wider grid below is a real fold-count difference, not a
# contradiction of the narrow-grid caution above.
AGE_CURVE_HR9_TEST_YEAR_START = 2010
AGE_CURVE_HR9_TEST_YEAR_END = 2019
AGE_CURVE_HR9_TEST_SAMPLE_SIZE = 500
AGE_CURVE_HR9_TEST_SEED = 0
ML_WALK_FORWARD_MIN_TRAIN_YEARS = 40
ML_WALK_FORWARD_TEST_BLOCK_YEARS = 10

AGE_CURVE_HR9_RIDGE_ALPHA_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
AGE_CURVE_HR9_GBM_PARAM_GRID = {
    "max_depth": [3, 5, None],
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200, 300],
}

# Results from scripts/train_dfs_ml_models.py, run 2026-07-26 against the
# FULL real persisted 2026 Statcast history (118 hitter dates / 103
# pitcher dates - retrained after fixing a real PAVE bug: the formula
# excluded strikeouts from the at-bat denominator alongside walks/HBP,
# inflating hit-rate-against for exactly the pitchers who strike the most
# batters out (see pitchers.py's module docstring and the README's "Real
# bug fixed: PAVE excluded strikeouts" section for the full story and the
# heuristic-level before/after numbers). starter_PAVE/Bullpen_PAVE
# (hitter features) and Expected_H_Allowed (a pitcher feature, itself
# PAVE-derived) all shifted, so all three models needed retraining even
# though none of their own feature SCHEMA changed - old model artifacts
# were kept, not deleted, since the schema is unchanged, but the weights
# are new). Nested walk-forward grid search on the earlier dates only,
# evaluated ONCE on the untouched final ML_FINAL_HOLDOUT_DATES-date block.
# All three signals cleared the bar (beat both the naive baseline AND the
# existing heuristic) and are now LIVE (see dfs_ml.apply_ml_overrides) -
# reported honestly, same as every other backtest here, and this WOULD
# have said so if a model hadn't cleared the bar (see git history / README
# for that framing):
#
#   DK_Points_Hitter (HistGradientBoostingRegressor, max_depth=2,
#   learning_rate=0.03, max_iter=100, min_samples_leaf=200):
#     MAE 4.6299 vs. naive-baseline MAE 4.6648 vs. heuristic MAE 4.7156,
#     correlation 0.161 (n=5,760) - essentially unchanged from the
#     pre-PAVE-fix retrain (0.162), as expected: this signal's heuristic
#     correlation was already near-zero for unrelated reasons (the
#     compute_matchup_adjustment ratio, not PAVE directly - see "Hitters"
#     below), so a PAVE fix alone wasn't expected to move it much.
#
#   Expected_H_Allowed (Ridge, alpha=30):
#     MAE 1.7685 vs. naive-baseline MAE 1.8395 vs. heuristic MAE 1.8032,
#     correlation 0.297 (n=481) - the PAVE fix alone already pulled the
#     heuristic from worse-than-baseline (2.6252 pre-fix, same dates) to
#     better-than-baseline (1.8032); this retrained model improves on
#     that fixed heuristic further still (1.7685), a genuine additional
#     gain on top of the formula fix, not just recovering ground the bug
#     had cost.
#
#   Expected_BB (Ridge, alpha=0.1):
#     MAE 1.0141 vs. naive-baseline MAE 1.0310 vs. heuristic MAE 1.0689,
#     correlation 0.130 (n=481) - Expected_BB doesn't consume PAVE
#     directly, so this retrain's small movement vs. the prior run
#     reflects more persisted history accumulating, not the PAVE fix
#     itself.
#
# Re-run scripts/train_dfs_ml_models.py (and update this comment) after
# any change to the DFS feature set, PAVE_WINDOWS, or DFS_PITCHER_WINDOWS -
# a stale saved model silently keeps serving old-feature-distribution
# predictions otherwise.

# Age Curves HR9 result, from scripts/train_age_curve_hr9_model.py (same
# 500-season, 2010-2019, seed-0 sample scripts/backtest_age_curve.py
# already used for the KNN number above - recomputed fresh, not assumed
# stale): HistGradientBoostingRegressor (max_depth=3, learning_rate=0.05,
# max_iter=100), trained ONLY on seasons before AGE_CURVE_HR9_TEST_YEAR_START
# (1871-2009, 16,324 rows) - MAE 0.3190 vs. naive-baseline MAE 0.3264 vs.
# KNN heuristic MAE 0.3312, correlation 0.359 vs. KNN's 0.321 (n=296/500).
# HR9 is genuinely the hardest of the four signals (single-season HR9 is
# substantially batted-ball-luck/park/defense-driven, not just a modeling
# gap - see age_curve_ml.py's module docstring) but a real multi-dimensional
# regression DID beat both the naive baseline and the single-dimension KNN
# search here, contrary to the honestly-stated-up-front possibility that it
# might not. Now LIVE for HR9 only (see build_age_curves.py's
# build_projections_for_group) - every other Age Curves metric (AVG/OBP/
# SLG/OPS/K9/BB9/FIP) is untouched and still served by the original KNN
# path, which was already beating its own baseline.

# --- Ceiling/volatility signal (dfs_ceiling.py) ---
#
# GPP (tournament) DFS lineups are won by boom/spike-game players, not
# players who reliably score near their own mean - a real, well-known DFS
# strategy concept (the user's own framing: "the winner won't have
# players getting 5 points across their lineup, they're more likely to
# have lucked into players averaging 15 or so"). Ceiling_DK_Points is the
# Pth percentile of a player's own REAL historical modeled DK points
# (dfs_backtest.compute_actual_hitter_dk_points/
# compute_actual_pitcher_dk_points applied per real game date they
# played) - an ADDITIONAL informational column alongside the existing
# mean projection (DK_Points_Hitter/DK_Points_Pitcher), never replacing
# it. See dfs_ceiling.py's module docstring for why this ships
# informational-only rather than as the optimizer's default objective.
DFS_CEILING_PERCENTILE = 90

# A player with fewer than this many real scored games has too little
# history for a meaningful per-player percentile (a rookie's 3rd game
# could literally BE their whole "ceiling" sample) - falls back to the
# GROUP-WIDE (all hitters', or all pitchers', pooled) percentile at the
# same level instead, the same small-sample philosophy
# MATCHUP_LEAGUE_PAVE_FALLBACK already uses. Modest, not a
# qualified-season bar - matches DFS_PITCHER_MIN_STARTS/AGE_CURVE_MIN_IP's
# reasoning that a small-but-real sample should still surface rather than
# being silently excluded.
DFS_CEILING_MIN_GAMES = 10

# Backtested (dfs_ceiling.backtest_ceiling_signal) against the same 20
# real game dates as the heuristic DFS backtest above, no-lookahead
# (Ceiling_DK_Points computed from ONLY history strictly before each test
# date). Two questions per player type: does a higher ceiling predict a
# better real day at all (correlation), and - the more DFS-relevant
# question - of the player-days that ACTUALLY landed in that date's real
# top decile, what fraction were ALSO top-decile by Ceiling_DK_Points
# going in, vs. by the existing mean projection instead ("capture rate"):
#
#   Hitters (n=5,430): correlation 0.127. Of 543 real top-decile hitter-
#   days, ceiling-ranking flagged 19.3% of them in advance vs. the mean
#   projection's 10.1% - the mean projection is barely better than the
#   ~10% base rate (i.e. DK_Points_Hitter has almost no power to predict
#   WHICH day a hitter booms), while ranking by real historical ceiling
#   is genuinely ~2x better than that base rate. A real, honest signal for
#   hitters, not overstated - still misses 4 of 5 real boom days, but a
#   meaningfully better-than-mean-projection way to look for them.
#
#   Pitchers (n=468): correlation 0.313 (numerically higher than hitters'),
#   but on capture rate the mean projection actually did SLIGHTLY BETTER
#   (27.7% vs. ceiling's 23.4% of 47 real top-decile pitcher-days) - no
#   clear edge for ceiling over the existing mean projection on the
#   pitcher side. Small sample (n=47 boom-days) makes this noisy, but
#   reported honestly as a non-result rather than rounded into a "works
#   great" story just because hitters showed a real signal.
#
# Conclusion: Ceiling_DK_Points is a genuinely validated upside signal for
# HITTERS specifically, not for pitchers. Ships informational-only either
# way (see dfs_ceiling.py's module docstring) - the optimizer's
# `--objective ceiling` flag is opt-in, not the default, and this mixed
# result is exactly why: a user choosing ceiling for a hitter-heavy GPP
# strategy has real backing; doing the same for pitcher selection does not
# yet. Re-run this backtest (and update this comment) after any change to
# DK_Points_Hitter/DK_Points_Pitcher's own formula.

# --- Boom_Adjusted_DK_Points (dfs_ceiling.compute_boom_adjusted_score) ---
#
# Neither pure mean nor pure ceiling: the user explicitly wants a player
# with real, frequent upside swings (sometimes 20, sometimes 5, sometimes
# 0, averaging 4.8) ranked ABOVE a metronomic player averaging a flat 5 -
# credit for genuine volatility, not a max-chasing boom-or-bust score and
# not blind to the mean either. Boom_Adjusted_DK_Points =
# DK_Points_Hitter/Pitcher (today's mean projection) + k * Upside_Deviation
# (a player's own real historical UPSIDE-ONLY semi-deviation - see
# dfs_ceiling.compute_upside_deviation's docstring for exactly why it's a
# semi-deviation, not plain stdev).
#
# The grid dfs_ceiling.backtest_boom_adjusted_signal searches to pick k
# from real data rather than guessing one.
DFS_BOOM_ADJUSTED_K_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Backtested (dfs_ceiling.backtest_boom_adjusted_signal) against the same
# 20-date sample as Ceiling_DK_Points's own backtest, same no-lookahead
# discipline, same correlation/capture-rate metrics, once per k in the
# grid above:
#
#   Hitters (n=5,430, 543 real top-decile days): correlation and capture
#   rate both rose MONOTONICALLY across the entire tested grid, never
#   turning over - k=0.0 (plain mean): correlation 0.009, capture 10.1%;
#   k=1.0: correlation 0.044, capture 12.5%; k=2.0 (grid max): correlation
#   0.061, capture 13.3%. The grid didn't localize a peak, but the
#   grid-max k=2.0 was deliberately NOT chosen: at k=2.0 the volatility
#   term (mean real Upside_Deviation 4.417) OUTWEIGHS the mean term (mean
#   real DK_Points_Hitter 5.185) by nearly 2:1, which just recreates pure
#   ceiling-chasing under a different name - directly contrary to the
#   user's explicit "neither pure boom nor pure mean" request. k=1.0 keeps
#   the two terms roughly balanced (mean contributes ~54% of a typical
#   score, deviation ~46%) while still capturing a real, validated
#   improvement over the plain mean (capture rate +24% relative, from
#   10.1% to 12.5%). Chosen for that balance, not because it's the
#   highest-scoring k in the grid.
#
#   Pitchers (n=468, 47 real top-decile days): correlation stayed flat
#   (0.328-0.340) and capture rate bounced non-monotonically (25.5%-29.8%)
#   across the whole grid, with NO k reliably beating k=0.0's baseline
#   27.7% - the same conclusion Ceiling_DK_Points's own backtest reached
#   for pitchers (no clear upside-signal edge). k=0.0 (falls back to the
#   plain mean projection) is the honest choice, not a guessed nonzero
#   value chasing sample noise.
#
# Re-run this backtest (and update this comment) after any change to
# DK_Points_Hitter/DK_Points_Pitcher's own formula.
DFS_BOOM_ADJUSTED_K_HITTER = 1.0
DFS_BOOM_ADJUSTED_K_PITCHER = 0.0

# --- Matchup_Boom_Score (dfs_ceiling.compute_matchup_boom_score, hitters only) ---
#
# Real evidence from an actual DK contest (winner spent $17,500 on
# pitchers, this project's own suggestion over $21,000, and even after
# the salary parity fix above a real slate's mean-projection lineup only
# totaled ~81 points) surfaced a deeper gap: neither Ceiling_DK_Points nor
# Boom_Adjusted_DK_Points reads TODAY's matchup on their volatility side -
# only the mean term does. This asks a genuinely different question: WHICH
# hitters are likely to boom today, given today's specific opponent, not
# just who has boomed historically.
#
# Backtested (dfs_ceiling.backtest_matchup_boom_signal, same 20-date
# no-lookahead sample, n=5,430, 689 real boom-days against an average
# no-lookahead threshold of ~14.0 points) - capture rate of those 689 real
# boom days (what fraction were flagged in advance by each signal's own
# top decile):
#
#   Mean projection (DK_Points_Hitter) alone:            10.6%
#   Matchup_Boom_Score (Boom_Rate * today's Matchup_Ratio): 14.1%
#   Boom_Rate alone (NO matchup adjustment at all):        17.3%
#
# Honest, somewhat surprising result: the matchup adjustment made it
# WORSE, not better. Boom_Rate alone beat the matchup-multiplied version
# by a real margin. Multiplying by Matchup_Ratio added noise here,
# consistent with Matchup_Ratio already being flagged elsewhere in this
# project (dfs.py's module docstring) as the single highest-risk modeling
# choice in the whole DFS feature set - already superseded by an ML model
# for the main hitter projection for exactly this reason. The same weak
# signal that dragged down DK_Points_Hitter drags down Matchup_Boom_Score
# when multiplied in.
#
# Because Boom_Rate is the real, validated win here, it's exposed as its
# own dfs_hitters.csv column (not just an internal ingredient of
# Matchup_Boom_Score) - Boom_Rate should be treated as the more trustworthy
# signal; Matchup_Boom_Score ships informational/exploratory only, not as
# an improvement over it. Reported plainly, not reframed as a win - same
# honesty standard as every other backtest in this project. Re-run this
# backtest (and update this comment) after any change to
# dfs.compute_matchup_adjustment's formula.

# --- Opponent offense adjustment (pitcher_matchup.py, pitchers only) ---
#
# Unlike hitters (Matchup_Ratio/Matchup_Hit_Probability/Matchup_Boom_Score
# above), dfs.compute_pitcher_dk_points had NO opponent-quality signal
# anywhere - not even in the base mean projection - before this module.
# K9/BB9/HR9/IP_per_start are all windowed averages of the PITCHER'S OWN
# recent form; nothing scaled Expected_H_Allowed/Expected_ER by how good
# the opposing offense actually is today (a real, user-observed gap: a
# pitcher with a great price-adjusted boom profile projects the same
# whether they're facing a last-place offense or a first-place one).
#
# pitcher_matchup.compute_opponent_offense_ratio blends the opponent's
# real team_bases_pg (teams.compute_offensive_edge's pure-offense half,
# before that function's own opponent subtraction - see its docstring for
# why the already-existing offensive_edge/true_power are NOT reusable here,
# both contaminated by netting out a STALE opponent, not today's) against
# the league average, clipped to PITCHER_MATCHUP_OFFENSE_CLIP so one
# extreme-outlier offense can't blow up a projection. weight=0.0 is the
# built-in null hypothesis - it returns exactly 1.0 (today's unadjusted
# heuristic), not an approximation of it - so the grid search below always
# includes an honest no-adjustment baseline to beat, not just a range of
# nonzero guesses.
PITCHER_MATCHUP_OFFENSE_CLIP = (0.85, 1.15)
PITCHER_MATCHUP_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

# Backtested (pitcher_matchup.backtest_pitcher_matchup_signal, real
# persisted Statcast, --days 90 - pitcher sample size is much smaller than
# the hitter-side backtests above, dozens of probable starters per date vs.
# thousands of plate appearances, so this needed a longer window than the
# 20-date default used elsewhere in this project to get a reportable
# sample). n=2,081 real pitcher-days, correlation and MAE of the adjusted
# DK_Points_Pitcher against that date's REAL Actual_DK_Points_Modeled, per
# weight in PITCHER_MATCHUP_WEIGHT_GRID:
#
#   weight=0.00 (baseline, no adjustment): correlation 0.3397, MAE 6.6736
#   weight=0.25:                            correlation 0.3407, MAE 6.6658
#   weight=0.50:                            correlation 0.3414, MAE 6.6612
#   weight=0.75:                            correlation 0.3416, MAE 6.6605
#   weight=1.00 (full, unblended ratio):    correlation 0.3417, MAE 6.6600
#
# Honest result: the direction is right (correlation rises and MAE falls
# monotonically as weight increases, and the win is FULLY monotonic across
# the whole grid, unlike Matchup_Boom_Score's own backtest which went
# backwards) - but the MAGNITUDE is tiny. Weight=1.0 vs. weight=0.0 is only
# a 0.6% relative correlation improvement and a 0.2% relative MAE
# improvement, nowhere near the real margins that justified this project's
# other nonzero defaults (e.g. Boom_Adjusted_DK_Points' k=1.0 needed a 24%
# relative capture-rate improvement over k=0.0 to be chosen). At n=2,081
# this small a gap isn't distinguishable from noise with any real
# confidence. PITCHER_MATCHUP_OFFENSE_WEIGHT stays 0.0 (today's unadjusted
# heuristic) for exactly that reason - Opponent_Offense_Ratio still ships
# as an informational-only column (see docs/dfs.js's pitcher table) rather
# than being silently dropped, since the DIRECTION is real evidence a user
# may still want to see, even though the MAGNITUDE doesn't clear this
# project's bar for changing the live default. Re-run this backtest (and
# update this comment) after any change to dfs.compute_pitcher_dk_points'
# formula or to teams.compute_offensive_edge's team_bases_pg.
PITCHER_MATCHUP_OFFENSE_WEIGHT = 0.0

# --- Value_Score (dfs_optimizer.py, "stars not superstars" roster construction) ---
#
# Real evidence from an actual DK contest: this project's own mean/
# ceiling/boom-adjusted optimizer objectives all independently converged
# on the SAME two most expensive pitchers, spending $19,900-$21,200 on
# pitching and leaving barely enough budget for 8 undifferentiated
# floor-priced hitters. Root cause, diagnosed directly against real
# numbers: Estimated_Salary's fixed $2,000 floor is a much smaller
# fraction of an elite player's price than a replacement-level player's,
# so ANY high scorer - regardless of position - gets a structurally
# better AVERAGE dollars-per-point rate purely from that floor dilution,
# even though the earlier parity fix already equalized the MARGINAL rate.
# An objective that maximizes raw point totals under a budget will always
# rationally chase that average-rate advantage and overpay for the 1-2
# biggest scorers, leaving nothing to build a real roster (a mix of
# reliable "consistent" floor plays and genuine "boom" upside plays, in
# the user's own framing - "consistent players carry their own, boom
# players pick up slack") with what's left.
#
# Value_Score = boom-adjusted points per $1,000 of Estimated_Salary -
# directly rewards being UNDERPRICED relative to real upside (a "star")
# over being fully priced-in already (a "superstar"), instead of chasing
# raw point totals regardless of cost. Computed for BOTH hitters and
# pitchers (dfs_optimizer.build_player_pool), unlike Boom_Adjusted_DK_Points/
# Matchup_Boom_Score above, which are hitter-validated or hitter-only.
#
# DFS_VALUE_BOOM_K_PITCHER gives pitchers real boom credit for this
# specific objective - explicitly NOT the same as DFS_BOOM_ADJUSTED_K_PITCHER
# (0.0) above, which was validated against a different, narrower question
# (does boom-adjusting predict WHICH DAY a pitcher booms - it didn't).
# Implemented here anyway, per explicit user direction, because Value_Score
# answers a different question: roster-construction diversification, not
# single-day prediction. Matches hitters' own validated k=1.0 for
# consistency - not independently re-validated for pitchers via a capture-
# rate backtest (that specific metric already showed no k helps there for
# THAT question), so treat this as a deliberate strategic choice, not a
# claimed statistical win.
DFS_VALUE_BOOM_K_PITCHER = 1.0

# Value_Score maximizes a per-dollar RATIO, which carries no pressure to
# spend anywhere close to the salary cap - confirmed via a real sanity
# check against actual production data: objective=Value_Score selected a
# full legal lineup for only $39,700 of the $50,000 cap (total DK_Points
# fell from 81.38 to 56.31 vs. objective=DK_Points on the same slate),
# because leaving $10,300 unspent doesn't cost the ratio objective
# anything. That's a real flaw, not the intended "stars not superstars"
# behavior - a usable lineup has to actually spend close to its budget.
#
# DFS_VALUE_MIN_SALARY_FRACTION adds a floor constraint
# (sum(Estimated_Salary) >= fraction * DFS_SALARY_CAP) so the optimizer is
# forced to use most of the cap even under Value_Score. Chosen via the
# same real pool, testing three fractions of the $50,000 cap:
#   0.85 (min $42,500): total_salary=$43,900  DK_Points=67.99  pitcher_salary=$16,900
#   0.90 (min $45,000): total_salary=$46,000  DK_Points=73.66  pitcher_salary=$19,000
#   0.95 (min $47,500): total_salary=$47,500  DK_Points=77.79  pitcher_salary=$19,900
# 0.95 defeats the purpose (pitcher spend creeps back to the same
# overpriced $19,900 the whole feature exists to avoid). 0.85 keeps
# pitcher spend near the real winning-lineup range ($17,500-$18,100 in
# the contest that originally motivated this) while still using most of
# the budget - the best balance of the three, not a formally optimized
# value.
DFS_VALUE_MIN_SALARY_FRACTION = 0.85

# --- Optimal Lineup (docs/dfs.html's "Optimal Lineup" tab, roster_positions.py,
# estimated_salary.py, dfs_optimizer.py, scripts/build_optimal_lineup.py) ---
#
# A salary-cap-and-position-slot DraftKings Classic MLB lineup optimizer,
# built on top of the DFS Player Rankings' own DK_Points_Hitter/
# DK_Points_Pitcher projections. IMPORTANT: DraftKings has no public API for
# real contest salaries - there is no free, ToS-compliant way to fetch them.
# Rather than scrape DraftKings (fragile, likely against their ToS) or
# require a manual daily CSV upload, Estimated_Salary here is a MODELED
# number derived from this project's own point projections - explicitly NOT
# a real DraftKings price. This was an explicit user decision ("build your
# best guess at pricing based on performance") after the tradeoffs were
# raised, not a default fallen into unnoticed. See estimated_salary.py's
# module docstring for the exact formula and every place this gets
# relabeled/disclaimed (config.py here, the CSV column name itself
# `Estimated_Salary` never bare `Salary`, docs/dfs.html's warning box,
# docs/dfs.js's rendered column header, README). A real DFS player could
# lose real money mistaking this for what DraftKings will actually charge -
# treat every one of those disclaimer sites as load-bearing, not decorative.

# DraftKings Classic MLB roster construction, confirmed live via web search
# (not memory) against https://www.draftkings.com/help/rules/mlb: 10 roster
# spots, no FLEX/UTIL slot, $50,000 salary cap.
DFS_ROSTER_SLOTS = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
DFS_SALARY_CAP = 50000

# Real DK salaries: a universal $2,000 floor and always $100 increments -
# both confirmed via the same source above. The ceiling below is NOT from
# an official DK table (DraftKings doesn't publish one - prices float
# algorithmically) - it's an anecdotal 2026 example from secondary sources
# (elite starters ~$9,500-$10,500), rounded outward to a defensible
# ceiling. This distinction (floor/increment are real DK rules; the
# ceiling is an estimate) is itself part of the honesty requirement here -
# don't let the presence of some real numbers imply the ceiling is equally
# authoritative.
DFS_ESTIMATED_SALARY_FLOOR = 2000
DFS_ESTIMATED_SALARY_ROUND_TO = 100

# --- Salary $/point parity fix (2026-07-26) ---
#
# v1 had SEPARATE reference point ranges and SEPARATE salary ceilings per
# position (DFS_REFERENCE_MIN/MAX_POINTS_HITTER/_PITCHER,
# DFS_ESTIMATED_SALARY_CEILING_HITTER/_PITCHER, both since removed) - each
# group independently min-max-scaled to its OWN range/ceiling. This was a
# real bug found in production: pitcher DK scoring naturally spans a much
# wider raw point range (~2.6-22.75, box-score categories piling up over 6
# innings) than hitters' hit-type-driven scoring (~2.6-4.7), so
# independently rescaling each to its own ceiling made a pitcher's
# marginal DK point worth roughly 5x a hitter's in salary terms - a pure
# scaling artifact, not real relative DFS value (confirmed by inspecting a
# real optimizer output: 2 elite pitchers alone consumed ~$22K of the
# $50K cap, leaving 8 hitter slots filled with replacement-level bats).
#
# Fixed by collapsing both position groups onto ONE shared reference point
# range and ONE shared salary ceiling (estimated_salary.py's
# compute_hitter_estimated_salary/compute_pitcher_estimated_salary both
# delegate to the same constants below) - one DK point is worth the same
# dollar amount regardless of position; a pitcher still costs far more in
# practice only because pitchers genuinely PROJECT far more points, which
# is the real, not artifactual, reason.
#
# Computed from a REAL 20-game-date backtest sample (dfs_backtest.
# backtest_dfs_projections, the same no-lookahead recompute every other
# backtest in this project uses), pooling both position groups' DK_Points
# together - not a single day's snapshot like the pre-fix v1 constants
# were, and not guessed. Hitters: DK_Points_Hitter ranged 0.1595-16.6757
# (mean 4.827, n=5,430). Pitchers: DK_Points_Pitcher ranged 0.1207-25.1412
# (mean 11.459, n=468) - the pitcher range fully spans the hitter range on
# both ends here, so the shared min/max below are effectively the pitcher
# extremes; that's expected, not a bug, since pitcher DK scoring
# genuinely spans a wider range than hitter scoring even after B's
# widening. With this shared scale, a real average hitter (~4.8 points)
# now prices around $3,700 and a real average pitcher (~11.5 points)
# around $6,100 - a plausible, non-degenerate spread. Compare to the old
# per-position scaling, where every pitcher's dollar-per-point rate was
# independently ~5x cheaper than every hitter's purely from the separate
# scaling - the optimizer exploited that mispricing by loading up on
# "cheap" pitcher points, which is the actual mechanism behind the real
# 2-elite-pitchers-eat-$22K bug described above.
#
# Acceptance check against real production DK_Points (2026-07-25,
# docs/data/dfs_hitters.csv/dfs_pitchers.csv): a hitter projecting exactly
# 8.0 points priced at $6,500 under the OLD scale (pinned near the hitter
# ceiling) vs. a pitcher projecting the SAME 8.0 points at only $4,400
# (still cheap on pitchers' much wider range) - a 48% price difference for
# an identical point total, purely the scaling artifact. Both now price
# identically at $4,800 under this shared scale.
DFS_REFERENCE_MIN_POINTS = 0.1207
DFS_REFERENCE_MAX_POINTS = 25.1412
DFS_ESTIMATED_SALARY_CEILING = 11000

# Statcast plate-appearance outcome values that count as a "completed" event
# (used to filter pitch-by-pitch data down to one row per at-bat outcome).
COUNTED_EVENTS = [
    "field_out",
    "force_out",
    "single",
    "double",
    "strikeout",
    "home_run",
    "grounded_into_double_play",
    "triple",
    "fielders_choice_out",
    "double_play",
    "field_error",
    "fielders_choice",
    "strikeout_double_play",
    "walk",
    "hit_by_pitch",
]

# --- NFL DFS: Data & Season Boundaries (nfl_data.py) ---
#
# The season this pipeline targets for live use. As of this build
# (2026-08-01), the 2026 NFL season has not started yet (kickoff is early
# September) - `nflreadpy.get_current_season()` itself confirmed live as
# returning 2025 (the last real completed season) with
# `get_current_week()` returning 22, i.e. the whole 2025 season including
# playoffs is done and there is no "upcoming week" to compute at all
# during the offseason. This is a real, expected gap for
# nfl_schedule.py's later "determine the upcoming week" logic to handle
# honestly, not a bug to engineer around here.
NFL_SEASON = 2026

# Real seasons backfilled by scripts/fetch_nfl_historical.py so
# nfl_dfs_backtest.py has real fuel immediately rather than waiting on
# the live season to slowly accumulate weeks - confirmed live that
# nflreadpy's real usable historical depth goes back to at least 1999
# (a 1999 load_player_stats call returned 16,839 real rows). Deliberately
# NOT that entire depth for this list - modern-era seasons (roster
# construction, offensive scheme, injury-report practices) are more
# representative of what a live 2026 projection will actually face than
# 1999-era football is, and a 10-season window already gives Phase 7's
# backtest a real sample in the tens of thousands of player-weeks.
NFL_HISTORICAL_SEASONS = list(range(2016, 2026))

NFL_RAW_DATA_DIR = "data/raw/nfl"

# --- NFL DFS: Rolling-Window Player Form (nfl_passing.py, nfl_rush_rec.py) ---
#
# Games-back windows, NOT day-count ones like MLB's WAVE_WINDOWS - NFL's
# weekly cadence and bye weeks make a calendar-count window silently
# under-sample (see the plan's "windows are game-count, not
# calendar-count" guiding principle, and config.LINEUP_WINDOW_GAMES's own
# precedent for the same reasoning). nfl_data.fetch_weekly_stats already
# omits any week a player didn't play, so ranking a player's own rows by
# recency and slicing the most recent N naturally skips byes/inactives
# without extra detection logic.
#
# PLACEHOLDER WEIGHTS - unlike WAVE_WINDOWS, these have NOT been
# backtested yet (no NFL backtest exists until Phase 7). Shaped the same
# way (heavier weight on the most recent window, a full-history anchor
# for stability) as a reasonable starting point only - expect these to
# change once nfl_dfs_backtest.py runs for real.
NFL_QB_WINDOWS = [
    (None, 0.20),
    (8, 0.30),
    (4, 0.50),
]
NFL_SKILL_WINDOWS = [
    (None, 0.20),
    (8, 0.30),
    (4, 0.50),
]

# Small-sample qualifiers, same role as DFS_PITCHER_MIN_STARTS - a QB/
# skill player with only 1-2 games of history has per-game rates that are
# close to pure noise (especially anything TD-rate-based). Gating on
# these is left to the DK-scoring consumer (nfl_dfs.py, Phase 4), same
# "expose the count, let the caller qualify" pattern pitcher_form.py
# already uses for DFS_PITCHER_MIN_STARTS.
NFL_QB_MIN_GAMES = 3
NFL_SKILL_MIN_GAMES = 3

# --- NFL DFS: Team Defense & Matchup (nfl_teams.py, nfl_matchup.py) ---
#
# Games-back windows for nfl_teams.compute_defense_rolling_rates - same
# "games-back, not day-count" reasoning as NFL_QB_WINDOWS/NFL_SKILL_WINDOWS
# above. PLACEHOLDER WEIGHTS, not yet backtested (see those constants'
# docstring - same caveat applies here).
NFL_DEFENSE_WINDOWS = [
    (None, 0.20),
    (8, 0.30),
    (4, 0.50),
]

# nfl_matchup.compute_opponent_adjustment_ratio's clip range, mirroring
# PITCHER_MATCHUP_OFFENSE_CLIP - keeps one extreme-outlier defense (a
# 2-game sample allowing an absurd amount) from blowing up an offensive
# player's projection. NFL_MATCHUP_DEFENSE_CLIP is the mirror-image clip
# for the OTHER adjustment direction - a DST's own projected points,
# scaled by how good the OPPOSING OFFENSE is (nfl_dst.py, Phase 4) - not
# consumed yet, defined here alongside its sibling since both are the
# same "matchup ratio" concept applied in opposite directions.
NFL_MATCHUP_OFFENSE_CLIP = (0.85, 1.15)
NFL_MATCHUP_DEFENSE_CLIP = (0.85, 1.15)

# Mirrors PITCHER_MATCHUP_WEIGHT_GRID - grid searched by a future
# nfl_dfs_backtest.py go/no-go run (Phase 7), not yet run for NFL.
NFL_MATCHUP_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

# Ships at 0.0 (informational-only, exactly reproduces the unadjusted
# heuristic) for the live 2026 season regardless of what a historical-
# seasons backtest shows, same "ship conservatively" reasoning as
# PITCHER_MATCHUP_OFFENSE_WEIGHT - NFL's smaller in-season sample and
# real year-over-year discontinuity (coaching/scheme/roster turnover)
# make this an even harder bar to clear than pitcher matchup's already-
# marginal real result. Revisit only after nfl_dfs_backtest.py
# (Phase 7) reports a real, non-noise margin over weight=0.0.
NFL_MATCHUP_WEIGHT = 0.0

# --- NFL DFS: DK Scoring (nfl_dfs.py) ---
#
# DraftKings NFL Classic scoring, confirmed live via web search against
# DraftKings' own real rules (not from memory) - see README's NFL DFS
# section for the citation. Unlike this project's MLB scoring, which had
# to approximate hit-type value from a linear signal, DK's real NFL
# categories map directly onto the per-game rate stats nfl_passing.py/
# nfl_rush_rec.py already compute - the actual risk here is entirely
# upstream in those windowed projections, not in this formula.
NFL_DK_PASS_YARD_POINTS = 0.04  # 1 point per 25 passing yards
NFL_DK_PASS_TD_POINTS = 4
NFL_DK_INTERCEPTION_POINTS = -1  # interception thrown
NFL_DK_RUSH_YARD_POINTS = 0.1  # 1 point per 10 rushing yards
NFL_DK_RUSH_TD_POINTS = 6
NFL_DK_RECEIVING_YARD_POINTS = 0.1  # 1 point per 10 receiving yards
NFL_DK_RECEIVING_TD_POINTS = 6
NFL_DK_RECEPTION_POINTS = 1  # full PPR
NFL_DK_FUMBLE_LOST_POINTS = -1
NFL_DK_2PT_POINTS = 2  # 2-point conversion, pass/run/catch all score the same
NFL_DK_100_YARD_BONUS = 3  # 100+ rushing OR receiving yards in a game (each counted separately)
NFL_DK_300_PASS_YARD_BONUS = 3  # 300+ passing yards in a game

# --- NFL DFS: DST Scoring (nfl_dst.py) ---
#
# Also confirmed live via web search - see README's NFL DFS section.
NFL_DK_DST_SACK_POINTS = 1
NFL_DK_DST_INT_POINTS = 2
NFL_DK_DST_FUMBLE_REC_POINTS = 2
NFL_DK_DST_TD_POINTS = 6  # defensive/return TD of any kind - see nfl_dst.py for which real columns feed this
NFL_DK_DST_SAFETY_POINTS = 2
NFL_DK_DST_BLOCKED_KICK_POINTS = 2

# Points-allowed bucket table: list of (upper_bound_inclusive, dk_points),
# checked in order, last entry's upper_bound is None ("and above"). Real
# DK rule: points allowed only counts points surrendered while the DST
# unit is on the field (a pick-six is charged to the DEFENSE that allowed
# it, not this team's own DST) - nflreadpy's per-game final score (used
# by nfl_dst.compute_points_allowed) doesn't make that distinction, a
# known, documented v1 simplification (the same category of approximation
# as nfl_dst.py's windowed-mean-through-the-bucket-table choice below).
NFL_DK_DST_POINTS_ALLOWED_BUCKETS = [
    (0, 10),
    (6, 7),
    (13, 4),
    (20, 1),
    (27, 0),
    (34, -1),
    (None, -4),
]

# --- NFL DFS: Estimated Salary (nfl_estimated_salary.py) ---
#
# Direct structural port of DFS_ESTIMATED_SALARY_*/DFS_REFERENCE_*_POINTS
# - see estimated_salary.py's module docstring for the full "why a shared
# reference range, not per-position" reasoning (equally true here: QB and
# skill-position DK scoring span very different raw point ranges, so a
# per-position-group scale would misprice a point the same way the old
# MLB hitter/pitcher split did).
#
# Reference range computed from REAL 2025 season DK_Points_QB/
# DK_Points_Skill/DK_Points_DST (nfl_dfs.compute_qb_dk_points/
# compute_skill_dk_points, nfl_dst.compute_dst_dk_points, run against
# nflreadpy's real load_player_stats/load_team_stats/load_schedules([2025])
# - full-season blended rates, not a single week's snapshot), pooling all
# three position groups together, real min/max (not a percentile) - same
# convention DFS_REFERENCE_MIN/MAX_POINTS used. QB: -0.33 to 24.52
# (n=81). Skill (RB/WR/TE): -0.05 to 27.89 (n=530). DST: -1.51 to 12.33
# (n=32). Unlike the MLB reference range, this one includes real negative
# values (a low-efficiency QB's interceptions, or a DST's worst
# points-allowed bucket, can both go net negative) - compute_estimated_salary's
# linear scaling handles that fine, same as any other range.
NFL_DFS_REFERENCE_MIN_POINTS = -1.5066
NFL_DFS_REFERENCE_MAX_POINTS = 27.8924
# DraftKings NFL Classic salary cap is $50,000 and $100 increments -
# confirmed live via web search (README's NFL DFS section has the
# citation), same real cap MLB Classic uses. The FLOOR is NOT an
# official DK table (DraftKings doesn't publish one - prices float
# algorithmically), same honesty caveat DFS_ESTIMATED_SALARY_FLOOR's own
# docstring makes for its ceiling: this is an estimate informed by real
# evidence (live search found real Week 1 2026 DST salaries topping out
# at $3,500, suggesting a real floor at or below $3,000), reusing MLB's
# own $2,000-$11,000 range as the defensible starting point since NFL's
# real floor/ceiling aren't independently confirmed.
NFL_DFS_ESTIMATED_SALARY_FLOOR = 2000
NFL_DFS_ESTIMATED_SALARY_CEILING = 11000
NFL_DFS_ESTIMATED_SALARY_ROUND_TO = 100
NFL_DFS_SALARY_CAP = 50000

# --- NFL DFS: Optimizer (nfl_dfs_optimizer.py) ---
#
# DK Classic's real 9-slot roster - confirmed live via web search
# alongside the salary cap above (README's NFL DFS section has the
# citation). Sums to 9. Passed directly to dfs_optimizer.solve_optimal_lineup
# (reused unmodified, not re-implemented - see nfl_dfs_optimizer.py's
# module docstring).
NFL_DFS_ROSTER_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}

# --- NFL DFS: Backtesting (nfl_dfs_backtest.py) ---
#
# Real no-lookahead backtest (nfl_dfs_backtest.backtest_nfl_dfs_projections,
# scripts/backtest_nfl_dfs_rankings.py) against the FULL real backfilled
# history (config.NFL_HISTORICAL_SEASONS, 2016-2025, data/raw/nfl/) - MAE
# and correlation of each projection against that week's REAL realized DK
# points, vs. a naive "always predict this whole sample's own mean"
# baseline (same go/no-go bar every other signal in this project uses:
# beat naive baseline AND a simpler heuristic by a real margin):
#
#   QB:    MAE 6.6609 vs. naive-baseline MAE 7.8183 (14.8% better), correlation 0.514 (n=6,358)
#   Skill: MAE 4.6839 vs. naive-baseline MAE 6.1814 (24.2% better), correlation 0.585 (n=50,988)
#   DST:   MAE 4.7188 vs. naive-baseline MAE 4.6271 (2.0% WORSE),  correlation 0.101 (n=5,490)
#
# QB/Skill also beat a simpler "flat, unweighted full-season-average"
# heuristic (config.NFL_QB_WINDOWS/NFL_SKILL_WINDOWS temporarily set to
# [(None, 1.0)] for the comparison run): QB correlation 0.499 vs. flat's
# 0.485 (MAE 6.6365 vs. 6.7259); Skill correlation 0.596 vs. flat's 0.578
# (MAE 4.5122 vs. 4.6339) - real but modest margins, confirming the
# RECENCY-WEIGHTING mechanism itself (not just "having any player-form
# signal at all") adds real value, on top of already beating the naive
# baseline by a wide margin. This validates the MECHANISM (Phase 2's
# rolling-window blend genuinely predicts real outcomes better than
# guessing or a flat average) - it does NOT validate the specific
# NFL_QB_WINDOWS/NFL_SKILL_WINDOWS weight VALUES (0.20/0.30/0.50), which
# remain an unrecalibrated first-pass placeholder (see those constants'
# own docstrings) - no full grid search over weight combinations has been
# run.
#
# DST is an honest NEGATIVE result: neither the windowed blend nor the
# flat heuristic (DST flat: correlation 0.049, MAE 4.5429 - also worse
# than naive) beats simply guessing the sample mean. nfl_dst.py's
# points-allowed-bucket-via-windowed-mean approximation (see that
# module's own docstring) is the most likely culprit - DST scoring is
# dominated by the highly game-specific, high-variance points-allowed
# category, which a multi-week rolling average of a notoriously noisy
# stat doesn't predict well. DST_Points ships (the optimizer/roster need
# a DST scoring column to function structurally) but should be treated
# as UNVALIDATED/weak, not a trustworthy signal - flagged here and in
# README rather than hidden, same "report honestly either way" standard
# every other backtest in this project holds to.
#
# NFL_MATCHUP_WEIGHT is NOT exercised by this backtest at all -
# nfl_matchup.py's opponent-adjustment ratio is not wired into
# nfl_dfs.compute_qb_dk_points/compute_skill_dk_points's formula (stays a
# separate, standalone, informational-only module - see nfl_matchup.py's
# own docstring), so there is nothing to grid-search yet; it already
# ships at 0.0 regardless, per the "ship conservatively" reasoning in
# that module's docstring.
#
# Reproduce: `python scripts/backtest_nfl_dfs_rankings.py` (needs
# data/raw/nfl/{weekly,team_stats,schedules}_<season>.parquet - see
# scripts/fetch_nfl_historical.py).

# --- NFL Bestball: Position Scarcity (nfl_bestball.compute_position_scarcity) ---
#
# NFL_BESTBALL_SCARCITY_MIN_GAMES (a real games_played >= 8 qualifier) was
# removed after real 2025 data showed it let in players with almost no real
# offensive role - e.g. a real return specialist with games_played=11 but
# exactly 1 real offensive snap all season. games_played only requires ANY
# real stat row that week (even a single special-teams play), not a
# meaningful offensive role. Replaced by NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE
# below - see nfl_bestball.compute_position_scarcity's own docstring for the
# full reasoning. games_played/games_missed are untouched everywhere else
# (the rankings table, the injury-history proxy) - only this qualifier
# changed.
#
# The snap-share qualifier itself was originally a real PER-GAME average
# (avg_offense_pct), which was then replaced by nfl_bestball.compute_
# player_snap_share's real SEASON-TOTAL share (a player's real total
# offensive snaps that season / their team's real total offensive snaps
# that season) - the per-game average let a real one-or-two-game emergency
# spot start at a high per-game rate qualify just as easily as a real
# every-week starter (e.g. a real 2025 backup QB's real 82% single-game
# rate across exactly 1 real game); the season-total share correctly drops
# that same real player to a real 5% share once measured against the
# team's full real season offensive-play total. games_played/dk_points_total
# etc. are unaffected - only how this specific qualifier is computed
# changed.
#
# 0.3 (30% of a team's real total season offensive snaps) is a simple,
# honest first-pass default - NOT backtest-derived, and deliberately lower
# than a "majority of snaps" bar would suggest, because a real SEASON-TOTAL
# share runs meaningfully lower than a per-game rate even for genuinely
# fantasy-relevant committee/complementary players (real 2025 example:
# Jaylen Warren and TreVeyon Henderson, both clearly real, draftable RB2/
# committee-role players with 200+ real season DK points, sit at real
# season shares of 0.47/0.46 - a 0.5 bar would have wrongly excluded real,
# relevant players like them). Easy to override via the build script's own
# flag if a different bar turns out to matter more.
NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE = 0.3

# Which nfl_bestball_rankings.csv column the bell-curve buckets and
# mean/std are computed over. dk_points_total (real realized season
# value), not dk_points_per_game - bestball drafting is about the whole
# season's production, and rate stats alone would erase exactly the
# playing-time/opportunity signal that makes a position scarce or deep in
# the first place.
NFL_BESTBALL_SCARCITY_VALUE_COLUMN = "dk_points_total"

# Tukey's IQR-fence multiplier for excluding real statistical outliers
# before computing a position's mean/std (see
# nfl_bestball._iqr_outlier_bounds/compute_position_scarcity) - 1.5 is the
# standard, textbook convention (not tuned/invented for this project), used
# specifically because it doesn't require an already-computed mean/std as
# an input, unlike a z-score-based rule (real NFL season-point totals among
# a snap-share-qualified population are often right-skewed, so a mean/std
# computed WITH the outliers already baked in would itself be distorted by
# them - see that function's own docstring for the real before/after
# numbers).
NFL_BESTBALL_SCARCITY_IQR_MULTIPLIER = 1.5

# --- NFL Bestball: Roster Construction & Draftable Pool Depth
# (nfl_bestball.compute_draftable_points_floor, compute_position_necessity) ---
#
# Confirmed live via web search (2026-08-07) against DraftKings' own
# published rules and major fantasy outlets (FantasyPros, Establish The
# Run, The Fantasy Footballers): DK Best Ball Mania (the flagship DK
# Best Ball tournament) draft rooms are real 12-team pods, 20 rounds/
# players per team, with a real 8-slot weekly starting lineup (QB, RB,
# RB, WR, WR, WR, TE, FLEX).
NFL_BESTBALL_DRAFT_POD_SIZE = 12

# Real published roster-construction strategy guidance gives typical
# PER-TEAM position counts drafted across those 20 rounds - "how many of
# each position do we actually want on our own team": QB 2-3, RB 5-7 (6
# the consistently-cited number across multiple real sources), WR 6-8,
# TE 1-2. This is the real "position necessity" input -
# compute_position_necessity compares it against how many real players
# at that position actually clear compute_position_scarcity's qualifier
# this season (real role AND real value), not just an invented "wanted"
# number in isolation.
NFL_BESTBALL_ROSTER_TARGET = {"QB": (2, 3), "RB": (5, 7), "WR": (6, 8), "TE": (1, 2)}

# Real user feedback: even a real 50%+ snap-share qualifier still let in
# real players nobody would actually draft (a real WR playing a
# meaningful complementary role but producing very little - snap share
# measures ROLE/health, not VALUE). This is a real, SEPARATE production
# floor, layered on top of (not instead of) the snap-share qualifier -
# real season-total dk_points_total, with the floor itself set at the
# real Nth-ranked player's own real total, N derived from real DraftKings
# Best Ball Mania roster-depth math, not an arbitrary points number.
#
# The midpoint of each real NFL_BESTBALL_ROSTER_TARGET range, times the
# real 12-team pod size, gives a real (not invented) estimate of how many
# players at each position a typical 12-team pool actually drafts - the
# real "draftable pool depth" (derived, not independently hardcoded, so
# this and NFL_BESTBALL_ROSTER_TARGET can never silently drift apart):
NFL_BESTBALL_DRAFTABLE_POOL_SIZE = {
    position: round((lo + hi) / 2 * NFL_BESTBALL_DRAFT_POD_SIZE)
    for position, (lo, hi) in NFL_BESTBALL_ROSTER_TARGET.items()
}

# --- NFL Bestball: Round Split (nfl_bestball.build_bestball_rankings) ---
#
# Confirmed live via web search (2026-08-08) against Establish The Run and
# 4for4: DK Best Ball Mania's real tournament structure runs a real
# "Round 1" across weeks 1-14 (cumulative points across those real weeks
# decide who advances out of each real 12-team draft pod - the top 2),
# followed by three real single-week knockout rounds - "Round 2" at real
# week 15, "Round 3" at real week 16, and the real championship "Round 4"
# at week 17. nfl_bestball_rankings.csv reports the real weeks-1-14 sum as
# r1_dk_points and the real weeks-15-17 sum as r2_r4_dk_points (one
# combined column, not split further per round) - a real, direct answer to
# "was this player producing early enough in the real season to matter for
# real pod advancement, not just totaling well by the end."
NFL_BESTBALL_ROUND1_END_WEEK = 14

# --- NFL Game Predictions (nfl_team_strength.py, nfl_game_picks.py) ---
#
# Real 1:1 structural port of the MLB Automated Game Picks pipeline
# (teams.py's Pyth Strength/SOS/Confidence mixture -> game_picks.py's
# composite win-probability model) - see nfl_team_strength.py's own
# module docstring for the full signal-by-signal mapping. EVERY constant
# below is a real, honestly-labeled STARTING POINT, not an asserted-
# correct value - nfl_game_picks_backtest.py trains on real 2025 weeks
# 1-7 and validates/re-fits against real 2025 weeks 8-18 (held out,
# including real historical closing-line moneylines already sitting in
# schedules_*.parquet) before any of this reaches a live 2026 pick. Same
# "hand-tuned, then honestly backtested" arc MLB's own TEAM_STRENGTH_WINDOWS/
# PYTHAGOREAN_EXPONENT/GAME_PICK_COMPOSITE_WEIGHTS went through - and the
# same real caveat NFL_QB_WINDOWS/NFL_SKILL_WINDOWS/NFL_DEFENSE_WINDOWS
# above already carry ("PLACEHOLDER WEIGHTS, not yet backtested").

# Games-back windows for nfl_team_strength.compute_strength_metrics.
# Tightened from the original (4, 8, None) start (which mirrored
# NFL_QB_WINDOWS/NFL_SKILL_WINDOWS/NFL_DEFENSE_WINDOWS's own shape) to
# (3, 7, None) - real follow-up (2026-09-02): a 17-18 game NFL season
# gets far less benefit from a long smoothing window than MLB's 162-game
# one does, since real personnel/scheme regime changes (a new starting
# QB, an OC change, a key injury) happen often enough that a season is
# too short to let a longer window "catch up" to the new reality on its
# own. This is NOT because single NFL games carry less real luck/variance
# than MLB's - if anything the standard finding is the opposite (a
# 17-game sample doesn't average out game-to-game variance the way 162
# games does, and famously low-persistence stats like raw turnover
# margin are a big part of why) - so tightening these windows is a real
# tradeoff (faster to react to a genuine regime shift, but also faster to
# overreact to one-game noise), not a free improvement. Genuinely
# re-validated by nfl_game_picks_backtest.py against the real 2025 test
# split (old 4/8/None vs this 3/7/None), not assumed better just because
# it reacts faster.
NFL_TEAM_STRENGTH_WINDOWS = [
    (3, 0.50),
    (7, 0.30),
    (None, 0.20),
]

# NFL's own real, commonly-cited Pythagorean win-expectation exponent
# (points-scored/points-allowed based) - NOT MLB's 1.83 (a different real
# sport with a different real scoring distribution). 2.37 is the widely-
# cited real NFL value in the sports-analytics literature (originally a
# Football Outsiders derivation) - a real, sourced starting point, same
# spirit as MLB's own "custom-tuned, not the classic 2" - genuinely
# re-fit against the real 2025 test split in nfl_game_picks_backtest.py,
# not assumed correct just because it's a commonly-cited number.
NFL_PYTHAGOREAN_EXPONENT = 2.37

# Same role/formula as NORMALIZATION_Z_SCALE/CONFIDENCE_SOS_WEIGHT above,
# a real starting point pending the same backtest - not reusing the MLB
# constants directly so tuning one sport's mixture can never silently
# move the other's.
NFL_NORMALIZATION_Z_SCALE = 0.15
NFL_CONFIDENCE_SOS_WEIGHT = 0.3

# Equal-weighted blend of the four team-level signals into one team
# composite rating, direct structural mirror of GAME_PICK_COMPOSITE_WEIGHTS -
# pyth_Strength/pyth_Confidence (the real record-based Pythagorean
# mixture) plus offensive_edge/defensive_edge (real passing_epa+
# rushing_epa+receiving_epa produced/allowed per game - see
# nfl_team_strength.py's own docstring) folded into true_power, same
# "true_power = avg(offensive_edge, defensive_edge)" shape as MLB's
# true_power = avg(offensive_edge, suppression_resistance). No NFL
# analog of suppression_resistance itself (baseball's "held under 3
# runs" shutout-innings framing has no honest 1:1 football translation) -
# defensive_edge fills that same STRUCTURAL role (a real, separate
# defensive-quality signal) with a real football-native stat instead of
# a force-fit port.
# Real follow-up (2026-09-02 - "we should include turnover ratio at a
# game level"): turnover_margin (nfl_team_strength.compute_team_turnover_margin -
# real takeaways minus real giveaways per game, same games-back blend as
# every other signal here) added as its own explicit 5th weight rather
# than folded into true_power - turnovers are a genuinely distinct
# quality dimension from EPA-based offensive/defensive edge (a team can
# be efficient per-play and still hemorrhage the ball), and keeping it
# separate lets the backtest validate/tune it independently instead of
# diluting it into an average. Rebalanced to 5 equal 0.20 weights (down
# from 4 equal 0.25 weights) - a real starting point, pending
# nfl_game_picks_backtest.py's own re-validation, same honest status as
# every other constant here.
# Real follow-up (2026-09-02 - "offensive efficiency (pts/drive)"):
# points_per_drive (nfl_team_strength.compute_team_points_per_drive - real
# points scored per real offensive drive, derived from play-by-play) added
# as a 6th signal, rebalanced to 6 equal ~0.1667 weights. A genuinely
# different efficiency lens than offensive_edge (per-PLAY EPA) - this
# measures how often real drives actually turn into points, not how
# valuable each individual play was - so kept as its own weight rather
# than folded into true_power/offensive_edge, same "let the backtest
# validate it independently" reasoning as turnover_margin above.
NFL_GAME_PICK_COMPOSITE_WEIGHTS = [
    ("pyth_Strength", 1 / 6),
    ("pyth_Confidence", 1 / 6),
    ("defensive_edge", 1 / 6),
    ("true_power", 1 / 6),
    ("turnover_margin", 1 / 6),
    ("points_per_drive", 1 / 6),
]

# A game is only "picked" if the favored side's win probability clears
# this bar - same role as GAME_PICK_MIN_PROBABILITY. Real NFL talent
# gaps tend to be larger than MLB's on a per-game basis, so this starts
# higher than MLB's 0.58 - a real, honest guess pending the backtest,
# not derived.
NFL_GAME_PICK_MIN_PROBABILITY = 0.62

# Same degenerate-input guard as GAME_PICK_RATING_FLOOR.
NFL_GAME_PICK_RATING_FLOOR = 0.05

# How much the QB-continuity adjustment (nfl_team_strength.py - the
# actual starting QB's own real rolling EPA-per-dropback quality,
# blended in when it differs from the team's recent primary QB) shifts
# a team's offensive rating, mirroring GAME_PICK_SUSCEPTIBILITY_WEIGHT's
# role. Ships conservative (informational lean, not a full swap to the
# backup's own thin-sample number) pending a real backtested weight,
# same "ship conservatively until proven" reasoning as NFL_MATCHUP_WEIGHT
# above.
NFL_QB_CONTINUITY_WEIGHT = 0.5

# Path to the saved NFL game-pick probability-calibration artifact
# (ml_models.fit_probability_calibration, trained by
# scripts/train_nfl_game_pick_calibration.py) - same real graceful-
# degradation contract as GAME_PICK_CALIBRATION_MODEL_PATH: a no-op
# until/unless a real trained artifact clears its own real-holdout bar.
NFL_GAME_PICK_CALIBRATION_MODEL_PATH = "data/models/nfl_game_pick_calibration_model.joblib"

# Same purpose as GAME_PICK_MODEL_VERSION, for nfl_game_predictions.py.
# Bumped v1 -> v2 (2026-09-02, tightened windows + turnover_margin added
# to the composite) so game_evaluation.py/the dashboard can segment the
# already-logged v1 picks (the old 4/8/full, 4-signal composite) from
# real picks made under this new logic - same "never silently rewrite
# already-logged history" discipline as MLB's own model_version bumps.
NFL_GAME_PICK_MODEL_VERSION = "v2"

# Walk-forward CV / final-holdout sizing for
# scripts/train_nfl_game_pick_calibration.py - same role as
# ML_FINAL_HOLDOUT_DATES/GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_DATES/
# GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_DATES, but in real NFL WEEK units,
# not calendar days - an NFL season has only ~16-18 real weeks of games
# total (vs. MLB's ~180 real days), so reusing the MLB day-count constants
# directly would leave zero real weeks to train calibration on at all.
# Sized against nfl_game_picks_backtest.py's own real replay (weeks 3-18,
# 16 real weeks): 3 held out as a real untouched final test, a minimum of
# 6 real weeks before the first walk-forward CV fold fires, 2-week test
# blocks after that - real starting points, not derived from a
# calibration-specific backtest of their own (unlike the model constants
# above, which nfl_game_picks_backtest.py itself validates).
NFL_GAME_PICK_ML_FINAL_HOLDOUT_WEEKS = 3
NFL_GAME_PICK_ML_WALK_FORWARD_MIN_TRAIN_WEEKS = 6
NFL_GAME_PICK_ML_WALK_FORWARD_TEST_BLOCK_WEEKS = 2
