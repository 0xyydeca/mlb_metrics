# mlb_metrics

Daily updated hitter/pitcher/team metrics (WAVE, PAVE, Game_Hit_Probability,
team Strength/Confidence) built from rolling windows of Statcast data, published
to `docs/` as a GitHub Pages dashboard.

## Project layout

- `src/mlb_metrics/` - the pipeline: `config.py` (window lengths and blend
  weights), `helpers.py` (event classifiers), `data.py` (Statcast fetch/raw
  persistence), `hitters.py`, `pitchers.py`, `teams.py` (metric computation),
  `pipeline.py` (orchestration + CLI), `predictions.py`/`evaluation.py`/
  `git_backtest.py` (hitter-pick backtesting - see below), `schedule.py`
  (MLB Stats API - probable pitchers, game schedules, final scores),
  `matchup.py` (batter-level opponent-pitching blend), `game_picks.py`/
  `game_predictions.py`/`game_evaluation.py` (Automated Game Picks - see
  below).
- `scripts/wave.py` - daily pipeline entrypoint (`python scripts/wave.py`).
- `scripts/run_backtest.py` - backtest entrypoint (`python scripts/run_backtest.py`).
- `scripts/evaluate_current_model.py` - read-only backtest report for the
  currently-live hitter-pick and game-pick models (see "Model versioning"
  below); doesn't write to `data/predictions/` or `docs/data/`.
- `scripts/fetch_lahman.py` / `scripts/build_age_curves.py` /
  `scripts/backtest_age_curve.py` - the Age Curves page's data pipeline
  (see "Age Curves" below); occasional-cadence, not part of the daily
  pipeline.
- `data/raw/` - persisted raw Statcast pulls, one parquet file per season,
  committed daily alongside the output CSVs so history accumulates run over run.
- `data/predictions/predictions.csv` - append-only log of every daily hitter
  pick (date, player, predicted probability, realized actual_hit once
  known). `data/predictions/game_predictions.csv` is the analogous log for
  Automated Game Picks, keyed on `game_pk` instead of a player id.
- `docs/data/` - the published `wave.csv` / `pave.csv` / `confidence.csv`
  consumed by the `docs/` dashboard, plus `backtest_summary.csv` (see below).
  `confidence.csv` includes `Bullpen_PAVE_PLUS`
  (and `Bullpen_BAA`/`Bullpen_Power_A`/`Bullpen_HR_Per`/`Bullpen_AtBats`): PAVE
  computed only from each team's relief appearances, not its starters - a hitter
  typically sees the bullpen for 1-3 of their 3-5 at-bats in a game, so evaluating
  a matchup against "the starter's PAVE" alone ignores most of the at-bats.
- `tests/` - pytest suite covering the event classifiers, the window-blend
  formulas (WAVE, PAVE, Bullpen_PAVE, Game_Hit_Probability), and the
  backtesting pipeline, against hand-computed expected values and synthetic
  end-to-end scenarios.

## Backtesting

Every daily `scripts/wave.py` run now also logs that day's top picks to
`data/predictions/predictions.csv` (before the games happen) and resolves
any previous day's pending picks against the newly-fetched data - this is
what finally answers whether WAVE/Game_Hit_Probability actually predicts
hits, which nothing in the project did before.

`scripts/run_backtest.py` additionally replays the ~40+ days of `wave.csv`
already sitting in git history through the same pick-selection logic (see
`git_backtest.py`), so there's a backtestable dataset immediately rather than
waiting weeks for new predictions to accumulate. It resolves against
`data/raw/` and, with `--fetch-missing`, against Statcast directly for any
date not already persisted - then writes `docs/data/backtest_summary.csv`
(hit rate by pick rank, Brier score, log loss, calibration) via
`evaluation.py`. A `Backtest` GitHub Actions workflow (`workflow_dispatch`)
runs this with real network access and commits the results.

Every pick is qualified by `BACKTEST_MIN_PLATE_APPEARANCES` (config.py,
default 30) before ranking - without it, a batter with a handful of at-bats
and one lucky hit can show a probability of 1.0 and dominate the picks on
pure sample-size noise.

## Beat the Streak Tracker (dashboard)

The dashboard's top section simulates actually playing Beat the Streak,
following its real rules rather than a simplified per-day win/loss model
(see `evaluation.streak_progression()`):

- A pick is graded "recommended" if a blended score - the mean of whichever
  of `predicted_probability` (Game_Hit_Probability), `probability`, and
  `Matchup_Hit_Probability` are available for that pick (see
  `evaluation._combined_probability`) - clears `DAILY_PICK_MIN_PROBABILITY`
  (config.py, default 0.77), else "speculative". This used to gate on
  Game_Hit_Probability alone at 0.80, which ignored the other two signals
  and produced zero-pick days whenever GHP landed just under the bar even
  with a strong matchup; blending all three (and lowering the bar to match,
  since a mean of two-or-three probabilities runs lower than GHP alone -
  see config.py's docstring for the backtest that picked 0.77) fixed that,
  empirically validated via a 42-day git-history-replay backtest.
- **The dashboard's top DAILY_PICK_MAX (2) candidates are ALWAYS shown,
  each individually graded** (`evaluation.graded_daily_picks`) - real user
  feedback: that 42-day replay never carried real (non-NaN)
  `Matchup_Hit_Probability` (it wasn't persisted to git history at the
  time), so once live runs started actually carrying it most days, the
  blended mean ran measurably lower than the replay ever exercised - a real
  5-day-straight stretch (2026-08-11 through 2026-08-15) landed the
  top-ranked candidate's combined probability at 0.71-0.77 every single
  day, just under the bar, leaving the dashboard blank despite real,
  already-qualified candidates existing all five days. Rather than
  re-chase a moving threshold, `DAILY_PICK_MIN_PROBABILITY`'s role changed
  instead of its value: it's no longer "the bar a day needs to clear to
  show anything" - it's now the recommended/speculative grade boundary,
  and the dashboard always shows its real top candidates, honestly graded,
  never blank on a weak-slate day. **Only "recommended"-grade picks still
  count toward the tracked streak/day_survival_rate** (`streak_progression`/
  `_recommended_picks`, unchanged) - a "speculative" day is still a real
  no-op for the streak, exactly like a zero-pick day was before. A date
  with literally NOTHING logged (a true off day, not a weak one) still
  gets the explicit `"no_pick"` row in the export rather than silently
  vanishing.
- A pick with an at-bat and no hit resets the streak to 0.
- A pick with zero at-bats that day (rained out, DNP, not in the lineup)
  is neutral - it neither advances nor resets the streak.
- Otherwise the streak increases by however many picks got a hit (0, 1,
  or 2).

Distinguishing "confirmed zero at-bats" from "outcome not known yet"
needed an `at_bats` field per pick (`predictions.py`) - a row only
resolves once its date falls within the coverage of the fetched/persisted
Statcast data, not just once that specific batter shows up in it.

This reads `docs/data/beat_the_streak_picks.csv` and
`beat_the_streak_summary.csv`, written by `evaluation.build_beat_the_streak_export()`
after every daily run and every backtest run - so the picks that make up the
streak are always the ones that were actually recommended *at the time*
(from `data/predictions/predictions.csv`), not recomputed with hindsight.

### Model versioning

`predictions.csv` is append-only - once a pick is logged, its
`predicted_probability`/which player was picked is frozen forever (only the
outcome fills in later). That means a selection-logic change (e.g. the
qualifier/ranking work above) never visibly moves the all-time accuracy
numbers until enough new-logic days outweigh the old ones, which can take a
long time. Every row is tagged with `model_version`
(`config.HITTER_MODEL_VERSION`, bumped whenever `select_picks`'s
qualifier/ranking logic meaningfully changes; git-history-reconstructed rows
from `git_backtest.py` are tagged `"legacy"` instead, since they don't
reflect current live logic) - `evaluation.summarize()`/
`build_beat_the_streak_export()` both take an optional `model_version` filter
so a recalibration's real forward performance is directly checkable
(`evaluation.summarize(log, model_version=config.HITTER_MODEL_VERSION)`)
instead of being diluted by history. `docs/data/beat_the_streak_summary_by_version.csv`
carries this same split (`all_time` plus the current version) for the
dashboard/anyone reading the CSVs directly. Game picks use the identical
pattern (`config.GAME_PICK_MODEL_VERSION`, `game_evaluation.build_game_picks_export`'s
own `model_version` filter).

Even with `model_version` tagging, a change made today still won't show up
in *live* picks until tomorrow's run - today's picks were already logged
before the change existed, and the append-only log correctly refuses to
rewrite them. `scripts/evaluate_current_model.py` answers "how does the
model I'd ship today actually perform" immediately instead of waiting:
it backtests the currently-live hitter-pick logic
(`git_backtest.reconstruct_historical_picks`, replaying `wave.csv` git
history through today's `select_picks()`) and the currently-live game-pick
logic (`game_picks_backtest.reconstruct_historical_game_picks_from_persisted`,
which recomputes `confidence.csv`/`pave.csv` fresh from persisted Statcast
per replayed date rather than replaying old git-committed snapshots, so a
signal added since - e.g. `Power_A_PLUS` - isn't silently skipped over the
way it would be replaying old commits), resolves both against real
outcomes, and prints an accuracy/Brier-score report. It's read-only - it
never writes to `data/predictions/` or `docs/data/`.

## Lineup awareness

Backtesting found ~30% of logged top-5 picks had zero at-bats on the day
they were picked - the batter wasn't even in the lineup. Two independent
fixes, both applied in `predictions.select_picks`:

- **Batting-order consistency** (`data.assign_batting_order`, `lineup.py`) -
  derived entirely from Statcast data already persisted, no new dependency,
  works retroactively. A batter only qualifies if their average batting-order
  slot over their current team's last `LINEUP_WINDOW_GAMES` games is in the
  top half (`LINEUP_TOP_HALF_MAX_SLOT`) *and* they've actually started at
  that rate (`LINEUP_MIN_START_RATE`) - together these exclude both a bench
  player's hot week and a recent call-up without a real track record. A
  mid-season trade resets the window to the batter's new team only.
- **Probable-pitcher matchup blending** (`schedule.py`, `matchup.py`) - a new
  dependency (`MLB-StatsAPI`, since `pybaseball` has no schedule/lineup
  support at all) fetches today's probable starters and, via a log5
  (odds-ratio) combination of the batter's own AB-level `WAVE` rate with the
  opposing starter's/bullpen's raw `PAVE`/`Bullpen_PAVE`, produces a
  `Matchup_Hit_Probability` for that specific matchup (see "Pick
  qualification" below for the full formula and why it actually drives picks
  now, not just a logged-alongside comparison). Also adds a "team is
  actually playing today" qualifier (an off day currently produces zero
  picks, not a stale recommendation).

Confirmed starting lineups (batting order 1-9 for *today's* game) are
explicitly out of scope for now - they post only ~2-4 hours before first
pitch, too late for the pipeline's 8am ET run regardless of data source, and
would need a second later run plus a decision about revising an
already-logged morning pick.

- **Recently-played eligibility** (`hitters.compute_last_game_dates`,
  `config.HITTER_MAX_DAYS_SINCE_LAST_GAME`, default 5 days) - a hitter's
  career-long PA total and season-long WAVE/Game_Hit_Probability rates stay
  high even after a week or more hurt/benched, so without this an inactive
  player could still surface as an official pick. `predictions.select_picks`
  excludes any hitter whose most recent completed game is more than the
  threshold before the pick date (column-gated, so old wave.csv snapshots
  without `Last_Game_Date` are unaffected). This is an eligibility/data-
  hygiene gate, not a new predictive signal - it doesn't touch ranking, only
  who's eligible to be ranked at all - so unlike the trained hit-probability
  model (still explicitly NOT wired into live picks) it didn't need its own
  backtest, the same reasoning that applies to the batting-order/lineup
  gates above. Real-data check on 2026-07-28 excluded 213 of 602 hitters
  (season-long stragglers - traded, released, injured) while correctly
  keeping actually-active players eligible.

## Pick qualification: probability vs. Game_Hit_Probability

`hitters.py` computes two independent estimates of "will this batter get a
hit in their next game": `probability` (a binomial-style model applied to
the at-bat-level WAVE rate, assuming `WAVE_TRIALS_PER_GAME` trials per game)
and `Game_Hit_Probability` (the directly observed, independently
recency-blended rate of games with >=1 hit - no binomial modeling). When
these two diverge, it's informative, not noise: a high `Game_Hit_Probability`
with a low `probability` is a batter who barely gets a hit most games (a lot
of 1-for-4/5s); a high `probability` with a low `Game_Hit_Probability` is
boom-or-bust (occasional multi-hit games mixed with a lot of 0-fors). Neither
pattern is the "reliable to get a hit today" signal a pick is supposed to
represent.

`predictions.select_picks` requires *both* columns to clear
`config.HITTER_MIN_PROBABILITY` (0.7) before a hitter qualifies as a
candidate at all, and by default ranks the qualified pool by `Approach`
(`Game_Hit_Probability * probability`, already computed in
`hitters.assemble_hitters`) rather than `Game_Hit_Probability` alone - a
`rank_metric` parameter picks *which* qualified hitters get chosen without
changing what `predicted_probability`/`metric` report (still
`Game_Hit_Probability`, so `DAILY_PICK_MIN_PROBABILITY` and the rest of the
Beat the Streak tracking logic are unaffected). This was empirically
validated, not just theorized: a 42-day git-history replay backtest (the
same technique `git_backtest.py` uses to reconstruct picks) found that
`Game_Hit_Probability`-only ranking scored a 0.284 Brier score on resolved
picks - effectively a coin flip - with the picks that actually cleared
`DAILY_PICK_MIN_PROBABILITY` hitting only 55% of the time. Adding the joint
qualifier and ranking by `Approach` improved the Brier score to 0.260 and
raised that same recommended-pick hit rate to 65%, without reducing pick
coverage (still at least one pick on all 42 backtested days). Matchup
quality (`Matchup_Hit_Probability`) still applies on top of this qualifier
when today's schedule is available - it isn't a substitute for
`probability`/`Game_Hit_Probability` both being solid, but a further
tiebreaker among hitters who already are.

`Automated Game Picks` (below) uses a much smaller resolved sample (69
backtested games as of this writing) - not enough to safely recalibrate its
`GAME_PICK_MIN_PROBABILITY` threshold or ranking without overfitting to
noise (its per-bin calibration isn't monotonic at that sample size). That
recalibration should wait until more games accumulate.

### Matchup quality: probability, Game_Hit_Probability, and the opposing pitching

A good matchup is now checked just as much as a hitter's own form, not
logged as an unused side metric. When today's probable starters are known,
`matchup.compute_matchup_hit_probability` combines a batter's own AB-level
`WAVE` rate with the opposing pitching's AB-level rate they're actually
projected to face - `PAVE`/`Bullpen_PAVE` (the real batting-average-against
scale, not the league-normalized `PAVE_PLUS` ratio `game_picks.py`'s
team-level model uses) blended by assumed at-bat share
(`MATCHUP_STARTER_AB_SHARE`/`MATCHUP_BULLPEN_AB_SHARE`) - via log5 (the
odds-ratio method, a generalization of Bill James' log5 formula for
combining two rates against a shared league baseline into the rate expected
specifically between them). The resulting matchup AB-level rate converts to
a per-game `Matchup_Hit_Probability` via the same binomial-trials formula
`probability` itself uses, so all three land on the same 0-1 scale.

`predictions.select_picks`'s joint qualifier (see above) extends
automatically to `Matchup_Hit_Probability` whenever it's present on the
table passed in - a hitter needs `probability`, `Game_Hit_Probability`,
*and* a good matchup all above `HITTER_MIN_PROBABILITY`, not just the first
two. `pipeline.run` ranks that qualified pool by `Matchup_Approach`
(`Approach * Matchup_Hit_Probability`) on days matchup data is available,
falling back to `Approach` alone (no matchup qualifier at all) if the
schedule fetch fails - the same resilience pattern as the fetch itself.
`predicted_probability`/`metric` logged still reflect `Game_Hit_Probability`
either way, so `DAILY_PICK_MIN_PROBABILITY`'s calibration is unaffected.

`WAVE`/`PAVE`/`Bullpen_PAVE` are all AB-level and reliably reproducible from
persisted Statcast at any past as-of-date, which made a 30-day backtest
possible even without git-history CSVs (raw `PAVE`/`WAVE` were never
persisted before this change). Replaying the actual `predictions.select_picks`
logic (the full joint qualifier, ranked by `Matchup_Approach`) against the
"recommended" subset that actually counts toward Beat the Streak
(rank<=2, `Game_Hit_Probability`>=0.80) raised the hit rate from 65.8%
(n=38, no matchup) to 75.0% (n=32, with matchup) - a real, meaningful
improvement over 29-30 days of history.

That backtest was only possible after fixing a real, pre-existing bug this
investigation surfaced: `data.assign_game_ids` used to reconstruct game
boundaries from scratch via an at_bat_number-reset counter, which silently
assumed one real game's rows were contiguous in the table - nothing
guaranteed that (`persist_raw_statcast` only sorts by `game_date`, and MLB
plays ~15 games in parallel most days, so their at-bat rows interleave in
whatever order Statcast/pybaseball delivered them). Confirmed on real data:
one sample batter's 96 real games were fragmented into 233 different
`game_id` values, deflating any from-scratch `Game_Hit_Probability`
reconstruction by roughly 40%. Fixed by grouping directly on Statcast's own
`game_pk` (MLB's real, already-unique game identifier - the same one
`game_picks_backtest.py` already used directly) instead of reconstructing
game identity at all; `game_id` itself stays a dense, date-ordered integer
(not raw `game_pk`, which isn't chronologically monotonic) so downstream
"latest game" logic (`teams.py`, `lineup.py`) is unaffected. Confirmed
against a real historical `wave.csv` commit: reconstructed
`Game_Hit_Probability` now matches the committed values exactly.

### Real bug fixed: PAVE excluded strikeouts from the AB denominator

Surfaced by a direct, specific complaint: a hitter's DFS matchup that day
was against Jacob Misiorowski, a dominant strikeout starter, and neither
`Matchup_Hit_Probability` nor the DFS pitcher projection treated that
matchup as tough at all. `pitchers._pave_rate` converts a hits-per-plate-
appearance rate into a hits-per-at-bat rate by excluding non-at-bat
events from the denominator - but the original formula excluded
strikeouts from that denominator ALONGSIDE walks/HBP
(`helpers.is_strikeout_walk_hbp`). A strikeout **is** an official at-bat;
only a walk or HBP isn't. Excluding it too shrinks the effective
denominator for every strikeout a pitcher records, which **inflates**
the computed hit-rate for exactly the pitchers who rack up the most
strikeouts - the more swings-and-misses a pitcher generates, the worse
the old formula made him look. Fixed by excluding only walks/HBP
(`helpers.is_non_at_bat_event`), leaving strikeouts in the AB count.

Confirmed against Misiorowski's real 2026 Statcast log (439 batters
faced: 173 K, 28 BB, 8 HBP, 61 hits): the old formula gave a full-season
PAVE of 0.265 (PAVE_PLUS ~0.93 - barely better than average); the fixed
formula gives 61/(439-28-8) = 0.151 - genuinely elite, matching his real
Cy-Young-caliber season.

**Re-backtested on the same real 20-date sample, same dates, before vs.
after the fix** (`dfs_backtest.backtest_dfs_projections` - the pitcher
side is the most directly affected, since `Expected_H_Allowed = PAVE *
Expected_IP * DFS_BATTERS_FACED_PER_INNING`):

| signal | MAE before | MAE after | naive-baseline MAE | correlation before | correlation after |
|---|---|---|---|---|---|
| `Expected_H_Allowed` vs. `Actual_H` | 2.6252 | **1.8032** | 1.8395 | 0.262 | 0.299 |
| `DK_Points_Pitcher` (combined) | 6.8840 | 6.8312 | 7.1776 | 0.327 | 0.332 |
| `DK_Points_Hitter` (heuristic) | 4.7161 | 4.7156 | 4.6879 | 0.010 | 0.010 |

`Expected_H_Allowed` is the headline result: its MAE was worse than the
naive baseline before this fix (a previously-documented, unresolved weak
signal - see "Pitchers" below) and now **beats** the baseline for the
first time, a direct, real confirmation that the bug fix - not just the
Misiorowski anecdote - measurably improved a signal this project had
already flagged as broken. `DK_Points_Pitcher`'s combined MAE improves
more modestly (it also blends the FIP-based ER estimate, which PAVE
doesn't touch). `DK_Points_Hitter`'s heuristic number is essentially
unchanged - expected, since that heuristic's correlation was already
near zero for unrelated, previously-documented reasons (the
`compute_matchup_adjustment` ratio - see "Hitters" below), and hitters'
live projection is served by the ML model, not this heuristic, anyway.

**What this fix reaches, project-wide**: every consumer of
`PAVE`/`PAVE_PLUS`/`Bullpen_PAVE` - `matchup.py`'s
`Matchup_Hit_Probability` (both the hitter matchup ratio above and its
Boom-Adjusted/Value_Score downstream uses), `game_picks.py`'s
susceptibility signal (blended against `Power_A_PLUS`, see
`GAME_PICK_SUSCEPTIBILITY_WEIGHT`), and `dfs.py`'s `Expected_H_Allowed`.
The DFS hitter/pitcher ML models (`dfs_ml.py`) were retrained after this
fix since `starter_PAVE`/`Bullpen_PAVE`/`Expected_H_Allowed` are direct
input features - see "Machine learning follow-up" below for the
retrained numbers.

**Automated Game Picks re-checked too, same 20-date sample, before vs.
after** (`game_picks_backtest.reconstruct_historical_game_picks_from_persisted`,
also a fresh recompute, not a git-history replay): accuracy actually went
from 57.9% to 56.1% (Brier 0.2470 -> 0.2486) - a small move in the WRONG
direction. On n=57 resolved games that's exactly one game's outcome
flipping, well within noise for a sample this size - not treated as a
real regression, but reported plainly rather than omitted because it
didn't confirm the hypothesis. The likely reason it barely moved at all
either way: `game_picks.py` blends `PAVE_PLUS` (a ratio re-normalized to
mean 1.0 across that day's qualified pitcher pool), not the raw PAVE
rate the DFS pitcher signals use directly - renormalization absorbs most
of a uniform formula shift, so this signal was always going to be far
less sensitive to the fix than `Expected_H_Allowed`'s raw AB-rate
calculation. `GAME_PICK_SUSCEPTIBILITY_WEIGHT` (0.5) was left unchanged;
one game on n=57 isn't grounds to recalibrate a weight that was itself
chosen from a real backtest.

**What could NOT be cleanly re-verified**: the 65.8%->75.0% Beat the
Streak hit-rate uplift reported just above, and the original
`Ceiling_DK_Points`/`Boom_Rate` capture-rate backtests, replay
git-history-committed CSV snapshots (`docs/data/wave.csv`/`confidence.csv`
at past commits) that were computed under the OLD, buggy PAVE - rewriting
that committed history to "fix" it retroactively isn't something this
project does. Those numbers stand as historically accurate for what was
live at the time; the corrected signal's real effect on live picks will
show up naturally as new daily data accumulates under the fixed formula
going forward, the same way any other model change's live impact is
observed.

### Platoon and park awareness

Two further adjustments to `Matchup_Hit_Probability`, on top of the base
batter-vs-starter blend above:

- **Platoon.** A batter's `WAVE` used to be a single hand-blended rate
  regardless of whether today's probable starter throws left- or
  right-handed - a real platoon split (common) got averaged away.
  `pitchers.compute_pitcher_throws` exposes each pitcher's own throwing
  hand (`Throws` in `pave.csv`, constant per pitcher), and
  `matchup._platoon_wave` now picks the batter's `WAVE_L`/`WAVE_R` (now
  also in `wave.csv`) to match, falling back to the blended `WAVE` when
  `Throws` is unknown (unannounced starter) or absent (old snapshots).
- **Park.** `teams.compute_park_factors` computes `Park_Factor` per team's
  home venue (each team's home games are treated as one park, since
  Statcast's pitch-level data doesn't carry a separate ballpark id) -
  combined runs/game at that venue relative to the across-all-parks
  average, the same PAVE_PLUS-style ratio convention (mean 1.0) used
  elsewhere. `matchup.py` looks up the actual venue for today's game (the
  home team, via `schedule_df`'s `is_home` - a road batter's *own* park is
  irrelevant, only the venue they're actually playing in matters) and
  scales the matchup AB rate by it, clipped to `MATCHUP_PARK_FACTOR_CLIP`
  and dialed by `MATCHUP_PARK_FACTOR_WEIGHT`.

Both were validated together via a 15-date persisted-Statcast backtest
(recomputing `wave.csv`/`pave.csv`/`confidence.csv` fresh per date, the
same technique `game_picks_backtest.py`'s persisted variant uses - neither
signal exists in old git-committed snapshots, so a git-history replay
can't see them): on the full qualified-candidate pool (n=43-52 resolved
picks per variant), platoon and park together raised the hit rate from
58% (neither) to 65% and cut the Brier score from 0.286 to 0.244 - the
best of every combination tried (platoon alone: 62%/0.261; park alone:
57%/0.293, roughly a wash by itself). See
`config.MATCHUP_PARK_FACTOR_WEIGHT`'s docstring for the full numbers,
including the smaller `rank<=2` "recommended" subset (n=11-18 - too small
to trust on its own, and noisier in the opposite direction on this
particular window). Revisit once more dates accumulate.

### Real batted-ball-quality signal (`hitters.compute_quality_of_contact`)

Every windowed hitter signal up to this point (`WAVE`, `Game_Hit_Probability`,
`Consistency`, `Approach`, the extended DK rates) is an outcome-RATE
statistic - hit/walk/RBI rates blended over recent games. None of them
measure contact QUALITY directly: how hard/well the ball was actually hit,
independent of whether it happened to find a fielder. The raw Statcast
columns for that (`launch_speed`, `estimated_ba_using_speedangle`,
`estimated_woba_using_speedangle`, `launch_speed_angle`) were already
present on every real Statcast row, just never used.

`compute_quality_of_contact(dt)` computes real recency-windowed
`Exit_Velo` (mean `launch_speed`), `Barrel_Rate` (share of batted balls
with `launch_speed_angle == 6` - Statcast's own real "barrel" bucket, not
an approximation reconstructed from the raw formula), `xBA` (mean
`estimated_ba_using_speedangle`), and `xwOBA` (mean
`estimated_woba_using_speedangle`), computed over real batted-ball rows
only (`helpers.is_batted_ball` filters to `type == "X"` - a strikeout/
walk/HBP has no batted-ball data and would otherwise silently bias every
rate toward 0/NaN). Blended by throws-side using the exact same
`_blend_windows`/`_side_window_agg` machinery `WAVE`/`WTB`/the extended DK
rates already use, reusing `config.WHOPS_WTB_WINDOWS`. Merged into
`wave.csv` and into `dfs_ml.HITTER_FEATURE_COLUMNS` (both the DK-points
regressor and the hit-probability classifier), so both models can learn
from contact quality directly instead of only ever seeing outcome rates.

**A real production bug found via a live retrain, not a synthetic test**:
`helpers.is_barrel` originally did `(launch_speed_angle == 6).astype(int)`.
Every synthetic test fixture supplied clean, non-null values, so `pytest`
passed cleanly (541 tests) - but the real persisted
`data/raw/statcast_2026.parquet`'s `launch_speed_angle` column has genuine
null values on some real tracked batted balls, and comparing a pandas
nullable-dtype column via `==` against those null rows returns `pd.NA`
(not `False`, per three-valued comparison logic), which `.astype(int)`
can't cast - `scripts/train_dfs_ml_models.py` crashed for real on the
first retrain attempt (GitHub Actions run 32280494858). Fixed with
`.fillna(False)` before casting, so "no tracked quality bucket" reads as
"not a barrel" - a real, honest 0, matching every other classifier's
missing-data convention in `helpers.py`. The retrain that followed the
fix succeeded end to end (run 32281354568) - see the significance report
above and the DFS ML retrain table below for what these four columns
actually contributed once real data was behind them.

### Real pitch-type-specific platoon matchup (`pitchers.compute_pitch_arsenal`, `hitters.compute_pitch_family_rates`, `matchup._pitch_arsenal_multiplier`)

A pitcher's real fastball/breaking/offspeed pitch mix (Statcast's own
`pitch_type` column) was sitting unused in the same already-persisted raw
data as the batted-ball-quality columns above - never used as a matchup
signal beyond the existing handedness platoon adjustment.
`helpers.pitch_type_family` groups every real `pitch_type` code into the
same three-bucket scheme Baseball Savant's own pitch-arsenal pages use.
`pitchers.compute_pitch_arsenal` computes a pitcher's real windowed USAGE
mix (what share of their actual pitches are fastball/breaking/offspeed -
needs every real pitch thrown, not just PA-ending ones, so it's the one
function in this project built on `pipeline.build_all_pitch_events` rather
than `data.completed_events`). `hitters.compute_pitch_family_rates`
computes the batter-side complement: WAVE split by the PA-ending pitch's
family instead of by pitcher handedness. Both are merged into
`dfs_ml.HITTER_FEATURE_COLUMNS` (`Fastball_WAVE`/`Breaking_WAVE`/
`Offspeed_WAVE` for the batter, `starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` for today's probable
starter) so the ML models can learn from this matchup directly, and
`matchup._pitch_arsenal_multiplier` blends them into `Matchup_Hit_Probability`
the same clip-then-weight-dial way `_park_factor_multiplier` does - ships
at `config.MATCHUP_PITCH_ARSENAL_WEIGHT = 0.0` (informational-only, exact
no-op) until a real historical-reconstruction backtest earns a nonzero
default, same "ship conservatively" precedent `PITCHER_MATCHUP_OFFENSE_WEIGHT`
already established - that backtest is a genuinely separate undertaking
from the ML-feature validation below and hasn't been run yet.

**Two real production bugs found via live retrains, not synthetic
tests** (both GitHub Actions runs against real persisted data, neither
caught by the 559-560 passing synthetic-fixture tests at the time):
1. `scripts/train_dfs_ml_models.py` crashed with a `TypeError` deep
   inside `matchup.compute_matchup_hit_probability`'s final `.clip(0, 1)`
   (run 32288246152). Root cause: `_pitch_arsenal_multiplier`'s
   `vs_league.replace(0, pd.NA)` silently upcast a float64 Series to
   `object` dtype - every arithmetic op downstream then ran through
   Python's own operators instead of numpy's, and on a real row
   `(1 - matchup_ab_rate) ** 3.5` produced a genuine Python `complex`
   value (Python's `**` does that for a negative base and a non-integer
   exponent; numpy's vectorized power would have just given `nan`) -
   `.clip()` then couldn't compare a complex number against an int.
   Fixed by using plain float division instead (`vs_starter`/`vs_league`
   can only ever be a genuine 0/0, never x/0, since both are weighted
   sums of the exact same `Fastball_WAVE`/`Breaking_WAVE`/`Offspeed_WAVE`
   terms) - confirmed against the exact failing case afterward: a real
   float64 `1.0`, never an object-dtype value.
2. (See the batted-ball-quality section above for the earlier, separate
   `is_barrel`/`pd.NA`-vs-`np.nan` bug this project already hit once -
   the SAME underlying class of bug (a pandas nullable/NA value leaking
   into a numeric computation and breaking a downstream cast or op)
   resurfaced in genuinely different code here, which is itself worth
   noting: `pd.NA` in an otherwise-float computation is a real, recurring
   footgun in this codebase, not a one-off mistake.

**Real retrain results** (GitHub Actions run 32289285787, after both
fixes): individually significant (p<0.0001 each) - real signal, not
noise, same pattern the batted-ball-quality columns showed. Not
significant in the combined multivariate model (collinear with existing
WAVE/Approach signal - see the significance report above for the exact
numbers) and the starter's own `starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` were never significant
either way. The full walk-forward hitter hit-probability model (all 28
features together) came out essentially unchanged from the immediately
prior run - adding these six columns didn't move the holdout numbers by
a meaningful margin, consistent with the significance report's finding
that they're collinear rather than independently predictive. Reported
honestly: this slice's real value so far is the two production bugs it
surfaced and fixed, not a measurable accuracy gain - the
`MATCHUP_PITCH_ARSENAL_WEIGHT` matchup multiplier remains unvalidated
and off (0.0) pending a real backtest.

### Real plate-discipline signal (`helpers.is_swing`/`is_whiff`/`is_out_of_zone`/`is_chase`, `hitters.compute_plate_discipline`)

The last raw signal this item set out to add: every pitch a hitter sees or
takes carries real swing/contact/zone information in Statcast's own
`description` and `zone` columns, already sitting unused in the same
persisted raw data the batted-ball-quality and pitch-arsenal work above
already tapped. `helpers.is_swing` classifies real Statcast `description`
codes into swing vs. take (bunts counted as swings - a documented
simplification, since bunts are ~0.25% of real pitches); `is_whiff`
classifies genuine swinging strikes (explicitly excluding `foul_tip`,
which Statcast scores as real contact, not a miss); `is_out_of_zone` uses
Statcast's own real `zone` 11-14 "chase quadrant" codes (its own
classification of the area just outside the 3x3 strike-zone grid, not
something derived from `plate_x`/`plate_z`/`sz_top`/`sz_bot`); `is_chase`
is the conjunction of the two. `pipeline.build_all_pitch_events` (already
built for the pitch-arsenal work above) was widened to carry
`description`/`zone` alongside `pitch_type`, and its blanket
`pitch_type`-null pre-filter was removed - `description`/`zone` are
populated on some rows where `pitch_type` is null, so a pre-filter keyed
only on `pitch_type` would have silently dropped real plate-discipline
data. `hitters.compute_plate_discipline` blends per-window `Whiff_Rate`
(swinging-strike rate on swings) and `Chase_Rate` (chase rate on
out-of-zone pitches) using the same rate-then-blend pattern every other
windowed signal in this codebase follows (`WAVE`, PAVE,
`compute_quality_of_contact`, `compute_pitch_arsenal`) - compute the RATE
per window first, then blend the windows' rates by weight, reusing
`config.WAVE_WINDOWS`. Merged into `wave.csv` and
`dfs_ml.HITTER_FEATURE_COLUMNS`.

**Real retrain results** (GitHub Actions run 32304349888, 2026-08-19 -
first clean run this item's three slices have had, no new production bug
surfaced): `Whiff_Rate` is significant both individually (coef -0.0980,
p<0.0001) and in the combined multivariate model (coef -0.1011,
p<0.0001) - the expected direction (a hitter who whiffs more often on
swings is less likely to get a hit) and, unlike most of the
batted-ball-quality/pitch-arsenal columns, it SURVIVES the multivariate
fit rather than collapsing to collinearity with existing WAVE/Approach
signal. `Chase_Rate` is only marginal individually (coef 0.0197,
p=0.0785) but reaches significance in the combined model (coef 0.0380,
p=0.0073) - both with a positive sign, which is counterintuitive (naively,
chasing more should predict fewer hits, not more) and worth flagging
honestly rather than glossing over; a plausible read is that `Chase_Rate`
is partly standing in for aggressive, high-contact-skill hitters once
`Whiff_Rate` is already in the model, but this hasn't been separately
tested. The full walk-forward hitter hit-probability model (now 30
features) improved slightly on this run: holdout log_loss=0.6802 (vs.
0.6817 the immediately prior run), ROC AUC=0.5726 (vs. 0.5679) - both
still beating naive-baseline (log_loss 0.6843) and the
`Game_Hit_Probability` heuristic (log_loss 0.7003, ROC AUC 0.5527) by a
real if modest margin. This is the first of item #1's three slices where
the new columns visibly moved the holdout numbers in the model's favor
rather than landing flat - a genuinely useful addition, not just informational.

### Hitter hit log (`data/predictions/hitter_hit_log.csv`, data asset - not yet a live signal)

Investigating why Beat the Streak's day-survival rate sits at ~50% found a
real gap: the only signal with enough logged pick history to test
(`Game_Hit_Probability`) showed no statistically significant relationship
with actual hit outcomes, and the sample was tiny (n=64 resolved picks) -
`predictions.csv` only logs the ~2 hitters/day that ever became an
official pick, not the full slate. A logistic regression fit against that
few, gated rows can't give trustworthy coefficients or significance
levels.

`dfs_backtest.assemble_hitter_hit_log` (run daily via
`scripts/build_hitter_hit_log.py`) builds the real training table instead:
one row per hitter per game, for **every** hitter with a game that date
(not just the ones that cleared the pick gates), with every feature
computed strictly before that game - `starter_PAVE`, `Bullpen_PAVE`,
`WAVE`/`WAVE_L`/`WAVE_R`, `Total_PA` (`PA_L + PA_R`),
`Game_Hit_Probability`, `Matchup_Hit_Probability`, `Park_Factor`, and the
rest of `dfs_ml.HITTER_FEATURE_COLUMNS` - plus a binary `Got_Hit` label
(did that batter record at least one hit that date, from real completed
Statcast events). Same no-lookahead recompute (`dfs_backtest._compute_date_outputs`)
and the same feature-building function (`dfs_ml.build_hitter_features`)
already used and tested for the DFS ML models, so this reuses trusted
machinery rather than hand-rolling a second merge path.

`scripts/build_hitter_hit_log.py` (no `--days` for a one-time full
historical backfill; `--days 10` for the daily incremental run
`daily_update.yml` uses) appends and dedupes on `(date, key_mlbam)` -
keep-last, so a freshly recomputed row always wins over a stale one and
re-running never creates duplicate rows.

This log is **purely a data asset today** - it feeds nothing live. Fitting
the actual logistic regression against it (and deciding whether it ever
replaces or augments `predictions.select_picks`'s probability gate) is a
follow-up once the log has accumulated real history.

### Fitting the predictive model: logistic regression vs. gradient boosting (`scripts/train_hitter_hit_model.py`)

Real answers to the "why is survival only ~50%, and which features are
actually significant" question, now that the hit log above fixes the
sample-size problem (n=26,955 hitter-games, PA-qualified, vs. the n=64
resolved picks the earlier ad hoc analysis had to work with). Two
deliverables from one script, both fit on rows filtered to
`Total_PA >= config.BACKTEST_MIN_PLATE_APPEARANCES` (the same gate
`predictions.select_picks` already applies before a hitter can be a pick
candidate - a handful of career plate appearances gives a mostly-noise
`WAVE`/`Game_Hit_Probability`, and letting that noise into the fit would
just drag down real coefficients):

**Significance report** (`statsmodels.Logit`, full history, real numbers
from 2026-08-19, after adding the batted-ball-quality columns, the
pitch-type-family columns, and the plate-discipline columns below):
individually (one univariate model per feature), `WAVE`, `WAVE_L`/`WAVE_R`,
`PA_L`/`PA_R`, `probability`, `Game_Hit_Probability`, `Consistency`,
`Approach`, `Expected_Bases`, `Expected_RBI`, `Exit_Velo`, `Barrel_Rate`,
`xBA`, `xwOBA`, `Fastball_WAVE`, `Breaking_WAVE`, `Offspeed_WAVE`,
`Whiff_Rate`, `Park_Factor`, and `Matchup_Hit_Probability` are all
significant (p<0.001) - notably including `Game_Hit_Probability` itself
(coef 0.2153, p<0.0001), the OPPOSITE of the much-earlier n=64 finding.
That earlier result wasn't wrong given its data - it was underpowered; a
64-pick sample simply can't reliably detect an effect of this size.
`Chase_Rate` is only marginal individually (coef 0.0197, p=0.0785).
`Expected_BB`, `Expected_HBP`, `starter_PAVE`, `Bullpen_PAVE`, `is_home`,
and `starter_fastball_rate`/`starter_breaking_rate`/`starter_offspeed_rate`
are not individually significant. In the combined (multivariate) model,
`probability`, `Game_Hit_Probability`, and `Consistency` show enormous
standard errors (~1,000,000+) and are unusable there - real, expected
multicollinearity, since `Consistency` (`Game_Hit_Probability -
probability`) and `Approach` (`Game_Hit_Probability * probability`) are
algebraically derived from those same two columns. `WAVE_L`, `WAVE_R`,
`PA_R`, `Park_Factor`, `Whiff_Rate` (coef -0.1011, p<0.0001), and
`Chase_Rate` (coef 0.0380, p=0.0073) remain significant once the others
are controlled for - `Whiff_Rate` in the expected direction (more whiffs,
fewer hits) and surviving the multivariate fit unlike most of the other
new columns; `Chase_Rate` significant but with a counterintuitive positive
sign, reported honestly rather than glossed over (see the plate-discipline
subsection above for a plausible read). **Reported honestly**:
`Exit_Velo`/`Barrel_Rate`/`xBA`/`xwOBA` and `Fastball_WAVE`/`Breaking_WAVE`/
`Offspeed_WAVE` are each individually significant on their own (real
signal, not noise), but none remain significant in the combined model
(p=0.48/0.49/0.93/0.45 for the first four, p=0.80/0.71/0.31 for the
pitch-family three) - they're substantially collinear with the existing
WAVE/Approach-family columns (a hitter who makes consistently hard
contact, or reliably ends ABs on fastballs successfully, also tends to
have a high recent WAVE), so this simple linear specification can't
cleanly credit them with independent lift once those columns are already
in the model. The starter's own `starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` never reach significance
either way (individually or combined, p=0.31-0.98) - a genuine, useful
finding on its own, not a failure to ship - see the walk-forward model
result just below, which reflects all 30 columns together.

**Walk-forward-validated predictive model** (`ml_models.py`'s
`WalkForwardDateSplit`/nested-holdout machinery, same discipline as the
three live DFS ML models): quant-analytics item #2 ("model family") -
until this run, this was a single `sklearn.LogisticRegression` with no
real alternative ever tried. `scripts/train_hitter_hit_model.py` now
grid-searches TWO candidates on the same walk-forward CV -
`LogisticRegression` (`config.HITTER_HIT_LOGIT_C_GRID`) and
`HistGradientBoostingClassifier` (new `config.HITTER_HIT_GBM_PARAM_GRID`,
reusing `DFS_HITTER_GBM_PARAM_GRID`'s exact values) - and keeps whichever
wins by walk-forward CV `neg_log_loss`, mirroring
`train_dfs_ml_models.py`'s own Ridge-vs-GBM selection for
`DK_Points_Hitter`.

**Real run (2026-08-20, n=5,186 holdout, 30 features)**:
`LogisticRegression` `best_params={'C': 0.3}` `cv_score=-0.6754`;
`HistGradientBoostingClassifier`
`best_params={'learning_rate': 0.03, 'max_depth': 3, 'max_iter': 100,
'min_samples_leaf': 200}` `cv_score=-0.6747` - GBM wins the CV comparison
and is selected. On the untouched holdout: log_loss 0.6795 vs.
naive-baseline 0.6843 vs. the `Game_Hit_Probability` heuristic's 0.7003 -
beats both bars, saved. **Reported honestly, not a clean win**: compared
against the immediately-prior LogisticRegression-only run on the same
holdout window (log_loss 0.6802, ROC AUC 0.5726 - see the plate-discipline
section above), GBM's log_loss is marginally better (0.6795 vs. 0.6802)
but its ROC AUC is actually WORSE (0.5654 vs. 0.5726) and its accuracy is
essentially flat (0.5669 vs. 0.5673). Two different proper-scoring metrics
moved in opposite directions on the same holdout - a real, mixed result on
this modest-sized holdout, not a decisive verdict that gradient boosting
beats logistic regression here. Neither of these two runs' numbers is
directly comparable to the original 2026-07-29 run (log_loss
0.6757/0.6815/0.6901, ROC AUC 0.575/0.564) - the holdout window itself
moved forward with the season across all these runs, not just the model/
feature set. Saved to `config.HITTER_HIT_PROBABILITY_MODEL_PATH`
(`data/models/hitter_hit_probability_model.joblib`); `HITTER_MODEL_VERSION`
is unchanged - it tracks the selection/gating logic (v3/v4), not which
sklearn class produces `Model_Hit_Probability`, and a model-family swap
under the same artifact schema isn't a version-worthy event under this
project's own convention.

**Refreshed after shipping shrinkage** (GitHub Actions run 32424947608,
2026-08-20, immediately after `WAVE_SHRINKAGE_STRENGTH`/
`GAME_HIT_PROB_SHRINKAGE_STRENGTH` shipped nonzero above): GBM still wins
the CV comparison (`cv_score=-0.6745` vs. Logit's `-0.6755`) and this time
it's a clean win on the holdout too, not the mixed result the immediately
prior run reported - log_loss 0.6773 (vs. 0.6795 before) and ROC AUC
0.5723 (vs. 0.5654 before, now beating the `Game_Hit_Probability`
heuristic's own 0.6797/0.5639 on both metrics). Not a controlled
before/after comparison (the holdout window also moved forward with the
season), but a genuinely healthier-looking result than the last run, and
one real data point that shrinkage's shipped defaults didn't hurt the
downstream model.

### Calibrating the raw model output: isotonic/Platt (`ml_models.fit_calibrated`)

Quant-analytics item #3 ("uncertainty quantification"), slice 2 of 3
(shrinkage → calibration → confidence intervals). `Model_Hit_Probability`
is the one genuine ML classifier score in this project - unlike
`probability`/`Game_Hit_Probability`/`Matchup_Hit_Probability` (all
hand-derived real-rate formulas), a classifier's raw `predict_proba`
output has no guarantee of being well-calibrated (GBMs especially are
known to run over/underconfident). `ml_models.fit_calibrated` wraps the
walk-forward-selected model in `sklearn.calibration.CalibratedClassifierCV`
(isotonic or sigmoid/Platt), refit through the SAME no-lookahead
`WalkForwardDateSplit` every other walk-forward fit in this project
already uses. `train_hitter_hit_model.py` now scores uncalibrated vs.
isotonic vs. sigmoid on the untouched holdout and keeps whichever wins by
**Brier score** (calibration's own proper-scoring metric) - reported
honestly either direction.

**Real run** (GitHub Actions run 32448967312, 2026-08-21, same holdout as
the "Refreshed after shipping shrinkage" numbers above):

| candidate | log_loss | brier | roc_auc | accuracy |
|---|---|---|---|---|
| uncalibrated | 0.6764 | **0.2417** | 0.5770 | 0.5733 |
| isotonic | 0.6769 | 0.2420 | 0.5728 | 0.5743 |
| sigmoid (Platt) | 0.6764 | **0.2417** | 0.5749 | 0.5733 |

**A real, honest NO-GO**: `uncalibrated` wins the Brier-score comparison
(tied with sigmoid, both beating isotonic) - this GBM's raw probabilities
were already reasonably calibrated on this holdout, so calibration didn't
earn its keep here. The shipped artifact is the plain uncalibrated model,
identical in kind to every prior run. Isotonic and sigmoid calibration
are each an ENSEMBLE of separately-fit per-fold models (not a single
monotonic transform of the one uncalibrated estimator's own scores), so
their ROC AUC isn't expected to match the uncalibrated model's exactly
even when calibration doesn't help - the small AUC spread above (0.5728-
0.5770) is real and expected, not a bug. `ml_models.fit_calibrated`
itself stays in place and gets tried on every future retrain - a
different holdout window or feature set could still earn a calibrated
win later, and this comparison now runs automatically every time.

**Permutation-importance report** (`sklearn.inspection.permutation_importance`,
same untouched holdout, chosen over SHAP specifically to avoid adding a
new heavy dependency to a project that has never needed one beyond
scikit-learn/statsmodels): a real, model-agnostic sanity check on what the
winning GBM is actually learning from, computed independently of the
significance report's own Logit coefficients above. **Refreshed alongside
the calibration comparison above** (same GitHub Actions run 32448967312):
top real features by mean importance: `Game_Hit_Probability` (0.00361),
`Consistency` (0.00254), `PA_L` (0.00091), `Whiff_Rate` (0.00085),
`Park_Factor` (0.00063), `PA_R` (0.00036), `Offspeed_WAVE` (0.00014),
`Chase_Rate` (0.00013), `Fastball_WAVE` (0.00012), `Bullpen_PAVE`
(0.00011), `xBA` (0.00010) - every other feature's importance rounds to
~0.0000-0.00002, several slightly negative (expected sampling noise for a
genuinely uninformative feature, not evidence of active harm). **A real,
honest divergence worth flagging**: `Whiff_Rate` ranking 4th here roughly
matches its own significance in the Logit above, but
`Matchup_Hit_Probability` (highly significant there, coef 0.0862
individually / 0.0539 combined, p<0.0001/p=0.0174) ranks near the very
bottom here (0.00005). GBM's tree splits absorb correlated signal
differently than a linear model's coefficients do - this isn't evidence
either method is wrong, just that "which features matter" genuinely
depends on which model family is
asking, a real quant lesson this comparison surfaces rather than papering
over.

The model shortlists candidates for the official Beat the Streak picks
(`predictions.select_picks`, see the next section) rather than ranking
the whole pool directly - see `HITTER_MODEL_VERSION`'s `"v4-model-shortlist"`
docstring and the "Wiring the model into Our Picks" section below for why.
`.github/workflows/ml_training_update.yml`'s weekly retrain job now
includes `scripts/train_hitter_hit_model.py` alongside the three DFS/
age-curve models, so this artifact stays current instead of being frozen
at its original training date.

### Bayesian shrinkage for small-sample hitters (`helpers.shrink_rate`, `hitters.compute_wave`/`compute_game_hit_probability`)

Quant-analytics item #3 ("uncertainty quantification"), slice 1 of 3
(shrinkage → calibration → confidence intervals). Before this slice, a
15-PA rookie and a 500-PA veteran ran through the exact same `count / n`
division inside `compute_wave` and `compute_game_hit_probability` - the
only sample-size protection anywhere was a hard external gate
(`config.BACKTEST_MIN_PLATE_APPEARANCES`) that fully excludes a player
below the line and fully includes one above it, with nothing in between.
`helpers.shrink_rate` adds real Beta-Binomial empirical-Bayes shrinkage:
`(count + prior_strength * prior_rate) / (n + prior_strength)`, where
`prior_strength` is a real-unit pseudo-observation count (at-bats for
`compute_wave`, games for `compute_game_hit_probability`) - not an
abstract 0-1 weight - pulling a low-`n` player's rate toward a real
league-average rate (computed from the same already-loaded data, no new
fetch, no lookahead) while barely moving a high-`n` veteran's. As with
every other tunable weight in this codebase, `prior_strength=0.0` is the
exact null hypothesis - byte-for-byte identical to the unshrunk rate, not
an approximation of it - so both constants shipped at `0.0` until a real
backtest earned a nonzero default.

**Real backtest results** (`scripts/backtest_shrinkage.py`, dispatched via
`.github/workflows/debug_backtest_shrinkage.yml` against the full
persisted 2026 season - GitHub Actions run 32415595512, 2026-08-20):
swept `{0, 25, 50}` for each constant independently through the real
no-lookahead historical reconstruction (`dfs_backtest.assemble_hitter_hit_log`),
scored against real `Got_Hit` outcomes.

| strength | WAVE full-pop log_loss (n=37,744) | WAVE PA-gated log_loss (n=33,035) | Game_Hit_Probability full-pop log_loss | Game_Hit_Probability PA-gated log_loss |
|---|---|---|---|---|
| 0 (unshrunk) | 0.9468 | 0.6908 | 1.2024 | 0.6983 |
| 25 | 0.6831 | 0.6805 | 0.6776 | **0.6752** |
| 50 | 0.6821 | **0.6798** | 0.6786 | 0.6762 |

(Game_Hit_Probability's own best PA-gated value was strength=25 at
0.6752, edging out strength=50's 0.6762.) The **full unfiltered
population's** log_loss at `strength=0` is dramatically worse than the
PA-gated subset (0.9468 vs. 0.6908 for WAVE; 1.2024 vs. 0.6983 for
Game_Hit_Probability) - real evidence of exactly the problem this slice
targets, since that population includes the small-sample rows a hard PA
gate excludes from live picks today but that an unshrunk rate still
handles badly. On the PA-gated population - the real go/no-go bar, since
that's what live Beat the Streak picks are actually exposed to -
shrinkage was a clean win for both signals, so:

- `config.WAVE_SHRINKAGE_STRENGTH` shipped at **50.0** (log_loss 0.6798 vs.
  0.6908 unshrunk).
- `config.GAME_HIT_PROB_SHRINKAGE_STRENGTH` shipped at **25.0** (log_loss
  0.6752 vs. 0.6983 unshrunk).

`config.BACKTEST_MIN_PLATE_APPEARANCES`'s hard gate stays in place
unremoved - it still serves an "insufficient data to trust even a shrunk
estimate at all" role that shrinkage alone doesn't replace. Since these
are real, earned nonzero defaults that change `WAVE`/`Game_Hit_Probability`
values (and everything downstream: `Approach`, `Consistency`,
`Matchup_Hit_Probability`, `Model_Hit_Probability`), the hitter
hit-probability model was retrained afterward via the normal
`ml_training_update.yml` weekly job so its training features reflect the
shrunk values.

### Confidence intervals on the full-season rate (`helpers.wilson_ci`)

Quant-analytics item #3 ("uncertainty quantification"), slice 3 of 3 -
the last piece, after shrinkage and calibration above. Every probability
this site produces was a bare point estimate with no interval anywhere.

The real complication: `WAVE`/`probability` and `Game_Hit_Probability`
are each a blend of 4 **overlapping, nested** recency windows
(`config.WAVE_WINDOWS`/`GAME_HIT_PROB_WINDOWS` - the 10-day window's
at-bats are a literal subset of the 30-day window's, which are a subset
of the 81-day window's, which are a subset of the full-season window's).
Those windows aren't independent samples, so there's no valid
closed-form confidence interval for the *blended* number itself. But
each blend's **full-season component** - real hits over real at-bats, or
real hit-games over real total-games, pooled across the whole season
with no recency weighting - genuinely is a single, well-defined binomial
proportion, with an exact, standard interval formula: the Wilson score
interval (`helpers.wilson_ci`, reusing
`statsmodels.stats.proportion.proportion_confint(method="wilson")` -
statsmodels is already a core project dependency, used elsewhere for
`Logit` significance reports, so this adds no new dependency).

**Deliberately excluded** (confirmed via `AskUserQuestion` rather than
fabricating an approximate interval for either): `Approach`
(`Game_Hit_Probability * probability`, a product of two already-blended
heuristics with no clean n) and `Consistency` (their difference);
`Matchup_Hit_Probability` (log5-combined with two multiplicative
adjustments - no clean n survives into it either); and
`Model_Hit_Probability` (the ML classifier's raw score - would need
bootstrap or conformal prediction, a separate, larger undertaking). Same
"don't ship fake precision" discipline slice 2 already followed when
calibration didn't earn a win. Only the combined (both-hands pooled)
full-season rate gets an interval - no separate `WAVE_L`/`WAVE_R`-side
intervals, keeping the new columns minimal.

**Computed on the RAW empirical count/n, never the shrunk point
estimate.** A confidence interval describes the sampling uncertainty of
the empirical estimator itself; `helpers.shrink_rate`'s shrunk output
(above) is a Bayesian point-estimate correction toward a league prior -
a complementary, not competing, treatment of the same small-sample
problem. `WAVE_CI_Low`/`WAVE_CI_High` and
`Game_Hit_Probability_CI_Low`/`_CI_High` are identical no matter what
`WAVE_SHRINKAGE_STRENGTH`/`GAME_HIT_PROB_SHRINKAGE_STRENGTH` are set to.
`probability_CI_Low`/`_CI_High` are the same `WAVE` interval pushed
through the existing `1 - (1-rate)**WAVE_TRIALS_PER_GAME` transform -
valid because that transform is strictly increasing over `[0, 1]`, so
transforming a valid interval's endpoints gives a valid interval for the
transformed quantity too, not an approximation.

Purely additive: `dfs_ml.HITTER_FEATURE_COLUMNS` is an explicit
allow-list (`hitter_feature_matrix`'s `.reindex(columns=...)`), so the
new columns don't touch the trained hitter hit-probability model at all -
**no retrain needed**, unlike shrinkage and calibration above. On the
dashboard, `WAVE`/`probability`/`Game_Hit_Probability` had no dedicated,
by-name display anywhere in `docs/app.js` - they only ever reached the
page through the generic "Top WAVE Players" table (`buildTable`), which
rendered every column as a raw, unformatted number. `buildTable` now
folds any `<X>_CI_Low`/`<X>_CI_High` pair into their point column `X`'s
own cell (`"28.4% (22.1%-35.2%)"`, the same `%`-formatting idiom used
everywhere else on the dashboard) instead of three separate raw columns -
a data-driven, forward-compatible widening (a no-op for any table with
no CI columns, e.g. Top PAVE Players; any future column following this
same naming convention gets folded in automatically).

This completes quant-analytics item #3 ("uncertainty quantification") in
full: Bayesian shrinkage for small-sample hitters, isotonic/Platt
calibration of the raw model output, and confidence intervals on the
full-season rate.

### Wiring the model into Our Picks (`pipeline.py`, `predictions.py`)

`pipeline.run()` resolves `predictions.select_picks`'s `rank_metric`
through a two-tier fallback, computed once per day: `Matchup_Approach`
(schedule/matchup data available) or `Approach` (no schedule at all).
Separately - independent of `rank_metric` - whenever the model artifact
loads and produces a real prediction for today's schedule
(`dfs_ml.predict_hitter_hit_probability` returns non-empty), its
`Model_Hit_Probability` column gets merged onto the pick pool and
`select_picks` uses that column's mere presence to narrow the
already-qualified pool down to a broad **shortlist** - its top
`config.HITTER_MODEL_SHORTLIST_SIZE` (10) candidates by
`Model_Hit_Probability` - BEFORE `rank_metric` ranks among that shortlist
and picks the final `top_n`. The model is a gate, not the ranker.

**This is a reversal of an earlier design (v3, `config.HITTER_MODEL_VERSION`)**
that let `Model_Hit_Probability` rank the WHOLE qualified pool directly,
gated on a probability threshold (`HITTER_MIN_MODEL_PROBABILITY`, since
removed). Real feedback after v3 shipped live (2026-08-05): a day it
surfaced a single hitter as the lone recommended pick and dropped another
hitter the user explicitly wanted, "because of their place in the
lineup" - a signal `Approach`/`Matchup_Approach` implicitly captures via
the `avg_batting_order`/`start_rate` qualifiers (a hitter who bats high
in an everyday lineup) that `Model_Hit_Probability` doesn't see directly
(it's not one of `dfs_ml.HITTER_FEATURE_COLUMNS`). The current design
(v4, `"v4-model-shortlist"`) keeps the model's original purpose - killing
pure hot-streak outliers by requiring a real matchup-aware model score at
all - while handing the final call back to the heuristic signal the model
doesn't capture. It also sidesteps `HITTER_MIN_MODEL_PROBABILITY`'s
calibration problem entirely: a rank-based cutoff needs no probability
threshold to derive or backtest.

`config.HITTER_MODEL_SHORTLIST_SIZE = 10` is an explicit user-specified
value, not backtest-derived like almost everything else in this file -
`scripts/backtest_selection_rule.py` (extended for this design, comparing
`heuristic_only` vs. `model_shortlist` variants head to head) should still
validate or inform retuning it, reported honestly either way, once that
real run is unblocked (see the script's own docstring for the sandbox
network limitation blocking it so far).

`predicted_probability`/`probability`/`Matchup_Hit_Probability`/
`Model_Hit_Probability` are all still logged to `predictions.csv` on
every pick regardless of whether the shortlist engaged that day, so
downstream evaluation/backtesting can always see the full picture.

Deliberately NOT bundled into this change (still, in v4 as in v3):
`evaluation.py`'s separate "recommended" gate
(`_combined_probability`/`config.DAILY_PICK_MIN_PROBABILITY`, which
decides which of the top-ranked candidates become the 0-2 picks shown as
"Our Picks" on the dashboard) still runs unchanged on top of whatever
`select_picks` returns - a deliberate fast-follow, not part of this
change (landing two new uncalibrated thresholds in one change would make
a good or bad outcome hard to attribute to either).

### Decision theory: streak-aware optimal stopping (`decision_theory.py`)

Quant-analytics item #4 ("decision theory for the actual game
structure"), slice 1 of 2 (a real correlation/joint-risk treatment
between same-day picks is the deferred other half). Every signal above
answers "how likely is a hit today" - none of them ever asked the
genuinely different question Beat the Streak's own rules pose: **given
the current streak, is today's best pick good enough to risk it, or is
sitting out worth more than playing a marginal day?**
`config.DAILY_PICK_MIN_PROBABILITY` (0.77) is a single fixed bar applied
identically every day, with zero awareness of streak length or how many
days remain - `predictions.select_picks` makes each day's decision
completely fresh. A real multi-day streak simulator already existed
(`evaluation.streak_progression`), but purely retrospectively - it scores
history after the fact, with no feedback into the day's own decision.

This is a genuine finite-horizon **optimal-stopping / dynamic-programming
problem** - the same "reservation threshold" shape as the classic
asset-selling/secretary problem - not something to approximate with a
hand-tuned heuristic. `decision_theory.solve_reservation_thresholds`
solves it exactly via backward induction: state = current streak `s`;
each of `horizon` remaining days has a real probability `p` (drawn from
the actual empirical sample of historical daily probabilities, not a
fitted distribution) of extending the streak; terminal utility (season
end) = the streak length itself (a deliberately simple, monotonic
utility - no discrete MLB.com prize-threshold modeling this slice, per
explicit choice). The Bellman recursion yields a closed-form decision
rule: PLAY iff `p > (V(s) - V(0)) / (V(s+gain) - V(0))` - a reservation
threshold that's provably non-decreasing in `s` (a longer streak needs a
better pick to risk it) and non-increasing as the horizon shrinks (less
time left to recover a reset lowers today's opportunity cost of sitting
out - a real, slightly counterintuitive result, not glossed over).
`gain` (picks-that-hit on a real non-reset day) is itself estimated from
real history (`decision_theory.estimate_gain`), not assumed to always
equal `config.DAILY_PICK_MAX`.

**Real backtest** (`scripts/backtest_streak_decision.py` - unlike every
prior slice, this needs no GitHub Actions dispatch at all:
`data/predictions/predictions.csv` is already checked into the repo and
readable with zero network dependency): loaded 71 real resolved days,
real empirical `gain=1.3125`. Solving the DP against this real history
and replaying it two ways - today's static 0.77 rule vs. the DP rule -
plus a bootstrap (1,000 resamples of the real historical `(p, outcome)`
day-pairs, reshuffling day order - consistent with the DP's own stated
days-are-i.i.d. simplification):

| | single real path | bootstrap mean final streak |
|---|---|---|
| Status quo (p&gt;=0.77) | final=4, longest=8, played 55/71 days | 1.258 |
| DP rule (streak/horizon-aware) | final=6, longest=8, played 71/71 days | 1.800 |

DP beat the status quo in 228/1,000 bootstrap resamples vs. 53 losses
(719 ties) - a real, if modest, edge on this history: **GO** for further
validation, not yet for live wiring (see below). **Reported honestly**:
in this specific real window the DP rule played every single day - not a
bug, a real mathematical property (at streak `s=0` the reservation
threshold is *always exactly 0*, since a failed play at `s=0` costs
nothing - there's nothing to protect), combined with this window's real
streaks never exceeding 8, at which point even the highest threshold
encountered (0.7856) still cleared against that day's real `p=0.908`.
The "protect an established big streak" behavior is real and provably
present in the model (see `tests/test_decision_theory.py`'s dedicated
hand-computed case, `s=5` correctly choosing to sit out a `p=0.3` day),
but this particular thin, modest-streak real history never actually
exercised it - an honest limitation of a 71-day backtest, not a flaw in
the reasoning.

**Deliberately not wired into `pipeline.run()`/`predictions.select_picks`
this slice** - informational/backtest-only, the same "validate offline
first, live wiring is a separate later decision" precedent
`train_hitter_hit_model.py`'s own docstring and `Matchup_Hit_Probability`'s
`weight=0.0` ship-conservatively pattern already established elsewhere in
this project. Also explicitly out of scope: a "1 pick vs. 2 picks"
decision (only binary PLAY-today's-existing-2-pick-rule/SIT this slice);
correlation/joint modeling between two same-day picks (item #4's other,
separate sub-problem - `p` is taken as the day's already-computed real
probability, not reopening that question here); and the i.i.d.-across-days
assumption is a real, stated simplification (serial correlation across
days - e.g. a league-wide hot/cold week - isn't modeled).

### Same-game diversification (`predictions._diversify_second_pick`)

Quant-analytics item #4, slice 2 of 2 - the deferred other half of the
streak-aware stopping rule above: "correlation from being in the same
game/weather." No weather data exists anywhere in this project (confirmed
by a real search - this slice can only address same-game correlation).
`game_pk` exists in raw Statcast and `schedule.normalize_schedule`'s
output, but was discarded before reaching `predictions.select_picks` -
`matchup.compute_matchup_hit_probability` returned only `[key_mlbam,
Matchup_Hit_Probability]`. A real local query against
`data/raw/statcast_2026.parquet` found 6 of 71 real two-pick days
(≈8.5%) had both picks in the same real game - real, but too thin to fit
an actual correlation coefficient from (statistically indefensible with
n=6).

Instead, `config.SAME_GAME_DIVERSIFICATION_MARGIN` (new, ships at **0.0
- the exact null hypothesis**) and `predictions._diversify_second_pick`
use a robust, sign-only argument: same-game correlation (shared weather/
park/pitching-matchup quality) can only ever raise the chance both picks
miss together relative to independence, never lower it - so when the #2
pick shares a game with #1, and a comparably-ranked candidate from a
DIFFERENT game exists within `margin`, preferring it is a real
improvement whose direction doesn't depend on knowing the exact
correlation magnitude. Only ever touches the #2 slot; #1 is never
affected; a missing `game_pk` column (every historical wave.csv-only
replay) is a no-op, same convention as every other optional qualifier in
`select_picks`. `pipeline.py` now merges `schedule_df`'s real `game_pk`
into `pick_pool` so live runs can actually use this.

**Real backtest** (`scripts/backtest_same_game_diversification.py`,
GitHub Actions run 32513681638, 2026-08-21, the SAME final
`ML_FINAL_HOLDOUT_DATES` (20) holdout window
`train_hitter_hit_model.py`/`backtest_selection_rule.py` validate
against): **a clean, honest NO-GO by construction** - of the 20 real
holdout dates (only 3 with a scored pick pair at all), **zero** had a
real same-game #1/#2 pair under today's unmodified ranking. Every margin
in the sweep (`0, 0.02, 0.05, 0.1`) produced byte-identical results
(`top_2_BOTH_hit_rate=0.6667`, the real `require_all=True` "both must
hit" Beat the Streak rule) because there was nothing for a nonzero margin
to possibly change in this particular window - not a failure of the
method, a real, honest reflection of how rare same-game pairs are within
this specific 20-date slice (contrast with the broader 71-date
`predictions.csv` sample above, which did show 6 real same-game pairs -
different, larger window, same underlying real rarity). The machinery
ships tested and ready to re-run as more real history accumulates;
`SAME_GAME_DIVERSIFICATION_MARGIN` stays at its null-hypothesis default
until a real backtest earns otherwise.

This completes quant-analytics item #4 ("decision theory for the actual
game structure") in full: streak-aware optimal stopping + same-game
diversification.

### Market benchmark confirmation (`scripts/debug_espn_odds.py`)

Quant-analytics item #6, slice 1 of at least 2 - "no market benchmark."
Unlike the NFL side's "FantasyPros ECR" precedent (which turns out not to
be a scrape at all - it's `nflreadpy` pulling a static CSV snapshot from a
GitHub mirror), no equivalent trick exists for MLB, and this project had
zero HTTP client dependency before this slice. Given the user's explicit
choice to pursue a real free source rather than an API key, real web
research (this repo's own sandbox blocks every odds-site domain directly,
so a real community project's source was read via `raw.githubusercontent.com`
instead) found something better than HTML scraping: ESPN's unofficial but
public, keyless, structured-JSON `site.api.espn.com/apis/site/v2/sports/
baseball/mlb/scoreboard` and `.../summary?event={id}` endpoints, which
still serve a real `pickcenter` array of moneyline odds. Per this
project's established `debug_nfl_data.py` pattern, this slice is
**confirmation-only**: a real dispatched script (`debug_espn_odds.py`,
`.github/workflows/debug_espn_odds.yml` - this sandbox can't reach ESPN
directly, but the Actions runner can) settles the real shape before any
ingestion logic gets written against assumed schemas. `requirements.txt`
gained its first HTTP client dependency, `requests`.

**Real findings** (GitHub Actions run 32516808493, 2026-08-21, `days_back`
default of 5):

- **Schema confirmed and stable.** Every real game on both dates queried
  (`20260821`, 15 events; `20260816`, 15 events) had exactly one
  `pickcenter` row (`provider.name="DraftKings"`, `id="100"`) with real
  `details`, `overUnder`, `spread`, `overOdds`/`underOdds`, and
  `awayTeamOdds`/`homeTeamOdds` objects carrying real `moneyLine` fields -
  e.g. away=+240/home=-261 (Cardinals at Phillies) de-vigging to a real
  home win probability of 0.7108 via the standard proportional method
  (`moneyline_to_implied_probability` + `devig`, both hand-sanity-checked
  before the dispatch, e.g. `-150` → exactly 0.6).
- **Team abbreviation crosswalk needed, but small.** ESPN uses `ARI` and
  `CHW`; this project's `schedule.TEAM_ID_TO_ABBREV` uses `AZ` and `CWS`.
  All other 28 real abbreviations matched exactly. Only these two need a
  mapping when this gets wired in for real.
- **Historical depth confirmed at 5 days back, not further.** The
  `20260816` query returned real `pickcenter`/moneyline data for games
  that had already been played by dispatch time - a genuinely encouraging
  sign for building a "beat the closing line" backtest, since the single
  biggest open risk going in was whether ESPN drops odds once a game
  finishes. This dispatch only tested exactly 5 days back, though - real
  depth beyond that (weeks/months, i.e. enough for a real historical
  backtest across this project's full holdout window) is still
  unconfirmed and would need its own real check before a backtest is
  promised.

**Honest bottom line:** this slice is a real GO on the schema/crosswalk
front (both are simple and stable) and a real, but only partially
confirmed, GO on historical depth (5 days back verified; further back is
not yet known). Slice 2 - wiring a real `Market_Home_Win_Probability`
column into `pipeline.py`/`game_evaluation.py` as a logged side-by-side
comparison, and a real historical "beat the closing line" backtest once
depth is confirmed further back - is a plausible next step but not yet
scoped or committed to.

### Real market wiring + "beat the closing line" (`market_odds.py`)

Quant-analytics item #6, slice 2 - promotes slice 1's confirmation-only
script into a real module (`src/mlb_metrics/market_odds.py`) and wires it
in as a logged, non-fatal comparison column, not a model input.
`pipeline.run()` fetches real ESPN market odds once per run in a
**separate** try/except from the schedule fetch, so a market-fetch
failure can never suppress real game-pick logging - the core feature
stays reliable even when ESPN is unreachable that day.
`game_predictions.select_game_picks` gained an optional
`market_probabilities` param and the log gained a real
`market_home_win_probability` column (migrated cleanly for older logs, no
data loss). `game_evaluation.build_game_picks_export` now reports the
market's own accuracy/Brier/log-loss the same way it already reports the
model's (same `evaluation.py` function-reuse pattern, not a
reimplementation), plus the item's literal stated goal: a real
**`beat_closing_line_rate`** - the fraction of resolved, market-available
games where the model's squared error against the real outcome (on a
common home-win-probability basis, not each side's own predicted-winner
basis) is strictly lower than the market's, with exact ties excluded from
both sides of the rate so it can never hide a tiny real comparison base.
`scripts/backfill_market_odds.py` (+ `.github/workflows/backfill_market_odds.yml`)
one-time-seeds the confirmed-reachable historical window using the same
dependency-injected-fetch pattern `game_predictions.resolve_game_predictions`
already established, so it's unit tested without real network. The
dashboard's History table picks up the raw column for free (data-driven
`buildTable`), and `renderGamePickStats` gained a real "Beat Closing Line"
summary tile.

**Real backfill** (GitHub Actions run 32519740032, 2026-08-21, the
default 5-day-back window, i.e. 2026-08-16 through 2026-08-20 - the exact
depth slice 1 confirmed ESPN still serves odds for):

- Real market data landed for **12 of 159** total resolved, above-threshold
  game picks - thin, exactly as expected from a first 5-day seed, not
  padded or extrapolated.
- **`market_accuracy` = 50.0%** on those 12 games, vs. this project's own
  **`accuracy` = 53.5%** across all 159 resolved games (different
  populations - the model's own 159-game number is NOT a fair comparison
  to the market's 12-game one; the real apples-to-apples comparison is
  `beat_closing_line_rate` below).
- **`beat_closing_line_rate` = 33.3%** (4 real model wins vs. 8 real
  market wins, zero ties, n=12 compared) - a real, honest **NO-GO** on
  this specific thin sample: the market beat this project's model on a
  real head-to-head Brier comparison more often than the reverse. n=12 is
  far too small to be statistically meaningful either way (a single
  extra game changes the rate by over 8 points) - this is a real starting
  number to track forward as `pipeline.run()` accumulates more real
  market-compared games day by day, not a final verdict on whether the
  model has a real edge over the closing line.

This completes quant-analytics item #6 ("no market benchmark") slice 2:
the market benchmark now exists, is logged, and is being compared
honestly - "we beat the closing line" is now a real, trackable number
(currently answered "not yet, on the data logged so far"), not an
unanswerable claim.

### Kelly-criterion bet sizing (`kelly.py`, `scripts/recommend_bets.py`)

A follow-up to the market benchmark above: turns "the model's probability
disagrees with the market's real price" into an actual recommended stake.
Three real design decisions worth stating plainly, since they run against
common intuition:

- **"Favorite" is irrelevant.** A favorite's higher win probability is
  already priced into its moneyline - betting favorites has no edge baked
  in by itself. The real signal is whether THIS project's own probability
  for a side diverges from what that side's real price implies, in either
  direction (an underdog the model likes more than the market does is
  just as real an edge as a favorite it likes more).
- **Single straight bets only - never parlays.** A parlay compounds the
  book's vig across every leg, and would need real joint-probability
  modeling this project doesn't have (`predictions._diversify_second_pick`
  is an explicit sign-only proxy, not a real correlation estimate) to
  ever be justified. `recommend_bets.py` never constructs or considers
  multi-leg bets.
- **No execution layer, and realistically none is coming.** Retail
  sportsbooks don't offer public betting APIs to individuals, and
  actively limit/ban bettors who show a persistent edge.
  `scripts/recommend_bets.py` is a report a human reads and places the
  bet from manually - not an automated pipeline stage.

**Mechanics**: `kelly.py` (pure, no `config` import, mirrors
`decision_theory.py`'s own separation of math from config wiring) -
`moneyline_to_net_odds` converts a real American moneyline to net odds
`b`; `kelly_fraction(probability, moneyline, fraction)` computes
`max((p*b - (1-p))/b, 0) * fraction` - clipped at 0 (no real edge means
no bet, never a negative stake), scaled by
`config.KELLY_FRACTION_MULTIPLIER` (half-Kelly by default - full Kelly is
only growth-optimal if the probability estimate is exactly right, and
this project's own estimates carry real error, per the Wilson CIs used
throughout). Critically, the edge computation compares the model's
probability against the REAL, VIGGED market price
(`market_odds.moneyline_to_implied_probability` on the real raw
moneyline - `market_odds.py` now carries `home_moneyline`/`away_moneyline`
alongside the existing de-vigged `market_home_win_probability`), not the
de-vigged one - a real bet is paid off at the real price, so Kelly's
net-odds term has to come from that same real price. `config.KELLY_MIN_EDGE`
(0.02) is a minimum-edge buffer against noise before any stake is even
considered.

`scripts/recommend_bets.py` reads that day's ALREADY-LOGGED game picks
(real `predicted_probability`/`predicted_winner` from `pipeline.run()`)
rather than recomputing from raw Statcast, and fetches fresh real market
odds for that exact date. Two independent safety checks before anything
is recommended: (1) the target date defaults to `schedule.today_local()`
and the script HARD REFUSES (raises, doesn't silently fall back) if the
log has no unresolved picks for that date - a real staleness bug was
caught in design review here, since `market_odds.py` matches games by
`(home_team, away_team)` only, with no date key, so silently defaulting
to "whatever's logged" risked blending one day's model probability with a
different day's real game; (2) a second, independent check against
`schedule.fetch_todays_games`' real MLB Stats API status, skipping any
game that isn't still `"Scheduled"`. It's mathematically provable that
the script never recommends both sides of one game when `min_edge > 0`
(the two sides' edges always sum to `-vig`, which is always negative for
a real book) - kept as a defensive runtime check anyway, since real data
anomalies are a more likely explanation than real arbitrage in this
pipeline.

**Honest current status - always printed, never hidden**: every run
prints the real `beat_closing_line_rate`/`n_beat_closing_line_compared`
from `game_evaluation.py` as a leading banner, with an explicit
"NOT YET STATISTICALLY VALIDATED" warning while `n` stays below
`config.KELLY_MIN_GAMES_FOR_CONFIDENCE` (100, a conservative round floor,
not a formal power calculation) - which it currently does (n=12, see
above). This script computes and shows real numbers; it does not claim
they're a proven strategy.

**Real dispatch verification** (GitHub Actions runs 32679390305/
32679454113/32679504206, 2026-08-24) confirmed both safety guards work
correctly end to end, and caught two real bugs along the way: (1) a
stdout/stderr buffering issue that printed the confidence banner AFTER a
later refusal message in the CI log despite running first in the code -
fixed with explicit `flush=True`; (2) the workflow was missing the
`MLB-StatsAPI` dependency `schedule.fetch_todays_games` needs - fixed by
adding it to the install step. Once fixed, a real run against the log's
actual last logged date (2026-08-21, three real days stale relative to
the dispatch) correctly refused to recommend anything: all 15 real games
logged that day were skipped because their real MLB Stats API status was
no longer `"Scheduled"` (the games had long since finished) - exactly the
intended safe behavior, not a bug. A genuinely positive real
recommendation requires the daily pipeline to have logged TODAY's real
slate first (this session's log happened to be stale); the underlying
edge/Kelly math itself is separately verified via hand-computed unit
tests (`tests/test_kelly.py`, `tests/test_recommend_bets.py`).

### Game-picks tracking pivot: bet-advised + real units won/lost (`game_predictions.advise_bets`)

Now that a real bet can be advised and sized, the Automated Game Picks
dashboard's tracking pivots to match: **replaces** the old
`above_threshold`-scoped accuracy/streak tracking (was the model's own
favorite pick right) with `bet_units`-scoped real P&L tracking (did
the bets the market actually disagreed with make money) - the real
question worth answering now, per direct user request. `above_threshold`
still gets published on every row for context; it just no longer drives
the headline scoring. `market_accuracy`/`market_brier_score`/
`beat_closing_line_rate` are untouched - a genuinely separate "are we
better forecasters than the market" question, not what changed here.

**Shared logic, not duplicated**: the same real edge/Kelly decision that
used to live only inside `scripts/recommend_bets.py` (`build_bet_recommendations`)
moved into `game_predictions.advise_bets` - `pipeline.run()` now calls it
directly on each day's real logged picks + fetched market odds, so the
SAME real decision that would be reported by `recommend_bets.py` gets
logged automatically as part of the daily pipeline, not just printed on
manual request. `recommend_bets.py` now calls this same shared function
instead of keeping its own copy.

**New logged columns** (`game_predictions.GAME_PREDICTION_COLUMNS`):
`bet_units`, `bet_side`, `bet_team`, `bet_moneyline`,
`bet_stake_fraction`, `bet_profit_units`. Stakes are reported in
**units**, not dollars - the standard sports-betting convention (bettors
track/report performance in bankroll-agnostic units since bankroll size
varies per person). `bet_units` is also the single real signal for
whether a bet was advised at all, replacing the old separate boolean:
`0.0` means no bet, any positive value is real units to risk. It's
computed as `bet_units = bet_stake_fraction / config.UNIT_SIZE_FRACTION`,
where `UNIT_SIZE_FRACTION` (a new constant, **0.01** - i.e. 1 unit = 1%
of bankroll, a common real convention, not a formally derived number)
defines what "1 unit" means as a fraction of bankroll. A real bettor
still converts units to their own real dollar stake at bet time via
`recommend_bets.py`'s own optional `--bankroll` flag, which this constant
doesn't touch. `bet_profit_units` is filled in only once
`resolve_game_predictions` knows the real outcome - real profit
(`bet_units * kelly.moneyline_to_net_odds(bet_moneyline)`) on a win,
`-bet_units` on a loss, and stays null for a non-advised (`bet_units ==
0`) or still-pending game (never a fabricated 0).

**Defended against a real doubleheader edge case**: `market_odds.py`
matches by `(home_team, away_team)` only (no date key - a disclosed,
pre-existing limitation), so a real doubleheader could in principle
produce two bet-advice rows for the same `game_pk`. `select_game_picks`
checks for this explicitly and drops both rows for any `game_pk` that
collides, with a warning, rather than picking one arbitrarily.

**Honest non-goals**: no historical backfill - raw moneylines were never
persisted for any of the 461 pre-existing logged rows, and ESPN's
confirmed real odds depth is only ~5 days back anyway, so this tracks
forward from when it shipped, same "accumulate forward" precedent
`game_picks.py`'s own docstring already established. `UNIT_SIZE_FRACTION`
just defines the unit/bankroll-fraction conversion, not a real
compounding-bankroll simulation - that's a real, more complex follow-up,
not this slice.

**Dashboard**: `renderGamePickStats` replaced its Current Streak/Best
Streak/Accuracy/Games Tracked tiles with Bet Streak/Best Bet Streak/Win
Rate/Bets Tracked/P&L (units, e.g. `+2.13u`, colored green/red by sign) -
Beat Closing Line stays, unchanged. `renderTodaysGamePicks`'s
"recommended" card highlight now reflects `bet_units > 0` instead of
`above_threshold`, and an advised card shows the real bet line
(team/moneyline/units to risk) instead of a dollar amount. Verified via a
local static-server + headless Chromium screenshot pass against synthetic
data matching the real new schema - tiles, the advised-bet highlight, and
the bet columns in the History table (fully data-driven, no JS change
needed there) all render correctly.

Full test suite: 650 passed (across `test_game_predictions.py`,
`test_game_evaluation.py`, `test_recommend_bets.py`, `test_pipeline.py` -
including a real end-to-end pipeline test that advises a bet in units on
day 1 and resolves it into a real positive unit profit on day 2).

### Quant-analytics item #5: backtest scope and statistical significance (`evaluation.py`)

Every real backtest rate this dashboard reports - `beat_closing_line_rate`,
`market_accuracy`, `win_rate_on_advised_bets`, `day_survival_rate` - is a
sample statistic computed from a real but often small `n`. A 12-game 33%
"beat the market" rate and a 1,200-game 33% rate are not the same amount
of evidence, and a bare percentage on a dashboard can't tell them apart -
this was flagged directly in this project's own README as an open gap
("item #5 ... was explicitly skipped") back when it was still true. This
slice closes it: every rate-based backtest metric now ships with a real
measure of whether its sample size actually supports trusting it.

**Three new statistics primitives (`evaluation.py`)**, each a thin,
verified wrapper around an established library - not a hand-derived
formula:
- `wilson_confidence_interval(successes, n, alpha=0.05)` - a scalar Wilson
  score CI, reusing `helpers.wilson_ci` (quant-analytics item #3's own
  per-row vectorized version) rather than reimplementing the same formula
  a second time.
- `binomial_significance(successes, n, null_probability=0.5)` - a
  two-sided exact binomial test p-value via `scipy.stats.binomtest`.
- `mean_significance(values, null_value=0.0)` - a one-sample two-sided
  t-test p-value via `scipy.stats.ttest_1samp`, for testing a real mean
  (not a count) against a null.

**Which metric gets which test, and why (this is the part that's easy to
get statistically wrong)**:
- `beat_closing_line_rate` gets BOTH a Wilson CI and a real
  `binomial_significance` p-value against a null of 0.5. This is a
  genuinely well-posed null: "whose squared error is lower on this game"
  is a symmetric coin flip under "no real skill difference between the
  model and the market," so 0.5 is the honest null, not just a convenient
  one.
- `market_accuracy` and `day_survival_rate` get a Wilson CI only, no
  p-value. A raw accuracy rate does NOT have a fair 0.5 null - real MLB
  home teams win somewhat more than half their games, so "vs. a coin
  flip" would misrepresent what "no skill" actually looks like here.
- `win_rate_on_advised_bets` gets a Wilson CI only, no `binomial_significance`
  p-value either - deliberately, even though it's a win/loss rate. Advised
  bets are placed at different real moneylines, so a win/loss count alone
  can't tell a good -150 favorite bet apart from a bad one; a 55% win rate
  on -150 favorites and a 55% win rate on +150 underdogs are very
  different real outcomes a win-rate-only test can't distinguish.
- Real bet profitability instead gets `roi_p_value` - a `mean_significance`
  t-test on every advised bet's real `bet_profit_units` against a null of
  0 (breaking even). This is the correctly-posed test for "did the
  advised bets actually make money, or is this within noise": each bet's
  real profit already prices in its real moneyline, which a win-rate test
  throws away.

**Wiring**: `game_evaluation._market_comparison_metrics`/
`_beat_closing_line_rate`/`_bet_pnl_metrics` each compute and return their
metric's CI (and p-value, where one applies) alongside the rate itself;
`build_game_picks_export`'s summary row gains
`market_accuracy_ci_low/high`, `beat_closing_line_rate_ci_low/high`,
`beat_closing_line_rate_p_value`, `win_rate_on_advised_bets_ci_low/high`,
and `roi_p_value`. `evaluation.build_beat_the_streak_export`'s summary
gains `day_survival_rate_ci_low/high`. No migration guard needed anywhere
here - unlike the per-row picks table, these summary rows are always
computed fresh from the full log on every pipeline run, never read back
from a persisted CSV.

**Dashboard**: `renderGamePickStats`/`renderStreakStats` gained a shared
`ciLabel`/`significanceLabel` formatter pair; the Win Rate, P&L, Beat
Closing Line, and Day Survival Rate tiles each now show a small sub-line
under their headline number - e.g. "95% CI 14–61%" or
"not significant (p=0.29)" - so a thin sample never reads as more certain
than it actually is. Verified via a local static-server + headless
Chromium screenshot pass against real formula output (computed via the
real `evaluation.*` functions themselves, not hand-typed numbers) written
into synthetic summary CSVs.

**Methodology page**: two new cards ("Exact Binomial Test," "One-Sample
t-Test") in the existing Statistical Significance Testing section
(alongside the Wald Test card), and the existing Wilson Score Confidence
Interval card's "How we use it" now also documents this scalar reuse.

**New dependency**: `scipy` added explicitly to `requirements.txt` -
it was already present transitively (both `scikit-learn` and
`statsmodels` depend on it), but `evaluation.py` now imports it directly,
so it's declared honestly rather than relied upon silently.

**Honest non-goal**: no significance testing was added for the DFS ML
holdout comparisons (log_loss/MAE vs. baseline) - those are continuous-
error metrics, not binomial rates or a simple one-sample mean, and need a
different real tool (e.g. a paired test between the model's and the
baseline's per-row errors) - a real, more involved follow-up, not this
slice.

Full test suite: 657 passed (7 new/updated tests across
`test_evaluation.py`, `test_game_evaluation.py` - hand-computed cases for
all three new primitives, plus real CI/p-value assertions wired through
`build_beat_the_streak_export`/`build_game_picks_export`'s summary rows).

### Real quant sanity-check: `KELLY_MIN_EDGE` raised 0.02 → 0.05 (2026-08-24)

The very first real day the bet-advice pipeline ran with real market
odds (2026-08-24), it advised bets on **8 of 10** real games - with
model-vs-market disagreements as large as 12 percentage points on the
de-vigged price. That's implausible as genuine value against a real,
liquid MLB moneyline market on the same day, let alone on 80% of a
day's slate at once - a real, sharp market essentially never leaves
that much on the table.

**Root cause (a real forecasting-skill gap, not a code bug)**: this
project's model probabilities cluster much closer to 50/50 than real
sportsbook lines do - the model is comparatively conservative/low-
spread, while a real book prices in starting pitcher, bullpen,
injuries, and more, more aggressively. When the model is less
confident than the market on every game, its comparatively muted
probability for whichever side the market is confident about
systematically LOOKS like value on the other side - even when it's
really just the model under-informing itself relative to the market,
not a real market inefficiency. This project's own
`beat_closing_line_rate` (0.357, n=14) is not yet statistically
distinguishable from a coin flip (`evaluation.binomial_significance`
p=0.42, using quant-analytics item #5's own new significance test) -
there is currently no proven evidence this model forecasts games
better than the market at all. Betting on every 2%+ disagreement was,
in effect, betting on the model's own forecasting noise.

**Fix (a stopgap, not a real solution)**: `config.KELLY_MIN_EDGE`
raised from 0.02 to **0.05** - filters out the smallest, least
convincing disagreements, but does NOT fix the underlying calibration/
skill gap. Recomputing today's real 8 advised games against the new
bar: 3 (AZ@CHC 3.5%, SEA@PHI 4.6%, ATH@MIN 4.9%) drop below 5% and were
corrected back to "no bet advised" in the real log
(`data/predictions/game_predictions.csv`) and re-exported to
`docs/data/game_picks_*.csv` - the games hadn't been played yet, so
this was a safe, honest correction, not a retroactive rewrite of a
resolved outcome. The remaining 5 (WSH@COL 7.0%, SF@CIN 5.2%, LAA@CLE
10.3%, DET@TB 5.6%, CWS@TEX 6.0%) still clear 5% - still a lot for one
day, an honest sign the deeper calibration question (real follow-up,
not this fix) still needs investigating, not that 0.05 is definitely
the right number either.

**Explicitly NOT done here** (raised, not decided, in the same
conversation): gating bet-advice on `beat_closing_line_rate` actually
clearing statistical significance before advising anything at all -
a more correct, structural fix than a bigger static threshold, but a
real design change of its own, not bundled into this stopgap.

Full test suite: 658 passed (no test changes needed - existing tests
either pass `min_edge` explicitly or use a fixture with a wide enough
margin to clear the new default).

### Real bet-sizing fix: uncertainty-scaled Kelly + a daily unit cap (2026-08-25)

Direct user ask: **"we need the units risked to not be arbitrary."**
Investigation confirmed the concern was real, not just a feeling: the
stake size on a given bet was the product of THREE separately-chosen
constants, each explicitly self-documented in `config.py` as a
convention, not a derived number - `KELLY_FRACTION_MULTIPLIER` (0.5,
"the standard practitioner default"), `KELLY_MIN_EDGE` (0.05, "not a
number backed by a formal calculation"), and `UNIT_SIZE_FRACTION` (0.01,
"a common real convention"). The real logged data showed the concrete
symptom: the 5 resolved advised bets on the books ranged **5.0 to 8.8
units** (5-8.8% of bankroll) each - large stakes for a model whose own
`beat_closing_line_rate` isn't yet statistically distinguishable from a
coin flip (see the sanity-check above). Only 5 real resolved bets exist
total - far too few to empirically backtest-tune any of the three
constants, which ruled out that approach.

**The fix grounds "how much to risk" in each team's own real record,
with zero new tunable constants beyond the user's own explicit cap**:

- `teams.compute_team_win_rate_ci` - one row per team: real
  season-to-date `win_rate` (from `build_team_record`'s real win/loss
  outcomes) plus a real Wilson score CI (`helpers.wilson_ci`, already
  established elsewhere in this project) on that binomial proportion.
  Naturally wide early in the season or for a thin sample, narrow once
  real games accumulate - merged into `teams.assemble_team_metrics`'
  output (`win_rate_CI_Low`/`win_rate_CI_High`), so it's part of the same
  `confidence` frame the daily pipeline already computes.
- `game_picks.apply_kelly_uncertainty` - for each game, combines both
  teams' real CI half-widths via standard root-sum-square error
  propagation (the two teams' records are independent real samples) into
  one real per-game uncertainty measure, then subtracts it from the raw
  win probability to get a real, worst-reasonable-case ("pessimistic")
  probability. A team missing from `confidence` entirely gets the same
  maximal degenerate half-width `wilson_ci`'s own `n=0` case already uses
  (0.5) - genuinely unknown is treated as maximally uncertain, not
  ordinary.
- `game_predictions.advise_bets` sizes each bet's stake off THAT real
  pessimistic probability at a full 1.0 Kelly multiplier, instead of the
  raw point estimate scaled by the old flat 0.5 - so a thin, uncertain
  team record now genuinely produces a smaller (or zero) stake, and a
  well-established one produces a stake close to what full Kelly would
  have said anyway. `KELLY_MIN_EDGE` eligibility is unchanged and still
  uses the raw model probability - "is there a real edge worth
  considering" stays a separate question from "how much to actually risk
  given that edge." Gated per-ROW on the pessimistic values actually
  being non-null (not just column presence, since the columns are now
  always persisted - see below) - any game without real team-record
  coverage cleanly falls back to the old flat-multiplier behavior instead
  of silently producing a NaN stake.
- **`config.KELLY_DAILY_UNIT_CAP = 5`** - a hard portfolio-level cap, in
  units, on the total stake advised across all of one day's bets
  combined - the user's own explicit number, not derived. If a day's
  total advised stake would exceed it, `advise_bets` scales EVERY advised
  stake for that date down proportionally (never dropping any single bet)
  so the day's total lands exactly at the cap.
- `KELLY_FRACTION_MULTIPLIER` is now a FALLBACK only - still the real
  multiplier used when a game has no real team-record coverage
  (`scripts/recommend_bets.py --kelly-fraction`, `game_picks_backtest.py`'s
  historical replay), no longer the default source of shrinkage for every
  bet.

**Persisted, not just used transiently**: `select_game_picks` now writes
`home_win_probability_pessimistic`/`away_win_probability_pessimistic`
onto every logged row (null when `confidence` wasn't given/didn't cover
that game) - `GAME_PREDICTION_COLUMNS` gains both, with the usual
migration guard backfilling `NA` for any pre-existing row. This matters
because `scripts/recommend_bets.py` re-derives its report from the
ALREADY-LOGGED picks rather than recomputing team confidence from raw
Statcast (a much heavier fetch than that script otherwise does) - without
persisting these two columns, the live pipeline's logged stake and the
report's recommended stake for the same game could silently disagree.
`pipeline.run()` now passes `confidence=outputs["confidence"]` into
`select_game_picks` so the live daily pipeline actually uses this sizing,
not just the tests.

New tests: `tests/test_teams.py` (Wilson CI correctness, smaller sample
= wider interval), `tests/test_game_picks.py` (`apply_kelly_uncertainty`
- combined half-width, zero-clipping, missing-team maximal uncertainty,
NaN passthrough), `tests/test_game_predictions.py` (pessimistic sizing
producing a real smaller stake than the old flat multiplier, the daily
cap's proportional scale-down, a different date's stakes staying
untouched, the per-row null fallback, persistence + migration of the two
new columns). Full test suite: 703 passed.

### Bet streak is now a DAY streak, not a per-bet streak (2026-08-25)

Direct user ask: **"the streak should be days (only add one per day or go
back to zero) if the cumulative bets made money that day."** The
previous `current_bet_streak`/`best_bet_streak` (`game_evaluation._bet_pnl_metrics`)
walked every real resolved advised bet in date order and incremented on
each individual winning bet - so a day with several advised bets could
move the streak by more than one step in a single day, and a day's own
net result didn't matter, only each bet's own sign.

**Fix**: `_bet_pnl_metrics` now groups resolved advised bets by real
calendar `date` and sums `bet_profit_units` per day first
(`resolved.groupby("date")["bet_profit_units"].sum()`), then runs the
same streak counter over those daily totals instead of over individual
bets - each day extends the streak by exactly 1 if its bets, summed
together, were profitable overall, and resets it to 0 otherwise,
regardless of how many bets were advised that day or how any one of them
did individually. A day with no advised bets at all neither extends nor
breaks it (it's simply not in the grouped total). The column names
(`current_bet_streak`/`best_bet_streak`) are unchanged - only what they
count changed - but the dashboard tile labels were renamed **Day
Streak**/**Best Day Streak** (`docs/app.js`) so the displayed number is
honestly labeled for what it now actually measures.

New/updated tests (`tests/test_game_evaluation.py`): the existing
plain-consecutive-wins test was renamed and re-asserted as a day streak
(one bet per day, so the numbers coincidentally match the old semantics -
kept to show that continuity), plus a new test with two bets on the same
day netting positive (one streak step, not two) followed by a second day
with a winning AND a losing bet that nets negative overall (a real
reset, proving the day's TOTAL profit drives the reset, not merely
"at least one bet lost that day").

Full test suite: 704 passed.

### The short-priced-favorite blowup risk: `kelly_fraction_multiplier` is stacked, not swapped out (2026-08-26)

Direct user challenge to the uncertainty-scaled Kelly fix above: **"choosing
whether we bet or not should not just be on whether we beat Vegas odds...
that 1 loss versus minimal return on the 3 wins minimizes returns, and if
[the underdog] wins just a couple times more than expected, we're working
at a loss despite the impressive win record."** Using the user's own
example (Dodgers -275 vs. Rockies): breakeven implied probability is
275/375 = **73.33%** and the real payout on a win is only `100/275 =
0.364` units per unit staked. That asymmetry makes the bet's real profitability extremely sensitive to
small errors in the probability estimate - concretely, if the model says
80% but the team's TRUE win probability is actually 75% (a real EV of
just +0.023 units per unit staked - barely profitable), and only 70%
(a real **-0.046** EV - a genuine loss): a 5-10 point overestimate, well
within this project's own documented calibration noise, flips real
profit to real loss. A real
20-bet simulation at this same price makes the fragility concrete: at a
true 75% win rate (expected 15 wins), the *expected* result is barely
positive (+0.45 units on 20 risked) - but 14 wins (well within one
binomial standard deviation of "expected") turns the same 20 bets into a
real **-0.91 unit loss**, despite a 70% win rate that would look
"impressive" on any accuracy-scored dashboard.

**Investigated first, not assumed**: simulating the exact Dodgers/Rockies
matchup through the already-shipped uncertainty-scaled Kelly (a
dominant team, 70-30 real record, n=162 - the largest possible real
in-season sample) found the existing pessimistic-probability mechanism
*already* zeroes out this exact bet when the model's raw estimate is
80% (pessimistic probability 70.1%, still under the 73.33% breakeven) -
the CI-based fix from the prior slice already does real work here.  But
push the model's raw confidence to 85% and the pessimistic probability
(75.1%) clears breakeven again, implying a **6.6-unit (6.6% of
bankroll) full-Kelly stake on a single game** - because a team's
SEASON-LONG win-rate Wilson CI can stay genuinely tight (a real,
well-established record) even though this one specific game still
carries real matchup-level uncertainty (starting pitcher, injuries,
etc.) that a season-aggregate CI simply can't see.

**Root cause**: `game_predictions.advise_bets` was treating the CI-based
pessimistic probability and the flat `kelly_fraction_multiplier` as
alternatives - full Kelly (1.0x) once a pessimistic probability was
available, the flat multiplier (0.5x) only as a fallback when one
wasn't. That swap was wrong: they defend against two different, both
real, sources of error (season-level team-quality uncertainty vs.
across-the-board probability-estimation error) and neither substitutes
for the other.

**Fix**: `kelly_fraction_multiplier` is now ALWAYS applied on top of
whichever probability sizes the bet - the code simplified to always call
`kelly.kelly_fraction(sizing_probability, moneyline, kelly_fraction_multiplier)`,
with only the *probability* (pessimistic vs. raw) still chosen per-row.
On the 85%-confidence Dodgers example above, this halves the single-game
stake from 6.6 units to **3.3 units** at the default half-Kelly
multiplier - real, direct risk reduction on exactly the scenario raised,
with no new tunable constant (the existing `KELLY_FRACTION_MULTIPLIER`
default, 0.5, is reused as-is).

**Explicitly not done here** (a real, more involved follow-up, not this
slice): a per-game (not per-team-season) uncertainty measure - e.g. the
model's own real historical calibration error conditional on how
extreme its predicted probability is - would target the exact
established-team/thin-single-game-signal gap this example exposed more
precisely than stacking a flat multiplier does. `KELLY_MIN_EDGE`
eligibility also still gates on the raw model probability rather than
the risk-adjusted one; a genuinely zero-stake "advised" row (sizing
clips to 0 either from the CI or the multiplier) is already
indistinguishable from "not advised" downstream (`bet_units == 0` is
the sole real signal, and `resolve_game_predictions` never scores a
`bet_units == 0` row), so this wasn't treated as a separate bug, but a
tighter eligibility gate remains a real option if the flat multiplier
alone proves insufficient once more real advised bets accumulate.

Updated tests (`tests/test_game_predictions.py`): the pessimistic-sizing
test now asserts the multiplier is genuinely still applied (smaller than
both a full-1.0-off-pessimistic stake and the old flat-multiplier-off-raw
stake); the `confidence`-vs-no-`confidence` comparison test already used
an explicit `kelly_fraction_multiplier=1.0` in both branches, so it was
unaffected by this change and still passes.

Full test suite: 704 passed.

### Rebalancing the blowup-risk fix: it was also killing genuinely appealing underdogs (2026-08-26)

Direct user follow-up, the morning after the fix above shipped: **"maybe
we scaled it down too much... this is now taking out games where the
line is genuinely appealing (some underdogs who have a chance)."**
Investigated with the day's own real, live data rather than assumed:
2026-08-26's slate logged 15 real games, all with `bet_units = 0.0`.
Three (NYM, MIA, ATH) had cleared the raw min-edge bar as candidates but
were zeroed out at the sizing step. Concretely, NYM (home, +143) showed
a real 10.7-point edge (model 51.9% vs. market-implied 41.2%) - a
genuinely appealing underdog line - but its real per-game pessimistic
probability (40.2%) fell just under the market's own price.

**Root cause, found in the real data, not assumed**: every team's real
season-to-date Wilson CI (`docs/data/confidence.csv`, 132-133 games
played - nearly a full season) sat at roughly an **8-point half-width**.
That's not a fluke or an early-season artifact - it's the real
`sqrt(n)` precision floor of a ~130-game binomial win-rate estimate, and
it barely narrows even at a full 162-game season. Combined
(root-sum-square) across two teams, that's consistently **~11-12 points
of haircut on every single game, all season long** - larger than most
real edges this project has ever advised on, so it was quietly erasing
genuine, honest edges (like NYM's above) right alongside the
implausible ones it was built to catch.

**Two real error sources need two different controls, not one shared
dial** - confirmed by testing both blowup-risk and underdog-value cases
against the same lever: loosening the CI enough to let the NYM edge back
through (real edge -0.99 -> +4.72 points) *also* reopens 3-12 units of
exposure on the original Dodgers/Rockies -275 blowup scenario, because
Kelly's stake sensitivity to the sizing probability is proportional to
`1 + 1/b` - large for short-priced favorites (small net odds `b`),
small for underdogs (large `b`). A single flat probability haircut,
tuned to catch one, unavoidably damages the other.

**Fix - two independent, complementary controls, each targeting the
real mechanism it addresses**:
- `config.KELLY_UNCERTAINTY_CI_ALPHA = 0.32` (a real, standard,
  NAMED ~68%/1-standard-error confidence level, not a hand-picked
  shrinkage constant) replaces the default 95% Wilson CI (alpha=0.05)
  *only* for `teams.compute_team_win_rate_ci`'s bet-sizing input -
  `helpers.wilson_ci`'s own 95% default is untouched everywhere else
  this project reports a real CI (win_rate_on_advised_bets, beat_closing_line_rate,
  etc.). Roughly halves the per-team half-width (~8pts -> ~4pts,
  ~11-12pts -> ~6pts combined), enough to let the real NYM edge clear
  again.
- `config.KELLY_MAX_SINGLE_BET_UNIT_CAP = 2` (a deliberate user-set risk
  limit, like `KELLY_DAILY_UNIT_CAP`, not derived) hard-clips any ONE
  game's stake directly in `game_predictions.advise_bets`, applied
  BEFORE the existing daily cap. This is the real, targeted defense
  against Kelly's own short-odds amplification - the loosened CI alone
  would let the Dodgers/Rockies example back in at 3-12 units depending
  on model confidence; the per-bet cap bounds it at exactly 2 units
  regardless.

**Honest, verified consequence, not hidden**: with this project's actual
production defaults (`KELLY_MIN_EDGE=0.05`, `KELLY_FRACTION_MULTIPLIER=0.5`),
a bet that just barely clears the minimum-edge bar already implies a
~5-unit half-Kelly stake before either cap - Kelly is aggressive by
design at even a "modest" edge. That means `KELLY_MAX_SINGLE_BET_UNIT_CAP`
will bind on most real advised bets, not just extreme outliers - Kelly's
own relative differentiation mostly only survives between "zero" and
"the per-bet cap," not much further above it. This is a real, deliberate
trade-off inherent to capping at all (the whole point is bounding the
worst case), not a bug, but worth naming plainly rather than letting the
per-bet cap's practical effect be a surprise.

**Re-verified against both real scenarios**: today's real NYM pick, run
through the fixed pipeline with actual production defaults, now advises
**2.0 units** (real pessimistic edge +4.7 points, an uncapped half-Kelly
stake of 4.0 units, clipped to the 2-unit per-bet cap). The original
Dodgers/Rockies -275 blowup case, re-checked at raw model confidences of
80/85/90%, now lands at exactly 2.0 units in all three cases (previously
0/3.3/12.7 units before this rebalancing) - bounded regardless of how
confident the model claims to be, rather than swinging unpredictably
between zero and double digits.

New/updated tests (`tests/test_game_predictions.py`): a dedicated
single-bet-cap test, a cap-layering test (two bets clipped by the
per-bet cap, one that isn't, all three then scaled together by the
daily cap - proving the two caps compose correctly and neither bet is
exempted from the daily cap just because it was already reduced once),
and the existing daily-cap/pessimistic-sizing tests updated to use
edges that isolate each behavior from the new per-bet cap. Also fixed a
real, verified pandas/pytest interaction bug hit while writing these:
`pandas.Series == pytest.approx(scalar)` silently returns all-`False`
elementwise in this project's pinned pandas/pytest versions (confirmed
directly, not assumed) rather than raising or comparing correctly -
replaced with a direct `(series - expected).abs().max() < tolerance`
check.

Full test suite: 705 passed.

### Two real follow-ups after the above shipped: same-day refresh and the daily schedule

**A same-day recompute gap, found and fixed by hand once (2026-08-26)**:
re-running `daily_update.yml` (or `scripts/wave.py` directly) again on a
date it already logged picks for does NOT refresh those still-pending
picks with newly-fixed logic - `game_predictions.append_game_predictions`
deliberately keeps an EXISTING logged row over a freshly-recomputed one
for the same `(date, game_pk, metric)`, correct behavior for protecting
already-resolved history, but it also meant the two fixes above didn't
actually reach that day's already-logged, still-pending games just by
re-running the pipeline. Concretely hit real: 2026-08-26's slate was
logged by the morning's run, both fixes merged a few hours later, and a
manual re-run of the pipeline still showed the pre-fix numbers (NYM/MIA/
ATH bet_units all 0.0) even though the freshly-regenerated
`confidence.csv` DID carry the new CI. Corrected by hand this one time -
recomputed those 3 real candidate games directly with the merged
production code (`game_picks.apply_kelly_uncertainty`, `kelly.kelly_fraction`)
against the real freshly-regenerated confidence data and each game's own
already-logged raw probability/moneyline, then wrote the corrected
`bet_units`/`bet_stake_fraction`/pessimistic-probability values back into
`data/predictions/game_predictions.csv` and regenerated
`docs/data/game_picks_picks.csv` (`pipeline.write_game_picks_export`) -
not hand-derived, the same real functions, just called directly on that
one date's real data. This is a real, standing gap (a same-day RE-run
doesn't refresh pending picks) - not fixed structurally here, only
patched for that one occurrence.

**Daily schedule moved earlier (2026-08-26)**: direct user report -
10 AM real Arizona time (`schedule.today_local()`'s own fixed UTC-7
reference zone), games about to start, site still showing stale picks.
`daily_update.yml`'s cron moved from `"0 12 * * *"` (noon UTC = 5 AM AZ)
to `"23 10 * * *"` (10:23 UTC = 3:23 AM AZ) - two real, separate reasons,
not just "earlier": (1) more real buffer before the earliest MLB games
(as early as ~9 AM AZ for a 12:05 PM ET getaway-day game) - roughly
5.5-6 hours of lead time instead of ~4; (2) off the top of the hour
(`:23`, not `:00`) - GitHub's own docs warn that scheduled workflows
"can be delayed during periods of high load," and the top of the hour
(especially a popular round-UTC time like noon) is exactly when that
load peaks. No DST drift to manage here, unlike most US timezones - real
Arizona time is fixed UTC-7 year-round, so this offset never needs
revisiting for that reason. **Does NOT fix the same-day refresh gap
above** - only a genuinely fresh calendar date benefits from the earlier
schedule automatically; a manual re-run later on an already-logged date
still needs the same by-hand correction until that gap is addressed
structurally.

### Game Picks History table: trimmed to a curated column set (2026-08-28)

Direct user report: **"we're dumping almost everything data wise into
the table, and it all gets jumbled. All we need is date, predicted
winner, predicted loser, model probability, market probability, bet
units, bet team, and status[, and bet profit units]."** The History
table (`docs/app.js`'s `renderGamePickHistory`) previously passed the
FULL raw row straight from `game_picks_picks.csv` into `buildTable`,
which renders one column per object key with the raw key name as the
header (no title-casing) - every field ever added to that CSV
(`game_pk`, `home_team`, `away_team`, `above_threshold`, the raw
`market_home_win_probability`, `bet_side`, `bet_moneyline`, ...) showed
up as an unlabeled, unformatted column.

**Two derived columns didn't exist yet and needed real logic, not just
column hiding**: `predicted_loser` (whichever of home/away ISN'T
`predicted_winner`) and a market probability that's actually comparable
to the model's - the existing `market_home_win_probability` is always
the HOME team's de-vigged probability, so showing it next to the
model's `predicted_probability` (always the model's OWN favored side)
silently compares the wrong side whenever the model favors the away
team. Added `game_evaluation.build_game_picks_export` computing both as
real pandas derivations (flip `market_home_win_probability` to
`1 - market_home_win_probability` whenever the away team is favored,
NaN-safe) rather than duplicating that logic in JS - kept in the same
Python data layer as every other real derived column in this project.

**`renderTodaysGamePicks` (the separate "today's picks" cards section)
reads the exact same loaded `picks` array as the History table**, so the
underlying CSV/columns weren't trimmed - only `renderGamePickHistory`'s
own display was, by building a small formatted object (Title Case
labels, `%` for probabilities, `-` for not-applicable bet fields, `+/-`
and a `u` suffix for profit units) before calling `buildTable`, the same
pattern `renderProbablePitchers` already established for a "nice" table
rather than a raw data dump.

Verified two ways: `tests/test_game_evaluation.py` (real Python
assertions on both derived columns, including the flip case), and a
small Node harness loading the real `app.js` against synthetic rows to
inspect the actual rendered `<table>` HTML directly (not just re-reading
the code) - confirmed the header row is exactly the 9 requested columns
and the away-favored row's Market Probability shows the correctly
flipped 60.0%, not the raw home-team 40.0%.

Full test suite: 706 passed.

### Real follow-up: game-pick probability calibration (2026-08-24, not shipped)

Direct follow-up to the `KELLY_MIN_EDGE` sanity-check above: on the 93
real games with both a model probability and a real de-vigged market
price, the model's `home_win_probability` was measurably narrower than
the market's on the exact same games (std 0.0347, max 63.7% vs. the
market's std 0.0588, max 72.7%) - the concrete mechanical cause of the
false-edge problem, not just "the model disagrees with the market
sometimes." A deeper, separate problem was also found: model and market
agree on which team is even favored on only 72.0% of those 93 games -
calibration can rescale confidence magnitude, but can't fix disagreement
about *who's* favored.

`ml_models.fit_probability_calibration` (isotonic or sigmoid/Platt, see
`scripts/train_game_pick_calibration.py`) was built to rescale the
heuristic's own raw probability against real resolved outcomes, wired
live via `game_picks.apply_calibration` (called from `pipeline.run()`
and the backtest reconstruction of "what today's live code would
produce") with the same graceful-degradation contract every other
optional ML artifact in this project already uses - falls back to the
raw, uncalibrated heuristic when no artifact exists or hasn't cleared
its own bar.

**Real dispatched result** (GitHub Actions run 32759747063, full
persisted log, 143 distinct dates, 1,578 train-pool rows / 274 final
holdout rows - 111 rows with a NaN raw probability, a genuine missing-
composite-data date, dropped before fitting): walk-forward CV picked
sigmoid (tied with isotonic at mean log_loss=0.6903). On the real,
untouched final holdout, calibrated (sigmoid) log_loss=0.6864 beat the
raw heuristic's 0.6933 - but did **not** beat the naive always-predict-
base-rate baseline (0.6814), so it was correctly **not saved** under
this project's strict "must beat both" bar. Notably, the calibrated
probability's spread on this holdout was narrower than the raw
heuristic's (std 0.0344 vs. 0.0641), the opposite direction of the
correction that's actually needed - a real sign this particular
population/fit isn't the fix, not a bug in the calibration code itself.

**Status: not shipped, reported honestly.** `game_picks.apply_calibration`
and the training script both stay live in the codebase (graceful no-op
until a real artifact clears the bar), but no artifact exists yet at
`config.GAME_PICK_CALIBRATION_MODEL_PATH`. The deeper favored-team-
disagreement finding (72.0% agreement) is a real, separate open
question a rescaling fix can't address on its own.

### Real infra fix: per-month Statcast persistence (2026-08-24)

Backfilling the real 2025 season (`scripts/backfill_statcast_season.py`,
built so this project's historical training data - independent of any
market-odds history - isn't limited to the current season only) produced
a single ~110.6MB `data/raw/statcast_2025.parquet`, and GitHub's real
push limit rejected it outright ("GH001: Large files detected...exceeds
GitHub's file size limit of 100.00 MB", a hard rejection, not a soft
warning) - all 742,080 fetched pitch rows were lost when the runner tore
down. The live production season's own `statcast_2026.parquet` was
already at 85MB with roughly half the season elapsed, putting the daily
pipeline's own commit step on the same real path to failure.

Fix: `data.persist_raw_statcast`/`load_persisted_statcast` now split by
real calendar month (`data/raw/statcast_<season>_<month>.parquet`)
instead of one file per season, keeping the exact same `(raw_dir,
season)` signature and still returning one combined DataFrame - none of
this project's ~20 existing callers needed changes. The real
`statcast_2026.parquet` was migrated (579,217 rows, byte-exact verified)
into 6 per-month files (~4-20MB each); a legacy-fallback read path keeps
this project's existing synthetic test fixtures (which write directly to
the old single-file layout) working unchanged.

Re-running the 2025 backfill after the fix succeeded cleanly: **742,080
real pitch rows across 190 distinct dates, 2,531 distinct games**, split
into 7 per-month files (largest 21.5MB), pushed without incident.

### Real infra fix: retry-with-rebase on every workflow's final push (2026-08-24)

`ml_training_update.yml`'s own training run - 50+ minutes of real work
before it ever reaches its final `git push` - lost that entire run's
trained artifacts to a plain non-fast-forward push rejection **twice in
a row** in practice (real GitHub Actions runs 32759747063 and
32766340800, both racing against a concurrent code push to the same
branch elsewhere in this same session). A single `git push` with no
retry was never going to survive a long-running job racing a human/
Claude actively pushing code to the same branch.

Fix: every workflow that commits+pushes generated output now retries
with a `git pull --rebase` between attempts (up to 5), not just
`ml_training_update.yml` - `backfill_statcast_season.yml`,
`daily_update.yml` (the core PRODUCTION pipeline, which runs on `main`
and can just as easily race a concurrent PR merge), `backtest.yml`,
`age_curves_update.yml`, `backfill_market_odds.yml`, and
`build_nfl_bestball_rankings.yml`. Safe in every case: each workflow's
commit step only ever touches its own specific path(s)
(`data/models/`, `data/raw/`, `docs/data/*.csv`, etc.), so rebasing onto
whatever else landed on the branch never has a real conflict to resolve.

### Save-gate policy change: drop the naive-baseline requirement (2026-08-24)

Every `train_*.py` script that fits a candidate ML replacement for an
existing heuristic (`train_game_pick_calibration.py`,
`train_game_pick_model.py`, `train_hitter_hit_model.py`,
`train_dfs_ml_models.py`, `train_age_curve_hr9_model.py`) previously
required a candidate to beat BOTH a naive always-predict-the-base-rate
baseline AND the existing heuristic before saving - exactly what
rejected the calibration result above (it beat the heuristic but not
the baseline). Explicit user direction: "our goal should be to get more
and more accurate, so as long as it beats our current model, save it."
The naive-baseline requirement is dropped from all five gates - the
naive baseline is still computed and printed for context (it's a useful
sanity signal), but no longer blocks a save. A candidate now saves
whenever it beats whatever's actually live for that signal today (the
heuristic, or KNN for HR9).

**Known limitation, not addressed by this change**: none of these five
scripts currently compare a new candidate against a *previously-saved*
ML artifact if one already exists (`hitter_hit_probability_model.joblib`
and `age_curve_hr9_model.joblib` both already exist) - each retrain only
re-compares against the heuristic/KNN, same as before. A retrain could
in theory overwrite an already-good saved model with a new one that
still beats the heuristic but is worse than what it's replacing. Real,
open follow-up, not fixed here.

**Also found and corrected while touching this code**: `train_age_curve_hr9_model.py`'s
save message claimed a saved artifact was "now live for HR9" - false
found via grep across `src/` (`AGE_CURVE_HR9_MODEL_PATH` is referenced
only in `config.py` and the training script itself, never loaded or
consulted by `age_curve.py`/`age_curve_ml.py`). Corrected to the same
honest "artifact only, NOT wired into live picks" wording every other
non-wired script already uses. Wiring it live is a separate, unstarted
task.

### Real feature-gap audit: home/road, day/night, weather, rest, umpire (2026-08-24)

Direct answer to "have we taken into account home/road splits, day/night
splits, weather, or other common features" for the game-pick model
(`game_picks.GAME_PICK_FEATURE_COLUMNS`, 10 composite/bullpen/starter
columns) - checked against the codebase and the real persisted Statcast
schema (119 columns), not assumed:

- **Home/road**: no explicit home-field-advantage term anywhere.
  `compute_game_win_probabilities` is a pure `home_rating / (home_rating
  + away_rating)` ratio with no home-field intercept/boost - real MLB
  home teams win ~53-54% historically, and none of that prior is baked
  in beyond whatever the composite ratings happen to encode (nothing
  home/away-specific). The hitter-level DFS model does carry `is_home`
  as a feature; the game-pick model does not carry anything equivalent.
- **Day/night**: not used anywhere. Real Statcast doesn't carry a direct
  `day_night` column, but `sv_id`'s embedded timestamp could derive it
  (or `game_type`/schedule fetch, unconfirmed without more digging) -
  no dedicated column has been built.
- **Weather**: real Statcast's 119 real columns (confirmed directly
  against `data/raw/statcast_2026_08.parquet`) carry no temperature,
  wind, humidity, or roof/dome column at all - not filtered out, never
  present. A weather feature would need a separate real data source
  (e.g. a stadium-keyed weather API), not something already sitting
  unused in persisted data like the batted-ball-quality/pitch-arsenal
  features earlier quant-analytics items found. Month-as-weather-proxy
  is unused too.
- **Rest days**: real, confirmed, unused. `batter_days_since_prev_game`/
  `batter_days_until_next_game`/`pitcher_days_since_prev_game`/
  `pitcher_days_until_next_game` are real columns already sitting in
  every persisted Statcast row (confirmed via grep - zero references
  anywhere in `src/`) - the exact same "sitting unused in already-
  persisted data" pattern quant-analytics item #1 found for batted-ball
  quality and pitch arsenal.
- **Umpire**: real, confirmed, unused. Statcast's own `umpire` column
  (home-plate umpire ID) is never referenced in `src/` - some umpires
  are real, well-documented hitter's/pitcher's umps.

None of this is built yet - reported as a real, verified gap audit, not
a commitment. Rest days and umpire are the most promising next slice
(real signal already sitting in persisted data, zero new fetch needed,
same low-cost pattern every prior quant-analytics item has followed);
weather would be the most expensive (a genuinely new external data
source, not just an unused column).

### Real build: hitter home/road split with wave logic (2026-08-24)

Direct follow-up to the feature-gap audit above. Explicit user direction:
"for each of our features, they should be taken with wave logic (someone
or a whole team might randomly struggle versus lefties or at home, for
example)" - a home/road split needs the exact same small-sample-noise
guard the platoon split (`WAVE_L`/`WAVE_R`) already gets, not a flat
season-long home-vs-away average that a short unlucky/lucky stretch could
distort.

`hitters.compute_home_road_split` (`WAVE_Home`/`WAVE_Away`) reuses
`_blend_windows`/`_side_window_agg`'s already-generalized `column`
parameter (previously proven for `compute_pitch_family_rates`'s pitch-type
split) - same `config.WAVE_WINDOWS` recency blend WAVE itself uses, no new
window scheme. The split key is Statcast's own `inning_topbot` ("Bot" =
bottom of the inning = the home team batting, "Top" = away) - directly
tells us, per PA, which side of the split it belongs to, with no need to
resolve the batter's own team identity. Wired into
`dfs_ml.HITTER_FEATURE_COLUMNS` (widening the hitter model's schema,
same "MUST be retired, not left in place" policy this project has applied
to every prior schema-widening slice - `hitter_hit_probability_model.joblib`
retired here too).

**Scope, honestly**: this is the hitter-level half of "someone... might
randomly struggle... at home." The team-level half ("a whole team might
randomly struggle... at home") - a real home-field-advantage term for
`game_picks.compute_game_win_probabilities`, recency-windowed the same
way - is real, separate follow-up work: it needs team game-by-game
home/away win/loss history (not currently assembled anywhere in this
project) rather than reusing existing per-PA infrastructure, so it isn't
bundled into this same change. Day/night is still unbuilt too (see the
audit above - Statcast carries no reliable, timezone-safe day/night
signal for historical PAs; a real derivation needs a new data source,
not just an unused column).

Full test suite: 678 passed (3 new - exact windowed arithmetic mirroring
`compute_pitch_family_rates`'s own test shape, unrecognized-`inning_topbot`
exclusion, and side-absent-reads-zero-not-dropped).

### Real build: rest days + umpire factor, tested before committing (2026-08-24)

Direct follow-up to the feature-gap audit's other two identified,
already-persisted-and-unused signals. Explicit user direction this time:
"use the identified features that we can use (whether derived or not).
Do a feature search and test feature significance before committing to
the model" - unlike `WAVE_Home`/`WAVE_Away` above (wired straight into
`dfs_ml.HITTER_FEATURE_COLUMNS`), these two go through a real
significance test FIRST, and only get wired into the live model if they
clear a real bar.

- `teams.compute_umpire_factor`: one row per real home-plate umpire
  (Statcast's own `umpire` id), that umpire's real hit rate allowed
  normalized to the across-all-umpires league average - same ratio
  convention as `Park_Factor`.
- `dfs_backtest.assemble_hitter_hit_log` now also carries two EXPLORATORY
  candidate columns, NOT part of `HITTER_FEATURE_COLUMNS` (so the live
  model's schema and every existing caller are completely unaffected):
  `Days_Rest` (Statcast's own real `batter_days_since_prev_game` for
  that date's game - already a real, no-lookahead fact, no derivation
  needed) and `Umpire_Factor` (computed from history STRICTLY BEFORE
  that date, same no-lookahead discipline every other feature here
  already follows, joined via that date's real home-plate umpire).
- `scripts/train_hitter_hit_model.py`'s existing significance report
  (`CANDIDATE_FEATURE_COLUMNS`) now tests both candidates - univariate
  AND combined with the full live feature set - alongside every existing
  feature, on the same real historical log, via the same
  `statsmodels.Logit` methodology already used for every other
  significance check in this project.

**Decision deferred to real dispatched numbers, not guessed**: neither
candidate is added to `HITTER_FEATURE_COLUMNS` in this change - that
only happens in an honest follow-up once a real GitHub Actions dispatch
reports their actual p-values. Reported here regardless of outcome, same
"non-significant is a real, useful answer" discipline the rest of this
project's significance reports already follow.

Full test suite: 682 passed (6 new - `compute_umpire_factor`'s own exact
arithmetic/missing-umpire/empty-input cases, plus
`assemble_hitter_hit_log`'s schema and real-value-population coverage
for both candidates).

**Real dispatched result (GitHub Actions run 32775346481, full history,
n=34,086 rows across 140 dates)** - neither candidate is added to
`HITTER_FEATURE_COLUMNS`:

- **`Days_Rest`**: univariate coef=-0.0300, p=**0.0064** (real, more rest
  correlates with a LOWER hit probability - plausibly IL/injury-stint
  rest rather than a true rust effect) - but combined with the full
  existing feature set, coef flips to +0.0057, p=**0.8054** (not
  significant). The univariate signal doesn't survive once the model
  already accounts for everything else - likely confounded with
  recency/playing-time signals already present (WAVE, `Last_Game_Date`-
  adjacent features), not independent information. **Not added.**
- **`Umpire_Factor`**: excluded from the report entirely - constant
  (zero variance) in the real data. Root cause confirmed directly
  against the real persisted file: Statcast's own `umpire` column is
  **0 non-null out of 92,154 real rows** in `data/raw/statcast_2026_08.parquet`
  - this project's real pybaseball/Statcast pull simply never populates
  it, a genuine real-world data-availability gap in the public feed, not
  a bug in `compute_umpire_factor` or the join logic. A real umpire
  signal would need a different data source entirely (e.g. a dedicated
  umpire-assignment feed) - out of scope here. **Cannot be tested, let
  alone used, with this project's current data.**

Real side note from the same report: `WAVE_Home`/`WAVE_Away` (already
live, shipped in the previous change) are strongly significant
univariately (p<0.0001 each) but not in the combined multivariate fit
(p=0.44/0.97) - the same multicollinearity-with-`WAVE`/`WAVE_L`/`WAVE_R`
pattern several other platoon-adjacent features already show in this
report, not evidence against the feature (see the significance report's
own docstring on standardization/collinearity).

The hitter hit-probability model retrained successfully under the
widened schema and saved (sigmoid calibration, beats `Game_Hit_Probability`
- artifact only, not wired live). The game-pick calibration artifact
also **finally saved** under the relaxed gate from the policy change
above (beats the raw heuristic, 0.6864 vs 0.6933) - wired live via
`game_picks.apply_calibration`, so live game picks are now genuinely
rescaled starting from this run.

### Real build: bullpen rest/readiness candidate, tested before committing (2026-08-25)

Direct follow-up ("what about bullpen rest/readiness"), same feature-
search discipline as the hitter-side candidates above, applied to the
game-pick model this time. `game_picks.GAME_PICK_FEATURE_COLUMNS`
already has `home_bullpen_pave_plus`/`away_bullpen_pave_plus` - a
season-long QUALITY aggregate - but nothing measuring current bullpen
FATIGUE: a bullpen can be excellent on paper and still gassed from three
straight extra-inning games.

- `pitchers.compute_bullpen_recent_workload`: real outs recorded by a
  team's relievers (`is_starter` False) in the `config.BULLPEN_FATIGUE_RECENT_DAYS`
  (2, first-pass/unvalidated) calendar days strictly before the target
  date - a single fixed recency CUTOFF (not a `WAVE_WINDOWS`-style multi-
  window blend), since this is a workload TOTAL, not a rate needing
  small-sample smoothing.
- `game_picks_backtest.assemble_game_pick_log` now also carries
  `home_bullpen_recent_outs`/`away_bullpen_recent_outs` as EXPLORATORY
  columns, NOT part of `GAME_PICK_FEATURE_COLUMNS` (live model schema
  unaffected).
- `scripts/train_game_pick_model.py`'s existing significance report now
  also tests both candidates, same `statsmodels.Logit` methodology as
  every other significance check in this project.

Real, already-existing infra reused directly: `data.label_pitcher_roles`/
`pipeline.build_pitcher_events_with_role` (the same role-labeling
`compute_bullpen_pave` already depends on) - no new fetch, no new
identity-resolution logic, same "game_pk is the real game id, not
`data.assign_game_ids`" convention this module already established.

Decision deferred to real dispatched numbers, same as the hitter-side
candidates: neither column is added to `GAME_PICK_FEATURE_COLUMNS` in
this change.

Full test suite: 686 passed (4 new - `compute_bullpen_recent_workload`'s
own exact arithmetic/empty-input/no-relief-in-window cases, plus
`assemble_game_pick_log`'s real-value population for both candidates).

**Real dispatched result (GitHub Actions run 32792241148, n=1,963 games)**
- clean, unambiguous negative, neither candidate added to
`GAME_PICK_FEATURE_COLUMNS`:

- `home_bullpen_recent_outs`: univariate p=**0.4727**, combined p=**0.6545**.
- `away_bullpen_recent_outs`: univariate p=**0.8985**, combined p=**0.6321**.

Neither clears any bar at all, univariate or combined - not a borderline
call like `Days_Rest`'s. The 2-day recency window
(`config.BULLPEN_FATIGUE_RECENT_DAYS`, explicitly flagged as a first-pass/
unvalidated choice) is the most likely place a real signal could still be
hiding - a different window length, or a rate-based measure (e.g. outs
per relief appearance, not a raw total) instead of a fixed lookback
total, are real, cheap follow-ups if this is worth revisiting; not
pursued further here since the finding is this clean.

Real side benefit of dispatching this: `train_game_pick_model.py`'s own
walk-forward predictive model (a full from-scratch LogisticRegression
replacement for the heuristic, previously reported as failing to beat
it) **saved for the first time** under the relaxed no-baseline-required
gate - `game_pick_win_probability_model.joblib`, artifact only, not
wired into live picks.

### Real build: broader bullpen-fatigue sweep, tested before committing (2026-08-25)

Direct follow-up to the clean negative above: "I want to see if other
applications of bullpen fatigue are significant, but I don't care if
they're cheap" - the 2-day recent-outs total was one specific way to
operationalize "bullpen fatigue," not the only one. Real, distinct
hypotheses added and tested, not micro-variations of the same idea:

- **Window sweep** (`config.BULLPEN_FATIGUE_CANDIDATE_WINDOWS = [1, 3, 5]`):
  the same `pitchers.compute_bullpen_recent_workload` at additional
  recency horizons - does the lookback length matter, independent of
  what's being measured.
- **`pitchers.compute_bullpen_distinct_relievers`**: workload BREADTH
  (how many different arms got used) instead of a raw outs total - the
  same total workload spread across many fresh arms is a very different
  bullpen state from that same workload concentrated on a couple of arms.
- **`pitchers.compute_bullpen_back_to_back_relievers`**: a sharper "which
  SPECIFIC arms are on zero rest" signal - counts relievers who appeared
  on both of the two most recent calendar dates (not "the last two games
  played," so a real scheduled off day correctly reads as nobody being
  back-to-back, rather than comparing across the gap).

All EXPLORATORY - `game_picks_backtest.BULLPEN_FATIGUE_CANDIDATE_COLUMNS`
is the single source of truth for the full column list, reused directly
by `train_game_pick_model.py`'s `CANDIDATE_FEATURE_COLUMNS` rather than a
second hardcoded copy. None of these are part of
`GAME_PICK_FEATURE_COLUMNS` in this change.

Tests: 6 new (`compute_bullpen_distinct_relievers`/
`compute_bullpen_back_to_back_relievers`'s own exact arithmetic -
including the off-day-gap-reads-zero case - plus real-value population
in `assemble_game_pick_log`). Full suite: 692 passed.

**Real dispatched result (GitHub Actions run 32813871428, n=1,963
games)** - a comprehensive negative across all 12 candidates. None comes
close to significant, univariate or combined:

| candidate | univariate p | combined p |
|---|---|---|
| `home_bullpen_recent_outs` (2d) | 0.4727 | 0.7405 |
| `away_bullpen_recent_outs` (2d) | 0.8985 | 0.8879 |
| `home_bullpen_recent_outs_1d` | 0.4153 | 0.9695 |
| `away_bullpen_recent_outs_1d` | 0.6632 | 0.5515 |
| `home_bullpen_recent_outs_3d` | 0.2987 | 0.8388 |
| `away_bullpen_recent_outs_3d` | 0.6031 | 0.2202 |
| `home_bullpen_recent_outs_5d` | 0.0912 | 0.3111 |
| `away_bullpen_recent_outs_5d` | 0.8869 | 0.6715 |
| `home_bullpen_distinct_relievers` | 0.5920 | 0.8274 |
| `away_bullpen_distinct_relievers` | 0.9725 | 0.9815 |
| `home_bullpen_back_to_back_relievers` | 0.8397 | 0.9347 |
| `away_bullpen_back_to_back_relievers` | 0.4511 | 0.6267 |

The closest thing to a signal - `home_bullpen_recent_outs_5d` at
p=0.0912 univariate - doesn't clear even the loosest conventional bar
(0.05), and drops further to p=0.3111 combined. Window length, workload
breadth, and specific-arms-on-zero-rest all tell the same story: none
of these bullpen-fatigue formulations shows real signal in this
project's data. **None added to `GAME_PICK_FEATURE_COLUMNS`.**

Real, honest interpretation: this doesn't prove bullpen fatigue doesn't
matter to real outcomes - it means none of these five ways of measuring
it, from real Statcast relief-appearance data alone, predicts who wins
the NEXT game beyond what `home_bullpen_pave_plus`/`Bullpen_PAVE_PLUS`
(the existing season-long quality signal) already captures. A genuinely
different data source (real bullpen-availability/usage-plan reporting,
not derivable from box-score-level Statcast) would be a different,
separate undertaking, not a variation on this same approach.

Real side note: the already-live `home_bullpen_pave_plus`/
`home_bullpen_power_a_plus` (quality, not fatigue) DO show real
univariate significance here (p=0.0442/p=0.0398) that fades in the
combined fit (p=0.4989/p=0.7216) - the same multicollinearity pattern
several other features in this report already show, not a new finding.

### Dashboard: Hit Streaks and Model Odds

The Beat the Streak section of the dashboard has three subtabs: **Our
Picks** (the official picks above - the model shortlists on days it's
available, per the wiring above, then `Approach`/`Matchup_Approach` picks
the final order), **Hit Streaks**, and **Model Odds**. The latter two
remain purely **informational** views alongside the official picks.

- **Hit Streaks** (`hitters.compute_current_hit_streaks`,
  `scripts/build_hit_streaks.py` → `docs/data/hit_streaks.csv`): each
  recently-active batter's real current consecutive-games-with-a-hit
  streak, counted from real completed Statcast events. A batter whose most
  recent game is more than `config.HIT_STREAK_RECENT_DAYS` (5) days old is
  excluded entirely, so an inactive/injured player's frozen streak doesn't
  crowd out who's actually hot right now. Never consulted by
  `predictions.select_picks` - informational only, same as before.
- **Model Odds** (`dfs_ml.predict_hitter_hit_probability`,
  `scripts/build_hitter_hit_predictions.py` → `docs/data/hitter_hit_predictions.csv`):
  today's PA-qualified hitters ranked by the trained hit-probability
  model's own predicted probability - the same model that gates the
  shortlist for Our Picks on days it's available, shown here
  independently for anyone who wants to see the model's own full ranking
  rather than just who it shortlisted.

Both scripts follow `build_dfs_rankings.py`'s resilience conventions
(missing input/model/failed schedule fetch leaves yesterday's output in
place rather than crashing or overwriting with nothing) and are wired into
`daily_update.yml` right after "Build DFS rankings."

## Automated Game Picks (dashboard)

A second, independent dashboard section predicts a winner for each of
today's games (not hitters) from six team-level signals: each team's
Pythagorean strength (`pyth_Strength`), Pythagorean confidence
(`pyth_Confidence`), suppression resistance (`suppression_resistance`), and
true power (`true_power`) - all from `confidence.csv` - adjusted by the
specific pitching each team is projected to face, via the same
clip-then-blend logic `matchup.py` uses for hitters
(`matchup.clip_and_blend_pitching_quality`, shared by both).

Pitching quality itself blends two complementary signals from the probable
starter and that team's bullpen: `PAVE_PLUS` (hit-rate against) and
`Power_A_PLUS` (total-bases-allowed rate against - a run-prevention/
ERA-like signal computed the same recency-windowed way as PAVE, see
`pitchers.py`; deliberately not raw ERA, which isn't computed from this
project's own Statcast data). `PAVE_PLUS` alone treats every hit allowed
identically, so a pitcher who mostly allows singles and one who mostly
allows home runs could look the same despite very different run-prevention
profiles - `config.GAME_PICK_SUSCEPTIBILITY_WEIGHT` (default 0.5, an equal
blend) controls the split, empirically validated via a 35-day
persisted-Statcast backtest: the 0.5/0.5 blend beat pure `PAVE_PLUS` on both
accuracy (56.3% vs 54.3%) and Brier score (0.2493 vs 0.2517).

- Every scheduled game is published (`game_predictions.select_game_picks`),
  flagged `above_threshold` if the favored side's win probability clears
  `GAME_PICK_MIN_PROBABILITY` (config.py, default 0.58 - much lower than the
  hitter picks' 0.77, since single-game MLB win probabilities are
  compressed near 50/50 even for real favorites) - the dashboard highlights
  the flagged games in Today's Picks rather than only ever showing them.
  `game_evaluation.build_game_picks_export`'s accuracy/streak numbers stay
  scoped to the `above_threshold` subset only - a below-threshold game's
  real outcome never moves those, exactly as before this change; only what
  gets *published* got broader.
- `home_win_probability = home_rating / (home_rating + away_rating)`, where
  each team's rating is its own offensive composite (equal-weighted blend
  of the four signals above, `GAME_PICK_COMPOSITE_WEIGHTS`) multiplied by
  the *opposing* team's pitching-weakness multiplier. This is a simple
  ratio, not log5 - these composites aren't calibrated win percentages, so
  a ratio is the honestly-explainable choice rather than borrowing false
  precision from a formula built for a different kind of input.
- Tracked with a plain win/loss accuracy and a simple consecutive-correct
  streak (`game_evaluation.build_game_picks_export`) - not Beat the
  Streak's reset-on-any-miss multi-pick rule, since that's MLB's specific
  hitter-streak game mechanic and doesn't apply to independent game-by-game
  picks.

This reads `docs/data/game_picks_picks.csv` and `game_picks_summary.csv`,
written after every daily run from `data/predictions/game_predictions.csv`
(a separate log from the hitter-pick `predictions.csv`, keyed on `game_pk`
instead of a player id). Resolution uses final scores fetched via
`schedule.fetch_game_results`, not Statcast.

**Historical backtest** (`scripts/run_game_picks_backtest.py`,
`game_picks_backtest.py`): unlike probable-pitcher/live-schedule data
(never persisted - see `Matchup_Hit_Probability`'s equivalent limitation
above), schedule/game *outcomes* are already fully reconstructable from
data this project already has: `confidence.csv`/`pave.csv` have been
committed daily (same commit as `wave.csv`) since the project started, in
exactly the as-of-that-date snapshot shape the model needs, and Statcast
itself already contains each game's real `game_pk`, actual starting
pitchers, and actual final score - no separate schedule-API history
needed. This replays the last N days (default 40) of `confidence.csv`/
`pave.csv` git history through `game_picks.compute_game_win_probabilities`,
using the *actual* starter in place of the announced "probable" one (a
reasonable stand-in in hindsight - the two agree the vast majority of the
time) and resolves every pick immediately against the real final score,
since a backtested pick's outcome is already known at reconstruction time
(no pending state, unlike a live pick). Writes into the same
`data/predictions/game_predictions.csv` log and
`docs/data/game_picks_picks.csv`/`game_picks_summary.csv` exports used by
the live daily system, exactly like `git_backtest.py` does for hitter
picks - so the dashboard shows one continuous history, not two separate
tracks.

This is a first-pass, unvalidated blend (see `game_picks.py`'s module
docstring), meant to be watched and compared against reality before being
trusted, not treated as ground truth.

### Game-pick logistic regression (`scripts/train_game_pick_model.py`, fit + report only)

Same two-phase pattern as the hitter-side model above, applied to the
team-level heuristic: `game_picks.build_game_features` exposes the ten raw,
unblended per-team/matchup ingredients (`home_composite`, `away_composite`,
`home`/`away_bullpen_pave_plus`, `home`/`away_bullpen_power_a_plus`,
`home`/`away_starter_pave_plus`, `home`/`away_starter_power_a_plus`) that
`compute_game_win_probabilities` combines into `home_win_probability`,
instead of only ever returning the already-blended ratio -
`compute_game_win_probabilities` itself is refactored to build on top of
`build_game_features` (a behavior-preserving change; all pre-existing
`test_game_picks.py` tests pass unmodified).
`game_picks_backtest.assemble_game_pick_log` replays every historical game
(no-lookahead, one row per `game_pk`, reusing the same as-of-date
`pipeline.compute_outputs` recompute `reconstruct_historical_game_picks_from_persisted`
uses) into a training table of those ten features, the heuristic's own
`home_win_probability` (carried through as a comparison column, not fed to
the model), and the real `Home_Won` outcome.

**Real numbers (2026-07-29, full persisted history, n=1,595 games across
121 dates)**: individually, only `home_starter_pave_plus` (p=0.0256) and
`home_starter_power_a_plus` (p=0.0335) are significant at p<0.05 -
`home_bullpen_pave_plus`/`home_bullpen_power_a_plus` are marginal
(p≈0.06); `home_composite`, `away_composite`, and all four away-side
pitching features are not significant. In the combined multivariate model
nothing clears p<0.05. On a 20-date holdout (n=264), the walk-forward
`LogisticRegression` essentially ties a coin flip (accuracy 0.500, ROC AUC
0.5035, log_loss 0.6937) and does not even beat the naive baseline
(0.6929), let alone the existing `home_win_probability` heuristic (log_loss
0.6874, ROC AUC 0.5603, clearly the stronger signal here) - **not saved**,
reported honestly per the same bar the hitter-side model had to clear.
Unlike the hitter side, this fit-and-report phase found no edge to
capture: the composite/bullpen signals `game_picks.py` already blends
carry real information the raw per-team features alone don't obviously
improve on with this data volume, at least not via a plain logistic
regression on today's feature set.

**Scope**: fit and report only, same as the hitter-side script - no
artifact was produced this run, and even if a future re-run (as more
history accumulates) does clear the bar, the design for actually using it
(nudging the heuristic's favored team's probability slightly toward the
model's pick, calibrated via its own backtest, mirroring how
`GAME_PICK_SUSCEPTIBILITY_WEIGHT` and `GAME_PICK_MIN_PROBABILITY` were both
calibrated rather than guessed) remains a separate, later decision.

## Probable Pitchers (dashboard)

A small daily list of today's games with each probable starter, their
throwing hand (`Throws`), and `PAVE`/`PAVE_PLUS`/`Power_A_PLUS`
(`schedule.build_probable_pitchers_table`),
written to `docs/data/probable_pitchers.csv` alongside the matchup blend
(same resilience as the rest of Part B - a failed/empty schedule fetch means
no file that day, not a stale one). A team with an unannounced or
unmatched starter still gets a row with blank pitcher fields, so the day's
full slate stays visible rather than silently missing teams.

## Age Curves (`docs/age-curves.html`, exploratory - separate page)

A standalone page, entirely separate from the daily pick pipeline: pick a
current hitter or pitcher and see how their season stat line compares to
historical players at the same age, plus a projection for next season
built from what those historical comparables actually did the following
year.

Uses Lahman's Baseball Database (1871-2025), read from CSVs committed
under `Lahman_Raw/` at the repo root (`lahman_data.py`) - downloaded by
hand from [SABR's mirror](https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd)
and refreshed manually, roughly once a season (see `Lahman_Raw/Readme.md`
for the refresh steps). Two automated alternatives were tried first and
both failed: pybaseball's own built-in `lahman` submodule downloads a zip
from the baseballdatabank GitHub repo at call time, and that download
started failing (`zipfile.BadZipFile`, reproduced in real CI with full
internet access - an upstream break, not a local network issue); the
`lahman` PyPI package needs no network fetch but turned out to be a single
abandoned release from 2022 with data frozen at the 2020 season. A
manually-refreshed, committed CSV sidesteps both failure modes.

A different data source than the rest of this project (Statcast), needed
because Lahman's historical seasons have no Statcast-derived signal
(WAVE/PAVE) to compare against. A current player is
put on the same stat basis - season `AVG`/`OBP`/`SLG`/`OPS` for hitters,
`K9`/`BB9`/`HR9`/`FIP` for pitchers (`traditional_stats.py`, reusing
`helpers.py`'s event classifiers, not recency-windowed like WAVE) - so
they're comparable to a single historical season. `key_mlbam` (this
project's own identity) bridges to Lahman's `playerID` via a two-hop
crosswalk: `chadwick_register()`'s `key_bbref` column matches Lahman
People's `bbrefID` (`lahman_data.build_crosswalk`).

Pitcher metrics deliberately **exclude ERA**: this project already avoids
ERA for its live pick models (`Power_A_PLUS` - ERA is too dependent on
defense/sequencing/inherited runners to isolate a pitcher's own
performance) and applies the same reasoning here. `K9`/`BB9`/`HR9` are the
three defense-independent "own stuff" component rates (mirroring
`AVG`/`OBP`/`SLG`'s role on the hitter side); `FIP` combines those same
three components into one number (mirroring `OPS`), using a fixed
constant (`config.FIP_CONSTANT = 3.10`, a pure additive shift
that doesn't affect comparable-search distances or projections) rather
than a real per-season constant, which would need ERA/runs data.

The page computes a **separate curve/projection per metric**
(`config.AGE_CURVE_HITTER_METRICS`: `AVG`, `OBP`, `SLG`, `OPS`;
`config.AGE_CURVE_PITCHER_METRICS`: `K9`, `BB9`, `HR9`, `FIP` - all
selectable on the page) rather than one blended number - power (`SLG`)
typically peaks earlier and declines faster than plate discipline (`OBP`),
while contact rate (`AVG`) tends to be more stable, and pitching
components age differently from each other too, so collapsing them into
one composite would hide that. A player's "closest historical comparables"
can therefore be a genuinely different set of players depending on which
metric is selected (e.g. their power comps vs. their contact comps).

`age_curve.py`'s method (v1, deliberately simple - see its module
docstring), run independently for each metric: find the
`AGE_CURVE_K_NEIGHBORS` (25) historical same-position seasons within
`AGE_CURVE_AGE_WINDOW` (1) year of the current player's age with the
closest value on that metric (a hand-implemented nearest-neighbor sort,
matching this project's pattern of implementing its own stats rather than
pulling in an ML library for one call - unchanged even though
scikit-learn is now a real dependency for the HR9 follow-up below), then
look at what those comparables actually did on that same
metric the *following* season. Comparables with no next season on record
(retired, hurt, released, or fell below `AGE_CURVE_MIN_AB`/`AGE_CURVE_MIN_IP`)
are excluded from the projection and that excluded fraction is reported
alongside it - survivorship bias made visible, not hidden. The projection
is a range (mean plus 25th/75th percentile), not a false-precision single
number.

The chart plots the player's own **real trajectory**, not just their
current season: for a player who already has Lahman-tracked history
(anyone who debuted before Lahman's most recent completed season), every
one of their own past ages on record is plotted as an actual point,
connected into a real multi-year line, with their current season and a
distinctly-marked projected next-season point at the end
(`age_curve_player_history.csv`, built by matching each current player's
own crosswalked Lahman `playerID` back into their own historical rows -
their real career arc, not a comparable's). A player with no Lahman
history yet (a very recent debut) only has their current + projected
points, same as before.

Known v1 simplifications (documented, not silently ignored): each metric's
comparable search is **independent** (ranked by that one stat alone, not a
joint multi-metric similarity - a future "one holistic comparable set,
several metric views" mode is a possible follow-up); **no era/park
adjustment** - comparing raw rates across very different offensive eras
(e.g. the high-offense late 1990s vs. a lower-average modern era) or
ballparks is a real limitation of this first pass.

`scripts/fetch_lahman.py` (converts `Lahman_Raw/*.csv` to
`data/raw/lahman/*.parquet`) and `scripts/build_age_curves.py` (writes
`docs/data/age_curve_projections.csv`/`age_curve_comparables.csv`/
`age_curve_league.csv`/`age_curve_player_history.csv`) run on their own
**occasional cadence** (weekly, via
`.github/workflows/age_curves_update.yml` - also manually triggerable) -
deliberately a separate workflow from `daily_update.yml`, since historical
comparables don't move day to day. `scripts/backtest_age_curve.py`
validates the projection method itself against a sample of real past
player-seasons (each with a known real next season to check against),
using only comparable data available at or before that test season's own
year - the same no-lookahead discipline as `git_backtest.py`/
`game_picks_backtest.py` elsewhere in this project.

**Validated against real data**: 500 sampled real player-seasons from
2010-2019 (out of 32,158 qualified historical hitter-seasons / 28,127
qualified historical pitcher-seasons, 1871-2025). Every hitter metric
beat the naive "always guess the sample mean" baseline, with a real
positive correlation between projected and actual next-season value -
strongest for the power/contact-combination metrics:

| metric | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| AVG | 0.0245 | 0.0254 | 0.352 | 355/500 |
| OBP | 0.0274 | 0.0284 | 0.429 | 355/500 |
| SLG | 0.0533 | 0.0582 | 0.473 | 355/500 |
| OPS | 0.0756 | 0.0789 | 0.446 | 355/500 |

(See `config.py`'s `AGE_CURVE_AGE_WINDOW` docstring for the full
methodology note - the 145/500 unscored seasons had no comparable with a
resolvable next season and are reported, not hidden.)

Pitcher metrics, same methodology: `K9` shows a strong signal (the
strongest of any Age Curves metric, hitter or pitcher - strikeout rate is
largely an "own stuff" skill that persists year to year); `BB9` and `FIP`
both clearly beat their baselines; `HR9`'s single-dimension comparable
search *used to be* reported honestly as a wash - it correlated with
next-season `HR9` (year-to-year home-run rate is notoriously volatile,
driven by batted-ball luck/park effects/defense as much as pitcher skill)
but its MAE was statistically indistinguishable from just guessing the
sample mean. A real ML follow-up fixed that (see "Machine learning
follow-up" below) - `HR9` in the table is now the ML result, and the KNN
comparable search remains what every other metric here uses:

| metric | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| K9 | 1.1476 | 1.5878 | 0.746 | 296/500 |
| BB9 | 0.6277 | 0.7776 | 0.616 | 296/500 |
| HR9 (KNN, superseded) | 0.3312 | 0.3264 | 0.321 | 296/500 |
| HR9 (ML, live) | 0.3190 | 0.3264 | 0.359 | 296/500 |
| FIP | 0.5907 | 0.6606 | 0.440 | 296/500 |

### Machine learning follow-up: HR9

`HR9` was the weakest signal in this project by a clear margin (README's
DFS section below has the other three), so it got a real ML attempt
(`age_curve_ml.py`, `scripts/train_age_curve_hr9_model.py`) rather than
further heuristic tuning: a gradient-boosting regression on all six
features together (`age`, `IP`, `K9`, `BB9`, `HR9`, `FIP`) instead of the
KNN path's single dimension (`HR9`'s own value). Trained ONLY on seasons
before 2010 (16,324 rows, 1871-2009) - a stricter, single-global-model
analogue of the KNN backtest's own no-lookahead discipline (year-blocked
cross-validation, `age_curve_ml.YearBlockedSplit`, picks hyperparameters
using only that pre-2010 pool) - then evaluated once on the exact same
500-season 2010-2019 holdout the KNN number above uses.

The honest expectation going in was that this might simply not be
fixable by any model built from a pitcher's own aggregate rate stats
(single-season HR9 is substantially batted-ball-luck/park/defense-driven).
That turned out to be too pessimistic: the model beat both the naive
baseline and the KNN search, and is now the live projection source for
`HR9` specifically - every other metric is untouched, still served by the
original KNN comparable search. See `config.py`'s Age Curves section for
the model's selected hyperparameters and `age_curve_ml.py`'s module
docstring for the full methodology.

## DFS Player Rankings (`docs/dfs.html`, `dfs.py`)

A standalone page ranking today's hitters and probable starting pitchers
by estimated DraftKings Classic MLB fantasy points. Originally a
**ranked list of good plays, not a salary-cap lineup optimizer** (no
salary data is ingested anywhere in this project, by design - the user's
original request was rankings/projections only). An "Optimal Lineup" tab
has since been added on top of these same projections - see "Optimal
Lineup" below for what changed and, critically, why its salaries are a
MODELED estimate, not real DraftKings prices.

DraftKings' scoring rules (`config.DFS_DK_*`) were confirmed live via web
search, not pulled from memory, with sources cited directly in
`config.py`:
[dknetwork.draftkings.com](https://dknetwork.draftkings.com/2020/05/29/beginner-mlb-dfs-scoring/)
and [draftkings.com/help/rules/2/59](https://www.draftkings.com/help/rules/2/59).
Hitters: single=3, double=5, triple=8, HR=10, run=2, RBI=2, BB=2, HBP=2,
SB=5, and (confirmed current as of July 2026) **no caught-stealing
penalty** - DraftKings removed it from the current ruleset. Pitchers:
2.25 points/inning (0.75/out), K=2, win=4, ER=-2, H=-0.6, BB=-0.6,
HBP=-0.6, plus rare complete-game/shutout/no-hitter bonuses.

### Hitters

DK pays non-linearly for hit type (a double isn't 2x a single's value: 5
!= 2*3), but this project only computes a linear `Expected_Bases` signal
(`hitters.compute_wtb`) - no per-player 1B/2B/3B/HR rate breakdown
exists. `DK_Points_Hitter_HitType` approximates the non-linear scoring
with a single calibrated "DK points per expected total base" constant,
`config.DFS_DK_POINTS_PER_TOTAL_BASE = 2.6998`, computed from **real**
Lahman batting 2015-2025 hit-type shares (not a guessed league average -
see the constant's docstring for the full arithmetic), applied to
`Expected_Bases` after scaling it for today's specific matchup via
`dfs.compute_matchup_adjustment` - a ratio of `Matchup_Hit_Probability`
(today's actual opposing pitcher, platoon+park adjusted) to the batter's
own blended `Game_Hit_Probability`.

**Widened beyond hit types (2026-07-26)**: `DK_Points_Hitter` also scores
walks, hit-by-pitch, and RBI on top of `DK_Points_Hitter_HitType`, via
`hitters.compute_extended_dk_rates` - the same recency-windowed,
platoon-blended architecture `compute_wave`/`compute_whops` already use
(`config.WHOPS_WTB_WINDOWS`), reused rather than inventing a new one.
- **BB/HBP**: trivial from the persisted Statcast `events` column -
  `helpers.is_walk_for_dk_scoring` (a walk OR an intentional walk;
  deliberately a separate classifier from the existing `helpers.is_walk`,
  which several already-validated signals - pitcher BB9, Age Curves BB9,
  the pitcher-side DFS backtest - depend on and which must NOT silently
  change scope) and `helpers.is_hit_by_pitch`.
- **RBI**: not a structured Statcast field, so `helpers.estimate_rbi`
  approximates it as `post_bat_score - bat_score` on the batter's own
  plate-appearance-ending row (both columns exist on every persisted row)
  - the standard Statcast RBI-approximation technique. Two accepted
  simplifications: a bases-loaded walk's forced-in run is credited as a
  "BB" event, not separately as an RBI (DK scores it as both; this
  doesn't), and this can't distinguish a batter's own go-ahead RBI from,
  e.g., a run that scored on the same play for an unrelated reason on rare
  multi-run plays. Clipped at 0 (an RBI can't be negative).

**Still excluded**: runs scored by the batter (would need real multi-PA
baserunner tracking across plate appearances within a game/inning - a
real but deferred v1.1 follow-up, not built) and stolen bases (Statcast
has no structured `events` value for a steal at all - it only appears in
the free-text `des` field, and is attributed to whoever was AT BAT during
the steal, not the runner - parsing that reliably would mean matching on
names, which conflicts with this project's numeric-ID-only join
convention everywhere else; explicitly scope-cut, not planned even for
v1.1).

**Backtest reality check**: widening the scored categories did NOT fix
`DK_Points_Hitter`'s underlying weak-signal problem - see "Machine
learning follow-up" below for the numbers and why the ML model, not this
heuristic, is what's actually live.

**Recency exclusion and position subtabs (2026-07-29)**: `scripts/build_dfs_rankings.py`
now drops any hitter whose most recent completed plate appearance
(`hitters.compute_last_game_dates`) is more than
`config.HITTER_MAX_DAYS_SINCE_LAST_GAME` (5) days before `--as-of-date` -
the same gate `predictions.select_picks` already applies to Beat the
Streak picks, reused directly. Without it, a hitter on a season-long
injured-list stay could still qualify on PA alone and keep getting
selected by the Optimal Lineup optimizer below (a real, live bug: Dansby
Swanson, injured for months, kept appearing). It also merges in each
hitter's DK Classic fielding slot (`dk_slot`, `roster_positions.py` - the
same eligibility lookup the optimizer already used) so the dashboard's
Hitters tab can offer per-position subtabs (All/C/1B/2B/3B/SS/OF)
alongside the existing full ranked list. A failed position-eligibility
fetch is non-fatal here (unlike a failed schedule fetch) - today's
rankings still publish, just with a blank `dk_slot` column and no
per-position filtering that day.

### Pitchers

Needed one new signal this project didn't have: `pitcher_form.py`'s
recency-windowed `K9`/`BB9`/`HR9`/innings-per-start, mirroring
`pitchers.compute_pave`'s windowing pattern but applied to start-level
data via its own separate `config.DFS_PITCHER_WINDOWS` (deliberately not
reusing `PAVE_WINDOWS`, since PAVE blends at-bat-level data while this
blends start-level data and needs wider windows to avoid tiny-sample
buckets). `DK_Points_Pitcher` combines expected innings, strikeouts,
walks, and hits allowed (the last via PAVE scaled by
`config.DFS_BATTERS_FACED_PER_INNING = 4.335`, computed from this
project's own persisted 2026 Statcast) plus an estimated earned-run
penalty. `Expected_ER` is derived from a windowed FIP
(`config.FIP_CONSTANT`, shared with Age Curves) rather than a real ER
signal - this project has a consistent house principle against modeling
ERA/earned runs directly, since they're defense/sequencing-dependent
rather than purely the pitcher's own skill (the same reasoning
`Power_A_PLUS` and Age Curves' pitcher metrics already use FIP over ERA
for).

**Explicitly excluded from v1**: Win (needs a win-probability estimate
too dependent on the pitcher's own team's offense to reasonably
approximate) and the rare discrete bonuses (complete game, shutout,
no-hitter). Relief pitchers are entirely out of scope - a meaningful
"innings per appearance" number only exists for a starter, not a
variable-length bullpen outing. A pitcher needs
`config.DFS_PITCHER_MIN_STARTS` (3) recorded starts and to be a team's
announced probable starter today to be ranked at all - no neutral
fallback for an unannounced pitcher.

### Cadence and pipeline

Unlike Age Curves (weekly - historical comparables barely move day to
day), DFS rankings run **daily**, wired directly into
`daily_update.yml` right after the model step
(`scripts/build_dfs_rankings.py`, writing
`docs/data/dfs_hitters.csv`/`dfs_pitchers.csv`), since matchups and
probable starters change every day and a weekly cadence would serve
stale matchups six days out of seven. A failed probable-pitcher fetch or
an empty schedule leaves the previous day's files untouched rather than
overwriting them with nothing.

`dfs_backtest.py` validates the projections with the same no-lookahead
discipline as `game_picks_backtest.py`/`backtest_age_curve.py`: for each
of a sample of real past game dates, `pipeline.compute_outputs` is
recomputed fresh from an as-of-date slice of persisted Statcast (nothing
after that date is visible), and the resulting projections are compared
against that date's real outcomes.

**Validated against real data**: 15 real game dates in July 2026,
recomputed with no lookahead. Reported honestly, not softened - this is
the ORIGINAL heuristic-only backtest; three of these five signals have
since been superseded by validated ML models (see "Machine learning
follow-up" below), kept here for the historical record:

| signal | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| `DK_Points_Hitter` (heuristic, hit-type only, superseded) | 3.7719 | 3.7496 | -0.004 | 4,156 |
| `Expected_IP` | 1.0257 | 1.0332 | 0.351 | 360 |
| `Expected_K` | 1.8971 | 1.9922 | 0.393 | 360 |
| `Expected_BB` (heuristic, superseded) | 1.0570 | 1.0380 | 0.182 | 360 |
| `Expected_H_Allowed` (heuristic, superseded) | 2.6545 | 1.7860 | 0.223 | 360 |
| `DK_Points_Pitcher` (combined, heuristic) | 6.9409 | 7.1970 | 0.306 | 360 |

The hitter ranking's real-world correlation was essentially zero -
`compute_matchup_adjustment`'s ratio (derived from hit-PROBABILITY
signals but applied multiplicatively to a TOTAL-BASES signal, flagged up
front as the single highest-risk modeling choice in this feature) did
**not** hold up at the single-game level. `Expected_H_Allowed`'s MAE was
actually worse than the naive baseline despite a positive correlation,
suggesting the PAVE-to-hits-allowed scaling was systematically off, not
just noisy. `Expected_IP`/`Expected_K` already beat their baselines and
were left alone - the ML follow-up below only targeted the three flagged
signals.

**Hitter scoring widened (2026-07-26)**: `DK_Points_Hitter` now also
scores BB/HBP/RBI, not just hit types (see `dfs.py`'s module docstring for
exactly what's included and why runs/stolen bases still aren't - no
reliable structured signal exists for either in the persisted Statcast
data). Re-running the same heuristic backtest against 20 recent game dates
with the widened scoring: MAE 4.7569 vs. 4.7457 naive-baseline MAE,
correlation 0.009 (n=5,430 scored). Widening the scored categories did
**not** fix the underlying weak-signal problem - the heuristic's
correlation is still essentially zero, because the flawed
`compute_matchup_adjustment` ratio still drives most of the point total
(`DK_Points_Hitter_HitType`) and the new BB/HBP/RBI terms are additive on
top of it, not a structural fix. The live default for `DK_Points_Hitter`
is the ML model below (trained directly on raw ingredients, bypassing
that ratio), not this heuristic.

### Machine learning follow-up: hitter and pitcher models

`DK_Points_Hitter`, `Expected_H_Allowed`, and `Expected_BB` each got a
real ML attempt (`dfs_ml.py`, `scripts/train_dfs_ml_models.py`) rather
than further heuristic tuning, once the numbers above showed they weren't
working. Trained on the FULL persisted 2026 Statcast history, with
walk-forward blocked cross-validation (`ml_models.WalkForwardDateSplit`)
for grid search and a final holdout (the most recent `config.ML_FINAL_HOLDOUT_DATES`
= 20 dates) the grid search never sees at all - the reported numbers below
come exclusively from that untouched holdout, refit on everything before
it, so they're a fair head-to-head against the original 15-20 date
heuristic numbers.

**Retrained 2026-07-26** after `DK_Points_Hitter`'s scoring widened to
include BB/HBP/RBI (see "Hitters" above) - the old model artifact was
deleted, not left in place, since its feature schema no longer matched
`dfs_ml.HITTER_FEATURE_COLUMNS`:

| signal | model | MAE | naive baseline MAE | heuristic MAE | correlation | n scored |
|---|---|---|---|---|---|---|
| `DK_Points_Hitter` | gradient boosting | 4.6716 | 4.7210 | 4.7569 | 0.162 | 5,642 |
| `Expected_H_Allowed` | Ridge (alpha=30) | 1.7385 | 1.8164 | 2.6241 | 0.302 | 468 |
| `Expected_BB` | Ridge (alpha=0.1) | 1.0233 | 1.0389 | 1.0744 | 0.120 | 468 |

`DK_Points_Hitter`'s MAE is on the wider post-BB/HBP/RBI point scale, so
not directly comparable in absolute terms to the pre-widening figure kept
in git history (3.6374) - but its correlation actually IMPROVED (0.162 vs.
0.145), a genuine gain, not just a like-for-like retrain.
`Expected_H_Allowed`/`Expected_BB`'s own inputs are unaffected by the
hitter-scoring widening - they were retrained only because more persisted
history had accumulated by this date; their small movement versus the
original run reflects that, not the widening itself.

All three beat both the naive baseline and their own prior heuristic, so
all three are now **live** (`dfs_ml.apply_ml_overrides`, called from
`scripts/build_dfs_rankings.py` every day) - a missing/not-yet-trained
model artifact falls back to the original heuristic automatically, so
nothing breaks if the weekly training workflow hasn't run yet. None of
these are strong signals - correlations of 0.12-0.30 mean there's still a
lot of unexplained variance - but each is a real, validated improvement
over both guessing and the prior heuristic, reported honestly rather than
oversold. The hitter model's biggest structural change: instead of
`compute_matchup_adjustment`'s flagged-risky ratio, it's trained directly
on the ratio's raw ingredients (the batter's own `WAVE`/
`Game_Hit_Probability`/`Consistency`/`Approach`, the opposing starter's
and bullpen's `PAVE`, `Park_Factor`, `is_home`, plus `Expected_BB`/
`Expected_HBP`/`Expected_RBI`) via `dfs_ml.build_hitter_features` -
letting the model find its own combination instead of inheriting the
multiplicative assumption that turned out not to hold up.

Model training runs weekly (`.github/workflows/ml_training_update.yml`,
same cadence rationale as Age Curves - re-fitting from full history is
too slow for the daily pipeline), committing `data/models/*.joblib`
alongside the code. Re-run `scripts/train_dfs_ml_models.py` (and update
the numbers above) after any change to the DFS feature set or windowing
constants.

**Retrained again 2026-07-26** after the PAVE bug fix above
(`starter_PAVE`/`Bullpen_PAVE` are direct hitter-model features;
`Expected_H_Allowed` - itself PAVE-derived - is a direct pitcher-model
feature), against the full persisted history (118 hitter dates / 103
pitcher dates):

| signal | model | MAE | naive baseline MAE | heuristic MAE | correlation | n scored |
|---|---|---|---|---|---|---|
| `DK_Points_Hitter` | gradient boosting | 4.6299 | 4.6648 | 4.7156 | 0.161 | 5,760 |
| `Expected_H_Allowed` | Ridge (alpha=30) | 1.7685 | 1.8395 | 1.8032 | 0.297 | 481 |
| `Expected_BB` | Ridge (alpha=0.1) | 1.0141 | 1.0310 | 1.0689 | 0.130 | 481 |

`DK_Points_Hitter`'s numbers are essentially unchanged from the prior
retrain (correlation 0.161 vs. 0.162) - expected, since PAVE isn't the
reason this signal's correlation is weak (see "Hitters" above).
`Expected_H_Allowed` is the interesting one: the PAVE fix alone already
turned its HEURISTIC from worse-than-baseline (MAE 2.6252 pre-fix) into
better-than-baseline (1.8032) on the identical dates - and this retrained
model improves on that fixed heuristic further still (1.7685), a real
additional gain stacked on top of the formula fix, not just the model
recovering ground the bug had cost it. All three again beat both the
naive baseline and their own (now-corrected) heuristic, so all three
stayed live with these updated weights.

**Retrained again 2026-08-19** after `dfs_ml.HITTER_FEATURE_COLUMNS`
widened to add `Exit_Velo`/`Barrel_Rate`/`xBA`/`xwOBA` (see "Real
batted-ball-quality signal" above), forcing a retire-and-retrain of the
two stale artifacts (`dfs_hitter_model.joblib`,
`hitter_hit_probability_model.joblib`) whose schema no longer matched:

| signal | model | MAE | naive baseline MAE | heuristic MAE | correlation | n scored |
|---|---|---|---|---|---|---|
| `DK_Points_Hitter` | gradient boosting | 4.3898 | 4.2960 | 4.3548 | 0.164 | 5,463 |
| `Expected_H_Allowed` | Ridge (alpha=30) | 1.7879 | 1.7959 | 1.9050 | 0.202 | 490 |
| `Expected_BB` | Ridge (alpha=0.1) | 0.9869 | 0.9790 | 0.9841 | 0.194 | 490 |

Reported honestly rather than silently kept as-is: `DK_Points_Hitter`'s
retrained model does **not** beat the naive baseline this run (4.3898 vs.
4.2960) - a reversal from the prior retrain above, where it beat both
bars. `Expected_BB` also misses the baseline (0.9869 vs. 0.9790), same as
several earlier runs. Both fall back to their heuristic automatically
(`ml_models.save_model`'s existing "beats baseline AND heuristic or don't
save" gate - see `dfs.py`'s docstring), so the live DFS Hitters tab keeps
serving the heuristic for these two exactly as it did before either model
ever existed - no crash, no silently-worse predictions shipped. Whether
this miss traces to the four new columns specifically, to a materially
different holdout window (the season has moved forward since the last
retrain, so the dates being fit/scored aren't the same), or plain
retrain-to-retrain noise on a signal whose correlation has hovered
0.16-0.30 the whole time isn't separable from this run alone - it isn't
evidence the new columns are bad, just that they didn't rescue this
particular model this time. `Expected_H_Allowed` (whose feature set
doesn't include the new columns at all) beat both bars again and stayed
live, essentially unaffected as expected. Next weekly retrain will show
whether `DK_Points_Hitter`/`Expected_BB` recover baseline on fresh data.

**Retrained again 2026-08-19** (same day, second run) after
`HITTER_FEATURE_COLUMNS` widened again to add `Fastball_WAVE`/
`Breaking_WAVE`/`Offspeed_WAVE`/`starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` (see "Real
pitch-type-specific platoon matchup" above): `DK_Points_Hitter` MAE
4.3982 (vs. 4.3898 just above, still short of the 4.2960 naive baseline -
still not saved), `Expected_H_Allowed` and `Expected_BB` both essentially
unchanged (1.7879 and 0.9869, identical to just above - expected, since
neither's feature set includes any of the six new columns). No
meaningful movement either direction from the pitch-family columns on
this signal.

**Retrained again 2026-08-19** (same day, third run) after
`HITTER_FEATURE_COLUMNS` widened once more to add `Whiff_Rate`/
`Chase_Rate` (see "Real plate-discipline signal" above): `DK_Points_Hitter`
MAE 4.3806 (vs. 4.3982 just above, still short of the 4.2960 naive
baseline - still not saved, heuristic keeps serving), correlation 0.165
(n=5,463). `Expected_H_Allowed`/`Expected_BB` don't include `Whiff_Rate`/
`Chase_Rate` in their own feature sets and are unaffected by this run.
Consistent with the pattern in the immediately-prior two retrains: these
two hitter-side batted-ball/plate-discipline signals move the hit-
probability classifier (see above) but haven't yet rescued the separate
`DK_Points_Hitter` points-regression model off its naive baseline.

## Optimal Lineup (`docs/dfs.html`'s "Optimal Lineup" tab, `dfs_optimizer.py`)

**`Estimated_Salary` is NOT a real DraftKings price.** DraftKings has no
public API for contest salaries - there is no free, ToS-compliant way to
fetch real ones. Scraping DraftKings' site was ruled out (fragile, likely
against their Terms of Service) and so was a manual daily CSV upload;
instead, on the user's own explicit direction ("build your best guess at
pricing based on performance"), `Estimated_Salary` is a MODELED number
derived entirely from this project's own `DK_Points_Hitter`/
`DK_Points_Pitcher` projections. Never treat it as a real, submittable
DraftKings price, and never assume a lineup built here is a valid budget
on the real platform - a real DFS player could lose real money making
either assumption. This is disclaimed everywhere the number surfaces: the
column is always named `Estimated_Salary` (never bare `Salary`), a
red-bordered warning box sits directly above the Optimal Lineup tab's
table, the rendered column header itself reads "Est. Salary (NOT a real
DK price)" (not relying solely on the box above it), and every module
docstring involved (`estimated_salary.py`, `dfs_optimizer.py`,
`roster_positions.py`) restates it.

**DraftKings Classic MLB roster rules**, confirmed live via web search
(not memory) against
[draftkings.com/help/rules/mlb](https://www.draftkings.com/help/rules/mlb):
10 roster spots - 2 P, C, 1B, 2B, 3B, SS, 3 OF (no FLEX/UTIL slot) - and a
$50,000 salary cap (`config.DFS_ROSTER_SLOTS`/`DFS_SALARY_CAP`). Real DK
salaries always use a $2,000 floor and $100 increments (also confirmed via
that source, `config.DFS_ESTIMATED_SALARY_FLOOR`/`_ROUND_TO`) - the salary
*ceiling* (`DFS_ESTIMATED_SALARY_CEILING`) is NOT from an official DK
table (DraftKings doesn't publish one; prices float algorithmically) -
it's an anecdotal 2026 example from secondary sources, rounded outward.
This floor/increment-are-real-but-ceiling-is-anecdotal distinction is
itself part of the honesty requirement here.

**The formula** (`estimated_salary.py`): linear min-max scaling of a
player's own `DK_Points_Hitter`/`DK_Points_Pitcher` into
[`DFS_ESTIMATED_SALARY_FLOOR`, `DFS_ESTIMATED_SALARY_CEILING`], clipped at
both ends, rounded to the nearest $100.

**Salary $/point parity fix (2026-07-26)**: the original v1 scaled
hitters and pitchers into SEPARATE reference point ranges and SEPARATE
salary ceilings, each independently min-max-scaled to its own group. This
was a real bug, caught by inspecting a real optimizer output: two elite
pitchers alone consumed ~$22K of the $50K cap, leaving the other 8 hitter
slots filled with replacement-level bats. The root cause wasn't that
pitchers are really worth that much more - it's that pitcher DK scoring
naturally spans a much wider raw point range (~2.6-22.75, box-score
categories piling up over 6 innings) than hitters' hit-type-driven
scoring (~2.6-4.7 pre-widening), so independently rescaling each group to
its own ceiling made a pitcher's marginal DK point worth roughly 5x a
hitter's in salary terms - a scaling artifact, not real relative DFS
value. Fixed by collapsing both groups onto ONE shared reference point
range (`DFS_REFERENCE_MIN_POINTS`/`DFS_REFERENCE_MAX_POINTS`) and ONE
shared ceiling (`DFS_ESTIMATED_SALARY_CEILING`) - both
`compute_hitter_estimated_salary`/`compute_pitcher_estimated_salary` now
delegate to the same constants, guaranteeing the same dollars-per-point
rate regardless of position (`test_hitter_and_pitcher_salary_share_same_dollar_per_point_rate`
is a direct regression test for this). A pitcher still costs far more
than a hitter in practice - just because pitchers genuinely PROJECT far
more points, the real reason, not a second hidden rescaling on top of it.

**Acceptance check, against real production `DK_Points` from 2026-07-25**
(`docs/data/dfs_hitters.csv`/`dfs_pitchers.csv`): under the OLD
per-position scale, a hitter projecting exactly 8.0 points priced at
**$6,500** (pinned to the hitter ceiling - 8 points was near the top of
hitters' whole observed range) while a pitcher projecting the SAME 8.0
points priced at only **$4,400** (still cheap relative to pitchers' much
wider range) - the same real point total was worth 48% more salary
depending on which position scored it, purely a scaling artifact. Under
the NEW shared scale, both price identically at **$4,800**. The real
top-2-pitchers-eat-the-budget case also improved: the two highest-scoring
real pitchers that day (22.75 and 22.39 points) combined for $21,800
under the old scale vs. $20,100 under the new one.

The shared reference range is computed from a REAL 20-game-date backtest
sample (`dfs_backtest.backtest_dfs_projections`, the same no-lookahead
recompute every other backtest in this project uses), pooling both
position groups' `DK_Points` together - not guessed, and not a single
day's snapshot like the pre-fix v1 constants were: hitters'
`DK_Points_Hitter` ranged 0.1595-16.6757 (mean 4.827, n=5,430); pitchers'
`DK_Points_Pitcher` ranged 0.1207-25.1412 (mean 11.459, n=468). The
pitcher range fully spans the hitter range on both ends, so the shared
min/max are effectively the pitcher extremes - expected, not a bug, since
pitcher DK scoring genuinely spans a wider range even after the hitter
scoring widened. Under this shared scale, a real average hitter (~4.8
points) now prices around $3,700 and a real average pitcher (~11.5
points) around $6,100 - a plausible, non-degenerate spread. This is still
a single-signal (points only) linear model - real DK pricing also
reflects season-long track record, popularity/ownership effects, and
other signals this project doesn't compute - a deliberate v1
simplification, not hidden.

**Position eligibility** (`roster_positions.py`): a genuinely new data
source for this project - no fielding-position data existed anywhere
before this. Fetched via the MLB Stats API's `people` endpoint
(`statsapi.get("people", {"personIds": ...})`), queried directly by
`key_mlbam` (this project's own player id - no name-matching needed).
Restricted to a player's single PRIMARY position, NOT DraftKings' real
multi-position eligibility (which DK derives from recent multi-position
starts) - a player whose primary position has no DK Classic slot at all
(most commonly "DH", a common primary role for several everyday hitters)
is excluded outright, a real v1 gap, not hidden. The field paths used here
are based on the MLB-StatsAPI package's own implementation and public
documentation, not a live-confirmed response (this sandbox cannot reach
statsapi.mlb.com at all - the same restriction `schedule.py`'s probable-
pitcher fetch already documents) - `scripts/debug_statsapi_positions.py`
(run through the `Debug statsapi` GitHub Actions workflow, real network
access) is the same bootstrapping step `schedule.py`'s own field paths
went through before being trusted in production.

**Injured/inactive hitters (2026-07-29)**: the optimizer's input,
`docs/data/dfs_hitters.csv`, now excludes any hitter who hasn't played
recently (see "Recency exclusion" under DFS Player Rankings' Hitters
section above) - previously nothing here checked *current* activity, only
season-long PA/rates, so a hitter on a long injured-list stay kept getting
optimally selected (a real, live bug: Dansby Swanson). `dfs_hitters.csv`
also now carries its own `dk_slot` column (for the dashboard's Hitters
position subtabs) - `scripts/build_optimal_lineup.py` drops it on read,
before `build_player_pool`'s own independent eligibility fetch/merge runs,
so the two don't collide into `dk_slot_x`/`dk_slot_y`.

**The optimizer** (`dfs_optimizer.py`): an exact MILP (mixed-integer
linear program), solved via [PuLP](https://github.com/coin-or/pulp)'s
bundled CBC solver - a new dependency, added because this is a real
knapsack/assignment problem (choosing 3 of ~80-100 OF-eligible candidates
alone is already a ~117,000-combination choice, before the shared salary-
cap budget across every other slot is considered), where a greedy
best-points-per-dollar pick per slot does NOT guarantee the true optimum.
The slot groups happen to be disjoint in this v1 design (one primary
position per player), which would let a hand-rolled DP solve this without
a new dependency - PuLP was chosen anyway since it directly expresses the
real constraints without careful group-merge bookkeeping, and because it
survives the near-certain follow-up of real DK multi-position eligibility
(which breaks the disjoint-groups assumption a DP would depend on, while
ILP trivially generalizes). A two-way player (appearing in both the
hitter and pitcher pools) is constrained to be selected at most once
across both roles, so the optimizer can never fill two roster slots with
the same real person. Returns no lineup (not a crash) if the slate can't
fill every slot or the cheapest possible full roster exceeds the cap -
`scripts/build_optimal_lineup.py` leaves the previous day's
`optimal_lineup.csv` in place in that case, same resilience pattern as a
failed schedule fetch elsewhere in this project.

`scripts/build_optimal_lineup.py` runs daily (`daily_update.yml`,
immediately after `scripts/build_dfs_rankings.py`), writing
`docs/data/optimal_lineup.csv` (the 10 selected players) and
`docs/data/dfs_salary_pool.csv` (every eligible player considered, for
transparency into what the optimizer actually saw).

### Slate filtering (client-side, `docs/dfs_solver.js`, 2026-07-31)

Real DraftKings contests split today's full slate into sub-contests by
game start time (early/main/night) - the daily batch lineup above always
optimizes across every game today, so it's never actually the optimal
lineup for any single real contest. The Optimal Lineup tab now lets a
user pick which games are in their contest (checkboxes, sorted by start
time) and click Analyze to re-solve the lineup for just those games -
entirely in the browser, using `docs/data/dfs_salary_pool.csv` (already
published daily), no new backend or network call.

**Why a hand-rolled JS solver, not a WASM/JS MILP library**: PapaParse is
the only third-party JS dependency anywhere on this site, matching its
no-build-step philosophy. `dfs_optimizer.py`'s own module docstring
already argues the problem is exactly solvable without PuLP: since every
pool row has exactly one `dk_slot`, the DraftKings roster slots are
disjoint groups, so this is a bounded multiple-choice knapsack per slot
merged over one shared salary budget - and since `Estimated_Salary` is
always a multiple of $100 (`config.DFS_ESTIMATED_SALARY_ROUND_TO`), the
budget dimension has only 501 discrete states at the $50,000 cap. A
per-slot dynamic-programming table (exact, not a heuristic - see
`docs/dfs_solver.js`'s own module docstring for the full algorithm)
merged via a knapsack-of-knapsacks solves a real slate in single-digit
milliseconds. Verified exact against the Python MILP directly: a randomized parity fuzz
test (`tests/test_dfs_solver_js.py`, 31 cases: 20 random pools, 10 with a
`min_salary` floor, 1 with a two-way-player collision) compares
`docs/dfs_solver.js`'s `solveOptimalLineupDP` against
`dfs_optimizer.solve_optimal_lineup` on the same random pools (including a
`min_salary` floor and a two-way-player collision) and asserts the two
exact solvers agree on both feasibility and the achievable objective
total - plus a direct production check: "select every game, objective
mean" reproduces the real `optimal_lineup.csv` exactly on live data.

The one thing the DP can't express natively is `dfs_optimizer.py`'s
cross-group "at most once per player" constraint for a true two-way
player (present in both the hitter and pitcher pools). Handled by solving
once per combination of "which single role stays available" when
duplicates exist (still exact, since each combination is itself an exact
solve and the real pool essentially never has more than one two-way
player) - more than 3 simultaneous duplicates (should not happen in
practice) falls back to keeping each duplicated player's higher-value row
and flags the result as approximate rather than an exponential blow-up.
**Known v1 limitation, matching `roster_positions.py`'s own documented
gap**: this DP depends on the disjoint-slot-groups property - if this
project ever implements real DraftKings multi-position eligibility (a
player legally fillable at more than one slot), the Python MILP already
generalizes to that case for free, while this DP would need to be
revisited.

Each game's real start time (`game_datetime`, threaded from
`schedule.normalize_schedule` through `dfs.py`'s
`HITTER_DFS_COLUMNS`/`PITCHER_DFS_COLUMNS` into `dfs_salary_pool.csv`) is
the raw statsapi `gameDate` field - confirmed live via the `Debug
statsapi` GitHub Actions workflow (run 30661418977, 2026-07-31): a bare,
unconditional sibling key of `gamePk`/`teams`/`status` on every raw game
dict, no extra hydrate needed, same as every other field path
`schedule.py` already trusts in production.

### Ceiling / volatility signal (`dfs_ceiling.py`)

GPP (tournament) DFS lineups are won by boom/spike-game players, not
players who reliably score near their own mean - a real, well-known DFS
strategy concept the user raised directly ("the winner won't have players
getting 5 points across their lineup, they're more likely to have lucked
into players averaging 15 or so"). Every signal in this project up to
this point was a MEAN projection; nothing measured upside.
`Ceiling_DK_Points` is the `config.DFS_CEILING_PERCENTILE`-th percentile
(90th) of a player's own REAL historical modeled DK points
(`dfs_backtest.compute_actual_hitter_dk_points`/
`compute_actual_pitcher_dk_points` applied per real game date they
played) - a genuine outcome-history signal, not a new projection model. A
player with fewer than `config.DFS_CEILING_MIN_GAMES` (10) real scored
games falls back to the group-wide (all hitters', or all pitchers')
percentile instead of a noisy small-sample per-player number
(`Ceiling_Source` marks which applies). Merged into `dfs_hitters.csv`/
`dfs_pitchers.csv` as an ADDITIONAL informational column, never replacing
`DK_Points_Hitter`/`DK_Points_Pitcher`.

**Backtested before trusting it** (`dfs_ceiling.backtest_ceiling_signal`,
same 20-date no-lookahead sample as the heuristic DFS backtest,
`Ceiling_DK_Points` computed from ONLY history strictly before each test
date): does ranking by ceiling actually surface real boom-game days
better than ranking by the existing mean projection? Measured as a
"capture rate" - of the player-days that ACTUALLY landed in that date's
real top decile, what fraction were ALSO top-decile by `Ceiling_DK_Points`
going in, vs. by the mean projection instead:

| player type | correlation | real top-decile days | ceiling capture rate | mean-projection capture rate |
|---|---|---|---|---|
| Hitters | 0.127 | 543 (n=5,430) | 19.3% | 10.1% |
| Pitchers | 0.313 | 47 (n=468) | 23.4% | 27.7% |

For hitters, ceiling-ranking is genuinely ~2x better than the mean
projection at flagging real boom days in advance (the mean projection is
barely above the ~10% base rate - `DK_Points_Hitter` has almost no power
to predict WHICH day a hitter booms). For pitchers, there's no clear
edge - the mean projection actually did slightly better on this (small,
n=47) sample. **Reported as a mixed result, not oversold**: this is a
validated upside signal for hitters specifically, not a blanket win.

Because of that mixed result, `Ceiling_DK_Points` ships
**informational-only** - the Optimal Lineup optimizer still defaults to
maximizing the mean projection. Pass `--objective ceiling` to
`scripts/build_optimal_lineup.py` (`dfs_optimizer.solve_optimal_lineup`'s
new `objective_column` parameter) to build a ceiling-maximizing lineup
instead - opt-in, not the default, until real backtest evidence supports
defaulting to it (which it does not yet, especially for pitchers). A
player with no `Ceiling_DK_Points` at all (no real scored history yet)
falls back to their own mean projection in the optimizer's pool, so
they're never silently unselectable under the ceiling objective.

**Explicit v1 scope cuts**: no real DK multi-position eligibility (single
primary position only - full-time DH players are excluded entirely); no
multi-lineup generation, no GPP-vs-cash-game strategy, no ownership-
projection/game-theory diversification (one single "best" lineup per
day); no live salary scraping or manual upload path (both explicitly
rejected in favor of the modeled-estimate approach); relief pitchers
remain out of scope (inherited from `dfs.py`'s own probable-starters-only
restriction). Salary-accuracy backtesting is also explicitly out of scope
- there's no real DraftKings price to validate `Estimated_Salary` against,
unlike every other backtested signal in this project.

### Boom-Adjusted DK Points (`dfs_ceiling.py`)

The user's own framing, verbatim: "player A scores 5 points literally
every night... player B sometimes scores 15, sometimes 5, sometimes 20,
and even sometimes 0, and he thus averages 4.8 - I'd rather have player B
than player A," followed by an explicit rejection of both extremes: "I
don't want straight boom or bust nor straight mean. Neither is
optimizing." Pure `Ceiling_DK_Points` (a single 90th-percentile game) is
exactly the "straight boom" case it rejects - it credits raw upside
regardless of how OFTEN a player actually reaches it. The plain mean is
the other rejected extreme.

`Boom_Adjusted_DK_Points = mean_projection + k * Upside_Deviation`, where
`Upside_Deviation` is a player's own real historical UPSIDE-ONLY
semi-deviation (`dfs_ceiling.compute_upside_deviation`): `sqrt(mean((x -
their_own_historical_mean).clip(lower=0) ** 2))` over ALL their real
games, not just the boom ones. Below-average games contribute exactly
0 - this is deliberately a semi-deviation, not plain standard deviation,
because plain stdev would reward bust-heavy inconsistency exactly as much
as boom-heavy inconsistency, the opposite of the point. Dividing by the
FULL game count (not just the boom subset) means a player who booms
rarely scores lower than one who booms often at similar magnitude -
real DFS value tracks boom FREQUENCY, not just size, which is exactly
what distinguished player B from a player who got lucky once.

**k was backtested, not guessed** (`dfs_ceiling.backtest_boom_adjusted_signal`,
same 20-date no-lookahead sample and capture-rate methodology as
`Ceiling_DK_Points`'s own backtest, run once per candidate k):

| player type | k=0.0 (mean) | k=1.0 (chosen, hitters) | k=2.0 (grid max) |
|---|---|---|---|
| Hitters correlation | 0.009 | 0.044 | 0.061 |
| Hitters capture rate | 10.1% | 12.5% | 13.3% |
| Pitchers correlation | 0.337 | 0.339 (k=0.75 peak) | 0.328 |
| Pitchers capture rate | 27.7% | 27.7% | 25.5% |

For hitters, correlation and capture rate rose monotonically across the
ENTIRE tested grid (0.0 through 2.0) without ever turning over - the grid
didn't localize a true peak. The grid-max k=2.0 was deliberately NOT
chosen anyway: a real check of magnitude (mean historical
`Upside_Deviation` 4.417 vs. mean `DK_Points_Hitter` 5.185) shows that at
k=2.0 the volatility term outweighs the mean term by nearly 2:1 - which
just recreates pure ceiling-chasing under a different name, the opposite
of what was asked for. **k=1.0** was chosen instead: it keeps the two
terms roughly balanced (mean contributes ~54% of a typical score,
deviation ~46%) while still capturing a real, validated improvement over
the plain mean (capture rate +24% relative). For pitchers, correlation
stayed flat and capture rate bounced non-monotonically across the whole
grid with no k reliably beating the k=0 baseline - the same conclusion
`Ceiling_DK_Points` reached for pitchers. **k=1.0 for hitters, k=0.0
(plain mean) for pitchers** (`config.DFS_BOOM_ADJUSTED_K_HITTER`/
`_PITCHER`).

Same shipping posture as `Ceiling_DK_Points`: an ADDITIONAL informational
column, not a default. Pass `--objective boom` to
`scripts/build_optimal_lineup.py` to build a lineup that maximizes
`Boom_Adjusted_DK_Points` instead of the plain mean - a genuine middle
ground between the "mean" and "ceiling" objectives, backed by the
balance/backtest reasoning above rather than picked arbitrarily.

### Matchup-aware boom score (`Matchup_Boom_Score`, hitters only)

Real evidence from an actual DraftKings contest surfaced the next gap:
the winning lineup spent $17,500 on its 2 pitchers and second place
$18,100, vs. this project's own suggested lineup spending over $21,000 -
and even after recalibrating the salary model with real backtest data
(see "Salary $/point parity fix" above), the optimizer's mean-projection
total for a real slate came out to only ~81 points, far below what
winning lineups actually need. The user's own accounting of what a
winning lineup needs: pitchers contribute roughly 50-60 points combined
(achievable from mean projections alone); the other ~100 points need to
come from 8 position players, and since only a handful of hitters
project anywhere near the ~12-point average that would require, **a few
specific hitters need to boom** - both the real 1st- and 2nd-place
lineups had 3 players combine for roughly 60 points between them. The
ask: predict WHICH hitters are most likely to boom today, based on their
matchup, not just who has boomed historically.

`Ceiling_DK_Points` and `Boom_Adjusted_DK_Points` are both matchup-BLIND
on their volatility side - only the mean term reads today's opponent.
`Matchup_Boom_Score` fixes that:

- `compute_boom_threshold`: a single FIXED, GROUP-WIDE point value (the
  `config.DFS_CEILING_PERCENTILE`-th percentile pooled across every real
  hitter-game, not each player's own percentile). A per-player threshold
  would be tautological here - by construction, roughly 10% of ANY
  player's own games clear their OWN 90th percentile, so every player's
  "boom rate" against their own bar would converge to about the same
  number regardless of how boom-prone they really are.
- `compute_boom_rate`: each player's real historical rate of clearing
  that SHARED bar - now players genuinely differ (a boom-prone hitter
  clears it far more than a steady one). Small-sample players (fewer
  than `config.DFS_CEILING_MIN_GAMES` real games) fall back to the
  group-wide clear rate.
- `compute_matchup_boom_score`: `Boom_Rate * Matchup_Ratio` - today's
  real opposing-pitcher/park-adjusted matchup signal
  (`dfs.compute_matchup_adjustment`'s output, already computed as part
  of `DK_Points_Hitter`) scales the player's own boom frequency up or
  down for today specifically. This is a RANKING signal, not a
  calibrated probability, despite "boom" in the name - never treat it as
  P(boom) in a literal sense.

**Backtested** (`dfs_ceiling.backtest_matchup_boom_signal`, same 20-date
no-lookahead sample, hitters only, n=5,430, 689 real boom-days against an
average no-lookahead threshold of ~14.0 points) - capture rate of those
689 real boom days, i.e. what fraction were flagged in advance by each
signal's own top decile:

| signal | capture rate |
|---|---|
| Mean projection (`DK_Points_Hitter`) alone | 10.6% |
| `Matchup_Boom_Score` (boom rate x today's matchup) | 14.1% |
| `Boom_Rate` alone (no matchup adjustment at all) | **17.3%** |

**Honest, somewhat surprising result: the matchup adjustment made it
WORSE, not better.** `Boom_Rate` alone - a player's plain historical
frequency of clearing the shared bar, with no read on today's opponent at
all - beat the matchup-multiplied version by a real margin (17.3% vs.
14.1%), and both comfortably beat the mean projection. Multiplying by
`Matchup_Ratio` added noise rather than signal here - consistent with
`Matchup_Ratio` already being flagged elsewhere in this project (see
`dfs.py`'s module docstring) as the single highest-risk modeling choice
in the whole DFS feature set, and already SUPERSEDED by an ML model for
the main hitter projection specifically because the heuristic ratio
didn't hold up. The same weak signal dragging down `DK_Points_Hitter`
drags down `Matchup_Boom_Score` too when multiplied in.

Because `Boom_Rate` is the real, validated win here, it's exposed as its
own column (not just an internal ingredient of `Matchup_Boom_Score`) -
use `Boom_Rate` to identify genuinely boom-prone hitters; treat
`Matchup_Boom_Score` as informational/exploratory only, not as an
improvement over it. This is reported plainly rather than reframed as a
win, matching how this project has always handled a backtest that didn't
confirm the hypothesis (see Age Curves HR9's KNN-vs-ML precedent,
Ceiling/Boom-Adjusted's pitcher non-results above).

**Re-checked after the PAVE bug fix above** (`Matchup_Ratio` reads
`Matchup_Hit_Probability`, a direct PAVE consumer; `Boom_Rate` itself is
historical-outcome-only and unaffected): n=5,547/675 real boom-days (a
slightly larger accumulated sample than the original 5,430/689, not a
strict same-day A/B) - mean projection 10.8%, `Matchup_Boom_Score` 14.5%,
`Boom_Rate` alone 17.6%. The ordering conclusion is UNCHANGED - `Boom_Rate`
alone still clearly beats `Matchup_Boom_Score` - with a small, consistent
uptick across all three from the corrected matchup input. The fix didn't
overturn this finding; `Matchup_Ratio` remains a weak signal for
reasons unrelated to the PAVE bug (see above).

**Hitters only** - pitchers have no `Matchup_Ratio` analog in this
project, and the earlier Ceiling/Boom-Adjusted backtests already found no
real upside-signal edge for pitchers, so this wasn't extended to a
foundation that had already shown no signal. Not wired into the
optimizer's `--objective` flag either (a full-lineup objective needs a
value for every roster slot including pitchers, which this doesn't
have) - it's a pure informational column for identifying which hitters to
prioritize, matching how it was actually asked for.

### Opponent offense adjustment (`Opponent_Offense_Ratio`, pitchers only, `pitcher_matchup.py`, 2026-07-31)

A second, distinct matchup gap, this one user-observed rather than
backtest-discovered: "matchup doesn't seem to be taken into account as
much as I'd like. It's choosing the same players for the most part
despite who they're facing. Does [a pitcher] have great boom potential
for the price? Yes. Will he boom off of [a tough offense]? Almost
certainly no." Checking the actual code confirmed the gap was real and,
unlike `Matchup_Boom_Score` above, genuinely untested: `dfs.
compute_pitcher_dk_points` had **no opponent-quality signal anywhere**,
not even in the base mean projection - `K9`/`BB9`/`HR9`/`IP_per_start`
are all windowed averages of the pitcher's OWN recent form. A pitcher's
projection was identical whether facing a last-place offense or a
first-place one.

`teams.compute_offensive_edge` already computes real, rolling
bases-scored-per-game per team (`team_bases_pg`), but its own
`offensive_edge`/`true_power` outputs net that out against whichever
opponent a team's OWN most recent game happened to be against - not
today's actual matchup - so both are contaminated for this use.
`team_bases_pg` itself has no opponent term at all and is now exposed
directly from `compute_offensive_edge` as the correct building block.

`pitcher_matchup.py`:

- `compute_opponent_offense_ratio`: today's opponent's `team_bases_pg`
  divided by the league average, blended toward a neutral 1.0 by a
  `weight` and clipped to `config.PITCHER_MATCHUP_OFFENSE_CLIP` (so one
  extreme-outlier offense can't blow up a projection). `weight=0.0`
  returns exactly 1.0 for every row - the built-in null hypothesis that
  reproduces today's unadjusted heuristic exactly, not an approximation
  of it.
- `attach_opponent_offense` / `compute_opponent_adjusted_pitcher_points`:
  scales `Expected_H_Allowed`/`Expected_ER` by the ratio and recomputes
  `DK_Points_Pitcher` from the scaled components. `Expected_K`/
  `Expected_BB`/`Expected_IP` are left unadjusted for v1 - a tougher
  offense plausibly affects those too, but there's no existing
  per-opponent signal for either yet, and scaling the two categories a
  bases-scored rate most directly predicts is the smallest, most
  defensible first step.

**Backtested** (`pitcher_matchup.backtest_pitcher_matchup_signal`, real
persisted Statcast, `--days 90` - the pitcher sample is much smaller than
the hitter-side backtests above, dozens of probable starters per date vs.
thousands of plate appearances, so this needed a longer window than the
20-date default used elsewhere in this project). n=2,081 real
pitcher-days, correlation and MAE of the adjusted `DK_Points_Pitcher`
against that date's REAL `Actual_DK_Points_Modeled`, per weight in
`config.PITCHER_MATCHUP_WEIGHT_GRID`:

| weight | correlation | MAE |
|---|---|---|
| 0.00 (baseline, no adjustment) | 0.3397 | 6.6736 |
| 0.25 | 0.3407 | 6.6658 |
| 0.50 | 0.3414 | 6.6612 |
| 0.75 | 0.3416 | 6.6605 |
| 1.00 (full, unblended ratio) | **0.3417** | **6.6600** |

**Honest result: the right direction, too small a magnitude.** Unlike
`Matchup_Boom_Score` above (which went backwards), this improves
correlation and lowers MAE monotonically across the entire grid - genuine
evidence the adjustment points the right way. But the size of the win is
tiny: weight=1.0 vs. weight=0.0 is only a 0.6% relative correlation
improvement and a 0.2% relative MAE improvement, nowhere near the real
margins that justified this project's other nonzero defaults (e.g.
`Boom_Adjusted_DK_Points`' `k=1.0` needed a 24% relative capture-rate
improvement over `k=0.0` to be chosen). At n=2,081 this small a gap isn't
distinguishable from noise with real confidence.

`config.PITCHER_MATCHUP_OFFENSE_WEIGHT` stays **0.0** for exactly that
reason - `DK_Points_Pitcher` is unchanged from before this module.
`Opponent_Offense_Ratio` still ships as an **informational-only column**
on the pitcher table (`scripts/build_dfs_rankings.py` calls
`attach_opponent_offense` with the weight-0.0 default, which is a no-op
ratio of exactly 1.0 today, but keeps the mechanism ready to flip on
without a code change if a future, larger-sample backtest clears the
bar) rather than being silently dropped - the DIRECTION is real evidence
worth showing, even though the MAGNITUDE doesn't clear this project's
bar for changing the live projection.

### Value_Score: "stars, not superstars" (`dfs_optimizer.py --objective value`)

None of the fixes above (salary parity, `Boom_Adjusted_DK_Points`,
`Matchup_Boom_Score`) actually solved the real complaint. Real evidence,
reported directly: on an actual slate, `mean`/`ceiling`/`boom` all
independently converged on the SAME two most expensive pitchers, spending
$19,900-$21,200 of the $50,000 cap on pitching and leaving barely enough
room for 8 undifferentiated floor-priced hitters. The user's diagnosis,
verified against the numbers rather than taken on faith: `Estimated_Salary`'s
fixed $2,000 floor (see "Salary $/point parity fix" above) is a much
smaller fraction of an elite player's price than a replacement-level
player's, so ANY high scorer - regardless of position - gets a
structurally better AVERAGE dollars-per-point rate purely from that floor
dilution, even though the parity fix already equalized the MARGINAL rate.
An objective that maximizes raw point totals under a budget will always
rationally chase that average-rate advantage and overpay for the 1-2
biggest scorers, leaving nothing to build a real roster - a mix of
reliable "consistent" floor plays and genuine "boom" upside plays
("consistent players carry their own, boom players pick up slack," in the
user's own framing) - with what's left. This is a roster-CONSTRUCTION
problem, not a single-day boom-prediction problem, so it's a different
question from the capture-rate backtests above - **implemented per
explicit user direction ("implement no matter the test"), not validated
against a capture-rate backtest the way `Ceiling_DK_Points`/`Boom_Rate`
were.**

`Value_Score = boom-adjusted points / (Estimated_Salary / 1000)` -
directly rewards being UNDERPRICED relative to real upside (a "star")
over being already fully priced-in (a "superstar"), computed for BOTH
hitters (reusing their own validated `Boom_Adjusted_DK_Points`, k=1.0) and
pitchers (a separate boom-adjusted computation using
`config.DFS_VALUE_BOOM_K_PITCHER = 1.0` - deliberately NOT the same as
`DFS_BOOM_ADJUSTED_K_PITCHER`'s validated 0.0, which answers the
different, narrower question of predicting which day a pitcher booms).

**A second real flaw, found via the same honest sanity check that shipped
this feature**: maximizing a per-dollar RATIO carries no pressure to
spend near the cap. Tested against real production data
(`docs/data/dfs_salary_pool.csv` recomputed with real `Upside_Deviation`):
`objective=value` with no floor picked a full legal lineup for only
$39,700 of the $50,000 cap - total `DK_Points` fell from 81.38 (mean
objective, same slate) to 56.31, because leaving $10,300 unspent doesn't
cost the ratio objective anything. Fixed with a `min_salary` floor
constraint on the same MILP (`solve_optimal_lineup`'s new `min_salary`
parameter: `sum(Estimated_Salary) >= min_salary`), set to
`config.DFS_VALUE_MIN_SALARY_FRACTION` (0.85) of the cap - chosen by
testing three fractions against the same real pool:

| min_salary fraction | total salary spent | total DK_Points | pitcher salary |
|---|---|---|---|
| 0.85 ($42,500) | $43,900 | 67.99 | $16,900 |
| 0.90 ($45,000) | $46,000 | 73.66 | $19,000 |
| 0.95 ($47,500) | $47,500 | 77.79 | $19,900 |

0.95 defeats the whole point - pitcher spend creeps back to the same
$19,900 this feature exists to avoid. 0.85 keeps pitcher spend near the
real winning-lineup range ($17,500-$18,100 in the contest that originally
motivated this work) while still using most of the budget - the best
balance of the three tested, not a formally optimized value.
`scripts/build_optimal_lineup.py` passes this floor only for
`--objective value`; `mean`/`ceiling`/`boom` are unchanged (no floor).

**`value` is now the daily production default** (`daily_update.yml` runs
`python scripts/build_optimal_lineup.py --objective value`), unlike
`ceiling`/`boom` which stay opt-in-only behind the flag. This is a
deliberate exception to this project's usual "don't default to an
unvalidated signal" rule: `mean`'s real-world failure mode (chronic
pitcher overspend) is exactly what motivated this whole feature, so
leaving `value` opt-in would mean the live "Optimal Lineup" tab kept
showing the broken behavior even after building the fix.
`dfs_optimizer.solve_optimal_lineup`'s own function default stays
`"DK_Points"` (mean) for library/test backward compatibility - only the
daily workflow's invocation changed.

## NFL DFS (`docs/nfl.html`, in progress)

An NFL analog of the MLB DFS pipeline above, mirroring its two core
ideas - rolling windows (not season averages) for player form, and
matchup analysis (opponent defense quality) - adapted for DraftKings NFL
Classic contests. Built incrementally; this section fills in with real
numbers/behavior as each phase lands.

Data source is `nflreadpy` (the actively-maintained nflverse successor to
the now-deprecated `nfl_data_py` - see its GitHub README for the
deprecation notice). Real field names/shapes referenced anywhere in this
project's NFL modules are confirmed live via
`scripts/debug_nfl_data.py`/`.github/workflows/debug_nfl_data.yml`, not
assumed - see `nfl_data.py`'s module docstring for the specific run cited.

Raw per-season tables (`weekly`, `schedules`, `injuries`,
`rosters_weekly`, `team_stats`, `snap_counts`) persist to
`data/raw/nfl/*.parquet` via `nfl_data.py`, one file per table per season
- the current season's file is overwritten wholesale on each fetch
(nflverse retroactively corrects stats, so an append-and-dedupe pattern
isn't safe), while completed historical seasons are fetched once via
`scripts/fetch_nfl_historical.py` (`config.NFL_HISTORICAL_SEASONS`) and
left alone. `snap_counts` is keyed by a real Pro-Football-Reference id
(`pfr_player_id`), a different id space than every other table's GSIS
`player_id` - crossed over via `rosters_weekly`'s own real `gsis_id`/
`pfr_id` columns wherever it's used (see `nfl_bestball.compute_player_snap_share`).

Player form is a **games-back rolling blend** (`nfl_passing.py` for QBs,
`nfl_rush_rec.py` for RB/WR/TE together, since both share DK's FLEX pool)
across `config.NFL_QB_WINDOWS`/`NFL_SKILL_WINDOWS` - games-back, not
day-count, unlike the MLB pipeline's `WAVE_WINDOWS` (see
`config.NFL_QB_WINDOWS`'s docstring for why NFL's bye weeks make a
calendar-count window the wrong shape). **Window weights are an
unvalidated first-pass placeholder** - no NFL backtest exists yet to
calibrate them against (see the Backtesting phase below, not yet built).

Matchup adjustment (`nfl_teams.py`/`nfl_matchup.py`) is a direct
structural port of `pitcher_matchup.py`'s opponent-offense ratio, applied
in the opposite direction: a QB/RB/WR/TE's projection is scaled by the
opponent DEFENSE's real recent pass/rush-yards-allowed rate
(`nfl_teams.compute_defense_rolling_rates`, derived from `weekly`'s own
`opponent_team` column - no separate schedule join needed for "what was
allowed"), looked up via that player's **upcoming** opponent from the
current week's schedule (`nfl_matchup.attach_matchup_adjustment`) -
never a defense's own last-played opponent, the same leakage bug class
`teams.compute_offensive_edge` had to be split apart to avoid for MLB.
Like `PITCHER_MATCHUP_OFFENSE_WEIGHT`, `config.NFL_MATCHUP_WEIGHT`
**defaults to 0.0** (informational-only) until a real NFL backtest earns
a nonzero live weight.

DK scoring (`nfl_dfs.py` for QB/RB/WR/TE, `nfl_dst.py` for DST) uses
DraftKings' real NFL Classic scoring rules, confirmed live via web
search against DraftKings' own published rules (not from memory):
0.04 pt/passing yard, 4 pts/passing TD, -1 pt/interception thrown, 0.1
pt/rushing or receiving yard, 6 pts/rushing or receiving TD, 1 pt/
reception (full PPR), -1 pt/fumble lost, 2 pts/2-point conversion, +3
bonus for 100+ rushing OR receiving yards in a game (scored separately -
a player can clear both), +3 bonus for 300+ passing yards in a game.
DST: 1 pt/sack, 2 pts/interception, 2 pts/fumble recovery, 2 pts/safety,
2 pts/blocked kick, 6 pts/defensive or return TD, plus a real non-linear
points-allowed bucket table (0 pts allowed = +10, down to 35+ pts
allowed = -4). DK NFL Classic uses a 9-slot roster (QB/RB/RB/WR/WR/WR/
TE/FLEX/DST) and a $50,000 salary cap - same real cap MLB Classic uses,
confirmed independently, not assumed just because MLB happens to match.

The 100+/300+ yardage bonuses are step functions, not linear in a
windowed mean - v1 scores them as an expected value (each player's own
historical rate of clearing the threshold, blended across the same
games-back windows as the rest of their projection), the same pattern
`hitters.compute_wave` already uses to turn a hit rate into a
probability. Flagged unvalidated pending a real NFL backtest (Phase 7).

`nfl_estimated_salary.py` is a direct structural port of
`estimated_salary.py` - same shared reference-point-range/floor/ceiling/
$100-round-to approach, same "never a real DraftKings price, always
`Estimated_Salary`" disclaimer. Reference range is computed from real
2025-season DK_Points_QB/DK_Points_Skill/DK_Points_DST output, not
guessed (see `config.py`'s NFL Estimated Salary section for the exact
numbers).

`nfl_roster_positions.py` maps `position` directly to a DK Classic slot
(QB/RB/WR/TE) - simpler than MLB's `roster_positions.py`, which needs its
own MLB Stats API fetch since Statcast carries no fielding-position data
at all; nflreadpy's tables already carry `position` directly. Every
RB/WR/TE gets TWO eligibility rows (their own slot AND `"FLEX"`) so
Phase 6's optimizer can handle FLEX with zero new constraint types.

The FLEX-slot optimizer needs almost no new logic. `nfl_dfs_optimizer.py`
reuses `dfs_optimizer.solve_optimal_lineup` UNMODIFIED - its existing
"cap any duplicated `key_mlbam` group at <= 1 selected row" MILP
constraint (originally built for the rare MLB two-way-player edge case)
now handles FLEX for real, since every RB/WR/TE gets two pool rows (own
slot + `"FLEX"`, same identity) from `nfl_roster_positions.py`. The
client-side `docs/nfl_dfs_solver.js` (a JS dynamic-programming
re-implementation, mirroring `docs/dfs_solver.js`) can't express that
same cross-group constraint directly, so it instead solves the ordinary
NO-FLEX problem exactly 3 times - once per hypothesis of which position
absorbs the extra slot (`{RB:3,WR:3,TE:1}` / `{RB:2,WR:4,TE:1}` /
`{RB:2,WR:3,TE:2}`) - and keeps the best, an exact (not approximate)
reduction since DK scores a FLEX RB identically to an RB-slot RB. Both
solvers verified against the same hand-constructed pool with a known-
by-construction optimum (`tests/test_nfl_dfs_optimizer.py` and
`docs/nfl_dfs_solver.test.js`), wired into CI alongside the MLB solver's
own Node tests.

**Real backtest results** (`nfl_dfs_backtest.py`, `scripts/backtest_nfl_dfs_rankings.py`,
no-lookahead, against the full real backfilled 2016-2025 history in
`data/raw/nfl/`): reported honestly either way, same standard every other
signal in this project is held to.

- **QB**: MAE 6.66 vs. a naive "always predict the sample mean" baseline's
  7.82 (14.8% better), correlation 0.514, n=6,358.
- **Skill (RB/WR/TE)**: MAE 4.68 vs. naive baseline's 6.18 (24.2% better),
  correlation 0.585, n=50,988.
- **DST**: MAE 4.72 vs. naive baseline's 4.63 - **worse than naive**,
  correlation 0.101 (essentially no signal), n=5,490.

QB/Skill also beat a simpler "flat, unweighted season-average" heuristic
(real but modest margins) - real evidence the RECENCY-WEIGHTING mechanism
itself adds value, not just "having a player-form signal at all." This
validates Phase 2's windowing MECHANISM; it does NOT validate the
specific window weight VALUES (still an unrecalibrated placeholder - see
`config.NFL_QB_WINDOWS`'s docstring).

**DST is an honest negative result** - `nfl_dst.py`'s points-allowed-
bucket-via-windowed-mean approximation doesn't beat guessing. `DST_Points`
still ships (the optimizer/roster need it structurally), but should be
treated as unvalidated/weak, not a trustworthy signal - see
`config.py`'s NFL Backtesting section for the full numbers and reasoning.

Not yet built: the weekly-cadence pipeline/workflows for the in-season
DFS Optimal Lineup subtab (needs live-season data this project doesn't
have yet).

### Bestball Draft Strategy (`docs/nfl.html`'s Preseason subtab, `nfl_bestball.py`)

**Its own visual theme.** `docs/style.css` is shared across every page, but
`docs/nfl.html` overrides the brand accent tokens (`--accent`,
`--accent-strong`, `--on-accent`) in its own inline `<style>` block -
football green instead of the baseball-orange every other page uses - and
swaps the header/favicon brand mark from the baseball-seam glyph to a
football-and-laces glyph. Scoped to this one file only (the shared
stylesheet and every other page's markup are untouched), so it's a purely
additive, page-local override rather than a new theming system.

Real preseason bestball drafts don't need a forward-looking weekly
projection - they need "how much value/risk did this player represent
LAST season, as a whole." `nfl_bestball.py` answers that with a real,
realized season total rather than a rolling-window projection: it reuses
`nfl_dfs_backtest.py`'s `compute_actual_qb_dk_points`/
`compute_actual_skill_dk_points` (already-validated real full-PPR DK
scoring from real box scores, see the backtest numbers above) summed
across every real regular-season week, for QB/RB/WR/TE (DST excluded -
bestball drafting doesn't need DST optimization the way weekly DFS does).

**Games played is the injury-history signal** - deliberately simple and
honest, not a full medical history: real games played vs. that player's
team's real regular-season game count (16 games through 2020, 17 from
2021 on - always read from the real schedule, never hardcoded), plus the
same figure for the prior season for a cheap repeat-injury-risk read.
`dk_points_per_game` and `games_missed` are kept as separate columns
rather than blended into one "draft score" - there's no backtestable
ground truth for what the right health/talent tradeoff weighting would
even be, so this project's "show raw signals honestly, don't blend
without a real backtest to justify it" standard applies here too.

**Real season-total offensive snap share** (`season_snap_share`,
`nfl_bestball.compute_player_snap_share`, from `nfl_data.fetch_snap_counts`
- real Pro-Football-Reference-sourced per-game snap counts via
`nflreadpy.load_snap_counts`) is also published on the rankings table -
a real playing-time-ROLE signal, distinct from games played (availability)
and points-per-game (production rate). This is a real SEASON-TOTAL share
(a player's real total offensive snaps that season, divided by their
team's real total offensive snaps that season - both summed across every
real game, not just the games the player themselves appeared in), not a
per-game average - a per-game average would let a real one-or-two-game
emergency spot start at a high per-game rate look identical to a real
every-week starter, exactly the small-sample problem a playing-time
signal needs to avoid. A team's real total offensive snaps for one game
is read as the max real `offense_snaps` among that team's players that
game (confirmed live: some player, almost always an O-line starter,
plays exactly 100% of a team's real offensive snaps in 541/544, ~99.4%,
of real 2025 team-games). It's crosswalked from the snap data's own
`pfr_player_id` join key to this project's `player_id`/GSIS id via
`fetch_rosters_weekly`'s real `gsis_id`/`pfr_id` columns (confirmed live
to cover 382 of 383, ~99.7%, of a real 2025 qualified population); a
player missing that crosswalk gets a real `NaN`, never a fabricated 0%.

**Real round-split scoring** (`r1_dk_points`/`r2_r4_dk_points`,
`nfl_bestball.build_bestball_rankings`) breaks the real season total into
the two real stages DraftKings Best Ball Mania tournaments actually run
on - confirmed live via web search against Establish The Run and 4for4:
a real "Round 1" spanning weeks 1-14 (cumulative real points across
those weeks decide who advances out of each real 12-team draft pod - the
top 2), followed by three real single-week knockout rounds at weeks 15,
16, and 17 ("Rounds 2-4", reported here as one combined real sum, not
split further). Both are real per-range sums of the exact same DK
scoring formula `dk_points_total` uses (`config.
NFL_BESTBALL_ROUND1_END_WEEK = 14`), not a different metric or a
simulated pod result - this project has no draft-pod/ADP data, so it
can't simulate real tournament advancement, only report how a player's
real production actually split across the two real stages. The
dashboard's Bestball table shows `R1 Score`/`R2-R4 Score` plus a real
season total at the end of the row, in place of the redundant `Games`
column (already covered by `Missed`/`Missed (Prior Yr)`) and the
`Season Snap %` column (kept in the underlying CSV for the position-
scarcity qualifier above, just not surfaced in this table).

**Real rank columns and live player search.** `build_bestball_rankings`
adds `overall_rank` (1-based, across every real position together) and
`position_rank` (the same real idea, computed separately within each
position, via pandas' own `rank(method="first")` so ties get consecutive
real ranks rather than an invented tie-break) - both surfaced as the
leading `Rank`/`Pos Rank` columns on the dashboard table, so a real
"QB12"/"WR3" read doesn't require counting rows by hand. A player search
box above the table filters by real player name as you type (`oninput`,
no button/Enter needed), combined with the existing position tabs.

The `Pos Rank` column also shows each player's real `points_z_score` in
parentheses (e.g. `1 (+2.8σ)`) - how many real standard deviations their
own real `dk_points_total` sits from their position's real,
outlier-excluded core mean (`compute_position_scarcity`'s own
`mean_dk_points`/`std_dk_points`, reused directly rather than a second
statistical basis). Deliberately the real SEASON-TOTAL z-score, not a
Round-1-only one - `position_rank` itself comes from `points_above_
replacement` (season-total based, further scaled by `necessity_ratio`),
so pairing it with a season-total z-score keeps the parenthetical
answering the same real question the rank does, rather than quietly
mixing in a different one. Computed for every real player in the table,
not just the snap-share/points-floor qualified ones (an honest "how far
from a typical qualified starter" read even for players who didn't clear
the bar); a real std of 0 or NaN (fewer than 2 real core players at that
position) gives a real NaN `points_z_score`, shown as a bare rank with no
parenthetical rather than a fabricated number.

**Real draft strategy is now baked directly into the ranking, not left
as a separate table to cross-reference.** Real user feedback: raw
`dk_points_total` alone was a misleading overall order - Josh Allen
ranked #4 overall on real 2025 points (389.6), well above where real
DraftKings ADP would ever put him, because QB is a real flat/deep
position (real CV 0.28 - see Draft Strategy below) where a QB drafted
many rounds later still scores close to a QB1's total. `overall_rank`/
`position_rank` are now computed from a new `points_above_replacement`
column instead of raw `dk_points_total`: each player's real
`dk_points_total` minus their OWN position's real roster-depth-derived
points floor (`compute_draftable_points_floor` - the same real floor
`compute_position_scarcity`'s qualifier already uses, see below) - a
real "how much do you actually lose by waiting on this position"
question, the same logic real ADP reflects. A flat position's floor
sits close to its ceiling (real 2025 QB: mean 260.2, floor 145.5 - not
much daylight), so even a QB1's raw points overstate their real value;
a scarce position's floor sits far below its stars (real 2025 RB: floor
43.6 vs. McCaffrey's 428.6), so real RB/WR production keeps most of its
raw-points rank. `position_rank` lands on the identical order either
way (subtracting one position's own constant floor can't reorder
players within that position), but `overall_rank` genuinely changes -
real 2025 effect: Josh Allen drops from raw-points rank #4 to real
`overall_rank` 13 (still `position_rank` 1, still the real #1 QB - the
adjustment is about cross-position value, not within-position ranking).
That's a real, substantial, directionally-correct move, though it lands
short of the ~25 the user's own recollection of DK ADP suggested -
expected, since real ADP also prices in things this project doesn't
model (injury risk, name value, rookie hype), and this metric is
independently derived from real roster-depth math, not fit to reproduce
a specific external number. `points_above_replacement` itself is also
published as a `Value` column for transparency. A position whose real
player pool doesn't reach its real roster-depth pool size gets a real
floor of 0 (not a fabricated exclusion) - falls back to ranking on raw
`dk_points_total`, same as before this feature existed.

Built via `scripts/build_nfl_bestball_rankings.py`
(`.github/workflows/build_nfl_bestball_rankings.yml`, `workflow_dispatch`
only) - a manual/one-time build, not a daily/weekly cron, since real
last-season stats don't change and this isn't meant to be a live-updated
feed.

**Position Scarcity** (`docs/data/nfl_position_scarcity.csv`,
`nfl_bestball.compute_position_scarcity`): a "how many difference-makers
exist at this position, and how many replacement-level guys" read for
draft strategy, not just a flat rank list. Per position (QB/RB/WR/TE):
the real total player pool size, how many of those players cleared a
real season-total-snap-share qualifier (`config.
NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE = 0.3`, "played at least 30% of
your team's total real offensive snaps this season" - a simple, honest,
not-backtest-derived first-pass default, deliberately lower than a
"majority of snaps" bar; see below for why), and a bell-curve breakdown
of those qualified players by how many standard deviations they sit
from their own position's mean (the standard empirical-rule bands - 1-2
SD, 2-3 SD, beyond 3 SD in both directions - with the central "within 1
SD" band further split into real quarter-SD slices, since most
draft-relevant players land there and one wide bucket would hide real
shape right where it matters most).

**This qualifier has gone through two real fixes, both driven by real
data, not guesswork:**

1. It used to be `games_played >= 8`, changed after real 2025 data
   showed it let in players with almost no real offensive role. A real
   return specialist (Jamal Agnew) had `games_played=11` - clearing the
   old bar easily - but played exactly 1 real offensive snap (0.18%
   share) all season and scored 0 real points; several similar
   special-teamers/inactive-but-rostered players sat right in the same
   "qualified" population as true starters. `games_played` only
   requires ANY real stat row that week (even a single special-teams
   play), not a meaningful offensive role - this is exactly why the
   table's mean and std were once suspiciously close together (real
   2025 WR: mean 99.2, std 83.1 under the old rule - std nearly as
   large as the mean, a genuine red flag).
2. The snap-share qualifier itself was then a real PER-GAME average
   (`avg_offense_pct`), which fixed the special-teamer problem but
   introduced a real, different small-sample problem in the opposite
   direction: a real one-or-two-game emergency spot start at a high
   per-game rate would qualify just as easily as a real every-week
   starter. Real example: Raiders backup QB Aidan O'Connell's real
   single-game rate was 82% across his exactly-1 real game played -
   comfortably clearing even a 50% per-game bar. `compute_player_snap_share`
   now computes a real SEASON-TOTAL share instead (real total snaps
   that season / team's real total offensive snaps that season, both
   summed over every real game, not just games the player appeared in)
   - the same O'Connell drops to a real 5.1% season share once measured
   against the Raiders' real full-season offensive-play total, and is
   correctly excluded.

Because a real season-total share runs meaningfully lower than a
per-game rate even for genuinely relevant players (real 2025 example:
Jaylen Warren and TreVeyon Henderson, both clearly real, draftable
committee-role RBs with 200+ real season DK points, sit at real season
shares of 0.47/0.46 - not the 0.5+ a "majority of snaps" framing might
suggest), the qualifier's threshold was lowered from 0.5 to 0.3
alongside this fix - see `config.NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE`'s
own docstring for the full real-data reasoning.

**A real production floor is ALSO required, layered on top of (not
instead of) the snap-share qualifier** (`nfl_bestball.
compute_draftable_points_floor`, `config.NFL_BESTBALL_DRAFTABLE_POOL_SIZE`).
Real user feedback: even at a real 50%+ snap-share bar, a position's
qualified-pool mean still didn't reflect "players who'd actually get
drafted" - real complementary-role players can clear a real snap-share
bar (a genuine role/health signal) while producing very little (a
separate value question snap share doesn't answer). The floor is set at
the real Nth-ranked player's own real `dk_points_total`, N derived from
real DraftKings Best Ball Mania roster-depth math - confirmed live via
web search against DraftKings' own published rules and major fantasy
outlets: real 12-team draft pods, 20 real roster spots/team, with real
published strategy guidance on typical per-team position counts (QB
2-3, RB 5-7 [6 the consistently-cited number], WR 6-8, TE 1-2) - the
midpoint of each real range, times the real 12-team pod size, gives a
real "how many players at this position does a typical pool actually
draft" depth: `{"QB": 30, "RB": 72, "WR": 84, "TE": 18}`. A position
with fewer real players than its pool size gets no real floor (real
data doesn't reach that deep) rather than a fabricated one.

**Real statistical outliers are excluded from the mean/std before the
bell curve is built, via Tukey's IQR-fence rule** (`config.
NFL_BESTBALL_SCARCITY_IQR_MULTIPLIER = 1.5`, the textbook convention,
not tuned for this project - chosen specifically because it doesn't
need an already-computed mean/std as an input, unlike a z-score rule,
which would be circular for exactly this problem). Those excluded
outliers still appear in the bell curve buckets (almost always in the
extreme bands, exactly where a real outlier belongs) - they're excluded
from the summary statistics, not from the table. Removing them does NOT
eliminate all spread, honestly: real production among a real qualified
population genuinely ranges from replacement-level to true
difference-makers, and that remaining spread is real, not leftover
contamination.

Real 2025 numbers as of this writing (via `python
scripts/build_nfl_bestball_rankings.py`), under BOTH the season-total
snap-share qualifier AND the roster-depth-based production floor - note
how much closer the WR/TE means now sit to real "would actually be
drafted" territory than under snap share alone (WR mean 127.8 -> 157.4,
TE 90.6 -> 172.8):

| Position | Total players | Qualified | Points floor | Outliers excluded | Mean DK pts | Std DK pts | Coefficient of variation |
|---|---|---|---|---|---|---|---|
| QB | 81 | 30 | 145.5 | 0 | 260.2 | 72.4 | 0.28 |
| RB | 151 | 46 | 43.6 | 1 | 198.4 | 82.4 | 0.42 |
| WR | 240 | 81 | 75.1 | 2 | 157.4 | 62.5 | 0.40 |
| TE | 136 | 17 | 130.8 | 1 | 172.8 | 23.9 | 0.14 |

Two honest real observations worth flagging:

- **RB's real points floor (43.6) is surprisingly low** and barely
  changed RB's qualified count (46, same as snap share alone) - real DK
  Best Ball roster-construction strategy deliberately drafts deep at RB
  (handcuffs/dart-throws for real injury insurance, a well-documented
  real strategy given the position's real injury volatility), so a
  rank-72 real cutoff lands on real committee/replacement-tier value,
  not a bug.
- **TE's qualified pool shrank the most** (68 -> 17) - a real reflection
  of how few NFL tight ends are real receiving weapons versus real
  in-line blockers who still play real meaningful snap shares without
  much real receiving production.

**Draft Strategy** (`docs/data/nfl_draft_strategy_takeaways.csv`,
`nfl_bestball.compute_draft_strategy_takeaways`): a real, numbers-driven
answer to "does this season's real spread argue for prioritizing this
position early in a draft, or waiting" - the direct question the bell
curve above raises but doesn't answer by itself. Ranks each position's
`coefficient_of_variation` (std/mean of its own real, outlier-excluded
core group) **relative to the other real positions this season**, not
against an invented absolute cutoff - "high" or "low" dispersion only
means something compared to the other real positions in the same real
season's data. Positions at or above the real median CV are read as the
more top-heavy/scarcer positions this season (a bigger real gap between
a difference-maker and a typical qualified player, so grabbing a proven
top-tier player there early carries more relative value); positions
below the median CV are read as flatter/deeper (real production is more
interchangeable, so it's generally safer to wait and spend an early
pick at a scarcer position instead).

This directly answers a natural question like "if QBs are all within 2
SD of the mean, should a TE be prioritized instead" - yes, exactly when
the other position's own real CV this season ranks higher (scarcer)
than QB's. Real 2025 read, under the corrected qualifier (season-total
snap share AND a real production floor): RB is now the SCARCEST
position (CV 0.42, real ranking #1 of 4), followed by WR (0.40), then
QB (0.28), with TE the flattest/deepest by a wide margin (CV 0.14) -
the fourth real read this table has produced across the qualifier's
three real fixes (originally QB flattest under games-played, then QB
scarcest under per-game average, then TE scarcest under season-total
share alone, now RB scarcest with the production floor added). Each
flip is itself a real illustration of how much the qualifier's
definition matters for this kind of relative comparison - not evidence
any one read was "wrong" so much as that a population contaminated with
the wrong players (special-teamers, short high-rate samples, or
low-value complementary role players) produces a real but misleading
CV, and only a population filtered on both real role AND real value
reflects players a real bestball drafter would actually be choosing
among.

**Position Necessity** (`docs/data/nfl_position_necessity.csv`,
`nfl_bestball.compute_position_necessity`): a real, distinct question from
the CV-based takeaway above - not "how spread out is production within
this position" but "how many of each position do we actually want on our
own 20-round roster, compared to how many real players are actually good
enough to draft this season." Compares real per-team roster-construction
targets (`config.NFL_BESTBALL_ROSTER_TARGET = {"QB": (2,3), "RB": (5,7),
"WR": (6,8), "TE": (1,2)}`, the same real DK Best Ball Mania guidance
`NFL_BESTBALL_DRAFTABLE_POOL_SIZE` is now DERIVED from, not independently
hardcoded) times a real 12-team pod (`config.NFL_BESTBALL_DRAFT_POD_SIZE
= 12`) against the position-scarcity table's own real `qualified_players`
count - real demand vs. real supply. Real 2025 read:

| Position | Roster target/team | Pod demand | Available (qualified) | Necessity ratio | Read |
|---|---|---|---|---|---|
| QB | 2-3 | 30 | 30 | 1.00 | roughly balanced |
| RB | 5-7 | 72 | 46 | 1.57 | real shortage |
| WR | 6-8 | 84 | 81 | 1.04 | roughly balanced |
| TE | 1-2 | 18 | 17 | 1.06 | roughly balanced |

RB is the one real standout: a 12-team pod wants 72 real RBs across its
20-round drafts, but only 46 real RBs actually clear this season's real
role+value bar - real bestball RB strategy (drafting deep/handcuffs for
real injury insurance, discussed above) means a real chunk of "the RBs a
pod drafts" are dart-throws, not genuinely startable difference-makers.

**This necessity_ratio is ALSO baked directly into `points_above_
replacement`/`overall_rank` in `nfl_bestball_rankings.csv`, not left as a
separate table to cross-reference only** - `build_bestball_rankings`
multiplies each player's VOR-adjusted value by their own position's real
necessity_ratio (both the points-floor subtraction and this multiplier
are per-position CONSTANTS, so `position_rank` is unaffected - only
`overall_rank` moves further). Real, honest compounding effect worth
flagging: since RB was already the position VOR rewarded most (see
above), stacking a real 1.57x necessity multiplier on top pushes RB even
further ahead - the real top 10 overall-ranked players are now ALL RBs
(previously a mix with WRs and Josh Allen in the top 10). Josh Allen's
`overall_rank` moves from 13 (VOR alone) to 26 (VOR + necessity) - QB's
necessity_ratio is right at 1.00 (roughly balanced), so this move is
entirely RB pulling further away, not QB being pushed down further. The
`read`/threshold cutoffs (>=1.1 shortage, <=0.9 surplus) are a simple,
honest first pass - NOT backtest-derived, easy to revisit once more real
seasons accumulate.

### Draft Assistant ("My Draft" - real, live-draft advice)

Everything above is a static preseason snapshot - it has no notion of an
in-progress draft. Real user ask: given a real current roster (e.g. 3 RBs,
2 WRs already drafted) and the pick you're at right now, what should you
actually do next? Three real gaps stood between the static tables above
and that ask, all addressed here: (1) real draft-state tracking - what's
on your roster, what pick you're at; (2) real market-consensus data to
judge "is this pick a reach"; (3) real snake-draft "picks until your next
turn" arithmetic.

**Real ECR data, not raw ADP.** Every raw ADP-provider domain (Fantasy
Football Calculator, FantasyPros itself, MyFantasyLeague) is blocked by
this sandbox's network egress policy, and this project has no HTTP client
dependency (see `requirements.txt`) to reach one directly regardless.
Instead, `nfl_ff_rankings.py` uses `nflreadpy.load_ff_rankings()`/
`load_ff_playerids()` (already-installed dependency, zero new packages) -
real FantasyPros Expert Consensus Rank (ECR) data, confirmed live via
`scripts/debug_nfl_data.py` (`.github/workflows/debug_nfl_data.yml`, run
31513644704, 2026-08-11) before any parsing code was written against
assumed field names, per this project's established data-confirmation
discipline. Filtered to real `ecr_type == "bo"` ("best-ball overall" - one
of nine real confirmed slices, the one that matches this project's
bestball focus directly and is already cross-position-comparable, like a
real draft pick number). Crosswalked from FantasyPros' own real player id
to this project's `player_id` (GSIS id) via `load_ff_playerids()`'s real
`fantasypros_id`/`gsis_id` columns - real confirmed coverage ~70.5%
(2211/3134 real 2025-rostered players), meaningfully lower than the 99.7%
`pfr_id` crosswalk `compute_player_snap_share` uses, so a real chunk of
players legitimately has no ECR match (absent from the output, never
fabricated). Unlike every other table on this page, ECR is a real
CURRENT-moment snapshot, not a per-season historical stat - fetched LIVE
at build time by `scripts/build_nfl_bestball_rankings.py` (not from
persisted parquet), and resiliently skipped (never a build failure) if
the real fetch fails that run.

**Roster gap and real necessity - buildable from data this project
already had.** `docs/nfl_draft_assistant.js` (real `node --test`
coverage, `docs/nfl_draft_assistant.test.js`, added to `ci.yml` - the
same real-test-coverage bar `docs/nfl_dfs_solver.js` already holds):
`computeRosterGap` compares your real entered roster against
`config.NFL_BESTBALL_ROSTER_TARGET`'s real per-team ranges (mirrored as a
JS constant, same "duplicate the real constant rather than share a
Python/JS module" convention `nfl_dfs_solver.js` already uses), reading
each position as real `need` (below the real min), `optional` (within
range), or `full` (at/above the real max). Combined with each position's
already-published real `necessity_ratio` (Position Necessity, above) -
every column a real, separate, transparent number, no invented composite
"take this player" score.

**No prediction of who's still on the board.** An earlier version of this
tool tried to guess a real "best remaining" player per position by
filtering `nflBestball` against ECR - real user feedback caught this
guessing at pick 59 that Josh Allen and Christian McCaffrey were still
available, when a real draft board would obviously have taken them long
before. The tool has no view of a live draft board and can't know what
OTHER teams have taken, so it stopped trying to guess. Instead, a second
search-and-add list ("Players I'm Considering For This Pick") lets you
name the specific players YOU'RE weighing right now, and the table
assesses only those real players: their real `points_above_replacement`
value, whether their position is a real roster need for you, the
position's real `necessity_ratio`, and a real reach/value read (below).
Drafting a considered player (adding them to My Roster) automatically
drops them from the considering list - they're no longer a decision to
weigh, they're on your team.

**Reach/value read and snake-draft math - the two pieces that needed real
ECR data.** `computeReachValueRead(ecr, ecrSd, pickNumber)`: a real,
honest comparison - your entered pick number against a player's real ECR
± their own real expert-rank standard deviation (no invented probability
model; both numbers come straight from the real confirmed `load_ff_rankings`
columns), applied per considered player rather than as a filter over the
whole player pool. A player with no real ECR match simply shows no read
(never a fabricated one), and is NOT excluded from consideration - you
told the tool you're weighing them, so it assesses what it honestly can.
`computeSnakeDraftPicksUntilNextTurn(draftSlot, podSize, pickNumber)`:
real snake-draft arithmetic (12-team pod, alternating direction each
round) from an entered real draft slot (1-12) and pick number - both
optional inputs; roster gap/necessity/considered-player assessment always
work without them.

**Entirely client-side, no account/server.** Both search-and-add lists
(My Roster, Players I'm Considering For This Pick) reuse the
already-loaded `nflBestball` array (no new CSV for search itself),
rendered as clickable candidate buttons (mirrors `docs/app.js`'s
`searchPlayer`/`showPlayer` pattern) with removable player cards (reusing
the existing `.removePitcher` button styling, laid out via the same
`.todaysPicks` flex-wrap container `docs/app.js` already uses elsewhere).
Your roster, considered-players list, draft slot, and pick number persist
to this browser's `localStorage` only (key `nflDraftAssistant.v1`) - a
public static GitHub Pages site with no per-user backend - with a "Clear
My Draft" button to reset.

**Preseason Notes** (`docs/data/nfl_draft_notes.csv`): a one-time,
hand-curated list of real training-camp storylines specifically for
players drafted in the April 2026 NFL Draft (depth-chart battles,
landing-spot opportunity, injury setbacks) from trusted sources (ESPN,
Yahoo Sports, SI, Fox Sports, NFL.com, team sites), each a short
paraphrase in original words with a link back to the source - never
reproduced article text. This list deliberately targets rookies: the
bestball rankings above are built entirely from real last-season stats,
so they have nothing to say about a player with no NFL track record
yet - this is where that gap gets filled in, once, by hand. This is
genuinely NOT automated - this codebase has zero precedent for fetching
external web content (no scraping/RSS/WebFetch anywhere), and the NFL
page's own design philosophy is "no new network call, same-origin CSV
fetches only."

## NFL Automated Game Picks (`docs/nfl.html`'s Game Picks tab)

A direct structural port of MLB's own Automated Game Picks (above) to
football, replicated signal-by-signal - not a from-scratch model. Real
NFL data made this genuinely easier to build than the MLB original in two
ways: `schedules_*.parquet` already carries real historical
`home_moneyline`/`away_moneyline`/`spread_line`/`total_line` for every
regular-season game back to 2016 (no separate `market_odds.py`-style
scraper needed), and `team_stats_*.parquet` already carries real,
already-computed `passing_epa`/`rushing_epa`/`receiving_epa` (no new
efficiency modeling needed from raw play-by-play).

**Team strength** (`nfl_team_strength.py`) mirrors `teams.py` exactly:
rolling win% and Pythagorean win% (`config.NFL_PYTHAGOREAN_EXPONENT=2.37`,
a real commonly-cited NFL literature value - not MLB's own 1.83) blended
across games-back windows (`config.NFL_TEAM_STRENGTH_WINDOWS`, 3/7/full
games). Real follow-up (2026-09-02): tightened from an initial 4/8/full
(an 18-week season's naive analog of MLB's 10/30/81/full) after a real
design discussion - an NFL season is too short for a long smoothing
window to "catch up" to a genuine regime change (a new starting QB, an
OC change, a key injury) the way MLB's 162-game season can, so recent
games are weighted more heavily. This is NOT because single NFL games
carry less real luck/variance than MLB's (the standard finding runs the
other way - a 17-game sample doesn't average out game-to-game noise the
way 162 games does, and raw turnover margin is famously low-persistence
year to year) - it's a real, honest tradeoff (faster to react to a
genuine shift, but also faster to overreact to one-game noise), validated
against the real backtest below rather than assumed. Each rolling signal
is z-normalized (`config.NFL_NORMALIZATION_Z_SCALE=0.15`) and mixed into
a Confidence signal the same way. `offensive_edge`/`defensive_edge`
replace MLB's `offensive_edge`/`suppression_resistance` with real total
EPA produced vs. allowed (the opponent's own `team_stats` row that same
week, joined via `opponent_team` - the same "opponent's own row for that
week IS what my defense allowed" pattern `nfl_teams.py` already used for
DFS matchups) - EPA is a real, well-regarded efficiency metric that fills
the same structural role as "held under 3 runs" without forcing an
invented baseball-specific threshold onto football.

**Turnover margin** (`nfl_team_strength.compute_team_turnover_margin`,
added 2026-09-02 - "we should include turnover ratio at a game level") -
real takeaways forced (`def_interceptions` + `fumble_recovery_opp`, a
recovery of the OPPONENT's own lost fumble) minus real turnovers given
away (`passing_interceptions` + `fumbles_lost_total`), both already
carried on a team's own `team_stats` row (no opponent join needed, unlike
offensive_edge/defensive_edge), blended across the same games-back
windows. Confirmed live: a team's own real turnovers-lost count matches
the real turnovers-forced count on the opponent's own row for that same
game in 541/544 (99.4%) of real 2025 team-games - the rare (~0.6%)
mismatch traces to a real fumble going out of bounds (possession changes
by rule with no recovery credited to either side), not a data error.

**Points per drive** (`nfl_team_strength.compute_team_points_per_drive`,
added 2026-09-02 - "offensive efficiency (pts/drive)") - a genuinely
different efficiency lens than `offensive_edge` (real per-PLAY EPA): how
often real drives actually turn into points, not how valuable each
individual play was. Sourced from real play-by-play (`nfl_data.fetch_pbp`,
`nflreadpy`'s `load_pbp` - not touched by this project before this, and
NOT bulk-backfilled across `config.NFL_HISTORICAL_SEASONS` the way every
other NFL table is, since one season alone is already ~20MB/~48.7k rows -
roughly 10x the combined size of every other NFL table this project
persists per season; fetched per-season on demand instead, current +
immediately-prior season, mirroring the live pipeline's own cold-start
pattern for every other table). Each real drive's own point value is the
real score change across it (`posteam_score_post` on the drive's last
real play minus `posteam_score` on its first), not guessed from
`fixed_drive_result`'s text label - this correctly handles a real 2-point
conversion, and correctly credits 0 points to the offense's own drive
when the DEFENSE scores against them mid-drive (a pick-six, a safety),
confirmed live against real 2025 plays. Considered and explicitly
NOT built: a homegrown "QBR-like" composite from the same play-by-play
data (`qb_epa`/`wpa`/`cpoe` are all real, confirmed-present columns) -
skipped since the model already has `passing_epa`/`passing_cpoe` doing
that job in the QB-continuity signal from the exact same underlying play
data, so a new composite would likely just duplicate existing signal for
real added engineering cost, not add new information; real ESPN Total
QBR itself is proprietary and isn't in `nflreadpy` at all, so it
couldn't be reproduced exactly regardless.

**QB continuity** (`nfl_team_strength.compute_qb_continuity_adjustment` +
`nfl_game_picks.py`'s own comparison) is the real NFL-specific signal
with no clean MLB analog - no MLB pitcher swings a team's win probability
the way an NFL starting QB does. `snap_counts_*.parquet`'s real per-game
`offense_pct` at the QB position identifies which QB has actually been
playing most recently (not just who a stale depth chart says); when the
game's own confirmed starter (`schedules_*.parquet`'s real `home_qb_id`/
`away_qb_id`) differs from that identified recent-primary QB, the team's
composite shifts toward the confirmed starter's own real rolling
`passing_epa_per_game` (`nfl_passing.compute_qb_rolling_stats`, reused
unchanged), weighted by `config.NFL_QB_CONTINUITY_WEIGHT=0.5`. Both
sides are z-normalized against the same population before differencing,
so when the confirmed starter IS the team's own recent-primary QB (the
ordinary case) the adjustment is exactly 0 with no separate "did the
starter change" branch needed.

`nfl_game_picks.compute_game_win_probabilities` combines six
z-normalized signals (`config.NFL_GAME_PICK_COMPOSITE_WEIGHTS`:
`pyth_Strength`, `pyth_Confidence`, `defensive_edge`, `true_power`,
`turnover_margin`, `points_per_drive`, equal-weighted at 1/6 each -
`turnover_margin`/`points_per_drive` were each added as their own
explicit weight rather than folded into `true_power`, since both are
genuinely distinct quality dimensions from EPA-based efficiency and
keeping them separate lets the backtest validate each independently)
plus the QB-continuity shift, then a simple ratio (not log5) into
`home_win_probability` - same "these composites aren't calibrated win
percentages" reasoning as the MLB original.

### Real backtest results (`scripts/run_nfl_game_picks_backtest.py`, `nfl_game_picks_backtest.py`)

Real no-lookahead week-by-week replay of the 2025 season (nflreadpy's own
real final scores and moneylines), split per the original build request
("use up to week 7 from last year to build and then the rest of the year
to test and calibrate models") into weeks 3-7 (train - weeks 1-2 are
excluded entirely, see below) and weeks 8-18 (test, held out). Reported
honestly, not cherry-picked:

| Split | Source | n | Accuracy | Brier | Beat closing line |
|---|---|---|---|---|---|
| Train (wk3-7) | Model | 76 | 56.6% | 0.244 | 36.8% |
| Train (wk3-7) | Market | 76 | 67.1% | 0.219 | - |
| Test (wk8-18) | Model | 164 | 57.9% | 0.246 | 36.0% |
| Test (wk8-18) | Market | 164 | 62.8% | 0.213 | - |

The raw heuristic beats a coin flip on real held-out data (57.9% test
accuracy) but trails the real market meaningfully (62.8%) and loses the
squared-error comparison on the majority of games (36.0% beat-closing-line,
below the 50% a genuinely competitive model would clear) - the exact same
finding MLB's own pre-calibration heuristic had (`home_win_probability`'s
real spread is far narrower than the market's own spread on the same
games), reported here with the same honesty rather than smoothed over.

**2026-09-02 real update** (tightened windows 4/8/full -> 3/7/full, plus
adding `turnover_margin` and `points_per_drive` as new composite signals,
in two real backtested steps): a real, modest, cumulative improvement,
not a breakthrough - train accuracy rose meaningfully (53.9% -> 56.6%),
test accuracy rose a bit (57.3% -> 57.9%), train beat-closing-line rose
(35.5% -> 36.8%), and test Brier improved slightly (0.247 -> 0.246).
Reported as-is rather than rounded up to a bigger win than the real
numbers show - still a narrow, unproven edge, not a solved model.

**Weeks 1 and 2 are excluded from every replay, for a real structural
reason, not an arbitrary cutoff**: week 1 has zero real prior-game history
within the season to build a rating from at all; week 2's history is
exactly one real game per team, so - confirmed live against the real 2025
season - every team's rolling rating collapses to the exact same real 0
(no games strictly before a team's own single game to draw a window from),
the whole population has zero variance, and z-normalization is a real,
honest 0/0 rather than a fabricated number. This is a genuine cold-start
property of validating a single season in isolation on purpose - the LIVE
pipeline (below) doesn't have this problem, since it carries real
prior-season history into early-season weeks instead.

**Calibration** (`scripts/train_nfl_game_pick_calibration.py`, same real
walk-forward-CV/final-holdout gate as MLB's own
`train_game_pick_calibration.py`) was fit against the same real 2025
replay and did **not** clear its own holdout bar yet (calibrated log_loss
0.702 vs. the raw heuristic's 0.695 on a 48-game final holdout - isotonic
badly overfit the real heuristic's very narrow probability range on this
little data). Nothing was saved; the live model runs uncalibrated
(`nfl_game_picks.apply_calibration`'s own graceful no-op) until a future
season's worth of additional real data can prove a real recalibration
out, exactly the same honest, unproven-until-earned status MLB's own
calibration artifact had before its first real backtest.

### Live pipeline (`nfl_pipeline.py`, `.github/workflows/nfl_weekly_update.yml`)

Runs weekly (Tuesday mornings, after Monday Night Football has resolved
and before the next week's slate firms up further - not daily, real NFL
games cluster Thu/Sun/Mon and this project has already learned firsthand
that shared GitHub Actions minutes are a real, finite budget). Each run:
fetches the current season fresh (falling back to the last persisted copy
of any table whose live fetch fails - real, live-confirmed preseason
case: nflverse doesn't publish a season's `team_stats`/`weekly`/`pbp`
file until real games start generating stats, and `snap_counts`/
`rosters_weekly` reject a season past their own currently-valid range
outright), resolves any picks from previously-logged pending weeks
against real final scores (a single bulk match against the freshly
fetched schedule - no per-date fetch loop needed, unlike MLB Stats API),
predicts the next real unplayed week, and writes
`docs/data/nfl_game_picks_*.csv`.

Logs to its own separate `data/predictions/nfl_game_predictions.csv` (a
distinct log from MLB's `game_predictions.csv`) and reuses `kelly.py`/
`config.KELLY_*` completely unchanged for bet sizing - `KELLY_DAILY_UNIT_CAP`/
`KELLY_MAX_SINGLE_BET_UNIT_CAP` apply **per sport**, so MLB's and NFL's
unit budgets are independent, never pooled. Verified end to end against
real live data: the actual pipeline script, run against the real 2026
preseason schedule (real Week 1 games, real already-posted moneylines,
2025 as cold-start history), produced real predictions and real
Kelly-sized bet advice - committed as the pipeline's first real output
(`data/predictions/nfl_game_predictions.csv`,
`docs/data/nfl_game_picks_*.csv`).

Like MLB's own daily log, this accumulates forward from when it ships -
no historical backfill of the live prediction log itself (the backtest
above is what validates the model against history, not the live log).
Playoff games (WC/DIV/CON/SB) are excluded from both the backtest and the
live pipeline's predictable-week logic - elimination-game dynamics, rest
advantages, and small sample don't fit the same statistical treatment as
the regular season.

## Running

```
pip install -r requirements.txt
python scripts/wave.py                        # today, using games through yesterday
python scripts/wave.py --as-of-date 2026-06-15 # re-run against a past date
```

`--as-of-date` only ever uses games strictly before that date, so a run can
never be computed from a game that hasn't happened yet (or is still in
progress) - this is also what lets the pipeline be re-run against any past
date once enough raw history has accumulated in `data/raw/`.

When `--as-of-date` isn't passed, "today" is `schedule.today_local()` -
Arizona time (fixed UTC-7, no DST), not the running machine's own local
time. `daily_update.yml`'s GitHub Actions runner is UTC, so a manual
`workflow_dispatch` run late in the Arizona evening (e.g. 8:45pm AZ =
3:45am UTC, already past UTC midnight) would otherwise publish tomorrow's
slate a day early for an Arizona audience. Same reasoning applies to
`scripts/build_dfs_rankings.py`/`build_hitter_hit_predictions.py`'s own
`--as-of-date` defaults. :)

## Tests

```
pip install -r requirements-dev.txt
pytest
```
