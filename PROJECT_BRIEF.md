# Project brief: mlb_metrics

## October 2 maintenance — nullable lineup repair

- Synced owner main from local `b77ef9a` to `48aa1d1bd0e1ffa91e7ceceb3c3aac78c1b1945c`; incoming changes were generated data/reports. Preserved the existing October 1 brief edits. Both briefs were compared; their dated current findings agree, while older implementation-complete/readiness entries remain historical.
- Reproduced the partial-lineup crash in isolated fixtures: 17 of 18 new helper regression cases failed before the fix. Missing `pd.NA` values raised an ambiguous-boolean error; all-missing CSV-like columns also rejected boolean/string scratch updates. A scratch-only overlay incorrectly marked other unknown players as nonstarters.
- Fixed `apply_confirmed_lineup_to_pool` to normalize nullable starter/status columns before scratch updates and apply confirmed-starter appearance only to known true flags. Unknown flags remain unknown; existing appearance estimates survive; scratches keep zero appearance. No formula, threshold, selection version, or frozen policy changed.
- Isolated pipeline unit tests from live lineup responses. Added a full pipeline doubleheader case with only one confirmed lineup, plus helper cases covering unmatched players, separate game IDs, explicit false flags, missing statuses and missing appearance estimates. Existing confirmed-starter and scratch behavior remain covered.
- Verification: 32 lineup helper tests and 14 pipeline cases pass; full Python suite **1,162 passed, one failed**, 1,163 warnings, 126.25 seconds. All **33 JavaScript tests pass**; `git diff --check` clean. Runtime: local Python 3.14.7, pandas 3.0.5, pytest 9.1.1; hosted Python 3.12 was not reproduced. Logs and verification metadata saved under `research/2026-10-02/` in the working-document folder.
- The sole full-suite failure is the existing production-snapshot integrity gate: four NYY–BAL sportsbook rows still have `source_status=ambiguous_match` and missing `game_pk`. No rows were silently removed, guessed, relabeled or remapped. Production history, configuration and readiness gates are unchanged by this repair.
- **Next bounded task:** resolve the ambiguous sportsbook event against exact MLB game identity, or preserve the original observations in explicit quarantine and prevent unmapped rows from entering production prediction snapshots. Keep the existing integrity gate. Then verify the owner-fork daily workflow reaches current pregame generation/publishing on Python 3.12; do not backdate predictions.
- Daily publishing is not yet verified repaired end-to-end. No daily update, training, capture, backtest or formal holdout scoring was dispatched for this maintenance task. Postseason cohort/coverage defects and independent lineup/model integration remain separate open work; real-money readiness remains insufficient.

## October 1 evening audit — current findings

- Report prepared from observed state at October 1, about 8:37 p.m. America/Phoenix (October 2 UTC), not a backdated morning forecast. [Interactive report](/Users/kyaryeh/.cursor/projects/Users-kyaryeh-Documents-git-mlb-metrics/canvases/MLB-october-readiness-report.canvas.tsx) includes upcoming games, executable-price snapshots, evidence comparisons, and five repair prompts. Raw evidence and pinned source manifest: `/Users/kyaryeh/Documents/ChatGPT/MLB stats/research/2026-10-01/`.
- Fresh remote references: owner `0e1bf251774e105643c757bdb395e096a1c82f6f`; upstream `a57ceab4e27d88f3e1bb732cb49cb3eb9f8fdc8f`. Local implementation remained at `b77ef9a`; remote data was read using pinned Git objects, without merging or training.
- Daily forecast publishing remains broken independently of the repaired quote collector. Latest [daily run 36896569861](https://github.com/0xyydeca/mlb_metrics/actions/runs/36896569861) failed: 10 failed, 1,134 passed on hosted Python 3.12. Nine pipeline failures reach `bool(pd.NA)` in `lineup_snapshots.py:432`; the other failure concerns four sportsbook rows missing `game_pk`. Downstream generation/publishing was skipped. Five latest daily runs inspected all failed. Next bounded engineering task: reproduce/fix nullable lineup handling and resolve or explicitly quarantine the invalid sportsbook rows without weakening tests; verify a hosted current pregame publish.
- Owner's latest published game-prediction date is September 15; upstream's is October 1. Both correctly predicted 405/739 matched resolved games (June 11–September 15). Logged theoretical sportsbook results remain negative: fork 21/50 actual bet-side wins, -7.1571 units, -6.29% ROI; upstream 39/121, -40.5737 units, -21.82% ROI. Cohorts differ; these are descriptive logs, not paired return experiments or Polymarket fills.
- Current remote prop metadata: 1,835 observation rows, 1,427 books, 21 captures over 10 dates; 42 rows lack game IDs and 41 lack player IDs. The 205 prospective decisions on three candidate dates are all passes, all use `same_time_market_mid_baseline`, and all have unknown lineup availability. Candidate dates are not certified eligible independent evidence dates.
- Confirmed cohort defect: `hit_prop_ops.classify_cohort` uses month >= 10, so September postseason games can be counted as regular season. Correct derived eligibility from verified MLB game type with an audit trail; preserve immutable decisions, policy versions, and holdouts. Current `LINEUP_API_SCHEMA_CONFIRMED=True` is a configuration flag, not evidence that paper ops supplied confirmed lineups.
- Official schedule showed the October 1 game final, no October 2 games, and four October 3 Division Series games. Full event details for all four October 3 Polymarket US events were fetched successfully; each had eight markets, none an eligible 1+ hit prop at observation time. The collector incorrectly reports zero events fetched because it counts via contract rows; fix request accounting independently of contract presence.
- Read-only October 3 moneyline books and exact rules are saved. Official US fee documentation at inspection gives theta 0.0695 effective October 1; adapter fee-version label still references September 17. Refresh verified fee provenance before decisions. No independent current forecasts support an edge calculation for these four games.
- This report task changed research artifacts and briefs only. It did not repair implementation, rerun full tests, dispatch workflows, backdate decisions, score prop holdouts, or enable betting. Actual venue and personal USD limits remain unknown. The earlier September 25 review target is historical; no supported real-money launch date exists.


## September 30 collection recovery and workflow repair

- **Hosted verification completed:** repair commit `ae75ea219b084661eb8c40787627266902a753a9` pushed to owner main. [Run 36769494111](https://github.com/0xyydeca/mlb_metrics/actions/runs/36769494111) passed dependency installation/imports on Python 3.12, captured 43 mapped contracts/books, logged 43 passes and zero buys, uploaded artifacts, and committed history as `7eaa284`. Synced back locally: 1,763 contract observations and 104 paper decisions, no duplicate capture/market keys or decision IDs. These counts include preserved prior rows and two September 30 capture cycles, not 1,763 independent bets.

- Verified GitHub run 36674930606 and recovered all 47 available `polymarket-quotes` artifacts dated September 19–30 UTC. Each ZIP matched GitHub's SHA-256 digest. Originals, artifact manifest, recovery script, and per-run inventory are saved in the working-document folder under `research/2026-09-30/recovery/`.
- Recovered 1,561 unique prop observation rows (1,227 captured books) on September 23, 24, 25, 26, 27, and 29. Merged with 116 existing rows without conflicting capture-ID/market-slug keys. Archives before September 23 contain no prop capture snapshots; September 28 captures contain no eligible contracts. All moneyline files remain preserved in original ZIPs, not merged into production here. Missing observations are not reconstructed.
- Confirmed production defect: paper ops and readiness crashed with missing `sklearn`, but shell wrappers returned success. Added pinned model dependencies, a fresh-runner import check, and visible nonzero failures for prop capture/ops while retaining always-run artifact upload. Removed unconditional half-hour formal scoring; registered checkpoint evaluation remains separate.
- Confirmed persistence defect: quote artifacts existed, but subsequent runners never restored the ignored prop store. Workflow now commits the compact contracts, capture summaries, and prospective ledger to main. Full response bodies remain artifacts; summaries retain exact capture ID/date/universe. Checkpoints are deliberately not persisted, so later runs can obtain refreshed books instead of skipping markets already captured.
- Verification: 45 focused tests passed; full suite 1,143 passed and one existing data-integrity test failed. Fetched main already contains four sportsbook rows with missing game_pk (provider event 401817088, September 25); no threshold/test was weakened and those rows were not silently removed. 33 JavaScript tests and workflow YAML/shell syntax checks passed. Local Python 3.14 differs from hosted Python 3.12. Workflow/dependency pushes now trigger one hosted verification run; generated-data commits do not retrigger it.
- Fresh isolated dependency install/import succeeded. A live September 30 capture obtained 43 contracts/books, all mapped to MLB game/player IDs and zero quarantined, with research-only flags. Actual capture ran locally; the generated host report describes the configured GitHub host, not proof of remote execution.
- Live paper command completed without the former import crash: 43 passes, zero buys, clean reconciliation. It logged today's decision time, never backdated decisions for recovered rows. Ops counts two candidate dates (September 19 and 30), not two validated evaluation dates. Current command supplies no confirmed lineup map and uses market-mid probabilities; it does not serve an independently validated prop model. Existing brief claims that all paper-model work is complete were too broad.
- Remaining blockers: sparse actual schedule cadence (recent runs several hours apart despite half-hour cron); genuine missing historical decisions; confirmed lineup/model integration; separate postseason evidence; owner's actual venue and personal USD limits. Frozen policy and evidence gates unchanged; no orders or nested-outcome scoring.

## September 30 automated maintenance — verified local state

- Checkout began clean at `ec1e070`. The implementation brief now records Prompts 1–4 delivered; the September 18 working-folder account of unfinished identity/model plumbing is historical. These completion claims were not fully re-audited today.
- Fixed paper-ops coverage provenance: metadata is loaded only from the selected capture ID, with matching embedded ID and local date. Missing, malformed, or mismatched snapshots remain explicit; another capture can no longer supply the denominator. Loaded-row counts and incomplete-day labeling cannot be overwritten by snapshot metadata. Forecast/selection policy and registered evidence gates are unchanged.
- Verification: 21 focused tests passed; full Python suite 1,140 passed (1,163 warnings); 33 JavaScript tests passed; final loader check 8 passed; `git diff --check` clean. Local Python 3.14.7/Node 24.2.0 differ from CI Python 3.12/Node 22, which were not reproduced.
- Added isolated regression tests for matched historical slices, unrelated newer snapshots, missing/malformed metadata, ID/date mismatches, invalid universe shape, and dates without rows.
- Local contracts contain 116 rows across September 18–19 only. The latest saved readiness report is dated September 20 and records one eligible date with `insufficient_evidence`; it is not a fresh September 30 evaluation. The September 25 review date in the older brief has passed. Remote captures and workflow success remain unverified.
- Next bounded task: inspect the existing capture workflow's recent remote runs/artifacts and recover legitimate prospective history if available; do not reconstruct missing prediction-time quotes or open nested outcomes early.
- No capture, training, ledger migration, readiness scoring, order placement, commit, or push was run in this maintenance task. Changes remain local. Actual venue and personal USD limits remain unresolved; betting stays disabled. Owner chose local Codex automation and no additional paid API budget.

## Prompt 4 — Formal readiness evaluation (registered checkpoint)

- Evaluated exact frozen policy `polymarket_us_mlb_hitter_hits_1plus_v1_frozen_v1` (`policy_hash=2f08c06d052b3115`) against preregistered protocol requirements.
- Pre-score audits: dataset denominators, cutoff integrity (outcomes unknown at decision), identity/rules fields, model/policy hashes, untouched evaluation period **not assigned** (floor unmet).
- Structural fold readiness and statistical precision reported **separately**; scoring blocked until structural floor + freeze assignment.
- **Verdict: `insufficient_evidence`** (`validation_status=insufficient_data`). Scoring not attempted. No thresholds lowered; no favorable subperiod selection; eval set not recycled for tuning.
- Denominators: **1** eligible independent decision date; **69** additional dates to structural floor (70); planning power target still ~197 date-blocks. Next permitted ops checkpoint: **7** dates (~6 remaining). Next registered review: **2026-09-25** (paper-system review, not betting launch).
- Report: `reports/model_validation/hit_prop_readiness_latest.json` (`report_hash` recorded). Mirrored into `hit_prop_paper_evaluation.json`.
- `BETTING_MODE=disabled`; real money not enabled. Software tests ≠ edge evidence.
- Checks: `tests/test_hit_prop_readiness.py` → **6 passed**.

## Prompt 3 — Operate frozen paper system

- Prospective ops under frozen policy; sim vs prospective stores separated; restart drill passed; GHA fail-soft collector.

## Prompts 1–2

- Identity/collection + forecast/paper protocol registered before nested outcomes.

## Purpose and owner decisions

- Personal MLB Polymarket research; profitability unproven.
- Manual bets only; `GAME_PREDICTION_MODE=shadow`, `BETTING_MODE=disabled`.
- Venue provisional Polymarket US (unconfirmed).
- Commit/push only to fork `0xyydeca/mlb_metrics` **`main`**.

## Commands

```bash
PYTHONPATH=src python scripts/run_hit_prop_readiness_evaluation.py

PYTHONPATH=src python scripts/run_hit_prop_paper_ops.py
PYTHONPATH=src python scripts/capture_hit_prop_research.py \
  --output-dir /tmp/mlb-hit-prop-research --persist-store

PYTHONPATH=src python -m pytest tests/test_hit_prop_readiness.py -q
```

## Readiness verdict

| Question | Verdict |
|---|---|
| Prop software / paper ops? | **Yes** (research-only). |
| Prop edge supported? | **No** — `insufficient_evidence`. |
| Real-money mode? | **No** (`BETTING_MODE=disabled`). |
| Next checkpoint | **7** eligible dates; review **2026-09-25**. |

## Next implementation task

Continue prospective collection under the frozen policy. Re-run readiness at 7/14/28/70 eligible-date checkpoints without opening nested scoring early or enabling real money. Preserve this negative/insufficient result in history.
