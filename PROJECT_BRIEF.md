# Project brief: mlb_metrics

## September 30 collection recovery and workflow repair

- Verified GitHub run 36674930606 and recovered all 47 available `polymarket-quotes` artifacts dated September 19–30 UTC. Each ZIP matched GitHub's SHA-256 digest. Originals, artifact manifest, recovery script, and per-run inventory are saved in the working-document folder under `research/2026-09-30/recovery/`.
- Recovered 1,561 unique prop observation rows (1,227 captured books) on September 23, 24, 25, 26, 27, and 29. Merged with 116 existing rows without conflicting capture-ID/market-slug keys. Archives before September 23 contain no prop capture snapshots; September 28 captures contain no eligible contracts. All moneyline files remain preserved in original ZIPs, not merged into production here. Missing observations are not reconstructed.
- Confirmed production defect: paper ops and readiness crashed with missing `sklearn`, but shell wrappers returned success. Added pinned model dependencies, a fresh-runner import check, and visible nonzero failures for prop capture/ops while retaining always-run artifact upload. Removed unconditional half-hour formal scoring; registered checkpoint evaluation remains separate.
- Confirmed persistence defect: quote artifacts existed, but subsequent runners never restored the ignored prop store. Workflow now commits the compact contracts, capture summaries, and prospective ledger to main. Full response bodies remain artifacts; summaries retain exact capture ID/date/universe. Checkpoints are deliberately not persisted, so later runs can obtain refreshed books instead of skipping markets already captured.
- Verification: 45 focused tests passed; full suite 1,143 passed and one existing data-integrity test failed. Fetched main already contains four sportsbook rows with missing game_pk (provider event 401817088, September 25); no threshold/test was weakened and those rows were not silently removed. 33 JavaScript tests and workflow YAML/shell syntax checks passed. Local Python 3.14 differs from hosted Python 3.12. Workflow/dependency pushes now trigger one hosted verification run; generated-data commits do not retrigger it.
- Fresh isolated dependency install/import succeeded. A live September 30 capture obtained 43 contracts/books, all mapped to MLB game/player IDs and zero quarantined, with research-only flags. Actual capture ran locally; the generated host report describes the configured GitHub host, not proof of remote execution.
- Live paper command completed without the former import crash: 43 passes, zero buys, clean reconciliation. It logged today's decision time, never backdated decisions for recovered rows. Ops counts two candidate dates (September 19 and 30), not two validated evaluation dates. Current command supplies no confirmed lineup map and uses market-mid probabilities; it does not serve an independently validated prop model. Existing brief claims that all paper-model work is complete were too broad.
- Remaining blockers: verify repaired hosted execution; sparse actual schedule cadence (recent runs several hours apart despite half-hour cron); genuine missing historical decisions; confirmed lineup/model integration; separate postseason evidence; owner's actual venue and personal USD limits. Frozen policy and evidence gates unchanged; no orders or nested-outcome scoring.

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
