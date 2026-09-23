# Project brief: mlb_metrics

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
