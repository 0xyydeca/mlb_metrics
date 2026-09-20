# Project brief: mlb_metrics

## September 20 hit-prop identity and collection (Prompt 1 — complete)

- Selected research market remains **MLB 1+ hitter hits** (separate from game-winner). Profitability unproven. Automated orders absent; `GAME_PREDICTION_MODE=shadow`, `BETTING_MODE=disabled`.
- Identity + rules layer complete:
  - Maps contracts to **`game_pk`** (moneyline registry + schedule) and **`key_mlbam`** (normalized name + team-scoped candidates + provider map). Provider IDs are never treated as MLB IDs.
  - Quarantines ambiguous names, wrong-day starts, doubleheaders, provider-ID conflicts, and **traded/wrong-team** matches.
  - Persists contract text, `rules_hash`, and **`rules_version=hit_prop_rules_v1`**. Distinguishes start+PA eligibility, walk-only PA vs AB, zero hits, non-participation → LFMP (nonbinary), postponement, missing outcomes.
- Collection universe (configured budgets `event_limit=16` / `book_limit=120`):
  - Records complete denominator + exclusions; incomplete fetch stays explicit.
  - Durable store + checkpoint restart under `data/polymarket/research/hit_props/`.
  - Failures retained in capture JSON / `hit_prop_collection_status_latest.json`.
- Host: reuse free GitHub Actions `polymarket_capture.yml` (`ubuntu-latest`). Sleep outside cron 15–02 UTC; no paid add-ons without explicit budget. Fail-soft hit-prop step + host report: `reports/polymarket/hit_prop_host_ops_latest.json`.
- Live acceptance capture 2026-09-19 (America/Phoenix late window): **1** remaining future event (`mlb-min-laa-2026-09-19`); **8** already started excluded; **18** 1+ hit contracts; **18/18** `game_pk` + `key_mlbam` mapped; **0** quarantined; **18** books; status `captured`; all `actionable=false`.
- Traceable example: [reports/polymarket/hit_prop_contract_example_latest.json](reports/polymarket/hit_prop_contract_example_latest.json) — Austin Martin 1+ hits, `game_pk=823976`, `key_mlbam=668885`, rules require start+PA else LFMP, book yes@0.60 / no@0.44 with request/receive timestamps.
- Checks: `tests/test_hit_prop_research.py` → **20 passed** (duplicate names, DH, walk-only, missing outcomes, invalid prices, provider failure, restart, traded/wrong-team, host ops).
- Venue: provisional Polymarket US (owner confirmation still open); public research continues labeled independently.
- **Software readiness (prop collection/identity):** Yes (research-only).
- **Evidence of edge:** No.

### Remaining blockers (prop)

1. Incomplete universe when event/book budgets bind (must stay explicit).
2. Players absent from hitter-log candidates need verified provider→`key_mlbam` entries.
3. Nested prop evaluation outcomes **not opened**; labeled history still thin.
4. Do not convert positive-AB hit rates into contract probabilities.
5. Venue US vs international unconfirmed; no real-money mode change.

## Prompt 2 start (register study before outcomes)

- Registered `polymarket_us_mlb_hitter_hits_1plus_v1` at `reports/model_validation/hit_prop_paper_protocol.json` **before** examining nested evaluation outcomes.
- Candidates, chronological periods, costs, eligibility, metrics, pass/fail, and date-block sample plan recorded. Evidence-collection estimate through 2026-09-25: **insufficient_data expected** (≤7 independent dates ≪ 70-date floor).
- Forecast model fitting / paper accounting freeze: continue through Sept 21–25 without claiming validation.

## September 18 audit notes (prior)

- Evidence and reproduction: [September 18 findings](</Users/kyaryeh/Documents/ChatGPT/MLB stats/research/2026-09-18/FINDINGS.md>).
- Game-winner verdict remains **`insufficient_evidence`**. Logged sportsbook P&L interpretive corrections from that audit stand; not Polymarket profit.

## Purpose and owner decisions

- Personal MLB Polymarket research; profitability is a goal, not an established result.
- Manual bets only; system must not place orders.
- Risk limits: agent paper units (20u / 5u max loss / 1u per-bet / 3u daily / 1u same-game / 1u same-team) — not USD bankroll.
- Venue US vs international unconfirmed → **provisional Polymarket US**.
- Commit/push only to fork `0xyydeca/mlb_metrics` **`main`**.

## Current plan

- Keep game-winner prospective protocol v2 collecting; do not bypass its gates.
- Advance 1+ hit prop research: identity+collection (**Prompt 1 done Sept 20**), then forecasts + paper study workflow (**Prompt 2**, Sept 21–25).
- Real-money pilot remains unauthorized.

## Commands

```bash
# Hit-prop research capture (research-only)
PYTHONPATH=src python scripts/capture_hit_prop_research.py \
  --output-dir /tmp/mlb-hit-prop-research --persist-store

# Register prop paper protocol (before evaluation outcomes)
PYTHONPATH=src python scripts/register_hit_prop_paper_protocol.py

PYTHONPATH=src python -m pytest tests/test_hit_prop_research.py -q
```

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| Prop identity + durable collection software? | **Yes** (research-only). |
| Prop edge / real-money pilot? | **No**. |
| Game-winner edge / real-money pilot? | **No** (`insufficient_evidence`). |
| Prop nested eval opened? | **No**. |

## Next implementation task

Prompt 2 (Sept 21–25): build separate prop forecasting + paper-evaluation path matched to contract participation rules; keep missing quotes missing; fit only in training folds; freeze policy; report insufficient evidence if floors unmet. Do not enable real money.
