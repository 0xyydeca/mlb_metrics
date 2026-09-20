# Project brief: mlb_metrics

## September 20–25 hit-prop Prompt 2 (forecasts + paper experiment)

- Separate prop forecast/eval path added; **game-winner protocol unchanged**.
- Target alignment: contract Yes ≈ `P(start+PA) × P(Got_Hit|qualify)`. Positive-AB rates **rejected** as silent contract payouts. `Game_Hit_Probability` used only as a conditional proxy when paired with an explicit qualify rate.
- Same-time market mid from executable yes/no only; missing historical quotes stay missing.
- Candidates registered: market mid baseline, contract-rule-adjusted baseball proxy, regularized market-residual logistic. Walk-forward calibration fits **training dates only**.
- Protocol + frozen paper policy written before nested outcomes:
  - `reports/model_validation/hit_prop_paper_protocol.json`
  - `reports/model_validation/hit_prop_frozen_policy.json` (`action=paper_only`, `BETTING_MODE=disabled`)
- Evaluation report: **`insufficient_data` / `insufficient_evidence`**. Nested outcomes **not opened**. Collection dates observed: **2**; **68** more independent dates to structural floor (70); planning power target ~197 date-blocks.
- Paper ledger path: size-limited asks walk, fees, delay/adverse stress, LFMP + binary settlement, skipped/unfilled. Demo: `reports/polymarket/hit_prop_forecast_settlement_example_latest.json`.
- Checks: `tests/test_hit_prop_paper.py` + `tests/test_hit_prop_research.py` → **27 passed**.
- **September 25 = paper-system review, not a betting launch.**
- **Software readiness (prop paper pipeline):** Yes (research-only).
- **Evidence of edge / validated model:** No.

### Remaining blockers (prop)

1. Nested labeled history far below 70-date floor (and ~197 planning power target).
2. Opportunity-model artifact not required for serving; qualify-rate joins still thin.
3. Venue US vs international unconfirmed; `BETTING_MODE=disabled`.
4. Do not convert positive-AB hit rates into contract probabilities.

## September 20 hit-prop identity and collection (Prompt 1 — complete)

- Maps `game_pk` / `key_mlbam`; quarantines ambiguous/DH/wrong-day/traded; `rules_version=hit_prop_rules_v1`.
- Durable research store + GHA host ops; live example Austin Martin `game_pk=823976`, `key_mlbam=668885`.

## September 18 audit notes (prior)

- Game-winner verdict remains **`insufficient_evidence`**.

## Purpose and owner decisions

- Personal MLB Polymarket research; profitability is a goal, not an established result.
- Manual bets only; system must not place orders.
- Risk limits: agent paper units (20u / 5u max loss / 1u per-bet / 3u daily / 1u same-game / 1u same-team) — not USD bankroll.
- Venue US vs international unconfirmed → **provisional Polymarket US**.
- Commit/push only to fork `0xyydeca/mlb_metrics` **`main`**.

## Current plan

- Keep game-winner prospective protocol v2 collecting; do not bypass its gates.
- Continue prop paper collection toward structural floors; do not claim validation or open nested outcomes early.
- Real-money pilot remains unauthorized.

## Commands

```bash
# Hit-prop research capture (research-only)
PYTHONPATH=src python scripts/capture_hit_prop_research.py \
  --output-dir /tmp/mlb-hit-prop-research --persist-store

# Register protocol + freeze paper policy (before nested outcomes)
PYTHONPATH=src python scripts/register_hit_prop_paper_protocol.py

# Paper evaluation (expect insufficient_data through Sept 25)
PYTHONPATH=src python scripts/run_hit_prop_paper_evaluation.py

# Paper ledger demo (no orders)
PYTHONPATH=src python scripts/run_hit_prop_paper_ledger.py

PYTHONPATH=src python -m pytest tests/test_hit_prop_paper.py tests/test_hit_prop_research.py -q
```

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| Prop identity + durable collection software? | **Yes** (research-only). |
| Prop forecast + paper ledger pipeline? | **Yes** (research-only). |
| Prop edge / validated model / real-money? | **No** (`insufficient_data`). |
| Game-winner edge / real-money pilot? | **No** (`insufficient_evidence`). |
| Prop nested eval opened? | **No**. |
| Sept 25 betting launch? | **No** (paper-system review only). |

## Next implementation task

Continue prospective prop quote/outcome collection under the frozen paper policy. Re-run `run_hit_prop_paper_evaluation.py` at checkpoints (7/14/28/70 dates). Open nested evaluation only after structural floors; keep insufficient_data as a valid result. Do not enable real money.
