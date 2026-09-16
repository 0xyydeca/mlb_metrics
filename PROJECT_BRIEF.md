# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner places bets manually; the system must not place orders.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- **Personal bankroll, max affordable loss, and exposure caps have not been supplied.** Stake guidance stays off until entered in the dashboard Risk tab.
- **Open question (venue):** which Polymarket venue the owner actually uses (US vs international) is still unconfirmed. Research remains **provisional Polymarket US** (`polymarket_us_provisional_research`).
- **Owner workflow:** commit and push only to fork `0xyydeca/mlb_metrics` **`main`**. Do not create feature branches for routine work. Do not open pull requests against `JMerchen/mlb_metrics`.

## Current plan

- Scope: pregame full-game winner contracts. Engineering priority only; profitability not demonstrated.
- **Prospective paper-evaluation protocol v2 is registered** and collection is initiated on GitHub Actions + local capture. Labeled nested-eval history remains insufficient for an edge claim.
- Target repo only: `0xyydeca/mlb_metrics`.
- Next: keep capture + paper-ledger logging running through the rest of 2026 and into 2027; report only at registered checkpoints; do not inspect the freeze tail during tuning.
- Full plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md).
- Daily ops: [docs/POLYMARKET_DAILY_OPS.md](docs/POLYMARKET_DAILY_OPS.md).

## Evidence snapshot (verified 2026-09-16)

- Modes unchanged: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.
- **Protocol:** `polymarket_us_game_winner_v2` (version 2), registered `2026-09-16T03:20:00Z`, hash in `reports/model_validation/polymarket_paper_protocol.json`.
- **Frozen policy:** paper-only / betting disabled; hash in `polymarket_frozen_policy.json`.
- **Exploratory dates (not untouched validation):** `2026-09-14`, `2026-09-15`.
- **Prospective collection start (America/Phoenix):** `2026-09-16`.
- **Checkpoints (game days):** 7, 14, 28, 70 — 7/14 are progress-only; promotion requires structural floor + gates.
- **Remaining 2026 regular season:** through **2026-09-27** → at most **12** calendar dates from collection start. Structural floor **70**; power plan ~**197** independent days for 0.01 log-loss edge. **This regular season cannot meet the floor.** Postseason (from 2026-09-29) is a **separate cohort** and does not fill the regular-season requirement.
- Evaluation verdict: **insufficient_evidence** / `insufficient_data` (0 Polymarket labeled nested-eval dates). Future observations are **not** claimed.
- Paper ledger continues logging candidates/passes before outcomes (114 passes / 0 buys on last cycle).
- Repository: https://github.com/0xyydeca/mlb_metrics

## Current technical facts

- Protocol registration (no outcome inspection): `scripts/register_polymarket_protocol.py`.
- Evaluation report (reproducible, fail-closed): `scripts/run_polymarket_paper_evaluation.py`.
- Collection host: GitHub Actions `polymarket_capture.yml` (ubuntu-latest) — capture `--with-baseball`, paper ledger cycle, artifacts. No paid services purchased; none missing for this path.
- External / operational blockers: venue unconfirmed; 30-minute cron cannot guarantee a 60s post-delay book; remaining season date count below floor.
- Foundation modules unchanged in role: registry, quote store, baseball snapshots, paper ledger/pipeline, decision board.

## Commands

```bash
# Register protocol only (before inspecting outcomes)
PYTHONPATH=src python scripts/register_polymarket_protocol.py

# Collection (read-only)
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --settle-date YYYY-MM-DD

# Health / audit
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# Checkpoint evaluation report (does not enable betting)
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py

# Focused tests
PYTHONPATH=src python -m pytest tests/test_polymarket_protocol_v2.py tests/test_paper_pipeline.py tests/test_paper_ledger.py tests/test_data_foundation.py -q
```

## Checks run

- `scripts/register_polymarket_protocol.py` → protocol v2 + frozen policy + collection status written; remaining season **12 < 70**.
- `scripts/run_polymarket_paper_evaluation.py` → **insufficient_evidence** / `insufficient_data`.
- `pytest` protocol/paper/foundation/decision/phase1 → **56 passed**.
- Paper ledger cycle → decisions logged; buys suppressed while gates fail.
- Capture workflow updated to persist baseball + run paper ledger + upload ledger artifacts (Actions host).

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| **Engineering / collection readiness?** | **Yes** — protocol registered, frozen policy saved, Actions collection path initiated, reports reproducible from saved inputs. |
| **Evidence of an edge / real-money readiness?** | **No** — `insufficient_evidence`; remaining 2026 regular season cannot supply the structural date floor; betting stays disabled. |

## Remaining blockers / gaps

- Confirm owner venue (US vs international).
- Supply bankroll + max affordable loss before stake guidance.
- Need ≥70 Polymarket labeled eligible dates (then freeze + outer folds); continue collection past 2026-09-27.
- Near-start denser quotes for conservative delay stress remain an operational gap on the 30-minute cron.

## Next implementation task

Keep authorized capture + paper-ledger logging running; settle when Finals exist; write checkpoint reports at 7/14 days without inspecting a freeze tail that has not been assigned yet.
