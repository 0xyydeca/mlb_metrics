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
- **MLB data/accounting foundation is engineering-ready** on fork **`main`** (mapping, books, fees, baseball snapshots, paper ledger cycle, settlement join, contract trace). Labeled evaluation history is still insufficient for an edge claim.
- Target repo only: `0xyydeca/mlb_metrics`.
- Next: keep quote + baseball capture running; accumulate labeled Polymarket dates to structural/power floors; re-run evaluation only at registered checkpoints.
- Full plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md).
- Daily ops: [docs/POLYMARKET_DAILY_OPS.md](docs/POLYMARKET_DAILY_OPS.md).

## Evidence snapshot (verified 2026-09-15 / refreshed 2026-09-16)

- Modes unchanged: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.
- Protocol / evaluation: verdict **insufficient_evidence** (`validation_status=insufficient_data`); Polymarket labeled eligible dates for nested eval still **0**.
- Live capture (2026-09-16): 40 books written, 0 HTTP failures; registry 57/57 mapped; baseball snapshots 15 usable persisted.
- Contract trace (real observed): `market_id=800763`, `game_pk=822849`, rules hash + quote + fee version `us_taker_theta_0.06_2026-07-01`. Conservative 60s delay **fails closed** when no newer book exists after delay (`missing_quote_after_manual_delay`). Two-book delay fills verified in tests. Report: `reports/polymarket/contract_trace_latest.json`.
- Paper cycle: 114 decisions logged (all **pass** while evidence gates fail); no production buys; fixtures remain under `fixture_*.csv` only.
- Missing for labeled eval: evidence gate pass (0 Polymarket nested-eval dates). Baseball snapshots are now persisted.
- Repository: https://github.com/0xyydeca/mlb_metrics

## Current technical facts

- Venue adapters: `venues/polymarket_us.py` (live), `polymarket_international.py` (explicit unsupported). Settlement API client: `fetch_market_settlement`.
- Registry / quotes: `market_contracts.py`, `quote_store.py` (as-of quote selection, ask/bid depth parsers, restart-safe dedupe).
- Baseball as-of: `game_baseball_snapshots.py` (persist default with `--with-baseball`).
- Paper ledger: `paper_ledger.py` + **`paper_pipeline.py`** (candidates/passes, delayed fills, settle from MLB Final results).
- Scripts: `capture_polymarket.py`, `health_polymarket.py`, `audit_polymarket_data.py`, `run_polymarket_paper_ledger.py`, `run_polymarket_paper_evaluation.py`, decision dashboard export/serve.
- Liquidity vs collector: empty/missing ask → persisted ineligible book; HTTP exceptions → `book_failures` / exit 2 when zero books.

## Commands

```bash
# Collection (read-only)
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/capture_polymarket.py --limit 50 --with-baseball --no-persist-baseball

# Health / audit (mapping ≠ usable prices)
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# Paper ledger + end-to-end contract trace
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --trace-only
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --settle-date 2026-09-15

# Evaluation protocol (does not enable betting)
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py

# Decision dashboard
PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball
PYTHONPATH=src python scripts/serve_decision_dashboard.py

# Focused tests
PYTHONPATH=src python -m pytest tests/test_paper_pipeline.py tests/test_paper_ledger.py tests/test_data_foundation.py tests/test_decision_board.py tests/test_polymarket_phase1.py -q
```

## Checks run

- `pytest` (paper pipeline + ledger + foundation + decision board + phase1) → **52 passed**.
- Live capture → 40 books written, baseball 15 usable persisted.
- Contract trace → **ok** for observed mapped contract (source → quote → fees); conservative delayed fill unfilled without a post-delay book; still missing `evidence_gate_pass`.
- Paper cycle → 114 passes, 0 buys (`BETTING_MODE=disabled`).
- Health → actionable suppressed when quotes age past 30s; mapping ≠ fresh prices.

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| **Engineering readiness** (foundation software for mapping, books, fees, as-of baseball, paper accounting, settlement join, trace)? | **Yes** — ready to operate and accumulate history. Automated orders remain absent. |
| **Evidence of an edge / real-money readiness?** | **No** — `insufficient_evidence`; 0 Polymarket labeled nested-eval dates; personal limits unset; venue unconfirmed. |

## Remaining blockers / gaps

- Confirm owner venue (US vs international).
- Supply bankroll + max affordable loss before stake guidance.
- Need Polymarket as-of books joined to official results across enough dates (floor 70; ~197 days for registered 0.01 log-loss power plan).
- Near-start 1-minute capture vs 30-minute cron remains an operational gap for the 30s actionable quote bar (recheck API covers manual action).
- Historical display-price backfill still unused for fill backtests (depth-only fills).

## Next implementation task

Continue capture; settle paper positions when Final results exist; re-run evaluation only at registered checkpoints without inspecting the freeze tail.
