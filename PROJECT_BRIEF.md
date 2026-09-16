# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner places bets manually; the system must not place orders.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- **Personal bankroll, max affordable loss, and per-bet / daily / same-game / same-team exposure caps have not been supplied.** Stake guidance stays off until the owner enters them in the dashboard Risk tab (browser localStorage only). Research continues with normalized paper units.
- **Open question (venue):** which Polymarket venue the owner actually uses (US vs international) is still unconfirmed. Research remains **provisional Polymarket US** and is labeled as such in capture/health output.
- **Owner workflow:** commit and push only to fork `0xyydeca/mlb_metrics` **`main`**. Do not create feature branches for routine work. Do not open pull requests against `JMerchen/mlb_metrics`.

## Current plan

- Scope: pregame full-game winner contracts. Engineering priority only; profitability not demonstrated.
- Data foundation, paper ledger / protocol, and **manual decision dashboard** are on fork **`main`**.
- Target repo only: `0xyydeca/mlb_metrics`.
- Next: accumulate Polymarket labeled history to the structural/power floors; keep collecting quotes; dashboard stays paper / Pass-only until gates + personal limits pass.
- Full plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md).
- Daily ops: [docs/POLYMARKET_DAILY_OPS.md](docs/POLYMARKET_DAILY_OPS.md).

## Evidence snapshot (verified 2026-09-15)

- Modes unchanged: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.
- Protocol registered before new outcomes: `reports/model_validation/polymarket_paper_protocol.json`.
- Exploratory dates (not untouched validation): `2026-09-14`, `2026-09-15`.
- Paper ledger hand examples reconcile (walk asks + fees + win settlement; canceled ≠ auto $0; display price ≠ fill).
- **Verdict: insufficient evidence** (`validation_status=insufficient_data`).
  - Polymarket labeled eligible dates for nested eval: **0** (structural floor 70; ~197 independent days planned for 0.01 log-loss edge at power 0.8 under σ=0.05 assumption).
  - Mapped contracts and quote index rows exist but are not a labeled evaluation set.
  - Sportsbook residual shadow history is diagnostic only; switching priors to Polymarket requires revalidation and is not claimed here.
- Repository: https://github.com/0xyydeca/mlb_metrics

## Current technical facts

- Research venue label: `polymarket_us_provisional_research`.
- Capture/health: `scripts/capture_polymarket.py`, `scripts/health_polymarket.py`, `scripts/audit_polymarket_data.py`.
- Paper ledger: `src/mlb_metrics/paper_ledger.py` (append-only decisions/positions under `data/polymarket/ledger/`).
- Research protocol + gates: `src/mlb_metrics/polymarket_research.py`.
- Evaluation runner: `scripts/run_polymarket_paper_evaluation.py`.
- Decision board: `src/mlb_metrics/decision_board.py`, export `scripts/export_polymarket_decision_board.py`, UI `docs/polymarket.html` + `docs/polymarket.js`, local host `scripts/serve_decision_dashboard.py` (quote recheck API; no orders).
- Board artifacts: `docs/data/polymarket_decision_board.csv`, `docs/data/polymarket_decision_meta.json`, mirrored `docs/data/polymarket_paper_evaluation.json`.
- Reports: `polymarket_paper_protocol.json`, `polymarket_frozen_policy.json`, `polymarket_paper_evaluation.json`.
- Explicit board states: Paper candidate; Pass—insufficient evidence; Pass—price too high; Pass—stale quote; Pass—missing inputs; Game started. High win probability alone never recommends.

## Commands

```bash
# Collection (read-only)
PYTHONPATH=src python scripts/capture_polymarket.py
PYTHONPATH=src python scripts/capture_polymarket.py --limit 50 --with-baseball

# Health / audit (mapping vs usable prices)
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# Register protocol + paper evaluation evidence report
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py

# Decision board export + dashboard
PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball
PYTHONPATH=src python scripts/serve_decision_dashboard.py
# → http://127.0.0.1:8765/polymarket.html

# Focused tests
PYTHONPATH=src python -m pytest tests/test_paper_ledger.py tests/test_decision_board.py tests/test_data_foundation.py -q
```

## Checks run

- `pytest tests/test_paper_ledger.py tests/test_decision_board.py` → **21 passed**.
- `scripts/run_polymarket_paper_evaluation.py` → wrote protocol/policy/evaluation; verdict **insufficient_evidence**.
- `scripts/export_polymarket_decision_board.py` → 114 board rows; `paper_candidates=0`, `actionable=0`, verdict `insufficient_evidence`.

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| Does the software work for inspection and Pass / no-bet decisions? | **Yes** — board export, explicit states, paper vs manual streams, risk config UI, recheck API (no orders). |
| Does prospective evidence support using real money? | **No** — `insufficient_evidence` / `insufficient_data`; `BETTING_MODE=disabled`; personal limits not configured. |

## Remaining blockers / gaps

- Need Polymarket as-of books joined to official results across enough dates for three complete outer blocks + freeze (floor 70 dates; power plan ~197 days for the registered 0.01 log-loss target under the stated σ).
- Owner venue confirmation still open.
- Owner must supply bankroll + max affordable loss (and confirm exposure caps) before stake guidance can enable.
- Residual sportsbook artifact still insufficient for independent promotion; not used as Polymarket evidence.
- Betting stays disabled while gates fail.

## Next implementation task

Keep quote capture running; join settled outcomes into the paper ledger automatically; re-run evaluation only at registered checkpoints without inspecting the freeze tail.
