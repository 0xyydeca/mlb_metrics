# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner places bets manually; the system must not place orders.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- **Personal bankroll, max affordable loss, and exposure caps have not been supplied.** Stake guidance stays off until entered in the dashboard Risk tab.
- **Open question (venue):** which Polymarket venue the owner actually uses (US vs international) is still unconfirmed. Research remains **provisional Polymarket US**.
- **Owner workflow:** commit and push only to fork `0xyydeca/mlb_metrics` **`main`**. Do not create feature branches for routine work. Do not open pull requests against `JMerchen/mlb_metrics`.

## Current plan

- MLB prospective paper protocol v2 remains active; collection continues (regular season alone cannot meet the 70-date floor).
- **Cross-sport reusable adapters delivered** (venues + sport schedule contracts). MLB paths/CSV semantics preserved.
- **Next research market selected (feasibility only): NFL pregame moneylines** — minimal public capture/mapping only. No NFL model training and no real-money use until a separate NFL paper protocol + validation.
- Target repo: `0xyydeca/mlb_metrics`. Full MLB plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md). Migration: [docs/SPORT_ADAPTER_MIGRATION.md](docs/SPORT_ADAPTER_MIGRATION.md).

## Evidence snapshot (verified 2026-09-16)

### MLB
- Modes: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.
- Protocol `polymarket_us_game_winner_v2`; verdict **insufficient_evidence** (0 labeled nested-eval dates). Remaining regular season ≤12 calendar dates vs floor 70.
- Future observations not claimed. No edge claimed.

### Market selection (registered before comparative outcomes)
- Protocol: `reports/model_validation/market_selection_protocol.json`.
- Feasibility: `reports/model_validation/market_selection_feasibility.json`.
- Statuses: NFL **feasible**; NBA **uncertain**; NHL **uncertain**.
- **Selected:** `nfl_pregame_moneyline` → `proceed_minimal_capture_only`.
- Live NFL sample (public US API): 20 markets discovered; **15 mapped** to `nfl:{game_id}`; 12–20 books written depending on run; HTTP failures 0 on last capture. Mapping required Eastern kickoff composition from nflreadpy `gameday`+`gametime`.
- Selection is **not** based on a profitable backtest. One NFL season is unlikely to hit the 70-date floor; multi-season collection still required for promotion-scale evidence.

## Completed vs future

| Completed now | Future (not done) |
|---|---|
| Sport adapter contracts (`sports/*`) | NBA/NHL adapters |
| `list_moneyline_markets(league=…)` + MLB wrapper | International venue |
| NFL registry/quotes namespaced under `data/polymarket/nfl/` | NFL paper protocol / nested eval |
| NFL capture script + mapping tests | NFL residual/heuristic models |
| Market-selection protocol + feasibility report | Real-money / stake guidance |
| MLB compatibility replay tests | Multi-sport dashboard |

## Commands

```bash
# MLB (unchanged)
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py

# Market selection (register before feasibility outcomes)
PYTHONPATH=src python scripts/register_market_selection_protocol.py
PYTHONPATH=src python scripts/write_market_selection_feasibility.py

# NFL minimal capture (does not touch MLB registry/quotes)
PYTHONPATH=src python scripts/capture_polymarket_nfl.py --limit 50

# Compatibility / focused tests
PYTHONPATH=src python -m pytest tests/test_sport_adapters.py tests/test_paper_ledger.py tests/test_data_foundation.py tests/test_polymarket_phase1.py -q
```

## Checks run

- `pytest` sport adapters + MLB polymarket/paper suites → **64 passed** (full focused set).
- MLB fixture fill/mapping replay equivalent after refactor.
- NFL capture → mapped contracts present; books stored under `data/polymarket/nfl/quotes` only.
- No unintended MLB production path writes from NFL capture.

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| MLB engineering / collection? | **Yes** (paper/protocol active). |
| MLB edge / real money? | **No**. |
| Cross-sport reuse scaffolding? | **Yes** (small adapters; not a framework rewrite). |
| NFL research capture? | **Started** (public data only). |
| NFL model / real money? | **No** — gated on future validation. |

## Remaining blockers / gaps

- Owner venue confirmation; personal risk limits.
- MLB + NFL both need multi-period labeled history for structural floors.
- NFL unmatched markets outside schedule lookahead still expected; deepen mapping QA.
- Near-start quote cadence vs 30s freshness bar.

## Next implementation task

Keep MLB collection running; optionally schedule NFL capture; register an NFL-specific paper protocol before any nested NFL evaluation outcomes.
