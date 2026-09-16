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
- **Next research market selected (feasibility only): NFL pregame moneylines** — minimal public capture/mapping only. No NFL model training and no real-money use until NFL paper protocol checkpoints are met and nested outcomes are opened.
- Target repo: `0xyydeca/mlb_metrics`. Full MLB plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md). Migration: [docs/SPORT_ADAPTER_MIGRATION.md](docs/SPORT_ADAPTER_MIGRATION.md).

## Evidence snapshot (verified 2026-09-16)

### MLB
- Modes: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.
- Protocol `polymarket_us_game_winner_v2`; verdict **insufficient_evidence** (0 labeled nested-eval dates). Remaining regular season ≤12 calendar dates vs floor 70.
- Future observations not claimed. No edge claimed.

### Market selection (registered before comparative outcomes)
- Protocol: `reports/model_validation/market_selection_protocol.json` (`market_selection_next_sport_v1`).
- Feasibility: `reports/model_validation/market_selection_feasibility.json`.
- Statuses: NFL **feasible**; NBA **uncertain**; NHL **uncertain**.
- **Selected:** `nfl_pregame_moneyline` → `proceed_minimal_capture_only`.
- Selection is **not** based on a profitable backtest. One NFL season is unlikely to hit the 70-date floor; multi-season collection still required for promotion-scale evidence.

### NFL live sample (public US API; limitations apply)
- Capture `2026-09-16T18:48:33Z`: **32** moneylines discovered; **30 mapped** to `nfl:{game_id}`; **32** books written; HTTP failures **0**.
- Registry cumulative: 52 rows (30 mapped / 22 unmatched). Quote index has executable asks; latest-slice top-of-book spreads observed ~0.005 (not treated as free liquidity; depth beyond BBO not claimed).
- Mapping requires Eastern kickoff composition from nflreadpy `gameday`+`gametime` (date-only midnight UTC rejected by tests).
- Sample limitations: single public-gateway series; unmatched markets excluded; venue still provisional US; no account access assumed.

### Registered NFL validation plan (nested outcomes not opened)
- `reports/model_validation/nfl_polymarket_paper_protocol.json` (`polymarket_us_nfl_game_winner_v1`).
- Gates: model development blocked; real money blocked; MLB evidence reuse forbidden.
- Structural floor still 70 eligible dates (or documented redesign). Freeze/holdout assigned only after floors.

## Completed vs future

| Completed now | Future (not done) |
|---|---|
| Sport adapter contracts (`sports/*`) | NBA/NHL adapters |
| `list_moneyline_markets(league=…)` + MLB wrapper | International venue |
| NFL registry/quotes namespaced under `data/polymarket/nfl/` | NFL nested eval outcomes |
| NFL capture script + mapping tests | NFL residual/heuristic Polymarket models |
| Market-selection protocol + feasibility | Real-money / stake guidance |
| NFL paper protocol registered (plan only) | Multi-sport dashboard |
| MLB compatibility replay tests | |

## Engineering / evidence estimates

| Item | Estimate |
|---|---|
| Capture/mapping (done minimal) | ~2–4 eng-days equivalent delivered |
| Nested-eval readiness (paper ledger join, settlement QA) | ~5–10 eng-days remaining |
| Meaningful validation vs 70-date floor | Multi-season collection (or explicit floor redesign) |
| Sources | Polymarket US sports/fees docs; in-repo `nfl_data` / adapters |

## Commands

```bash
# MLB (unchanged)
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py

# Market selection (register before feasibility outcomes)
PYTHONPATH=src python scripts/register_market_selection_protocol.py
PYTHONPATH=src python scripts/write_market_selection_feasibility.py
PYTHONPATH=src python scripts/register_nfl_paper_protocol.py

# NFL minimal capture (does not touch MLB registry/quotes)
PYTHONPATH=src python scripts/capture_polymarket_nfl.py --limit 50

# Compatibility / focused tests
PYTHONPATH=src python -m pytest tests/test_sport_adapters.py tests/test_paper_ledger.py tests/test_data_foundation.py tests/test_polymarket_phase1.py -q
```

## Checks run

- `pytest` sport adapters + MLB polymarket/paper suites → **47 passed** (focused set including NFL midnight-reject mapping).
- NFL capture refreshed → 30/32 mapped; books under `data/polymarket/nfl/quotes` only.
- NFL paper protocol written; nested outcomes **not** opened.
- No unintended MLB production path writes from NFL capture.

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| MLB engineering / collection? | **Yes** (paper/protocol active). |
| MLB edge / real money? | **No**. |
| Cross-sport reuse scaffolding? | **Yes** (small adapters; not a framework rewrite). |
| NFL research capture? | **Started** (public data only). |
| NFL model / real money? | **No** — gated on protocol checkpoints. |

## Remaining blockers / gaps

- Owner venue confirmation; personal risk limits.
- MLB + NFL both need multi-period labeled history for structural floors.
- NFL unmatched markets outside schedule lookahead still expected; deepen mapping QA.
- Near-start quote cadence vs 30s freshness bar.
- Unknowns that could reverse NFL selection: systematic missing US books; confirmed different venue; material postpone/neutral-site settlement divergence.

## Next implementation task

Keep MLB collection running; schedule NFL capture; meet NFL protocol checkpoints before any nested NFL evaluation outcomes.
