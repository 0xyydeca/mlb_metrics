# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner places bets manually; the system must not place orders.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- **Risk limits (owner delegated 2026-09-16):** agent set **normalized paper units** only — bankroll 20u, max loss 5u, per-bet 1u, daily 3u, same-game 1u, same-team 1u. Not a personal USD bankroll; not from an income target; `real_money_usd_stakes_enabled=false`. Override with explicit USD later if/when a pilot is considered.
- **Open question (venue):** which Polymarket venue the owner actually uses (US vs international) is still unconfirmed. Research remains **provisional Polymarket US**.
- **Owner workflow:** commit and push only to fork `0xyydeca/mlb_metrics` **`main`**. Do not create feature branches for routine work. Do not open pull requests against `JMerchen/mlb_metrics`.

## Current plan

- MLB prospective paper protocol v2 remains active; collection continues (regular season alone cannot meet the 70-date floor).
- **Gated manual pilot workflow delivered** (readiness dual conclusions, rejects, pauses, real vs paper reconcile, exposure-review protocol). Real-money pilot is **not** authorized.
- Cross-sport adapters + NFL minimal capture remain as previously selected (research capture only).
- Target repo: `0xyydeca/mlb_metrics`. Full MLB plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md). Ops: [docs/POLYMARKET_DAILY_OPS.md](docs/POLYMARKET_DAILY_OPS.md).

## Evidence snapshot (verified 2026-09-16)

### Dual readiness (pilot)
- Report: `reports/model_validation/polymarket_pilot_readiness.json`.
- **Software operates correctly:** Yes (inspection / Pass / no-bet).
- **Evidence supports limited real-money pilot:** **No**.
- `pilot_authorized=false`. `edge_claimed=false`. Modes: shadow / betting disabled.
- Market/protocol checked: `mlb_pregame_moneyline` / `polymarket_us_game_winner_v2`.
- Specific no-bet reasons include: `evidence_verdict:insufficient_evidence`, `validation_status:insufficient_data`, `insufficient_labeled_dates:0<70`, gate failures, `BETTING_MODE=disabled`, `risk_limits_are_paper_units_not_usd_bankroll`.
- Pause: evidence deterioration keeps new bets blocked; existing exposure preserved.
- Exposure-increase review registered: `polymarket_us_pilot_exposure_review_v1` (no recovery boosting).
- Risk limits file: `data/polymarket/pilot/risk_limits.json` (paper units).

### MLB paper protocol
- Protocol `polymarket_us_game_winner_v2`; verdict **insufficient_evidence** (0 labeled nested-eval dates). Remaining regular season ≤12 calendar dates vs floor 70.
- Future observations not claimed. No edge claimed.

### Market selection / NFL
- NFL selected for minimal public capture only; nested NFL outcomes not opened. See prior feasibility artifacts.

## Completed vs future

| Completed now | Future (not done) |
|---|---|
| Pilot readiness dual conclusions + report | Real-money pilot authorization |
| Manual dashboard fields + recheck/rejects | Owner risk limits + venue confirmation |
| Real fill ledger path + reconcile vs paper | ≥70 labeled dates + nested pass |
| Pause conditions + exposure-review protocol | Any stake increase |
| NFL minimal capture (prior task) | NFL nested eval / models |

## Commands

```bash
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball
PYTHONPATH=src python scripts/write_pilot_readiness_report.py
PYTHONPATH=src python scripts/register_pilot_exposure_review.py
PYTHONPATH=src python scripts/serve_decision_dashboard.py

PYTHONPATH=src python -m pytest tests/test_manual_pilot.py tests/test_decision_board.py -q
```

## Checks run

- `pytest` `test_manual_pilot` + `test_decision_board` → **18 passed** (rejects, pauses, reconcile, restart, exposure review).
- Readiness report written with `software=True`, `evidence_pilot=False`, `pilot_authorized=False`.

## Readiness verdict (separate)

| Question | Verdict |
|---|---|
| MLB engineering / collection / pilot UI? | **Yes**. |
| MLB edge / real-money pilot? | **No**. |
| NFL research capture? | Started (prior). |
| NFL / MLB models for betting? | **No**. |

## Remaining before a pilot

1. ≥70 Polymarket-labeled eligible dates; nested eval `validated_passed` + `edge_supported`.
2. Owner: replace paper-unit caps with explicit USD bankroll/max-loss if/when considering a pilot.
3. Owner venue confirmation; then consider `BETTING_MODE` (orders still manual).
4. Do not lower thresholds; do not treat market-only fallback as independent model.

## Next implementation task

Continue MLB (+ optional NFL) collection; re-run readiness at registered checkpoints only. Do not enable real money until remaining items above are met.
