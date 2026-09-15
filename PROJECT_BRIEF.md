# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner places bets manually; the system must not place orders.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- Personal bankroll, acceptable losses, and account/trade-execution authorization have not been supplied. Research and engineering can proceed with public data and paper results.

## Current plan

- Start with pregame full-game winner contracts. Engineering priority only; profitability not demonstrated.
- Phase 0 completed on `codex/phase-0-evidence-foundation` (`0e357fc`).
- Phase 1 base adapter landed on `codex/phase-1-polymarket-adapter` (`565201a`).
- Follow-up on the same branch: live StatsAPI mapping schedule + scheduled Polymarket capture workflow.
- Next: Phase 2 as-of baseball inputs (lineups) while quotes accumulate; then paper ledger / decision display.
- Full plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md).

## Evidence snapshot

- Planning audit: September 14, 2026, commit `6eb45e3`.
- Phase 0: `0e357fc`. Phase 1 adapter: `565201a`.
- Mapping/capture hardening: in progress on `codex/phase-1-polymarket-adapter` (uncommitted until owner commits).
- Repository: https://github.com/0xyydeca/mlb_metrics

## Current technical facts

- Selected research venue: **Polymarket US** (public gateway). International remains unsupported stub.
- Mapping schedule: live StatsAPI for `today_local()` + `POLYMARKET_SCHEDULE_LOOKAHEAD_DAYS` (3), merged with snapshot rows for gaps.
- Capture: `scripts/capture_polymarket.py` / `scripts/audit_polymarket_data.py`.
- Workflow: `.github/workflows/polymarket_capture.yml` cron `:27/:57` in the MLB window; uploads quote artifacts; commits registry + coverage JSON only.
- Registry/coverage are trackable; quote Parquet remains gitignored.
- Modes: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.

## Mapping/capture follow-up (verified)

- Root cause of low mapping: schedule snapshots lagged Polymarket listings (Sep 15 evening games absent).
- Fix: `market_contracts.load_mapping_schedule()` prefers live StatsAPI games.
- Live check after fix: **30 mapped / 12 unmatched** on a 30–50 market pull (unmatched mostly Sep 17–18 beyond prior 2-day window; lookahead raised to 3).
- Wrong-day same matchups stay unmatched via time tolerance (e.g. Sep 14 CIN/LAD vs Sep 15 listing).
- Tests: `tests/test_polymarket_phase1.py` **12 passed**.

## Remaining blockers

- Residual model still lacks enough sportsbook history / artifact.
- Polymarket quote history just starting; no paper ledger / decision UI yet.
- Owner bankroll and live-trading authorization still unknown; betting stays disabled.
- Actions schedule on the fork may lag until the workflow has run a few times.

## Next implementation task

Phase 2: confirmed lineup / as-of baseball inputs for game-winner features, while scheduled Polymarket capture accumulates books.

## Sources

- [Polymarket US sports data](https://docs.polymarket.us/data-guide/sports-data)
- [US market book](https://docs.polymarket.us/api-reference/markets/get-market-book)
- [US fees](https://docs.polymarket.us/fees)
- [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md)
