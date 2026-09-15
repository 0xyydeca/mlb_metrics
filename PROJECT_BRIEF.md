# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner places bets manually; the system must not place orders.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- Personal bankroll, acceptable losses, and account/trade-execution authorization have not been supplied. Research and engineering can proceed with public data and paper results.
- **Open question (venue):** which Polymarket venue the owner actually uses (US vs international) is still unconfirmed. Research remains **provisional Polymarket US** and is labeled as such in capture/health output.

## Current plan

- Scope: pregame full-game winner contracts. Engineering priority only; profitability not demonstrated.
- Phase 0 + Phase 1 base are on fork `main`. Data-foundation hardening (capture reliability, coverage waterfall, lineup/starter snapshots) is on `codex/data-foundation-trustworthy-manual`.
- Target repo only: `0xyydeca/mlb_metrics` (do not open PRs against `JMerchen/mlb_metrics`).
- Next after this branch merges: paper ledger / decision display (Phase 3), while quotes accumulate.
- Full plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md).

## Evidence snapshot (verified 2026-09-15)

- Planning audit baseline: commit `6eb45e3`.
- Prior Phase 0/1 commits: `0e357fc`, `565201a`, `282a3aa`.
- Saved coverage report that showed **42 mapped / 0 books / stale quotes** was diagnosed as a **collector run with zero successful book fetches** (failures previously swallowed), not proof of empty market liquidity. Live recheck on 2026-09-15: books parse successfully; sample run captured **25/25 eligible books** with `http_failures=0`.
- Live API schema check: `GET /v1/markets/{slug}/book` returns `marketData.bids` / `offers` with `{px:{value}, qty}` and `state`; units are USD share cost in `[0,1]` and contract quantities.
- Mapping ≠ usable prices: registry can be 42/42 mapped while only the contracts with fresh eligible books count as usable (example after `--limit 25`: usable_prices=25; older mapped rows may be stale).
- Lineup Stage A: schedule `lineups.homePlayers/awayPlayers` expose MLBAM `id` when announced; **no `battingOrder` field**. List index is announced order only and can diverge from post-start boxscore (verified Final `game_pk=824465`). Today’s slate often has lineups unconfirmed until near start; probable pitchers are available earlier.
- Modes unchanged: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.
- Repository: https://github.com/0xyydeca/mlb_metrics

## Current technical facts

- Research venue label: `polymarket_us_provisional_research` (public gateway; no paid dependency for capture).
- Capture: `scripts/capture_polymarket.py` (retries/backoff, book failure taxonomy, liquidity vs HTTP failure, coverage waterfall, optional `--with-baseball`).
- Health: `scripts/health_polymarket.py` and `scripts/audit_polymarket_data.py` — report mapping and usable price coverage separately.
- Quotes: date-partitioned Parquet + index; idempotent by `raw_response_hash`; gitignored; workflow artifacts upload quotes.
- Registry/coverage JSON are trackable.
- Baseball inputs: `game_baseball_snapshots.py` + lineup parser with `LINEUP_API_SCHEMA_CONFIRMED=True` and explicit `batting_order_source=schedule_lineups_list_index`.
- Fee schedules versioned in `config.POLYMARKET_US_FEE_SCHEDULES` (θ=0.06; θ=0.0695 from `2026-09-17T03:59:00Z`).

## Commands

```bash
# Collection (read-only)
PYTHONPATH=src python scripts/capture_polymarket.py
PYTHONPATH=src python scripts/capture_polymarket.py --limit 50 --with-baseball

# Health / audit (mapping vs usable prices)
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# Lineup schema probe
PYTHONPATH=src python scripts/debug_statsapi_lineups.py
```

Runtime note: expect roughly 0.2–0.5s per market book; a ~40-market capture is typically well under 60s when the gateway is healthy. Actionable freshness target remains 30s per contract — cron every 30 minutes cannot keep all contracts fresh between runs; decisions must re-check.

## Checks run

- `pytest tests/test_data_foundation.py tests/test_polymarket_phase1.py tests/test_lineup_snapshots.py tests/test_market_odds_snapshots.py tests/test_game_residual_model.py` → **73 passed**.
- Live capture + health on 2026-09-15 → books captured; sample contract traced (mapping, book, fee version, home executable buy, probable pitchers; lineups unconfirmed for today’s games).

## Remaining blockers / gaps

- Owner venue confirmation (US vs international) still open.
- Residual model still lacks enough sportsbook history / artifact for independent promotion.
- Confirmed batting lineups are often absent until near first pitch; game-winner usability currently keys off **probable pitchers**, with lineup status recorded separately.
- Paper ledger / decision UI not built yet.
- Fork Actions Polymarket workflow had **0 runs** observed at diagnosis time; schedule may still need a few successful executions.
- Do not treat mapped_rate as tradable coverage.

## Next implementation task

Phase 3 paper ledger + local decision display that consumes traced quotes/fees/baseball pass reasons, still with betting disabled.
