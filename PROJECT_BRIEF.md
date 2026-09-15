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
- Phase 1 (read-only Polymarket US adapter + registry + quote store) implemented on `codex/phase-1-polymarket-adapter`.
- Next: improve mapping coverage / scheduled capture reliability, then Phase 2 as-of baseball inputs (lineups) overlapping continued quote collection.
- Full plan: [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md).

## Evidence snapshot

- Planning audit: September 14, 2026, commit `6eb45e3`.
- Phase 0: September 14–15, 2026, `0e357fc` on `codex/phase-0-evidence-foundation`.
- Phase 1: September 14–15, 2026, branch `codex/phase-1-polymarket-adapter`.
- Repository: https://github.com/0xyydeca/mlb_metrics

## Current technical facts

- Selected research venue: **Polymarket US** (`POLYMARKET_VENUE_SELECTED=polymarket_us`, public gateway `https://gateway.polymarket.us`).
- International venue adapter exists only as an explicit unsupported stub.
- Capture entrypoints: `scripts/capture_polymarket.py`, `scripts/audit_polymarket_data.py`.
- Registry: `data/polymarket/registry/contracts.csv` (local; gitignored).
- Quotes: date-partitioned Parquet under `data/polymarket/quotes/` (gitignored).
- ESPN/DraftKings sportsbook odds remain the existing residual-model prior path; Polymarket quotes are a separate venue dataset.
- Modes: `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`.

## Phase 1 completed (verified)

1. **Venue interface** — `src/mlb_metrics/venues/base.py` with capabilities, fee schedule, market/book types; `order_placement` always false for US.
2. **Polymarket US adapter** — league MLB event discovery, full-game moneyline parse, order-book fetch, versioned taker fee schedules (0.06 now; 0.0695 from 2026-09-17T03:59Z).
3. **Orientation safety** — long/short outcomes retain team abbreviations; tests assert long is not assumed home (fixture: SD long / COL home).
4. **Contract registry** — `market_contracts.py` maps by home/away + start-time tolerance; statuses `mapped` / `unmatched` / `ambiguous` / `incomplete_teams`; preserves `first_observed_at_utc`.
5. **Quote store** — append-only Parquet + CSV index; idempotent on `raw_response_hash`; freshness helper uses `POLYMARKET_QUOTE_MAX_AGE_SECONDS=30`.
6. **Executable buy helper** — long uses best ask; short uses `1 - best_bid` (documented); display prices are not treated as fills.

### Live capture check (this session)

- `python scripts/capture_polymarket.py --limit 15`
- Discovered 15 moneylines; schedule rows available: 156.
- Registry: **3 mapped / 12 unmatched / 0 ambiguous** on that sample (many events outside current schedule-snapshot coverage or timing).
- Wrote books into `data/polymarket/quotes/` and coverage JSON under `reports/polymarket/` (both gitignored except keep/README).
- Modes remained shadow/disabled.

## Checks run

- `pytest tests/test_polymarket_phase1.py`: **9 passed**
- Related safety/residual workflow tests: **30 passed** together with Phase 1 tests
- Live read-only capture against Polymarket US gateway: succeeded
- Unverified: multi-day 95% mapping coverage target; GitHub Actions cron for Polymarket capture; international venue

## Remaining blockers

- Mapping coverage depends on fresh MLB schedule snapshots aligned to Polymarket start times; unmatched rate is still high on a first sample.
- Residual model still lacks enough sportsbook history / artifact; Polymarket history collection has just started.
- No paper ledger / decision UI yet (later phases).
- Owner bankroll and live-trading authorization still unknown; betting stays disabled.

## Next implementation task

Raise Polymarket↔`game_pk` mapping coverage (schedule freshness + doubleheader/postpone cases), optionally add a scheduled read-only capture workflow, then Phase 2 lineup/as-of baseball inputs while quotes accumulate.

## Sources

- [Polymarket US sports data](https://docs.polymarket.us/data-guide/sports-data)
- [US market book](https://docs.polymarket.us/api-reference/markets/get-market-book)
- [US fees](https://docs.polymarket.us/fees)
- [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md)
