# Sport adapter migration notes

## Goal

Reuse verified Polymarket venue/quote/paper infrastructure across sports **without**
weakening MLB correctness or rewriting the stack as a framework.

## What is shared (do not fork)

| Component | Location | Notes |
|---|---|---|
| Venue discovery / books / fees | `venues/base.py`, `venues/polymarket_us.py` | Use `list_moneyline_markets(league=...)`; MLB wrapper retained |
| Quotes, depth, freshness, storage | `quote_store.py` | Pass a sport-specific `store_dir` |
| Paper fills, fees, exposure, settle math | `paper_ledger.py` | Sport-agnostic taker accounting |
| Decision states / readiness patterns | `decision_board.py` (MLB UI today) | Reuse patterns; do not force NFL into MLB board CSV |

## What stays sport-specific

| Concern | MLB | NFL (first expansion) |
|---|---|---|
| Native game identity | `game_pk` (int) | `sport_game_key` = `nfl:{game_id}` |
| Player identity | `key_mlbam` | NFL ids via existing nfl pipelines — not `key_mlbam` |
| Schedule / results | StatsAPI + `schedule.py` | `sports/nfl.py` + `nfl_data.py` |
| Registry path | `data/polymarket/registry/` | `data/polymarket/nfl/registry/` |
| Quote path | `data/polymarket/quotes/` | `data/polymarket/nfl/quotes/` |
| Research protocol | `polymarket_us_game_winner_v2` | Separate protocol required before NFL nested eval |

## Compatibility rules

1. MLB CSV column names and paths remain stable for existing consumers.
2. Never coerce NFL `game_id` into `game_pk`.
3. Operational collector fixes ≠ evaluation-version changes.
4. Replaying saved MLB fixtures must produce equivalent fills/mapping (see `tests/test_sport_adapters.py`).

## Explicit non-goals (this migration)

- No multi-sport dashboard rewrite.
- No NFL residual model or real-money enablement.
- No purchase of third-party data feeds.
