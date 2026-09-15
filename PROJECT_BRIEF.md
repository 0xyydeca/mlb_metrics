# Project brief: mlb_metrics

## Purpose and owner decisions

- The owner uses this project for personal MLB bets on Polymarket and wants accurate picks that help win money. Profitability is a goal, not an established result.
- The owner delegates market selection to the agent based on evidence about wins, odds, costs, and risk.
- The owner requests a thorough implementation plan and autonomous progress without repeated approval questions for already authorized work.
- Personal bankroll, acceptable losses, and account/trade-execution authorization have not been supplied. Research and engineering can proceed with public data and paper results.

## Current plan

- Start with pregame full-game winner contracts because the repository already has game-win models, labels, schedule matching, and a market-residual framework. This is the selected engineering priority; superior profitability has not been demonstrated.
- Phase 0 (evidence foundation) is implemented on branch `codex/phase-0-evidence-foundation`. Next: Phase 1 — Polymarket venue adapter, contract registry, and quote capture.
- Prioritize MLB betting work. Preserve existing DFS, NFL, Beat the Streak, and age-curve features without treating them as current expansion priorities.
- Full work sequence, file-level backlog, acceptance criteria, evidence gates, and effort estimates: [MLB Polymarket development plan](MLB_POLYMARKET_PLAN.md).

## Evidence snapshot

- Planning audit: September 14, 2026, America/Phoenix, commit `6eb45e3d234416d54ddec5cf13ab6aa013c1be2a`.
- Phase 0 implementation: September 14–15, 2026, branch `codex/phase-0-evidence-foundation` (based on that commit).
- Repository: https://github.com/0xyydeca/mlb_metrics
- Public fork of JMerchen/mlb_metrics.

## Current technical facts

- Python processing: `src/mlb_metrics/`; entrypoints: `scripts/`; static HTML/CSS/JavaScript dashboard: `docs/`; dashboard data: `docs/data/`.
- Raw baseball data, prediction logs, and model artifacts live under `data/`.
- Current market ingestion uses ESPN sportsbook moneylines, with DraftKings preferred. Native Polymarket integration is not yet implemented.
- CI uses Python 3.12 and Node 22. Core Python dependencies are pinned in `requirements.txt` / `requirements-dev.txt` / `requirements-odds-capture.txt` to the locally verified versions.
- Default date convention is America/Phoenix.
- Current settings: `HITTER_SELECTION_MODE="shadow"`, `STREAK_POLICY_MODE="shadow"`, `GAME_PREDICTION_MODE="shadow"`, `BETTING_MODE="disabled"`, `LINEUP_API_SCHEMA_CONFIRMED=False`.

## Phase 0 completed (verified)

1. **Validation reports preserved**
   - `.gitignore` whitelists `game_residual_nested.json`, `game_residual_betting_gate.json`, and streak-policy report names.
   - `game_residual_training.yml` force-adds those JSONs, uploads `residual-validation-reports` artifacts (`if: always()`), and still commits when present.
   - Trainer writes insufficient-data reports when the pick log or market-aligned frame is empty.
   - Checked: `git check-ignore` reports the residual JSON paths as trackable; unrelated `foo.json` remains ignored.

2. **Training horizon + complete folds**
   - Default residual history horizon: `GAME_RESIDUAL_TRAINING_HISTORY_DATES = 120`.
   - Structural minimum: `GAME_RESIDUAL_MIN_DATES_FOR_COMPLETE_OUTER_FOLDS = 70`.
   - Residual nested validation requires complete outer/inner test blocks (`require_complete_test_blocks=True`).
   - Boundary tests: 60→2 folds, 61→2 folds (no partial trailing block), 70→3 folds with production freeze/block constants.
   - Holdout (`GAME_RESIDUAL_BETTING_FREEZE_DATES = 10`) unchanged.

3. **Honest market-only fallback labels**
   - Missing/failed residual artifacts export `probability_source="market_only_fallback"` with `fallback_used` / `fallback_reason`.
   - Independent residual label `market_residual` is reserved for loaded-artifact predictions.

4. **Closing vs prediction-time separation**
   - `_resolve_closing_market_probability` no longer fills missing closing with prediction-time prices.
   - `paired_market_scoring_differences` defaults to strict closing; `market_price_source="prediction_time"` is explicit for non-closing research.
   - Export `market_*` accuracy metrics use logged prediction-time prices; `beat_closing_*` requires true closing snapshots.
   - Closing window capped by `MARKET_ODDS_CLOSING_MAX_AGE_MINUTES = 180`.

5. **Distinct run/status visibility**
   - Report statuses: `insufficient_data`, `validated_failed`, `validated_passed`.
   - Reports include `artifact_saved` / `artifact_loaded`; trainer prints a `RUN OUTCOME` line.
   - `residual_training_status` / status script expose validation status and artifact flags.

6. **Reproducibility + write isolation**
   - Pinned requirements to the verified local environment (pandas/numpy/scikit-learn/joblib/etc.).
   - Autouse conftest redirects streak/lineup/audit/shadow/report paths away from production `data/predictions` and `reports/model_validation`.
   - Added residual artifact save/load/predict compatibility test.

## Checks run

- Focused Phase 0 suites: **87 passed**.
- Full `pytest`: **1007 passed**; 8 git-init tests failed only under the sandbox (no `git init`); re-run with full permissions: **20/20 passed** in the git/backtest modules.
- Node CI checks: `docs/dfs_solver.test.js`, `docs/nfl_dfs_solver.test.js`, `docs/nfl_draft_assistant.test.js` — **33/33 passed**.
- Modes remain `GAME_PREDICTION_MODE=shadow`, `BETTING_MODE=disabled`.
- Unverified here: end-to-end GitHub Actions residual-training run committing/uploading reports on `main` (workflow YAML and local force-add behavior verified; remote dispatch not executed).

## Remaining blockers (unchanged by Phase 0)

- Only ~11 valid sportsbook prediction dates / ~147 morning-aligned games; zero complete residual outer folds possible today.
- No residual model artifact; shadow rows are market fallbacks until a gate-passing train saves one.
- No Polymarket venue adapter, contract registry, fee model, or executable quote store.
- Confirmed lineup API still gated (`LINEUP_API_SCHEMA_CONFIRMED=False`).
- Odds-capture freshness can still lag under Actions scheduling; health workflow may fail when checkout data is stale.
- Owner venue/account, bankroll, and live-trading authorization remain unknown; betting stays disabled.

## Next implementation task

**Phase 1 item 5–6 from the plan:** selected-venue Polymarket adapter + contract registry, then durable quote recorder/health, using public read-only APIs only. Do not enable live betting.

## Sources

- Owner statements establish purpose, delegated market prioritization, and autonomous Phase 0 implementation.
- [MLB_POLYMARKET_PLAN.md](MLB_POLYMARKET_PLAN.md)
- [US API](https://docs.polymarket.us/api-reference/introduction), [US price history](https://docs.polymarket.us/api-reference/price-history/get-price-history), [US order books](https://docs.polymarket.us/api-reference/markets/get-market-book), [US fees](https://docs.polymarket.us/fees), [US sports settlement](https://docs.polymarket.us/faqs/sports-faqs).
- [International market data](https://docs.polymarket.com/market-data/overview), [international fees](https://docs.polymarket.com/trading/fees).
