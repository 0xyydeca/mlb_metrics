# MLB Polymarket development plan

**Purpose:** Build a reliable personal decision tool that estimates MLB outcomes, compares those estimates with prices the owner can actually obtain, and identifies opportunities only when evidence supports them after costs and risk.

**Planning decision:** Start with pregame, full-game winner contracts. Reuse the existing baseball pipeline and market-residual model framework. First repair the evidence pipeline, then add Polymarket data, then test whether an edge exists. Expansion to other markets follows measured results.

**Status:** Evidence review and implementation plan completed. Production fixes, Polymarket integration, and forward validation described below are not yet implemented. No demonstrated Polymarket profitability has been established.

**Evidence snapshot:** Repository `0xyydeca/mlb_metrics`, commit `6eb45e3d234416d54ddec5cf13ab6aa013c1be2a`, reviewed September 14, 2026, America/Phoenix. Workflow evidence includes an earlier same-day commit and is identified separately. Local materials were the owner's AGENTS.md and PROJECT_BRIEF.md. Their instructions were treated as project context; the owner's current request governs this plan.

## 1. Decisions made

### Initial market: pregame game winners

The current code already predicts game winners, records final scores, matches games by MLB identifiers, and implements a model that learns adjustments to market probabilities. That makes game-winner markets the shortest path to a valid end-to-end comparison. This is an engineering priority, not a claim that they offer the highest profits.

Use one scheduled decision near first pitch, initially **30 minutes before scheduled start**, with a final data/price refresh before displaying an actionable signal. Record earlier snapshots for research. The 30-minute timing is a design starting point to validate, not an optimized result. A delayed start, pitcher change, lineup change, or stale quote invalidates the recommendation until recomputed.

Evaluate candidate markets in this order:

1. **Full-game winners:** first implementation and primary benchmark.
2. **Totals and run lines:** only after a run-distribution model, push/settlement handling, and historical price coverage exist. A win-probability model does not supply these probabilities.
3. **Player outcomes:** only when actual venue listings, participation rules, and reliable lineup inputs match a modeled outcome. Beat the Streak's zero-at-bat semantics must not be copied into a contract with different settlement rules.
4. **In-play contracts:** later, because the current daily pipeline does not provide the event latency, game-state models, or execution records required.
5. **Futures:** later, because season simulation and long holding periods require a different validation setup.

Keep DFS, NFL, and age-curve functionality working, but direct new research effort toward the first market. Do not delete those features as a prerequisite.

### Objective: better decisions at available prices

Use net expected value and uncertainty to rank eligible opportunities. Track accuracy, but do not maximize win rate in isolation.

For an ordinary contract that pays $1 for a win and $0 otherwise, buying one share at price `c` has estimated pre-cost expected profit `p − c`, where `p` is the estimated probability. Subtract fees and execution costs. For example, a 60% chance bought for $0.70 loses an expected $0.10 per share before costs despite winning most of the time. This is an illustration, not a prediction about a particular game.

For contracts with cancellation or intermediate settlement values, use the expected settlement payout across those states instead of this binary shortcut. The general calculation is **expected payout minus total acquisition cost**.

### Venue must be explicit

Polymarket US and the international venue have different APIs, market identifiers, execution mechanics, and fees. The user's intended account/venue has not been verified. Build a small provider interface and identify the venue from actual market URLs and metadata; do not infer it from the computer's timezone. US public discovery is available without trading credentials, as is international public market data. [US API](https://docs.polymarket.us/api-reference/introduction), [international market data](https://docs.polymarket.com/market-data/overview).

Research can proceed using public data. The selected venue must be attached to every observation and result; a profitable result on one venue cannot establish profitability on the other.

## 2. What the evidence shows now

### A. The project has a substantial foundation

There are dedicated modules for metrics, predictions, schedule snapshots, odds capture, chronological model validation, model artifacts, and betting-readiness checks. The September 14 residual-training workflow reports **1,007 tests passed** on commit `1680c1845ac1564e643f23f310e8948665c2b029`. That is useful software evidence, not evidence of profitable bets. [Training run](https://github.com/0xyydeca/mlb_metrics/actions/runs/34883973780).

### B. Current odds are sportsbook odds

The inspected `market_odds.py` obtains ESPN moneylines, with DraftKings as the preferred provider. Its implied probabilities and sportsbook settlement math are not executable Polymarket orders. A Polymarket adapter, contract crosswalk, fee model, and execution-aware ledger are required. [Odds implementation](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/src/mlb_metrics/market_odds.py).

### C. The usable market history is much smaller than the file size suggests

The committed odds file contains **946 rows**, covering September 3–14. Of these, 473 are marked post-start. Restricting to valid, matched, pre-start opening/morning/lineup-lock records leaves **147 games across 11 dates**, all morning snapshots. Other pregame intraday rows can support appropriately timestamped research, but must not be relabeled as earlier observations. [Snapshot file](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/data/predictions/market_odds_snapshots.csv).

The earlier training run independently reports 147 valid market games, 11 dates, **110 aligned training rows**, zero possible outer folds, and no saved residual artifact. Its inventory had 916 rows before subsequent captures. These are different snapshots, not inconsistent counts. [Training run](https://github.com/0xyydeca/mlb_metrics/actions/runs/34883973780).

### D. A successful training workflow currently means insufficient evidence

The run produced a report but did not save a model. It finished successfully because insufficient data is an allowed research outcome. The current shadow file has **147 rows with empty artifact IDs**, and every residual probability equals the market probability. The inference function explicitly falls back to the market when the artifact is absent. The exporter nevertheless stamps `probability_source="market_residual"`; that needs a clearer fallback label. [Shadow records](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/data/predictions/game_residual_shadow_predictions.csv), [model and exporter](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/src/mlb_metrics/game_residual_model.py).

### E. Validation reports are being lost between runs

`.gitignore` excludes `reports/model_validation/**`, except README.md and .gitkeep. The training workflow stages that directory without overriding the exclusions. Its log says a JSON report was written, followed by no staged changes; the inspected repository contains no report. This is a concrete persistence defect, separate from whether the model passes validation. [Ignore rules](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/.gitignore), [workflow](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/.github/workflows/game_residual_training.yml).

### F. The default training horizon cannot meet its own fold gate

The training command defaults to the latest 60 game dates. Validation reserves 10 dates, starts with 30 training dates, and then uses 10-date test blocks. With 60 dates this allows only **two outer folds**, while promotion requires at least three. I reproduced the boundary behavior using the repository's fold-building function:

- 11 valid dates: zero outer folds.
- 60 valid dates: two 10-date outer test blocks.
- 61 valid dates: three blocks, but the last contains only one date.
- 70 valid dates: three complete 10-date blocks.

Raise the available-history horizon and explicitly require complete evaluation blocks. Do not reduce the holdout or minimum evidence simply to obtain a passing badge. Seventy dates is a structural minimum for this configuration, not a statistical guarantee. [Trainer](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/scripts/train_game_residual_model.py), [fold construction](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/src/mlb_metrics/model_validation.py), [configuration](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/src/mlb_metrics/config.py).

### G. Intraday freshness is not reliable enough

The odds-health workflow failed at 23:05 UTC on September 14 because its checked-out data was **195 minutes old**, exceeding its 90-minute threshold. A capture run starting around the same time does not make the earlier stale period disappear. Root causes may include scheduling delays and readers using older committed snapshots; the log establishes staleness, not a complete root-cause diagnosis. [Failed health run](https://github.com/0xyydeca/mlb_metrics/actions/runs/34907180483).

### H. Confirmed lineups are implemented but gated off

`LINEUP_API_SCHEMA_CONFIRMED=False`; the lineup adapter returns empty while this is false. The 53 committed lineup-lock run records contain 39 `no_changes` and 14 `no_games_in_window` statuses, with no recorded refresh. The plan must verify actual API parsing and ensure lineup information reaches the game-win model, not just hitter inference. [Lineup adapter](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/src/mlb_metrics/lineup_snapshots.py), [run records](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/data/predictions/lineup_lock_runs.csv).

### I. Existing accuracy is descriptive, not a trading result

The game log contains 752 rows and 740 resolved rows: 403 correct winners, or **54.46%**, with a Brier score of **0.2496**. The v1 subset is 359/663, or 54.15%; 77 resolved rows are labeled legacy. These figures mix dates and historical development conditions and are not an untouched evaluation of a current model. They do not include Polymarket fills or costs, and no Polymarket return can be inferred from them. One game ID appears on two dates, one pending and one resolved; investigate rescheduling semantics without deleting history. [Game log](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/data/predictions/game_predictions.csv).

### J. Closing-price labels need stricter provenance

`game_evaluation._resolve_closing_market_probability` falls back to a logged prediction-time probability when no closing snapshot exists. This can mix morning and closing comparisons under a closing label. Require a distinct provenance flag and report true closing comparisons separately. Also cap how far before start a quote can be and still qualify as a closing observation. [Evaluation implementation](https://github.com/0xyydeca/mlb_metrics/blob/6eb45e3d234416d54ddec5cf13ab6aa013c1be2a/src/mlb_metrics/game_evaluation.py).

## 3. Target system

Use the current Python package and static dashboard. Add small modules around well-defined data boundaries rather than rewrite the project.

The flow is:

1. MLB schedule, historical baseball data, starter announcements, confirmed lineups, and available weather forecasts.
2. Venue-specific market discovery, contract rules, fee metadata, and timestamped order books.
3. Exact game/contract mapping and immutable observations.
4. Features assembled as they were knowable at the decision time.
5. Baseline probability and a validated baseball adjustment.
6. Expected payout, executable cost, uncertainty, and exposure checks.
7. A dated recommendation or an explicit reason to pass.
8. A paper ledger followed by venue settlement and performance attribution.

### Proposed additions

These names describe planned files, not existing capabilities:

- `src/mlb_metrics/venues/base.py`: common market, quote, cost, and settlement interface.
- `src/mlb_metrics/venues/polymarket_us.py` and `polymarket_international.py`: venue-specific parsing. Implement the selected venue first; keep the other an explicit unsupported capability until tested.
- `src/mlb_metrics/market_contracts.py`: MLB game-to-contract mapping, rule hashes, outcome orientation, and reschedule handling.
- `src/mlb_metrics/quote_store.py`: append-only quote observations and collector health.
- `src/mlb_metrics/polymarket_execution.py`: cost estimation and hypothetical fills; no live-order dependency in the research path.
- `src/mlb_metrics/decision_policy.py`: pure eligibility and price-limit decisions shared by backtest and dashboard.
- `src/mlb_metrics/paper_ledger.py`: immutable decisions, hypothetical fills, settlements, and cash-flow accounting.
- `scripts/capture_polymarket.py`, `scripts/audit_polymarket_data.py`, `scripts/run_polymarket_backtest.py`, and `scripts/build_polymarket_dashboard.py`: small operational entrypoints.

Reuse `game_residual_model.py`, `model_validation.py`, `schedule_snapshots.py`, `game_picks.py`, and baseball feature modules. Adapt their inputs and evaluation where the venue requires different semantics.

### Data contracts

Every table has a schema version, UTC timestamps, source identity, and uniqueness checks.

- **Market registry:** venue, event ID, contract ID, outcome/instrument ID, `game_pk`, teams, market type, line if applicable, scheduled start, trading status, settlement rule text/hash, mapping status, and mapping evidence. Never assume Yes always means the home team.
- **Quote observations:** source and receive timestamps, request time, contract/outcome ID, bids and asks with sizes, fee configuration, tick/lot sizes, suspension status, and raw-response hash. Preserve observations that are rejected; exclude them through explicit eligibility fields.
- **Baseball observations:** event time, first observed time, scheduled/actual start status, announced starters, lineup state and order, source, and revision lineage.
- **Prediction record:** decision ID, contract ID, game ID, as-of time, feature cutoff, data manifest hash, model artifact/version, baseline probability, model probability, calibration version, fallback state, uncertainty method, and intended evaluation cohort.
- **Decision record:** eligibility, reason codes, observed executable price, maximum acceptable price, assumed quantity, expected cost, edge estimate, stress results, and policy version.
- **Paper ledger:** requested and filled quantities, fill assumptions and times, costs, payout, settlement status/revisions, and realized net P&L. Report open positions separately from settled returns.

Raw quote data should use date-partitioned Parquet plus a small transactional local store/index for the initial collector. Export compact dashboard data and evidence manifests. Preserve current historical files; do not move or delete them without a checked migration. Avoid committing every order-book refresh to Git. Keep personal bankroll and any future account records outside public dashboard exports.

## 4. Implementation sequence and acceptance criteria

### Phase 0 — Repair evidence and reproducibility

**Estimated engineering effort: 2–3 focused days.** No market-data waiting is required for these fixes.

1. Whitelist intended report JSON files in `.gitignore`; retain failed and insufficient-data reports. Upload reports as workflow artifacts even if later steps fail. Verify retrieval from a fresh checkout/artifact download.
2. Replace the implicit 60-date training cap with an explicit coverage-aware horizon. Expose date range and eligible-date counts in every run. Preflight the number and size of possible folds before expensive feature reconstruction.
3. Require complete outer test blocks for promotion; add boundary tests for 60, 61, and 70 valid dates with a 10-date freeze. Persist exact fold membership and an immutable final evaluation period.
4. Make status explicit: `insufficient_data`, `validated_failed`, `validated_passed`, `artifact_saved`, and `artifact_loaded`. A completed job must not imply readiness.
5. Label missing-artifact output as `market_only_fallback`, retain the actual fallback reason, and prevent fallback output from being counted as independent model performance.
6. Remove morning-as-closing substitution from strict reports; retain any compatibility report under its correct label.
7. Pin the tested dependency environment, including model serialization versions. Add one bounded artifact load/predict compatibility check.
8. Detect unintended writes during tests and research. The observed training job ended with a modified streak-policy CSV; identify which step wrote it and route test/research output to isolated directories.

**Done when:** a clean run with insufficient data leaves a retrievable report with accurate status; a fresh run can load that report; fold counts match tests; missing artifacts cannot produce a misleading model label; research does not alter production predictions; all relevant tests pass.

### Phase 1 — Establish Polymarket contract and price coverage

**Estimated engineering effort: 3–5 days. Start recording as early as possible.**

1. Discover current and historical MLB markets through official public APIs. Record venue identity and supported capabilities. Confirm market rules, not just title similarity.
2. Match contracts to MLB IDs using teams, scheduled time, game number, and provider IDs. Quarantine ambiguous matches and verify postponed, resumed, canceled, and doubleheader examples.
3. Capture both outcome costs and order-book depth. A displayed probability or last trade is not a fill quote. US price history contains book-derived display prices without order size, and Yes/No prices can sum above one because of the spread. Preserve those meanings. [US history specification](https://docs.polymarket.us/api-reference/price-history/get-price-history), [US order book](https://docs.polymarket.us/api-reference/markets/get-market-book).
4. Read market-specific fees and tick/lot limits. Version fee assumptions by venue and effective time. Do not hardcode one permanent sports fee or count promotional rebates after expiration. [US fees](https://docs.polymarket.us/fees), [international fees](https://docs.polymarket.com/trading/fees).
5. Keep book snapshots, historical display prices, trade prints, and sportsbook benchmarks as distinct data classes. Historical backfill without size supports exploratory price analysis, not a verified fill backtest.
6. Track first observation time as well as source event time so revised historical responses cannot be mistaken for information available earlier.

**Initial operational targets, selected for engineering validation:** cover at least 95% of eligible listed game-winner contracts over seven consecutive game days; every emitted decision has an unambiguous mapping and complete cost fields. Capture near-start prices at least every minute where supported, with a fresh public quote check for any actionable display. The quote-age limit begins at 30 seconds; stale or missing quotes suppress the action. These are service targets, not predictions of market liquidity.

**Done when:** sampled ordinary and exceptional games map correctly; 100% of emitted decisions trace to a saved quote and rules version; replaying capture is idempotent; missing, suspended, stale, or ambiguous inputs produce a reasoned pass.

### Phase 2 — Make baseball inputs trustworthy at decision time

**Estimated engineering effort: 3–5 days, overlapping price collection.**

1. Inspect real MLB API lineup responses for scheduled, confirmed, changed, scratched, and started games. Implement parsing against recorded responses; only then enable the schema flag.
2. Carry lineup strength and starter certainty into the game-win feature frame. Test both sides and absent inputs. A hitter-only overlay is insufficient for a game-winner decision.
3. Assemble pitcher, bullpen, offense, defense, park, and rest features from data available before the decision. Compare a small feature set first. Add weather only with forecast issue timestamps and historical forecast availability.
4. Treat missing required starters/lineups as a separate unqualified policy state initially. An unconfirmed-lineup strategy can be tested later as its own cohort.
5. Reconcile schedule-time cutoffs with actual prediction timestamps. Existing fixed morning cutoffs must not attach a later announcement to an earlier decision.
6. Support doubleheaders without merging games; exclude game-one results from a game-two prediction made before those results were available. Preserve postponed-game identity and resolution lineage.
7. Generate a coverage waterfall: all games → mapped contracts → usable prices → usable baseball snapshots → complete features → resolved outcomes. Give a reason for every removed row.

**Done when:** a future result or revised lineup cannot change a historical replay; missing inputs remain visible; source time and observation time both satisfy the as-of join; repeated replay gives the same feature manifest and predictions.

### Phase 3 — Build the Polymarket replay and paper ledger

**Estimated engineering effort: 3–5 days. Depends on Phases 1–2 schemas.**

1. Replay in timestamp order with a single shared decision policy. Log every candidate and every no-bet decision, not only eventual winners or selected bets.
2. Start with immediately executable, size-limited hypothetical purchases and hold to settlement. Use the observed ask/depth and actual venue fee semantics. Do not assume limit orders fill because price merely touched a level.
3. Stress execution with worse quotes, added delay, reduced available size, and rejected/partial fills. Use one, two, and five ticks of adverse movement and 30/60/120-second delay scenarios as test cases, not assumed real slippage.
4. Handle the exact venue settlement price, including non-binary outcomes when rules require them. Never automatically treat canceled events as $0 losses or refunds without checking the contract. US MLB settlement explicitly addresses official results and independent doubleheader contracts. [US sports rules](https://docs.polymarket.us/faqs/sports-faqs), [settlement endpoint](https://docs.polymarket.us/api-reference/markets/get-market-settlement).
5. Reconcile cash flows: acquisition cost + fees, settlement or exit proceeds, and remaining open exposure. Define ROI as settled net profit divided by settled acquisition costs; also show total committed capital and unresolved exposure.
6. Compare simple one-share results with equal-risk paper positions. Keep sizing performance separate from probability quality. Use the same valid opportunity set for all models.

**Done when:** hand-calculated examples reconcile; outcome orientation, partial fills, fee rounding, cancellation, and rescheduling tests pass; duplicate decisions do not create duplicate exposure; optimistic and conservative execution assumptions are clearly separated.

### Phase 4 — Determine whether the model adds information

**Estimated engineering effort: 4–7 days once usable history exists. Evidence accumulation may take much longer.**

Compare these candidates on identical games and decision times:

1. A documented market-probability baseline derived from the selected venue's book, with crossed/stale books excluded. Keep it separate from the executable ask used for costs.
2. The existing heuristic model, calibrated using training data only.
3. A regularized market-residual logistic model: market probability plus a learned baseball adjustment.
4. A small nonlinear residual challenger only after the linear comparison is reproducible.

The existing residual formulation is a useful starting point: `logit(final probability) = logit(market probability) + baseball adjustment`. It asks whether baseball information improves the current market estimate. Use the same prediction-time market baseline during training and serving; replacing sportsbook priors with Polymarket priors requires revalidation.

Research protocol:

- Register the hypothesis, candidate features, model family, calibration, entry timing, and threshold search before opening evaluation outcomes.
- Use expanding chronological training and inner folds for all model and strategy choices. Outer folds estimate performance after those choices. Treat market selection and execution timing as tuning too.
- Retain at least three complete outer blocks plus a separate final freeze period. Begin with the repository's 30/10/10 structure, then perform a precision analysis before enlarging claims. Smaller sample counts must not be sold as statistically sufficient.
- Fit preprocessing, calibration, thresholds, and missing-value policies inside training only. Avoid repeatedly revisiting an outer/final holdout to make the model look better.
- Keep a registry of which dates were inspected. Data reviewed during this audit is exploratory; it is not newly untouched validation evidence.
- Evaluate every market-covered game for probability quality and the predeclared decision subset for trading results. Show selection coverage so a high score on a tiny subset is visible.
- Report Brier score and log loss (probability error), calibration by probability band, and paired differences against the market. Bootstrap whole game days and assess multi-day blocks as a sensitivity check.
- Report net ROI, profit per settled position, worst drawdown, loss streaks, fill rate, costs, stale-price rejection rate, and concentration by week, team, probability band, and time to start.
- Use closing-line movement as a supporting diagnostic with quote-age and contract provenance. It is not a substitute for net returns or calibration.
- Test each feature family by adding/removing it. Retain new complexity only when its gains survive untouched evaluation and plausible cost stress.

**Done when:** a saved report identifies the selected model, exact data/fold hashes, sample sizes, uncertainty, costs, comparison results, and a pass/fail/insufficient conclusion. A no-edge result is a valid research result and leaves betting disabled.

### Phase 5 — Build a useful personal dashboard

**Estimated engineering effort: 2–4 days; can proceed with clearly labeled paper data.**

The primary screen should answer: **Which contract, at what maximum price, why, and how fresh is the information?**

Each opportunity shows the game, start time, venue and contract link, side, rules summary, current executable cost, model probability, maximum acceptable price after costs, expected-value estimate, uncertainty, lineup/starter status, quote age, and paper/live status. Display the quoted quantity so an attractive price is not presented as available for unlimited size.

Use plain statuses: `Paper candidate`, `Pass: price too high`, `Pass: stale quote`, `Pass: missing lineup`, `Pass: insufficient evidence`, and `Game started`. A high win probability alone must never earn a “recommended” label.

Add an evidence view with dataset coverage, model/version, validation status, realistic paper performance, open exposure, and reasons trades were skipped. Separate old model history, current model results, sportsbook research, and venue-specific paper results.

Serve the first version locally. Static published data can display analysis, but near-start recommendations require a fresh quote check; a once-daily CSV must not look like a live tradable price. Keep personal account data private.

**Done when:** the owner can trace any candidate to its saved quote, prediction, and contract; stale data automatically removes actionability; all charts label the model/cohort/time period; empty evidence never appears as a zero-risk or passing result.

### Phase 6 — Forward validation and operations

**Engineering effort: 1–3 days for instrumentation; then observation on actual game days.**

Run a frozen paper policy through a predeclared prospective period. Log decisions before outcomes; account for skipped, unfilled, open, and settled positions. Daily operational checks can detect outages without reopening model-selection decisions. Evaluate performance at registered checkpoints rather than stopping at the first profitable streak.

Retain GitHub Actions for tests and batch training. For near-start collection, use a restartable collector with durable storage and a heartbeat. The initial local service requires the machine to stay awake; it suppresses stale recommendations after sleep or connection loss. A hosted always-on instance is a later deployment choice with explicit operating cost. Existing public/free endpoints are the initial data budget.

Monitor per contract: quote age, baseball freshness, missing mappings, missing artifacts, failed requests, clock skew, storage writes, and settlement lag. A global “latest row is fresh” check cannot establish that every game's price is fresh.

**Done when:** restarting and reconnecting preserve exactly-once decision IDs; forced outages result in explicit passes; the ledger reconciles; a fresh deployment can load the exact validated artifact; a rollback restores a known baseline while keeping audit history.

## 5. Evidence gates

These are proposed project criteria selected before new performance testing. They are not statements that the current system passes or that passing guarantees future profit.

### Gate A — Data and accounting

- Every eligible row has an exact contract mapping, correct outcome orientation, valid cutoff, fee version, and traceable quote.
- No post-start observations or future announcements enter a pregame feature set.
- Every paper position reconciles to acquisition costs and actual settlement semantics.
- Collector coverage reaches the initial seven-day operational target; all detected gaps remain excluded and reported.

### Gate B — Model evidence

- At least three full chronological outer test blocks and a separately registered final evaluation period.
- Point estimates improve on the same-time market for both Brier score and log loss. Register log loss as the primary metric; its paired day-block 95% interval should exclude zero improvement before claiming demonstrated predictive superiority.
- Calibration and high-confidence errors are reported with sample sizes. Require enough precision relative to the intended pricing edge, rather than choose an arbitrary high win-rate target.
- Reproducibility and feature availability pass on a fresh environment. The saved model, report, feature schema, training cutoff, and policy hash agree.

### Gate C — Strategy evidence

- Positive net return under realistic recorded prices/costs on untouched evaluation, with a day-block 95% lower bound above zero before claiming demonstrated profitability.
- Results remain positive under the registered conservative cost scenario and are not explained by one lucky week. Report the result after excluding the best week; preserve the existing concentration check as an additional constraint.
- Sufficient numbers of bets and independent dates are set through a precision/power assessment. The current 30-bet/100-game policy minimums are floors, not proof of a small edge.
- Closing-price evidence is genuinely closing, appropriately fresh, and venue-specific. Missing closing observations are missing evidence.
- Prospective paper performance meets the registered protocol. Extend observation when inconclusive; do not tune on the same period and still call it forward validation.

### Gate D — Practical use

- A current recommendation has fresh executable cost and sufficient quantity, passes all required input checks, and remains below its maximum acceptable price.
- A tested per-game/portfolio exposure policy and loss budget exist. The owner's personal bankroll and loss tolerance are unknown; analysis remains normalized and paper-only until those values are provided through ordinary project configuration.
- Research/market selection authority is not account or trade-execution authority. This plan introduces no wallet actions, funds transfers, or automated order placement.

“No bet” is the required output whenever a gate fails. The system can still be operationally useful by showing honest forecasts and explaining why an edge is unproven.

## 6. Risk and sizing research

Separate three questions: whether the prediction is useful, whether the price is favorable, and how much exposure a hypothetical portfolio can sustain.

Start with unlevered paper positions and compare one-share and equal-risk policies. Later test fractional Kelly using calibrated probabilities and conservative uncertainty adjustments. Do not let aggressive sizing manufacture an attractive return chart.

Use one net directional exposure per game initially. Group correlated positions by game, team, date, and shared drivers such as weather. Do not treat opposite sides or related player outcomes as independent opportunities.

Stress a normalized paper bankroll across a small predeclared range of per-game and total-exposure limits. Choose the research policy using training/simulation only; it is not the owner's approved personal risk budget. Include price gaps, liquidity loss, probability miscalibration, and clusters of losses. Report both settled-equity drawdown and total open capital at risk.

Do not increase exposure to recover losses. If the model degrades or inputs fail, revert to paper/market-baseline reporting and investigate the cause using the existing audit trail.

## 7. Engineering checks that matter

Retain the current Python and JavaScript suites. Add focused tests for the new behavior instead of duplicating implementation details:

- Report JSON survives version control/artifact publishing, including insufficient-data reports.
- Fold boundary arithmetic, complete-block requirements, fixed freeze IDs, and no train/test overlap.
- Missing model and prediction-error fallbacks have correct provenance and cannot pass as independent residual evidence.
- Venue outcome inversion, fees, rounding, lot sizes, depth-weighted cost, partial fills, and zero fills.
- Doubleheader mapping, postponed/resumed games, changed pitchers, scratched hitters, stale lineups, and start-time changes.
- Future-data perturbation leaves earlier features and decisions unchanged.
- Repeated capture or replay does not duplicate snapshots/decisions/positions.
- Stale quotes, missing cost data, suspended contracts, and clock skew suppress recommendations.
- Settlement at $0, $1, and intermediate values reconciles ledger balances; settlement corrections preserve history.
- Dashboard labels correctly distinguish paper, legacy, fallback, pending, and validated results.

Run read-only public API contract checks separately from deterministic tests. Save representative responses with timestamps; do not require changing live games for unit tests. Before each implementation merge, run relevant regressions and the repository's applicable CI checks. Model changes also require the corresponding data/validation report.

## 8. Prioritized delivery backlog

Each item should be a small reviewable change with an acceptance result and rollback path.

1. **P0: Retain validation evidence.** `.gitignore`, residual-training workflow, report retention check. Depends on nothing. Rollback affects report publishing only.
2. **P0: Fix training horizon and fold completeness.** Trainer, configuration, validation tests. Depends on nothing. Keep model modes unchanged.
3. **P0: Honest fallback and closing provenance.** Residual export, game evaluation, dashboard labels. Depends on nothing. Preserve historical rows with legacy provenance labels.
4. **P0: Baseline reproducibility and write isolation.** Dependency lock, read-only research checks, test output locations. Resolve the observed unexpected CSV write.
5. **P1: Contract registry and selected-venue adapter.** New venue/mapping modules and fixture tests. Depends on interface/schema decision; supports public read-only data.
6. **P1: Durable quote recorder and health.** Collector, store, per-contract freshness and recovery tests. Depends on item 5. Begin collection immediately after this works.
7. **P1: Real lineup parsing and as-of game features.** Existing lineup/schedule modules, game feature path. Can proceed alongside item 6.
8. **P1: Venue cost and settlement ledger.** Execution simulator and ledger. Depends on items 5–6; includes no order submission.
9. **P1: Frozen replay and market-residual comparison.** Shared decision policy, validation reports and cohort definitions. Depends on items 2, 5–8 and adequate data.
10. **P2: Personal dashboard and traceability.** Price limit, freshness, evidence view, paper results. Depends on registry/ledger schemas; UI work can use recorded paper fixtures.
11. **P2: Prospective paper evaluation and operations.** Frozen policy, incident/restart drill, observed performance. Depends on item 9; calendar duration depends on sample precision and game availability.
12. **P3: Evidence-led expansion.** Totals/run lines, then selected player markets. Requires an independent target/settlement model and the same gates; do not reuse a game-winner pass as approval for every market.

## 9. Delivery timing and resource plan

The engineering ranges above imply roughly **3–6 working weeks** for a robust first paper system, depending on API/data issues and how much can be reused. This is a planning estimate, not a completion promise. Existing test coverage and residual-model structure reduce rebuild work.

Suggested order:

- **First 2–3 days:** correct report retention, horizon, fallback labels, closing labels, and reproducibility.
- **First week:** establish the selected venue's registry and begin valid quote collection; produce a coverage report.
- **Weeks 2–3:** complete lineup inputs, replay, ledger, and first operational dashboard.
- **Weeks 3–6:** run model comparisons when the data gate permits; harden collection and the frozen paper policy.
- **Thereafter:** forward observation until registered evidence thresholds are met. A fast implementation cannot manufacture historical order books or additional independent games.

At present, there are only 11 valid sportsbook prediction dates and no collected Polymarket book history in the inspected files. The existing fold setup needs 70 valid dates for three complete blocks plus its holdout. That is at least 59 additional valid dates if relying only on those 11, before accounting for market mismatch or missing features. Do not promise a current-season validation finish; assess historical vendor coverage and season/postseason differences before using backfill.

Start with public endpoints and existing compute; do not assume a paid data subscription or new hosting budget. Measure request volume, storage growth, collection uptime, and reconstruction runtime in Phase 1. Evaluate paid historical data only after demonstrating that its timestamp, contract, and depth coverage would fill a material gap. Do not buy a large dataset merely to increase row counts.

## 10. Decisions that remain evidence-dependent

These are not blockers to the first engineering tasks and do not require repeated approval questions:

- Which Polymarket venue and exact contracts correspond to the owner's intended use: resolve from actual market URLs/metadata when available.
- Whether game winners offer any net edge: answer with matched-time validation and prospective records.
- Whether weather, lineup strength, or bullpen information adds value beyond the market: answer with feature comparisons and availability audits.
- Whether historical price data is sufficient for executable backtests: inspect depth/size coverage and timestamp semantics; otherwise limit claims to price research and collect forward books.
- Whether an always-on host is needed: measure the local collector's uptime and fail-closed behavior; separate engineering readiness from a hosting purchase.
- Personal bankroll and acceptable losses: retain as unknowns; they do not prevent research or paper validation.

## 11. Deliverables and definition of success

The finished project should provide:

1. Versioned, sourced MLB/contract/quote datasets with coverage and leakage checks.
2. Reproducible probability comparisons against a correctly timed market baseline.
3. A contract-specific simulator and forward paper ledger with realistic costs.
4. A concise personal dashboard with contract link, current cost, maximum price, evidence, and expiry.
5. Separate data, model, strategy, and operational readiness reports.
6. Recovery procedures and an audit trail that survive outages and model changes.

Success means the system can demonstrate when its recommendations are supported and when to pass, with measured performance at actual venue prices. Reliable software can be built; a profitable edge must be earned by the evidence and may not exist for a tested strategy.

## 12. Work completed for this plan

- Read both owner-provided project documents.
- Inspected repository structure, current configuration, odds and schedule handling, lineup gate, model validation, training workflow, prediction/evaluation logic, and relevant tracked CSVs.
- Read the latest 15 workflow summaries and detailed logs for the recent residual-training run and a failed odds-health run.
- Computed row/date/coverage and descriptive game-pick counts from the five inspected CSVs.
- Reproduced the validation fold boundary issue using the repository's fold function in isolation.
- Checked current official Polymarket documentation for US/international data access, price history, fees, order books, and settlement.

This was an evidence audit and planning task. No full local pipeline or training run was performed, no code fix was merged, and no trade was made. The observed GitHub test run is historical evidence from its stated commit, not a claim of newly passing tests after changes.
