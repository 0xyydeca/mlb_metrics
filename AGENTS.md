# Working instructions

Applies to all work in this project.

## Shared working preferences

- Read PROJECT_BRIEF.md and the files relevant to the task before making changes.
- Keep instructions concise, concrete, and testable. Put changing facts, priorities, constraints, and open questions in PROJECT_BRIEF.md; do not duplicate them here.
- Distinguish verified facts, user decisions, proposals, and unknowns. Never invent project details, results, or preferences.
- Make routine research and implementation decisions autonomously within the requested scope. Record unresolved inputs and continue independent work; do not repeatedly ask for approval of already authorized work.
- In the completion summary, state what changed, what was checked, the results, and any unverified behavior.

## Project rules

- Check implementation and tests when README claims conflict; record unresolved discrepancies in the brief.
- Keep metric windows, weights, thresholds, and model-version constants in src/mlb_metrics/config.py. Do not silently change formulas or their units.
- Use the project's schedule.today_local() convention for default dates. Test timezone boundaries when changing date logic.
- Use game_pk for game identity and key_mlbam for MLB player identity. For hitter modeling, retain both doubleheader games; do not join outcomes by player/date alone.
- For historical features, use only information available before the prediction cutoff. Fit preprocessing and tune models on training folds only; keep evaluation and freeze periods out of tuning.
- Preserve prediction provenance and audit history. Do not rewrite started or resolved picks; same-day pregame refreshes must follow existing supersession rules and retain audit snapshots.
- Keep unknown outcomes distinct from confirmed zero at-bats. Test both cases when changing outcome resolution.
- Change prediction selection versions when qualifier or ranking logic meaningfully changes; retain compatibility with legacy prediction rows and model artifacts. Label market-only fallbacks explicitly and exclude them from independent-model results.
- Persist failed and insufficient-data validation reports as well as passing reports; verify that a fresh environment can retrieve them. Do not promote models or enable betting by bypassing validation/readiness gates. Use actual reports; never fabricate passing results. Keep research outputs separate from published predictions.
- Before evaluating a betting signal, match its predicted outcome to the exact market contract and settlement rules; record the price timestamp and cost/execution assumptions. Report predictive accuracy separately from net returns, with evaluation dates, sample size, and uncertainty. Never describe profitability as guaranteed.
- When comparing markets or strategies, use registered, complete unseen evaluation periods and a market-price baseline; report drawdown, liquidity limits, and sensitivity to costs alongside returns. Treat market selection as part of tuning, keep final holdouts untouched, and permit a no-bet conclusion when evidence is insufficient.
- Keep venue identity, contract rules, executable quotes, and prediction-time versus closing-price provenance explicit. A missing closing quote must remain missing in closing-line reports.
- Keep synthetic fixtures out of production data. Validate data changes for required keys, duplicates, missing values, and downstream CSV consumers.
- Inspect workflow write effects before running it: daily updates, backfills, training, and backtests can change tracked data or models. Run them only when those effects are within the requested task.
- For code changes, run relevant regression tests and the applicable CI checks listed in PROJECT_BRIEF.md. Add focused tests for changed formulas, prediction semantics, or bug fixes. Report checks that could not run; do not claim a model improvement from unit tests alone.
