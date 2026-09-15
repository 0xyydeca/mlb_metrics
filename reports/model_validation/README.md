# Model validation reports

Machine-readable nested rolling-origin validation output. Generated locally /
in CI. Optional freeze dates listed in each report must not be repeatedly
inspected during development.

Residual training always writes:

- `game_residual_nested.json` — nested validation + probability promotion gate
- `game_residual_betting_gate.json` — betting promotion companion report

These files are force-committed by the Game Residual Training workflow and
uploaded as the `residual-validation-reports` artifact, including
`insufficient_data` and `validated_failed` outcomes. A green workflow is not
model readiness.

Complete DraftKings-style DFS targets write `dfs_complete_targets_nested.json`
(see `scripts/train_dfs_complete_model.py`). Legacy partial-target models
remain the live benchmark until that report's promotion gate passes.

Betting readiness is checked by `scripts/check_betting_readiness.py`
(fail-closed). Do not fabricate gate reports. `GAME_PREDICTION_MODE` and
`BETTING_MODE` must stay shadow/disabled until every readiness check passes.
