# Model validation reports

Machine-readable nested rolling-origin validation output (`nested_validation_report.json`). Generated locally / in CI — not enormous raw fold-prediction dumps. Optional freeze dates listed in each report must not be repeatedly inspected during development.

Complete DraftKings-style DFS targets write `dfs_complete_targets_nested.json` (see `scripts/train_dfs_complete_model.py`). Legacy partial-target models remain the live benchmark until that report's promotion gate passes.

Betting readiness is checked by `scripts/check_betting_readiness.py` (fail-closed). Do not fabricate gate reports. `GAME_PREDICTION_MODE` and `BETTING_MODE` must stay shadow/disabled until every readiness check passes.

