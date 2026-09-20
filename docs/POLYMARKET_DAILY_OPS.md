# Polymarket prospective collection + gated manual pilot — daily ops

Manual only. No automated orders. Venue label: **provisional Polymarket US** until confirmed.

**Protocol:** `polymarket_us_game_winner_v2` (registered before new prospective outcomes).  
**Pilot market:** `mlb_pregame_moneyline` (candidates in protocol — market-only fallback is never an independent model).  
**Collection start (America/Phoenix):** 2026-09-16. Exploratory (not untouched): 2026-09-14, 2026-09-15.  
**Honest capacity:** remaining 2026 regular season cannot meet structural floor 70 alone — keep collecting.

## Dual readiness (do not conflate)

| Conclusion | Current |
|---|---|
| Software operates for Pass / no-bet inspection | **Yes** (shadow / betting disabled) |
| Evidence supports limited real-money pilot | **No** (`insufficient_evidence`) |

Report: `reports/model_validation/polymarket_pilot_readiness.json`  
Exposure-increase rules: `reports/model_validation/polymarket_pilot_exposure_review_protocol.json` (never raise stakes to recover losses or meet an income deadline).

## Host / runtime

- Authorized: GitHub Actions `polymarket_capture.yml` on `ubuntu-latest` (no paid add-ons).
- Optional local runs with the same scripts.
- Missing host: none for this collection path. Do not buy services.
- Sleep: outside cron windows (`27,57` at 15–23 UTC and 0–2 UTC) the Actions host is asleep.
- Outage: API failures stay in reports; no invented quotes; modes stay shadow/disabled.
- Hit-prop research capture reuses the same workflow (fail-soft step); reports under `reports/polymarket/hit_prop_*`.

## Startup / daily

```bash
# Protocol registration (does not inspect outcomes)
PYTHONPATH=src python scripts/register_polymarket_protocol.py
PYTHONPATH=src python scripts/register_pilot_exposure_review.py

# Daily collection
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# Decision board + pilot readiness (always write even when evidence fails)
PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball
PYTHONPATH=src python scripts/write_pilot_readiness_report.py

# Dashboard (recheck API; no orders)
PYTHONPATH=src python scripts/serve_decision_dashboard.py
# open http://127.0.0.1:8765/polymarket.html

# After Finals (settlement join)
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --skip-cycle --settle-date YYYY-MM-DD

# Checkpoint report only — never enables betting
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py
```

## Manual pilot rules

1. Read the dual banner first. Software YES + evidence NO ⇒ **no bet**.
2. Configure Risk limits only when ready: dedicated bankroll, max affordable loss, per-bet, daily, correlated (same-game + same-team). Until then: **normalized paper units** only.
3. Recheck public quote before any purchase; reject stale quotes (>30s), prices above max acceptable, missing evidence, excess exposure, paused pilot, or stale board export (>5m).
4. Place orders yourself on Polymarket if you choose. Record real fills under Manual positions / `data/polymarket/pilot/real_fills.csv`. Keep real P&L separate from paper.
5. Pauses (ops failure, loss limit, evidence deterioration, owner) block new open exposure; they do **not** erase existing exposure.
6. Never increase stakes to recover losses. Any later increase requires exposure-review checkpoints.

## What remains before a real-money pilot

1. ≥70 Polymarket-labeled eligible dates with nested evaluation **validated_passed** and verdict **edge_supported**.
2. Frozen policy present and hash-matched (already on disk for v2; must stay matched after future evals).
3. Owner supplies the five risk-limit families (not derived from income).
4. Owner confirms venue; only then consider changing `BETTING_MODE` (orders still manual).
5. Do not lower thresholds or promote market-only fallback as an independent model.

## Collection rules (unchanged)

1. Log every eligible candidate and pass **before** outcomes.
2. Do not inspect or assign a freeze tail until the structural floor exists.
3. Operational collector fixes ≠ evaluation-version / policy changes.
4. Compare models only on identical opportunities vs same-time market mid.
5. Never claim future observations have already occurred.
6. If evidence remains thin, preserve **insufficient_data** — that is a valid result.
