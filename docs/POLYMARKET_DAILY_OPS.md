# Polymarket prospective collection — daily ops

Manual only. No automated orders. Venue label: **provisional Polymarket US** until confirmed.

**Protocol:** `polymarket_us_game_winner_v2` (registered before new prospective outcomes).  
**Collection start (America/Phoenix):** 2026-09-16. Exploratory (not untouched): 2026-09-14, 2026-09-15.  
**Honest capacity:** remaining 2026 regular season (through 2026-09-27) has ≤12 calendar dates vs structural floor 70 — **insufficient alone**; keep collecting into 2027. Postseason is a separate cohort.

## Host / runtime

- Authorized: GitHub Actions `polymarket_capture.yml` on `ubuntu-latest` (no paid add-ons).
- Optional local runs with the same scripts.
- Missing host: none for this collection path. Do not buy services.

## Startup

```bash
# One-time / when protocol constants change (does not inspect outcomes)
PYTHONPATH=src python scripts/register_polymarket_protocol.py

# Daily collection
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# After Finals (settlement join)
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --skip-cycle --settle-date YYYY-MM-DD

# Checkpoint report only (7 / 14 / 28 / 70 game days) — never enables betting
PYTHONPATH=src python scripts/run_polymarket_paper_evaluation.py
```

## Rules

1. Log every eligible candidate and pass **before** outcomes.
2. Do not inspect or assign a freeze tail until the structural floor exists.
3. Operational collector fixes ≠ evaluation-version / policy changes.
4. Compare models only on identical opportunities vs same-time market mid.
5. Never claim future observations have already occurred.
6. If evidence remains thin, preserve **insufficient_data** — that is a valid result.
