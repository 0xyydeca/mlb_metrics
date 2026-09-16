# Polymarket decision / foundation — daily ops

Manual only. No automated orders. Venue research label: **provisional Polymarket US** until you confirm US vs international.

## Startup

```bash
# 1) Capture books + baseball (persists snapshots by default)
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball

# 2) Health / audit (mapping ≠ usable prices)
PYTHONPATH=src python scripts/health_polymarket.py
PYTHONPATH=src python scripts/audit_polymarket_data.py

# 3) Trace one observed contract → quote → fees → paper fill math
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --trace-only

# 4) Log candidates/passes (buys suppressed while evidence gates fail)
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py

# 5) Settle any open paper positions after Finals
PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py --skip-cycle --settle-date YYYY-MM-DD

# 6) Dashboard (optional)
PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball
PYTHONPATH=src python scripts/serve_decision_dashboard.py
# → http://127.0.0.1:8765/polymarket.html
```

## Checklist

1. If `evidence_verdict` ≠ `edge_supported`, correct action is **no bet**.
2. Trace file: `reports/polymarket/contract_trace_latest.json` — must show venue, `game_pk`, orientation, rules hash, quote times, fee version, paper cost.
3. Empty ask / ineligible book = liquidity; HTTP failures in capture report = collector problem.
4. Stale quotes (>30s) and missing baseball suppress actionable output.
5. Never increase stakes to recover losses. Configure personal limits before stake guidance.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_paper_pipeline.py tests/test_paper_ledger.py tests/test_data_foundation.py tests/test_decision_board.py tests/test_polymarket_phase1.py -q
```
