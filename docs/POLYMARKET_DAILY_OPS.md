# Polymarket decision dashboard — daily ops

Manual only. No automated orders. Personal bankroll / max-loss must be set in the Risk tab before stake guidance can appear.

## Startup

```bash
# 1) Refresh public books (optional but recommended)
PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball

# 2) Export board + copy evaluation into docs/data/
PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball

# 3) Serve docs + quote recheck API
PYTHONPATH=src python scripts/serve_decision_dashboard.py
# Open http://127.0.0.1:8765/polymarket.html
```

Static-only (no recheck API):

```bash
cd docs && python -m http.server 8765
```

## Checklist

1. Read the readiness banner. If `evidence_verdict` ≠ `edge_supported`, the correct action is **no bet**.
2. For each row answer: contract link + side, max acceptable price, permitted exposure units, evidence (policy hash / validation), quote age.
3. Recheck public price before any manual purchase. Stale tabs and quotes invalidate actionability.
4. Record real fills under Manual positions; compare to paper assumptions. Settled net ≠ open exposure.
5. Never increase stakes to recover losses.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_decision_board.py tests/test_paper_ledger.py -q
```
