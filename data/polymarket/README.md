# Polymarket local data

- `registry/` — contract registry CSV from capture (`contracts.csv` may be tracked)
- `quotes/` — date-partitioned Parquet order-book observations (**gitignored**)

Price unit: USD share cost in `[0,1]`. Quantity unit: contracts.
Capture is public read-only. Never commit account credentials or order secrets.

Mapping success in the registry is **not** the same as usable price coverage —
see `reports/polymarket/coverage_latest.json` waterfall / health scripts.
