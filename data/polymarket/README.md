# Polymarket local data

- `registry/` — contract registry CSV written by capture (gitignored except `.gitkeep`)
- `quotes/` — date-partitioned Parquet order-book observations (gitignored)

Never commit personal account data or order credentials here. Capture is
public read-only.
