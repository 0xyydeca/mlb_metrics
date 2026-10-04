# Comparing the hitter model with Polymarket prices after fees

This read-only benchmark compares independent three-state forecasts with saved
Polymarket US 1+ hit quotes. It does not score evaluation outcomes, change the
frozen paper policy, certify an edge, or place orders.

## Calculation

For the identical venue, market slug, game ID, player ID and rules hash:

1. Select the latest quote received at or before the forecast (at most 30 seconds old).
2. Retain its two-sided midpoint as the market payout reference.
3. Select the first quote at least 60 seconds after the forecast, at most 30 seconds
   after that deadline, and strictly before the scheduled start. This is the
   hypothetical delayed execution quote. Missing quotes remain missing.
4. Require sufficient displayed Yes-ask size for the configured one-contract
   quantity. Add the standard US rounded order fee cap to the delayed ask.
5. Compare model payout bounds against both the same-time midpoint and all-in cost:
   lower payout = P(qualifying hit); upper payout = lower + P(nonqualifying).
   Subtract all-in cost from both bounds for model-implied expected-net bounds.

These are hypothetical top-of-book calculations, not observed fills or realized
profit. A conditional hit probability cannot be compared directly to a contract
price when non-participation has a nonbinary payout. The bounds also do not model
interrupted-game settlement. There is no exit trade, rebate, deeper-book fill,
slippage stress or claim of favorable fill certainty in this comparison.

The [official US fee schedule](https://docs.polymarket.us/fees), verified October 4,
2026, takes effect October 1 at 14:00 UTC. Standard taker theta is 0.0695; rounded
fees use nearest-cent, half-to-even rounding. For split fills, cumulative rounding
caps the order fee; this benchmark conservatively uses that cap. Combos and the
international venue are unsupported. Earlier quotes are rejected rather than
backdated to today's verified fee schedule. Later fee changes require a new
verified schedule.

## Forecast input

CSV fields required for a comparable row:

- `venue_id`, `market_slug`, `game_pk`, `key_mlbam`, `rules_hash`
- `forecast_time_utc`, `scheduled_start_utc`, `feature_cutoff_utc`,
  `training_end_time_utc`, `artifact_created_at_utc` (timezone-aware timestamps)
- `artifact_sha256`, `prediction_time_availability_certified` (actual boolean)
- `probability_source=independent_three_state_model`
- `p_nonqualifying`, `p_qualifying_no_hit`, `p_qualifying_hit` (sum to one)

Metadata must come from actual saved provenance. Do not fill old predictions with
today's model hash or invent pregame creation times to make them pass. The current
development model has uncertified historical timing and is not yet an eligible
prospective forecast source. Duplicate forecast identity/times and ambiguous
quote timestamps are rejected. All exclusions remain visible.

The quote input uses the existing hit-prop contracts CSV schema, including exact
identity/rules, receipt and request timestamps, Yes/No asks, displayed Yes size,
participation rules, matching fee coefficient, book status, and mapped/unquarantined status.

## Run

```sh
PYTHONPATH=src python scripts/compare_hit_prop_market.py \
  --forecasts /path/to/forecasts.csv \
  --quotes data/polymarket/research/hit_props/contracts.csv \
  --output-dir /path/to/new-comparison-run
```

Only the new output directory is written. Existing runs cannot be overwritten;
failed runs retain a JSON report. Report hashes identify both inputs. The runner
reads identity/forecast/quote fields, never outcome labels. Realized return and
formal probability-performance scoring stay with the registered readiness workflow.

## October 4 actual data check

14,447 development predictions cover August 4–September 14. The 1,975 saved
contract observations cover September 18–October 4. Their exact game/player key
intersection is zero. Consequently the saved report is `insufficient_evidence`,
with zero matched comparisons and no realized ROI. Historical OHLC or last-trade
prices would not establish executable asks, sizes and delayed fills. This task
has not established that an archival source can recover those missing observations.
