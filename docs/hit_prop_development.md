# Three-state hit-prop development model

This experiment uses previously inspected development dates ending September 14,
2026. It is separate from the frozen prospective paper policy and cannot enable
betting. The model is not imported by the production prediction pipeline.

## Target and evidence

The three classes are nonqualifying (did not start or had no PA), qualifying with
zero hits, and qualifying with at least one hit. The label reader uses official
MLB final regular-season boxscores, exact game/player/team identity, batting order
and substitution status. Unknown players/counts and rescheduled games with
unverified completion cutoffs are excluded with explicit reasons. A walk-only
appearance counts as participation. An early substitute's hit does not count as a
qualifying hit. Existing historical logs are never relabeled in place.

The fixed classifier uses regularized logistic regression. Its 13 features are
listed in config.py; starting-pitcher and matchup-derived inputs are excluded.
Median imputation and scaling are fitted within each chronological training fold.
The initial training period is 28 observed dates, followed by 14-date test blocks.
These dates were previously inspected: this is development, not a pristine
holdout. The baseline uses training-only class frequencies on identical rows.
Neither that comparison nor software tests establish an edge over market prices.

Historical feature timestamps are reconstructed date cutoffs. A nominal UTC
midnight stamp does not prove prior-day games had finished or that the provider
had published their final data. Future production use needs verified feature
availability, current lineup evidence and separate prospective validation.
Historical start-rate features also retain the existing reconstructed batting-order
method; official boxscore correction here applies only to target labels.

## Reproduction

Use a separate, new output directory and explicit input files:

```sh
PYTHONPATH=src python scripts/run_hit_prop_development.py \
  --input /path/to/hitter_opportunity_development.csv \
  --schedule /path/to/official_schedule.json \
  --boxscore-dir /path/to/boxscores \
  --output-dir /path/to/new-development-run
```

Schedule input is an MLB Stats API schedule response. Each `<game_pk>.json` in the
boxscore directory wraps the corresponding official boxscore response as
`{"game_pk": 123, "payload": {...}}`; collection snapshots should also retain
source URL and retrieval time. The October 4 research collection and input hashes
are recorded in the working-document research directory. Do not substitute
post-registration evaluation dates or overwrite a previous run.

The runner writes a JSON report even on failure, exclusions, official labels,
forward development predictions, snapshot hashes, and a development-only model
bundle. It reads explicit files and writes only to the requested new directory;
it performs no network requests or production updates. A fitted artifact is saved
only when chronological evaluation can run.

## Payout limits

If the three state probabilities are `(p_nonqualifying, p_no_hit, p_hit)`, the
expected Yes-share payout lies between `p_hit` and `p_hit + p_nonqualifying` when
the non-participation payout is unknown but bounded between zero and one. This is
not a profit forecast or a confidence interval. Game-interruption nonbinary
settlement is not included in this target. Execution prices, fees, liquidity,
nonbinary payout provenance, and untouched prospective results remain required.
