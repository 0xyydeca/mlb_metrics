# Independent prospective research archive

Run `PYTHONPATH=src python scripts/capture_hit_prop_forward.py` from the repository.
The existing Polymarket capture workflow runs this after a successful prop capture,
then runs `--audit-only`. It uses public read-only books and never places orders.
The pinned development artifact is used only in this separate research stream;
it is not promoted into the published picks or frozen paper policy.

## What is recorded

- Full input bytes for the contract registry, WAVE features and pinned model;
  content-addressed SHA-256 blobs retain repeated snapshots without overwriting.
- A registration fixing model hash, features, entry cutoff, quote delay/age,
  fee regime, quantity, request budget and selection order. A changed registration
  fails instead of silently changing the experiment.
- The current Phoenix-date/latest-capture universe, with exclusion reasons for
  duplicates, missing identities/features, past cutoffs and budget omissions.
- The first eligible independent three-state forecast for each exact venue,
  market ID/slug, game, player and rules hash, saved at least 30 minutes before
  scheduled start. Both doubleheader game IDs remain distinct.
- The exact feature matrix and prediction-time book; a separate persistence
  receipt; and the first attempted execution book after the fixed 60-second
  delay. Quotes outside the 30-second window, crossed books and insufficient
  displayed size remain missing hypothetical fills. No closing quote substitution.
- One-share hypothetical acquisition cost including the standard rounded fee
  cap. This is not an actual fill; minimum quantity is checked when supplied by
  the book adapter. Unknown venue minimums and actual execution remain unverified.

Files are atomically created without replacement. Incomplete runs, missing
receipts and lost delayed quotes remain visible. Restarting never regenerates
an existing forecast or substitutes a later quote for a missed window. New model
or policy versions require a reviewed, separate archive; old records remain.

`--audit-only` reopens inputs, verifies hashes and timing, and reconciles price
and fee arithmetic. SHA-256 detects inconsistencies, not malicious rewriting of
all files or independent clock tampering. The existing GitHub workflow commits
these files and uploads artifacts, providing an additional publication record;
verify its actual completion before claiming hosted collection ran successfully.

## What remains blocked

The archive is prospective **research**, not a validated track record. Historical
model training availability is uncertified, and formal holdout assignment remains
unassigned. Current bounds for nonparticipation remain unchanged. Regular-season
and postseason counts are separate. Neither is automatically a new eligible date
in the existing registered evaluation.

This command does not retrieve settlement outcomes or calculate aggregate returns.
After the registered model/provenance, untouched-assignment and sample-size gates
are satisfied, the evaluator must join actual venue settlements to these immutable
contract identities and report probability quality versus the same-time market,
net returns after recorded costs, missing settlements, drawdown, uncertainty and
cost sensitivity. Fractional settlement evidence uses `hit_prop_settlement`;
unverified normalized evidence is not sufficient for final validation. No final
settlement price may be used as a forecast-time LFMP estimate.

No historical-date option exists. Late implementation cannot produce a pregame
forecast for a game whose cutoff has passed. The CLI writes only within its
explicit output directory; a missing input or model mismatch persists a failure
report and does not trigger training, backfill or model replacement.
