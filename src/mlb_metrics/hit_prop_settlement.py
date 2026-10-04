"""Evidence-backed fractional payouts, separate from forecast-time estimates.

Input is an explicitly normalized settlement record, not an undocumented venue
API response. The caller supplies the archived source bytes and normalized
record; the hash binds them, but does not authenticate the provider or attest
that normalization was correct. Never substitute a quote for a final payout.
"""
from __future__ import annotations

import hashlib
import math
from urllib.parse import urlparse

import pandas as pd


IDENTITY_FIELDS = ('venue_id', 'market_id', 'market_slug', 'game_pk', 'key_mlbam', 'rules_hash')


def resolve_fractional_payout(*, contract, evidence, source_bytes, as_of_utc):
    """Return an audited Yes payout or an explicit pending result; never guess."""
    def pending(reason):
        return {'status': 'pending', 'reason': reason, 'yes_payout': None}

    def stamp(value):
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts) or ts.tzinfo is None:
                return None
            return ts.tz_convert('UTC')
        except (ValueError, TypeError, OverflowError):
            return None

    if not isinstance(evidence, dict) or not isinstance(contract, dict):
        return pending('missing_evidence_or_contract')
    for key in IDENTITY_FIELDS:
        expected, actual = contract.get(key), evidence.get(key)
        if not isinstance(expected, str) or not expected.strip() or expected != actual:
            return pending('identity_mismatch_' + key)
    if contract['venue_id'] != 'polymarket_us':
        return pending('unsupported_venue')
    if contract.get('settlement_on_non_participation') != 'last_fair_market_price':
        return pending('unsupported_nonparticipation_rule')
    if evidence.get('record_type') != 'final_settlement' or evidence.get('final') is not True:
        return pending('not_final_settlement')
    if evidence.get('settlement_reason') != 'non_participation':
        return pending('unsupported_settlement_reason')
    if evidence.get('payout_side') != 'YES' or evidence.get('payout_unit') != 'USD_per_contract':
        return pending('ambiguous_payout_side_or_unit')
    source_url = evidence.get('source_url')
    try:
        url = urlparse(source_url if isinstance(source_url, str) else '')
        valid_url = url.scheme == 'https' and (url.hostname == 'polymarket.us' or (url.hostname or '').endswith('.polymarket.us')) and not url.username and not url.password
    except ValueError:
        valid_url = False
    if not valid_url:
        return pending('unsupported_source_url')
    if not isinstance(source_bytes, bytes) or not source_bytes:
        return pending('missing_source_snapshot')
    digest = hashlib.sha256(source_bytes).hexdigest()
    if evidence.get('source_sha256') != digest:
        return pending('source_hash_mismatch')
    as_of, received, settled = map(stamp, (as_of_utc, evidence.get('received_at_utc'), evidence.get('settled_at_utc')))
    if any(x is None for x in (as_of, received, settled)):
        return pending('invalid_evidence_timestamp')
    if settled > received or received > as_of:
        return pending('evidence_not_yet_available')
    px = evidence.get('yes_payout')
    if isinstance(px, bool):
        return pending('invalid_payout')
    try:
        px = float(px)
    except (TypeError, ValueError, OverflowError):
        return pending('invalid_payout')
    if not math.isfinite(px) or not 0 <= px <= 1:
        return pending('invalid_payout')
    return {'status': 'resolved', 'reason': 'recorded_final_nonparticipation_payout',
            'yes_payout': px,
            'source_sha256': digest, 'source_url': source_url,
            'received_at_utc': received.isoformat(), 'settled_at_utc': settled.isoformat(),
            'normalization_independently_verified': False}
