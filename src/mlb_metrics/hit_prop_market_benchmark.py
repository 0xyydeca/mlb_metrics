"""Price-matched research comparison, without opening settlement outcomes.

Quotes are reference observations, not actual fills. All rows remain paper-only.
A three-state forecast is compared as an expected-payout interval, never as a
conditional hit probability against an unconditional contract price.
"""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_EVEN
from collections import Counter
import math
import numpy as np
import pandas as pd
from mlb_metrics import config

IDENTITY = ['venue_id', 'market_slug', 'game_pk', 'key_mlbam', 'rules_hash']


def number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def timestamp(value):
    try:
        ts = pd.Timestamp(value)
        return ts.tz_convert('UTC') if not pd.isna(ts) and ts.tzinfo else None
    except (ValueError, TypeError):
        return None


def standard_fee_cap(price, quantity):
    """Current standard US order fee cap; no combos, rebates, or early sale."""
    p, q = number(price), number(quantity)
    if p is None or q is None or not 0 < p < 1 or q <= 0:
        raise ValueError('Invalid fee inputs')
    exact = Decimal(str(config.HIT_PROP_BENCHMARK_TAKER_THETA))*Decimal(str(q))*Decimal(str(p))*(1-Decimal(str(p)))
    return float(exact.quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN))


def coverage_audit(forecasts, quotes):
    """Only identities/dates; never access outcome columns or score holdouts."""
    def pairs(frame):
        if not {'game_pk','key_mlbam'} <= set(frame):
            return set()
        result=set()
        for game,player in zip(frame.game_pk,frame.key_mlbam):
            g,p=number(game),number(player)
            if g is not None and p is not None and g>0 and p>0 and g.is_integer() and p.is_integer():
                result.add((int(g),int(p)))
        return result
    def span(frame,field):
        if field not in frame or frame[field].dropna().empty:
            return None
        v=frame[field].dropna().astype(str)
        return [v.min(),v.max()]
    return {'forecast_rows':len(forecasts),'quote_rows':len(quotes),
            'forecast_dates':span(forecasts,'date'),'quote_dates':span(quotes,'requested_local_date'),
            'overlapping_game_player_keys':len(pairs(forecasts)&pairs(quotes))}


def identity(row):
    for k in IDENTITY:
        v=row.get(k)
        if v is None or pd.isna(v) or not str(v).strip():
            return None
    g,p=number(row['game_pk']),number(row['key_mlbam'])
    if any(v is None or v<=0 or not v.is_integer() for v in (g,p)):
        return None
    return (str(row['venue_id']),str(row['market_slug']),int(g),int(p),str(row['rules_hash']))


def compare_forecasts(forecasts,quotes):
    """Deterministic same-time baseline and 60-second delayed ask comparison.

    No evaluation outcomes are read; edge intervals are model-implied only.
    Missing delayed books stay missing, never replaced by current/closing prices.
    """
    audit=coverage_audit(forecasts,quotes)
    rows=[]
    grouped={}
    for q in quotes.to_dict('records'):
        key=identity(q)
        ts=timestamp(q.get('book_receive_time_utc'))
        if key is not None and ts is not None:
            grouped.setdefault(key,[]).append((ts,q))
    for values in grouped.values():
        values.sort(key=lambda v:v[0])
    records=forecasts.to_dict('records')
    counts=Counter((identity(f),timestamp(f.get('forecast_time_utc'))) for f in records)
    for i,f in enumerate(records):
        out={'forecast_row':i,'game_pk':f.get('game_pk'),'key_mlbam':f.get('key_mlbam'),
             'market_slug':f.get('market_slug'),'venue_id':f.get('venue_id'),
             'rules_hash':f.get('rules_hash'),'artifact_sha256':f.get('artifact_sha256'),
             'status':'excluded','actionable':False}
        reasons=[]
        key=identity(f);decision=timestamp(f.get('forecast_time_utc'))
        start=timestamp(f.get('scheduled_start_utc'))
        if key is None:reasons.append('missing_or_invalid_contract_identity')
        if f.get('venue_id')!='polymarket_us':reasons.append('unsupported_or_missing_venue')
        if decision is None or start is None or decision>=start:reasons.append('invalid_or_poststart_forecast_time')
        for field in ['feature_cutoff_utc','training_end_time_utc','artifact_created_at_utc']:
            ts=timestamp(f.get(field))
            if ts is None or decision is None or ts>decision:reasons.append('missing_or_late:'+field)
        digest=f.get('artifact_sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest.lower()):
            reasons.append('missing_artifact_hash')
        certified=f.get('prediction_time_availability_certified')
        if not isinstance(certified,(bool,np.bool_)) or not certified:reasons.append('uncertified_prediction_time_inputs')
        if f.get('probability_source')!='independent_three_state_model':reasons.append('not_independent_model')
        probs=[number(f.get('p_'+name)) for name in ('nonqualifying','qualifying_no_hit','qualifying_hit')]
        if any(x is None or not 0<=x<=1 for x in probs) or not math.isclose(sum(x or 0 for x in probs),1,abs_tol=1e-9):
            reasons.append('invalid_three_state_probabilities')
        marker=(key,decision)
        if key is not None and decision is not None and counts[marker]>1:reasons.append('duplicate_forecast_identity_time')
        if reasons:
            out['reasons']=';'.join(reasons);rows.append(out);continue
        available=grouped.get(key,[])
        now=[v for v in available if v[0]<=decision and (decision-v[0]).total_seconds()<=config.HIT_PROP_BENCHMARK_QUOTE_AGE]
        deadline=decision+pd.Timedelta(seconds=config.HIT_PROP_BENCHMARK_DELAY_SECONDS)
        later=[v for v in available if deadline<=v[0]<start and (v[0]-deadline).total_seconds()<=config.HIT_PROP_BENCHMARK_QUOTE_AGE]
        if not now or not later:
            out['reasons']='missing_prediction_time_quote' if not now else 'missing_delayed_execution_quote'
            rows.append(out);continue
        ts,q=now[-1];fill_ts,fill=later[0]
        # Multiple conflicting observations at an identical timestamp are ambiguous.
        if sum(t==ts for t,_ in now)>1 or sum(t==fill_ts for t,_ in later)>1:
            out['reasons']='ambiguous_quote_timestamp';rows.append(out);continue
        for t,obs in [(ts,q),(fill_ts,fill)]:
            ask,no,size=number(obs.get('yes_buy_price')),number(obs.get('no_buy_price')),number(obs.get('yes_buy_size'))
            req=timestamp(obs.get('book_request_time_utc'));obs_start=timestamp(obs.get('scheduled_start_utc'))
            if (ask is None or no is None or not 0<ask<1 or not 0<no<1 or 1-no>ask or size is None or size<config.HIT_PROP_BENCHMARK_QUANTITY
                or req is None or req>t or obs_start!=start or obs.get('book_status')!='captured'
                or str(obs.get('quarantined')).lower()!='false' or obs.get('game_mapping_status')!='mapped' or obs.get('player_mapping_status')!='mapped'
                or number(obs.get('fee_coefficient'))!=config.HIT_PROP_BENCHMARK_TAKER_THETA
                or obs.get('stat')!='hits' or number(obs.get('threshold'))!=1
                or obs.get('settlement_on_non_participation')!='last_fair_market_price'
                or str(obs.get('requires_starting_lineup')).lower()!='true' or str(obs.get('requires_plate_appearance')).lower()!='true'):
                reasons.append('invalid_quote_identity_rules_book_or_size')
        if fill_ts<timestamp(config.HIT_PROP_BENCHMARK_FEE_EFFECTIVE_UTC):reasons.append('fee_schedule_not_verified_for_quote_date')
        if reasons:
            out['reasons']=';'.join(reasons);rows.append(out);continue
        ask=float(fill['yes_buy_price']);quantity=config.HIT_PROP_BENCHMARK_QUANTITY
        fee=standard_fee_cap(ask,quantity);cost=ask+fee/quantity
        mid=(float(q['yes_buy_price'])+1-float(q['no_buy_price']))/2
        lower,upper=probs[2],probs[2]+probs[0]
        out.update(status='paper_price_comparison',reasons='',forecast_time_utc=decision.isoformat(),
            quote_time_utc=ts.isoformat(),execution_quote_time_utc=fill_ts.isoformat(),
            prediction_capture_id=q.get('capture_id'),execution_capture_id=fill.get('capture_id'),
            market_mid_payout_reference=mid,execution_yes_ask=ask,quantity=quantity,
            rounded_order_fee_cap=fee,all_in_cost_per_contract=cost,
            model_expected_payout_lower=lower,model_expected_payout_upper=upper,
            model_minus_market_mid_lower=lower-mid,model_minus_market_mid_upper=upper-mid,
            after_fee_expected_net_lower=lower-cost,after_fee_expected_net_upper=upper-cost,
            fee_source=config.HIT_PROP_BENCHMARK_FEE_SOURCE,closing_quote_used=False)
        rows.append(out)
    result=pd.DataFrame(rows)
    matched=int((result.status=='paper_price_comparison').sum()) if not result.empty else 0
    report={'status':'paper_comparison_only' if matched else 'insufficient_evidence',
            'coverage':audit,'matched_comparisons':matched,'excluded_forecasts':len(rows)-matched,
            'exclusion_reason_counts':result.loc[result.status=='excluded','reasons'].value_counts().to_dict() if not result.empty else {},
            'realized_returns':None,'outcomes_opened':False,'real_money_ready':False,
            'fee_source':config.HIT_PROP_BENCHMARK_FEE_SOURCE,
            'assumptions':{'quantity':config.HIT_PROP_BENCHMARK_QUANTITY,'delay_seconds':config.HIT_PROP_BENCHMARK_DELAY_SECONDS,
                'quote_max_age_seconds':config.HIT_PROP_BENCHMARK_QUOTE_AGE,'fee_model':'standard US rounded order fee cap; no rebates; single buy held to settlement',
                'fills':'Top-of-book size-limited hypothetical execution; not actual fills',
                'nonbinary':'Expected payout bounds, not a conditional hit probability or invented LFMP',
                'baseline':'Same-time midpoint is a payout reference, not a fee-adjusted probability',
                'evaluation':'No outcome scoring or promotion; registered formal evaluation remains separate'}}
    return result,report
