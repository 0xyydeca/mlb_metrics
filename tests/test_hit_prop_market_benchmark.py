import numpy as np
import pandas as pd
import pytest
from mlb_metrics import hit_prop_market_benchmark as b


def fixture():
    common=dict(venue_id='polymarket_us',market_slug='slug',game_pk=1,key_mlbam=2,rules_hash='r',scheduled_start_utc='2026-10-04T23:00:00Z')
    f=common|dict(forecast_time_utc='2026-10-04T22:00:00Z',feature_cutoff_utc='2026-10-04T21:00:00Z',training_end_time_utc='2026-09-14T23:00:00Z',artifact_created_at_utc='2026-10-04T20:00:00Z',artifact_sha256='a'*64,prediction_time_availability_certified=True,probability_source='independent_three_state_model',p_nonqualifying=.1,p_qualifying_no_hit=.3,p_qualifying_hit=.6)
    q=common|dict(book_request_time_utc='2026-10-04T21:59:58Z',book_receive_time_utc='2026-10-04T21:59:59Z',yes_buy_price=.55,no_buy_price=.47,yes_buy_size=3,book_status='captured',fee_coefficient=.0695,quarantined=False,game_mapping_status='mapped',player_mapping_status='mapped',stat='hits',threshold=1,settlement_on_non_participation='last_fair_market_price',requires_starting_lineup=True,requires_plate_appearance=True)
    later=q|dict(book_request_time_utc='2026-10-04T22:00:59Z',book_receive_time_utc='2026-10-04T22:01:00Z',yes_buy_price=.56)
    return pd.DataFrame([f]),pd.DataFrame([q,later])


def test_correct_market_reference_delay_fees_and_nonbinary_bounds():
    f,q=fixture();rows,r=b.compare_forecasts(f,q);x=rows.iloc[0]
    assert r['matched_comparisons']==1 and not r['outcomes_opened']
    assert x.market_mid_payout_reference==pytest.approx(.54)
    assert x.all_in_cost_per_contract==pytest.approx(.58)
    assert x.after_fee_expected_net_lower==pytest.approx(.02)
    assert x.after_fee_expected_net_upper==pytest.approx(.12)
    assert not x.actionable and r['realized_returns'] is None


@pytest.mark.parametrize('field,value', [('prediction_time_availability_certified',False),('prediction_time_availability_certified','True'),('probability_source','same_time_market_mid_baseline'),('feature_cutoff_utc','2026-10-04T22:01:00Z'),('artifact_created_at_utc','2026-10-05T00:00:00Z'),('artifact_sha256','bad'),('p_nonqualifying',np.nan),('p_qualifying_hit',1.2),('game_pk',1.5),('forecast_time_utc','2026-10-04T22:00:00'),('forecast_time_utc','2026-10-04T23:00:00Z')])
def test_unverified_or_leaking_forecast_excluded(field,value):
    f,q=fixture();f[field]=value;_,r=b.compare_forecasts(f,q);assert r['matched_comparisons']==0


@pytest.mark.parametrize('field,value',[('fee_coefficient',0),('fee_coefficient',None),('fee_coefficient',.06),('key_mlbam',3),('game_pk',2),('rules_hash','other'),('venue_id','polymarket_international'),('market_slug','other'),('yes_buy_size',.5),('quarantined',True),('yes_buy_price',0),('no_buy_price',.1),('book_status','failed'),('requires_starting_lineup',False),('threshold',2),('scheduled_start_utc','2026-10-04T23:01:00Z')])
def test_wrong_contract_bad_size_or_invalid_book_excluded(field,value):
    f,q=fixture();q[field]=value;_,r=b.compare_forecasts(f,q);assert r['matched_comparisons']==0


def test_missing_execution_quote_never_uses_old_or_closing_price():
    f,q=fixture();_,r=b.compare_forecasts(f,q.iloc[:1]);assert r['matched_comparisons']==0
    q.loc[1,'book_receive_time_utc']='2026-10-04T23:00:00Z'
    _,r=b.compare_forecasts(f,q);assert r['matched_comparisons']==0


def test_duplicates_excluded_without_cherry_picking():
    f,q=fixture();_,r=b.compare_forecasts(pd.concat([f,f]),q);assert r['matched_comparisons']==0
    _,r=b.compare_forecasts(f,pd.concat([q,q]));assert r['matched_comparisons']==0


def test_observed_fee_examples_and_bad_inputs():
    assert b.standard_fee_cap(.65,1000)==15.81
    assert b.standard_fee_cap(.5,1000)==17.38
    assert b.standard_fee_cap(.5,1)==.02
    for p,q in [(0,1),(.5,0),(.5,float('inf')),(True,1)]:
        with pytest.raises(ValueError):b.standard_fee_cap(p,q)


def test_no_overlap_audit_and_empty_inputs():
    f,q=fixture();f['game_pk']=9
    _,r=b.compare_forecasts(f,q);assert r['coverage']['overlapping_game_player_keys']==0
    _,r=b.compare_forecasts(pd.DataFrame(),pd.DataFrame());assert r['status']=='insufficient_evidence'


def test_apparent_midpoint_advantage_can_lose_after_spread_and_fees():
    f,q=fixture();f['p_nonqualifying']=0;f['p_qualifying_no_hit']=.44;f['p_qualifying_hit']=.56
    rows,_=b.compare_forecasts(f,q);r=rows.iloc[0]
    assert r.model_minus_market_mid_lower>0
    assert r.after_fee_expected_net_upper<0


def test_current_fee_schedule_never_backdated():
    f,q=fixture()
    for table in [f,q]:
        for col in table.columns:
            if col.endswith('_utc'):
                table[col]=table[col].str.replace('2026-10-04','2026-09-30')
    rows,r=b.compare_forecasts(f,q)
    assert r['matched_comparisons']==0
    assert 'fee_schedule_not_verified_for_quote_date' in rows.iloc[0].reasons
