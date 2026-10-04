import copy
import json

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, hit_prop_development as dev


def game():
    return {'gamePk': 1, 'officialDate': '2026-07-01', 'gameType': 'R', 'status': {'abstractGameState': 'Final'}}


def box(order='100', sub=False, pa=3, ab=3, hits=1):
    return {'teams': {'home': {'team': {'abbreviation': 'LAD'}, 'batters': [10], 'players': {
        'ID10': {'person': {'id': 10}, 'battingOrder': order,
                 'gameStatus': {'isSubstitute': sub, 'isOnBench': False},
                 'stats': {'batting': {'plateAppearances': pa, 'atBats': ab, 'hits': hits}}}}}}}


def row():
    return dict(date='2026-07-01', game_pk=1, key_mlbam=10, team='LAD')


@pytest.mark.parametrize('kwargs, expected', [({},2), ({'hits':0},1), ({'order':'101','sub':True},0),
    ({'pa':1,'ab':0,'hits':0},1), ({'pa':0,'ab':0,'hits':0},0), ({'pa':None},None),
    ({'hits':4},None), ({'order':'100','sub':True},None), ({'sub':None},None)])
def test_official_labels(kwargs, expected):
    assert dev.official_player_label(game(),box(**kwargs),row())[0] == expected


def test_bench_is_confirmed_only_with_complete_membership():
    b=box();p=b['teams']['home']['players']['ID10'];p.pop('battingOrder');p['gameStatus']['isOnBench']=True
    assert dev.official_player_label(game(),b,row())[0] is None
    b['teams']['home']['batters']=[]
    assert dev.official_player_label(game(),b,row())[0] == 0


@pytest.mark.parametrize('changes', [{'gamePk':2},{'officialDate':'2026-07-02'}, {'gameType':'D'},
    {'status':{'abstractGameState':'Live'}}, {'resumeDate':'2026-07-03'}])
def test_wrong_or_unfinished_games_excluded(changes):
    assert dev.official_player_label(game()|changes,box(),row())[0] is None


def test_missing_player_and_wrong_team_stay_unknown():
    assert dev.official_player_label(game(),box(),row()|{'key_mlbam':11})[0] is None
    assert dev.official_player_label(game(),box(),row()|{'team':'SD'})[0] is None


def frame():
    return pd.DataFrame([row() | {'feature_as_of_timestamp':'2026-07-01T00:00:00Z'} |
                         {k:0.5 for k in config.HIT_PROP_DEV_FEATURES}])


def test_label_preparation_ignores_reconstructed_labels_and_leaky_features(tmp_path):
    (tmp_path/'1.json').write_text(json.dumps({'game_pk':1,'payload':box(order='101',sub=True)}))
    raw=frame().assign(Started=1, Hits=1, Batting_Order=1, starter_PAVE=999)
    kept, excluded, hashes=dev.prepare_development_frame(raw,{'dates':[{'games':[game()]}]},tmp_path)
    assert kept.target_state.tolist()==[0]
    assert not {'Started','Hits','Batting_Order','starter_PAVE'} & set(kept)
    assert excluded.empty and '1' in hashes


@pytest.mark.parametrize('change', [{'date':'2026-09-18'}, {'game_pk':1.1},
    {'key_mlbam':True}, {'feature_as_of_timestamp':'2026-07-01T00:01:00Z'}, {'feature_as_of_timestamp':None}])
def test_bad_identity_cutoff_or_protected_date_rejected(tmp_path,change):
    raw=frame()
    for k,v in change.items():raw[k]=v
    with pytest.raises(ValueError):dev.prepare_development_frame(raw,{},tmp_path)


def test_doubleheader_and_missing_boxscores(tmp_path):
    raw=pd.concat([frame(),frame().assign(game_pk=2)],ignore_index=True)
    kept,excluded,_=dev.prepare_development_frame(raw,{},tmp_path)
    assert kept.empty and len(excluded)==2
    with pytest.raises(ValueError,match='Duplicate'):
        dev.prepare_development_frame(pd.concat([frame(),frame()]),{},tmp_path)


def test_payout_bounds_keep_nonbinary_mass():
    np.testing.assert_allclose(dev.payout_bounds([[.2,.3,.5],[0,.4,.6]]),[[.5,.7],[.6,.6]])
    for bad in [[[.2,.3,.6]], [[-.1,.3,.8]], [[np.nan,.3,.7]], [[.5,.5]]]:
        with pytest.raises(ValueError):dev.payout_bounds(bad)


def synthetic_history():
    rows=[]
    for day in range(56):
        for target in range(3):
            rows.append({'date':(pd.Timestamp('2026-07-01')+pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                         'game_pk':day+1,'key_mlbam':target+1,'target_state':target,
                         **{k:float(day%7+target) for k in config.HIT_PROP_DEV_FEATURES}})
    return pd.DataFrame(rows)


def test_forward_folds_and_preprocessing_never_learn_test_data():
    data=synthetic_history();p,report=dev.chronological_development(data)
    assert (p.training_end_date<p.date).all() and report['formal_holdout'] is False
    changed=data.copy();mask=changed.date>'2026-07-28';changed.loc[mask,'target_state']=(changed.loc[mask,'target_state']+1)%3
    changed.loc[mask,'WAVE']=1e6
    before=dev.fit_model(data[~mask]);after=dev.fit_model(changed[~mask])
    np.testing.assert_allclose(before[1].mean_,after[1].mean_)
    p2,_=dev.chronological_development(changed.drop(columns=['WAVE']).assign(WAVE=data.WAVE))
    cols=['p_'+name for name in dev.CLASS_NAMES]
    np.testing.assert_allclose(p.loc[p.fold==0,cols],p2.loc[p2.fold==0,cols])
    assert report['returns'] is None and not report['real_money_ready']


def test_small_history_persists_insufficient_result():
    p,r=dev.chronological_development(synthetic_history().iloc[:3])
    assert p.empty and r['status']=='insufficient_data'
