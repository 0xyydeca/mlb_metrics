import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, hit_prop_forward as forward
from mlb_metrics.venues.polymarket_us import rules_hash


class Clock:
    def __init__(self):
        self.value = pd.Timestamp('2026-10-04T18:00:00Z')
    def __call__(self):
        return self.value
    def sleep(self, seconds):
        self.value += pd.Timedelta(seconds=seconds)


class Model:
    classes_ = np.array([0, 1, 2])
    def predict_proba(self, frame):
        return np.array([[.2, .3, .5]] * len(frame))


class Adapter:
    def __init__(self, clock, fail_second=False, changes=None):
        self.clock, self.calls, self.fail_second, self.changes = clock, 0, fail_second, changes or {}
    def fetch_market_book(self, slug, *, market_id, fee_coefficient):
        self.calls += 1
        if self.calls > 1 and self.fail_second:
            raise TimeoutError('fixture delayed quote unavailable')
        book = dict(venue_id='polymarket_us', market_id=market_id, market_slug=slug,
                    request_time_utc=self.clock().isoformat(), receive_time_utc=self.clock().isoformat(),
                    eligible=True, suspended=False, best_bid=.48, best_ask=.52, best_ask_size=10,
                    fee_coefficient=.0695, min_trade_qty=None) | self.changes
        return SimpleNamespace(to_dict=lambda: book)


def contract(**changes):
    rules = 'Must be in the starting lineup and record a plate appearance; otherwise last fair market price.'
    return dict(venue_id='polymarket_us', market_id='12', market_slug='slug', game_pk=22, key_mlbam=33,
                rules_text=rules, rules_hash=rules_hash(rules), scheduled_start_utc='2026-10-04T22:00:00Z',
                requested_local_date='2026-10-04', home_team='NYY', away_team='BOS',
                game_type='R', game_type_source='mlb_statsapi_game_pk', quarantined=False,
                game_mapping_status='mapped', player_mapping_status='mapped', stat='hits', threshold=1,
                requires_starting_lineup=True, requires_plate_appearance=True,
                settlement_on_non_participation='last_fair_market_price', fee_coefficient=.0695) | changes


@pytest.fixture
def inputs(monkeypatch):
    model_bytes = b'fixture model never loaded as pickle'
    monkeypatch.setattr(config, 'HIT_PROP_FORWARD_MODEL_SHA256', forward.sha(model_bytes))
    artifact = dict(model=Model(), model_version=config.HIT_PROP_DEV_MODEL_VERSION,
                    features=config.HIT_PROP_DEV_FEATURES, training_end_date='2026-09-14',
                    production_serving_allowed=False, research_only=True)
    monkeypatch.setattr(forward.joblib, 'load', lambda _: artifact)
    wave = {k: .2 for k in config.HIT_PROP_DEV_FEATURES}
    wave.update(key_mlbam=33, team='NYY', Last_Game_Date='2026-10-03')
    return dict(model_bytes=model_bytes, wave_bytes=pd.DataFrame([wave]).to_csv(index=False).encode(),
                contracts_bytes=pd.DataFrame([contract()]).to_csv(index=False).encode())


def capture(tmp_path, inputs, adapter=None, clock=None):
    clock = clock or Clock()
    return forward.capture(root=tmp_path, **inputs, adapter=adapter or Adapter(clock), clock=clock, sleep=clock.sleep)


def test_complete_capture_reopens_in_fresh_audit(tmp_path, inputs):
    report = capture(tmp_path, inputs)
    assert report['forecasts_saved'] == report['executable_quotes_matched'] == 1
    audit = forward.audit_store(tmp_path)
    assert audit['intact_pregame_forecasts'] == audit['execution_quotes_matched'] == 1
    assert audit['integrity_errors'] == []
    assert audit['formal_eligible_dates'] == 0
    assert audit['outcomes_opened'] is False
    path = next((tmp_path/'forecasts').glob('*.json'))
    row = json.loads(path.read_bytes())
    assert row['prediction_time_availability_certified'] is False
    assert row['probabilities'] == dict(nonqualifying=.2, qualifying_no_hit=.3, qualifying_hit=.5)


def test_restart_never_replaces_first_forecast(tmp_path, inputs):
    capture(tmp_path, inputs)
    before = {p.name: p.read_bytes() for p in (tmp_path/'forecasts').glob('*.json')}
    second = capture(tmp_path, inputs)
    assert second['forecasts_saved'] == 0
    assert second['exclusions'][0]['reason'] == 'first_forecast_already_saved'
    assert before == {p.name: p.read_bytes() for p in (tmp_path/'forecasts').glob('*.json')}


def test_failed_delayed_book_preserves_forecast_and_missing_fill(tmp_path, inputs):
    clock = Clock()
    report = capture(tmp_path, inputs, Adapter(clock, fail_second=True), clock)
    assert report['forecasts_saved'] == 1 and report['executable_quotes_matched'] == 0
    audit = forward.audit_store(tmp_path)
    assert audit['missing_execution'] == 1 and audit['integrity_errors'] == []


@pytest.mark.parametrize('changes', [
    dict(scheduled_start_utc='2026-10-04T18:30:00Z'), dict(scheduled_start_utc='2026-10-04T18:00:00Z'),
    dict(scheduled_start_utc=None), dict(scheduled_start_utc='2026-10-04T22:00:00'),
    dict(requested_local_date='2026-10-03'), dict(quarantined=True),
    dict(game_mapping_status='unknown'), dict(player_mapping_status='unknown'),
    dict(game_type_source='guess'), dict(rules_hash='wrong'), dict(fee_coefficient=.01),
    dict(home_team='LAD', away_team='SD'), dict(threshold=2), dict(requires_starting_lineup=False),
])
def test_invalid_candidates_are_persisted_exclusions(tmp_path, inputs, changes):
    inputs['contracts_bytes'] = pd.DataFrame([contract(**changes)]).to_csv(index=False).encode()
    clock = Clock(); adapter = Adapter(clock)
    report = capture(tmp_path, inputs, adapter, clock)
    assert report['forecasts_saved'] == 0 and adapter.calls == 0
    if changes.get('requested_local_date') == '2026-10-03':
        assert report['historical_or_other_date_rows_not_replayed'] == 1
    else:
        assert len(report['exclusions']) == 1
    assert list((tmp_path/'runs').glob('*/report.json'))


@pytest.mark.parametrize('changes', [dict(best_bid=.53), dict(best_ask_size=.1), dict(eligible=False),
    dict(suspended=True), dict(market_id='other'), dict(fee_coefficient=.02), dict(min_trade_qty=2),
    dict(receive_time_utc='2026-10-04T18:01:00Z')])
def test_bad_books_never_become_forecasts(tmp_path, inputs, changes):
    clock = Clock()
    report = capture(tmp_path, inputs, Adapter(clock, changes=changes), clock)
    assert report['forecasts_saved'] == 0 and report['exclusions']


def test_model_change_fails_before_deserialization_and_retains_report(tmp_path, inputs, monkeypatch):
    inputs['model_bytes'] = b'different artifact'
    monkeypatch.setattr(forward.joblib, 'load', lambda _: pytest.fail('must check digest first'))
    report = capture(tmp_path, inputs)
    assert report['status'] == 'failed'
    assert list((tmp_path/'runs').glob('*/report.json'))


@pytest.mark.parametrize('folder', ['forecasts', 'blobs', 'receipts'])
def test_tampering_is_detected(tmp_path, inputs, folder):
    capture(tmp_path, inputs)
    path = next((tmp_path/folder).glob('*'))
    path.write_bytes(b'corrupt')
    assert forward.audit_store(tmp_path)['status'] == 'integrity_failure'


def test_atomic_no_overwrite(tmp_path):
    path = tmp_path/'record.json'
    forward.publish_once(path, b'first')
    with pytest.raises(FileExistsError):
        forward.publish_once(path, b'second')
    assert path.read_bytes() == b'first'
    assert not list(tmp_path.glob('.pending-*'))


def test_same_player_doubleheader_retained(tmp_path, inputs):
    inputs['contracts_bytes'] = pd.DataFrame([contract(), contract(game_pk=23, market_id='13', market_slug='slug2')]).to_csv(index=False).encode()
    assert capture(tmp_path, inputs)['forecasts_saved'] == 2


def test_duplicate_contract_is_not_arbitrarily_selected(tmp_path, inputs):
    inputs['contracts_bytes'] = pd.DataFrame([contract(), contract()]).to_csv(index=False).encode()
    report = capture(tmp_path, inputs)
    assert report['forecasts_saved'] == 0 and len(report['exclusions']) == 2


def test_latest_capture_selected_without_discarding_archived_inputs(tmp_path, inputs):
    rows = [contract(capture_id='capture1'), contract(capture_id='capture2')]
    inputs['contracts_bytes'] = pd.DataFrame(rows).to_csv(index=False).encode()
    report = capture(tmp_path, inputs)
    assert report['forecasts_saved'] == 1
    assert (tmp_path/'blobs'/forward.sha(inputs['contracts_bytes'])).read_bytes() == inputs['contracts_bytes']


def test_missing_receipt_is_not_evidence(tmp_path, inputs):
    capture(tmp_path, inputs)
    next((tmp_path/'receipts').glob('*')).unlink()
    assert forward.audit_store(tmp_path)['status'] == 'integrity_failure'


def test_pending_execution_after_crash_not_fabricated(tmp_path, inputs):
    capture(tmp_path, inputs)
    next((tmp_path/'execution').glob('*')).unlink()
    audit = forward.audit_store(tmp_path)
    assert audit['missing_execution'] == 1 and audit['execution_quotes_matched'] == 0


def test_policy_change_cannot_silently_reuse_stream(tmp_path, inputs, monkeypatch):
    capture(tmp_path, inputs)
    monkeypatch.setattr(config, 'HIT_PROP_FORWARD_ENTRY_MINUTES', 1)
    report = capture(tmp_path, inputs)
    assert report['status'] == 'failed'
    assert 'registration changed' in report['error']


def test_budget_omissions_stay_in_denominator(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(config, 'HIT_PROP_FORWARD_MAX_CONTRACTS', 1)
    inputs['contracts_bytes'] = pd.DataFrame([contract(), contract(market_id='13', market_slug='slug2')]).to_csv(index=False).encode()
    report = capture(tmp_path, inputs)
    assert report['forecasts_saved'] == 1
    assert any(x['reason'] == 'request_budget_omission' for x in report['exclusions'])


def test_inference_crossing_cutoff_never_archived_as_pregame(tmp_path, inputs, monkeypatch):
    clock = Clock()
    original = forward.development.predict_states
    def late(*args):
        value = original(*args)
        clock.value = pd.Timestamp('2026-10-04T22:00:00Z')
        return value
    monkeypatch.setattr(forward.development, 'predict_states', late)
    assert capture(tmp_path, inputs, clock=clock)['forecasts_saved'] == 0


def test_execution_record_tamper_detected(tmp_path, inputs):
    capture(tmp_path, inputs)
    p = next((tmp_path/'execution').glob('*.json'))
    data = json.loads(p.read_bytes()); data['acquisition_cost'] = 0
    p.write_text(json.dumps(data))
    assert forward.audit_store(tmp_path)['status'] == 'integrity_failure'


def test_forecast_already_saved_when_delay_starts(tmp_path, inputs):
    clock = Clock()
    def check_sleep(seconds):
        assert len(list((tmp_path/'forecasts').glob('*.json'))) == 1
        assert len(list((tmp_path/'receipts').glob('*.json'))) == 1
        clock.sleep(seconds)
    result = forward.capture(root=tmp_path, **inputs, adapter=Adapter(clock), clock=clock, sleep=check_sleep)
    assert result['executable_quotes_matched'] == 1
