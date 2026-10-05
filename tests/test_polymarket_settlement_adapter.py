"""Settlement response validation; fixtures do not establish live API semantics."""
import hashlib
import json
import pytest
from mlb_metrics.venues.polymarket_us import PolymarketUSAdapter


def adapter(monkeypatch, payload):
    client = PolymarketUSAdapter(max_retries=0, min_interval_seconds=0)
    monkeypatch.setattr(client, '_http_get_json', lambda path: payload)
    return client


@pytest.mark.parametrize('payload', [None, [], {}, {'settlement':.4},
    {'slug':'other','settlement':.4}, {'slug':'','settlement':.4},
    {'slug':'market','settlement':True}, {'slug':'market','settlement':False},
    {'slug':'market','settlement':float('nan')}, {'slug':'market','settlement':float('inf')},
    {'slug':'market','settlement':-.1}, {'slug':'market','settlement':1.1},
    {'slug':'market','settlement':None}, {'slug':'market','settlement':'bad'}])
def test_invalid_response_is_rejected(monkeypatch, payload):
    with pytest.raises(RuntimeError):
        adapter(monkeypatch,payload).fetch_market_settlement('market')


@pytest.mark.parametrize('value', [0, .37, 1, '0.37'])
def test_valid_fractional_response_preserves_evidence(monkeypatch,value):
    payload={'slug':'market','settlement':value,'fixture_note':'source data'}
    result=adapter(monkeypatch,payload).fetch_market_settlement('market')
    assert result['settlement']==float(value)
    assert result['response_payload']==payload
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert result['response_payload_sha256']==hashlib.sha256(raw).hexdigest()
    assert result['request_time_utc']<=result['receive_time_utc']
    assert 'final' not in result


@pytest.mark.parametrize('slug',[None,'', '  ', 12])
def test_invalid_request_does_not_call_provider(monkeypatch,slug):
    client=PolymarketUSAdapter()
    monkeypatch.setattr(client,'_http_get_json',lambda _:pytest.fail('invalid request reached network'))
    with pytest.raises(ValueError):client.fetch_market_settlement(slug)


def test_market_slug_path_is_encoded(monkeypatch):
    client=PolymarketUSAdapter()
    paths=[]
    def respond(path):
        paths.append(path)
        return {'slug':'market/part','settlement':.4}
    monkeypatch.setattr(client,'_http_get_json',respond)
    client.fetch_market_settlement('market/part')
    assert paths==['/v1/markets/market%2Fpart/settlement']
