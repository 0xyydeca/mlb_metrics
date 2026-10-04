import hashlib
import pytest
from mlb_metrics import hit_prop_settlement as payout, hit_prop_paper as paper, paper_ledger

SOURCE = b'{"test_fixture": "normalized settlement evidence"}'
CONTRACT = dict(venue_id='polymarket_us', market_id='12', market_slug='hitter-hit', game_pk='22', key_mlbam='33', rules_hash='rules1', settlement_on_non_participation='last_fair_market_price')


def evidence(**changes):
    return CONTRACT | dict(record_type='final_settlement', final=True, settlement_reason='non_participation', payout_side='YES', payout_unit='USD_per_contract', source_url='https://api.polymarket.us/fixture', source_sha256=hashlib.sha256(SOURCE).hexdigest(), received_at_utc='2026-10-04T22:00:00Z', settled_at_utc='2026-10-04T21:00:00Z', yes_payout=.37) | changes


def resolve(record):
    return payout.resolve_fractional_payout(contract=CONTRACT, evidence=record, source_bytes=SOURCE, as_of_utc='2026-10-04T23:00:00Z')


@pytest.mark.parametrize('value', [0, .37, 1])
def test_fractional_and_boundary_payouts(value):
    result = resolve(evidence(yes_payout=value))
    assert result['status'] == 'resolved'
    assert result['yes_payout'] == value
    assert result['normalization_independently_verified'] is False


@pytest.mark.parametrize('key', payout.IDENTITY_FIELDS)
def test_identity_mismatch(key):
    assert resolve(evidence(**{key: 'wrong'}))['status'] == 'pending'


@pytest.mark.parametrize('changes', [
    dict(final=False), dict(final='True'), dict(record_type='quote'),
    dict(payout_side='NO'), dict(payout_unit='cents'), dict(source_sha256='wrong'),
    dict(source_url='https://polymarket.us.attacker.example/'), dict(source_url='http://polymarket.us/'),
    dict(received_at_utc='2026-10-05T00:00:00Z'), dict(settled_at_utc='2026-10-04T22:30:00Z'),
    dict(received_at_utc='2026-10-04T22:00:00'), dict(settled_at_utc=None),
    dict(settlement_reason='postponed'), dict(yes_payout=True), dict(yes_payout=-.1),
    dict(yes_payout=1.1), dict(yes_payout=float('nan')), dict(yes_payout=float('inf')),
    dict(yes_payout=None), dict(yes_payout='bad'),
])
def test_invalid_evidence_stays_pending(changes):
    assert resolve(evidence(**changes))['status'] == 'pending'


def trade(**changes):
    args = dict(yes_asks=[{'price':.5,'size':10}], decision_time_utc='2026-10-04T20:00:00Z', market_id='12', market_slug='hitter-hit', game_pk=22, key_mlbam=33, player_name='Fixture', model_name='fixture', model_probability=.6, market_mid_probability=.5, settlement={'settlement_class':'last_fair_market_price'}, settlement_contract=CONTRACT, settlement_evidence=evidence(), settlement_source_bytes=SOURCE, settlement_as_of_utc='2026-10-04T23:00:00Z')
    return paper.simulate_prop_paper_trade(**(args | changes))


def test_realized_fractional_accounting_includes_acquisition_cost():
    result = trade()
    position = result['position']
    assert position['status'] == 'settled'
    assert position['proceeds'] == pytest.approx(.37)
    assert position['net_pnl'] == pytest.approx(.37 - result['purchase']['acquisition_cost'])
    assert position['payout_source'] == 'recorded_settlement'


def test_invalid_record_does_not_settle():
    result = trade(settlement_evidence=evidence(final=False))
    assert result['position']['status'] == 'open'
    assert result['position']['net_pnl'] is None
    assert result['position']['payout_evidence']['reason'] == 'not_final_settlement'


def test_no_mixing_evidence_and_assumption():
    with pytest.raises(ValueError, match='mix'):
        trade(lfmp_price=.5)


def test_position_identity_bound():
    with pytest.raises(ValueError, match='match position'):
        trade(game_pk=23)


@pytest.mark.parametrize('value', [-1, 2, float('nan'), float('inf'), True])
def test_ledger_rejects_invalid_fractional_payout(value):
    with pytest.raises(ValueError):
        paper_ledger.settle_position(filled_qty=1, acquisition_cost=.5, selected_team='YES', winning_team=None, settlement_px=value)


@pytest.mark.parametrize('classification', ['last_fair_market_price_or_unresolved', 'binary_yes', 'unknown_pending'])
def test_evidence_does_not_override_other_settlement_classes(classification):
    with pytest.raises(ValueError, match='unambiguous'):
        trade(settlement={'settlement_class': classification})
