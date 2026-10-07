"""Never relabel an explicitly different provider market as the requested one."""
import pytest
from mlb_metrics.venues import polymarket_us as venue


def parse(payload):
    return venue.parse_book_response(payload, market_id='123', market_slug='requested',
        request_time_utc='2026-10-07T16:00:00Z', receive_time_utc='2026-10-07T16:00:01Z',
        fee=venue.select_fee_schedule(as_of_utc='2026-10-07T16:00:01Z'))


def body():
    return {'bids':[{'px':.4,'qty':2}], 'offers':[{'px':.5,'qty':2}], 'state':'MARKET_STATE_OPEN'}


@pytest.mark.parametrize('slug', ['other', '', None, 123])
@pytest.mark.parametrize('wrapped', [True, False])
def test_explicit_wrong_or_invalid_response_identity_rejected(slug, wrapped):
    row=body() | {'marketSlug':slug}
    with pytest.raises(ValueError, match='marketSlug'):
        parse({'marketData':row} if wrapped else row)


@pytest.mark.parametrize('wrapped', [True, False])
def test_matching_identity_preserves_prices(wrapped):
    row=body() | {'marketSlug':'requested'}
    book=parse({'marketData':row} if wrapped else row)
    assert book.market_slug=='requested'
    assert book.best_ask==.5 and book.best_ask_size==2


def test_legacy_absent_identity_does_not_assert_provider_verification():
    book=parse({'marketData':body()})
    assert book.market_slug=='requested'
    # Legacy request-bound parsing remains supported; no independent identity
    # certification field is fabricated for a response that omitted the slug.
    assert 'provider_identity_verified' not in book.to_dict()
