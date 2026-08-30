from __future__ import annotations

import pytest

from forex_execution.client import OandaOrderRejected, OandaPracticeClient, _signed_units
from forex_execution.config import oanda_practice_config


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(oanda_practice_config, 'OANDA_PRACTICE_SERVER', 'https://example.test', raising=False)
    monkeypatch.setattr(oanda_practice_config, 'OANDA_PRACTICE_TOKEN', 'fake-token', raising=False)
    monkeypatch.setattr(oanda_practice_config, 'OANDA_PRACTICE_ACCOUNT_ID', '101-001-1234567-001', raising=False)
    return OandaPracticeClient()


def test_signed_units_long_stays_positive():
    assert _signed_units('long', 100) == 100


def test_signed_units_short_goes_negative():
    assert _signed_units('short', 100) == -100


def test_signed_units_rejects_non_positive_input():
    with pytest.raises(ValueError, match='positive'):
        _signed_units('long', 0)
    with pytest.raises(ValueError, match='positive'):
        _signed_units('short', -5)


def test_place_market_order_builds_long_order_body(client, monkeypatch):
    captured = {}

    def fake_post(path, body):
        captured['path'] = path
        captured['body'] = body
        return {'orderFillTransaction': {'id': '42'}}

    monkeypatch.setattr(client, '_post', fake_post)
    client.place_market_order('EUR/USD', side='long', units=100, take_profit_price=1.105, stop_loss_price=1.095)

    assert captured['path'] == '/orders'
    order = captured['body']['order']
    assert order['instrument'] == 'EUR_USD'
    assert order['units'] == '100'
    assert order['type'] == 'MARKET'
    assert order['timeInForce'] == 'FOK'
    assert order['takeProfitOnFill'] == {'price': '1.10500'}
    assert order['stopLossOnFill'] == {'price': '1.09500'}
    assert 'trailingStopLossOnFill' not in order


def test_place_market_order_builds_short_order_with_negative_units(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client, '_post', lambda path, body: captured.update(body=body) or {'orderFillTransaction': {}})
    client.place_market_order('USD/JPY', side='short', units=1000, trailing_stop_distance=0.30)

    order = captured['body']['order']
    assert order['units'] == '-1000'
    assert order['trailingStopLossOnFill'] == {'distance': '0.30000'}


def test_place_market_order_raises_on_order_cancel_transaction(client, monkeypatch):
    monkeypatch.setattr(
        client, '_post',
        lambda path, body: {'orderCancelTransaction': {'reason': 'MARKET_HALTED'}},
    )
    with pytest.raises(OandaOrderRejected, match='MARKET_HALTED'):
        client.place_market_order('EUR/USD', side='long', units=100)


def test_close_position_long_sets_long_units(client, monkeypatch):
    captured = {}

    def fake_put(path, body):
        captured['path'] = path
        captured['body'] = body
        return {}

    monkeypatch.setattr(client, '_put', fake_put)
    client.close_position('EUR/USD', side='long')

    assert captured['path'] == '/positions/EUR_USD/close'
    assert captured['body'] == {'longUnits': 'ALL'}


def test_close_position_short_sets_short_units_and_supports_partial_close(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(client, '_put', lambda path, body: captured.update(body=body) or {})
    client.close_position('USD/JPY', side='short', units=50)

    assert captured['body'] == {'shortUnits': '50'}
