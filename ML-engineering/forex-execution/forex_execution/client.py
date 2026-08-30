from __future__ import annotations

import logging
from typing import Literal

import requests
from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_fixed

from forex_execution.config import oanda_practice_config
from forex_execution.headers import get_oanda_headers

logger = logging.getLogger(__name__)

Side = Literal['long', 'short']


class OandaOrderRejected(RuntimeError):
    """Raised when OANDA accepts the HTTP request (2xx) but reports the order itself
    was cancelled rather than filled -- e.g. MARKET_HALTED, INSUFFICIENT_MARGIN. OANDA's
    v20 API returns 201 for both outcomes, so a bare raise_for_status() would miss this."""


def _is_not_client_error(exc: BaseException) -> bool:
    """True unless `exc` is an HTTPError with a 4xx status -- those are deterministic
    (bad instrument, bad units, bad auth), not worth retrying for the same guaranteed
    outcome. Mirrors ETL-forex-time-series-data's SwapRateETL/CandlestickETL convention."""
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return True
    return not (400 <= exc.response.status_code < 500)


def _signed_units(side: Side, units: int) -> int:
    if units <= 0:
        raise ValueError(f'units must be positive (side controls direction): got {units}')
    return units if side == 'long' else -units


class OandaPracticeClient:
    """Thin wrapper around OANDA's v20 REST API, hardcoded to the practice server
    (see forex_execution.config.oanda_practice_config) -- this class has no notion of
    a live server at all, so there's no config path that could accidentally place a
    real-money trade."""

    def __init__(self) -> None:
        self.server = oanda_practice_config.OANDA_PRACTICE_SERVER
        self.account_id = oanda_practice_config.OANDA_PRACTICE_ACCOUNT_ID
        self.headers = get_oanda_headers()

    def _accounts_url(self, path: str) -> str:
        return f'{self.server}/v3/accounts/{self.account_id}{path}'

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.RequestException) & retry_if_exception(_is_not_client_error),
        reraise=True,
    )
    def _get(self, path: str) -> dict:
        r = requests.get(self._accounts_url(path), headers=self.headers)
        r.raise_for_status()
        return r.json()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.RequestException) & retry_if_exception(_is_not_client_error),
        reraise=True,
    )
    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(self._accounts_url(path), headers=self.headers, json=body)
        r.raise_for_status()
        return r.json()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.RequestException) & retry_if_exception(_is_not_client_error),
        reraise=True,
    )
    def _put(self, path: str, body: dict) -> dict:
        r = requests.put(self._accounts_url(path), headers=self.headers, json=body)
        r.raise_for_status()
        return r.json()

    def get_account_summary(self) -> dict:
        return self._get('/summary')

    def get_open_positions(self) -> dict:
        return self._get('/openPositions')

    def get_open_trades(self) -> dict:
        return self._get('/openTrades')

    def place_market_order(
        self,
        instrument: str,
        side: Side,
        units: int,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        trailing_stop_distance: float | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Places a MARKET order (FOK -- fill-or-kill, no partial fills left resting).
        `units` is always a positive count; `side` controls direction, so callers never
        have to remember OANDA's own positive-long/negative-short units convention.
        take_profit_price/stop_loss_price are absolute prices; trailing_stop_distance is
        a price distance (OANDA's own convention), not a percentage."""
        order: dict = {
            'type': 'MARKET',
            'instrument': instrument.replace('/', '_'),
            'units': str(_signed_units(side, units)),
            'timeInForce': 'FOK',
            'positionFill': 'DEFAULT',
        }
        if take_profit_price is not None:
            order['takeProfitOnFill'] = {'price': f'{take_profit_price:.5f}'}
        if stop_loss_price is not None:
            order['stopLossOnFill'] = {'price': f'{stop_loss_price:.5f}'}
        if trailing_stop_distance is not None:
            order['trailingStopLossOnFill'] = {'distance': f'{trailing_stop_distance:.5f}'}
        if client_order_id is not None:
            order['clientExtensions'] = {'id': client_order_id}

        rj = self._post('/orders', {'order': order})
        if 'orderCancelTransaction' in rj:
            reason = rj['orderCancelTransaction'].get('reason', 'UNKNOWN')
            raise OandaOrderRejected(f'{instrument} {side} {units}: order created but cancelled by OANDA ({reason})')
        logger.info('Filled %s %s %d', side, instrument, units)
        return rj

    def close_position(self, instrument: str, side: Side, units: int | Literal['ALL'] = 'ALL') -> dict:
        """Closes all or part of an open position in one direction. OANDA scopes long
        and short units separately even for the same instrument (hedging accounts can
        hold both at once), hence the required `side`."""
        units_str = str(units) if units == 'ALL' else str(abs(int(units)))
        body = {'longUnits': units_str} if side == 'long' else {'shortUnits': units_str}
        return self._put(f'/positions/{instrument.replace("/", "_")}/close', body)
