# forex-execution

Places and manages trades on OANDA's **practice** server. There is no live-trading
code path in this package at all -- `forex_execution.config.oanda_practice_config`
only ever reads `OANDA_PRACTICE_*` environment variables, so there's no config
mistake that could route an order to a real-money account.

## Setup

```bash
uv sync --extra dev
source ~/environment_variables/oanda_practice.sh  # OANDA_PRACTICE_SERVER/TOKEN/ACCOUNT_ID
```

## Checking the account (read-only, safe to run any time)

```bash
uv run python scripts/account_status.py
```

## Placing an order

Supports both directions (`--side long`/`--side short`), an absolute take-profit
price, an absolute stop-loss price, and a trailing-stop distance (a price distance,
per OANDA's own convention -- not a percentage). `--units` is always a positive
count; `--side` controls direction, so you never have to remember OANDA's own
positive-long/negative-short units convention.

Every script defaults to a **dry run** -- it prints exactly what it would submit
and makes no network call. Pass `--yes` to actually submit.

```bash
# Dry run -- prints the order, submits nothing
uv run python scripts/place_order.py --instrument EUR/USD --side long --units 100 \
    --take-profit 1.10500 --stop-loss 1.09500

# Actually submit it
uv run python scripts/place_order.py --instrument EUR/USD --side long --units 100 \
    --take-profit 1.10500 --stop-loss 1.09500 --yes

# Short, with a trailing stop instead of a fixed take-profit/stop-loss
uv run python scripts/place_order.py --instrument USD/JPY --side short --units 1000 \
    --trailing-stop-distance 0.30 --yes
```

Orders are `MARKET`/`FOK` (fill-or-kill -- no partial fill left resting). OANDA's
v20 API can return HTTP 201 even when it cancels the order it just created (e.g.
`MARKET_HALTED`, `INSUFFICIENT_MARGIN`) -- `place_market_order` checks the response
body for `orderCancelTransaction` and raises `OandaOrderRejected` in that case rather
than reporting a false success.

## Closing a position

```bash
uv run python scripts/close_position.py --instrument EUR/USD --side long --yes
uv run python scripts/close_position.py --instrument EUR/USD --side long --units 50 --yes  # partial
```

## Tests

```bash
uv run pytest
```

All tests mock the HTTP layer (`_get`/`_post`/`_put`) -- none of them make a real
network call.
