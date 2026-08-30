"""Places a MARKET order against the OANDA practice account configured via
OANDA_PRACTICE_SERVER/OANDA_PRACTICE_TOKEN/OANDA_PRACTICE_ACCOUNT_ID.

Prints the order it's about to submit and requires --yes to actually send it --
without --yes this is a dry run (build + print the request body, no network call).

Usage:
    uv run python scripts/place_order.py --instrument EUR/USD --side long --units 100 \\
        --take-profit 1.10500 --stop-loss 1.09500 --yes

    uv run python scripts/place_order.py --instrument USD/JPY --side short --units 1000 \\
        --trailing-stop-distance 0.30 --yes
"""

from __future__ import annotations

import argparse
import json

from forex_execution.client import OandaOrderRejected, OandaPracticeClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--instrument', required=True, help='e.g. EUR/USD')
    parser.add_argument('--side', required=True, choices=['long', 'short'])
    parser.add_argument('--units', required=True, type=int, help='Always positive; --side controls direction')
    parser.add_argument('--take-profit', type=float, default=None, help='Absolute take-profit price')
    parser.add_argument('--stop-loss', type=float, default=None, help='Absolute stop-loss price')
    parser.add_argument('--trailing-stop-distance', type=float, default=None,
                         help='Trailing-stop distance in price units (not a percentage), e.g. 0.0050 for EUR/USD')
    parser.add_argument('--client-order-id', default=None)
    parser.add_argument('--yes', action='store_true', help='Actually submit the order. Without this: dry run only.')
    args = parser.parse_args()

    print(f'{"SUBMITTING" if args.yes else "DRY RUN (pass --yes to actually submit)"}:')
    print(f'  {args.side.upper()} {args.units} units of {args.instrument} @ MARKET (practice account)')
    if args.take_profit is not None:
        print(f'  take-profit: {args.take_profit}')
    if args.stop_loss is not None:
        print(f'  stop-loss: {args.stop_loss}')
    if args.trailing_stop_distance is not None:
        print(f'  trailing-stop distance: {args.trailing_stop_distance}')

    if not args.yes:
        return

    client = OandaPracticeClient()
    try:
        rj = client.place_market_order(
            instrument=args.instrument,
            side=args.side,
            units=args.units,
            take_profit_price=args.take_profit,
            stop_loss_price=args.stop_loss,
            trailing_stop_distance=args.trailing_stop_distance,
            client_order_id=args.client_order_id,
        )
    except OandaOrderRejected as exc:
        print(f'REJECTED: {exc}')
        raise SystemExit(1) from None

    print(json.dumps(rj, indent=2))


if __name__ == '__main__':
    main()
