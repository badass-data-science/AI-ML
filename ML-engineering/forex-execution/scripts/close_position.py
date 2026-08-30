"""Closes all (or part) of an open position on the OANDA practice account.

Usage:
    uv run python scripts/close_position.py --instrument EUR/USD --side long --yes
    uv run python scripts/close_position.py --instrument EUR/USD --side long --units 50 --yes
"""

from __future__ import annotations

import argparse
import json
from typing import Literal

from forex_execution.client import OandaPracticeClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--instrument', required=True, help='e.g. EUR/USD')
    parser.add_argument('--side', required=True, choices=['long', 'short'])
    parser.add_argument('--units', type=int, default=None, help='Omit to close the entire position')
    parser.add_argument('--yes', action='store_true', help='Actually submit the close. Without this: dry run only.')
    args = parser.parse_args()

    units: int | Literal['ALL'] = 'ALL' if args.units is None else args.units
    print(f'{"SUBMITTING" if args.yes else "DRY RUN (pass --yes to actually submit)"}:')
    print(f'  close {units} units of {args.side.upper()} {args.instrument} (practice account)')

    if not args.yes:
        return

    client = OandaPracticeClient()
    rj = client.close_position(instrument=args.instrument, side=args.side, units=units)
    print(json.dumps(rj, indent=2))


if __name__ == '__main__':
    main()
