"""Read-only: prints the OANDA practice account's summary and any open positions/trades.
Safe to run any time -- makes no state-changing calls.

Usage:
    uv run python scripts/account_status.py
"""

from __future__ import annotations

import json

from forex_execution.client import OandaPracticeClient


def main() -> None:
    client = OandaPracticeClient()

    print('=== Account summary ===')
    print(json.dumps(client.get_account_summary(), indent=2))

    print('\n=== Open positions ===')
    print(json.dumps(client.get_open_positions(), indent=2))

    print('\n=== Open trades ===')
    print(json.dumps(client.get_open_trades(), indent=2))


if __name__ == '__main__':
    main()
