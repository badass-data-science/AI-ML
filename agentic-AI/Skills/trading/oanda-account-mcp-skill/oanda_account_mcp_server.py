from __future__ import annotations

import datetime
import os
from urllib.parse import parse_qs, urlparse

import oandapyV20
import oandapyV20.endpoints.accounts as v20_accounts
import oandapyV20.endpoints.transactions as v20_transactions
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

mcp = FastMCP("oanda-account")


# ---------------------------------------------------------------------------
# Oanda client factory
# ---------------------------------------------------------------------------

def _client() -> tuple[oandapyV20.API, str]:
    """Build an authenticated Oanda v20 API client from environment variables."""
    token       = os.environ["OANDA_API_TOKEN"]
    account_id  = os.environ["OANDA_ACCOUNT_ID"]
    environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
    return oandapyV20.API(access_token=token, environment=environment), account_id


# ---------------------------------------------------------------------------
# Weekly P&L helper
# ---------------------------------------------------------------------------

def _week_start_rfc3339() -> str:
    """RFC3339 timestamp for Monday 00:00:00 UTC of the current week."""
    today  = datetime.date.today()
    monday = today - datetime.timedelta(days=today.weekday())
    dt     = datetime.datetime(monday.year, monday.month, monday.day,
                               tzinfo=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _weekly_realized_pl(client: oandapyV20.API, account_id: str) -> float:
    """
    Sum realized P&L from every transaction since Monday 00:00:00 UTC.

    TransactionList returns page URLs rather than transactions directly.
    Each page is fetched via TransactionIDRange using the from/to IDs
    embedded in the page URL query string.
    """
    params = {"from": _week_start_rfc3339(), "pageSize": 1000}
    r = v20_transactions.TransactionList(account_id, params=params)
    client.request(r)

    total_pl = 0.0

    for page_url in r.response.get("pages", []):
        qs      = parse_qs(urlparse(page_url).query)
        from_id = qs["from"][0]
        to_id   = qs["to"][0]

        r2 = v20_transactions.TransactionIDRange(
            account_id, params={"from": from_id, "to": to_id}
        )
        client.request(r2)

        for tx in r2.response.get("transactions", []):
            if "pl" in tx:
                total_pl += float(tx["pl"])

    return total_pl


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class AccountState(BaseModel):
    model_config = ConfigDict(extra="allow")

    balance:             float = Field(description="Realized account balance, excluding open position P&L")
    weekly_drawdown_pct: float = Field(ge=0.0, description="Realized drawdown this week as a positive percentage; 0 if the week is profitable")
    open_positions:      int   = Field(ge=0,   description="Number of currently open positions")
    nav:                 float = Field(description="Net asset value — balance plus unrealized P&L on open positions")
    weekly_pl:           float = Field(description="Realized P&L for the current week; negative indicates a loss")
    currency:            str   = Field(description="Account base currency (e.g. USD)")


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------

@mcp.tool()
def get_account_state() -> AccountState:
    """
    Retrieve current Oanda account state for use with The Risk Desk expert system.

    Fetches the account summary via the Oanda v20 REST API and aggregates
    realized P&L from all transactions since Monday 00:00:00 UTC to compute
    the current weekly drawdown.

    The returned AccountState can be passed directly to The Risk Desk's
    evaluate_trade() tool.

    Required environment variables:
        OANDA_API_TOKEN    — v20 API bearer token
        OANDA_ACCOUNT_ID   — account ID (e.g. 001-001-XXXXXXX-001)

    Optional environment variables:
        OANDA_ENVIRONMENT  — 'practice' or 'live'  (default: 'practice')

    This tool is informational only and is not financial advice.
    """
    client, account_id = _client()

    r = v20_accounts.AccountSummary(account_id)
    client.request(r)
    summary = r.response["account"]

    balance        = float(summary["balance"])
    nav            = float(summary["NAV"])
    open_positions = int(summary.get("openPositionCount", 0))
    currency       = str(summary["currency"])

    weekly_pl = _weekly_realized_pl(client, account_id)

    if weekly_pl < 0.0:
        week_open_balance   = balance - weekly_pl
        weekly_drawdown_pct = abs(weekly_pl) / week_open_balance * 100.0
    else:
        weekly_drawdown_pct = 0.0

    return AccountState(
        balance=balance,
        weekly_drawdown_pct=round(weekly_drawdown_pct, 4),
        open_positions=open_positions,
        nav=nav,
        weekly_pl=weekly_pl,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Oanda account MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
