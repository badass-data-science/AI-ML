from __future__ import annotations

import datetime
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

import clips
import oandapyV20
import oandapyV20.endpoints.accounts as v20_accounts
import oandapyV20.endpoints.pricing as v20_pricing
import oandapyV20.endpoints.transactions as v20_transactions
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from python_tools_and_shortcuts.ai.fuzzylogic.FuzzyInterpolator import FuzzyInterpolator
from python_tools_and_shortcuts.econometrics.ticker_prices import get_most_recent_ticker_close_value

# ---------------------------------------------------------------------------
# VIX fuzzy configuration — mirrors vix-fuzzy-mcp-skill exactly so regime
# classification is consistent across both skills
# ---------------------------------------------------------------------------

_VIX_SET_NAMES = ['very low', 'low', 'medium low', 'medium', 'medium high', 'high', 'very high']

_VIX_MEMBERSHIP_RANGES = {
    'very low':   [9.140000343322754,  12.869999885559082],
    'low':        [9.140000343322754,  12.869999885559082, 15.0600004196167],
    'medium low': [12.869999885559082, 15.0600004196167,   17.450000762939453],
    'medium':     [15.0600004196167,   17.450000762939453, 20.649999618530273],
    'medium high':[17.450000762939453, 20.649999618530273, 25.110000610351562],
    'high':       [20.649999618530273, 25.110000610351562, 82.69000244140625],
    'very high':  [25.110000610351562, 82.69000244140625],
}

# Maps from dominant fuzzy set to the four Risk Desk regime levels
_FUZZY_SET_TO_REGIME: dict[str, str] = {
    'very low':    'calm',
    'low':         'calm',
    'medium low':  'normal',
    'medium':      'normal',
    'medium high': 'elevated',
    'high':        'elevated',
    'very high':   'crisis',
}


def _dominant_vix_set(memberships: dict[str, float]) -> str:
    """Return the fuzzy set with the highest membership degree.
    Ties are broken toward the higher-indexed (more conservative) set."""
    return max(memberships, key=lambda s: (memberships[s], _VIX_SET_NAMES.index(s)))


# ---------------------------------------------------------------------------
# Oanda account helpers — mirrors oanda-account-mcp-skill exactly
# ---------------------------------------------------------------------------

def _oanda_client() -> tuple[oandapyV20.API, str]:
    token       = os.environ["OANDA_API_TOKEN"]
    account_id  = os.environ["OANDA_ACCOUNT_ID"]
    environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
    return oandapyV20.API(access_token=token, environment=environment), account_id


def _week_start_rfc3339() -> str:
    today  = datetime.date.today()
    monday = today - datetime.timedelta(days=today.weekday())
    dt     = datetime.datetime(monday.year, monday.month, monday.day,
                               tzinfo=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _weekly_realized_pl(client: oandapyV20.API, account_id: str) -> float:
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


def _pair_to_oanda(pair: str) -> str:
    return pair.replace("/", "_")


def _current_session() -> str:
    hour = datetime.datetime.now(datetime.timezone.utc).hour
    if hour < 7:
        return "tokyo"
    if hour < 12:
        return "london"
    if hour < 16:
        return "overlap"
    if hour < 21:
        return "new-york"
    return "sydney"


RULES_DIR = Path(__file__).parent / "clips_rules"

mcp = FastMCP("risk-desk")

# One shared environment — rules are loaded once, reset before each evaluation
_env = clips.Environment()


def _load_rules() -> None:
    _env.load(str(RULES_DIR / "templates.clp"))
    _env.load(str(RULES_DIR / "pair_profiles.clp"))
    _env.load(str(RULES_DIR / "regime_rules.clp"))
    _env.load(str(RULES_DIR / "drawdown_rules.clp"))
    _env.load(str(RULES_DIR / "liquidity_rules.clp"))
    _env.load(str(RULES_DIR / "concentration_rules.clp"))


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

MAJOR_PAIRS = Literal["EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD", "USD/JPY", "USD/CHF", "USD/CAD"]
DIRECTION   = Literal["long", "short"]
REGIME      = Literal["calm", "normal", "elevated", "crisis"]
SESSION     = Literal["london", "new-york", "tokyo", "sydney", "overlap"]


class TradeProposal(BaseModel):
    pair:      MAJOR_PAIRS
    direction: DIRECTION
    size_pct:  float = Field(gt=0, description="Position size as a percentage of account balance")


class MarketRegime(BaseModel):
    vix_level: float  = Field(gt=0, description="Most recent VIX close value")
    regime:    REGIME = Field(description="Fuzzy regime classification from vix-fuzzy-mcp-skill")


class AccountState(BaseModel):
    balance:             float = Field(gt=0)
    weekly_drawdown_pct: float = Field(ge=0, description="Realised drawdown this week as a positive percentage")
    open_positions:      int   = Field(ge=0)


class PairLiquidity(BaseModel):
    pair:        MAJOR_PAIRS
    spread_pips: float  = Field(gt=0)
    session:     SESSION


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    result:   Literal["APPROVED", "BLOCKED", "MODIFIED"]
    rule_id:  str
    reason:   str
    severity: Literal["info", "warning", "critical"]


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")

    overall:  Literal["APPROVED", "BLOCKED", "MODIFIED"]
    verdicts: list[Verdict]
    summary:  str


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_market_regime() -> MarketRegime:
    """
    Fetch the current VIX value and classify it into a market regime using
    fuzzy set membership.

    Uses the same fuzzy set definitions as the vix-fuzzy-mcp-skill so that
    regime classification is consistent across the trading skill suite.

    Regime levels:
    - calm:     dominant fuzzy set is 'very low' or 'low'     (VIX roughly < 13)
    - normal:   dominant fuzzy set is 'medium low' or 'medium' (VIX roughly 13–21)
    - elevated: dominant fuzzy set is 'medium high' or 'high'  (VIX roughly 21–35)
    - crisis:   dominant fuzzy set is 'very high'              (VIX roughly > 35)

    Returns a MarketRegime that can be passed directly to evaluate_trade().

    This tool is informational only and is not financial advice.
    """
    vix = get_most_recent_ticker_close_value('^VIX')

    if not math.isfinite(vix):
        raise RuntimeError(f"Invalid VIX value returned: {vix!r}")
    if vix < 0.0:
        raise RuntimeError(f"Unexpected negative VIX value: {vix!r}")

    fi = FuzzyInterpolator(_VIX_SET_NAMES, _VIX_MEMBERSHIP_RANGES)
    memberships = fi.interpolate_membership(vix)
    dominant = _dominant_vix_set(memberships)
    regime = _FUZZY_SET_TO_REGIME[dominant]

    return MarketRegime(vix_level=vix, regime=regime)


@mcp.tool()
def get_account_state() -> AccountState:
    """
    Retrieve current Oanda account state for use with evaluate_trade().

    Fetches the account summary via the Oanda v20 REST API and aggregates
    realized P&L from all transactions since Monday 00:00:00 UTC to compute
    the current weekly drawdown.

    Required environment variables:
        OANDA_API_TOKEN    — v20 API bearer token
        OANDA_ACCOUNT_ID   — account ID (e.g. 001-001-XXXXXXX-001)

    Optional environment variables:
        OANDA_ENVIRONMENT  — 'practice' or 'live'  (default: 'practice')

    This tool is informational only and is not financial advice.
    """
    client, account_id = _oanda_client()

    r = v20_accounts.AccountSummary(account_id)
    client.request(r)
    summary = r.response["account"]

    balance        = float(summary["balance"])
    open_positions = int(summary.get("openPositionCount", 0))

    weekly_pl = _weekly_realized_pl(client, account_id)

    if weekly_pl < 0.0:
        week_open_balance   = balance - weekly_pl
        weekly_drawdown_pct = round(abs(weekly_pl) / week_open_balance * 100.0, 4)
    else:
        weekly_drawdown_pct = 0.0

    return AccountState(
        balance=balance,
        weekly_drawdown_pct=weekly_drawdown_pct,
        open_positions=open_positions,
    )


@mcp.tool()
def get_pair_liquidity(pair: MAJOR_PAIRS) -> PairLiquidity:
    """
    Fetch the current bid/ask spread for a forex major pair and infer the
    active trading session from UTC time.

    Spread is retrieved live from the Oanda v20 pricing endpoint and
    expressed in pips (0.0001 per pip for most pairs; 0.01 for JPY pairs).
    Session is inferred using standard interbank open/close times:

        Tokyo    00:00–07:00 UTC
        London   07:00–12:00 UTC
        Overlap  12:00–16:00 UTC  (London + New York both open)
        New York 16:00–21:00 UTC
        Sydney   21:00–00:00 UTC

    Returns a PairLiquidity that can be passed directly to evaluate_trade().

    Required environment variables:
        OANDA_API_TOKEN   — v20 API bearer token
        OANDA_ACCOUNT_ID  — account ID (e.g. 001-001-XXXXXXX-001)

    Optional environment variables:
        OANDA_ENVIRONMENT — 'practice' or 'live'  (default: 'practice')

    This tool is informational only and is not financial advice.
    """
    client, account_id = _oanda_client()

    r = v20_pricing.PricingInfo(account_id, params={"instruments": _pair_to_oanda(pair)})
    client.request(r)

    price     = r.response["prices"][0]
    bid       = float(price["closeoutBid"])
    ask       = float(price["closeoutAsk"])
    pip_size  = 0.01 if "JPY" in pair else 0.0001
    spread_pips = round((ask - bid) / pip_size, 1)

    return PairLiquidity(pair=pair, spread_pips=spread_pips, session=_current_session())


@mcp.tool()
def evaluate_trade(
    trade:     TradeProposal,
    regime:    MarketRegime,
    account:   AccountState,
    liquidity: PairLiquidity,
) -> RiskAssessment:
    """
    Evaluate a proposed forex trade against The Risk Desk expert system rules.

    Asserts the trade proposal, market regime, account state, and pair liquidity
    as facts into a CLIPS forward-chaining inference engine, runs the engine,
    and returns a structured risk assessment with a complete rule-firing trace.

    Overall result precedence: BLOCKED > MODIFIED > APPROVED.
    An empty verdicts list means the trade passed all checks with no issues.

    This tool is informational only and is not financial advice.
    """
    _env.reset()

    _env.assert_string(
        f'(trade-proposal'
        f' (pair "{trade.pair}")'
        f' (direction {trade.direction})'
        f' (size-pct {trade.size_pct}))'
    )
    _env.assert_string(
        f'(market-regime'
        f' (vix-level {regime.vix_level})'
        f' (regime {regime.regime}))'
    )
    _env.assert_string(
        f'(account-state'
        f' (balance {account.balance})'
        f' (weekly-drawdown-pct {account.weekly_drawdown_pct})'
        f' (open-positions {account.open_positions}))'
    )
    _env.assert_string(
        f'(pair-liquidity'
        f' (pair "{liquidity.pair}")'
        f' (spread-pips {liquidity.spread_pips})'
        f' (session {liquidity.session}))'
    )

    _env.run()

    verdicts: list[Verdict] = []
    for fact in _env.facts():
        if fact.template.name == "risk-verdict":
            verdicts.append(Verdict(
                result=str(fact["result"]),
                rule_id=str(fact["rule-id"]),
                reason=str(fact["reason"]),
                severity=str(fact["severity"]),
            ))

    results = {v.result for v in verdicts}
    if "BLOCKED" in results:
        overall = "BLOCKED"
    elif "MODIFIED" in results:
        overall = "MODIFIED"
    else:
        overall = "APPROVED"

    fired = len(verdicts)
    summary = (
        f"{fired} rule{'s' if fired != 1 else ''} fired. "
        f"Overall verdict: {overall}."
    )

    return RiskAssessment(overall=overall, verdicts=verdicts, summary=summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run The Risk Desk MCP server over stdio."""
    _load_rules()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
