#
# Load useful libraries
#
from __future__ import annotations

import math
from typing import Literal

import yfinance as yf
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

#
# Declare global color types
#
Color = Literal["green", "yellow", "red"]

#
# Initiate an MCP server object
#
mcp = FastMCP("vix-market-indicator")

#
# Define a schema for tool results
#
class VixResult(BaseModel):
    """Structured result returned by the VIX MCP tool."""

    symbol: str = Field(description="Ticker symbol used to retrieve VIX data.")
    period: str = Field(description="Yahoo Finance lookback period used for retrieval.")
    vix: float = Field(description="Most recent VIX close value.")
    color_based_interpretation: Color = Field(
        description="Simple volatility status indicator: green, yellow, or red."
    )
    interpretation: str = Field(description="Plain-English interpretation of the VIX level.")

#
# Define a function for classifying VIX values by warning color
#
def classify_vix(vix: float) -> Color:
    """Classify VIX into a simple color-based volatility regime."""

    if vix >= 30.0:
        return "red"
    if vix >= 20.0:
        return "yellow"
    return "green"

#
# Provide a prose description of each color warning
#
def explain_color(color: Color) -> str:
    """Return a simple human-readable explanation for the color classification."""

    explanations = {
        "green": "Lower-volatility environment based on this simple threshold model.",
        "yellow": "Elevated-volatility environment based on this simple threshold model.",
        "red": "High-volatility environment based on this simple threshold model.",
    }
    return explanations[color]


@mcp.tool()
def get_most_recent_vix(period: str = "7d") -> VixResult:
    """
    Retrieve the most recent VIX close and classify it using a simple color indicator.

    Args:
        period: Yahoo Finance history period to request. Common examples include
            '1d', '5d', '7d', '1mo', '3mo', '6mo', and '1y'.

    Returns:
        A structured VIX result containing the latest close, color indicator,
        and plain-English interpretation.

    Notes:
        This tool is informational only and is not financial advice.
    """

    allowed_periods = {"1d", "5d", "7d", "1mo", "3mo", "6mo", "1y"}

    if period not in allowed_periods:
        raise ValueError(
            f"Unsupported period={period!r}. "
            f"Allowed values are: {sorted(allowed_periods)}"
        )

    symbol = "^VIX"
    history = yf.Ticker(symbol).history(period=period)

    if history.empty or "Close" not in history.columns:
        raise RuntimeError(f"No VIX close data returned for period={period!r}.")

    closes = history["Close"].dropna()

    if closes.empty:
        raise RuntimeError("VIX close data was returned but contained no valid values.")

    vix = float(closes.iloc[-1])

    if not math.isfinite(vix):
        raise RuntimeError(f"Invalid VIX value returned: {vix!r}.")

    if vix < 0.0:
        raise RuntimeError(f"Unexpected negative VIX value returned: {vix!r}.")

    color = classify_vix(vix)

    return VixResult(
        symbol=symbol,
        period=period,
        vix=vix,
        color_based_interpretation=color,
        interpretation=explain_color(color),
    )


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
