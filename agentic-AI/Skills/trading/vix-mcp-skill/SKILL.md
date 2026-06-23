# VIX Market Indicator MCP Skill

## Purpose

This skill exposes an MCP-compatible tool that retrieves the most recent VIX close
from Yahoo Finance and returns a simple color-based volatility interpretation.

## Tool

### `get_most_recent_vix`

Retrieves the most recent VIX close value using the `^VIX` ticker and classifies
the result as:

- `green`: VIX < 20
- `yellow`: VIX >= 20 and < 30
- `red`: VIX >= 30

## Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `period` | string | `7d` | Yahoo Finance lookback period. Allowed values: `1d`, `5d`, `7d`, `1mo`, `3mo`, `6mo`, `1y`. |

## Output

Returns a structured object containing:

```json
{
  "symbol": "^VIX",
  "period": "7d",
  "vix": 17.42,
  "color_based_interpretation": "green",
  "interpretation": "Lower-volatility environment based on this simple threshold model."
}
```

## Notes

This skill is informational only. It does not provide investment advice,
trading recommendations, or portfolio management guidance.

## Runtime

Requires Python 3.10+ and the following Python packages:

- `mcp[cli]`
- `yfinance`
- `pydantic`

Run locally with:

```bash
uv run vix_mcp_server.py
```
