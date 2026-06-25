# Fuzzy VIX Market Indicator MCP Skill

## Purpose

This skill exposes an MCP-compatible tool that retrieves the most recent VIX indicator close value from Yahoo Finance and specifies its degree of membership in ordered fuzzy sets describing the VIX value's magnitude.

## Tool

### `get_fuzzy_set_membership_of_most_recent_vix`

Retrieves the most recent VIX indicator close value and specifies its degree of membership in ordered fuzzy sets describing the VIX value's magnitude. The ordinal fuzzy set names increase linguistically with respect to increase in VIX.

## Output

Returns a structured object containing:

```json
{
  "VIX ordinal increasing fuzzy set names": [
    "very low",
    "low",
    "medium low",
    "medium",
    "medium high",
    "high",
    "very high"
  ],
  "VIX fuzzy set membership": {
    "very low": 0.0,
    "low": 0.0,
    "medium low": 0.0,
    "medium": 0.4031248342245224,
    "medium high": 0.5968751657754776,
    "high": 0.0,
    "very high": 0.0
  }
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
uv run vix_fuzzy_mcp_server.py
```
