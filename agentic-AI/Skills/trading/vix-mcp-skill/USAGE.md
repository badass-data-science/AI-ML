# Usage

## Start the server

```bash
uv run vix_mcp_server.py
```

## Tool

### `get_most_recent_vix`

Input:

```json
{
  "period": "7d"
}
```

Output:

```json
{
  "symbol": "^VIX",
  "period": "7d",
  "vix": 17.42,
  "color_based_interpretation": "green",
  "interpretation": "Lower-volatility environment based on this simple threshold model."
}
```

## Supported periods

- `1d`
- `5d`
- `7d`
- `1mo`
- `3mo`
- `6mo`
- `1y`

## Financial disclaimer

The output is informational only. It is not investment advice, trading advice,
or a recommendation to buy, sell, or hold any security.
