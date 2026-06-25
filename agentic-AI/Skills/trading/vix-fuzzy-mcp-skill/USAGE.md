# Usage

## Start the server

```bash
uv run vix_fuzzy_mcp_server.py
```

## Tool

### `get_fuzzy_set_membership_of_most_recent_vix`

Output:

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

## Financial disclaimer

The output is informational only. It is not investment advice, trading advice,
or a recommendation to buy, sell, or hold any security.
