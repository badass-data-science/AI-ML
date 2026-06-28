# Oanda Account MCP Skill

An MCP skill for querying live Oanda account state via the [v20 REST API](https://developer.oanda.com/rest-live-v20/introduction/).

Returns account balance, NAV, open position count, and weekly realized drawdown — the three inputs The Risk Desk expert system needs to evaluate circuit breaker rules.

## MCP tool

**`get_account_state()`** — no arguments required. Returns:

| Field | Description |
|-------|-------------|
| `balance` | Realized account balance (excludes unrealized P&L) |
| `nav` | Net asset value including unrealized P&L on open positions |
| `weekly_pl` | Realized P&L since Monday 00:00:00 UTC |
| `weekly_drawdown_pct` | Weekly loss as a percentage of week-opening balance; 0 if profitable |
| `open_positions` | Number of currently open positions |
| `currency` | Account base currency |

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OANDA_API_TOKEN` | yes | — | v20 API bearer token |
| `OANDA_ACCOUNT_ID` | yes | — | Account ID (e.g. `001-001-XXXXXXX-001`) |
| `OANDA_ENVIRONMENT` | no | `practice` | `practice` or `live` |

## Installation

```bash
uv pip install -e .
```

## Usage with The Risk Desk

```
get_account_state()       → AccountState
get_market_regime()       → MarketRegime      # from risk-desk-mcp-skill
evaluate_trade(...)       → RiskAssessment    # from risk-desk-mcp-skill
```

See `example_mcp_config.json` for Claude Desktop / OpenClaw configuration.
