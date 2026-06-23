# VIX MCP Skill

A small MCP-compatible AI agent skill that retrieves the latest VIX close from
Yahoo Finance and returns a simple green/yellow/red volatility indicator.

## What this provides

This package exposes one MCP tool:

```text
get_most_recent_vix(period: str = "7d")
```

The tool returns structured data like:

```json
{
  "symbol": "^VIX",
  "period": "7d",
  "vix": 17.42,
  "color_based_interpretation": "green",
  "interpretation": "Lower-volatility environment based on this simple threshold model."
}
```

## Color thresholds

| VIX value | Color |
|---:|---|
| `< 20` | `green` |
| `>= 20` and `< 30` | `yellow` |
| `>= 30` | `red` |

## Installation

Using `uv`:

```bash
cd vix-mcp-skill
uv sync
```

Or using `pip`:

```bash
cd vix-mcp-skill
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the MCP server

```bash
uv run vix_mcp_server.py
```

Or, after installing with `pip`:

```bash
vix-mcp-server
```

This server runs over stdio, which is the usual local transport for desktop MCP
clients.

## MCP client configuration example

Use an absolute path for the project directory:

```json
{
  "mcpServers": {
    "vix-market-indicator": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/vix-mcp-skill",
        "run",
        "vix_mcp_server.py"
      ]
    }
  }
}
```

## Testing with MCP Inspector

```bash
npx -y @modelcontextprotocol/inspector
```

Point the inspector at the local command used to run this server.

## Development notes

- Do not use `print()` for normal logging in a stdio MCP server. Stdout is used
  by MCP's JSON-RPC protocol.
- The tool validates the `period` argument before calling Yahoo Finance.
- The tool rejects empty, missing, NaN, infinite, or negative VIX values.

## Disclaimer

This skill is informational only and is not financial advice.
