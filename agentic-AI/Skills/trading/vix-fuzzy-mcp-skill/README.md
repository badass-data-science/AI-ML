# Fuzzy VIX MCP Skill

A small MCP-compatible AI agent skill that retrieves the most recent VIX indicator close value from Yahoo Finance and specifies its degree of membership in ordered fuzzy sets describing the VIX value's magnitude. The ordinal fuzzy set names increase linguistically with respect to increase in VIX.

## What this provides

This package exposes one MCP tool:

```text
get_fuzzy_set_membership_of_most_recent_vix()
```

The tool returns structured data like:

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

## Installation

Using `uv`:

```bash
cd vix-fuzzy-mcp-skill
uv sync
```

Or using `pip`:

```bash
cd vix-fuzzy-mcp-skill
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the MCP server

```bash
uv run vix_fuzzy_mcp_server.py
```

Or, after installing with `pip`:

```bash
vix-fuzzy-mcp-server
```

This server runs over stdio, which is the usual local transport for desktop MCP
clients.

## MCP client configuration example

Use an absolute path for the project directory:

```json
{
  "mcpServers": {
    "vix-fuzzy-indicator": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/vix-fuzzy-mcp-skill",
        "run",
        "vix_fuzzy_mcp_server.py"
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
