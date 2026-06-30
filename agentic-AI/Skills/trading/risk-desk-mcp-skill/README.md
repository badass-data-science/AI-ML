# The Risk Desk

An MCP skill that wraps a [CLIPS](https://www.clipsrules.net/) expert system to act as an auditable risk gatekeeper for forex LLM agents.

When an LLM agent proposes a trade, The Risk Desk evaluates it against a set of explicit, human-readable rules using forward-chaining inference. Every decision comes with a complete rule-firing trace — no black box.

---

## How it works

```
LLM agent proposes trade (natural language)
    → structured fact extraction
        → The Risk Desk (this MCP skill)
            ← get_market_regime   → VIX level + fuzzy regime classification
            ← get_account_state   → balance, weekly drawdown, open positions
            ← get_pair_liquidity  → live spread in pips + active session
        → CLIPS inference engine fires rules
    → APPROVED / BLOCKED / MODIFIED + rule-firing trace
→ LLM generates plain-English explanation for the trader
```

The three data-fetching tools (`get_market_regime`, `get_account_state`, `get_pair_liquidity`) are designed to be called first to gather live market context, and their outputs passed directly into `evaluate_trade`. The CLIPS engine then runs forward-chaining inference — every rule that fires produces a `Verdict`, and the overall result is the most severe verdict across all of them.

---

## MCP tools

### `get_market_regime() → MarketRegime`

Fetches the current VIX close price (via yfinance) and classifies it into one of four regime levels using fuzzy set membership — the same fuzzy set definitions used by the `vix-fuzzy-mcp-skill`, so regime classification is consistent across the trading skill suite.

| Regime | VIX range (approx.) | Dominant fuzzy set |
|--------|--------------------|--------------------|
| calm | < 13 | very low / low |
| normal | 13–21 | medium low / medium |
| elevated | 21–35 | medium high / high |
| crisis | > 35 | very high |

Returns a `MarketRegime` that can be passed directly to `evaluate_trade`.

---

### `get_account_state() → AccountState`

Fetches the current Oanda account summary and aggregates all realized P&L since Monday 00:00 UTC to compute the weekly drawdown percentage.

Returns an `AccountState` containing:
- `balance` — current account balance
- `weekly_drawdown_pct` — realized drawdown this week as a positive percentage
- `open_positions` — number of currently open positions

Requires `OANDA_API_TOKEN` and `OANDA_ACCOUNT_ID` environment variables.

---

### `get_pair_liquidity(pair) → PairLiquidity`

Fetches the live bid/ask spread for a major pair from the Oanda v20 pricing endpoint, expresses it in pips, and infers the current trading session from UTC time using standard interbank hours:

| Session | UTC hours |
|---------|-----------|
| Tokyo | 00:00–07:00 |
| London | 07:00–12:00 |
| Overlap (London + New York) | 12:00–16:00 |
| New York | 16:00–21:00 |
| Sydney | 21:00–00:00 |

Returns a `PairLiquidity` that can be passed directly to `evaluate_trade`.

---

### `evaluate_trade(trade, regime, account, liquidity) → RiskAssessment`

The core evaluation tool. Asserts the four inputs as facts into a CLIPS environment, runs the forward-chaining inference engine, and returns a structured risk assessment.

**Inputs:**
- `trade` — pair, direction (long/short), and position size as a percentage of account balance
- `regime` — from `get_market_regime()`
- `account` — from `get_account_state()`
- `liquidity` — from `get_pair_liquidity(pair)`

**Output — `RiskAssessment`:**
- `overall` — `APPROVED`, `BLOCKED`, or `MODIFIED` (precedence: BLOCKED > MODIFIED > APPROVED)
- `verdicts` — list of `Verdict` objects, one per fired rule, each with `rule_id`, `result`, `reason`, and `severity`
- `summary` — human-readable one-liner with fired rule count and overall verdict

An empty `verdicts` list means the trade passed all checks cleanly.

---

## Rule categories

| Category | Rule ID | Verdict | Severity | Condition |
|----------|---------|---------|----------|-----------|
| Regime-gated | REGIME-001a | BLOCKED | critical | Long a risk-on pair when VIX regime is crisis |
| Regime-gated | REGIME-001b | BLOCKED | critical | Short a risk-on pair when VIX regime is crisis |
| Regime-gated | REGIME-002 | MODIFIED | warning | Position size > 1.0% when VIX regime is elevated |
| Circuit breaker | DD-001 | BLOCKED | critical | Weekly drawdown ≥ 5% |
| Circuit breaker | DD-002 | APPROVED | warning | Weekly drawdown ≥ 3% (approaching circuit breaker) |
| Liquidity | LIQ-001 | BLOCKED | critical | Spread > 3.0 pips |
| Liquidity | LIQ-002 | APPROVED | info | EUR/USD or GBP/USD during Tokyo session |
| Concentration | CONC-001 | BLOCKED | critical | 5 or more open positions |
| Concentration | CONC-002 | APPROVED | warning | 3–4 open positions (approaching limit) |

### Pair risk characters

The regime rules use a static risk character per pair and direction to determine whether a trade increases or decreases portfolio risk under market stress.

| Pair | Long | Short |
|------|------|-------|
| EUR/USD | risk-on | risk-off |
| GBP/USD | risk-on | risk-off |
| AUD/USD | risk-on | risk-off |
| NZD/USD | risk-on | risk-off |
| USD/JPY | risk-on | risk-off |
| USD/CHF | neutral | risk-off |
| USD/CAD | risk-off | risk-on |

USD/CAD is the notable inversion: buying USD against a commodity currency is risk-off. USD/CHF long is neutral because both currencies carry safe-haven characteristics; the short (buying CHF) is the cleaner risk-off signal.

---

## Dependencies

- [`clipspy`](https://clipspy.readthedocs.io/) — Python bindings for the CLIPS expert system shell
- [`mcp`](https://github.com/modelcontextprotocol/python-sdk) — Model Context Protocol server SDK
- [`pydantic`](https://docs.pydantic.dev/) — structured input/output validation
- [`oandapyV20`](https://github.com/hootnot/oanda-api-v20) — Oanda v20 REST API client
- [`yfinance`](https://github.com/ranaroussi/yfinance) — VIX price data
- [`python-tools-and-shortcuts`](https://github.com/badass-data-science/python-tools-and-shortcuts) — fuzzy logic interpolation utilities

---

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv pip install -e .
```

To also install test dependencies:

```bash
uv pip install -e .[dev]
```

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OANDA_API_TOKEN` | yes | — | Oanda v20 API bearer token |
| `OANDA_ACCOUNT_ID` | yes | — | Oanda account ID (e.g. `001-001-XXXXXXX-001`) |
| `OANDA_ENVIRONMENT` | no | `practice` | `practice` or `live` |

---

## Running from the command line

The server communicates over stdio (standard MCP transport). Running it directly is useful for smoke-testing the install:

```bash
uv run risk-desk-mcp-server
```

The server will start and wait for MCP messages on stdin. Use Ctrl-C to exit. In normal use you won't run it directly — your MCP client (Claude Desktop, OpenClaw, etc.) starts and manages the process.

---

## Adding to OpenClaw

Open your OpenClaw MCP server configuration file and add an entry under `mcpServers`:

```json
{
  "mcpServers": {
    "risk-desk": {
      "command": "uv",
      "args": ["run", "risk-desk-mcp-server"],
      "cwd": "/path/to/risk-desk-mcp-skill",
      "env": {
        "OANDA_API_TOKEN": "your-token-here",
        "OANDA_ACCOUNT_ID": "001-001-XXXXXXX-001",
        "OANDA_ENVIRONMENT": "practice"
      }
    }
  }
}
```

Replace `/path/to/risk-desk-mcp-skill` with the absolute path to this directory, and fill in your Oanda credentials. The `OANDA_ENVIRONMENT` key can be omitted if you are using a practice account (it defaults to `practice`).

After saving the config, restart OpenClaw. The Risk Desk tools will appear as `risk-desk/get_market_regime`, `risk-desk/get_account_state`, `risk-desk/get_pair_liquidity`, and `risk-desk/evaluate_trade`.

---

## Running the tests

```bash
uv pip install -e .[dev]
pytest
```

The test suite covers all eight rule categories (both fire and no-fire cases, including exact threshold boundaries), the pure helper functions, overall verdict precedence, and the summary string format. Tests require a working install of all dependencies including `clipspy`.

---

## Project structure

```
risk_desk_mcp_server/
    __init__.py              # MCP server, tools, and data-fetching helpers
    clips_rules/
        templates.clp        # CLIPS fact templates
        pair_profiles.clp    # Static risk characters for the seven major pairs
        regime_rules.clp     # REGIME-001a/b, REGIME-002
        drawdown_rules.clp   # DD-001, DD-002
        liquidity_rules.clp  # LIQ-001, LIQ-002
        concentration_rules.clp  # CONC-001, CONC-002
tests/
    test_risk_desk.py        # pytest test suite
example_mcp_config.json      # Minimal MCP client config snippet
pyproject.toml
```

---

*This tool is informational only and is not financial advice.*
