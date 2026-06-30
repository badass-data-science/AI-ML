# The Risk Desk

An MCP skill that wraps a [CLIPS](https://www.clipsrules.net/) expert system to act as an auditable risk gatekeeper for forex LLM agents.

When an LLM agent proposes a trade, The Risk Desk evaluates it against a set of explicit, human-readable rules using forward-chaining inference. Every decision comes with a complete rule-firing trace — no black box.

## Architecture

```
LLM agent proposes trade (natural language)
    → structured fact extraction
        → The Risk Desk (this MCP skill)
            ← VIX fuzzy MCP skill     → market regime fact
            ← broker MCP skill        → account state fact
            ← forex data source       → pair liquidity fact
        → CLIPS inference engine fires rules
    → APPROVED / BLOCKED / MODIFIED + rule trace
→ LLM generates plain-English explanation
```

## Rule categories

| Category | Rule IDs | Description |
|----------|----------|-------------|
| Regime-gated | REGIME-001a/b | Block risk-on trades when VIX is in crisis |
| Regime-gated | REGIME-002 | Cap position size at 1.0% during elevated VIX |
| Circuit breaker | DD-001 | Hard halt when weekly drawdown ≥ 5% |
| Circuit breaker | DD-002 | Soft warning when weekly drawdown ≥ 3% |
| Liquidity | LIQ-001 | Block when spread exceeds 3.0 pips |
| Liquidity | LIQ-002 | Warn when trading European pairs in Tokyo session |
| Concentration | CONC-001 | Hard block when 5 or more positions are open |
| Concentration | CONC-002 | Soft warning when 3–4 positions are open |

## Covered pairs

The seven major forex pairs: EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/JPY, USD/CHF, USD/CAD.

Each pair carries a directional risk character (risk-on / risk-off / neutral) that regime rules use to decide whether a proposed trade adds or removes risk relative to prevailing market stress.

## Dependencies

- [`clipspy`](https://clipspy.readthedocs.io/) — Python bindings for the CLIPS expert system shell
- [`mcp`](https://github.com/modelcontextprotocol/python-sdk) — Model Context Protocol server SDK
- [`pydantic`](https://docs.pydantic.dev/) — structured input/output validation

## Installation

```bash
uv pip install -e .
```

## MCP tool

**`evaluate_trade`** — accepts a trade proposal, market regime, account state, and pair liquidity; returns a `RiskAssessment` with `overall` verdict and a list of `Verdict` objects, one per fired rule.

See `example_mcp_config.json` for Claude Desktop / OpenClaw configuration.
