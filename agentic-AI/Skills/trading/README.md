# Agent Skills Related to Trading

A collection of homegrown trading-related AI agent skills.

* ***vix-mcp-skill:*** Retrieves the most recent VIX value and assigns a color code to it indicating warning level. The only thing that makes this different from a simple stock ticker lookup is the added color indicator, otherwise we could just have written a tool that looks up any stock ticker because "^VIX" is a ticker symbol for this index.

* ***vix-fuzzy-mcp-skill:*** Retrieves the most recent VIX value and specifies its degree of membership in ordered fuzzy sets describing the VIX value's magnitude (e.g. "medium", "high"), rather than a single color code.

* ***vix_fuzzy_shared:*** Shared VIX fetching and fuzzy set classification logic used by both `vix-fuzzy-mcp-skill` and `risk-desk-mcp-skill`, so the two skills compute VIX regime identically from one source instead of each keeping its own copy.

* ***oanda-account-mcp-skill:*** Queries live Oanda account state (balance, NAV, open position count, weekly realized drawdown) via the Oanda v20 REST API.

* ***risk-desk-mcp-skill ("The Risk Desk"):*** Wraps a CLIPS expert system to act as an auditable risk gatekeeper for forex LLM agents — evaluates proposed trades against explicit rules (regime, drawdown, liquidity, concentration) and returns an APPROVED/BLOCKED/MODIFIED verdict with a full rule-firing trace.
