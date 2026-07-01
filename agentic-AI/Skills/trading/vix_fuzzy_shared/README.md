# vix-fuzzy-shared

Shared VIX fuzzy-set constants and classification helpers used by both
[`vix-fuzzy-mcp-skill`](../vix-fuzzy-mcp-skill) and
[`risk-desk-mcp-skill`](../risk-desk-mcp-skill), so VIX fuzzy-set
membership is computed identically everywhere in the trading skill
suite instead of each skill keeping its own copy of the constants and
interpolation logic.

## API

- `VIX_SET_NAMES` — the seven ordinal fuzzy set names, increasing with VIX
- `VIX_MEMBERSHIP_RANGES` — the membership function ranges per set
- `get_most_recent_vix() -> float` — fetches and validates the most recent VIX close value
- `interpolate_vix_membership(vix: float) -> dict` — returns `{'fuzzy set membership': {...}, 'value range': {...}}` for a VIX value
