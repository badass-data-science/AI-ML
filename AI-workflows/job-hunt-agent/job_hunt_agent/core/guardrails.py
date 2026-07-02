"""Case-insensitive scanning for the vault's standing content rules.

Run on every assembled draft before it's written to disk. A hit never causes
the draft to be silently altered or dropped — the text is written as-is and
the violation is surfaced as a warning, so a human sees exactly what tripped
the check and why, rather than having content silently stripped out from
under them.
"""

from __future__ import annotations

import re

from job_hunt_agent.core.vault_models import VaultSnapshot


def _word_boundary_pattern(term: str) -> re.Pattern:
    # Word-boundary, case-insensitive match. Plain substring matching on a
    # short acronym like "RAG" produces false positives inside ordinary
    # words ("paragraph", "leveraging") — \b anchors prevent that while
    # still matching "RAG" as its own token, hyphenated, or punctuated.
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def scan_for_violations(text: str, vault: VaultSnapshot) -> list[str]:
    """Return a list of human-readable violation messages, empty if clean."""
    violations: list[str] = []

    for term in vault.forbidden_terms:
        if _word_boundary_pattern(term).search(text):
            violations.append(
                f"forbidden term found: {term!r} — this is a standing content rule, "
                "see Notes/career-positioning.md in the vault"
            )

    for skill in vault.excluded_aspirational_skills:
        if _word_boundary_pattern(skill).search(text):
            violations.append(
                f"excluded aspirational skill found: {skill!r} — not yet a real, "
                "evidenced skill, see Notes/skills-vault-status.md in the vault"
            )

    return violations
