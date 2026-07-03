"""Best-effort helpers for working with raw posting text.

Nothing here is authoritative — these are conveniences for a human (or an
assisted writing pass) doing the company-specific paragraph, which the
vault's own rules say must always be written fresh. This module only ever
saves a re-read of the raw posting; it never writes anything.
"""

from __future__ import annotations

import re

# "**About Acme Corp.** We build..." — heading and body sharing one line,
# bold-wrapped. Requires both the opening and closing ** so the middle
# [^*\n]* (which can't itself contain *) naturally stops at the real
# boundary — a non-greedy version of this without the closing ** anchor
# matches zero characters and leaves the company name stuck in the body.
_BOLD_ABOUT_INLINE_RE = re.compile(r"(?i)^\*\*about\b[^*\n]*\*\*[:.]?\s*")

# "About Acme Corp" (plain, or a markdown heading) as its own whole line —
# body is whatever follows, same paragraph or the next one.
_ABOUT_HEADING_LINE_RE = re.compile(r"(?i)^about\b")


def extract_company_brief(text: str) -> str | None:
    """Best-effort extraction of a posting's "About <Company>" blurb.

    Looks for a paragraph that reads like an "About ..." heading — whether
    bold-inline with the body on the same line, or its own heading line with
    the body following — and returns the prose after it, pulling from the
    next paragraph if the heading's own paragraph has nothing else in it.
    Falls back to the first substantial paragraph if no such heading is
    found at all — in practice, a posting's mission statement is usually the
    first real paragraph even without one.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    for i, para in enumerate(paragraphs):
        inline_match = _BOLD_ABOUT_INLINE_RE.match(para)
        if inline_match:
            body = para[inline_match.end():].strip()
        else:
            first_line = para.splitlines()[0].strip("# *").strip()
            if not _ABOUT_HEADING_LINE_RE.match(first_line):
                continue
            body = "\n".join(para.splitlines()[1:]).strip()

        if not body and i + 1 < len(paragraphs):
            body = paragraphs[i + 1]
        body = " ".join(body.split())
        if body:
            return body

    for para in paragraphs:
        candidate = " ".join(para.split())
        if len(candidate) > 100 and not candidate.startswith("#"):
            return candidate

    return None
