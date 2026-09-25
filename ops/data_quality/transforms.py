"""
transforms.py — Pure chunk post-processing used by the v2 ingestion.

Both functions exist because the Great Expectations suite failed on the Module 1 chunks:

  * merge_small_chunks — Module 1 split markdown on every header, which left 232 of 426
    chunks (54%) under 200 characters, e.g. "## Current version\\n**v2** is the current
    stable version." A chunk like that embeds poorly and, when it wins retrieval, hands the
    LLM a heading with no instructions under it (Module 1's Q037 failure).
  * the same GE run also caught a 19-character orphan chunk in a runbook PDF:
    "(RPO is 5 minutes)." — the recursive splitter cut the end of a sentence off into its
    own chunk, separating the RPO fact from the failover procedure it belongs to. Orphans
    like this (any doc type, < ORPHAN_CHARS) are folded back into the previous chunk.
  * redact_pii — resolved tickets contain real-looking customer email addresses. Those
    should never be embedded, stored in Pinecone metadata, or quoted back to a CSM.

Pure functions (dicts in, dicts out) so they are trivially unit-testable.
"""

from __future__ import annotations

import itertools
import re

# Company / placeholder addresses that appear in product docs on purpose.
ALLOWED_EMAIL_DOMAINS = ("helix.io", "example.com", "company.com")

_allowed = "|".join(re.escape(d) for d in ALLOWED_EMAIL_DOMAINS)

# The same patterns are used by the GE suite (detection) and here (redaction), so the
# two can never drift apart.
PII_PATTERNS: dict[str, str] = {
    "email": rf"[A-Za-z0-9._%+-]+@(?!(?:{_allowed})\b)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{{2,}}",
    "phone": r"(?<![\w-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])",
    "ssn": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "credit_card": r"(?<!\d)(?:\d{4}[ -]){3}\d{4}(?!\d)",
}

ORPHAN_CHARS = 100

_COMPILED = {name: re.compile(p) for name, p in PII_PATTERNS.items()}


def find_pii(text: str) -> dict[str, int]:
    """Count PII matches per type (used for reporting; never returns the values)."""
    return {name: len(rx.findall(text)) for name, rx in _COMPILED.items() if rx.search(text)}


def redact_pii(text: str) -> str:
    for name, rx in _COMPILED.items():
        text = rx.sub(f"[REDACTED_{name.upper()}]", text)
    return text


def redact_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Returns (new_chunks, number_of_chunks_changed)."""
    out, changed = [], 0
    for c in chunks:
        new_text = redact_pii(c["text"])
        if new_text != c["text"]:
            changed += 1
        out.append({**c, "text": new_text})
    return out, changed


def merge_small_chunks(
    chunks: list[dict], min_chars: int = 400, max_chars: int = 1500
) -> list[dict]:
    """Merge consecutive *markdown* chunks of the same document until each is >= min_chars.

    - Never merges across documents (keeps doc_id citations correct).
    - Never merges past max_chars.
    - Markdown sections are merged up to min_chars. PDFs/tickets keep their Module 1
      chunking (1000-char recursive splitter / one chunk per ticket) except that an orphan
      fragment shorter than ORPHAN_CHARS is folded into the chunk before it.
    - Chunk ids are re-numbered "<doc_id>::<i>" so they stay unique and ordered.
    """
    if min_chars <= 0:
        return [dict(c) for c in chunks]

    out: list[dict] = []
    for doc_id, group in itertools.groupby(chunks, key=lambda c: c["doc_id"]):
        group = list(group)
        if group[0].get("doc_type") != "markdown":
            kept: list[dict] = []
            for c in group:
                if kept and len(c["text"]) < ORPHAN_CHARS and c["text"] not in kept[-1]["text"]:
                    kept[-1] = {**kept[-1], "text": f"{kept[-1]['text']} {c['text']}"}
                elif kept and len(c["text"]) < ORPHAN_CHARS:
                    continue  # fully contained in the previous chunk's overlap
                else:
                    kept.append(dict(c))
            for i, c in enumerate(kept):
                out.append({**c, "id": f"{doc_id}::{i}", "chunk_index": i})
            continue

        texts: list[str] = []
        buf: str | None = None
        for c in group:
            t = c["text"]
            if buf is None:
                buf = t
            elif len(buf) < min_chars and len(buf) + 2 + len(t) <= max_chars:
                buf = f"{buf}\n\n{t}"
            else:
                texts.append(buf)
                buf = t
        if buf is not None:
            # A small trailing section gets folded into the previous chunk if it fits.
            if texts and len(buf) < min_chars and len(texts[-1]) + 2 + len(buf) <= max_chars:
                texts[-1] = f"{texts[-1]}\n\n{buf}"
            else:
                texts.append(buf)

        template = group[0]
        for i, t in enumerate(texts):
            out.append({**template, "id": f"{doc_id}::{i}", "text": t, "chunk_index": i})
    return out
