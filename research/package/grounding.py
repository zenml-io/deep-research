"""Grounding analysis utilities for assembled investigation packages.

Pure functions — no LLM calls, no I/O.  Used by the assembly checkpoint
to validate citation density before producing an InvestigationPackage.
"""

from __future__ import annotations

import re

from research.contracts.evidence import EvidenceLedger

# Regex to match [evidence_id] references in report text
_CITATION_RE = re.compile(r"\[([^\]]+)\]")

# Fenced code blocks (``` ... ```) — stripped before citation extraction so
# square brackets in code examples (JSON arrays, permission manifests, etc.)
# don't get misread as evidence citations.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)

# Minimum sentence length to count as "substantive" (skip short fragments)
_MIN_SENTENCE_LENGTH = 20


class GroundingError(Exception):
    """Raised when grounding checks fail during assembly."""


class CitationResolutionError(Exception):
    """Raised when citation IDs don't resolve to ledger entries."""


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks so their contents don't feed the citation regex."""
    return _FENCED_CODE_RE.sub("", text)


def extract_citation_ids(text: str) -> set[str]:
    """Extract all [evidence_id] references from markdown text.

    Filters out:
    - fenced code blocks (JSON examples, code snippets)
    - markdown link text like ``[label](url)``
    - http(s) URLs and section anchors inside brackets
    - checkbox patterns (``[ ]`` / ``[x]``)
    - multi-line captures (evidence IDs never span lines)
    """
    stripped = _strip_code_blocks(text)
    ids: set[str] = set()
    for match in _CITATION_RE.finditer(stripped):
        candidate = match.group(1)
        # Skip if followed by '(' — that's a markdown link
        end = match.end()
        if end < len(stripped) and stripped[end] == "(":
            continue
        if candidate.startswith("http") or candidate.startswith("#"):
            continue
        if candidate in ("x", " ", "X"):
            continue
        # Evidence IDs are single-line, quote-free tokens; multiline or
        # quoted captures are overwhelmingly JSON / code leakage.
        if "\n" in candidate or '"' in candidate:
            continue
        ids.add(candidate)
    return ids


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for grounding density analysis.

    Simple sentence splitting on period, exclamation, question mark
    followed by whitespace. Only returns substantive sentences
    (>= _MIN_SENTENCE_LENGTH characters after stripping).
    """
    raw = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LENGTH]


def compute_grounding_density(report_content: str, valid_ids: set[str]) -> float:
    """Compute the fraction of substantive sentences containing valid citations.

    A sentence is "grounded" if it contains at least one [evidence_id]
    that resolves to the valid_ids set.

    Returns a float in [0.0, 1.0]. Returns 1.0 if there are no
    substantive sentences (vacuously true).
    """
    sentences = split_sentences(report_content)
    if not sentences:
        return 1.0

    grounded = 0
    for sentence in sentences:
        cited = extract_citation_ids(sentence)
        if cited & valid_ids:  # any valid citation
            grounded += 1

    return grounded / len(sentences)


def validate_citations(
    report_content: str, ledger: EvidenceLedger
) -> tuple[set[str], set[str]]:
    """Validate that all citation IDs resolve to ledger entries.

    Returns (valid_ids, unresolved_ids).
    """
    ledger_ids = {item.evidence_id for item in ledger.items}
    cited_ids = extract_citation_ids(report_content)
    valid = cited_ids & ledger_ids
    unresolved = cited_ids - ledger_ids
    return valid, unresolved
