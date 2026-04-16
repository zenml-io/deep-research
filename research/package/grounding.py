"""Grounding analysis utilities for assembled investigation packages.

Pure functions — no LLM calls, no I/O.  Used by the assembly checkpoint
to validate citation density before producing an InvestigationPackage.
"""

from __future__ import annotations

import re

from research.contracts.evidence import EvidenceLedger

# Regex to match [evidence_id] references in report text
_CITATION_RE = re.compile(r"\[([^\]]+)\]")

# Minimum sentence length to count as "substantive" (skip short fragments)
_MIN_SENTENCE_LENGTH = 20


class GroundingError(Exception):
    """Raised when grounding checks fail during assembly."""


class CitationResolutionError(Exception):
    """Raised when citation IDs don't resolve to ledger entries."""


def extract_citation_ids(text: str) -> set[str]:
    """Extract all [evidence_id] references from markdown text.

    Filters out common markdown patterns that aren't evidence citations
    (e.g. [link text](url), section headers).
    """
    ids: set[str] = set()
    for match in _CITATION_RE.finditer(text):
        candidate = match.group(1)
        # Skip if it looks like a markdown link text (followed by parentheses)
        end = match.end()
        if end < len(text) and text[end] == "(":
            continue
        # Skip common non-citation patterns
        if candidate.startswith("http") or candidate.startswith("#"):
            continue
        # Skip checkbox patterns like [x] or [ ]
        if candidate in ("x", " ", "X"):
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
