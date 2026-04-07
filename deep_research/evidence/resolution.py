from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deep_research.models import EvidenceCandidate, EvidenceLedger


def resolve_selected_entries(ledger: EvidenceLedger) -> list[EvidenceCandidate]:
    """Return the best available selected entries from a ledger.

    Handles three ledger shapes:
    1. Phase 2+ ledger with explicit selected/rejected buckets.
    2. Legacy entries-only ledger where ``selected`` was set on each entry.
    3. Legacy entries-only ledger without selection flags — returns all entries.
    """
    if ledger.selected or ledger.rejected:
        return ledger.selected
    if any("selected" in c.model_fields_set for c in ledger.entries):
        return [c for c in ledger.entries if c.selected]
    selected = [c for c in ledger.entries if c.selected]
    return selected or ledger.entries


def resolve_coverage_entries(ledger: EvidenceLedger) -> list[EvidenceCandidate]:
    """Like ``resolve_selected_entries`` but also includes legacy considered
    entries that were never explicitly selected or rejected (for coverage
    scoring purposes).
    """
    if ledger.selected or ledger.rejected:
        entries = list(ledger.selected)
        selected_keys = {e.key for e in entries}
        rejected_keys = {e.key for e in ledger.rejected}
        for candidate in ledger.considered:
            if (
                "selected" not in candidate.model_fields_set
                and candidate.key not in selected_keys
                and candidate.key not in rejected_keys
            ):
                entries.append(candidate)
        return entries
    if any("selected" in c.model_fields_set for c in ledger.entries):
        return [
            c
            for c in ledger.entries
            if c.selected or "selected" not in c.model_fields_set
        ]
    selected = [c for c in ledger.entries if c.selected]
    return selected or ledger.entries
