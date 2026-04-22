"""Package materialization and export utilities.

Re-exports the public API from submodules:

- ``assembly`` — pure derived-metadata computation
- ``export`` — filesystem serialization / deserialization
"""

from __future__ import annotations

from research.contracts.package import EvidenceStats
from research.package.assembly import compute_evidence_stats, compute_run_summary
from research.package.export import read_package, sanitize_path_component, write_package
from research.package.grounding import (
    CitationResolutionError,
    GroundingError,
    compute_grounding_density,
    extract_citation_ids,
    split_sentences,
    validate_citations,
)

__all__ = [
    "CitationResolutionError",
    "EvidenceStats",
    "GroundingError",
    "compute_evidence_stats",
    "compute_grounding_density",
    "compute_run_summary",
    "extract_citation_ids",
    "read_package",
    "sanitize_path_component",
    "split_sentences",
    "validate_citations",
    "write_package",
]
