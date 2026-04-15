"""Package materialization and export utilities.

Re-exports the public API from submodules:

- ``assembly`` — pure derived-metadata computation
- ``export`` — filesystem serialization / deserialization
"""

from __future__ import annotations

from research.package.assembly import EvidenceStats, compute_evidence_stats, compute_run_summary
from research.package.export import read_package, sanitize_path_component, write_package

__all__ = [
    "EvidenceStats",
    "compute_evidence_stats",
    "compute_run_summary",
    "read_package",
    "sanitize_path_component",
    "write_package",
]
