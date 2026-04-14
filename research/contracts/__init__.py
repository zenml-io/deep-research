"""V2 data contracts — re-exports for convenient imports."""

from research.contracts.base import StrictBase
from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceItem, EvidenceLedger
from research.contracts.plan import ResearchPlan, SubagentTask

__all__ = [
    "StrictBase",
    "ResearchBrief",
    "ResearchPlan",
    "SubagentTask",
    "EvidenceItem",
    "EvidenceLedger",
]
