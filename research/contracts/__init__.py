"""V2 data contracts — re-exports for convenient imports."""

from research.contracts.base import StrictBase
from research.contracts.brief import ResearchBrief
from research.contracts.decisions import SubagentFindings, SupervisorDecision
from research.contracts.evidence import EvidenceItem, EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import (
    CouncilComparison,
    CouncilPackage,
    EvidenceStats,
    InvestigationPackage,
    RunMetadata,
)
from research.contracts.plan import ResearchPlan, SubagentTask
from research.contracts.reports import (
    CritiqueDimensionScore,
    CritiqueReport,
    DraftReport,
    FinalReport,
)

__all__ = [
    "StrictBase",
    "ResearchBrief",
    "ResearchPlan",
    "SubagentTask",
    "EvidenceItem",
    "EvidenceLedger",
    "SubagentFindings",
    "SupervisorDecision",
    "IterationRecord",
    "DraftReport",
    "CritiqueDimensionScore",
    "CritiqueReport",
    "FinalReport",
    "EvidenceStats",
    "RunMetadata",
    "InvestigationPackage",
    "CouncilComparison",
    "CouncilPackage",
]
