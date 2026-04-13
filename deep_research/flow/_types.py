"""Shared types, constants, and helpers used across the flow sub-modules."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import NamedTuple

from pydantic import BaseModel, Field

from deep_research.config import ResearchConfig
from deep_research.enums import DeliverableMode, StopReason
from deep_research.models import (
    CoherenceResult,
    CoverageScore,
    CritiqueResult,
    EvidenceLedger,
    GroundingResult,
    IterationRecord,
    RawToolResult,
    RenderPayload,
    RequestClassification,
    ResearchPlan,
    ResearchPreferences,
    SearchAction,
    SeededEntities,
    SupervisorDecision,
    ToolCallRecord,
)


APPROVE_PLAN_WAIT_NAME = "approve_plan"
CLARIFY_BRIEF_WAIT_NAME = "clarify_brief"

# Deliverable modes the writer fully supports; other modes still render but
# surface a degradation note in the final package.
_FULLY_SUPPORTED_MODES: frozenset[DeliverableMode] = frozenset(
    {
        DeliverableMode.RESEARCH_PACKAGE,
        DeliverableMode.FINAL_REPORT,
        DeliverableMode.COMPARISON_MEMO,
    }
)


def _flow() -> ModuleType:
    """Return the live ``research_flow`` module so checkpoint monkeypatches are honoured."""
    from deep_research.flow import research_flow

    return research_flow


@dataclass(frozen=True)
class RunState:
    """Resolved inputs the iteration loop + downstream helpers consume."""

    brief: str
    config: ResearchConfig
    classification: RequestClassification
    preferences: ResearchPreferences
    clarify_options: "ClarifyOptions | None" = None
    seeded_entities: SeededEntities | None = None


class IterationLoopOutput(NamedTuple):
    ledger: EvidenceLedger
    iteration_history: tuple[IterationRecord, ...]
    provider_usage_summary: dict[str, int]
    spent_usd: float
    stop_reason: StopReason
    council_models: list[str]
    active_elapsed_seconds: int
    wall_elapsed_seconds: int
    total_tokens: int


class CritiqueBundle(NamedTuple):
    renders: list[RenderPayload]
    critique_result: CritiqueResult | None
    spent_usd: float


class JudgeBundle(NamedTuple):
    grounding_result: GroundingResult | None
    coherence_result: CoherenceResult | None
    spent_usd: float


class WaveSearchResult(NamedTuple):
    decision: SupervisorDecision
    raw_results: list[RawToolResult]
    supervisor_cost: float
    search_cost: float
    warnings: list[str]
    tool_calls: list[ToolCallRecord]
    step_latencies_ms: dict[str, int]
    total_tokens: int


class WaveTriageResult(NamedTuple):
    candidates: list
    relevance_cost: float
    step_latencies_ms: dict[str, int]
    total_tokens: int


class WaveEnrichResult(NamedTuple):
    ledger: EvidenceLedger
    step_latencies_ms: dict[str, int]


class WaveScoreResult(NamedTuple):
    coverage: CoverageScore
    step_latencies_ms: dict[str, int]


class WaveFeedbackResult(NamedTuple):
    carryover_actions: list[SearchAction]
    reason: str | None
    # Why: deep-tier synthesis-to-unsupported-claims feedback is capped by
    # SelectionPolicyConfig.feedback_loop_max_iterations; we track the count
    # here so the cap survives replay.
    feedback_iteration_count: int = 0


class _ReplanOutcome(NamedTuple):
    plan: ResearchPlan
    uncovered_subtopics: list[str]
    unanswered_questions: list[str]
    replans_used: int
    should_continue: bool


class ClarifyOptions(BaseModel):
    clarified_brief: str | None = Field(default=None)
    scope_adjustment: str | None = Field(default=None)
    source_preference: str | None = Field(default=None)
    depth_preference: str | None = Field(default=None)
    comparison_targets: list[str] = Field(default_factory=list)
    deliverable_mode: str | None = Field(default=None)


class PlanApproval(BaseModel):
    approved: bool = True
    notes: str | None = None
    scope_adjustment: str | None = None
    source_preference: str | None = None
    deliverable_mode: str | None = None


def merge_provider_counts(
    existing: dict[str, int],
    raw_results: list[RawToolResult],
) -> dict[str, int]:
    """Return a new dict with provider counts updated from ``raw_results``."""
    updated = dict(existing)
    for raw_result in raw_results:
        updated[raw_result.provider] = updated.get(raw_result.provider, 0) + 1
    return updated


def build_tool_call_records(
    raw_results: list[RawToolResult],
) -> list[ToolCallRecord]:
    """Build immutable tool-call records from raw results."""
    records: list[ToolCallRecord] = []
    for raw_result in raw_results:
        provider_label = f" via {raw_result.provider}" if raw_result.provider else ""
        if raw_result.ok:
            summary = f"{raw_result.tool_name}{provider_label} succeeded"
            status = "ok"
        else:
            detail = f": {raw_result.error}" if raw_result.error else ""
            summary = f"{raw_result.tool_name}{provider_label} failed{detail}"
            status = "error"
        records.append(
            ToolCallRecord(
                tool_name=raw_result.tool_name,
                status=status,
                provider=raw_result.provider,
                summary=summary,
            )
        )
    return records
