import json
from datetime import datetime as _datetime
from math import isfinite
from typing import Annotated, Literal

from pydantic import (
    AnyUrl,
    BaseModel,
    computed_field,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from deep_research.enums import SourceKind, StopReason, Tier


UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_iso8601_timestamp(value: str | None) -> str | None:
    """Return a timestamp only if it parses as an ISO-8601 string."""
    if value is None:
        return None

    normalized_value = value.replace("Z", "+00:00")
    try:
        _datetime.fromisoformat(normalized_value)
    except ValueError as exc:
        raise ValueError("timestamp must be a parseable ISO-8601 string") from exc

    return value


def _validate_json_mapping_keys(value: object) -> None:
    """Recursively validate that every nested mapping inside raw metadata uses string keys.

    Raw metadata must stay JSON-serializable for package persistence, so nested dict-like
    structures cannot contain non-string keys anywhere in the value tree.
    """
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError("raw_metadata mapping keys must be strings")
            _validate_json_mapping_keys(nested_value)
        return

    if isinstance(value, list | tuple):
        for item in value:
            _validate_json_mapping_keys(item)


class ResearchPlan(StrictBaseModel):
    goal: str
    key_questions: list[str]
    subtopics: list[str]
    queries: list[str]
    sections: list[str]
    success_criteria: list[str]
    query_groups: dict[str, list[str]] = Field(default_factory=dict)
    allowed_source_groups: list[str] = Field(default_factory=list)
    approval_status: Literal["not_requested", "pending", "approved", "rejected"] = (
        "not_requested"
    )


class EvidenceSnippet(StrictBaseModel):
    text: str
    source_locator: str | None = None


class EvidenceCandidate(StrictBaseModel):
    key: str
    title: str
    url: AnyUrl
    snippets: list[EvidenceSnippet] = Field(default_factory=list)
    provider: str
    source_kind: SourceKind
    quality_score: UnitFloat = 0.0
    relevance_score: UnitFloat = 0.0
    authority_score: UnitFloat = 0.0
    freshness_score: UnitFloat = 0.0
    matched_subtopics: list[str] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    raw_metadata: dict[str, object] = Field(default_factory=dict)
    selected: bool = False

    @model_validator(mode="after")
    def validate_raw_metadata(self) -> "EvidenceCandidate":
        try:
            _validate_json_mapping_keys(self.raw_metadata)
            json.dumps(self.raw_metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw_metadata must be JSON-serializable") from exc
        return self


class DedupeEvent(StrictBaseModel):
    duplicate_key: str
    canonical_key: str
    match_basis: Literal["doi", "arxiv_id", "canonical_url", "title"]


class EvidenceLedger(StrictBaseModel):
    considered: list[EvidenceCandidate] = Field(default_factory=list)
    selected: list[EvidenceCandidate] = Field(default_factory=list)
    rejected: list[EvidenceCandidate] = Field(default_factory=list)
    dedupe_log: list[DedupeEvent] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def populate_considered_from_entries(cls, data: object) -> object:
        if isinstance(data, dict) and "entries" in data:
            data = dict(data)
            entries = data.pop("entries")
            if "considered" not in data:
                data["considered"] = entries
            else:
                normalized_considered = [
                    EvidenceCandidate.model_validate(candidate)
                    for candidate in data["considered"]
                ]
                normalized_entries = [
                    EvidenceCandidate.model_validate(candidate) for candidate in entries
                ]
                if normalized_considered != normalized_entries:
                    raise ValueError(
                        "entries must match considered when both are provided"
                    )
        return data

    @computed_field
    @property
    def entries(self) -> list[EvidenceCandidate]:
        return self.considered


class SelectionItem(StrictBaseModel):
    candidate_key: str
    rationale: str
    bridge_note: str | None = None
    matched_subtopics: list[str] = Field(default_factory=list)
    reading_time_minutes: StrictInt | None = None
    ordering_rationale: str | None = None

    @model_validator(mode="after")
    def validate_reading_time(self) -> "SelectionItem":
        if self.reading_time_minutes is not None and self.reading_time_minutes < 0:
            raise ValueError("reading_time_minutes must be non-negative")
        return self


class SelectionGraph(StrictBaseModel):
    items: list[SelectionItem] = Field(default_factory=list)
    gap_coverage_summary: list[str] = Field(default_factory=list)


class IterationRecord(StrictBaseModel):
    iteration: StrictInt
    new_candidate_count: StrictInt = 0
    accepted_candidate_count: StrictInt = 0
    rejected_candidate_count: StrictInt = 0
    coverage: StrictFloat = 0.0
    coverage_delta: StrictFloat = 0.0
    uncovered_subtopics: list[str] = Field(default_factory=list)
    estimated_cost_usd: StrictFloat = 0.0
    tool_calls: list["ToolCallRecord"] = Field(default_factory=list)
    continue_reason: str | None = None
    stop_reason: StopReason | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "IterationRecord":
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.new_candidate_count < 0:
            raise ValueError("new_candidate_count must be non-negative")
        if self.accepted_candidate_count < 0:
            raise ValueError("accepted_candidate_count must be non-negative")
        if self.rejected_candidate_count < 0:
            raise ValueError("rejected_candidate_count must be non-negative")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be between 0.0 and 1.0")
        if not isfinite(self.coverage_delta):
            raise ValueError("coverage_delta must be finite")
        if not isfinite(self.estimated_cost_usd):
            raise ValueError("estimated_cost_usd must be finite")
        if self.estimated_cost_usd < 0.0:
            raise ValueError("estimated_cost_usd must be non-negative")
        return self


class IterationTrace(StrictBaseModel):
    iterations: list[IterationRecord] = Field(default_factory=list)


class CritiqueDimensionScore(StrictBaseModel):
    name: str
    score: UnitFloat
    rationale: str


class CritiqueResult(StrictBaseModel):
    dimensions: list[CritiqueDimensionScore] = Field(default_factory=list)
    summary: str
    revision_suggestions: list[str] = Field(default_factory=list)
    revision_recommended: bool = False


class GroundingVerdict(StrictBaseModel):
    citation: str
    candidate_key: str | None = None
    supported: bool
    rationale: str


class GroundingResult(StrictBaseModel):
    score: UnitFloat
    verdicts: list[GroundingVerdict] = Field(default_factory=list)


class CoherenceResult(StrictBaseModel):
    relevance: UnitFloat
    logical_flow: UnitFloat
    completeness: UnitFloat
    consistency: UnitFloat
    summary: str


class RenderPayload(StrictBaseModel):
    name: str
    content_markdown: str
    citation_map: dict[str, str] = Field(default_factory=dict)
    structured_content: dict[str, object] | None = None
    generated_at: str | None = None

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str | None) -> str | None:
        return _validate_iso8601_timestamp(value)


class RunSummary(StrictBaseModel):
    run_id: str
    brief: str
    tier: Tier
    stop_reason: StopReason
    status: str
    estimated_cost_usd: float = 0.0
    elapsed_seconds: int = 0
    iteration_count: int = 0
    provider_usage_summary: dict[str, int] = Field(default_factory=dict)
    council_enabled: bool = False
    council_size: int = 1
    council_models: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value: str | None) -> str | None:
        return _validate_iso8601_timestamp(value)

    @model_validator(mode="after")
    def validate_phase_one_metadata(self) -> "RunSummary":
        if not isfinite(self.estimated_cost_usd):
            raise ValueError("estimated_cost_usd must be finite")
        if self.estimated_cost_usd < 0.0:
            raise ValueError("estimated_cost_usd must be non-negative")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.iteration_count < 0:
            raise ValueError("iteration_count must be non-negative")
        if self.council_size < 1:
            raise ValueError("council_size must be at least 1")
        for provider, usage_count in self.provider_usage_summary.items():
            if usage_count < 0:
                raise ValueError(
                    f"provider_usage_summary[{provider!r}] must be non-negative"
                )
        return self


class InvestigationPackage(StrictBaseModel):
    run_summary: RunSummary
    research_plan: ResearchPlan
    evidence_ledger: EvidenceLedger
    selection_graph: SelectionGraph
    iteration_trace: IterationTrace
    renders: list[RenderPayload]
    critique_result: CritiqueResult | None = None
    grounding_result: GroundingResult | None = None
    coherence_result: CoherenceResult | None = None


class CoverageScore(StrictBaseModel):
    subtopic_coverage: UnitFloat
    source_diversity: UnitFloat
    evidence_density: UnitFloat
    total: UnitFloat
    uncovered_subtopics: list[str] = Field(default_factory=list)


class ToolCallRecord(StrictBaseModel):
    tool_name: str
    status: str
    provider: str | None = None
    summary: str | None = None


class IterationBudget(StrictBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @model_validator(mode="after")
    def validate_non_negative(self) -> "IterationBudget":
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.estimated_cost_usd < 0.0:
            raise ValueError("estimated_cost_usd must be non-negative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        return self


class RawToolResult(StrictBaseModel):
    tool_name: str
    provider: str
    payload: dict
    ok: bool = True
    error: str | None = None


class SupervisorDecision(StrictBaseModel):
    rationale: str
    search_actions: list[str]


class SupervisorCheckpointResult(StrictBaseModel):
    raw_results: list[RawToolResult]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RelevanceCheckpointResult(StrictBaseModel):
    candidates: list[EvidenceCandidate]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RequestClassification(StrictBaseModel):
    audience_mode: str
    freshness_mode: str
    recommended_tier: Tier
    needs_clarification: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_clarification_state(self) -> "RequestClassification":
        if self.needs_clarification:
            if (
                self.clarification_question is None
                or not self.clarification_question.strip()
            ):
                raise ValueError(
                    "clarification_question must be set when clarification is needed"
                )
        elif self.clarification_question is not None:
            raise ValueError(
                "clarification_question must be None when clarification is not needed"
            )
        return self
