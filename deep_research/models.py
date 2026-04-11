import json
from datetime import datetime
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

from deep_research.enums import (
    DeliverableMode,
    PlanningMode,
    SourceGroup,
    SourceKind,
    StopReason,
    Tier,
)


UnitFloat = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validate_iso8601_timestamp(value: str | None) -> str | None:
    """Return a timestamp only if it parses as an ISO-8601 string."""
    if value is None:
        return None

    normalized_value = value.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized_value)
    except ValueError as exc:
        raise ValueError("timestamp must be a parseable ISO-8601 string") from exc

    return value


def validate_json_mapping_keys(value: object) -> None:
    """Recursively validate that every nested mapping inside raw metadata uses string keys.

    Raw metadata must stay JSON-serializable for package persistence, so nested dict-like
    structures cannot contain non-string keys anywhere in the value tree.
    """
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError("raw_metadata mapping keys must be strings")
            validate_json_mapping_keys(nested_value)
        return

    if isinstance(value, list | tuple):
        for item in value:
            validate_json_mapping_keys(item)


class ResearchPreferences(StrictBaseModel):
    """User intent extracted from the research brief.

    Advisory fields shape LLM decisions. Exclusion fields are hard constraints
    enforced at the provider registry level.
    """

    audience: str | None = None
    freshness: str | None = None
    deliverable_mode: DeliverableMode = DeliverableMode.RESEARCH_PACKAGE
    preferred_source_groups: list[SourceGroup] = Field(default_factory=list)
    excluded_source_groups: list[SourceGroup] = Field(default_factory=list)
    preferred_providers: list[str] = Field(default_factory=list)
    excluded_providers: list[str] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    time_window_days: int | None = None
    planning_mode: PlanningMode = PlanningMode.BROAD_SCAN
    cost_bias: str | None = None
    speed_bias: str | None = None


class ResearchPlan(StrictBaseModel):
    """Structured plan the planner produces for a research brief.

    Captures the goal, guiding questions, subtopic decomposition, seed queries,
    output sections, and success criteria that the downstream supervisor loop
    and writer will rely on.
    """

    goal: str = Field(
        ...,
        description="One-sentence statement of what the research run should achieve.",
    )
    key_questions: list[str] = Field(
        ...,
        description="Specific questions the final deliverable must answer.",
    )
    subtopics: list[str] = Field(
        ...,
        description="Decomposition of the goal into focused subtopics to investigate.",
    )
    queries: list[str] = Field(
        ...,
        description="Initial search queries seeded for the supervisor to run.",
    )
    sections: list[str] = Field(
        ...,
        description="Section titles the writer should use to structure the deliverable.",
    )
    success_criteria: list[str] = Field(
        ...,
        description="Observable criteria used to judge whether the run succeeded.",
    )
    query_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Optional grouping of queries keyed by subtopic or theme for targeted search.",
    )
    allowed_source_groups: list[str] = Field(
        default_factory=list,
        description="Optional allow-list of source groups (e.g. academic, news) the plan should stay within.",
    )
    approval_status: Literal["not_requested", "pending", "approved", "rejected"] = (
        Field(
            default="not_requested",
            description="Human-in-the-loop approval state for the plan before execution proceeds.",
        )
    )


class EvidenceSnippet(StrictBaseModel):
    text: str = Field(..., description="Extracted text content from the source.")
    source_locator: str | None = Field(
        default=None,
        description="Page number, section anchor, or paragraph reference within the source.",
    )


class EvidenceCandidate(StrictBaseModel):
    key: str = Field(..., description="Unique identifier for this evidence candidate.")
    title: str = Field(..., description="Title or headline of the source document.")
    url: AnyUrl = Field(..., description="URL of the source document.")
    snippets: list[EvidenceSnippet] = Field(
        default_factory=list,
        description="Extracted text snippets from the source.",
    )
    provider: str = Field(
        ...,
        description="Search provider that returned this result (e.g. 'brave', 'arxiv').",
    )
    source_kind: SourceKind = Field(
        ..., description="Classification of the source type."
    )
    quality_score: UnitFloat = Field(
        default=0.0, description="Overall quality estimate (0.0 to 1.0)."
    )
    relevance_score: UnitFloat = Field(
        default=0.0, description="Relevance to the research brief (0.0 to 1.0)."
    )
    authority_score: UnitFloat = Field(
        default=0.0, description="Source authority estimate (0.0 to 1.0)."
    )
    freshness_score: UnitFloat = Field(
        default=0.0, description="Recency score (0.0 to 1.0)."
    )
    matched_subtopics: list[str] = Field(
        default_factory=list,
        description="Plan subtopics this evidence addresses.",
    )
    doi: str | None = Field(
        default=None, description="Digital Object Identifier if available."
    )
    arxiv_id: str | None = Field(
        default=None, description="ArXiv paper ID if available."
    )
    raw_metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Provider-specific metadata preserved for downstream processing.",
    )
    selected: bool = Field(
        default=False,
        description="True if this candidate was selected for the final deliverable.",
    )

    @model_validator(mode="after")
    def validate_raw_metadata(self) -> "EvidenceCandidate":
        try:
            validate_json_mapping_keys(self.raw_metadata)
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
    candidate_key: str = Field(
        ..., description="Key of the EvidenceCandidate this item refers to."
    )
    rationale: str = Field(
        ..., description="Why this candidate was selected for the deliverable."
    )
    bridge_note: str | None = Field(
        default=None,
        description="How this item connects to adjacent items in reading order.",
    )
    matched_subtopics: list[str] = Field(
        default_factory=list, description="Subtopics this selection covers."
    )
    reading_time_minutes: StrictInt | None = Field(
        default=None, description="Estimated reading time in minutes."
    )
    ordering_rationale: str | None = Field(
        default=None,
        description="Why this item appears at this position in the reading order.",
    )

    @model_validator(mode="after")
    def validate_reading_time(self) -> "SelectionItem":
        if self.reading_time_minutes is not None and self.reading_time_minutes < 0:
            raise ValueError("reading_time_minutes must be non-negative")
        return self


class SelectionGraph(StrictBaseModel):
    """Curator's ordered shortlist of evidence chosen for the final deliverable.

    Explains which candidates survived curation, how they connect, and which
    subtopic gaps the selection still leaves open.
    """

    items: list[SelectionItem] = Field(
        default_factory=list,
        description="Ordered list of curated selection items with rationales and bridge notes.",
    )
    gap_coverage_summary: list[str] = Field(
        default_factory=list,
        description="Human-readable notes describing subtopics that remain under-covered after curation.",
    )


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
    """Reviewer agent's critique of a draft deliverable.

    Scores the draft across multiple rubric dimensions and produces
    actionable revision suggestions for the writer.
    """

    dimensions: list[CritiqueDimensionScore] = Field(
        default_factory=list,
        description="Per-dimension rubric scores with short rationales from the reviewer.",
    )
    summary: str = Field(
        ...,
        description="Overall qualitative summary of the draft's strengths and weaknesses.",
    )
    revision_suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete, actionable revision instructions the writer should apply.",
    )
    revision_recommended: bool = Field(
        default=False,
        description="True when the reviewer believes another writer pass is warranted.",
    )


class CritiqueCheckpointResult(StrictBaseModel):
    critique: CritiqueResult
    budget: "IterationBudget" = Field(default_factory=lambda: IterationBudget())


class GroundingVerdict(StrictBaseModel):
    citation: str
    candidate_key: str | None = None
    supported: bool
    rationale: str


class GroundingResult(StrictBaseModel):
    """Judge verdict on whether claims in the draft are grounded in cited evidence.

    Aggregates per-citation verdicts into a single grounding score used by
    the control loop to decide whether to continue iterating.
    """

    score: UnitFloat = Field(
        ...,
        description="Overall fraction of claims judged to be supported by their citations (0.0 to 1.0).",
    )
    verdicts: list[GroundingVerdict] = Field(
        default_factory=list,
        description="Per-citation grounding verdicts with rationale and resolved candidate key.",
    )


class GroundingCheckpointResult(StrictBaseModel):
    grounding: GroundingResult
    budget: "IterationBudget" = Field(default_factory=lambda: IterationBudget())


class CoherenceResult(StrictBaseModel):
    """Judge verdict on the structural quality of a draft deliverable.

    Captures independent 0-1 scores across relevance, flow, completeness, and
    consistency so the control loop can decide on further revision.
    """

    relevance: UnitFloat = Field(
        ...,
        description="How well the draft addresses the research brief (0.0 to 1.0).",
    )
    logical_flow: UnitFloat = Field(
        ...,
        description="Quality of argument progression and transitions between sections (0.0 to 1.0).",
    )
    completeness: UnitFloat = Field(
        ...,
        description="Extent to which the draft covers all required subtopics and questions (0.0 to 1.0).",
    )
    consistency: UnitFloat = Field(
        ...,
        description="Internal consistency of claims, terminology, and framing (0.0 to 1.0).",
    )
    summary: str = Field(
        ...,
        description="Qualitative rationale summarising the structural assessment.",
    )


class CoherenceCheckpointResult(StrictBaseModel):
    coherence: CoherenceResult
    budget: "IterationBudget" = Field(default_factory=lambda: IterationBudget())


class RenderPayload(StrictBaseModel):
    name: str
    content_markdown: str
    citation_map: dict[str, str] = Field(default_factory=dict)
    structured_content: dict[str, object] | None = None
    generated_at: str | None = None

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str | None) -> str | None:
        return validate_iso8601_timestamp(value)


class RenderProse(StrictBaseModel):
    """Writer agent's prose rendering of a deliverable (pre-citation-mapping).

    Used as the structured output type for the writer agent when it emits
    a markdown draft for a specific deliverable mode.
    """

    content_markdown: str = Field(
        ...,
        description="Full markdown body of the rendered deliverable produced by the writer.",
    )
    render_label: str = Field(
        default="",
        description="Short label identifying this render (e.g. 'full_report', 'backing_report').",
    )
    deliverable_mode: DeliverableMode = Field(
        default=DeliverableMode.RESEARCH_PACKAGE,
        description="Which deliverable flavour this render targets.",
    )


class RevisionCheckpointResult(StrictBaseModel):
    renders: list[RenderPayload]
    budget: "IterationBudget" = Field(default_factory=lambda: IterationBudget())


class RenderCheckpointResult(StrictBaseModel):
    render: RenderPayload
    budget: "IterationBudget" = Field(default_factory=lambda: IterationBudget())


class RenderSettingsSnapshot(StrictBaseModel):
    writer_model: str


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
        return validate_iso8601_timestamp(value)

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


class RunMetadataStamp(StrictBaseModel):
    """Immutable run identity and wall-clock start captured at flow kickoff.

    Produced inside a checkpoint so replay returns the same ``run_id`` and
    ``started_at`` the original run observed, keeping non-determinism behind
    the checkpoint replay boundary.
    """

    run_id: str = Field(
        ...,
        description="Stable identifier for this run, shared across all RunSummary snapshots.",
    )
    started_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp (Z-suffixed) marking when the flow kicked off.",
    )

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: str) -> str:
        validated = validate_iso8601_timestamp(value)
        if validated is None:
            raise ValueError("started_at must be a parseable ISO-8601 string")
        return validated


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
    render_settings: RenderSettingsSnapshot | None = None
    preferences: ResearchPreferences | None = None
    preference_degradations: list[str] = Field(default_factory=list)


class CoverageScore(StrictBaseModel):
    subtopic_coverage: UnitFloat = Field(
        ...,
        description="Fraction of plan subtopics judged to be covered by the gathered evidence (0.0 to 1.0).",
    )
    source_diversity: UnitFloat = Field(
        ...,
        description="Diversity of information sources used, normalized to 0.0-1.0.",
    )
    evidence_density: UnitFloat = Field(
        ...,
        description="Density of evidence relative to the number of key questions, normalized to 0.0-1.0.",
    )
    total: UnitFloat = Field(
        ...,
        description="Simple average of subtopic_coverage, source_diversity, and evidence_density, rounded to 4 decimal places.",
    )
    uncovered_subtopics: list[str] = Field(
        default_factory=list,
        description="Subtopics from the plan that the scorer judged to be insufficiently covered.",
    )


class ToolCallRecord(StrictBaseModel):
    tool_name: str = Field(..., description="Name of the tool that was invoked.")
    status: str = Field(..., description="Outcome status: 'ok' or 'error'.")
    provider: str | None = Field(
        default=None, description="Search provider that handled the call."
    )
    summary: str | None = Field(
        default=None, description="Human-readable one-line summary of the call outcome."
    )


class IterationBudget(StrictBaseModel):
    input_tokens: int = Field(default=0, description="Number of input tokens consumed.")
    output_tokens: int = Field(
        default=0, description="Number of output tokens generated."
    )
    total_tokens: int = Field(default=0, description="Sum of input and output tokens.")
    estimated_cost_usd: float = Field(
        default=0.0, description="Estimated cost in USD for this budget period."
    )

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
    tool_name: str = Field(
        ..., description="Name of the tool that produced this result."
    )
    provider: str = Field(
        ..., description="Search provider or tool backend identifier."
    )
    payload: dict = Field(..., description="Raw result payload from the tool.")
    ok: bool = Field(default=True, description="True if the tool call succeeded.")
    error: str | None = Field(
        default=None, description="Error message if the tool call failed."
    )


class SearchAction(StrictBaseModel):
    query: str = Field(..., description="The search query string to execute.")
    rationale: str = Field(
        ..., description="Why this query was chosen and what gap it targets."
    )
    preferred_providers: list[str] = Field(
        default_factory=list, description="Provider names to prefer for this query."
    )
    preferred_source_kinds: list[SourceKind] = Field(
        default_factory=list, description="Source types to prefer."
    )
    recency_days: StrictInt | None = Field(
        default=None, description="Limit results to the last N days."
    )
    max_results: StrictInt | None = Field(
        default=None, description="Maximum number of results to return."
    )

    @model_validator(mode="after")
    def validate_limits(self) -> "SearchAction":
        if self.recency_days is not None and self.recency_days <= 0:
            raise ValueError("recency_days must be positive when provided")
        if self.max_results is not None and self.max_results <= 0:
            raise ValueError("max_results must be positive when provided")
        return self


class SearchExecutionResult(StrictBaseModel):
    raw_results: list[RawToolResult] = Field(default_factory=list)
    budget: "IterationBudget" = Field(default_factory=lambda: IterationBudget())


class SupervisorDecision(StrictBaseModel):
    """Supervisor agent's decision for the next iteration of the research loop.

    Contains the reasoning, the concrete search actions to execute this turn,
    and a status flag signalling whether enough evidence has been gathered.
    """

    rationale: str = Field(
        ...,
        description="Brief explanation of why these search actions were chosen and what gap they target.",
    )
    search_actions: list[SearchAction] = Field(
        default_factory=list,
        description="Search actions (query + preferences) the supervisor wants executed this turn.",
    )
    status: Literal["continue", "complete"] = Field(
        default="continue",
        description=(
            "'complete' if the supervisor believes enough evidence has been gathered "
            "to answer the brief; 'continue' otherwise."
        ),
    )


class ReplanDecision(StrictBaseModel):
    """Replanner agent's decision on whether to adjust the research plan mid-run."""

    should_replan: bool = Field(
        ...,
        description="True if the replanner believes the plan should be adjusted to improve coverage.",
    )
    rationale: str = Field(
        ...,
        description="Explanation of why replanning is or isn't needed.",
    )
    updated_subtopics: list[str] = Field(
        default_factory=list,
        description="Revised subtopic list if replanning; empty if no change.",
    )
    updated_queries: list[str] = Field(
        default_factory=list,
        description="New seed queries to try after replanning; empty if no change.",
    )


class SupervisorCheckpointResult(StrictBaseModel):
    decision: SupervisorDecision = Field(
        default_factory=lambda: SupervisorDecision(rationale="", search_actions=[])
    )
    raw_results: list[RawToolResult] = Field(default_factory=list)
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RelevanceScorerOutput(StrictBaseModel):
    """Structured output from the relevance scorer agent.

    Wraps the scored candidates so the LLM can return the updated list in
    a single, schema-validated payload.
    """

    candidates: list[EvidenceCandidate] = Field(
        ...,
        description=(
            "Full list of candidates with updated relevance_score and matched_subtopics "
            "fields reflecting the scorer's judgement."
        ),
    )


class RelevanceCheckpointResult(StrictBaseModel):
    candidates: list[EvidenceCandidate]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RequestClassification(StrictBaseModel):
    """Classifier agent's interpretation of an incoming research brief.

    Captures audience/freshness framing, a recommended run tier, optional
    clarification state, and extracted user preferences used downstream.
    """

    audience_mode: str = Field(
        ...,
        description="Inferred target audience style (e.g. 'technical', 'executive', 'general').",
    )
    freshness_mode: str = Field(
        ...,
        description="Inferred recency requirement (e.g. 'evergreen', 'recent', 'breaking').",
    )
    recommended_tier: Tier = Field(
        ...,
        description="Suggested effort tier for this brief (quick, standard, deep, custom).",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True when the brief is too ambiguous to run without a follow-up question.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Follow-up question to surface to the user; required when needs_clarification is True.",
    )
    preferences: ResearchPreferences = Field(
        default_factory=ResearchPreferences,
        description="Structured user preferences extracted from the brief (source groups, freshness, biases).",
    )

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
