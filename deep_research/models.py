from pydantic import AnyUrl, BaseModel, ConfigDict, Field, model_validator

from deep_research.enums import SourceKind, StopReason, Tier


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchPlan(StrictBaseModel):
    goal: str
    key_questions: list[str]
    subtopics: list[str]
    queries: list[str]
    sections: list[str]
    success_criteria: list[str]


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
    quality_score: float = 0.0
    relevance_score: float = 0.0
    selected: bool = False

    @model_validator(mode="after")
    def validate_scores(self) -> "EvidenceCandidate":
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("relevance_score must be between 0.0 and 1.0")
        return self


class EvidenceLedger(StrictBaseModel):
    entries: list[EvidenceCandidate] = Field(default_factory=list)


class SelectionItem(StrictBaseModel):
    candidate_key: str
    rationale: str


class SelectionGraph(StrictBaseModel):
    items: list[SelectionItem] = Field(default_factory=list)


class IterationRecord(StrictBaseModel):
    iteration: int
    new_candidate_count: int = 0
    coverage: float = 0.0

    @model_validator(mode="after")
    def validate_ranges(self) -> "IterationRecord":
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.new_candidate_count < 0:
            raise ValueError("new_candidate_count must be non-negative")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be between 0.0 and 1.0")
        return self


class IterationTrace(StrictBaseModel):
    iterations: list[IterationRecord] = Field(default_factory=list)


class RenderPayload(StrictBaseModel):
    name: str
    content_markdown: str
    citation_map: dict[str, str] = Field(default_factory=dict)


class RunSummary(StrictBaseModel):
    run_id: str
    brief: str
    tier: Tier
    stop_reason: StopReason
    status: str


class InvestigationPackage(StrictBaseModel):
    run_summary: RunSummary
    research_plan: ResearchPlan
    evidence_ledger: EvidenceLedger
    selection_graph: SelectionGraph
    iteration_trace: IterationTrace
    renders: list[RenderPayload]


class CoverageScore(StrictBaseModel):
    subtopic_coverage: float
    source_diversity: float
    evidence_density: float
    total: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "CoverageScore":
        for value in (
            self.subtopic_coverage,
            self.source_diversity,
            self.evidence_density,
            self.total,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("coverage scores must be between 0.0 and 1.0")
        return self


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
