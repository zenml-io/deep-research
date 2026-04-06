from pydantic import BaseModel, Field

from deep_research.enums import StopReason, Tier


class ResearchPlan(BaseModel):
    goal: str
    key_questions: list[str]
    subtopics: list[str]
    queries: list[str]
    sections: list[str]
    success_criteria: list[str]


class EvidenceSnippet(BaseModel):
    title: str | None = None
    url: str | None = None
    excerpt: str | None = None


class EvidenceCandidate(BaseModel):
    snippet: EvidenceSnippet
    rationale: str | None = None


class EvidenceLedger(BaseModel):
    entries: list[EvidenceCandidate]


class SelectionItem(BaseModel):
    candidate_index: int | None = None


class SelectionGraph(BaseModel):
    items: list[SelectionItem]


class IterationRecord(BaseModel):
    iteration: int | None = None


class IterationTrace(BaseModel):
    iterations: list[IterationRecord]


class RenderPayload(BaseModel):
    kind: str | None = None
    content: str | None = None


class RunSummary(BaseModel):
    run_id: str
    brief: str
    tier: Tier
    stop_reason: StopReason
    status: str


class InvestigationPackage(BaseModel):
    run_summary: RunSummary
    research_plan: ResearchPlan
    evidence_ledger: EvidenceLedger
    selection_graph: SelectionGraph
    iteration_trace: IterationTrace
    renders: list[RenderPayload]


class CoverageScore(BaseModel):
    subtopic_coverage: float
    source_diversity: float
    evidence_density: float
    total: float


class ToolCallRecord(BaseModel):
    tool_name: str
    status: str
    provider: str | None = None
    summary: str | None = None


class IterationBudget(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RawToolResult(BaseModel):
    tool_name: str
    provider: str
    payload: dict
    ok: bool = True
    error: str | None = None


class SupervisorDecision(BaseModel):
    rationale: str
    search_actions: list[str]


class SupervisorCheckpointResult(BaseModel):
    raw_results: list[RawToolResult]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RelevanceCheckpointResult(BaseModel):
    candidates: list[EvidenceCandidate]
    budget: IterationBudget = Field(default_factory=IterationBudget)


class RequestClassification(BaseModel):
    audience_mode: str
    freshness_mode: str
    recommended_tier: Tier
    needs_clarification: bool = False
    clarification_question: str | None = None
