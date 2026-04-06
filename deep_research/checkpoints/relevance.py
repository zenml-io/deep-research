from kitaru import checkpoint

from deep_research.agents.relevance_scorer import build_relevance_scorer_agent
from deep_research.config import ResearchConfig
from deep_research.models import (
    EvidenceCandidate,
    RelevanceCheckpointResult,
    ResearchPlan,
)


@checkpoint(type="llm_call")
def score_relevance(
    candidates: list[EvidenceCandidate],
    plan: ResearchPlan,
    config: ResearchConfig,
) -> RelevanceCheckpointResult:
    """Checkpoint: score each candidate's relevance to the research plan via LLM."""
    agent = build_relevance_scorer_agent(config.relevance_scorer_model)
    prompt = {
        "plan": plan.model_dump(mode="json"),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    return agent.run_sync(prompt).output
