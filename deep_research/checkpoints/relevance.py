import json

from kitaru import checkpoint

from deep_research.agents.relevance_scorer import build_relevance_scorer_agent
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.evidence.ledger import truncate_ledger_for_context
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    EvidenceCandidate,
    EvidenceLedger,
    RelevanceCheckpointResult,
    ResearchPlan,
)
from deep_research.observability import span


@checkpoint(type="llm_call")
def score_relevance(
    candidates: list[EvidenceCandidate],
    plan: ResearchPlan,
    config: ResearchConfig,
) -> RelevanceCheckpointResult:
    """Checkpoint: score each candidate's relevance to the research plan via LLM."""
    with span("score_relevance", candidate_count=len(candidates)):
        agent = build_relevance_scorer_agent(config.relevance_scorer_model)
        truncated_candidates = truncate_ledger_for_context(
            EvidenceLedger(considered=candidates),
            max_chars=config.relevance_context_budget_chars,
            role="relevance",
            snippet_budget_chars=config.context_snippet_budget_chars,
        ).considered
        prompt = {
            "plan": plan.model_dump(mode="json"),
            "candidates": [
                candidate.model_dump(mode="json") for candidate in truncated_candidates
            ],
        }
        result = agent.run_sync(json.dumps(prompt, indent=2))
        return RelevanceCheckpointResult(
            candidates=result.output.candidates,
            budget=budget_from_agent_result(
                result,
                ModelPricing.model_validate(config.relevance_scorer_pricing),
            ),
        )
