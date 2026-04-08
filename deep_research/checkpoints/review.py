import json

from kitaru import checkpoint

from deep_research.agents.reviewer import build_reviewer_agent
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    CritiqueCheckpointResult,
    CritiqueResult,
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
)


@checkpoint(type="llm_call")
def review_renders(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> CritiqueCheckpointResult:
    """Checkpoint: critique eager renders against the current package context."""
    agent = build_reviewer_agent(config.review_model)
    prompt = {
        "renders": [render.model_dump(mode="json") for render in renders],
        "plan": plan.model_dump(mode="json"),
        "selection": selection.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
    }
    result = agent.run_sync(json.dumps(prompt, indent=2))
    return CritiqueCheckpointResult(
        critique=CritiqueResult.model_validate(result.output),
        budget=budget_from_agent_result(
            result,
            ModelPricing.model_validate(config.review_pricing),
        ),
    )
