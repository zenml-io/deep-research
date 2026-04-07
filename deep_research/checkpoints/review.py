from kitaru import checkpoint

from deep_research.agents.reviewer import build_reviewer_agent
from deep_research.config import ResearchConfig
from deep_research.models import (
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
) -> CritiqueResult:
    """Checkpoint: critique eager renders against the current package context."""
    agent = build_reviewer_agent(config.review_model)
    prompt = {
        "renders": [render.model_dump(mode="json") for render in renders],
        "plan": plan.model_dump(mode="json"),
        "selection": selection.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
    }
    return agent.run_sync(prompt).output
