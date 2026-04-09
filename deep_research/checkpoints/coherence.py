import json

from kitaru import checkpoint

from deep_research.agents.judge import build_coherence_judge_agent
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    CoherenceCheckpointResult,
    CoherenceResult,
    RenderPayload,
    ResearchPlan,
)


@checkpoint(type="llm_call")
def verify_coherence(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    config: ResearchConfig,
) -> CoherenceCheckpointResult:
    """Checkpoint: judge coherence of the final eager renders against the plan."""
    agent = build_coherence_judge_agent(config.judge_model)
    prompt = {
        "renders": [render.model_dump(mode="json") for render in renders],
        "plan": plan.model_dump(mode="json"),
    }
    result = agent.run_sync(json.dumps(prompt, indent=2))
    return CoherenceCheckpointResult(
        coherence=CoherenceResult.model_validate(result.output),
        budget=budget_from_agent_result(
            result,
            ModelPricing.model_validate(config.judge_pricing),
        ),
    )
