import json

from kitaru import checkpoint

from deep_research.agents.judge import build_grounding_judge_agent
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    EvidenceLedger,
    GroundingCheckpointResult,
    GroundingResult,
    RenderPayload,
)


@checkpoint(type="llm_call")
def judge_grounding(
    renders: list[RenderPayload],
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> GroundingCheckpointResult:
    """Checkpoint: judge citation grounding on the final eager renders."""
    agent = build_grounding_judge_agent(config.judge_model)
    prompt = {
        "renders": [render.model_dump(mode="json") for render in renders],
        "ledger": ledger.model_dump(mode="json"),
    }
    result = agent.run_sync(json.dumps(prompt, indent=2))
    return GroundingCheckpointResult(
        grounding=GroundingResult.model_validate(result.output),
        budget=budget_from_agent_result(
            result,
            ModelPricing.model_validate(config.judge_pricing),
        ),
    )
