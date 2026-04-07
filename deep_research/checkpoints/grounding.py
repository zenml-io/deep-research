from kitaru import checkpoint

from deep_research.agents.judge import build_grounding_judge_agent
from deep_research.config import ResearchConfig
from deep_research.models import EvidenceLedger, GroundingResult, RenderPayload


@checkpoint(type="llm_call")
def judge_grounding(
    renders: list[RenderPayload],
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> GroundingResult:
    """Checkpoint: judge citation grounding on the final eager renders."""
    agent = build_grounding_judge_agent(config.judge_model)
    prompt = {
        "renders": [render.model_dump(mode="json") for render in renders],
        "ledger": ledger.model_dump(mode="json"),
    }
    return agent.run_sync(prompt).output
