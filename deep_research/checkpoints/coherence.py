from kitaru import checkpoint

from deep_research.agents.judge import build_coherence_judge_agent
from deep_research.config import ResearchConfig
from deep_research.models import CoherenceResult, RenderPayload, ResearchPlan


@checkpoint(type="llm_call")
def judge_coherence(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    config: ResearchConfig,
) -> CoherenceResult:
    """Checkpoint: judge coherence of the final eager renders against the plan."""
    agent = build_coherence_judge_agent(config.judge_model)
    prompt = {
        "renders": [render.model_dump(mode="json") for render in renders],
        "plan": plan.model_dump(mode="json"),
    }
    return agent.run_sync(prompt).output
