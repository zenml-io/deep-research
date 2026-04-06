from kitaru import checkpoint

from deep_research.agents.planner import build_planner_agent
from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import RequestClassification, ResearchPlan


@checkpoint(type="llm_call")
def build_plan(
    brief: str, classification: RequestClassification, tier: Tier
) -> ResearchPlan:
    """Checkpoint: generate a structured research plan from the brief and tier."""
    del classification
    model_name = ResearchConfig.for_tier(tier).planner_model
    return build_planner_agent(model_name).run_sync(brief).output
