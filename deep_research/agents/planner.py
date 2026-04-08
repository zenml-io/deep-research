from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import ResearchPlan
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_planner_agent(model_name: str) -> Agent[None, ResearchPlan]:
    """Create a Kitaru-wrapped PydanticAI agent for generating research plans."""
    return wrap_agent(
        Agent(
            model_name,
            name="planner",
            output_type=ResearchPlan,
            instructions=load_prompt("planner"),
        )
    )
