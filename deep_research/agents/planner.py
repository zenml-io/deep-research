from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import ResearchPlan
from deep_research.prompts.loader import load_prompt


def build_planner_agent(model_name: str):
    """Create a Kitaru-wrapped PydanticAI agent for generating research plans."""
    return kp.wrap(
        Agent(
            model_name,
            name="planner",
            output_type=ResearchPlan,
            instructions=load_prompt("planner"),
        )
    )
