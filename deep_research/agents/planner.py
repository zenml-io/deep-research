from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import ResearchPlan
from deep_research.prompts.loader import load_prompt


def build_planner_agent(model_name: str):
    return kp.wrap(
        Agent(
            model_name,
            name="planner",
            output_type=ResearchPlan,
            system_prompt=load_prompt("planner"),
        )
    )
