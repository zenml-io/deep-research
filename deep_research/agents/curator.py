from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import SelectionGraph
from deep_research.prompts.loader import load_prompt


def build_curator_agent(model_name: str):
    return kp.wrap(
        Agent(
            model_name,
            name="curator",
            result_type=SelectionGraph,
            system_prompt=load_prompt("curator"),
        )
    )
