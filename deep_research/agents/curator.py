from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import SelectionGraph
from deep_research.prompts.loader import load_prompt


def build_curator_agent(model_name: str):
    """Create a Kitaru-wrapped PydanticAI agent for curating evidence selections."""
    return kp.wrap(
        Agent(
            model_name,
            name="curator",
            output_type=SelectionGraph,
            instructions=load_prompt("curator"),
        )
    )
