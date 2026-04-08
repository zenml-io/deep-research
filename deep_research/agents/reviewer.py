from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import CritiqueResult
from deep_research.prompts.loader import load_prompt


def build_reviewer_agent(model_name: str):
    """Create a Kitaru-wrapped PydanticAI agent for render critique."""
    return kp.wrap(
        Agent(
            model_name,
            name="reviewer",
            output_type=CritiqueResult,
            instructions=load_prompt("reviewer"),
        )
    )
