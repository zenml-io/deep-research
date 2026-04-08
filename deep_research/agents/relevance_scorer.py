from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import RelevanceCheckpointResult
from deep_research.prompts.loader import load_prompt


def build_relevance_scorer_agent(model_name: str):
    """Create a Kitaru-wrapped PydanticAI agent for scoring evidence relevance."""
    return kp.wrap(
        Agent(
            model_name,
            name="relevance_scorer",
            output_type=RelevanceCheckpointResult,
            instructions=load_prompt("relevance_scorer"),
        )
    )
