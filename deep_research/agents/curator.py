from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import SelectionGraph
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_curator_agent(model_name: str) -> Agent[None, SelectionGraph]:
    """Create a Kitaru-wrapped PydanticAI agent for curating evidence selections."""
    return wrap_agent(
        Agent(
            model_name,
            name="curator",
            output_type=SelectionGraph,
            instructions=load_prompt("curator"),
        )
    )
