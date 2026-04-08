from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import RenderProse
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_writer_agent(model_name: str) -> Agent[None, RenderProse]:
    """Create a Kitaru-wrapped PydanticAI agent for rendering report content."""
    return wrap_agent(
        Agent(
            model_name,
            name="writer",
            output_type=RenderProse,
            instructions=load_prompt("writer"),
        )
    )
