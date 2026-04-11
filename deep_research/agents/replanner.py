from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import ReplanDecision
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=4)
def build_replanner_agent(model_name: str) -> Agent[None, ReplanDecision]:
    """Create a Kitaru-wrapped PydanticAI agent that decides whether to replan."""
    return wrap_agent(
        Agent(
            model_name,
            name="replanner",
            output_type=ReplanDecision,
            instructions=load_prompt("replanner"),
        ),
    )
