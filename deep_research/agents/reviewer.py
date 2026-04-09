from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import CritiqueResult
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_reviewer_agent(model_name: str) -> Agent[None, CritiqueResult]:
    """Create a Kitaru-wrapped PydanticAI agent for render critique."""
    return wrap_agent(
        Agent(
            model_name,
            name="reviewer",
            output_type=CritiqueResult,
            instructions=load_prompt("reviewer"),
        )
    )
