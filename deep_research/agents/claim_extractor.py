from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import ClaimInventory
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=4)
def build_claim_extractor_agent(model_name: str) -> Agent[None, ClaimInventory]:
    """Create a Kitaru-wrapped PydanticAI agent that extracts and grounds claims."""
    return wrap_agent(
        Agent(
            model_name,
            name="claim_extractor",
            output_type=ClaimInventory,
            instructions=load_prompt("claim_extractor"),
        )
    )
