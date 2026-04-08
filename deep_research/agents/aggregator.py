from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import InvestigationPackage
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_aggregator_agent(model_name: str) -> Agent[None, InvestigationPackage]:
    """Create a Kitaru-wrapped PydanticAI agent for aggregating investigation results."""
    return wrap_agent(
        Agent(
            model_name,
            name="aggregator",
            output_type=InvestigationPackage,
            instructions=load_prompt("aggregator"),
        )
    )
