from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import InvestigationPackage
from deep_research.prompts.loader import load_prompt


def build_aggregator_agent(model_name: str):
    return kp.wrap(
        Agent(
            model_name,
            name="aggregator",
            output_type=InvestigationPackage,
            system_prompt=load_prompt("aggregator"),
        )
    )
