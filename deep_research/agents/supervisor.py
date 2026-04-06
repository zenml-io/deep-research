from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.prompts.loader import load_prompt


def build_supervisor_agent(
    model_name: str, toolsets: list[object], tools: list[object]
):
    return kp.wrap(
        Agent(
            model_name,
            name="supervisor",
            system_prompt=load_prompt("supervisor"),
            toolsets=toolsets,
            tools=tools,
        ),
        tool_capture_config={"mode": "full"},
    )
