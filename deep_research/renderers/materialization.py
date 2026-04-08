"""Plain helpers for synthesizing prose from deterministic render scaffolds.

These helpers stay Kitaru-free so lazy package IO can materialize renders without
importing checkpoint-decorated modules. Checkpoint wrappers should live elsewhere.
"""

import json
import re

from deep_research.config import ModelPricing
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import RenderCheckpointResult, RenderPayload
from deep_research.prompts.loader import load_prompt


_CITATION_RE = re.compile(r"\[\d+\]")


def materialize_render_payload(
    scaffold: RenderPayload,
    *,
    writer_model: str,
    prompt_name: str,
    pricing: ModelPricing,
) -> RenderCheckpointResult:
    """Run the writer agent against a scaffold and return validated synthesized prose."""
    from deep_research.agents.writer import build_writer_agent

    agent = build_writer_agent(writer_model)
    prompt = {
        "render_prompt": load_prompt(prompt_name),
        "render_name": scaffold.name,
        "citation_map": scaffold.citation_map,
        "structured_content": scaffold.structured_content,
    }
    result = agent.run_sync(json.dumps(prompt, indent=2))
    render = scaffold.model_copy(
        update={"content_markdown": result.output.content_markdown}
    )
    citations = set(_CITATION_RE.findall(render.content_markdown))
    unknown_citations = sorted(citations.difference(render.citation_map))
    if unknown_citations:
        raise ValueError(
            f"Unknown citation markers in render output: {unknown_citations}"
        )
    return RenderCheckpointResult(
        render=render,
        budget=budget_from_agent_result(result, pricing),
    )
