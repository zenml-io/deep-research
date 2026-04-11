"""Plain helpers for synthesizing prose from deterministic render scaffolds.

These helpers stay Kitaru-free so lazy package IO can materialize renders without
importing checkpoint-decorated modules. Checkpoint wrappers should live elsewhere.
"""

import re

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ModelPricing
from deep_research.enums import DeliverableMode
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    RenderCheckpointResult,
    RenderPayload,
    ResearchPreferences,
)
from deep_research.prompts.loader import load_prompt


_CITATION_RE = re.compile(r"\[\d+\]")


def prompt_for_mode(
    base_prompt_name: str,
    preferences: ResearchPreferences | None,
) -> str:
    """Load trusted mode-specific guidance for writer materialization."""
    base_prompt = load_prompt(base_prompt_name)
    if preferences is None:
        return base_prompt

    parts: list[str] = []
    mode = preferences.deliverable_mode
    if mode != DeliverableMode.RESEARCH_PACKAGE:
        parts.append(f"Deliverable mode: {mode.value}.")

    if mode == DeliverableMode.COMPARISON_MEMO and preferences.comparison_targets:
        targets = " vs ".join(preferences.comparison_targets)
        parts.append(
            f"This is a comparison between {targets}. Structure the output to compare"
            " these targets fairly with balanced coverage."
        )
    elif mode == DeliverableMode.RECOMMENDATION_BRIEF:
        parts.append(
            "The user wants a recommendation. Lead with the recommendation,"
            " then provide supporting evidence and trade-offs."
        )
    elif mode == DeliverableMode.ANSWER_ONLY:
        parts.append(
            "The user wants a concise direct answer. Be brief and focused."
            " Skip extensive background unless critical to the answer."
        )

    if preferences.audience:
        parts.append(f"Target audience: {preferences.audience}.")
    if preferences.freshness:
        parts.append(f"Freshness preference: {preferences.freshness}.")

    preamble = " ".join(parts)
    if not preamble:
        return base_prompt
    return f"{preamble}\n\n{base_prompt}"


def materialize_render_payload(
    scaffold: RenderPayload,
    *,
    writer_model: str,
    prompt_name: str,
    pricing: ModelPricing,
    preferences: ResearchPreferences | None = None,
) -> RenderCheckpointResult:
    """Run the writer agent against a scaffold and return validated synthesized prose."""
    from deep_research.agents.writer import build_writer_agent

    agent = build_writer_agent(writer_model)
    prompt = {
        "trusted_render_guidance": prompt_for_mode(prompt_name, preferences),
        "trusted_context": {
            "render_name": scaffold.name,
            "citation_map": scaffold.citation_map,
            "preferences": (
                preferences.model_dump(mode="json") if preferences is not None else None
            ),
        },
        "untrusted_render_input": scaffold.structured_content,
    }
    result = agent.run_sync(
        serialize_prompt_payload(prompt, label="writer render payload")
    )
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
