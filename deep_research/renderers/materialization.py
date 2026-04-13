"""Plain helpers for synthesizing prose from deterministic render scaffolds.

These helpers stay Kitaru-free so lazy package IO can materialize renders without
importing checkpoint-decorated modules. Checkpoint wrappers should live elsewhere.
"""

import re
from collections.abc import Mapping

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

_FULL_REPORT_SHAPE_REMINDER = (
    "Do not degrade this mode into a generic final report. Honor the requested"
    " deliverable shape and section framing."
)


def _mode_hints(
    mode: DeliverableMode,
    preferences: ResearchPreferences,
    *,
    prompt_name: str,
) -> list[str]:
    """Return ordered guidance strings for the given deliverable mode."""
    parts: list[str] = []

    if mode == DeliverableMode.RESEARCH_PACKAGE:
        return parts

    parts.append(f"Deliverable mode: {mode.value}.")

    if mode == DeliverableMode.COMPARISON_MEMO:
        if preferences.comparison_targets:
            targets = " vs ".join(preferences.comparison_targets)
            parts.append(
                f"This is a comparison between {targets}. Structure the output to"
                " compare these targets fairly with balanced coverage."
            )
        else:
            parts.append(
                "This deliverable is a comparison memo. Compare the strongest named"
                " systems or approaches from the evidence instead of writing a generic"
                " essay. Include implications and limitations."
            )
        parts.append(
            "Use explicit comparison sections. Include a concise comparison overview,"
            " a head-to-head analysis, implications for this repo or user decision,"
            " and a limitations section."
        )
        if prompt_name == "writer_full_report":
            parts.append(_FULL_REPORT_SHAPE_REMINDER)

    elif mode == DeliverableMode.RECOMMENDATION_BRIEF:
        parts.append(
            "The user wants a recommendation. Lead with the recommendation,"
            " then provide supporting evidence and trade-offs."
        )
        parts.append(
            "Use explicit sections for the recommendation, supporting rationale,"
            " alternatives considered, risks, and next steps."
        )
        if prompt_name == "writer_full_report":
            parts.append(_FULL_REPORT_SHAPE_REMINDER)

    elif mode == DeliverableMode.ANSWER_ONLY:
        parts.append(
            "The user wants a concise direct answer. Be brief and focused."
            " Skip extensive background unless critical to the answer."
        )

    if preferences.audience:
        parts.append(f"Target audience: {preferences.audience}.")
    if preferences.freshness:
        parts.append(f"Freshness preference: {preferences.freshness}.")

    return parts


def _prompt_for_mode(
    base_prompt_name: str,
    preferences: ResearchPreferences | None,
) -> str:
    """Return the base prompt with mode-specific guidance prepended when applicable."""
    base_prompt = load_prompt(base_prompt_name)
    if preferences is None:
        return base_prompt
    hints = _mode_hints(
        preferences.deliverable_mode, preferences, prompt_name=base_prompt_name
    )
    if not hints:
        return base_prompt
    return " ".join(hints) + "\n\n" + base_prompt


def _trim_structure(value: object, *, string_budget: int, list_budget: int) -> object:
    if isinstance(value, str):
        return value[:string_budget]
    if isinstance(value, list):
        return [
            _trim_structure(item, string_budget=string_budget, list_budget=list_budget)
            for item in value[:list_budget]
        ]
    if isinstance(value, Mapping):
        return {
            key: _trim_structure(
                nested, string_budget=string_budget, list_budget=list_budget
            )
            for key, nested in value.items()
        }
    return value


def _truncate_render_input(
    structured_content: object,
    *,
    max_chars: int | None,
    snippet_budget_chars: int,
) -> object:
    if max_chars is None:
        return structured_content
    current = structured_content
    for list_budget in (50, 25, 10, 5, 3):
        candidate = _trim_structure(
            current,
            string_budget=snippet_budget_chars,
            list_budget=list_budget,
        )
        if len(serialize_prompt_payload(candidate, label="writer input")) <= max_chars:
            return candidate
        current = candidate
    return current


def materialize_render_payload(
    scaffold: RenderPayload,
    *,
    writer_model: str,
    prompt_name: str,
    pricing: ModelPricing,
    preferences: ResearchPreferences | None = None,
    max_context_chars: int | None = None,
    snippet_budget_chars: int = 600,
) -> RenderCheckpointResult:
    """Run the writer agent against a scaffold and return a validated render result.

    Raises ValueError if the writer produces citation markers not present in the
    scaffold's citation_map.
    """
    from deep_research.agents.writer import build_writer_agent

    agent = build_writer_agent(writer_model)
    truncated_input = _truncate_render_input(
        scaffold.structured_content,
        max_chars=max_context_chars,
        snippet_budget_chars=snippet_budget_chars,
    )
    prompt = {
        "trusted_render_guidance": _prompt_for_mode(prompt_name, preferences),
        "trusted_context": {
            "render_name": scaffold.name,
            "citation_map": scaffold.citation_map,
            "preferences": (
                preferences.model_dump(mode="json") if preferences is not None else None
            ),
        },
        "untrusted_render_input": truncated_input,
    }
    result = agent.run_sync(
        serialize_prompt_payload(prompt, label="writer render payload")
    )
    render = scaffold.model_copy(
        update={"content_markdown": result.output.content_markdown}
    )
    unknown_citations = sorted(
        set(_CITATION_RE.findall(render.content_markdown)).difference(render.citation_map)
    )
    if unknown_citations:
        raise ValueError(
            f"Unknown citation markers in render output: {unknown_citations}"
        )
    return RenderCheckpointResult(
        render=render,
        budget=budget_from_agent_result(result, pricing),
    )
