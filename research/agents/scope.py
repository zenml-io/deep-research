"""Scope agent — normalizes a raw user request into a ResearchBrief."""

from __future__ import annotations

from research.agents._factory import BudgetAwareAgent, _build_agent
from research.contracts import ResearchBrief


def build_scope_agent(model_name: str) -> BudgetAwareAgent:
    return _build_agent(
        model_name,
        name="scope",
        prompt_name="scope",
        output_type=ResearchBrief,
    )
