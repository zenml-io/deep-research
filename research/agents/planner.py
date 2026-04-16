"""Planner agent — decomposes a ResearchBrief into a ResearchPlan."""

from __future__ import annotations

from research.agents._factory import BudgetAwareAgent, _build_agent
from research.contracts import ResearchPlan


def build_planner_agent(model_name: str) -> BudgetAwareAgent:
    return _build_agent(
        model_name,
        name="planner",
        prompt_name="planner",
        output_type=ResearchPlan,
    )
