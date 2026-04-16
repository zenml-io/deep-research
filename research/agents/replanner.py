"""Replanner agent — revises a ResearchPlan after critique feedback."""

from __future__ import annotations

from research.agents._factory import BudgetAwareAgent, _build_agent
from research.contracts import ResearchPlan


def build_replanner_agent(model_name: str) -> BudgetAwareAgent:
    return _build_agent(
        model_name,
        name="replanner",
        prompt_name="replanner",
        output_type=ResearchPlan,
    )
