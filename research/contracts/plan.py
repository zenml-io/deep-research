"""Research plan and subagent task contracts."""

from __future__ import annotations

from research.contracts.base import StrictBase


class ResearchPlan(StrictBase):
    """Structured investigation plan derived from a brief.

    Defines the goal, decomposition into subtopics and questions,
    query strategies, expected report sections, and success criteria.
    """

    goal: str
    """High-level objective of the investigation."""

    key_questions: list[str]
    """Core questions the investigation must answer."""

    subtopics: list[str] = []
    """Decomposed subtopics to explore."""

    query_strategies: list[str] = []
    """Search strategies to employ (e.g. 'arxiv keyword search')."""

    sections: list[str] = []
    """Expected sections in the final report."""

    success_criteria: list[str] = []
    """Criteria for determining investigation completeness."""


class SubagentTask(StrictBase):
    """A unit of work assigned to a research subagent.

    Describes what to investigate and hints for search strategy,
    scoped to a single subtopic.
    """

    task_description: str
    """What the subagent should investigate."""

    target_subtopic: str
    """Which subtopic this task addresses."""

    search_strategy_hints: list[str] = []
    """Optional hints for the subagent's search approach."""
