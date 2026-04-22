"""Supervisor decisions and subagent findings contracts."""

from __future__ import annotations

from research.contracts.base import StrictBase
from research.contracts.plan import SubagentTask


class SubagentFindings(StrictBase):
    """Per-task synthesis results from a subagent.

    Captures the distilled findings, source references, excerpts,
    and optional confidence notes produced by a single subagent run.
    """

    findings: list[str]
    """Distilled findings from the subagent's research."""

    source_references: list[str] = []
    """References to sources consulted."""

    excerpts: list[str] = []
    """Verbatim excerpts supporting the findings."""

    confidence_notes: str | None = None
    """Optional notes on confidence/reliability of findings."""


class SupervisorDecision(StrictBase):
    """Supervisor's decision after evaluating the current research state.

    Indicates whether to continue or stop, what gaps remain,
    and what subagent tasks to dispatch next.
    """

    done: bool
    """Whether the supervisor considers the investigation complete."""

    rationale: str
    """Explanation for the decision."""

    gaps: list[str] = []
    """Identified gaps in current coverage."""

    subagent_tasks: list[SubagentTask] = []
    """Tasks to dispatch to subagents in the next iteration."""

    pinned_evidence_ids: list[str] = []
    """Evidence IDs the supervisor wants to preserve/highlight."""
