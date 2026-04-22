"""Iteration record contract — captures one research loop cycle."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from research.contracts.base import StrictBase
from research.contracts.decisions import SubagentFindings, SupervisorDecision


class IterationRecord(StrictBase):
    """Record of a single research iteration.

    Captures the supervisor decision, subagent results, and
    iteration-level metrics for auditability.
    """

    iteration_index: Annotated[int, Field(ge=0)]
    """Zero-based index of this iteration."""

    supervisor_decision: SupervisorDecision
    """The supervisor's decision for this iteration."""

    subagent_results: list[SubagentFindings] = []
    """Results from subagents dispatched this iteration."""

    ledger_size: Annotated[int, Field(ge=0)] = 0
    """Number of evidence items in the ledger after this iteration."""

    supervisor_done_ignored: bool = False
    """Whether supervisor `done=True` was observed but ignored by config."""

    cost_usd: Annotated[float, Field(ge=0.0)] = 0.0
    """Cost in USD incurred during this iteration."""

    duration_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    """Wall-clock duration of this iteration in seconds."""
