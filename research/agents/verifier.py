"""Verifier agent — checks report claims against the evidence ledger."""

from __future__ import annotations

from research.agents._factory import BudgetAwareAgent, _build_agent
from research.contracts.reports import VerificationReport


def build_verifier_agent(model_name: str) -> BudgetAwareAgent:
    return _build_agent(
        model_name,
        name="verifier",
        prompt_name="verifier",
        output_type=VerificationReport,
    )
