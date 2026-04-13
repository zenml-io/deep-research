"""Claim extraction pass (post-render, pre-assembly)."""

from __future__ import annotations

from deep_research.config import ResearchConfig
from deep_research.flow._types import _flow
from deep_research.models import (
    ClaimInventory,
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
)


def run_claim_extraction_if_enabled(
    renders: list[RenderPayload],
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    config: ResearchConfig,
) -> tuple[ClaimInventory | None, float]:
    """Extract claims when enabled for the current tier."""
    if not config.claim_extraction_enabled:
        return None, 0.0
    result = _flow().extract_claims.submit(renders, ledger, plan, config).load()
    return result.inventory, result.budget.estimated_cost_usd
