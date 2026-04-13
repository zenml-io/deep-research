"""Critique and judge passes (reviewer + grounding/coherence judges)."""

from __future__ import annotations

from deep_research.config import ResearchConfig
from deep_research.flow._types import (
    CritiqueBundle,
    JudgeBundle,
    _flow,
)
from deep_research.models import (
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
)


def run_critique_if_enabled(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> CritiqueBundle:
    """Run reviewer and apply revisions when critique is enabled."""
    flow = _flow()
    if not config.critique_enabled:
        return CritiqueBundle(renders=renders, critique_result=None, spent_usd=0.0)
    critique_checkpoint = flow.critique_reports.submit(
        renders, plan, selection, ledger, config
    ).load()
    critique_result = critique_checkpoint.critique
    revision_checkpoint = flow.apply_revisions.submit(
        renders, critique_result, plan, config
    ).load()
    return CritiqueBundle(
        renders=revision_checkpoint.renders,
        critique_result=critique_result,
        spent_usd=(
            critique_checkpoint.budget.estimated_cost_usd
            + revision_checkpoint.budget.estimated_cost_usd
        ),
    )


def run_judges_if_enabled(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> JudgeBundle:
    """Run grounding + coherence judges in parallel when enabled."""
    flow = _flow()
    if not config.judge_enabled:
        return JudgeBundle(grounding_result=None, coherence_result=None, spent_usd=0.0)
    grounding_future = flow.verify_grounding.submit(renders, ledger, config)
    coherence_future = flow.verify_coherence.submit(renders, plan, config)
    grounding_checkpoint = grounding_future.load()
    coherence_checkpoint = coherence_future.load()
    return JudgeBundle(
        grounding_result=grounding_checkpoint.grounding,
        coherence_result=coherence_checkpoint.coherence,
        spent_usd=(
            grounding_checkpoint.budget.estimated_cost_usd
            + coherence_checkpoint.budget.estimated_cost_usd
        ),
    )
