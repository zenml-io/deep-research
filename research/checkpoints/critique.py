"""Critique checkpoint — reviewer agent(s) produce a CritiqueReport.

This is the mandatory provider-crossing checkpoint. On the deep tier,
two reviewers on different providers produce independent critiques that
are merged deterministically (union of issues, scores averaged).
"""

import json
import logging

from kitaru import checkpoint

from research.agents.reviewer import build_reviewer_agent
from research.contracts.evidence import EvidenceLedger
from research.contracts.plan import ResearchPlan
from research.contracts.reports import (
    CritiqueDimensionScore,
    CritiqueReport,
    DraftReport,
)

logger = logging.getLogger(__name__)


def _merge_critiques(a: CritiqueReport, b: CritiqueReport) -> CritiqueReport:
    """Merge two independent critiques deterministically.

    - Issues: union (deduplicated, preserving order)
    - Dimension scores: averaged (matched by dimension name)
    - require_more_research: either True => True
    - Provenance: combined
    """
    # Merge issues (deduplicated, preserving order)
    seen: set[str] = set()
    merged_issues: list[str] = []
    for issue in a.issues + b.issues:
        if issue not in seen:
            seen.add(issue)
            merged_issues.append(issue)

    # Merge dimension scores (average matching dimensions, keep unique ones)
    a_dims = {d.dimension: d for d in a.dimensions}
    b_dims = {d.dimension: d for d in b.dimensions}
    all_dims = sorted(set(a_dims) | set(b_dims))

    merged_dims: list[CritiqueDimensionScore] = []
    for dim_name in all_dims:
        if dim_name in a_dims and dim_name in b_dims:
            avg_score = (a_dims[dim_name].score + b_dims[dim_name].score) / 2.0
            explanation = (
                f"[Reviewer 1] {a_dims[dim_name].explanation} "
                f"[Reviewer 2] {b_dims[dim_name].explanation}"
            )
            merged_dims.append(
                CritiqueDimensionScore(
                    dimension=dim_name,
                    score=avg_score,
                    explanation=explanation,
                )
            )
        elif dim_name in a_dims:
            merged_dims.append(a_dims[dim_name])
        else:
            merged_dims.append(b_dims[dim_name])

    # Merge provenance
    merged_provenance = list(a.reviewer_provenance) + list(b.reviewer_provenance)

    return CritiqueReport(
        dimensions=merged_dims,
        require_more_research=a.require_more_research or b.require_more_research,
        issues=merged_issues,
        reviewer_provenance=merged_provenance,
    )


@checkpoint(type="llm_call")
def run_critique(
    draft: DraftReport,
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    model_name: str,
    second_model_name: str | None = None,
) -> CritiqueReport:
    """Checkpoint: critique a draft report.

    On standard tier: single reviewer on model_name.
    On deep tier: two reviewers on different models (model_name and
    second_model_name). Single reviewer failure is tolerated; both fail = error.

    Args:
        draft: The draft report to critique.
        plan: The research plan (for completeness checking).
        ledger: The evidence ledger (for grounding verification).
        model_name: PydanticAI model string for the primary reviewer.
        second_model_name: Optional second reviewer model (deep tier).

    Returns:
        A CritiqueReport (merged if dual-reviewer).
    """
    prompt = json.dumps(
        {
            "draft": draft.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "ledger": ledger.model_dump(mode="json"),
        },
        indent=2,
    )

    if second_model_name is None:
        # Single reviewer (standard tier)
        agent = build_reviewer_agent(model_name)
        return agent.run_sync(prompt).output

    # Dual reviewer (deep tier)
    results: list[CritiqueReport] = []
    errors: list[Exception] = []

    for i, mn in enumerate([model_name, second_model_name]):
        try:
            agent = build_reviewer_agent(mn)
            result = agent.run_sync(prompt).output
            result = CritiqueReport(
                dimensions=result.dimensions,
                require_more_research=result.require_more_research,
                issues=result.issues,
                reviewer_provenance=[f"reviewer_{i + 1}:{mn}"],
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Reviewer %d (%s) failed: %s", i + 1, mn, exc)
            errors.append(exc)

    if len(results) == 0:
        raise RuntimeError(f"Both reviewers failed: {errors}")

    if len(results) == 1:
        logger.warning("Only one of two reviewers succeeded; using single critique")
        return results[0]

    return _merge_critiques(results[0], results[1])
