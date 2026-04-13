from datetime import UTC, datetime

from deep_research.enums import DeliverableMode
from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.models import InvestigationPackage, RenderPayload


def _effective_deliverable_mode(package: InvestigationPackage) -> DeliverableMode:
    if package.preferences is None:
        return DeliverableMode.FINAL_REPORT
    return package.preferences.deliverable_mode


def _effective_sections(
    package: InvestigationPackage, mode: DeliverableMode
) -> list[str]:
    if mode == DeliverableMode.COMPARISON_MEMO:
        return [
            "Executive Summary",
            "Comparison Overview",
            "Head-to-Head Analysis",
            "Implications for This Repo",
            "Recommendations",
            "Limitations",
        ]
    if mode == DeliverableMode.RECOMMENDATION_BRIEF:
        return [
            "Recommendation",
            "Why This Direction",
            "Alternatives Considered",
            "Risks and Caveats",
            "Next Steps",
            "Limitations",
        ]

    return [section for section in package.research_plan.sections if section.strip()]


def _claim_inventory_summary(package: InvestigationPackage) -> dict[str, object] | None:
    inventory = package.claim_inventory
    if inventory is None:
        return None
    return {
        "total_claims": inventory.total_claims,
        "supported_ratio": inventory.supported_ratio,
        "unsupported_ratio": inventory.unsupported_ratio,
        "trivial_ratio": inventory.trivial_ratio,
        "per_subtopic_coverage": inventory.per_subtopic_coverage,
        "claims": [claim.model_dump(mode="json") for claim in inventory.claims],
    }


def _repo_gap_hints(package: InvestigationPackage) -> list[str]:
    gaps: list[str] = []
    if package.grounding_result is not None and package.grounding_result.score < 0.8:
        gaps.append(
            "Grounding remains weak; recommendations should stay conservative and call out verification gaps."
        )
    if package.coherence_result is not None and package.coherence_result.completeness < 0.8:
        gaps.append(
            "Coverage is incomplete; identify which subtopics still lack direct evidence."
        )
    if package.selection_graph.gap_coverage_summary:
        gaps.extend(package.selection_graph.gap_coverage_summary)
    return gaps


def _render_report(
    package: InvestigationPackage,
    *,
    render_name: str,
    mode: DeliverableMode,
) -> RenderPayload:
    citation_map = {
        f"[{index}]": item.candidate_key
        for index, item in enumerate(package.selection_graph.items, start=1)
    }
    selected_candidates = resolve_selected_entries(package.evidence_ledger)
    return RenderPayload(
        name=render_name,
        content_markdown="",
        structured_content={
            "brief": package.run_summary.brief,
            "goal": package.research_plan.goal,
            "sections": _effective_sections(package, mode),
            "selection_items": [
                item.model_dump(mode="json") for item in package.selection_graph.items
            ],
            "selected_candidates": [
                candidate.model_dump(mode="json")
                for candidate in selected_candidates
            ],
            "iteration_trace": [
                iteration.model_dump(mode="json")
                for iteration in package.iteration_trace.iterations
            ],
            "stop_reason": package.run_summary.stop_reason.value,
        },
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def render_full_report(package: InvestigationPackage) -> RenderPayload:
    """Build a deterministic full-report scaffold from package state."""
    return _render_report(
        package,
        render_name="full_report",
        mode=_effective_deliverable_mode(package),
    )


def render_comparison_memo(package: InvestigationPackage) -> RenderPayload:
    """Build a deterministic comparison-memo scaffold from package state."""
    return _render_report(
        package,
        render_name="comparison_memo",
        mode=DeliverableMode.COMPARISON_MEMO,
    )


def render_recommendation_brief(package: InvestigationPackage) -> RenderPayload:
    """Build a deterministic recommendation-brief scaffold from package state."""
    return _render_report(
        package,
        render_name="recommendation_brief",
        mode=DeliverableMode.RECOMMENDATION_BRIEF,
    )
