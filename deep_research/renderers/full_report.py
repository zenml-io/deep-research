from datetime import UTC, datetime

from deep_research.models import InvestigationPackage, RenderPayload


def render_full_report(package: InvestigationPackage) -> RenderPayload:
    """Build a deterministic full-report scaffold from package state."""
    citation_map = {
        f"[{index}]": item.candidate_key
        for index, item in enumerate(package.selection_graph.items, start=1)
    }
    return RenderPayload(
        name="full_report",
        content_markdown="",
        structured_content={
            "brief": package.run_summary.brief,
            "goal": package.research_plan.goal,
            "sections": package.research_plan.sections,
            "selection_items": [
                item.model_dump(mode="json") for item in package.selection_graph.items
            ],
            "selected_candidates": [
                candidate.model_dump(mode="json")
                for candidate in package.evidence_ledger.selected
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
