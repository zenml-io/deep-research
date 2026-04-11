from kitaru import checkpoint

from deep_research.models import CritiqueResult, RenderPayload, ResearchPlan
from deep_research.observability import span


@checkpoint(type="tool_call")
def apply_revisions(
    renders: list[RenderPayload],
    critique: CritiqueResult,
    plan: ResearchPlan,
) -> list[RenderPayload]:
    """Checkpoint: apply one bounded critique-informed revision pass."""
    with span("apply_revisions", render_count=len(renders)):
        del plan
        revised: list[RenderPayload] = []
        for render in renders:
            structured_content = dict(render.structured_content or {})
            structured_content.update(
                {
                    "critique_summary": critique.summary,
                    "revision_suggestions": critique.revision_suggestions,
                }
            )
            revised.append(
                render.model_copy(update={"structured_content": structured_content})
            )
        return revised
