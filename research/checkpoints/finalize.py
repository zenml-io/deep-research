"""Finalize checkpoint — applies critique to produce a FinalReport."""

import json
import logging

from kitaru import checkpoint

from research.agents.finalizer import build_finalizer_agent
from research.contracts.reports import CritiqueReport, DraftReport, FinalReport

logger = logging.getLogger(__name__)


@checkpoint(type="llm_call")
def run_finalize(
    draft: DraftReport,
    critique: CritiqueReport,
    model_name: str,
) -> FinalReport | None:
    """Checkpoint: apply critique to draft, producing a FinalReport.

    On failure, returns None. The flow should check
    ``allow_unfinalized_package`` to decide whether to ship
    draft+critique without a final report.

    Args:
        draft: The draft report.
        critique: The critique report to apply.
        model_name: PydanticAI model string for the finalizer agent.

    Returns:
        A FinalReport, or None if the finalizer failed.
    """
    try:
        agent = build_finalizer_agent(model_name)
        prompt = json.dumps(
            {
                "draft": draft.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
            },
            indent=2,
        )
        return agent.run_sync(prompt).output
    except Exception as exc:
        logger.warning("Finalizer failed: %s — draft and critique preserved", exc)
        return None
