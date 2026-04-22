"""Finalize checkpoint — applies critique to produce a FinalReport."""

import json
import logging

from kitaru import checkpoint

from research.agents.finalizer import build_finalizer_agent
from research.contracts.evidence import EvidenceLedger
from research.contracts.reports import CritiqueReport, DraftReport, FinalReport

logger = logging.getLogger(__name__)


@checkpoint(type="llm_call")
def run_finalize(
    draft: DraftReport,
    critique: CritiqueReport,
    ledger: EvidenceLedger,
    model_name: str,
    stop_reason: str | None = None,
) -> FinalReport | None:
    """Checkpoint: apply critique to draft, producing a FinalReport.

    The finalizer agent returns plain markdown text (``output_type=str``).
    This checkpoint wraps the output in a :class:`FinalReport` by extracting
    section headings from the markdown.

    On failure, returns None. The flow should check
    ``allow_unfinalized_package`` to decide whether to ship
    draft+critique without a final report.

    Args:
        draft: The draft report.
        critique: The critique report to apply.
        ledger: The evidence ledger (for grounded repairs).
        model_name: PydanticAI model string for the finalizer agent.
        stop_reason: Why the research loop terminated.

    Returns:
        A FinalReport, or None if the finalizer failed.
    """
    try:
        agent = build_finalizer_agent(model_name)
        prompt = json.dumps(
            {
                "draft": draft.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
                "ledger": ledger.model_dump(mode="json"),
                "stop_reason": stop_reason,
            },
            indent=2,
        )
        markdown: str = agent.run_sync(prompt).output
        return FinalReport.from_markdown(markdown, stop_reason=stop_reason)
    except Exception as exc:
        logger.warning("Finalizer failed: %s — draft and critique preserved", exc)
        return None
