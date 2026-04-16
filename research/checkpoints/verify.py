"""Verification checkpoint — checks report claims against the evidence ledger."""

import json
import logging

from kitaru import checkpoint

from research.agents.verifier import build_verifier_agent
from research.contracts.evidence import EvidenceLedger
from research.contracts.reports import DraftReport, FinalReport, VerificationReport

logger = logging.getLogger(__name__)


@checkpoint(type="llm_call")
def run_verify(
    report: FinalReport | DraftReport,
    ledger: EvidenceLedger,
    model_name: str,
) -> VerificationReport | None:
    """Checkpoint: verify claims in *report* against *ledger*.

    Returns None on verifier failure; flow treats that as "verification skipped"
    and assembles the package with verification=None.
    """
    try:
        agent = build_verifier_agent(model_name)
        prompt = json.dumps(
            {
                "report": report.model_dump(mode="json"),
                "ledger": ledger.model_dump(mode="json"),
            },
            indent=2,
        )
        return agent.run_sync(prompt).output
    except Exception as exc:
        logger.warning("Verifier failed: %s — skipping verification", exc)
        return None
