"""Assembly checkpoint — materializes InvestigationPackage with mechanical checks.

ZERO LLM calls. This is a @checkpoint(type="tool_call") that computes
derived metadata, validates grounding density, resolves citation IDs
against the evidence ledger, and produces the final InvestigationPackage.
"""

import logging

from kitaru import checkpoint

from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import InvestigationPackage, RunMetadata, ToolProviderManifest
from research.contracts.plan import ResearchPlan
from research.contracts.reports import (
    CritiqueReport,
    DraftReport,
    FinalReport,
    VerificationReport,
)
from research.package.grounding import (
    CitationResolutionError,
    GroundingError,
    compute_grounding_density,
    validate_citations,
)
from research.prompts import get_prompt_hashes

logger = logging.getLogger(__name__)

# Non-fatal underlength thresholds
_UNDERLENGTH_WORD_THRESHOLD = 300
_UNDERLENGTH_LEDGER_MIN = 5


@checkpoint(type="tool_call")
def assemble_package(
    metadata: RunMetadata,
    brief: ResearchBrief,
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iterations: list[IterationRecord],
    draft: DraftReport | None,
    critique: CritiqueReport | None,
    final_report: FinalReport | None,
    tool_provider_manifest: ToolProviderManifest,
    revised_plan: ResearchPlan | None = None,
    grounding_min_ratio: float = 0.7,
    strict_grounding: bool = False,
    verification: VerificationReport | None = None,
) -> InvestigationPackage:
    """Checkpoint: assemble the final InvestigationPackage.

    ZERO LLM calls. Runs mechanical checks:
    1. Citation resolution: all [evidence_id] refs must exist in ledger
    2. Grounding density: ratio of grounded sentences >= grounding_min_ratio
    3. Schema validation: InvestigationPackage must be valid

    When ``strict_grounding`` is False (the default), a grounding density
    below threshold logs a warning but does NOT crash.  The density is
    always recorded in ``metadata.grounding_density`` for downstream
    consumers to inspect.

    Args:
        metadata: Run-level metadata.
        brief: The research brief.
        plan: The originally approved research plan.
        revised_plan: Supplemental-loop revised plan, if any.
        ledger: The evidence ledger.
        iterations: Per-iteration records.
        draft: Draft report (may be None).
        critique: Critique report (may be None).
        final_report: Final report (may be None).
        tool_provider_manifest: Durable provider/tool manifest for this run.
        grounding_min_ratio: Minimum grounding density (0.0-1.0).
        strict_grounding: If True, raise GroundingError when density is
            below threshold.  If False (default), log a warning instead.

    Returns:
        A valid InvestigationPackage.

    Raises:
        CitationResolutionError: If any citation IDs don't resolve.
        GroundingError: Only when strict_grounding=True and density is
            below threshold.
    """
    # Determine which report to check (prefer final, fall back to draft)
    report_content = None
    if final_report is not None:
        report_content = final_report.content
    elif draft is not None:
        report_content = draft.content

    grounding_density: float | None = None

    if report_content is not None:
        if not ledger.items:
            # Grounding density is meaningless when the ledger is empty —
            # there's no evidence to cite (e.g. all subagents failed).
            # Produce the package anyway with a warning instead of crashing.
            logger.warning(
                "Ledger is empty — skipping grounding check. "
                "Report quality is degraded (no evidence to cite)."
            )
        else:
            valid_ids, unresolved_ids = validate_citations(report_content, ledger)

            if unresolved_ids:
                raise CitationResolutionError(
                    f"Unresolved citation IDs: {sorted(unresolved_ids)}"
                )

            grounding_density = compute_grounding_density(report_content, valid_ids)

            if grounding_density < grounding_min_ratio:
                if strict_grounding:
                    raise GroundingError(
                        f"Grounding density {grounding_density:.2f} below threshold "
                        f"{grounding_min_ratio:.2f}"
                    )
                logger.warning(
                    "Grounding density %.2f below threshold %.2f "
                    "(strict_grounding=False, proceeding anyway)",
                    grounding_density,
                    grounding_min_ratio,
                )
            else:
                logger.info(
                    "Grounding density: %.2f (threshold: %.2f)",
                    grounding_density,
                    grounding_min_ratio,
                )

    # Non-fatal underlength warning: flag short reports when evidence is plentiful
    if report_content is not None:
        word_count = len(report_content.split())
        if (
            word_count < _UNDERLENGTH_WORD_THRESHOLD
            and len(ledger.items) >= _UNDERLENGTH_LEDGER_MIN
        ):
            logger.warning(
                "Report is short (%d words) relative to evidence volume "
                "(%d ledger items). Expected at least %d words.",
                word_count,
                len(ledger.items),
                _UNDERLENGTH_WORD_THRESHOLD,
            )

    prompt_hashes = get_prompt_hashes()

    # Stamp grounding density into metadata (copy with updated field)
    stamped_metadata = metadata.model_copy(
        update={"grounding_density": grounding_density}
    )

    return InvestigationPackage(
        schema_version="1.0",
        metadata=stamped_metadata,
        brief=brief,
        plan=plan,
        revised_plan=revised_plan,
        ledger=ledger,
        iterations=iterations,
        draft=draft,
        critique=critique,
        final_report=final_report,
        verification=verification,
        prompt_hashes=prompt_hashes,
        tool_provider_manifest=tool_provider_manifest,
    )
