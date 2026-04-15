"""Assembly checkpoint — materializes InvestigationPackage with mechanical checks.

ZERO LLM calls. This is a @checkpoint(type="tool_call") that computes
derived metadata, validates grounding density, resolves citation IDs
against the evidence ledger, and produces the final InvestigationPackage.
"""

import logging
import re

from kitaru import checkpoint

from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import InvestigationPackage, RunMetadata
from research.contracts.plan import ResearchPlan
from research.contracts.reports import CritiqueReport, DraftReport, FinalReport
from research.prompts import get_prompt_hashes

logger = logging.getLogger(__name__)

# Regex to match [evidence_id] references in report text
_CITATION_RE = re.compile(r"\[([^\]]+)\]")

# Minimum sentence length to count as "substantive" (skip short fragments)
_MIN_SENTENCE_LENGTH = 20


class GroundingError(Exception):
    """Raised when grounding checks fail during assembly."""


class CitationResolutionError(Exception):
    """Raised when citation IDs don't resolve to ledger entries."""


def _extract_citation_ids(text: str) -> set[str]:
    """Extract all [evidence_id] references from markdown text.

    Filters out common markdown patterns that aren't evidence citations
    (e.g. [link text](url), section headers).
    """
    ids: set[str] = set()
    for match in _CITATION_RE.finditer(text):
        candidate = match.group(1)
        # Skip if it looks like a markdown link text (followed by parentheses)
        end = match.end()
        if end < len(text) and text[end] == "(":
            continue
        # Skip common non-citation patterns
        if candidate.startswith("http") or candidate.startswith("#"):
            continue
        # Skip checkbox patterns like [x] or [ ]
        if candidate in ("x", " ", "X"):
            continue
        ids.add(candidate)
    return ids


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for grounding density analysis.

    Simple sentence splitting on period, exclamation, question mark
    followed by whitespace. Only returns substantive sentences
    (>= _MIN_SENTENCE_LENGTH characters after stripping).
    """
    raw = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LENGTH]


def _compute_grounding_density(report_content: str, valid_ids: set[str]) -> float:
    """Compute the fraction of substantive sentences containing valid citations.

    A sentence is "grounded" if it contains at least one [evidence_id]
    that resolves to the valid_ids set.

    Returns a float in [0.0, 1.0]. Returns 1.0 if there are no
    substantive sentences (vacuously true).
    """
    sentences = _split_sentences(report_content)
    if not sentences:
        return 1.0

    grounded = 0
    for sentence in sentences:
        cited = _extract_citation_ids(sentence)
        if cited & valid_ids:  # any valid citation
            grounded += 1

    return grounded / len(sentences)


def _validate_citations(
    report_content: str, ledger: EvidenceLedger
) -> tuple[set[str], set[str]]:
    """Validate that all citation IDs resolve to ledger entries.

    Returns (valid_ids, unresolved_ids).
    """
    ledger_ids = {item.evidence_id for item in ledger.items}
    cited_ids = _extract_citation_ids(report_content)
    valid = cited_ids & ledger_ids
    unresolved = cited_ids - ledger_ids
    return valid, unresolved


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
    grounding_min_ratio: float = 0.7,
    strict_grounding: bool = False,
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
        plan: The research plan.
        ledger: The evidence ledger.
        iterations: Per-iteration records.
        draft: Draft report (may be None).
        critique: Critique report (may be None).
        final_report: Final report (may be None).
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

    # Run grounding checks on report content
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
            valid_ids, unresolved_ids = _validate_citations(report_content, ledger)

            if unresolved_ids:
                raise CitationResolutionError(
                    f"Unresolved citation IDs: {sorted(unresolved_ids)}"
                )

            grounding_density = _compute_grounding_density(report_content, valid_ids)

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
    _UNDERLENGTH_WORD_THRESHOLD = 300
    _UNDERLENGTH_LEDGER_MIN = 5
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

    # Record prompt hashes
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
        ledger=ledger,
        iterations=iterations,
        draft=draft,
        critique=critique,
        final_report=final_report,
        prompt_hashes=prompt_hashes,
    )
