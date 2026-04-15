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
) -> InvestigationPackage:
    """Checkpoint: assemble the final InvestigationPackage.

    ZERO LLM calls. Runs mechanical checks:
    1. Citation resolution: all [evidence_id] refs must exist in ledger
    2. Grounding density: ratio of grounded sentences >= grounding_min_ratio
    3. Schema validation: InvestigationPackage must be valid

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

    Returns:
        A valid InvestigationPackage.

    Raises:
        CitationResolutionError: If any citation IDs don't resolve.
        GroundingError: If grounding density is below threshold.
    """
    # Determine which report to check (prefer final, fall back to draft)
    report_content = None
    if final_report is not None:
        report_content = final_report.content
    elif draft is not None:
        report_content = draft.content

    # Run grounding checks on report content
    if report_content is not None:
        valid_ids, unresolved_ids = _validate_citations(report_content, ledger)

        if unresolved_ids:
            raise CitationResolutionError(
                f"Unresolved citation IDs: {sorted(unresolved_ids)}"
            )

        density = _compute_grounding_density(report_content, valid_ids)
        if density < grounding_min_ratio:
            raise GroundingError(
                f"Grounding density {density:.2f} below threshold "
                f"{grounding_min_ratio:.2f}"
            )

        logger.info(
            "Grounding density: %.2f (threshold: %.2f)",
            density,
            grounding_min_ratio,
        )

    # Record prompt hashes
    prompt_hashes = get_prompt_hashes()

    return InvestigationPackage(
        schema_version="1.0",
        metadata=metadata,
        brief=brief,
        plan=plan,
        ledger=ledger,
        iterations=iterations,
        draft=draft,
        critique=critique,
        final_report=final_report,
        prompt_hashes=prompt_hashes,
    )
