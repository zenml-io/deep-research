"""Derived-metadata computation for investigation packages.

Pure functions — no LLM calls, no IO. These compute human-readable
summaries and statistics from an already-assembled package.
"""

from __future__ import annotations

from urllib.parse import urlparse

from research.contracts.package import EvidenceStats, InvestigationPackage


def compute_run_summary(package: InvestigationPackage) -> str:
    """Return a human-readable text summary of the research run.

    Includes question, tier, iteration count, cost, stop reason,
    and evidence/report stats.
    """
    meta = package.metadata
    lines: list[str] = [
        f"Research Run: {meta.run_id}",
        f"Tier: {meta.tier}",
        f"Question: {package.brief.topic}",
        f"Started: {meta.started_at}",
    ]

    if meta.completed_at:
        lines.append(f"Completed: {meta.completed_at}")

    lines.append(f"Iterations: {meta.total_iterations}")
    lines.append(f"Cost: ${meta.total_cost_usd:.4f}")

    if meta.stop_reason:
        lines.append(f"Stop reason: {meta.stop_reason}")

    evidence_count = len(package.ledger.items)
    lines.append(f"Evidence items: {evidence_count}")

    if package.final_report is not None:
        lines.append("Report: final report available")
    elif package.draft is not None:
        lines.append("Report: draft only")
    else:
        lines.append("Report: none")

    return "\n".join(lines)


def compute_evidence_stats(package: InvestigationPackage) -> EvidenceStats:
    """Compute statistics about the evidence ledger."""
    items = package.ledger.items
    domains: set[str] = set()
    providers: set[str] = set()
    items_with_doi = 0
    items_with_arxiv_id = 0
    items_with_url = 0
    iterations: set[int] = set()

    for item in items:
        iterations.add(item.iteration_added)

        if item.provider:
            providers.add(item.provider)

        if item.doi:
            items_with_doi += 1

        if item.arxiv_id:
            items_with_arxiv_id += 1

        url = item.canonical_url or item.url
        if url:
            items_with_url += 1
            parsed = urlparse(url)
            if parsed.hostname:
                domains.add(parsed.hostname)

    return {
        "total_items": len(items),
        "unique_domains": sorted(domains),
        "providers": sorted(providers),
        "items_with_doi": items_with_doi,
        "items_with_arxiv_id": items_with_arxiv_id,
        "items_with_url": items_with_url,
        "iterations_represented": sorted(iterations),
    }
