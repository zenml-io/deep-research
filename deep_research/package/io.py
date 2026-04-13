import shutil
from pathlib import Path

from deep_research.config import ModelPricing, ResearchConfig
from deep_research.models import (
    ClaimInventory,
    EvidenceLedger,
    InvestigationPackage,
    RenderPayload,
    ResearchPlan,
    RunSummary,
)
from deep_research.renderers.full_report import (
    render_full_report as build_full_report_scaffold,
)
from deep_research.renderers.materialization import materialize_render_payload


_RESERVED_RENDER_NAMES = {"full_report.md"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def sanitize_path_component(value: str, *, field_name: str) -> str:
    """Validate a single path segment and reject traversal, separators, or padding."""
    if (
        value in {"", ".", ".."}
        or value != value.strip()
        or any(sep in value for sep in ("/", "\\"))
    ):
        raise ValueError(f"Unsafe path component for {field_name}: {value!r}")
    return value


def _write_text(content: str, path: Path) -> None:
    """Write text to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# Preserved public aliases: both forms are part of the package IO surface.
write_markdown = _write_text
write_json = _write_text


def _write_model_json(model, path: Path) -> None:
    """Serialize a Pydantic model to indented JSON at the given path."""
    _write_text(model.model_dump_json(indent=2), path)


def reset_directory(path: Path) -> None:
    """Replace a generated-output directory with an empty copy at the same path."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _render_summary_markdown(run_summary: RunSummary) -> str:
    """Render a RunSummary as a compact markdown summary section."""
    wall = run_summary.wall_elapsed_seconds
    wall_display = "n/a" if wall is None else str(wall)
    return (
        "# Summary\n\n"
        f"Run ID: {run_summary.run_id}\n\n"
        f"Status: {run_summary.status}\n"
        f"Elapsed seconds: {run_summary.elapsed_seconds}\n"
        f"Active elapsed seconds: {run_summary.active_elapsed_seconds}\n"
        f"Wall elapsed seconds: {wall_display}\n"
    )


def _render_seeded_entities_section(plan: ResearchPlan) -> list[str]:
    """Return markdown lines for the seeded entities block, or an empty list."""
    if plan.seeded_entities is None:
        return []
    s = plan.seeded_entities
    groups = [
        ("Projects", s.projects),
        ("Benchmarks", s.benchmarks),
        ("Products", s.products),
        ("Companies", s.companies),
        ("Key terms", s.key_terms),
    ]
    lines = ["## Seeded Entities", ""]
    lines.extend(
        f"- {label}: {', '.join(values) if values else 'none'}"
        for label, values in groups
    )
    lines.append("")
    return lines


def _render_plan_markdown(plan: ResearchPlan) -> str:
    """Render a ResearchPlan as a structured markdown document."""
    lines = [
        "# Research Plan",
        "",
        f"Goal: {plan.goal}",
        "",
        f"Approval status: {plan.approval_status}",
        "",
        *_render_seeded_entities_section(plan),
        "## Key Questions",
        *[f"- {q}" for q in plan.key_questions],
        "",
        "## Query Groups",
    ]
    for group_name, queries in plan.query_groups.items():
        lines.append(f"- {group_name}")
        lines.extend(f"  - {query}" for query in queries)
    return "\n".join(lines) + "\n"


def _render_ledger_markdown(ledger: EvidenceLedger) -> str:
    """Render an EvidenceLedger as a flat markdown list of entries."""
    lines = [
        "# Evidence Ledger",
        "",
        f"Entries: {len(ledger.entries)}",
        "",
    ]
    for entry in ledger.entries:
        lines.extend(
            [
                f"## {entry.key}",
                f"- Title: {entry.title}",
                f"- URL: {entry.url}",
                f"- Selected: {'yes' if entry.selected else 'no'}",
            ]
        )
    return "\n".join(lines) + "\n"


def render_claim_inventory_markdown(claim_inventory: ClaimInventory) -> str:
    """Render the optional claim inventory as a compact markdown artifact.

    Each claim is rendered with its status, confidence, evidence keys, and
    optional covered subtopics and verification reasoning.
    """
    lines = [
        "# Claim Inventory",
        "",
        f"Total claims: {claim_inventory.total_claims}",
        f"Supported ratio: {claim_inventory.supported_ratio:.2f}",
        f"Unsupported ratio: {claim_inventory.unsupported_ratio:.2f}",
        f"Trivial ratio: {claim_inventory.trivial_ratio:.2f}",
        "",
    ]
    for index, claim in enumerate(claim_inventory.claims, start=1):
        evidence_keys = ", ".join(claim.supporting_candidate_keys) or "none"
        lines.extend(
            [
                f"## Claim {index}",
                claim.claim_text,
                "",
                f"- Status: {claim.support_status}",
                f"- Confidence: {claim.confidence_score:.2f}",
                f"- Evidence keys: {evidence_keys}",
            ]
        )
        if claim.covered_subtopics:
            lines.append(f"- Covered subtopics: {', '.join(claim.covered_subtopics)}")
        if claim.verification_reasoning:
            lines.append(f"- Verification: {claim.verification_reasoning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Package write helpers
# ---------------------------------------------------------------------------


def _resolve_render_file_names(package: InvestigationPackage) -> list[str]:
    """Sanitize and validate render file names, rejecting duplicates and reserved names."""
    names = [
        f"{sanitize_path_component(render.name, field_name='render.name')}.md"
        for render in package.renders
    ]
    normalized = {name.casefold() for name in names}
    if len(names) != len(normalized):
        raise ValueError("Duplicate render output filename")
    if normalized & _RESERVED_RENDER_NAMES:
        raise ValueError(
            "Duplicate render output filename: 'full_report' is reserved"
        )
    return names


def _write_evidence_artifacts(
    package: InvestigationPackage, evidence_dir: Path
) -> None:
    """Write ledger JSON/markdown and optional claim inventory to evidence_dir."""
    _write_model_json(package.evidence_ledger, evidence_dir / "ledger.json")
    _write_text(_render_ledger_markdown(package.evidence_ledger), evidence_dir / "ledger.md")
    if package.claim_inventory is not None:
        _write_model_json(package.claim_inventory, evidence_dir / "claims.json")
        _write_text(
            render_claim_inventory_markdown(package.claim_inventory),
            evidence_dir / "claims.md",
        )


def _write_iteration_artifacts(
    package: InvestigationPackage, iterations_dir: Path
) -> None:
    """Write one JSON file per iteration snapshot under iterations_dir."""
    for iteration in package.iteration_trace.iterations:
        _write_model_json(
            iteration, iterations_dir / f"{iteration.iteration:03d}.json"
        )


def _write_render_artifacts(
    package: InvestigationPackage,
    render_file_names: list[str],
    renders_dir: Path,
) -> None:
    """Write each render's markdown content to renders_dir."""
    for render, file_name in zip(package.renders, render_file_names, strict=True):
        _write_text(render.content_markdown, renders_dir / file_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# This package IO helper may execute an LLM call outside Kitaru tracking after the
# main flow has completed.
def write_full_report(
    package: InvestigationPackage,
    run_dir: Path,
    config: ResearchConfig | None = None,
) -> RenderPayload:
    """Generate and write the canonical full report render on demand.

    Resolves the writer model from config (preferred) or package.render_settings.
    Raises ValueError if neither is available or if the resulting render name
    is not 'full_report'.
    """
    if config is not None:
        writer_model = config.writer_model
    elif package.render_settings is not None:
        writer_model = package.render_settings.writer_model
    else:
        raise ValueError("write_full_report requires config or package.render_settings")
    render = materialize_render_payload(
        build_full_report_scaffold(package),
        writer_model=writer_model,
        prompt_name="writer_full_report",
        pricing=ModelPricing(),
    ).render
    if render.name != "full_report":
        raise ValueError("Expected render.name to be 'full_report'")
    _write_text(render.content_markdown, run_dir / "renders" / "full_report.md")
    return render


def write_package(package: InvestigationPackage, output_dir: Path) -> Path:
    """Persist an investigation package as JSON and rendered markdown files.

    Creates a subdirectory named after the run ID inside output_dir. The
    renders/ and iterations/ subdirectories are always reset (emptied then
    recreated) to avoid stale files from previous writes.

    Returns the path to the run directory.
    """
    run_dir_name = sanitize_path_component(
        package.run_summary.run_id, field_name="run_id"
    )
    render_file_names = _resolve_render_file_names(package)

    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    reset_directory(run_dir / "renders")
    reset_directory(run_dir / "iterations")

    _write_model_json(package, run_dir / "package.json")
    _write_text(_render_summary_markdown(package.run_summary), run_dir / "summary.md")
    _write_model_json(package.research_plan, run_dir / "plan.json")
    _write_text(_render_plan_markdown(package.research_plan), run_dir / "plan.md")

    _write_evidence_artifacts(package, run_dir / "evidence")
    _write_iteration_artifacts(package, run_dir / "iterations")
    _write_render_artifacts(package, render_file_names, run_dir / "renders")

    return run_dir


def read_package(run_dir: Path) -> InvestigationPackage:
    """Load an InvestigationPackage from a previously written run directory."""
    return InvestigationPackage.model_validate_json(
        (run_dir / "package.json").read_text(encoding="utf-8")
    )
