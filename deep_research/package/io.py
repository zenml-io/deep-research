import shutil
from pathlib import Path

from deep_research.config import ModelPricing, ResearchConfig
from deep_research.models import InvestigationPackage, RenderPayload
from deep_research.renderers.full_report import (
    render_full_report as build_full_report_scaffold,
)
from deep_research.renderers.materialization import materialize_render_payload


def sanitize_path_component(value: str, *, field_name: str) -> str:
    """Validate a single path segment and reject traversal, separators, or padding.

    The package writer uses run IDs and render names as file-system path components.
    This helper enforces that each value is one clean segment, not an empty value,
    reserved traversal token, or a string that smuggles in directory separators.
    """
    if value in {"", ".", ".."}:
        raise ValueError(f"Unsafe path component for {field_name}: {value!r}")
    if value != value.strip():
        raise ValueError(f"Unsafe path component for {field_name}: {value!r}")
    if any(separator in value for separator in ("/", "\\")):
        raise ValueError(f"Unsafe path component for {field_name}: {value!r}")
    return value


def write_markdown(content: str, path: Path) -> None:
    """Write markdown content to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(content: str, path: Path) -> None:
    """Write serialized JSON text to disk after ensuring the parent directory exists.

    This helper mirrors `write_markdown()` so package persistence can create nested
    output folders on demand before emitting stable JSON artifacts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def reset_directory(path: Path) -> None:
    """Replace a generated-output directory with an empty copy at the same path.

    Package writes are expected to be idempotent, so this helper removes any stale
    render or iteration files from a prior run before recreating the directory.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
# This package IO helper may execute an LLM call outside Kitaru tracking after the
# main flow has completed.
def write_full_report(
    package: InvestigationPackage,
    run_dir: Path,
    config: ResearchConfig | None = None,
) -> RenderPayload:
    """Generate and write the canonical full report render on demand."""
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
    write_markdown(render.content_markdown, run_dir / "renders" / "full_report.md")
    return render


def write_package(package: InvestigationPackage, output_dir: Path) -> Path:
    """Persist an investigation package as JSON and rendered markdown files."""
    run_dir_name = sanitize_path_component(
        package.run_summary.run_id,
        field_name="run_id",
    )
    render_file_names = [
        f"{sanitize_path_component(render.name, field_name='render.name')}.md"
        for render in package.renders
    ]
    normalized_render_file_names = {name.casefold() for name in render_file_names}
    if len(render_file_names) != len(normalized_render_file_names):
        raise ValueError("Duplicate render output filename")

    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    reset_directory(run_dir / "renders")
    reset_directory(run_dir / "iterations")
    write_json(
        package.model_dump_json(indent=2),
        run_dir / "package.json",
    )
    write_markdown(
        (
            "# Summary\n\n"
            f"Run ID: {package.run_summary.run_id}\n\n"
            f"Status: {package.run_summary.status}\n"
        ),
        run_dir / "summary.md",
    )
    write_json(
        package.research_plan.model_dump_json(indent=2),
        run_dir / "plan.json",
    )
    plan_lines = [
        "# Research Plan",
        "",
        f"Goal: {package.research_plan.goal}",
        "",
        f"Approval status: {package.research_plan.approval_status}",
        "",
        "## Key Questions",
    ]
    plan_lines.extend(
        f"- {question}" for question in package.research_plan.key_questions
    )
    plan_lines.extend(["", "## Query Groups"])
    for group_name, queries in package.research_plan.query_groups.items():
        plan_lines.append(f"- {group_name}")
        plan_lines.extend(f"  - {query}" for query in queries)
    write_markdown("\n".join(plan_lines) + "\n", run_dir / "plan.md")
    write_json(
        package.evidence_ledger.model_dump_json(indent=2),
        run_dir / "evidence" / "ledger.json",
    )
    ledger_lines = [
        "# Evidence Ledger",
        "",
        f"Entries: {len(package.evidence_ledger.entries)}",
        "",
    ]
    for entry in package.evidence_ledger.entries:
        ledger_lines.extend(
            [
                f"## {entry.key}",
                f"- Title: {entry.title}",
                f"- URL: {entry.url}",
                f"- Selected: {'yes' if entry.selected else 'no'}",
            ]
        )
    write_markdown("\n".join(ledger_lines) + "\n", run_dir / "evidence" / "ledger.md")
    for iteration in package.iteration_trace.iterations:
        write_json(
            iteration.model_dump_json(indent=2),
            run_dir / "iterations" / f"{iteration.iteration:03d}.json",
        )
    for render, render_file_name in zip(
        package.renders, render_file_names, strict=True
    ):
        write_markdown(render.content_markdown, run_dir / "renders" / render_file_name)
    return run_dir


def read_package(run_dir: Path) -> InvestigationPackage:
    """Load an InvestigationPackage from a previously written run directory."""
    return InvestigationPackage.model_validate_json(
        (run_dir / "package.json").read_text(encoding="utf-8")
    )
