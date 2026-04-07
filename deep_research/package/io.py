import shutil
import importlib
from pathlib import Path

from deep_research.models import InvestigationPackage, RenderPayload


def _sanitize_path_component(value: str, *, field_name: str) -> str:
    """Validate that a string is safe for use as a single path component."""
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


def _write_json(content: str, path: Path) -> None:
    """Write JSON content to disk, creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _reset_directory(path: Path) -> None:
    """Recreate a directory from scratch so generated outputs start clean."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _render_summary_markdown(package: InvestigationPackage) -> str:
    """Render the top-level run summary markdown file."""
    return (
        "# Summary\n\n"
        f"Run ID: {package.run_summary.run_id}\n\n"
        f"Status: {package.run_summary.status}\n"
    )


def _render_plan_markdown(package: InvestigationPackage) -> str:
    """Render the research plan markdown companion file."""
    lines = [
        "# Research Plan",
        "",
        f"Goal: {package.research_plan.goal}",
        "",
        f"Approval status: {package.research_plan.approval_status}",
        "",
        "## Key Questions",
    ]
    lines.extend(f"- {question}" for question in package.research_plan.key_questions)
    lines.extend(["", "## Query Groups"])
    for group_name, queries in package.research_plan.query_groups.items():
        lines.append(f"- {group_name}")
        lines.extend(f"  - {query}" for query in queries)
    return "\n".join(lines) + "\n"


def _render_ledger_markdown(package: InvestigationPackage) -> str:
    """Render a markdown view of the evidence ledger entries."""
    lines = [
        "# Evidence Ledger",
        "",
        f"Entries: {len(package.evidence_ledger.entries)}",
        "",
    ]
    for entry in package.evidence_ledger.entries:
        lines.extend(
            [
                f"## {entry.key}",
                f"- Title: {entry.title}",
                f"- URL: {entry.url}",
                f"- Selected: {'yes' if entry.selected else 'no'}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_full_report(package: InvestigationPackage, run_dir: Path) -> RenderPayload:
    """Generate and write the canonical full report render on demand."""
    renderer_module = importlib.import_module("deep_research.renderers.full_report")
    render = renderer_module.render_full_report(package)
    if render.name != "full_report":
        raise ValueError("Expected render.name to be 'full_report'")
    write_markdown(render.content_markdown, run_dir / "renders" / "full_report.md")
    return render


def write_package(package: InvestigationPackage, output_dir: Path) -> Path:
    """Persist an investigation package as JSON and rendered markdown files."""
    run_dir_name = _sanitize_path_component(
        package.run_summary.run_id,
        field_name="run_id",
    )
    reserved_render_file_names = {"full_report.md"}
    render_file_names = [
        f"{_sanitize_path_component(render.name, field_name='render.name')}.md"
        for render in package.renders
    ]
    normalized_render_file_names = {
        name.casefold() for name in [*render_file_names, *reserved_render_file_names]
    }
    if len(render_file_names) + len(reserved_render_file_names) != len(
        normalized_render_file_names
    ):
        raise ValueError("Duplicate render output filename")

    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _reset_directory(run_dir / "renders")
    _reset_directory(run_dir / "iterations")
    _write_json(
        package.model_dump_json(indent=2),
        run_dir / "package.json",
    )
    write_markdown(_render_summary_markdown(package), run_dir / "summary.md")
    _write_json(
        package.research_plan.model_dump_json(indent=2),
        run_dir / "plan.json",
    )
    write_markdown(_render_plan_markdown(package), run_dir / "plan.md")
    _write_json(
        package.evidence_ledger.model_dump_json(indent=2),
        run_dir / "evidence" / "ledger.json",
    )
    write_markdown(_render_ledger_markdown(package), run_dir / "evidence" / "ledger.md")
    for iteration in package.iteration_trace.iterations:
        _write_json(
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
