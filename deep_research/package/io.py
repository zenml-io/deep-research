from pathlib import Path

from deep_research.models import InvestigationPackage


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


def write_package(package: InvestigationPackage, output_dir: Path) -> Path:
    """Persist an investigation package as JSON and rendered markdown files."""
    run_dir_name = _sanitize_path_component(
        package.run_summary.run_id,
        field_name="run_id",
    )
    render_file_names = [
        f"{_sanitize_path_component(render.name, field_name='render.name')}.md"
        for render in package.renders
    ]
    if len(render_file_names) != len(set(render_file_names)):
        raise ValueError("Duplicate render output filename")

    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "package.json").write_text(
        package.model_dump_json(indent=2),
        encoding="utf-8",
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
