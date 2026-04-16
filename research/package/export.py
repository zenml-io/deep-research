"""Serialize and deserialize InvestigationPackage to/from disk.

Public API
----------
- ``write_package`` — writes a structured directory with JSON artifacts.
- ``read_package`` — reads back from ``package.json``.
- ``sanitize_path_component`` — validates a string for safe use in paths.
"""

from __future__ import annotations

from pathlib import Path

from research.contracts.package import InvestigationPackage


def resolve_package_run_dir(output_dir: Path | str, run_id: str) -> Path:
    """Resolve the durable run directory for *run_id* under *output_dir*."""
    safe_run_id = sanitize_path_component(run_id, field_name="run_id")
    return Path(output_dir) / safe_run_id


def sanitize_path_component(value: str, *, field_name: str) -> str:
    """Validate *value* for safe use as a single path component.

    Rejects empty strings, ``"."``, ``".."``, strings containing
    path separators (``/`` or ``\\``), and strings with leading or
    trailing whitespace.

    Raises
    ------
    ValueError
        If *value* is unsafe.
    """
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(
            f"{field_name} must not have leading/trailing whitespace: {value!r}"
        )
    if value == "":
        raise ValueError(f"{field_name} must not be empty")
    if value in (".", ".."):
        raise ValueError(f"{field_name} must not be '.' or '..': {value!r}")
    if "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must not contain path separators: {value!r}")
    return value


def _write_text(content: str, path: Path) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_package(
    package: InvestigationPackage,
    output_dir: Path,
) -> Path:
    """Write *package* as a structured directory tree.

    Creates ``<output_dir>/<run_id>/`` containing:

    - ``package.json`` — full serialized package
    - ``report.md`` — final report, draft fallback, or placeholder
    - ``evidence/ledger.json`` — serialized evidence ledger
    - ``iterations/<NNN>.json`` — per-iteration snapshots

    Returns the path to the run directory.
    """
    run_dir = resolve_package_run_dir(output_dir, package.metadata.run_id)

    # Full package
    _write_text(package.model_dump_json(indent=2), run_dir / "package.json")

    # Report (best available)
    if package.final_report is not None:
        report_content = package.final_report.content
    elif package.draft is not None:
        report_content = package.draft.content
    else:
        report_content = "_No report was generated for this research run._"
    _write_text(report_content, run_dir / "report.md")

    # Evidence ledger
    _write_text(
        package.ledger.model_dump_json(indent=2),
        run_dir / "evidence" / "ledger.json",
    )

    # Per-iteration snapshots
    for record in package.iterations:
        filename = f"{record.iteration_index:03d}.json"
        _write_text(
            record.model_dump_json(indent=2),
            run_dir / "iterations" / filename,
        )

    return run_dir


def read_package(run_dir: Path) -> InvestigationPackage:
    """Deserialize an ``InvestigationPackage`` from *run_dir*.

    Reads ``<run_dir>/package.json`` and validates via Pydantic.

    Raises
    ------
    FileNotFoundError
        If *run_dir* or ``package.json`` does not exist.
    """
    package_path = Path(run_dir) / "package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"No package.json found in {run_dir}")
    raw = package_path.read_text(encoding="utf-8")
    return InvestigationPackage.model_validate_json(raw)
