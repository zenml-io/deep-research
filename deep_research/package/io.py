from pathlib import Path

from deep_research.models import InvestigationPackage


def write_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_package(package: InvestigationPackage, output_dir: Path) -> Path:
    run_dir = output_dir / package.run_summary.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "package.json").write_text(
        package.model_dump_json(indent=2),
        encoding="utf-8",
    )
    for render in package.renders:
        write_markdown(
            render.content_markdown, run_dir / "renders" / f"{render.name}.md"
        )
    return run_dir


def read_package(run_dir: Path) -> InvestigationPackage:
    return InvestigationPackage.model_validate_json(
        (run_dir / "package.json").read_text(encoding="utf-8")
    )
