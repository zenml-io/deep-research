from pathlib import Path

import pytest

from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
)
from deep_research.package.assembly import assemble_package
from deep_research.package.io import read_package, write_markdown, write_package


def make_package(
    *, run_id: str = "run-1", render_names: list[str] | None = None
) -> InvestigationPackage:
    names = render_names or ["reading_path"]

    return InvestigationPackage(
        run_summary={
            "run_id": run_id,
            "brief": "brief",
            "tier": "standard",
            "stop_reason": "max_iterations",
            "status": "completed",
        },
        research_plan={
            "goal": "goal",
            "key_questions": [],
            "subtopics": [],
            "queries": [],
            "sections": [],
            "success_criteria": [],
        },
        evidence_ledger={"entries": []},
        selection_graph={"items": []},
        iteration_trace={"iterations": []},
        renders=[
            {
                "name": name,
                "content_markdown": f"# {name}",
                "citation_map": {},
            }
            for name in names
        ],
    )


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"

    write_markdown("# Title", path)

    assert path.read_text() == "# Title"


def test_write_and_read_package_round_trip(tmp_path: Path) -> None:
    sample_package = make_package()

    run_dir = write_package(sample_package, tmp_path)
    restored = read_package(run_dir)

    assert (run_dir / "renders" / "reading_path.md").read_text(encoding="utf-8") == (
        "# reading_path"
    )
    assert restored == sample_package


@pytest.mark.parametrize("run_id", ["", ".", "..", "run/1", "run\\1"])
def test_write_package_rejects_invalid_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="Unsafe path component"):
        write_package(make_package(run_id=run_id), tmp_path)


@pytest.mark.parametrize(
    "render_name", ["", ".", "..", "notes/part-1", "notes\\part-1"]
)
def test_write_package_rejects_invalid_render_name(
    tmp_path: Path, render_name: str
) -> None:
    with pytest.raises(ValueError, match="Unsafe path component"):
        write_package(make_package(render_names=[render_name]), tmp_path)


def test_write_package_rejects_duplicate_render_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate render output filename"):
        write_package(
            make_package(render_names=["reading_path", "reading_path"]), tmp_path
        )


def test_assemble_package_builds_investigation_package() -> None:
    package = assemble_package(
        run_summary=RunSummary(
            run_id="run-1",
            brief="brief",
            tier="standard",
            stop_reason="max_iterations",
            status="completed",
        ),
        research_plan=ResearchPlan(
            goal="goal",
            key_questions=[],
            subtopics=[],
            queries=[],
            sections=[],
            success_criteria=[],
        ),
        evidence_ledger=EvidenceLedger(entries=[]),
        selection_graph=SelectionGraph(items=[]),
        iteration_trace=IterationTrace(iterations=[]),
        renders=[
            RenderPayload(
                name="reading_path",
                content_markdown="# Reading Path",
                citation_map={},
            )
        ],
    )

    assert isinstance(package, InvestigationPackage)
    assert package.run_summary.run_id == "run-1"
    assert package.renders[0].name == "reading_path"
