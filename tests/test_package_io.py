from pathlib import Path

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


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"

    write_markdown("# Title", path)

    assert path.read_text() == "# Title"


def test_write_and_read_package_round_trip(tmp_path: Path) -> None:
    sample_package = InvestigationPackage(
        run_summary={
            "run_id": "run-1",
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
                "name": "reading_path",
                "content_markdown": "# Reading Path",
                "citation_map": {},
            }
        ],
    )

    run_dir = write_package(sample_package, tmp_path)
    restored = read_package(run_dir)

    assert restored == sample_package


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
