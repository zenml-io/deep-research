from deep_research.models import EvidenceLedger, ResearchPlan
from deep_research.tools.bash_executor import run_bash
from deep_research.tools.state_reader import read_gaps, read_plan


def test_read_plan_returns_serializable_plan() -> None:
    plan = ResearchPlan(
        goal="g",
        key_questions=[],
        subtopics=["a"],
        queries=[],
        sections=[],
        success_criteria=[],
    )

    result = read_plan(plan)

    assert result["goal"] == "g"


def test_read_gaps_reports_missing_subtopics() -> None:
    plan = ResearchPlan(
        goal="g",
        key_questions=[],
        subtopics=["a", "b"],
        queries=[],
        sections=[],
        success_criteria=[],
    )
    ledger = EvidenceLedger(entries=[])

    assert read_gaps(plan, ledger) == ["a", "b"]


def test_run_bash_blocks_dangerous_commands() -> None:
    result = run_bash("rm -rf /")

    assert result.ok is False
