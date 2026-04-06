from deep_research.models import EvidenceCandidate, EvidenceLedger, ResearchPlan
from deep_research.tools.bash_executor import run_bash
from deep_research.tools.state_reader import read_gaps, read_plan


def test_read_plan_returns_serializable_plan() -> None:
    plan = ResearchPlan(
        goal="g",
        key_questions=["k1"],
        subtopics=["a"],
        queries=["q1"],
        sections=["s1"],
        success_criteria=["c1"],
    )

    result = read_plan(plan)

    assert result == plan.model_dump()


def test_read_gaps_reports_missing_subtopics() -> None:
    plan = ResearchPlan(
        goal="g",
        key_questions=[],
        subtopics=["a", "b"],
        queries=[],
        sections=[],
        success_criteria=[],
    )
    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="k1",
                title="b",
                url="https://example.com/b",
                provider="test",
                source_kind="web",
            )
        ]
    )

    assert read_gaps(plan, ledger) == ["a"]


def test_run_bash_returns_success_for_allowlisted_command() -> None:
    result = run_bash("pwd")

    assert result.ok is True
    assert result.tool_name == "run_bash"
    assert result.provider == "bash"
    assert result.payload["returncode"] == 0
    assert isinstance(result.payload["stdout"], str)


def test_run_bash_returns_failed_result_for_malformed_command() -> None:
    result = run_bash('"')

    assert result.ok is False
    assert result.error is not None


def test_run_bash_blocks_interpreter_bypass_attempts() -> None:
    result = run_bash("python3 -c 'print(1)'")

    assert result.ok is False
    assert result.error == "command not allowed"
