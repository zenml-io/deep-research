import importlib
import sys
import types

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import (
    EvidenceLedger,
    IterationBudget,
    RawToolResult,
    ResearchPlan,
    SupervisorCheckpointResult,
)


def _clear_modules(*names: str) -> None:
    """Remove modules from sys.modules so tests can force a clean import."""
    for name in names:
        sys.modules.pop(name, None)


def _clear_council_module() -> None:
    """Clear the council checkpoint module before re-importing it in tests."""
    _clear_modules("deep_research.checkpoints.council")


def _install_kitaru_checkpoint_stub(monkeypatch):
    """Install a minimal Kitaru checkpoint decorator and capture decorated names."""
    decorated = []

    def checkpoint(*, type):
        def decorator(func):
            func._checkpoint_type = type
            decorated.append((func.__name__, type))
            return func

        return decorator

    monkeypatch.setitem(
        sys.modules, "kitaru", types.SimpleNamespace(checkpoint=checkpoint)
    )
    return decorated


def _install_supervisor_checkpoint_stub(monkeypatch) -> None:
    """Stub the supervisor checkpoint module so council tests stay isolated."""

    def fake_execute(plan, ledger, iteration, config, uncovered_subtopics=None):
        return SupervisorCheckpointResult(raw_results=[])

    monkeypatch.setitem(
        sys.modules,
        "deep_research.checkpoints.supervisor",
        types.SimpleNamespace(run_supervisor=fake_execute),
    )


def _sample_plan() -> ResearchPlan:
    """Return a multi-subtopic research plan fixture for council aggregation tests.

    Council tests use this plan to ensure override models and merged generator outputs
    operate against realistic plan data instead of an overly trivial placeholder.
    """
    return ResearchPlan(
        goal="Answer the brief",
        key_questions=["What changed?", "Why does it matter?"],
        subtopics=["status", "impact"],
        queries=["topic overview", "topic impact"],
        sections=["Summary", "Details"],
        success_criteria=["Cover both subtopics"],
    )


def _import_council_module():
    """Import the council checkpoint module after clearing any cached copy."""
    _clear_council_module()
    return importlib.import_module("deep_research.checkpoints.council")


def test_aggregate_council_results_flattens_all_generators(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    _install_supervisor_checkpoint_stub(monkeypatch)
    module = _import_council_module()

    grouped = [
        SupervisorCheckpointResult(
            raw_results=[RawToolResult(tool_name="a", provider="m1", payload={})],
            budget=IterationBudget(
                input_tokens=6,
                output_tokens=4,
                total_tokens=10,
                estimated_cost_usd=0.01,
            ),
        ),
        SupervisorCheckpointResult(
            raw_results=[RawToolResult(tool_name="b", provider="m2", payload={})],
            budget=IterationBudget(
                input_tokens=12,
                output_tokens=8,
                total_tokens=20,
                estimated_cost_usd=0.02,
            ),
        ),
    ]

    merged = module.aggregate_council_results(grouped)

    assert [item.tool_name for item in merged.raw_results] == ["a", "b"]
    assert merged.budget.input_tokens == 18
    assert merged.budget.output_tokens == 12
    assert merged.budget.total_tokens == 30
    assert merged.budget.estimated_cost_usd == 0.03


def test_run_council_generator_uses_override_model_when_provided(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    _install_supervisor_checkpoint_stub(monkeypatch)
    module = _import_council_module()

    config = ResearchConfig.for_tier(Tier.STANDARD)
    plan = _sample_plan()
    ledger = EvidenceLedger(entries=[])
    captured = []

    def fake_execute(plan_arg, ledger_arg, iteration_arg, config_arg, uncovered_subtopics=None):
        captured.append(
            {
                "plan": plan_arg,
                "ledger": ledger_arg,
                "iteration": iteration_arg,
                "config": config_arg,
                "uncovered_subtopics": uncovered_subtopics,
            }
        )
        return SupervisorCheckpointResult(raw_results=[])

    monkeypatch.setattr(module, "run_supervisor", fake_execute)

    result = module.run_council_generator(
        plan,
        ledger,
        2,
        model_name="override-supervisor-model",
        config=config,
    )

    assert result == SupervisorCheckpointResult(raw_results=[])
    assert captured == [
        {
            "plan": plan,
            "ledger": ledger,
            "iteration": 2,
            "config": config.model_copy(
                update={"supervisor_model": "override-supervisor-model"}
            ),
            "uncovered_subtopics": None,
        }
    ]
    assert ("run_council_generator", "llm_call") in decorated
