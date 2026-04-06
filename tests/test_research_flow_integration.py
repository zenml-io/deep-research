import importlib
import sys
import types

from deep_research.config import ResearchConfig
from deep_research.enums import StopReason, Tier
from deep_research.models import (
    CoverageScore,
    EvidenceLedger,
    IterationBudget,
    RawToolResult,
    RenderPayload,
    ResearchPlan,
    RelevanceCheckpointResult,
    RequestClassification,
    SelectionGraph,
    SupervisorCheckpointResult,
)


def _load_research_flow_module():
    def checkpoint(*, type):
        def decorator(func):
            func._checkpoint_type = type
            func.submit = lambda *args, **kwargs: types.SimpleNamespace(
                load=lambda: func(*args, **kwargs)
            )
            return func

        return decorator

    def flow(func):
        def run(*args, **kwargs):
            return func(*args, **kwargs)

        func.run = run
        return func

    sys.modules["kitaru"] = types.SimpleNamespace(
        checkpoint=checkpoint,
        flow=flow,
        log=lambda **kwargs: None,
        wait=lambda **kwargs: None,
    )
    sys.modules["kitaru.adapters"] = types.SimpleNamespace(
        pydantic_ai=types.SimpleNamespace(wrap=lambda agent, **kwargs: agent)
    )
    sys.modules["pydantic_ai"] = types.SimpleNamespace(Agent=object)
    sys.modules.pop("deep_research.flow.research_flow", None)
    return importlib.import_module("deep_research.flow.research_flow")


def _sample_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Learn Kitaru",
        key_questions=["What is it?"],
        subtopics=["overview"],
        queries=["learn kitaru"],
        sections=["Summary"],
        success_criteria=["Return a package"],
    )


def _sample_supervisor_result() -> SupervisorCheckpointResult:
    return SupervisorCheckpointResult(
        raw_results=[
            RawToolResult(
                tool_name="search",
                provider="test",
                payload={"items": []},
            )
        ],
        budget=IterationBudget(),
    )


def _sample_coverage() -> CoverageScore:
    return CoverageScore(
        subtopic_coverage=1.0,
        source_diversity=1.0,
        evidence_density=1.0,
        total=1.0,
    )


def _patch_success_path(module, monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "build_plan",
        lambda brief, classification, tier: _sample_plan(),
    )
    monkeypatch.setattr(module, "normalize_evidence", lambda raw_results: [])
    monkeypatch.setattr(
        module,
        "score_relevance",
        lambda candidates, plan, config: RelevanceCheckpointResult(
            candidates=[],
            budget=IterationBudget(),
        ),
    )
    monkeypatch.setattr(module, "merge_evidence", lambda scored, ledger: ledger)
    monkeypatch.setattr(
        module, "evaluate_coverage", lambda ledger, plan: _sample_coverage()
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=True,
            reason=StopReason.CONVERGED,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        lambda ledger, plan: SelectionGraph(items=[]),
    )
    monkeypatch.setattr(
        module,
        "render_reading_path",
        lambda selection: RenderPayload(name="reading_path", content_markdown="# RP\n"),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        lambda selection, ledger, plan: RenderPayload(
            name="backing_report",
            content_markdown="# BR\n",
        ),
    )


def test_research_flow_returns_package(monkeypatch) -> None:
    module = _load_research_flow_module()

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        lambda *args, **kwargs: RequestClassification(
            audience_mode="technical",
            freshness_mode="current",
            recommended_tier=Tier.STANDARD,
            needs_clarification=False,
            clarification_question=None,
        ),
    )
    _patch_success_path(module, monkeypatch)
    monkeypatch.setattr(
        module, "run_supervisor", lambda *args, **kwargs: _sample_supervisor_result()
    )

    handle = module.research_flow.run(
        "learn kitaru", config=ResearchConfig.for_tier(Tier.STANDARD)
    )
    result = handle.wait()

    assert result is not None


def test_council_flow_aggregates_multiple_generator_results(monkeypatch) -> None:
    module = _load_research_flow_module()

    class FakeFuture:
        def __init__(self, payload):
            self.payload = payload

        def load(self):
            return self.payload

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        lambda *args, **kwargs: RequestClassification(
            audience_mode="technical",
            freshness_mode="current",
            recommended_tier=Tier.DEEP,
            needs_clarification=False,
            clarification_question=None,
        ),
    )
    _patch_success_path(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "run_council_generator",
        type(
            "G",
            (),
            {
                "submit": staticmethod(
                    lambda *args, **kwargs: FakeFuture(_sample_supervisor_result())
                )
            },
        )(),
    )

    config = ResearchConfig.for_tier(Tier.DEEP).model_copy(
        update={"council_mode": True, "council_size": 3}
    )
    handle = module.research_flow.run("learn kitaru", config=config)
    result = handle.wait()

    assert result is not None
