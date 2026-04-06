import importlib
import sys
import types
from contextlib import contextmanager

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


def _as_checkpoint(func):
    """Add .submit() to a plain function so it behaves like a Kitaru checkpoint."""

    def submit(*args, after=None, id=None, **kwargs):
        result = func(*args, **kwargs)
        return types.SimpleNamespace(load=lambda: result)

    func.submit = submit
    return func


@contextmanager
def _preserve_modules(*names: str):
    sentinel = object()
    originals = {name: sys.modules.get(name, sentinel) for name in names}
    try:
        yield
    finally:
        for name, value in originals.items():
            if value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_research_flow_module():
    class FakeHandle:
        def __init__(self, value):
            self._value = value

        def wait(self):
            return self._value

    def checkpoint(*, type):
        def decorator(func):
            func._checkpoint_type = type

            def submit(*args, after=None, id=None, **kwargs):
                result = func(*args, **kwargs)
                return types.SimpleNamespace(load=lambda: result)

            func.submit = submit
            return func

        return decorator

    def flow(func):
        def run(*args, **kwargs):
            return FakeHandle(func(*args, **kwargs))

        func.run = run
        return func

    with _preserve_modules("kitaru", "kitaru.adapters", "pydantic_ai"):
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
        _as_checkpoint(lambda brief, classification, tier: _sample_plan()),
    )
    monkeypatch.setattr(
        module,
        "normalize_evidence",
        _as_checkpoint(lambda raw_results: []),
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        _as_checkpoint(
            lambda candidates, plan, config: RelevanceCheckpointResult(
                candidates=[],
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "merge_evidence",
        _as_checkpoint(lambda scored, ledger: ledger),
    )
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        _as_checkpoint(lambda ledger, plan: _sample_coverage()),
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
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "render_reading_path",
        _as_checkpoint(
            lambda selection: RenderPayload(
                name="reading_path", content_markdown="# RP\n"
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        _as_checkpoint(
            lambda selection, ledger, plan: RenderPayload(
                name="backing_report",
                content_markdown="# BR\n",
            )
        ),
    )


def test_load_research_flow_module_restores_stubbed_modules() -> None:
    sentinel_kitaru = object()
    sentinel_adapters = object()
    sentinel_pydantic_ai = object()

    original_kitaru = sys.modules.get("kitaru", sentinel_kitaru)
    original_adapters = sys.modules.get("kitaru.adapters", sentinel_adapters)
    original_pydantic_ai = sys.modules.get("pydantic_ai", sentinel_pydantic_ai)

    _load_research_flow_module()

    assert sys.modules.get("kitaru", sentinel_kitaru) is original_kitaru
    assert sys.modules.get("kitaru.adapters", sentinel_adapters) is original_adapters
    assert sys.modules.get("pydantic_ai", sentinel_pydantic_ai) is original_pydantic_ai


def test_research_flow_returns_package(monkeypatch) -> None:
    module = _load_research_flow_module()

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda *args, **kwargs: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
                recommended_tier=Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
            )
        ),
    )
    _patch_success_path(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(lambda *args, **kwargs: _sample_supervisor_result()),
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
        _as_checkpoint(
            lambda *args, **kwargs: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
                recommended_tier=Tier.DEEP,
                needs_clarification=False,
                clarification_question=None,
            )
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
