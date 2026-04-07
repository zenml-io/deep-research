from contextlib import contextmanager
from datetime import datetime
import importlib
import sys
import types

from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
    SelectionItem,
)


def _as_checkpoint(func):
    """Add .submit() to a plain function so it behaves like a Kitaru checkpoint."""

    def submit(*args, after=None, id=None, **kwargs):
        result = func(*args, **kwargs)
        return types.SimpleNamespace(load=lambda: result)

    func.submit = submit
    return func


def _install_kitaru_stub() -> None:
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


def _load_module(module_name: str):
    with _preserve_modules("kitaru", "kitaru.adapters", "pydantic_ai"):
        _install_kitaru_stub()
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    return module


def _make_package() -> InvestigationPackage:
    return InvestigationPackage(
        run_summary=RunSummary(
            run_id="run-1",
            brief="Map the package render surface",
            tier="standard",
            stop_reason="converged",
            status="completed",
        ),
        research_plan=ResearchPlan(
            goal="Answer the brief",
            key_questions=["What matters?"],
            subtopics=["core"],
            queries=["example query"],
            sections=["Overview"],
            success_criteria=["Produce a summary"],
        ),
        evidence_ledger=EvidenceLedger(),
        selection_graph=SelectionGraph(
            items=[SelectionItem(candidate_key="source-1", rationale="useful")]
        ),
        iteration_trace=IterationTrace(),
        renders=[],
    )


def _assert_iso8601_timestamp(value: str | None) -> None:
    assert isinstance(value, str)
    datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_load_module_restores_stubbed_dependencies(monkeypatch) -> None:
    original_kitaru = types.ModuleType("kitaru")
    original_kitaru_adapters = types.ModuleType("kitaru.adapters")
    original_pydantic_ai = types.ModuleType("pydantic_ai")

    monkeypatch.setitem(sys.modules, "kitaru", original_kitaru)
    monkeypatch.setitem(sys.modules, "kitaru.adapters", original_kitaru_adapters)
    monkeypatch.setitem(sys.modules, "pydantic_ai", original_pydantic_ai)

    _load_module("deep_research.renderers.reading_path")

    assert sys.modules["kitaru"] is original_kitaru
    assert sys.modules["kitaru.adapters"] is original_kitaru_adapters
    assert sys.modules["pydantic_ai"] is original_pydantic_ai


def test_render_reading_path_returns_markdown_payload() -> None:
    module = _load_module("deep_research.renderers.reading_path")
    selection = SelectionGraph(
        items=[SelectionItem(candidate_key="a", rationale="useful")]
    )

    payload = module.render_reading_path(selection)

    assert payload.name == "reading_path"
    assert "[1] a: useful" in payload.content_markdown
    assert payload.structured_content == {
        "items": [
            {
                "citation": "[1]",
                "candidate_key": "a",
                "rationale": "useful",
            }
        ]
    }
    assert payload.citation_map == {"[1]": "a"}
    _assert_iso8601_timestamp(payload.generated_at)
    assert module.render_reading_path._checkpoint_type == "llm_call"


def test_render_backing_report_returns_goal_and_selected_count() -> None:
    module = _load_module("deep_research.renderers.backing_report")
    selection = SelectionGraph(
        items=[SelectionItem(candidate_key="a", rationale="useful")]
    )
    ledger = EvidenceLedger()
    plan = ResearchPlan(
        goal="Answer the brief",
        key_questions=["What matters?"],
        subtopics=["core"],
        queries=["example query"],
        sections=["Overview"],
        success_criteria=["Produce a summary"],
    )

    payload = module.render_backing_report(selection, ledger, plan)

    assert payload.name == "backing_report"
    assert "Goal: Answer the brief" in payload.content_markdown
    assert "Selected: 1" in payload.content_markdown
    assert "[1] a" in payload.content_markdown
    assert payload.structured_content == {
        "goal": "Answer the brief",
        "selected_count": 1,
        "selection_keys": ["a"],
    }
    assert payload.citation_map == {"[1]": "a"}
    _assert_iso8601_timestamp(payload.generated_at)
    assert module.render_backing_report._checkpoint_type == "llm_call"


def test_render_full_report_returns_lazy_package_payload() -> None:
    module = _load_module("deep_research.renderers.full_report")
    package = _make_package()

    payload = module.render_full_report(package)

    assert payload.name == "full_report"
    assert "Brief: Map the package render surface" in payload.content_markdown
    assert "## Selected Evidence" in payload.content_markdown
    assert "[1] source-1: useful" in payload.content_markdown
    assert payload.structured_content == {
        "run_id": "run-1",
        "selected_count": 1,
        "selection_keys": ["source-1"],
    }
    assert payload.citation_map == {"[1]": "source-1"}
    _assert_iso8601_timestamp(payload.generated_at)
    assert module.render_full_report._checkpoint_type == "llm_call"


def test_research_flow_uses_renderer_checkpoints(monkeypatch) -> None:
    module = _load_module("deep_research.flow.research_flow")
    selection = SelectionGraph(
        items=[SelectionItem(candidate_key="selected-1", rationale="useful")]
    )
    reading_path_calls = []
    backing_report_calls = []

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: ResearchPlan(
                goal="Answer the brief",
                key_questions=["What matters?"],
                subtopics=["core"],
                queries=["example query"],
                sections=["Overview"],
                success_criteria=["Produce a summary"],
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_iteration",
        lambda *args, **kwargs: types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        ),
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
            lambda candidates, plan, config: types.SimpleNamespace(
                candidates=[],
                budget=types.SimpleNamespace(estimated_cost_usd=0.0),
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
        _as_checkpoint(lambda ledger, plan: types.SimpleNamespace(total=1.0)),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=True,
            reason=module.StopReason.CONVERGED,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        _as_checkpoint(lambda ledger, plan, config=None: selection),
    )
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "render_reading_path",
        _as_checkpoint(
            lambda rendered_selection: (
                reading_path_calls.append(rendered_selection)
                or RenderPayload(
                    name="reading_path", content_markdown="# Reading Path\n"
                )
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        _as_checkpoint(
            lambda rendered_selection, rendered_ledger, rendered_plan: (
                backing_report_calls.append(
                    (rendered_selection, rendered_ledger, rendered_plan)
                )
                or RenderPayload(
                    name="backing_report", content_markdown="# Backing Report\n"
                )
            )
        ),
    )

    package = module.research_flow.run("brief")

    assert reading_path_calls == [selection]
    assert backing_report_calls == [
        (selection, EvidenceLedger(), package.research_plan)
    ]
    assert [render.name for render in package.renders] == [
        "reading_path",
        "backing_report",
    ]
