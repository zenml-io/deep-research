import ast
from contextlib import contextmanager
from datetime import datetime
import importlib
import inspect
from pathlib import Path
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
    """Install lightweight Kitaru and PydanticAI stubs used to import renderer modules.

    Renderer and flow tests only need the checkpoint and flow decorators to attach test
    helpers, so these stubs avoid pulling in the real runtime integration stack.
    """

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
    """Temporarily preserve selected modules while renderer import-time stubs are active.

    This keeps the import helper from permanently overwriting globally cached modules
    when tests swap in fake Kitaru and PydanticAI implementations.
    """
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
    """Import a module under temporary Kitaru stubs and then restore globals."""
    with _preserve_modules("kitaru", "kitaru.adapters", "pydantic_ai"):
        _install_kitaru_stub()
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    return module


def _make_package() -> InvestigationPackage:
    """Build a compact but realistic investigation package fixture for renderer tests.

    The fixture includes a run summary, plan, selection graph, and empty render list so
    renderer modules can be exercised without depending on the full research flow.
    """
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
    """Assert that a render timestamp is present and ISO-8601 parseable."""
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
                "bridge_note": None,
                "matched_subtopics": [],
                "reading_time_minutes": None,
                "ordering_rationale": None,
            }
        ],
        "gap_coverage_summary": [],
    }
    assert payload.citation_map == {"[1]": "a"}
    _assert_iso8601_timestamp(payload.generated_at)
    assert not hasattr(module.render_reading_path, "_checkpoint_type")
    assert not hasattr(module.render_reading_path, "submit")


def test_render_reading_path_handles_richer_selection_items() -> None:
    module = _load_module("deep_research.renderers.reading_path")
    selection = SelectionGraph(
        items=[
            SelectionItem(
                candidate_key="candidate-1",
                rationale="Foundational source.",
                bridge_note="Read before the operational guide.",
                matched_subtopics=["replay"],
                reading_time_minutes=12,
                ordering_rationale="Most authoritative first.",
            )
        ],
        gap_coverage_summary=["operations"],
    )

    payload = module.render_reading_path(selection)

    assert "Foundational source." in payload.content_markdown
    assert "Bridge: Read before the operational guide." in payload.content_markdown
    assert "Subtopics: replay" in payload.content_markdown
    assert "Uncovered subtopics: operations" in payload.content_markdown
    assert payload.structured_content["items"][0]["candidate_key"] == "candidate-1"
    assert payload.structured_content["gap_coverage_summary"] == ["operations"]


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
    assert not hasattr(module.render_backing_report, "_checkpoint_type")
    assert not hasattr(module.render_backing_report, "submit")


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


def test_renderer_modules_stay_pure_and_checkpoint_wrappers_live_under_checkpoints() -> (
    None
):
    reading_module = _load_module("deep_research.renderers.reading_path")
    backing_module = _load_module("deep_research.renderers.backing_report")
    full_report_module = _load_module("deep_research.renderers.full_report")
    rendering_checkpoint_module = _load_module("deep_research.checkpoints.rendering")

    for module, function_name in (
        (reading_module, "render_reading_path"),
        (backing_module, "render_backing_report"),
        (full_report_module, "render_full_report"),
    ):
        func = getattr(module, function_name)
        assert not hasattr(func, "_checkpoint_type")
        assert not hasattr(func, "submit")

    for function_name in (
        "render_reading_path",
        "render_backing_report",
        "render_full_report",
    ):
        checkpoint_func = getattr(rendering_checkpoint_module, function_name)
        assert checkpoint_func._checkpoint_type == "llm_call"
        assert hasattr(checkpoint_func, "submit")


def test_underscore_prefixed_functions_have_detailed_docstrings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in sorted(
        [
            *repo_root.joinpath("deep_research").glob("**/*.py"),
            *repo_root.joinpath("tests").glob("**/*.py"),
        ]
    ):
        module_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module_ast):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("_"):
                continue
            docstring = ast.get_docstring(node)
            qualified_name = f"{path.relative_to(repo_root)}::{node.name}"
            if not docstring:
                failures.append(f"missing docstring: {qualified_name}")
                continue
            if len(docstring.split()) < 10:
                failures.append(f"thin docstring: {qualified_name} -> {docstring!r}")

    assert not failures, "\n".join(failures)


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
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: types.SimpleNamespace(
                raw_results=[],
                budget=types.SimpleNamespace(estimated_cost_usd=0.0),
            )
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
        _as_checkpoint(lambda scored, ledger, config=None: ledger),
    )
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        _as_checkpoint(
            lambda ledger, plan: types.SimpleNamespace(
                total=1.0, uncovered_subtopics=[]
            )
        ),
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


def test_research_flow_imports_render_checkpoint_wrappers() -> None:
    module = _load_module("deep_research.flow.research_flow")

    assert (
        module.render_reading_path.__module__ == "deep_research.checkpoints.rendering"
    )
    assert (
        module.render_backing_report.__module__ == "deep_research.checkpoints.rendering"
    )


def test_research_flow_passes_config_into_merge_checkpoint(monkeypatch) -> None:
    module = _load_module("deep_research.flow.research_flow")
    merge_calls = []

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
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: types.SimpleNamespace(
                raw_results=[],
                budget=types.SimpleNamespace(estimated_cost_usd=0.0),
            )
        ),
    )
    monkeypatch.setattr(
        module, "normalize_evidence", _as_checkpoint(lambda raw_results: [])
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
        _as_checkpoint(
            lambda scored, ledger, *, config=None: (
                merge_calls.append((scored, ledger, config)) or ledger
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        _as_checkpoint(
            lambda ledger, plan: types.SimpleNamespace(
                total=1.0, uncovered_subtopics=[]
            )
        ),
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
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "render_reading_path",
        _as_checkpoint(
            lambda selection: RenderPayload(
                name="reading_path", content_markdown="# Reading Path\n"
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        _as_checkpoint(
            lambda selection, ledger, plan: RenderPayload(
                name="backing_report", content_markdown="# Backing Report\n"
            )
        ),
    )
    monkeypatch.setattr(module, "log", lambda **kwargs: None)

    module.research_flow.run(
        "brief",
        config=module.ResearchConfig.for_tier(module.Tier.STANDARD).model_copy(
            update={"source_quality_floor": 0.45}
        ),
    )

    assert len(merge_calls) == 1
    _, _, received_config = merge_calls[0]
    assert received_config is not None
    assert received_config.source_quality_floor == 0.45
