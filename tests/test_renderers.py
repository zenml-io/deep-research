import importlib
import sys
import types

from deep_research.models import (
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
    SelectionItem,
)


def _install_kitaru_stub() -> None:
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


def _load_module(module_name: str):
    _install_kitaru_stub()
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_render_reading_path_returns_markdown_payload() -> None:
    module = _load_module("deep_research.renderers.reading_path")
    selection = SelectionGraph(
        items=[SelectionItem(candidate_key="a", rationale="useful")]
    )

    payload = module.render_reading_path(selection)

    assert payload.name == "reading_path"
    assert "useful" in payload.content_markdown
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
    assert module.render_backing_report._checkpoint_type == "llm_call"


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
        lambda brief, config: types.SimpleNamespace(
            recommended_tier=module.Tier.STANDARD,
            needs_clarification=False,
            clarification_question=None,
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        lambda brief, classification, tier: ResearchPlan(
            goal="Answer the brief",
            key_questions=["What matters?"],
            subtopics=["core"],
            queries=["example query"],
            sections=["Overview"],
            success_criteria=["Produce a summary"],
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
    monkeypatch.setattr(module, "normalize_evidence", lambda raw_results: [])
    monkeypatch.setattr(
        module,
        "score_relevance",
        lambda candidates, plan, config: types.SimpleNamespace(
            candidates=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        ),
    )
    monkeypatch.setattr(module, "merge_evidence", lambda scored, ledger: ledger)
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        lambda ledger, plan: types.SimpleNamespace(total=1.0),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=True,
            reason=module.StopReason.CONVERGED,
        ),
    )
    monkeypatch.setattr(module, "build_selection_graph", lambda ledger, plan: selection)
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "render_reading_path",
        lambda rendered_selection: (
            reading_path_calls.append(rendered_selection)
            or RenderPayload(name="reading_path", content_markdown="# Reading Path\n")
        ),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        lambda rendered_selection, rendered_ledger, rendered_plan: (
            backing_report_calls.append(
                (rendered_selection, rendered_ledger, rendered_plan)
            )
            or RenderPayload(
                name="backing_report", content_markdown="# Backing Report\n"
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
