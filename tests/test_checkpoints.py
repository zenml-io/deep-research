import importlib
import inspect
import sys
import types

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import (
    CoverageScore,
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
    InvestigationPackage,
    IterationBudget,
    IterationTrace,
    RawToolResult,
    RequestClassification,
    RenderPayload,
    ResearchPlan,
    RelevanceCheckpointResult,
    RunSummary,
    SelectionGraph,
    StopReason,
    SupervisorCheckpointResult,
)


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _clear_checkpoint_modules() -> None:
    _clear_modules(
        "deep_research.checkpoints.classify",
        "deep_research.checkpoints.plan",
        "deep_research.checkpoints.supervisor",
        "deep_research.checkpoints.normalize",
        "deep_research.checkpoints.relevance",
        "deep_research.checkpoints.merge",
        "deep_research.checkpoints.evaluate",
        "deep_research.checkpoints.select",
        "deep_research.checkpoints.assemble",
    )


def _install_supervisor_factory_dependency_stubs(monkeypatch):
    wrap_calls = []

    class FakeAgent:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    def wrap(agent, tool_capture_config=None):
        wrapped = {
            "agent": agent,
            "tool_capture_config": tool_capture_config,
        }
        wrap_calls.append(wrapped)
        return wrapped

    monkeypatch.setitem(
        sys.modules, "pydantic_ai", types.SimpleNamespace(Agent=FakeAgent)
    )
    monkeypatch.setitem(
        sys.modules,
        "kitaru.adapters",
        types.SimpleNamespace(pydantic_ai=types.SimpleNamespace(wrap=wrap)),
    )
    monkeypatch.setitem(
        sys.modules,
        "kitaru",
        types.SimpleNamespace(
            adapters=types.SimpleNamespace(pydantic_ai=types.SimpleNamespace(wrap=wrap))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.prompts.loader",
        types.SimpleNamespace(load_prompt=lambda name: f"prompt:{name}"),
    )
    return wrap_calls


def _load_supervisor_factory_module():
    _clear_modules("deep_research.agents.supervisor")
    return importlib.import_module("deep_research.agents.supervisor")


def _install_kitaru_checkpoint_stub(monkeypatch):
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


def _install_agent_builder_stubs(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, output):
            self._output = output

        def run_sync(self, prompt):
            calls.append(prompt)
            return types.SimpleNamespace(output=self._output)

    classifier_output = RequestClassification(
        audience_mode="technical",
        freshness_mode="current",
        recommended_tier=Tier.STANDARD,
    )
    planner_output = ResearchPlan(
        goal="Understand the topic",
        key_questions=["What matters?"],
        subtopics=["core"],
        queries=["example query"],
        sections=["Overview"],
        success_criteria=["Produce a summary"],
    )
    supervisor_output = SupervisorCheckpointResult(
        raw_results=[
            RawToolResult(
                tool_name="search",
                provider="brave",
                payload={"results": [{"title": "A", "url": "https://a.example"}]},
            )
        ],
        budget=IterationBudget(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    relevance_output = RelevanceCheckpointResult(
        candidates=[
            EvidenceCandidate(
                key="c-1",
                title="Relevant",
                url="https://relevant.example",
                snippets=[EvidenceSnippet(text="alpha")],
                provider="brave",
                source_kind="web",
                quality_score=0.6,
                relevance_score=0.8,
            )
        ],
        budget=IterationBudget(input_tokens=8, output_tokens=4, total_tokens=12),
    )
    selection_output = SelectionGraph(items=[])

    def build_factory(output, bucket):
        def builder(model_name, *args, **kwargs):
            bucket.append({"model_name": model_name, "args": args, "kwargs": kwargs})
            return FakeAgent(output)

        return builder

    classifier_calls = []
    planner_calls = []
    supervisor_calls = []
    relevance_calls = []
    curator_calls = []

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.classifier",
        types.SimpleNamespace(
            build_classifier_agent=build_factory(classifier_output, classifier_calls)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.planner",
        types.SimpleNamespace(
            build_planner_agent=build_factory(planner_output, planner_calls)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.supervisor",
        types.SimpleNamespace(
            build_supervisor_agent=build_factory(supervisor_output, supervisor_calls)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.relevance_scorer",
        types.SimpleNamespace(
            build_relevance_scorer_agent=build_factory(
                relevance_output, relevance_calls
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.curator",
        types.SimpleNamespace(
            build_curator_agent=build_factory(selection_output, curator_calls)
        ),
    )

    return {
        "run_calls": calls,
        "classifier_calls": classifier_calls,
        "planner_calls": planner_calls,
        "supervisor_calls": supervisor_calls,
        "relevance_calls": relevance_calls,
        "curator_calls": curator_calls,
    }


def _import_checkpoint_module(name: str):
    _clear_checkpoint_modules()
    return importlib.import_module(name)


def _sample_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Answer the brief",
        key_questions=["What changed?", "Why does it matter?"],
        subtopics=["status", "impact"],
        queries=["topic overview", "topic impact"],
        sections=["Summary", "Details"],
        success_criteria=["Cover both subtopics"],
    )


def test_normalize_evidence_accepts_raw_results(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.normalize")

    result = RawToolResult(
        tool_name="search",
        provider="brave",
        payload={
            "results": [
                {
                    "title": "A",
                    "url": "https://a.example",
                    "snippet": "x",
                }
            ]
        },
    )
    normalized = module.normalize_evidence([result])

    assert normalized[0].title == "A"
    assert normalized[0].provider == "brave"
    assert module.normalize_evidence._checkpoint_type == "tool_call"


def test_normalize_evidence_accepts_raw_results_with_items_payload(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.normalize")

    result = RawToolResult(
        tool_name="search",
        provider="brave",
        payload={
            "items": [
                {
                    "title": "B",
                    "url": "https://b.example",
                    "description": "y",
                }
            ]
        },
    )
    normalized = module.normalize_evidence([result])

    assert normalized[0].title == "B"
    assert normalized[0].snippets[0].text == "y"


def test_supervisor_factory_uses_checkpoint_result_contract(monkeypatch) -> None:
    wrap_calls = _install_supervisor_factory_dependency_stubs(monkeypatch)

    module = _load_supervisor_factory_module()

    module.build_supervisor_agent("test-model", toolsets=[], tools=[])

    assert wrap_calls[0]["agent"].kwargs["output_type"] is SupervisorCheckpointResult


def test_classify_request_uses_configured_model_and_returns_output(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.classify")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={"classifier_model": "classifier-test-model"}
    )

    result = module.classify_request("Need a concise answer", config)

    assert result.recommended_tier is Tier.STANDARD
    assert calls["classifier_calls"] == [
        {"model_name": "classifier-test-model", "args": (), "kwargs": {}}
    ]
    assert calls["run_calls"] == ["Need a concise answer"]
    assert ("classify_request", "llm_call") in decorated


def test_build_plan_uses_tier_model_and_passes_brief(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.plan")
    classification = RequestClassification(
        audience_mode="technical",
        freshness_mode="current",
        recommended_tier=Tier.DEEP,
    )

    result = module.build_plan("Research Kitaru", classification, Tier.QUICK)

    assert result.goal == "Understand the topic"
    assert calls["planner_calls"] == [
        {
            "model_name": ResearchConfig.for_tier(Tier.QUICK).planner_model,
            "args": (),
            "kwargs": {},
        }
    ]
    assert calls["run_calls"] == ["Research Kitaru"]
    assert ("build_plan", "llm_call") in decorated


def test_run_supervisor_delegates_to_shared_helper(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    _install_agent_builder_stubs(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={"supervisor_model": "supervisor-test-model"}
    )
    plan = _sample_plan()
    ledger = EvidenceLedger(entries=[])
    captured = []

    def fake_execute(plan_arg, ledger_arg, iteration_arg, config_arg, model_name=None):
        captured.append(
            {
                "plan": plan_arg,
                "ledger": ledger_arg,
                "iteration": iteration_arg,
                "config": config_arg,
                "model_name": model_name,
            }
        )
        return SupervisorCheckpointResult(raw_results=[])

    monkeypatch.setattr(module, "_execute_supervisor_turn", fake_execute)

    result = module.run_supervisor(plan, ledger, 2, config)

    assert result == SupervisorCheckpointResult(raw_results=[])
    assert captured == [
        {
            "plan": plan,
            "ledger": ledger,
            "iteration": 2,
            "config": config,
            "model_name": None,
        }
    ]
    assert ("run_supervisor", "llm_call") in decorated


def test_execute_supervisor_turn_uses_override_model_when_provided(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")
    config = ResearchConfig.for_tier(Tier.STANDARD)

    result = module._execute_supervisor_turn(
        _sample_plan(),
        EvidenceLedger(entries=[]),
        1,
        config,
        model_name="override-supervisor-model",
    )

    assert result.raw_results[0].tool_name == "search"
    assert calls["supervisor_calls"] == [
        {
            "model_name": "override-supervisor-model",
            "args": (),
            "kwargs": {"toolsets": [], "tools": []},
        }
    ]
    assert calls["run_calls"]


def test_score_relevance_uses_configured_model(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.relevance")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={"relevance_scorer_model": "relevance-test-model"}
    )
    candidates = [
        EvidenceCandidate(
            key="c-1",
            title="Candidate",
            url="https://candidate.example",
            provider="brave",
            source_kind="web",
        )
    ]

    result = module.score_relevance(candidates, _sample_plan(), config)

    assert result.candidates[0].title == "Relevant"
    assert calls["relevance_calls"] == [
        {"model_name": "relevance-test-model", "args": (), "kwargs": {}}
    ]
    assert ("score_relevance", "llm_call") in decorated


def test_merge_evidence_preserves_existing_and_adds_unique_entries(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.merge")

    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="existing",
                title="Existing",
                url="https://x.example",
                provider="docs",
                source_kind="docs",
            )
        ]
    )
    scored = [
        EvidenceCandidate(
            key="duplicate",
            title="Duplicate",
            url="https://x.example",
            provider="search",
            source_kind="web",
        ),
        EvidenceCandidate(
            key="new",
            title="New",
            url="https://y.example",
            provider="search",
            source_kind="web",
        ),
    ]

    merged = module.merge_evidence(scored, ledger)

    assert [entry.key for entry in merged.entries] == ["existing", "new"]
    assert module.merge_evidence._checkpoint_type == "tool_call"


def test_evaluate_coverage_computes_serializable_score(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="1",
                title="Status",
                url="https://status.example",
                snippets=[EvidenceSnippet(text="Current status and context")],
                provider="brave",
                source_kind="web",
            ),
            EvidenceCandidate(
                key="2",
                title="Impact",
                url="https://impact.example",
                snippets=[EvidenceSnippet(text="Impact details and consequences")],
                provider="docs",
                source_kind="docs",
            ),
        ]
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert isinstance(coverage, CoverageScore)
    assert 0.0 <= coverage.total <= 1.0
    assert coverage.subtopic_coverage == 1.0
    assert module.evaluate_coverage._checkpoint_type == "tool_call"


def test_build_selection_graph_returns_selected_entries(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(entries=[])

    result = module.build_selection_graph(ledger, _sample_plan())

    assert result.items == []
    assert calls["curator_calls"] == [
        {
            "model_name": ResearchConfig.for_tier(Tier.STANDARD).curator_model,
            "args": (),
            "kwargs": {},
        }
    ]
    assert ("build_selection_graph", "llm_call") in decorated


def test_assemble_package_checkpoint_wraps_terminal_package(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.assemble")

    package = module.assemble_package(
        run_summary=RunSummary(
            run_id="run-1",
            brief="brief",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
        ),
        research_plan=ResearchPlan(
            goal="goal",
            key_questions=["q"],
            subtopics=["topic"],
            queries=["query"],
            sections=["Overview"],
            success_criteria=["done"],
        ),
        evidence_ledger=EvidenceLedger(entries=[]),
        selection_graph=SelectionGraph(items=[]),
        iteration_trace=IterationTrace(iterations=[]),
        renders=[RenderPayload(name="reading_path", content_markdown="# Reading Path")],
    )

    assert isinstance(package, InvestigationPackage)
    assert package.run_summary.run_id == "run-1"
    assert ("assemble_package", "tool_call") in decorated


def test_assemble_package_checkpoint_avoids_keyword_only_args(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.assemble")

    parameter_kinds = [
        parameter.kind
        for parameter in inspect.signature(module.assemble_package).parameters.values()
    ]

    assert inspect.Parameter.KEYWORD_ONLY not in parameter_kinds
