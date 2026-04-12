import importlib
import inspect
import json
import sys
import types

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import (
    CoherenceResult,
    CritiqueResult,
    CoverageScore,
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
    GroundingResult,
    InvestigationPackage,
    IterationBudget,
    IterationTrace,
    RawToolResult,
    RenderProse,
    RequestClassification,
    RenderPayload,
    ResearchPlan,
    RelevanceCheckpointResult,
    RunSummary,
    SearchAction,
    SupervisorDecision,
    SelectionGraph,
    StopReason,
    SupervisorCheckpointResult,
)


def _clear_modules(*names: str) -> None:
    """Remove named modules so checkpoint tests can import fresh copies."""
    for name in names:
        sys.modules.pop(name, None)


def _clear_checkpoint_modules() -> None:
    """Clear all checkpoint modules that these tests re-import under stubs."""
    _clear_modules(
        "deep_research.checkpoints.classify",
        "deep_research.checkpoints.plan",
        "deep_research.checkpoints.supervisor",
        "deep_research.checkpoints.review",
        "deep_research.checkpoints.revise",
        "deep_research.checkpoints.grounding",
        "deep_research.checkpoints.coherence",
        "deep_research.checkpoints.normalize",
        "deep_research.checkpoints.search",
        "deep_research.checkpoints.relevance",
        "deep_research.checkpoints.merge",
        "deep_research.checkpoints.evaluate",
        "deep_research.checkpoints.select",
        "deep_research.checkpoints.assemble",
    )


def _install_supervisor_factory_dependency_stubs(monkeypatch):
    """Install prompt, adapter, and agent stubs for supervisor factory tests."""
    wrap_calls = []

    class FakeAgent:
        def __init__(self, model_name, **kwargs):
            """Record constructor inputs so supervisor factory tests can inspect them later."""
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
    """Import the supervisor agent factory after clearing its cached module."""
    _clear_modules("deep_research.agents._kitaru", "deep_research.agents.supervisor")
    return importlib.import_module("deep_research.agents.supervisor")


def _install_kitaru_checkpoint_stub(monkeypatch):
    """Install a minimal Kitaru checkpoint decorator and record decorated functions."""
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
    """Stub all agent builders used by checkpoint tests and capture their calls."""
    calls = []

    class FakeAgent:
        def __init__(self, output):
            """Store the canned output object returned by the fake agent `run_sync` call."""
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
        decision=SupervisorDecision(rationale="Need more sources."),
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
    review_output = CritiqueResult(
        dimensions=[],
        summary="Review summary",
        revision_suggestions=["Clarify the lead."],
        revision_recommended=True,
    )
    grounding_output = GroundingResult(score=1.0, verdicts=[])
    coherence_output = CoherenceResult(
        relevance=0.9,
        logical_flow=0.8,
        completeness=0.85,
        consistency=0.95,
        summary="Coherent overall.",
    )
    writer_output = RenderProse(
        content_markdown="# Revised Content\nRevised based on critique.",
        render_label="revised",
    )

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
    reviewer_calls = []
    grounding_calls = []
    coherence_calls = []
    writer_calls = []

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
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.reviewer",
        types.SimpleNamespace(
            build_reviewer_agent=build_factory(review_output, reviewer_calls)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.judge",
        types.SimpleNamespace(
            build_grounding_judge_agent=build_factory(
                grounding_output, grounding_calls
            ),
            build_coherence_judge_agent=build_factory(
                coherence_output, coherence_calls
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.writer",
        types.SimpleNamespace(
            build_writer_agent=build_factory(writer_output, writer_calls)
        ),
    )

    return {
        "run_calls": calls,
        "classifier_calls": classifier_calls,
        "planner_calls": planner_calls,
        "supervisor_calls": supervisor_calls,
        "relevance_calls": relevance_calls,
        "curator_calls": curator_calls,
        "reviewer_calls": reviewer_calls,
        "grounding_calls": grounding_calls,
        "coherence_calls": coherence_calls,
        "writer_calls": writer_calls,
    }


def _import_checkpoint_module(name: str):
    """Import a checkpoint module after clearing cached checkpoint-module imports first.

    These tests install lightweight stubs into `sys.modules`, so each import must start
    from a clean slate to ensure decorators and agent builders are rebound correctly.
    """
    _clear_checkpoint_modules()
    return importlib.import_module(name)


def _sample_plan() -> ResearchPlan:
    """Return a representative research plan fixture shared across checkpoint tests.

    The fixture includes multiple subtopics and key questions so selection, coverage,
    and prompt-shaping tests exercise more than the most trivial single-field case.
    """
    return ResearchPlan(
        goal="Answer the brief",
        key_questions=["What changed?", "Why does it matter?"],
        subtopics=["status", "impact"],
        queries=["topic overview", "topic impact"],
        sections=["Summary", "Details"],
        success_criteria=["Cover both subtopics"],
    )


def test_extract_candidates_accepts_raw_results(monkeypatch) -> None:
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
    normalized = module.extract_candidates([result])

    assert normalized[0].title == "A"
    assert normalized[0].provider == "brave"
    assert module.extract_candidates._checkpoint_type == "tool_call"


def test_extract_candidates_uses_payload_source_kind_and_baseline_quality(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.normalize")

    result = RawToolResult(
        tool_name="search",
        provider="arxiv",
        payload={
            "source_kind": "paper",
            "results": [
                {
                    "title": "DPO Survey",
                    "url": "https://arxiv.org/abs/2401.00001",
                    "description": "Survey abstract",
                    "arxiv_id": "2401.00001",
                }
            ],
        },
    )

    normalized = module.extract_candidates([result])

    assert normalized[0].source_kind == "paper"
    assert normalized[0].quality_score >= 0.9


def test_extract_candidates_accepts_raw_results_with_items_payload(monkeypatch) -> None:
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
    normalized = module.extract_candidates([result])

    assert normalized[0].title == "B"
    assert normalized[0].snippets[0].text == "y"


def test_evidence_ledger_accepts_bucket_style_constructor_inputs() -> None:
    candidate_payload = {
        "key": "candidate-1",
        "title": "Example",
        "url": "https://example.com/source-1",
        "provider": "arxiv",
        "source_kind": "paper",
        "selected": True,
    }

    ledger = EvidenceLedger.model_validate(
        {
            "considered": [candidate_payload],
            "selected": [candidate_payload],
            "rejected": [],
        }
    )

    assert ledger.considered[0].key == "candidate-1"
    assert ledger.selected[0].key == "candidate-1"


def test_supervisor_factory_uses_checkpoint_result_contract(monkeypatch) -> None:
    wrap_calls = _install_supervisor_factory_dependency_stubs(monkeypatch)

    module = _load_supervisor_factory_module()

    module.build_supervisor_agent("test-model", toolsets=[], tools=[])

    assert wrap_calls[0]["agent"].kwargs["output_type"] is SupervisorDecision


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
    assert len(calls["run_calls"]) == 1
    payload = json.loads(calls["run_calls"][0])
    assert payload["brief"] == "Research Kitaru"
    assert ("build_plan", "llm_call") in decorated


def test_run_supervisor_returns_structured_decision_and_mcp_raw_results(
    monkeypatch,
) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    supervisor_calls = []

    supervisor_output = SupervisorDecision(
        rationale="Need paper search next.",
        search_actions=[
            SearchAction(
                query="direct preference optimization survey",
                rationale="Need recent paper coverage.",
                preferred_providers=["arxiv", "semantic_scholar"],
            )
        ],
    )

    def build_supervisor_agent(model_name, *args, **kwargs):
        supervisor_calls.append(
            {"model_name": model_name, "args": args, "kwargs": kwargs}
        )

        def run_sync(prompt):
            return types.SimpleNamespace(
                output=supervisor_output,
                usage=lambda: types.SimpleNamespace(
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                ),
                all_messages=lambda: [
                    types.SimpleNamespace(
                        parts=[
                            types.SimpleNamespace(
                                part_kind="tool-return",
                                tool_name="search",
                                content={
                                    "provider": "mcp",
                                    "payload": {
                                        "source_kind": "web",
                                        "results": [],
                                    },
                                },
                            )
                        ]
                    )
                ],
            )

        return types.SimpleNamespace(run_sync=run_sync)

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.supervisor",
        types.SimpleNamespace(build_supervisor_agent=build_supervisor_agent),
    )

    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "supervisor_model": "supervisor-test-model",
            "max_tool_calls_per_cycle": 4,
            "tool_timeout_sec": 30,
        }
    )
    result = module.run_supervisor(
        _sample_plan(), EvidenceLedger(entries=[]), 2, config
    )

    assert result.decision.search_actions[0].preferred_providers == [
        "arxiv",
        "semantic_scholar",
    ]
    assert result.raw_results[0].provider == "mcp"
    assert result.budget.total_tokens == 18
    assert supervisor_calls == [
        {
            "model_name": "supervisor-test-model",
            "args": (),
            "kwargs": {
                "toolsets": [],
                "tools": supervisor_calls[0]["kwargs"]["tools"],
            },
        }
    ]
    assert ("run_supervisor", "llm_call") in decorated


def test_run_supervisor_surfaces_structured_trace_warning_raw_result(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "pydantic_ai",
        types.SimpleNamespace(__version__="9.9.9"),
    )

    def build_supervisor_agent(model_name, *args, **kwargs):
        def run_sync(prompt, hooks=None):
            if hooks and "after_tool_call" in hooks:
                hooks["after_tool_call"](
                    "search",
                    RawToolResult(
                        tool_name="search",
                        provider="mcp",
                        payload={"source_kind": "web", "results": []},
                    ),
                )
            return types.SimpleNamespace(
                output=SupervisorDecision(rationale="Need more coverage."),
                usage=lambda: types.SimpleNamespace(
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                ),
                all_messages=lambda: [
                    types.SimpleNamespace(
                        parts=[
                            types.SimpleNamespace(
                                part_kind="tool-return",
                                tool_name="search",
                                content={"provider": "mcp", "unexpected": True},
                            )
                        ]
                    )
                ],
            )

        return types.SimpleNamespace(run_sync=run_sync)

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.supervisor",
        types.SimpleNamespace(build_supervisor_agent=build_supervisor_agent),
    )

    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "supervisor_model": "supervisor-test-model",
            "max_tool_calls_per_cycle": 4,
            "tool_timeout_sec": 30,
        }
    )

    result = module.run_supervisor(
        _sample_plan(), EvidenceLedger(entries=[]), 2, config
    )

    assert result.raw_results[-1].tool_name == "supervisor_trace_warning"
    assert result.raw_results[-1].provider == "supervisor"
    assert result.raw_results[-1].ok is False
    assert result.raw_results[-1].payload["iteration"] == 2
    assert result.raw_results[-1].payload["dropped_part_count"] == 1
    assert result.raw_results[-1].payload["warning_codes"] == [
        "unhandled_tool_return_keys:provider,unexpected"
    ]
    assert result.raw_results[-1].payload["pydantic_ai_version"] == "9.9.9"


def test_run_supervisor_builds_real_provider_surface_and_richer_prompt(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    calls = []

    def build_supervisor_agent(model_name, *args, **kwargs):
        calls.append({"model_name": model_name, "args": args, "kwargs": kwargs})

        def run_sync(prompt):
            calls.append({"prompt": prompt})
            return types.SimpleNamespace(
                output=SupervisorDecision(rationale="Need more coverage."),
                usage=lambda: types.SimpleNamespace(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                ),
                all_messages=lambda: [],
            )

        return types.SimpleNamespace(run_sync=run_sync)

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.supervisor",
        types.SimpleNamespace(build_supervisor_agent=build_supervisor_agent),
    )

    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "supervisor_model": "override-supervisor-model",
            "max_tool_calls_per_cycle": 7,
            "tool_timeout_sec": 45,
        }
    )
    sentinel_toolset = object()
    sentinel_tools = [object(), object(), object()]

    monkeypatch.setattr(
        module,
        "build_supervisor_surface",
        lambda plan, ledger, *, uncovered_subtopics, tool_timeout_sec, allow_bash_tool=False, mcp_servers=None: (
            [sentinel_toolset],
            sentinel_tools,
        ),
    )

    result = module.run_supervisor(
        _sample_plan(),
        EvidenceLedger(entries=[]),
        1,
        config,
    )

    assert result.decision.rationale == "Need more coverage."
    assert calls[0:1] == [
        {
            "model_name": "override-supervisor-model",
            "args": (),
            "kwargs": {"toolsets": [sentinel_toolset], "tools": sentinel_tools},
        }
    ]
    import json

    prompt = json.loads(calls[1]["prompt"])
    assert prompt["uncovered_subtopics"] == ["status", "impact"]
    assert prompt["max_tool_calls"] == 7
    assert prompt["tool_timeout_sec"] == 45
    assert prompt["enabled_providers"] == config.enabled_providers


def test_extract_mcp_raw_results_accepts_multiple_search_like_shapes(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")

    result = types.SimpleNamespace(
        all_messages=lambda: [
            types.SimpleNamespace(
                parts=[
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content=RawToolResult(
                            tool_name="search",
                            provider="mcp",
                            payload={"source_kind": "web", "results": []},
                        ),
                    ),
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content={
                            "tool_name": "search",
                            "provider": "mcp",
                            "payload": {"source_kind": "paper", "items": []},
                        },
                    ),
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content={
                            "provider": "mcp",
                            "source_kind": "web",
                            "results": [],
                        },
                    ),
                ]
            )
        ]
    )

    raw_results = module.extract_mcp_raw_results(result)

    assert [item.provider for item in raw_results] == ["mcp", "mcp", "mcp"]
    assert raw_results[0].payload["source_kind"] == "web"
    assert raw_results[1].payload["source_kind"] == "paper"
    assert raw_results[2].payload["results"] == []


def test_extract_mcp_raw_results_ignores_non_search_like_shapes(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")

    result = types.SimpleNamespace(
        all_messages=lambda: [
            types.SimpleNamespace(
                parts=[
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content="not-a-dict",
                    ),
                    types.SimpleNamespace(
                        part_kind="tool-return",
                        tool_name="search",
                        content={"provider": "mcp"},
                    ),
                    types.SimpleNamespace(
                        part_kind="text",
                        tool_name="search",
                        content={
                            "provider": "mcp",
                            "source_kind": "web",
                            "results": [],
                        },
                    ),
                ]
            )
        ]
    )

    assert module.extract_mcp_raw_results(result) == []


def test_run_supervisor_uses_explicit_pricing_when_present(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)

    def build_supervisor_agent(model_name, *args, **kwargs):
        def run_sync(prompt):
            return types.SimpleNamespace(
                output=SupervisorDecision(rationale="Need more coverage."),
                usage=lambda: types.SimpleNamespace(
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                ),
                all_messages=lambda: [],
            )

        return types.SimpleNamespace(run_sync=run_sync)

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.supervisor",
        types.SimpleNamespace(build_supervisor_agent=build_supervisor_agent),
    )

    module = _import_checkpoint_module("deep_research.checkpoints.supervisor")
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "supervisor_pricing": {
                "input_per_million_usd": 100.0,
                "output_per_million_usd": 200.0,
            }
        }
    )

    result = module.run_supervisor(
        _sample_plan(),
        EvidenceLedger(entries=[]),
        1,
        config,
    )

    assert result.budget.input_tokens == 11
    assert result.budget.output_tokens == 7
    assert result.budget.estimated_cost_usd == 0.0025


def test_execute_searches_returns_raw_results_and_provider_budget(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.search")

    class FakeProvider:
        name = "arxiv"

        def estimate_cost_usd(self, query_count: int) -> float:
            return 0.0

        def search(self, queries, *, max_results_per_query=10, recency_days=None):
            return [
                RawToolResult(
                    tool_name="search",
                    provider="arxiv",
                    payload={"source_kind": "paper", "results": []},
                )
            ]

    monkeypatch.setattr(
        module,
        "ProviderRegistry",
        lambda config: types.SimpleNamespace(
            providers_for=lambda action, **kwargs: [FakeProvider()]
        ),
    )

    result = module.execute_searches(
        SupervisorDecision(
            rationale="Need paper search.",
            search_actions=[SearchAction(query="rlhf survey", rationale="Find papers")],
        ),
        ResearchConfig.for_tier(Tier.STANDARD),
    )

    assert result.raw_results[0].provider == "arxiv"
    assert result.budget.estimated_cost_usd == 0.0
    assert ("execute_searches", "tool_call") in decorated


def test_execute_searches_dedupes_duplicate_search_actions(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.search")
    search_calls = []

    class FakeProvider:
        name = "semantic_scholar"

        def estimate_cost_usd(self, query_count: int) -> float:
            return 0.02 * query_count

        def search(self, queries, *, max_results_per_query=10, recency_days=None):
            search_calls.append(
                {
                    "queries": list(queries),
                    "max_results_per_query": max_results_per_query,
                    "recency_days": recency_days,
                }
            )
            return [
                RawToolResult(
                    tool_name="search",
                    provider="semantic_scholar",
                    payload={"source_kind": "paper", "results": []},
                )
            ]

    monkeypatch.setattr(
        module,
        "ProviderRegistry",
        lambda config: types.SimpleNamespace(
            providers_for=lambda action, **kwargs: [FakeProvider()]
        ),
    )

    result = module.execute_searches(
        SupervisorDecision(
            rationale="Need paper search.",
            search_actions=[
                SearchAction(
                    query="rlhf survey",
                    rationale="Find papers",
                    preferred_providers=["semantic_scholar"],
                    recency_days=30,
                    max_results=3,
                ),
                SearchAction(
                    query="RLHF survey",
                    rationale="Duplicate phrasing should dedupe",
                    preferred_providers=["semantic_scholar"],
                    recency_days=30,
                    max_results=3,
                ),
            ],
        ),
        ResearchConfig.for_tier(Tier.STANDARD),
    )

    assert len(search_calls) == 1
    assert search_calls[0] == {
        "queries": ["rlhf survey"],
        "max_results_per_query": 3,
        "recency_days": 30,
    }
    assert len(result.raw_results) == 1
    assert result.budget.estimated_cost_usd == 0.02


def test_review_and_judge_checkpoints_use_configured_models(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    review_module = _import_checkpoint_module("deep_research.checkpoints.review")
    revise_module = _import_checkpoint_module("deep_research.checkpoints.revise")
    grounding_module = _import_checkpoint_module("deep_research.checkpoints.grounding")
    coherence_module = _import_checkpoint_module("deep_research.checkpoints.coherence")
    config = ResearchConfig.for_tier(Tier.DEEP).model_copy(
        update={
            "review_model": "review-model",
            "judge_model": "judge-model",
            "review_pricing": {
                "input_per_million_usd": 100.0,
                "output_per_million_usd": 200.0,
            },
            "judge_pricing": {
                "input_per_million_usd": 100.0,
                "output_per_million_usd": 200.0,
            },
        }
    )
    renders = [RenderPayload(name="reading_path", content_markdown="# RP")]

    critique_result = review_module.critique_reports(
        renders,
        _sample_plan(),
        SelectionGraph(items=[]),
        EvidenceLedger(entries=[]),
        config,
    )
    revision_result = revise_module.apply_revisions(
        renders,
        critique_result.critique,
        _sample_plan(),
        config,
    )
    revised = revision_result.renders
    grounding_result = grounding_module.verify_grounding(
        revised,
        EvidenceLedger(entries=[]),
        config,
    )
    coherence_result = coherence_module.verify_coherence(
        revised,
        _sample_plan(),
        config,
    )

    assert critique_result.critique.summary == "Review summary"
    assert critique_result.budget.estimated_cost_usd == 0.0
    assert revision_result.budget.estimated_cost_usd == 0.0
    assert revised[0].name == "reading_path"
    assert (
        revised[0].content_markdown == "# Revised Content\nRevised based on critique."
    )
    assert revised[0].structured_content["critique_summary"] == "Review summary"
    assert revised[0].structured_content["revision_applied"] is True
    assert grounding_result.grounding.score == 1.0
    assert grounding_result.budget.estimated_cost_usd == 0.0
    assert coherence_result.coherence.summary == "Coherent overall."
    assert coherence_result.budget.estimated_cost_usd == 0.0
    assert calls["reviewer_calls"] == [
        {"model_name": "review-model", "args": (), "kwargs": {}}
    ]
    assert calls["grounding_calls"] == [
        {"model_name": "judge-model", "args": (), "kwargs": {}}
    ]
    assert calls["coherence_calls"] == [
        {"model_name": "judge-model", "args": (), "kwargs": {}}
    ]
    assert ("critique_reports", "llm_call") in decorated
    assert ("apply_revisions", "llm_call") in decorated
    assert ("verify_grounding", "llm_call") in decorated
    assert ("verify_coherence", "llm_call") in decorated


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


def test_update_ledger_preserves_existing_and_adds_unique_entries(monkeypatch) -> None:
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

    merged = module.update_ledger(scored, ledger)

    assert [entry.key for entry in merged.entries] == ["existing", "new"]
    assert module.update_ledger._checkpoint_type == "tool_call"


def test_update_ledger_uses_configured_quality_floor(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.merge")
    ledger = EvidenceLedger()
    scored = [
        EvidenceCandidate(
            key="candidate-1",
            title="Borderline",
            url="https://example.com/borderline",
            provider="search",
            source_kind="web",
            quality_score=0.45,
            relevance_score=0.6,
        )
    ]
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={"source_quality_floor": 0.5}
    )

    merged = module.update_ledger(scored, ledger, config)

    assert merged.selected == []
    assert [candidate.key for candidate in merged.rejected] == ["candidate-1"]


def test_update_ledger_preserves_existing_dedupe_log(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.merge")

    ledger = EvidenceLedger(
        considered=[
            EvidenceCandidate(
                key="candidate-1",
                title="Replay",
                url="https://example.com/replay",
                provider="docs",
                source_kind="docs",
                doi="10.1000/replay",
                quality_score=0.8,
                selected=True,
            )
        ],
        selected=[
            EvidenceCandidate(
                key="candidate-1",
                title="Replay",
                url="https://example.com/replay",
                provider="docs",
                source_kind="docs",
                doi="10.1000/replay",
                quality_score=0.8,
                selected=True,
            )
        ],
        rejected=[],
        dedupe_log=[
            {
                "duplicate_key": "candidate-0",
                "canonical_key": "candidate-1",
                "match_basis": "doi",
            }
        ],
    )
    scored = [
        EvidenceCandidate(
            key="candidate-2",
            title="Unique",
            url="https://example.com/unique",
            provider="search",
            source_kind="web",
            quality_score=0.7,
            relevance_score=0.6,
        )
    ]

    merged = module.update_ledger(scored, ledger)

    assert [event.duplicate_key for event in merged.dedupe_log] == ["candidate-0"]


def test_update_ledger_does_not_duplicate_historical_dedupe_events_on_replay(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.merge")

    canonical = EvidenceCandidate(
        key="candidate-1",
        title="Replay",
        url="https://example.com/replay",
        provider="docs",
        source_kind="docs",
        doi="10.1000/replay",
        quality_score=0.8,
        selected=True,
    )
    duplicate = EvidenceCandidate(
        key="candidate-0",
        title="Replay duplicate",
        url="https://example.com/replay",
        provider="search",
        source_kind="web",
        quality_score=0.4,
        selected=False,
    )
    ledger = EvidenceLedger(
        entries=[canonical, duplicate],
        dedupe_log=[
            {
                "duplicate_key": "candidate-0",
                "canonical_key": "candidate-1",
                "match_basis": "canonical_url",
            }
        ],
    )

    merged = module.update_ledger([], ledger)

    assert [
        (event.duplicate_key, event.canonical_key, event.match_basis)
        for event in merged.dedupe_log
    ] == [("candidate-0", "candidate-1", "canonical_url")]


def _install_coverage_scorer_stub(monkeypatch, canned_output):
    """Stub the coverage scorer agent factory so the evaluate checkpoint uses a fake LLM."""
    calls = []

    def build_coverage_scorer_agent(model_name):
        calls.append(model_name)

        def run_sync(prompt):
            return types.SimpleNamespace(output=canned_output)

        return types.SimpleNamespace(run_sync=run_sync)

    monkeypatch.setitem(
        sys.modules,
        "deep_research.agents.coverage_scorer",
        types.SimpleNamespace(build_coverage_scorer_agent=build_coverage_scorer_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "deep_research.observability",
        types.SimpleNamespace(
            bootstrap_logfire=lambda: None,
            span=lambda *args, **kwargs: __import__("contextlib").nullcontext(),
        ),
    )
    return calls


def _sample_config() -> ResearchConfig:
    return ResearchConfig.for_tier(Tier.STANDARD)


def test_score_coverage_delegates_to_llm_agent(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    canned = CoverageScore(
        subtopic_coverage=0.8,
        source_diversity=0.7,
        evidence_density=0.6,
        total=0.7,
        uncovered_subtopics=["impact"],
    )
    scorer_calls = _install_coverage_scorer_stub(monkeypatch, canned)
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
        ]
    )

    config = _sample_config()
    coverage = module.score_coverage(ledger, _sample_plan(), config)

    assert isinstance(coverage, CoverageScore)
    assert coverage.total == 0.7
    assert coverage.uncovered_subtopics == ["impact"]
    assert scorer_calls == [config.coverage_scorer_model]
    assert module.score_coverage._checkpoint_type == "llm_call"


def test_score_coverage_returns_zeros_for_empty_ledger(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    # Agent should NOT be called for empty entries, so provide a dummy
    _install_coverage_scorer_stub(monkeypatch, None)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    coverage = module.score_coverage(
        EvidenceLedger(entries=[]), _sample_plan(), _sample_config()
    )

    assert coverage.subtopic_coverage == 0.0
    assert coverage.source_diversity == 0.0
    assert coverage.evidence_density == 0.0
    assert coverage.total == 0.0
    assert coverage.uncovered_subtopics == ["status", "impact"]


def test_score_coverage_empty_ledger_with_rejected_only(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    _install_coverage_scorer_stub(monkeypatch, None)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    rejected_candidate = EvidenceCandidate(
        key="rejected-1",
        title="Status",
        url="https://status.example",
        snippets=[EvidenceSnippet(text="Current status and context")],
        provider="brave",
        source_kind="web",
        quality_score=0.1,
        selected=False,
    )
    ledger = EvidenceLedger(
        considered=[rejected_candidate],
        selected=[],
        rejected=[rejected_candidate],
    )

    coverage = module.score_coverage(ledger, _sample_plan(), _sample_config())

    assert coverage.subtopic_coverage == 0.0
    assert coverage.total == 0.0


def test_score_coverage_passes_config_model_to_agent_factory(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    canned = CoverageScore(
        subtopic_coverage=1.0,
        source_diversity=1.0,
        evidence_density=1.0,
        total=1.0,
    )
    scorer_calls = _install_coverage_scorer_stub(monkeypatch, canned)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="1",
                title="Status",
                url="https://status.example",
                snippets=[EvidenceSnippet(text="info")],
                provider="brave",
                source_kind="web",
            ),
        ]
    )

    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={"coverage_scorer_model": "test-model-override"}
    )
    module.score_coverage(ledger, _sample_plan(), config)

    assert scorer_calls == ["test-model-override"]


def test_rank_evidence_returns_selected_entries(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(entries=[])

    result = module.rank_evidence(ledger, _sample_plan())

    assert result.items == []
    assert result.gap_coverage_summary == ["status", "impact"]
    assert ("rank_evidence", "tool_call") in decorated


def test_rank_evidence_uses_selected_entries_and_gap_summary(
    monkeypatch,
) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(
        considered=[
            EvidenceCandidate(
                key="candidate-1",
                title="Replay",
                url="https://example.com/replay",
                provider="docs",
                source_kind="docs",
                matched_subtopics=["replay"],
                quality_score=0.8,
                relevance_score=0.7,
                authority_score=0.8,
                selected=True,
            ),
            EvidenceCandidate(
                key="candidate-2",
                title="Durability",
                url="https://example.com/durability",
                provider="paper",
                source_kind="paper",
                matched_subtopics=["durability"],
                quality_score=0.9,
                relevance_score=0.6,
                authority_score=0.95,
                snippets=[
                    EvidenceSnippet(text="Longer source"),
                    EvidenceSnippet(text="Second snippet"),
                ],
                selected=True,
            ),
        ],
        selected=[
            EvidenceCandidate(
                key="candidate-1",
                title="Replay",
                url="https://example.com/replay",
                provider="docs",
                source_kind="docs",
                matched_subtopics=["replay"],
                quality_score=0.8,
                relevance_score=0.7,
                authority_score=0.8,
                selected=True,
            ),
            EvidenceCandidate(
                key="candidate-2",
                title="Durability",
                url="https://example.com/durability",
                provider="paper",
                source_kind="paper",
                matched_subtopics=["durability"],
                quality_score=0.9,
                relevance_score=0.6,
                authority_score=0.95,
                snippets=[
                    EvidenceSnippet(text="Longer source"),
                    EvidenceSnippet(text="Second snippet"),
                ],
                selected=True,
            ),
        ],
        rejected=[],
        dedupe_log=[],
    )
    plan = ResearchPlan(
        goal="Answer the brief",
        key_questions=["What matters?"],
        subtopics=["replay", "durability", "operations"],
        queries=["example query"],
        sections=["Overview"],
        success_criteria=["Produce a summary"],
    )

    result = module.rank_evidence(ledger, plan)

    assert [item.candidate_key for item in result.items] == [
        "candidate-2",
        "candidate-1",
    ]
    assert result.items[0].matched_subtopics == ["durability"]
    assert result.items[0].reading_time_minutes == 6
    assert (
        result.items[0].ordering_rationale
        == "Higher quality, authority, and relevance sources appear earlier."
    )
    assert result.gap_coverage_summary == ["operations"]
    assert ("rank_evidence", "tool_call") in decorated


def test_rank_evidence_uses_selected_flags_for_legacy_entries_only_ledger(
    monkeypatch,
) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="candidate-1",
                title="Replay",
                url="https://example.com/replay",
                provider="docs",
                source_kind="docs",
                matched_subtopics=["status"],
                quality_score=0.8,
                relevance_score=0.7,
                authority_score=0.8,
                selected=True,
            ),
            EvidenceCandidate(
                key="candidate-2",
                title="Durability",
                url="https://example.com/durability",
                provider="paper",
                source_kind="paper",
                matched_subtopics=["impact"],
                quality_score=0.9,
                relevance_score=0.6,
                authority_score=0.95,
                selected=False,
            ),
        ]
    )

    result = module.rank_evidence(ledger, _sample_plan())

    assert [item.candidate_key for item in result.items] == ["candidate-1"]
    assert result.gap_coverage_summary == ["impact"]
    assert ("rank_evidence", "tool_call") in decorated


def test_rank_evidence_ignores_all_rejected_legacy_entries_only_ledger(
    monkeypatch,
) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="candidate-1",
                title="Status",
                url="https://example.com/status",
                provider="docs",
                source_kind="docs",
                matched_subtopics=["status"],
                selected=False,
            )
        ]
    )

    result = module.rank_evidence(ledger, _sample_plan())

    assert result.items == []
    assert result.gap_coverage_summary == ["status", "impact"]
    assert ("rank_evidence", "tool_call") in decorated


def test_rank_evidence_uses_text_coverage_for_legacy_selected_entry_without_matched_subtopics(
    monkeypatch,
) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="candidate-1",
                title="Status update",
                url="https://example.com/status",
                provider="docs",
                source_kind="docs",
                snippets=[EvidenceSnippet(text="Current status and context")],
                selected=True,
            )
        ]
    )

    result = module.rank_evidence(ledger, _sample_plan())

    assert [item.candidate_key for item in result.items] == ["candidate-1"]
    assert result.gap_coverage_summary == ["impact"]
    assert ("rank_evidence", "tool_call") in decorated


def test_rank_evidence_treats_matched_subtopics_case_insensitively(
    monkeypatch,
) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(
        selected=[
            EvidenceCandidate(
                key="candidate-1",
                title="Replay",
                url="https://example.com/replay",
                provider="docs",
                source_kind="docs",
                matched_subtopics=["replay"],
                selected=True,
            )
        ]
    )
    plan = ResearchPlan(
        goal="Answer the brief",
        key_questions=["What matters?"],
        subtopics=["Replay", "Impact"],
        queries=["example query"],
        sections=["Overview"],
        success_criteria=["Produce a summary"],
    )

    result = module.rank_evidence(ledger, plan)

    assert [item.candidate_key for item in result.items] == ["candidate-1"]
    assert result.gap_coverage_summary == ["Impact"]
    assert ("rank_evidence", "tool_call") in decorated


def test_rank_evidence_breaks_score_ties_by_candidate_key(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.select")
    shared_scores = {
        "quality_score": 0.8,
        "authority_score": 0.7,
        "relevance_score": 0.6,
        "selected": True,
    }
    ledger = EvidenceLedger(
        selected=[
            EvidenceCandidate(
                key="candidate-b",
                title="Second by key",
                url="https://example.com/b",
                provider="docs",
                source_kind="docs",
                **shared_scores,
            ),
            EvidenceCandidate(
                key="candidate-a",
                title="First by key",
                url="https://example.com/a",
                provider="docs",
                source_kind="docs",
                **shared_scores,
            ),
        ]
    )

    result = module.rank_evidence(ledger, _sample_plan())

    assert [item.candidate_key for item in result.items] == [
        "candidate-a",
        "candidate-b",
    ]
    assert result.items[0].ordering_rationale == (
        "Higher quality, authority, and relevance sources appear earlier."
    )


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
