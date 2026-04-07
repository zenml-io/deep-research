import importlib
import inspect
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
    _clear_modules("deep_research.agents.supervisor")
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
    }


def _import_checkpoint_module(name: str):
    """Import a checkpoint module after clearing cached checkpoint modules."""
    _clear_checkpoint_modules()
    return importlib.import_module(name)


def _sample_plan() -> ResearchPlan:
    """Return a representative research plan fixture for checkpoint tests."""
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


def test_run_supervisor_uses_configured_model(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

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

    assert result.raw_results[0].tool_name == "search"
    assert calls["supervisor_calls"] == [
        {
            "model_name": "supervisor-test-model",
            "args": (),
            "kwargs": {
                "toolsets": [],
                "tools": calls["supervisor_calls"][0]["kwargs"]["tools"],
            },
        }
    ]
    assert ("run_supervisor", "llm_call") in decorated


def test_run_supervisor_builds_real_provider_surface_and_richer_prompt(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

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
        lambda plan, ledger, *, uncovered_subtopics, tool_timeout_sec, mcp_servers=None: (
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

    assert result.raw_results[0].tool_name == "search"
    assert calls["supervisor_calls"] == [
        {
            "model_name": "override-supervisor-model",
            "args": (),
            "kwargs": {"toolsets": [sentinel_toolset], "tools": sentinel_tools},
        }
    ]
    prompt = calls["run_calls"][0]
    assert prompt["uncovered_subtopics"] == ["status", "impact"]
    assert prompt["max_tool_calls"] == 7
    assert prompt["tool_timeout_sec"] == 45


def test_review_and_judge_checkpoints_use_configured_models(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)
    calls = _install_agent_builder_stubs(monkeypatch)

    review_module = _import_checkpoint_module("deep_research.checkpoints.review")
    revise_module = _import_checkpoint_module("deep_research.checkpoints.revise")
    grounding_module = _import_checkpoint_module("deep_research.checkpoints.grounding")
    coherence_module = _import_checkpoint_module("deep_research.checkpoints.coherence")
    config = ResearchConfig.for_tier(Tier.DEEP).model_copy(
        update={"review_model": "review-model", "judge_model": "judge-model"}
    )
    renders = [RenderPayload(name="reading_path", content_markdown="# RP")]

    critique = review_module.review_renders(
        renders,
        _sample_plan(),
        SelectionGraph(items=[]),
        EvidenceLedger(entries=[]),
        config,
    )
    revised = revise_module.revise_renders(renders, critique, _sample_plan())
    grounding = grounding_module.judge_grounding(
        revised,
        EvidenceLedger(entries=[]),
        config,
    )
    coherence = coherence_module.judge_coherence(
        revised,
        _sample_plan(),
        config,
    )

    assert critique.summary == "Review summary"
    assert revised[0].name == "reading_path"
    assert revised[0].structured_content["critique_summary"] == "Review summary"
    assert grounding.score == 1.0
    assert coherence.summary == "Coherent overall."
    assert calls["reviewer_calls"] == [
        {"model_name": "review-model", "args": (), "kwargs": {}}
    ]
    assert calls["grounding_calls"] == [
        {"model_name": "judge-model", "args": (), "kwargs": {}}
    ]
    assert calls["coherence_calls"] == [
        {"model_name": "judge-model", "args": (), "kwargs": {}}
    ]
    assert ("review_renders", "llm_call") in decorated
    assert ("revise_renders", "llm_call") in decorated
    assert ("judge_grounding", "llm_call") in decorated
    assert ("judge_coherence", "llm_call") in decorated


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


def test_merge_evidence_uses_configured_quality_floor(monkeypatch) -> None:
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

    merged = module.merge_evidence(scored, ledger, config)

    assert merged.selected == []
    assert [candidate.key for candidate in merged.rejected] == ["candidate-1"]


def test_merge_evidence_preserves_existing_dedupe_log(monkeypatch) -> None:
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

    merged = module.merge_evidence(scored, ledger)

    assert [event.duplicate_key for event in merged.dedupe_log] == ["candidate-0"]


def test_merge_evidence_does_not_duplicate_historical_dedupe_events_on_replay(
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

    merged = module.merge_evidence([], ledger)

    assert [
        (event.duplicate_key, event.canonical_key, event.match_basis)
        for event in merged.dedupe_log
    ] == [("candidate-0", "candidate-1", "canonical_url")]


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


def test_evaluate_coverage_ignores_rejected_only_evidence(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    selected_candidate = EvidenceCandidate(
        key="selected-1",
        title="Status",
        url="https://status.example",
        snippets=[EvidenceSnippet(text="Current status and context")],
        provider="brave",
        source_kind="web",
        quality_score=0.8,
        selected=True,
    )
    rejected_candidate = EvidenceCandidate(
        key="rejected-1",
        title="Impact",
        url="https://impact.example",
        snippets=[EvidenceSnippet(text="Impact details and consequences")],
        provider="docs",
        source_kind="docs",
        quality_score=0.1,
        selected=False,
    )
    ledger = EvidenceLedger(
        considered=[selected_candidate, rejected_candidate],
        selected=[selected_candidate],
        rejected=[rejected_candidate],
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.5


def test_evaluate_coverage_ignores_all_rejected_evidence(monkeypatch) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
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

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.0


def test_evaluate_coverage_uses_selected_flags_for_legacy_entries_only_ledger(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="selected-1",
                title="Status",
                url="https://status.example",
                snippets=[EvidenceSnippet(text="Current status and context")],
                provider="brave",
                source_kind="web",
                selected=True,
            ),
            EvidenceCandidate(
                key="rejected-1",
                title="Impact",
                url="https://impact.example",
                snippets=[EvidenceSnippet(text="Impact details and consequences")],
                provider="docs",
                source_kind="docs",
                selected=False,
            ),
        ]
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.5


def test_evaluate_coverage_counts_matched_subtopics_without_text_match(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    selected_candidate = EvidenceCandidate(
        key="selected-1",
        title="Operational guide",
        url="https://ops.example",
        snippets=[EvidenceSnippet(text="Playbook and runbook")],
        provider="docs",
        source_kind="docs",
        matched_subtopics=["impact"],
        quality_score=0.8,
        selected=True,
    )
    ledger = EvidenceLedger(
        considered=[selected_candidate],
        selected=[selected_candidate],
        rejected=[],
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.5


def test_evaluate_coverage_reports_uncovered_subtopics_and_preserves_checkpoint_metadata(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    selected_candidate = EvidenceCandidate(
        key="selected-1",
        title="Status",
        url="https://status.example",
        snippets=[EvidenceSnippet(text="Current status and context")],
        provider="brave",
        source_kind="web",
        quality_score=0.8,
        selected=True,
    )
    ledger = EvidenceLedger(
        considered=[selected_candidate],
        selected=[selected_candidate],
        rejected=[],
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.5
    assert coverage.uncovered_subtopics == ["impact"]
    assert module.evaluate_coverage._checkpoint_type == "tool_call"


def test_evaluate_coverage_reports_all_subtopics_as_uncovered_without_entries(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    coverage = module.evaluate_coverage(EvidenceLedger(entries=[]), _sample_plan())

    assert coverage.uncovered_subtopics == ["status", "impact"]


def test_evaluate_coverage_combines_selected_and_legacy_considered_entries(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    legacy_candidate = EvidenceCandidate(
        key="legacy-1",
        title="Impact briefing",
        url="https://impact.example",
        provider="docs",
        source_kind="docs",
        matched_subtopics=["impact"],
    )
    selected_candidate = EvidenceCandidate(
        key="selected-1",
        title="Status update",
        url="https://status.example",
        provider="brave",
        source_kind="web",
        matched_subtopics=["status"],
        quality_score=0.8,
        selected=True,
    )
    ledger = EvidenceLedger(
        considered=[legacy_candidate, selected_candidate],
        selected=[selected_candidate],
        rejected=[],
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 1.0
    assert coverage.uncovered_subtopics == []


def test_evaluate_coverage_combines_selected_and_legacy_entries_only_ledger(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    legacy_candidate = EvidenceCandidate(
        key="legacy-1",
        title="Impact briefing",
        url="https://impact.example",
        provider="docs",
        source_kind="docs",
        matched_subtopics=["impact"],
    )
    selected_candidate = EvidenceCandidate(
        key="selected-1",
        title="Status update",
        url="https://status.example",
        provider="brave",
        source_kind="web",
        matched_subtopics=["status"],
        selected=True,
    )
    ledger = EvidenceLedger(entries=[legacy_candidate, selected_candidate])

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 1.0
    assert coverage.uncovered_subtopics == []


def test_evaluate_coverage_uses_legacy_considered_entries_when_selected_and_rejected_are_empty(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    legacy_candidate = EvidenceCandidate(
        key="legacy-1",
        title="Impact briefing",
        url="https://impact.example",
        provider="docs",
        source_kind="docs",
        matched_subtopics=["impact"],
    )
    ledger = EvidenceLedger(considered=[legacy_candidate], selected=[], rejected=[])

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.5
    assert coverage.uncovered_subtopics == ["status"]


def test_evaluate_coverage_ignores_all_rejected_legacy_entries_only_ledger(
    monkeypatch,
) -> None:
    _install_kitaru_checkpoint_stub(monkeypatch)
    module = _import_checkpoint_module("deep_research.checkpoints.evaluate")

    ledger = EvidenceLedger(
        entries=[
            EvidenceCandidate(
                key="rejected-1",
                title="Status",
                url="https://status.example",
                snippets=[EvidenceSnippet(text="Current status and context")],
                provider="brave",
                source_kind="web",
                selected=False,
            )
        ]
    )

    coverage = module.evaluate_coverage(ledger, _sample_plan())

    assert coverage.subtopic_coverage == 0.0


def test_build_selection_graph_returns_selected_entries(monkeypatch) -> None:
    decorated = _install_kitaru_checkpoint_stub(monkeypatch)

    module = _import_checkpoint_module("deep_research.checkpoints.select")
    ledger = EvidenceLedger(entries=[])

    result = module.build_selection_graph(ledger, _sample_plan())

    assert result.items == []
    assert result.gap_coverage_summary == ["status", "impact"]
    assert ("build_selection_graph", "llm_call") in decorated


def test_build_selection_graph_uses_selected_entries_and_gap_summary(
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

    result = module.build_selection_graph(ledger, plan)

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
    assert ("build_selection_graph", "llm_call") in decorated


def test_build_selection_graph_uses_selected_flags_for_legacy_entries_only_ledger(
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

    result = module.build_selection_graph(ledger, _sample_plan())

    assert [item.candidate_key for item in result.items] == ["candidate-1"]
    assert result.gap_coverage_summary == ["impact"]
    assert ("build_selection_graph", "llm_call") in decorated


def test_build_selection_graph_ignores_all_rejected_legacy_entries_only_ledger(
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

    result = module.build_selection_graph(ledger, _sample_plan())

    assert result.items == []
    assert result.gap_coverage_summary == ["status", "impact"]
    assert ("build_selection_graph", "llm_call") in decorated


def test_build_selection_graph_uses_text_coverage_for_legacy_selected_entry_without_matched_subtopics(
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

    result = module.build_selection_graph(ledger, _sample_plan())

    assert [item.candidate_key for item in result.items] == ["candidate-1"]
    assert result.gap_coverage_summary == ["impact"]
    assert ("build_selection_graph", "llm_call") in decorated


def test_build_selection_graph_treats_matched_subtopics_case_insensitively(
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

    result = module.build_selection_graph(ledger, plan)

    assert [item.candidate_key for item in result.items] == ["candidate-1"]
    assert result.gap_coverage_summary == ["Impact"]
    assert ("build_selection_graph", "llm_call") in decorated


def test_build_selection_graph_breaks_score_ties_by_candidate_key(monkeypatch) -> None:
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

    result = module.build_selection_graph(ledger, _sample_plan())

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
