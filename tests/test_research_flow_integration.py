import importlib
import sys
import types
from contextlib import contextmanager

from deep_research.config import ResearchConfig
from deep_research.enums import StopReason, Tier
from deep_research.models import (
    CoherenceCheckpointResult,
    CoverageScore,
    CritiqueCheckpointResult,
    EvidenceLedger,
    GroundingCheckpointResult,
    IterationBudget,
    RawToolResult,
    RenderCheckpointResult,
    RenderPayload,
    ResearchPlan,
    ResearchPreferences,
    RelevanceCheckpointResult,
    RequestClassification,
    SearchAction,
    SearchExecutionResult,
    SelectionGraph,
    SupervisorDecision,
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
    """Temporarily preserve selected modules while integration-test stubs are installed.

    The helper restores any existing runtime modules after each import so lightweight
    stubs do not leak out of the integration-test setup path.
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


def _load_research_flow_module():
    """Import the flow module under lightweight stubs that mimic integration behavior.

    The helper installs fake flow handles and checkpoint submission semantics so tests
    can exercise orchestration logic without needing the full Kitaru runtime.
    """

    class FakeHandle:
        def __init__(self, value):
            """Store the flow return value so tests can mimic Kitaru handle semantics."""
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
        sys.modules.pop("deep_research.checkpoints.council", None)
        return importlib.import_module("deep_research.flow.research_flow")


def _sample_plan() -> ResearchPlan:
    """Return a representative research plan fixture used by integration happy paths.

    The plan stays small enough for test readability while still resembling the shape of
    a real orchestrated plan consumed by the flow.
    """
    return ResearchPlan(
        goal="Learn Kitaru",
        key_questions=["What is it?"],
        subtopics=["overview"],
        queries=["learn kitaru"],
        sections=["Summary"],
        success_criteria=["Return a package"],
    )


def _sample_supervisor_result() -> SupervisorCheckpointResult:
    """Return a minimal supervisor result fixture that still matches flow expectations.

    Integration tests use this canned result to exercise normalization, merge, and cost
    plumbing without depending on real provider output.
    """
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
    """Return a full-coverage fixture for integration tests that should stop cleanly.

    Happy-path flow tests use this score to drive convergence logic toward a completed
    package without needing multiple research iterations.
    """
    return CoverageScore(
        subtopic_coverage=1.0,
        source_diversity=1.0,
        evidence_density=1.0,
        total=1.0,
    )


def _render_result(name: str, markdown: str) -> RenderCheckpointResult:
    return RenderCheckpointResult(
        render=RenderPayload(name=name, content_markdown=markdown),
        budget=IterationBudget(),
    )


def _patch_success_path(module, monkeypatch) -> None:
    """Patch the research flow module into a single-iteration happy path."""
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(lambda brief, classification, tier: _sample_plan()),
    )
    monkeypatch.setattr(
        module,
        "execute_searches",
        _as_checkpoint(lambda decision, config, **kwargs: SearchExecutionResult()),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "extract_candidates",
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
        "update_ledger",
        _as_checkpoint(lambda scored, ledger, config=None: ledger),
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "score_coverage",
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
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(
            lambda selection, ledger, plan, config, **kwargs: _render_result(
                "reading_path", "# RP\n"
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda selection, ledger, plan, iteration_trace, provider_usage_summary, stop_reason, config, **kwargs: (
                _render_result(
                    "backing_report",
                    "# BR\n",
                )
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(
            lambda *args, **kwargs: CritiqueCheckpointResult(
                critique=module.CritiqueResult(
                    dimensions=[],
                    summary="critique",
                    revision_suggestions=[],
                    revision_recommended=False,
                ),
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "apply_revisions",
        _as_checkpoint(lambda renders, critique, plan: renders),
    )
    monkeypatch.setattr(
        module,
        "verify_grounding",
        _as_checkpoint(
            lambda *args, **kwargs: GroundingCheckpointResult(
                grounding=module.GroundingResult(score=1.0, verdicts=[]),
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(
            lambda *args, **kwargs: CoherenceCheckpointResult(
                coherence=module.CoherenceResult(
                    relevance=1.0,
                    logical_flow=1.0,
                    completeness=1.0,
                    consistency=1.0,
                    summary="coherent",
                ),
                budget=IterationBudget(),
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
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(
            lambda *args, **kwargs: CritiqueCheckpointResult(
                critique=module.CritiqueResult(
                    dimensions=[],
                    summary="critique",
                    revision_suggestions=["tighten"],
                    revision_recommended=True,
                ),
                budget=IterationBudget(estimated_cost_usd=0.4),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "apply_revisions",
        _as_checkpoint(lambda renders, critique, plan: renders),
    )
    monkeypatch.setattr(
        module,
        "verify_grounding",
        _as_checkpoint(
            lambda *args, **kwargs: GroundingCheckpointResult(
                grounding=module.GroundingResult(score=1.0, verdicts=[]),
                budget=IterationBudget(estimated_cost_usd=0.3),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(
            lambda *args, **kwargs: CoherenceCheckpointResult(
                coherence=module.CoherenceResult(
                    relevance=1.0,
                    logical_flow=1.0,
                    completeness=1.0,
                    consistency=1.0,
                    summary="coherent",
                ),
                budget=IterationBudget(estimated_cost_usd=0.2),
            )
        ),
    )

    handle = module.research_flow.run(
        "learn kitaru", config=ResearchConfig.for_tier(Tier.DEEP)
    )
    result = handle.wait()

    assert result is not None
    record = result.iteration_trace.iterations[0]

    assert record.new_candidate_count == 0
    assert record.accepted_candidate_count == 0
    assert record.rejected_candidate_count == 0
    assert record.coverage == 1.0
    assert record.coverage_delta == 1.0
    assert record.uncovered_subtopics == []
    assert record.tool_calls == [
        module.ToolCallRecord(
            tool_name="search",
            status="ok",
            provider="test",
            summary="search via test succeeded",
        )
    ]
    assert record.continue_reason is None
    assert record.stop_reason is StopReason.CONVERGED
    assert (
        result.render_settings.writer_model
        == ResearchConfig.for_tier(Tier.DEEP).writer_model
    )
    assert result.run_summary.estimated_cost_usd == 0.9
    assert result.critique_result.summary == "critique"
    assert result.grounding_result.score == 1.0
    assert result.coherence_result.summary == "coherent"


def test_council_flow_aggregates_multiple_generator_results(monkeypatch) -> None:
    module = _load_research_flow_module()

    class FakeFuture:
        def __init__(self, payload):
            """Store the payload that `.load()` should yield for fake council futures."""
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


def test_research_flow_combines_supervisor_and_builtin_search_results(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    normalized_batches = []

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
        _as_checkpoint(
            lambda *args, **kwargs: SupervisorCheckpointResult(
                decision=SupervisorDecision(
                    rationale="search next",
                    search_actions=[
                        SearchAction(query="learn kitaru", rationale="fill gaps")
                    ],
                ),
                raw_results=[
                    RawToolResult(
                        tool_name="mcp_search",
                        provider="openai",
                        payload={"items": ["a"]},
                    )
                ],
                budget=IterationBudget(estimated_cost_usd=0.3),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "execute_searches",
        _as_checkpoint(
            lambda decision, config, **kwargs: SearchExecutionResult(
                raw_results=[
                    RawToolResult(
                        tool_name="provider_search",
                        provider="semantic_scholar",
                        payload={"items": ["b"]},
                    )
                ],
                budget=IterationBudget(estimated_cost_usd=0.2),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "extract_candidates",
        _as_checkpoint(
            lambda raw_results: normalized_batches.append(list(raw_results)) or []
        ),
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
    )

    handle = module.research_flow.run("learn kitaru")
    result = handle.wait()

    assert normalized_batches == [
        [
            RawToolResult(
                tool_name="mcp_search",
                provider="openai",
                payload={"items": ["a"]},
            ),
            RawToolResult(
                tool_name="provider_search",
                provider="semantic_scholar",
                payload={"items": ["b"]},
            ),
        ]
    ]
    assert result.run_summary.provider_usage_summary == {
        "openai": 1,
        "semantic_scholar": 1,
    }
    assert result.iteration_trace.iterations[0].tool_calls == [
        module.ToolCallRecord(
            tool_name="mcp_search",
            status="ok",
            provider="openai",
            summary="mcp_search via openai succeeded",
        ),
        module.ToolCallRecord(
            tool_name="provider_search",
            status="ok",
            provider="semantic_scholar",
            summary="provider_search via semantic_scholar succeeded",
        ),
    ]
    assert result.run_summary.estimated_cost_usd == 0.5
