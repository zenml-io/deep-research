import importlib
import sys
import types
from datetime import datetime

from deep_research.models import (
    CoherenceCheckpointResult,
    CoherenceResult,
    CoverageScore,
    CritiqueCheckpointResult,
    CritiqueResult,
    GroundingCheckpointResult,
    GroundingResult,
    IterationBudget,
    RawToolResult,
    RenderCheckpointResult,
    RenderPayload,
    RequestClassification,
    ResearchPlan,
    ResearchPreferences,
    RelevanceCheckpointResult,
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


def _load_research_flow_module():
    """Import the flow module with lightweight Kitaru and PydanticAI test stubs."""

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
    sys.modules.pop("deep_research.flow.research_flow", None)
    return importlib.import_module("deep_research.flow.research_flow")


def _render_result(
    name: str,
    markdown: str,
    *,
    cost: float = 0.0,
) -> RenderCheckpointResult:
    return RenderCheckpointResult(
        render=RenderPayload(name=name, content_markdown=markdown),
        budget=IterationBudget(estimated_cost_usd=cost),
    )


def test_research_flow_uses_council_path_when_enabled(monkeypatch) -> None:
    module = _load_research_flow_module()
    council_calls = []

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda *args, **kwargs: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
                recommended_tier=module.Tier.DEEP,
                needs_clarification=False,
                clarification_question=None,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda *args, **kwargs: ResearchPlan(
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "run_council_generator",
        type(
            "CouncilGenerator",
            (),
            {
                "submit": staticmethod(
                    lambda *args, **kwargs: types.SimpleNamespace(
                        load=lambda: (
                            council_calls.append((args, kwargs))
                            or SupervisorCheckpointResult(
                                raw_results=[],
                                budget=IterationBudget(),
                            )
                        )
                    )
                )
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "aggregate_council_results",
        _as_checkpoint(
            lambda results: SupervisorCheckpointResult(
                raw_results=[],
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("should not run")
            )
        ),
    )
    monkeypatch.setattr(
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        _as_checkpoint(
            lambda candidates, plan, config: RelevanceCheckpointResult(
                candidates=[], budget=IterationBudget()
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
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
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
        _as_checkpoint(
            lambda ledger, plan: CoverageScore(
                subtopic_coverage=1.0,
                source_diversity=1.0,
                evidence_density=1.0,
                total=1.0,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=True, reason=module.StopReason.CONVERGED
        ),
    )
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(
            lambda *args, **kwargs: CritiqueCheckpointResult(
                critique=CritiqueResult(dimensions=[], summary="skip"),
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
                grounding=GroundingResult(score=1.0, verdicts=[]),
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(
            lambda *args, **kwargs: CoherenceCheckpointResult(
                coherence=CoherenceResult(
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
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    module.research_flow.run(
        "brief",
        config=module.ResearchConfig.for_tier(module.Tier.DEEP).model_copy(
            update={"council_mode": True, "council_size": 3}
        ),
    )

    assert len(council_calls) == 3


def test_research_flow_uses_supervisor_path_when_council_disabled(monkeypatch) -> None:
    module = _load_research_flow_module()
    supervisor_calls = []

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda *args, **kwargs: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
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
            lambda *args, **kwargs: ResearchPlan(
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: (
                supervisor_calls.append((args, kwargs))
                or SupervisorCheckpointResult(raw_results=[], budget=IterationBudget())
            )
        ),
    )
    monkeypatch.setattr(
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        _as_checkpoint(
            lambda candidates, plan, config: RelevanceCheckpointResult(
                candidates=[], budget=IterationBudget()
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
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
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
        _as_checkpoint(
            lambda ledger, plan: CoverageScore(
                subtopic_coverage=1.0,
                source_diversity=1.0,
                evidence_density=1.0,
                total=1.0,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=True, reason=module.StopReason.CONVERGED
        ),
    )
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    module.research_flow.run(
        "brief",
        config=module.ResearchConfig.for_tier(module.Tier.STANDARD),
    )

    assert len(supervisor_calls) == 1


def test_research_flow_recomputes_tier_after_clarification(monkeypatch) -> None:
    module = _load_research_flow_module()
    classifications = [
        types.SimpleNamespace(
            recommended_tier=module.Tier.STANDARD,
            needs_clarification=True,
            clarification_question="Need more detail?",
            preferences=ResearchPreferences(),
        ),
        types.SimpleNamespace(
            recommended_tier=module.Tier.DEEP,
            needs_clarification=False,
            clarification_question=None,
            preferences=ResearchPreferences(),
        ),
    ]
    build_plan_tiers = []

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(lambda brief, config: classifications.pop(0)),
    )
    monkeypatch.setattr(
        module,
        "wait",
        lambda **kwargs: (
            "clarified brief" if kwargs["name"] == "clarify_brief" else True
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: (
                build_plan_tiers.append(tier)
                or types.SimpleNamespace(goal=f"plan for {brief}")
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
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        _as_checkpoint(lambda ledger, config: ledger),
        raising=False,
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(
            lambda *args, **kwargs: CritiqueCheckpointResult(
                critique=CritiqueResult(dimensions=[], summary="skipped"),
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
                grounding=GroundingResult(score=1.0, verdicts=[]),
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(
            lambda *args, **kwargs: CoherenceCheckpointResult(
                coherence=CoherenceResult(
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
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run("initial brief")

    assert build_plan_tiers == [module.Tier.DEEP]
    assert package["run_summary"].tier is module.Tier.DEEP
    assert package["run_summary"].brief == "clarified brief"


def test_research_flow_rebuilds_default_config_after_clarification_tier_change(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    classifications = [
        types.SimpleNamespace(
            recommended_tier=module.Tier.STANDARD,
            needs_clarification=True,
            clarification_question="Need more detail?",
            preferences=ResearchPreferences(),
        ),
        types.SimpleNamespace(
            recommended_tier=module.Tier.DEEP,
            needs_clarification=False,
            clarification_question=None,
            preferences=ResearchPreferences(),
        ),
    ]
    observed_configs = []

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(lambda brief, config: classifications.pop(0)),
    )
    monkeypatch.setattr(
        module,
        "wait",
        lambda **kwargs: (
            "clarified brief" if kwargs["name"] == "clarify_brief" else True
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: types.SimpleNamespace(goal="goal")
        ),
    )

    def fake_run_supervisor(
        plan, ledger, iteration, config, uncovered_subtopics=None, **kwargs
    ):
        observed_configs.append(config)
        return types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        )

    monkeypatch.setattr(module, "run_supervisor", _as_checkpoint(fake_run_supervisor))
    monkeypatch.setattr(
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(
            lambda *args, **kwargs: CritiqueCheckpointResult(
                critique=CritiqueResult(dimensions=[], summary="skipped"),
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
                grounding=GroundingResult(score=1.0, verdicts=[]),
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(
            lambda *args, **kwargs: CoherenceCheckpointResult(
                coherence=CoherenceResult(
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
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    module.research_flow.run("initial brief")

    config = observed_configs[0]
    deep_defaults = module.ResearchConfig.for_tier(module.Tier.DEEP)
    assert config.tier is module.Tier.DEEP
    assert config.max_iterations == deep_defaults.max_iterations
    assert config.cost_budget_usd == deep_defaults.cost_budget_usd
    assert config.time_box_seconds == deep_defaults.time_box_seconds
    assert config.critique_enabled is deep_defaults.critique_enabled
    assert config.judge_enabled is deep_defaults.judge_enabled


def test_research_flow_passes_elapsed_seconds_to_convergence(monkeypatch) -> None:
    module = _load_research_flow_module()
    monotonic_values = iter([10.0, 16.2, 16.2])

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: types.SimpleNamespace(goal="goal")
        ),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
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
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
        _as_checkpoint(
            lambda ledger, plan: types.SimpleNamespace(
                total=0.1, uncovered_subtopics=[]
            )
        ),
    )

    def fake_check_convergence(*args, **kwargs):
        if kwargs["elapsed_seconds"] >= 6:
            return types.SimpleNamespace(
                should_stop=True,
                reason=module.StopReason.TIME_EXHAUSTED,
            )
        return types.SimpleNamespace(should_stop=False, reason=None)

    monkeypatch.setattr(module, "check_convergence", fake_check_convergence)
    monkeypatch.setattr(module, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run("brief")

    assert package["run_summary"].stop_reason is module.StopReason.TIME_EXHAUSTED


def test_research_flow_populates_phase_one_run_summary_metadata(monkeypatch) -> None:
    module = _load_research_flow_module()
    monotonic_values = iter([100.0, 102.2, 104.9, 107.8])
    supervisor_results = iter(
        [
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="tavily", tool_name="search", ok=True, error=None
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=1.25),
            ),
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="tavily", tool_name="search", ok=True, error=None
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=1.25),
            ),
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="tavily", tool_name="search", ok=True, error=None
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=1.25),
            ),
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="serpapi", tool_name="search", ok=True, error=None
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=2.0),
            ),
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="serpapi", tool_name="search", ok=True, error=None
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=2.0),
            ),
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        provider="openai", tool_name="search", ok=True, error=None
                    ),
                    types.SimpleNamespace(
                        provider="serpapi", tool_name="search", ok=True, error=None
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=2.0),
            ),
        ]
    )
    relevance_results = iter(
        [
            types.SimpleNamespace(
                candidates=[types.SimpleNamespace(key="c1")],
                budget=types.SimpleNamespace(estimated_cost_usd=0.5),
            ),
            types.SimpleNamespace(
                candidates=[types.SimpleNamespace(key="c2")],
                budget=types.SimpleNamespace(estimated_cost_usd=0.25),
            ),
        ]
    )
    coverages = iter([0.35, 0.8])

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: types.SimpleNamespace(goal="goal")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_council_generator",
        type(
            "CouncilGenerator",
            (),
            {
                "submit": staticmethod(
                    lambda *args, **kwargs: types.SimpleNamespace(
                        load=lambda: next(supervisor_results)
                    )
                )
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "aggregate_council_results",
        _as_checkpoint(lambda results: results[0]),
    )
    monkeypatch.setattr(
        module,
        "extract_candidates",
        _as_checkpoint(lambda raw_results: list(raw_results)),
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        _as_checkpoint(lambda candidates, plan, config: next(relevance_results)),
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
        _as_checkpoint(
            lambda ledger, plan: types.SimpleNamespace(
                total=next(coverages), uncovered_subtopics=[]
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=kwargs["elapsed_seconds"] >= 4,
            reason=(
                module.StopReason.CONVERGED if kwargs["elapsed_seconds"] >= 4 else None
            ),
        ),
    )
    monkeypatch.setattr(module, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("reading_path", "# RP", cost=0.2)
        ),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR", cost=0.1)
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    config = module.ResearchConfig.for_tier(module.Tier.STANDARD).model_copy(
        update={
            "council_mode": True,
            "council_size": 3,
            "supervisor_model": "provider/supervisor",
        }
    )

    package = module.research_flow.run("brief", config=config)

    run_summary = package["run_summary"]
    iteration_trace = package["iteration_trace"]

    assert run_summary.estimated_cost_usd == 4.3
    assert run_summary.elapsed_seconds == 7
    assert run_summary.iteration_count == 2
    assert run_summary.provider_usage_summary == {
        "openai": 3,
        "tavily": 1,
        "serpapi": 1,
    }
    assert run_summary.council_enabled is True
    assert run_summary.council_size == 3
    assert run_summary.council_models == ["provider/supervisor"] * 3
    assert datetime.fromisoformat(run_summary.started_at.replace("Z", "+00:00"))
    assert datetime.fromisoformat(run_summary.completed_at.replace("Z", "+00:00"))
    assert [record.estimated_cost_usd for record in iteration_trace.iterations] == [
        1.75,
        2.25,
    ]


def test_research_flow_records_richer_iteration_trace_and_logs_stop_metadata(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    supervisor_results = iter(
        [
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        tool_name="search",
                        provider="openai",
                        ok=True,
                    ),
                    types.SimpleNamespace(
                        tool_name="browse",
                        provider="tavily",
                        ok=False,
                        error="timeout",
                    ),
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=0.6),
            ),
            types.SimpleNamespace(
                raw_results=[
                    types.SimpleNamespace(
                        tool_name="search",
                        provider="openai",
                        ok=True,
                    )
                ],
                budget=types.SimpleNamespace(estimated_cost_usd=0.4),
            ),
        ]
    )
    relevance_results = iter(
        [
            types.SimpleNamespace(
                candidates=[types.SimpleNamespace(key="accepted-1")],
                budget=types.SimpleNamespace(estimated_cost_usd=0.2),
            ),
            types.SimpleNamespace(
                candidates=[types.SimpleNamespace(key="accepted-2")],
                budget=types.SimpleNamespace(estimated_cost_usd=0.3),
            ),
        ]
    )
    merged_ledgers = iter(
        [
            types.SimpleNamespace(
                selected=[types.SimpleNamespace(key="accepted-1")],
                rejected=[types.SimpleNamespace(key="rejected-1")],
            ),
            types.SimpleNamespace(
                selected=[
                    types.SimpleNamespace(key="accepted-1"),
                    types.SimpleNamespace(key="accepted-2"),
                ],
                rejected=[types.SimpleNamespace(key="rejected-1")],
            ),
        ]
    )
    coverages = iter(
        [
            types.SimpleNamespace(total=0.35, uncovered_subtopics=["gaps-a", "gaps-b"]),
            types.SimpleNamespace(total=0.8, uncovered_subtopics=[]),
        ]
    )
    decisions = iter(
        [
            types.SimpleNamespace(should_stop=False, reason=None),
            types.SimpleNamespace(
                should_stop=True,
                reason=module.StopReason.CONVERGED,
            ),
        ]
    )
    convergence_calls = []
    log_calls = []

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: types.SimpleNamespace(goal="goal")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(lambda *args, **kwargs: next(supervisor_results)),
    )
    monkeypatch.setattr(
        module,
        "extract_candidates",
        _as_checkpoint(lambda raw_results: list(raw_results)),
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        _as_checkpoint(lambda candidates, plan, config: next(relevance_results)),
    )
    monkeypatch.setattr(
        module,
        "update_ledger",
        _as_checkpoint(lambda scored, ledger, config=None: next(merged_ledgers)),
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
        _as_checkpoint(lambda ledger, plan: next(coverages)),
    )

    def fake_check_convergence(*args, **kwargs):
        kwargs["history_snapshot"] = list(args[1])
        convergence_calls.append(kwargs)
        return next(decisions)

    monkeypatch.setattr(module, "check_convergence", fake_check_convergence)
    monkeypatch.setattr(module, "log", lambda **kwargs: log_calls.append(kwargs))
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run("brief")

    records = package["iteration_trace"].iterations

    assert convergence_calls[0]["history_snapshot"] == []
    assert len(convergence_calls[1]["history_snapshot"]) == 1
    assert convergence_calls[1]["history_snapshot"][0].iteration == 0
    assert convergence_calls[1]["history_snapshot"][0].coverage == 0.35
    assert [call["new_candidate_count"] for call in convergence_calls] == [2, 1]
    assert records[0].accepted_candidate_count == 1
    assert records[0].rejected_candidate_count == 1
    assert records[0].coverage_delta == 0.35
    assert records[0].uncovered_subtopics == ["gaps-a", "gaps-b"]
    assert records[0].tool_calls == [
        module.ToolCallRecord(
            tool_name="search",
            status="ok",
            provider="openai",
            summary="search via openai succeeded",
        ),
        module.ToolCallRecord(
            tool_name="browse",
            status="error",
            provider="tavily",
            summary="browse via tavily failed: timeout",
        ),
    ]
    assert records[0].stop_reason is None
    assert records[0].continue_reason == "remaining uncovered subtopics: gaps-a, gaps-b"
    assert records[1].accepted_candidate_count == 2
    assert records[1].rejected_candidate_count == 1
    assert records[1].coverage_delta == 0.45
    assert records[1].uncovered_subtopics == []
    assert records[1].stop_reason is module.StopReason.CONVERGED
    assert records[1].continue_reason is None
    assert log_calls == [
        {
            "iteration": 0,
            "coverage": 0.35,
            "coverage_delta": 0.35,
            "uncovered_subtopics": ["gaps-a", "gaps-b"],
            "new_candidate_count": 2,
            "accepted_candidate_count": 1,
            "rejected_candidate_count": 1,
            "tool_summaries": [
                "search via openai succeeded",
                "browse via tavily failed: timeout",
            ],
            "stop_reason": None,
            "continue_reason": "remaining uncovered subtopics: gaps-a, gaps-b",
            "spent_usd": 0.8,
        },
        {
            "iteration": 1,
            "coverage": 0.8,
            "coverage_delta": 0.45,
            "uncovered_subtopics": [],
            "new_candidate_count": 1,
            "accepted_candidate_count": 2,
            "rejected_candidate_count": 1,
            "tool_summaries": ["search via openai succeeded"],
            "stop_reason": module.StopReason.CONVERGED,
            "continue_reason": None,
            "spent_usd": 1.5,
        },
    ]


def test_research_flow_records_completion_after_eager_render_loads(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    current_time = {"value": 10.0}

    class FakeFuture:
        def __init__(self, value, *, loaded_time: float):
            """Store a payload and the clock value that should be observed when loading it."""
            self.value = value
            self.loaded_time = loaded_time

        def load(self):
            current_time["value"] = self.loaded_time
            return self.value

    monkeypatch.setattr(module, "monotonic", lambda: current_time["value"])
    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: types.SimpleNamespace(
                goal="goal",
                subtopics=["topic"],
                key_questions=["q"],
            )
        ),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
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
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        types.SimpleNamespace(
            submit=lambda selection, ledger, plan, config, **kwargs: FakeFuture(
                _render_result("reading_path", "# RP"),
                loaded_time=14.0,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        types.SimpleNamespace(
            submit=lambda selection, ledger, plan, iteration_trace, provider_usage_summary, stop_reason, config, **kwargs: (
                FakeFuture(
                    _render_result("backing_report", "# BR"),
                    loaded_time=18.0,
                )
            )
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run("brief")

    assert package["run_summary"].elapsed_seconds == 8


def test_research_flow_preserves_disabled_critique_and_judges(monkeypatch) -> None:
    module = _load_research_flow_module()

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda *args, **kwargs: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
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
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: SupervisorCheckpointResult(
                raw_results=[],
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
        _as_checkpoint(
            lambda ledger, plan: CoverageScore(
                subtopic_coverage=1.0,
                source_diversity=1.0,
                evidence_density=1.0,
                total=1.0,
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("should not run")
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_grounding",
        _as_checkpoint(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("should not run")
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("should not run")
            )
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run(
        "brief",
        config=module.ResearchConfig.for_tier(module.Tier.STANDARD),
    )

    assert package["critique_result"] is None
    assert package["grounding_result"] is None
    assert package["coherence_result"] is None


def test_research_flow_runs_critique_and_judges_when_enabled(monkeypatch) -> None:
    module = _load_research_flow_module()

    critique = CritiqueResult(
        dimensions=[],
        summary="critique",
        revision_suggestions=["tighten"],
        revision_recommended=True,
    )
    critique_checkpoint = CritiqueCheckpointResult(
        critique=critique,
        budget=IterationBudget(estimated_cost_usd=0.4),
    )
    coherence = CoherenceResult(
        relevance=1.0,
        logical_flow=1.0,
        completeness=1.0,
        consistency=1.0,
        summary="coherent",
    )
    grounding_checkpoint = GroundingCheckpointResult(
        grounding=GroundingResult(score=1.0, verdicts=[]),
        budget=IterationBudget(estimated_cost_usd=0.3),
    )
    coherence_checkpoint = CoherenceCheckpointResult(
        coherence=coherence,
        budget=IterationBudget(estimated_cost_usd=0.2),
    )

    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda *args, **kwargs: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
                recommended_tier=module.Tier.DEEP,
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
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: SupervisorCheckpointResult(
                raw_results=[],
                budget=IterationBudget(),
            )
        ),
    )
    monkeypatch.setattr(
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
        _as_checkpoint(
            lambda ledger, plan: CoverageScore(
                subtopic_coverage=1.0,
                source_diversity=1.0,
                evidence_density=1.0,
                total=1.0,
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(
        module,
        "critique_reports",
        _as_checkpoint(lambda *args, **kwargs: critique_checkpoint),
    )
    monkeypatch.setattr(
        module,
        "apply_revisions",
        _as_checkpoint(lambda renders, critique, plan: renders),
    )
    monkeypatch.setattr(
        module,
        "verify_grounding",
        _as_checkpoint(lambda *args, **kwargs: grounding_checkpoint),
    )
    monkeypatch.setattr(
        module,
        "verify_coherence",
        _as_checkpoint(lambda *args, **kwargs: coherence_checkpoint),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run(
        "brief",
        config=module.ResearchConfig.for_tier(module.Tier.DEEP),
    )

    assert package["critique_result"].summary == "critique"
    assert package["grounding_result"].score == 1.0
    assert package["coherence_result"].summary == "coherent"
    assert package["run_summary"].estimated_cost_usd == 0.9


def test_research_flow_preserves_overrides_when_clarification_changes_tier(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    incoming_config = module.ResearchConfig.for_tier(module.Tier.STANDARD).model_copy(
        update={
            "council_mode": True,
            "council_size": 5,
            "supervisor_model": "custom/supervisor",
            "planner_model": "custom/planner",
            "cost_budget_usd": 9.5,
            "time_box_seconds": 321,
        }
    )
    classifications = [
        types.SimpleNamespace(
            recommended_tier=module.Tier.STANDARD,
            needs_clarification=True,
            clarification_question="Clarify?",
            preferences=ResearchPreferences(),
        ),
        types.SimpleNamespace(
            recommended_tier=module.Tier.DEEP,
            needs_clarification=False,
            clarification_question=None,
            preferences=ResearchPreferences(),
        ),
    ]
    observed_configs = []

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(lambda brief, config: classifications.pop(0)),
    )
    monkeypatch.setattr(
        module,
        "wait",
        lambda **kwargs: (
            "clarified brief" if kwargs["name"] == "clarify_brief" else True
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: types.SimpleNamespace(goal="goal")
        ),
    )

    def fake_run_council_generator(
        plan, ledger, iteration, model_name, config, uncovered_subtopics=None, **kwargs
    ):
        observed_configs.append(config)
        return types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        )

    monkeypatch.setattr(
        module,
        "run_council_generator",
        type(
            "CouncilGenerator",
            (),
            {
                "submit": staticmethod(
                    lambda *args, **kwargs: types.SimpleNamespace(
                        load=lambda: fake_run_council_generator(*args, **kwargs)
                    )
                )
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "aggregate_council_results",
        _as_checkpoint(lambda results: results[-1]),
    )
    monkeypatch.setattr(
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    module.research_flow.run("initial brief", config=incoming_config)

    config = observed_configs[0]
    assert config.tier is module.Tier.DEEP
    assert config.council_mode is True
    assert config.council_size == 5
    assert config.supervisor_model == "custom/supervisor"
    assert config.planner_model == "custom/planner"
    assert config.cost_budget_usd == 9.5
    assert config.time_box_seconds == 321


def test_research_flow_calls_all_checkpoints(monkeypatch) -> None:
    """Verify the flow materializes every checkpoint via .submit().load()."""
    module = _load_research_flow_module()
    called = set()

    def track(name, result):
        def fn(*args, **kwargs):
            called.add(name)
            return result

        return _as_checkpoint(fn)

    classification = types.SimpleNamespace(
        recommended_tier=module.Tier.STANDARD,
        needs_clarification=False,
        clarification_question=None,
        preferences=ResearchPreferences(),
    )
    plan = types.SimpleNamespace(goal="goal", subtopics=["topic"], key_questions=["q"])

    monkeypatch.setattr(module, "classify_request", track("classify", classification))
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(module, "build_plan", track("plan", plan))
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: types.SimpleNamespace(
                decision=SupervisorDecision(
                    rationale="search next",
                    search_actions=[SearchAction(query="q", rationale="because")],
                ),
                raw_results=[],
                budget=types.SimpleNamespace(estimated_cost_usd=0.0),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "execute_searches",
        track(
            "execute_searches",
            SearchExecutionResult(budget=IterationBudget()),
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "extract_candidates", track("normalize", []))
    monkeypatch.setattr(
        module,
        "score_relevance",
        track(
            "score",
            types.SimpleNamespace(
                candidates=[],
                budget=types.SimpleNamespace(estimated_cost_usd=0.0),
            ),
        ),
    )
    monkeypatch.setattr(
        module, "update_ledger", track("merge", module.EvidenceLedger())
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        track("enrich_candidates", module.EvidenceLedger()),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "score_coverage",
        track("coverage", types.SimpleNamespace(total=1.0, uncovered_subtopics=[])),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: types.SimpleNamespace(
            should_stop=True,
            reason=module.StopReason.CONVERGED,
        ),
    )
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        track("selection", SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        track("reading", _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        track("backing", _render_result("backing_report", "# BR")),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    module.research_flow.run("brief")

    assert called == {
        "classify",
        "plan",
        "execute_searches",
        "normalize",
        "score",
        "merge",
        "enrich_candidates",
        "coverage",
        "selection",
        "reading",
        "backing",
    }


def test_research_flow_passes_loaded_values_to_terminal_checkpoints(
    monkeypatch,
) -> None:
    """Verify renderers receive loaded values and assemble_package gets correct args."""
    module = _load_research_flow_module()
    selection = SelectionGraph(items=[])
    ledger = module.EvidenceLedger()
    plan = ResearchPlan(
        goal="goal",
        key_questions=["k"],
        subtopics=["status"],
        queries=["q"],
        sections=["Summary"],
        success_criteria=["c"],
    )
    captured = {}

    def fake_render_reading(
        sel, current_ledger, current_plan, current_config, **kwargs
    ):
        captured["reading_selection"] = sel
        captured["reading_ledger"] = current_ledger
        captured["reading_plan"] = current_plan
        captured["reading_config"] = current_config
        return _render_result("reading_path", "# RP", cost=0.2)

    def fake_render_backing(
        sel,
        current_ledger,
        current_plan,
        iteration_trace,
        provider_usage_summary,
        stop_reason,
        current_config,
        **kwargs,
    ):
        captured["backing_selection"] = sel
        captured["backing_ledger"] = current_ledger
        captured["backing_plan"] = current_plan
        captured["backing_iteration_trace"] = iteration_trace
        captured["backing_provider_usage_summary"] = provider_usage_summary
        captured["backing_stop_reason"] = stop_reason
        captured["backing_config"] = current_config
        return _render_result("backing_report", "# BR", cost=0.1)

    def fake_assemble(**kwargs):
        captured["assemble_kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(lambda brief, classification, tier: plan),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
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
        module, "extract_candidates", _as_checkpoint(lambda raw_results: [])
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
        "update_ledger",
        _as_checkpoint(lambda scored, current_ledger, config=None: ledger),
    )
    monkeypatch.setattr(
        module,
        "enrich_candidates",
        _as_checkpoint(lambda current_ledger, config: ledger),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "score_coverage",
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: selection),
    )
    monkeypatch.setattr(
        module, "write_reading_path", _as_checkpoint(fake_render_reading)
    )
    monkeypatch.setattr(
        module, "write_backing_report", _as_checkpoint(fake_render_backing)
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(fake_assemble))

    module.research_flow.run("brief")

    assert captured["reading_selection"] is selection
    assert captured["reading_ledger"] is ledger
    assert captured["reading_plan"] is plan
    assert captured["backing_selection"] is selection
    assert captured["backing_ledger"] is ledger
    assert captured["backing_plan"] is plan
    assert captured["backing_iteration_trace"].iterations[0].stop_reason is (
        module.StopReason.CONVERGED
    )
    assert captured["backing_provider_usage_summary"] == {}
    assert captured["backing_stop_reason"] is module.StopReason.CONVERGED
    assert captured["assemble_kwargs"]["selection_graph"] is selection
    assert captured["assemble_kwargs"]["renders"] == [
        RenderPayload(name="reading_path", content_markdown="# RP"),
        RenderPayload(name="backing_report", content_markdown="# BR"),
    ]
    assert captured["assemble_kwargs"]["run_summary"].estimated_cost_usd == 0.3
    assert len(captured["assemble_kwargs"]["renders"]) == 2


def test_research_flow_executes_built_in_searches_then_fetches_before_coverage(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    call_order = []
    normalized_inputs = []

    supervisor_raw = RawToolResult(
        tool_name="supervisor_search",
        provider="openai",
        payload={"items": ["supervisor"]},
    )
    built_in_raw = RawToolResult(
        tool_name="provider_search",
        provider="serpapi",
        payload={"items": ["built-in"]},
    )
    decision = SupervisorDecision(
        rationale="search next",
        search_actions=[SearchAction(query="q", rationale="because")],
    )

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: ResearchPlan(
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: SupervisorCheckpointResult(
                decision=decision,
                raw_results=[supervisor_raw],
                budget=IterationBudget(estimated_cost_usd=0.6),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "execute_searches",
        _as_checkpoint(
            lambda incoming_decision, config, **kwargs: (
                call_order.append("execute_searches")
                or SearchExecutionResult(
                    raw_results=[built_in_raw],
                    budget=IterationBudget(estimated_cost_usd=0.4),
                )
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "extract_candidates",
        _as_checkpoint(
            lambda raw_results: (
                call_order.append("extract_candidates")
                or normalized_inputs.append(list(raw_results))
                or [types.SimpleNamespace(key="candidate-1")]
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        _as_checkpoint(
            lambda candidates, plan, config: RelevanceCheckpointResult(
                candidates=[],
                budget=IterationBudget(estimated_cost_usd=0.2),
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
        _as_checkpoint(
            lambda ledger, config: call_order.append("enrich_candidates") or ledger
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "score_coverage",
        _as_checkpoint(
            lambda ledger, plan: (
                call_order.append("score_coverage")
                or CoverageScore(
                    subtopic_coverage=1.0,
                    source_diversity=1.0,
                    evidence_density=1.0,
                    total=1.0,
                )
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run("brief")

    assert call_order == [
        "execute_searches",
        "extract_candidates",
        "enrich_candidates",
        "score_coverage",
    ]
    assert normalized_inputs == [[supervisor_raw, built_in_raw]]
    assert package["run_summary"].estimated_cost_usd == 1.2
    assert package["run_summary"].provider_usage_summary == {"openai": 1, "serpapi": 1}
    assert package["iteration_trace"].iterations[0].tool_calls == [
        module.ToolCallRecord(
            tool_name="supervisor_search",
            status="ok",
            provider="openai",
            summary="supervisor_search via openai succeeded",
        ),
        module.ToolCallRecord(
            tool_name="provider_search",
            status="ok",
            provider="serpapi",
            summary="provider_search via serpapi succeeded",
        ),
    ]


def test_research_flow_skips_built_in_search_execution_when_no_search_actions(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()
    normalized_inputs = []

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: types.SimpleNamespace(
                recommended_tier=module.Tier.STANDARD,
                needs_clarification=False,
                clarification_question=None,
                preferences=ResearchPreferences(),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda brief, classification, tier: ResearchPlan(
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(
            lambda *args, **kwargs: SupervisorCheckpointResult(
                decision=SupervisorDecision(rationale="no-op", search_actions=[]),
                raw_results=[],
                budget=IterationBudget(estimated_cost_usd=0.1),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "execute_searches",
        _as_checkpoint(
            lambda decision, config, **kwargs: (_ for _ in ()).throw(
                AssertionError("should not run")
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "extract_candidates",
        _as_checkpoint(
            lambda raw_results: normalized_inputs.append(list(raw_results)) or []
        ),
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
        _as_checkpoint(
            lambda ledger, plan: CoverageScore(
                subtopic_coverage=1.0,
                source_diversity=1.0,
                evidence_density=1.0,
                total=1.0,
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    package = module.research_flow.run("brief")

    assert normalized_inputs == [[]]
    assert package["run_summary"].provider_usage_summary == {}
    assert package["iteration_trace"].iterations[0].tool_calls == []


def test_research_flow_council_submission_ids_include_iteration(monkeypatch) -> None:
    module = _load_research_flow_module()
    council_submission_ids = []
    coverage_totals = iter([0.1, 1.0])
    convergence_decisions = iter(
        [
            types.SimpleNamespace(should_stop=False, reason=None),
            types.SimpleNamespace(
                should_stop=True,
                reason=module.StopReason.CONVERGED,
            ),
        ]
    )

    monkeypatch.setattr(
        module,
        "classify_request",
        _as_checkpoint(
            lambda brief, config: RequestClassification(
                audience_mode="technical",
                freshness_mode="current",
                recommended_tier=module.Tier.DEEP,
                needs_clarification=False,
                clarification_question=None,
            )
        ),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "build_plan",
        _as_checkpoint(
            lambda *args, **kwargs: ResearchPlan(
                goal="goal",
                key_questions=["k"],
                subtopics=["status"],
                queries=["q"],
                sections=["Summary"],
                success_criteria=["c"],
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "run_council_generator",
        type(
            "CouncilGenerator",
            (),
            {
                "submit": staticmethod(
                    lambda *args, **kwargs: types.SimpleNamespace(
                        load=lambda: (
                            council_submission_ids.append(kwargs["id"])
                            or SupervisorCheckpointResult(
                                decision=SupervisorDecision(
                                    rationale="no search",
                                    search_actions=[],
                                ),
                                raw_results=[],
                                budget=IterationBudget(),
                            )
                        )
                    )
                )
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "aggregate_council_results",
        _as_checkpoint(lambda results: results[0]),
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
        _as_checkpoint(
            lambda ledger, plan: CoverageScore(
                subtopic_coverage=1.0,
                source_diversity=1.0,
                evidence_density=1.0,
                total=next(coverage_totals),
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "check_convergence",
        lambda *args, **kwargs: next(convergence_decisions),
    )
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "rank_evidence",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "write_reading_path",
        _as_checkpoint(lambda *args, **kwargs: _render_result("reading_path", "# RP")),
    )
    monkeypatch.setattr(
        module,
        "write_backing_report",
        _as_checkpoint(
            lambda *args, **kwargs: _render_result("backing_report", "# BR")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", _as_checkpoint(lambda **kwargs: kwargs))

    module.research_flow.run(
        "brief",
        config=module.ResearchConfig.for_tier(module.Tier.DEEP).model_copy(
            update={
                "council_mode": True,
                "council_size": 2,
                "max_iterations": 2,
                "critique_enabled": False,
                "judge_enabled": False,
            }
        ),
    )

    assert council_submission_ids == [
        "council_0_0",
        "council_0_1",
        "council_1_0",
        "council_1_1",
    ]
