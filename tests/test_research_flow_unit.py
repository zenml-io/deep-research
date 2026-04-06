import importlib
import sys
import types

from deep_research.models import SelectionGraph


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


def test_run_iteration_uses_council_when_enabled(monkeypatch) -> None:
    module = _load_research_flow_module()
    called = {"council": False}

    def fake_council(*args, **kwargs):
        called["council"] = True
        return type(
            "SupervisorCheckpointResult",
            (),
            {
                "raw_results": [],
                "budget": type("Budget", (), {"estimated_cost_usd": 0.0})(),
            },
        )()

    monkeypatch.setattr(module, "_run_council_iteration", fake_council)

    module._run_iteration(
        plan=None,
        ledger=None,
        iteration=0,
        config=type("C", (), {"council_mode": True, "council_size": 3})(),
        council_models=["m1", "m2", "m3"],
    )

    assert called["council"] is True


def test_run_iteration_uses_supervisor_when_council_disabled(monkeypatch) -> None:
    module = _load_research_flow_module()
    called = {"supervisor": False}

    def fake_supervisor(*args, **kwargs):
        called["supervisor"] = True
        return type(
            "SupervisorCheckpointResult",
            (),
            {
                "raw_results": [],
                "budget": type("Budget", (), {"estimated_cost_usd": 0.0})(),
            },
        )()

    monkeypatch.setattr(module, "run_supervisor", fake_supervisor)

    module._run_iteration(
        plan=None,
        ledger=None,
        iteration=0,
        config=type("C", (), {"council_mode": False})(),
        council_models=["m1"],
    )

    assert called["supervisor"] is True


def test_run_iteration_loads_supervisor_artifact_when_needed(monkeypatch) -> None:
    module = _load_research_flow_module()
    loaded = {"count": 0}
    payload = type(
        "SupervisorCheckpointResult",
        (),
        {
            "raw_results": [],
            "budget": type("Budget", (), {"estimated_cost_usd": 0.0})(),
        },
    )()

    class FakeArtifact:
        def load(self):
            loaded["count"] += 1
            return payload

    monkeypatch.setattr(
        module, "run_supervisor", lambda *args, **kwargs: FakeArtifact()
    )

    result = module._run_iteration(
        plan=None,
        ledger=None,
        iteration=0,
        config=type("C", (), {"council_mode": False})(),
        council_models=["m1"],
    )

    assert result is payload
    assert loaded["count"] == 1


def test_resolve_council_models_uses_configured_supervisor_model() -> None:
    module = _load_research_flow_module()
    config = type(
        "C",
        (),
        {"council_size": 3, "supervisor_model": "openai/gpt-4o-mini"},
    )()

    assert module._resolve_council_models(config) == [
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini",
    ]


def test_research_flow_recomputes_tier_after_clarification(monkeypatch) -> None:
    module = _load_research_flow_module()
    classifications = [
        types.SimpleNamespace(
            recommended_tier=module.Tier.STANDARD,
            needs_clarification=True,
            clarification_question="Need more detail?",
        ),
        types.SimpleNamespace(
            recommended_tier=module.Tier.DEEP,
            needs_clarification=False,
            clarification_question=None,
        ),
    ]
    build_plan_tiers = []

    monkeypatch.setattr(
        module,
        "classify_request",
        lambda brief, config: classifications.pop(0),
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
        lambda brief, classification, tier: (
            build_plan_tiers.append(tier)
            or types.SimpleNamespace(goal=f"plan for {brief}")
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        lambda ledger, plan: SelectionGraph(items=[]),
    )
    monkeypatch.setattr(module, "assemble_package", lambda **kwargs: kwargs)

    package = module.research_flow.run("initial brief")

    assert build_plan_tiers == [module.Tier.DEEP]
    assert package["run_summary"].tier is module.Tier.DEEP
    assert package["run_summary"].brief == "clarified brief"


def test_research_flow_passes_elapsed_seconds_to_convergence(monkeypatch) -> None:
    module = _load_research_flow_module()
    monotonic_values = iter([10.0, 16.2])

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
        lambda brief, classification, tier: types.SimpleNamespace(goal="goal"),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
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
        lambda ledger, plan: types.SimpleNamespace(total=0.1),
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
        "build_selection_graph",
        lambda ledger, plan: SelectionGraph(items=[]),
    )
    monkeypatch.setattr(module, "assemble_package", lambda **kwargs: kwargs)

    package = module.research_flow.run("brief")

    assert package["run_summary"].stop_reason is module.StopReason.TIME_EXHAUSTED


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
        ),
        types.SimpleNamespace(
            recommended_tier=module.Tier.DEEP,
            needs_clarification=False,
            clarification_question=None,
        ),
    ]
    observed_configs = []

    monkeypatch.setattr(
        module,
        "classify_request",
        lambda brief, config: classifications.pop(0),
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
        lambda brief, classification, tier: types.SimpleNamespace(goal="goal"),
    )

    def fake_run_iteration(plan, ledger, iteration, config, council_models):
        observed_configs.append(config)
        return types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        )

    monkeypatch.setattr(module, "_run_iteration", fake_run_iteration)
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        lambda ledger, plan: SelectionGraph(items=[]),
    )
    monkeypatch.setattr(module, "assemble_package", lambda **kwargs: kwargs)

    module.research_flow.run("initial brief", config=incoming_config)

    config = observed_configs[0]
    assert config.tier is module.Tier.DEEP
    assert config.council_mode is True
    assert config.council_size == 5
    assert config.supervisor_model == "custom/supervisor"
    assert config.planner_model == "custom/planner"
    assert config.cost_budget_usd == 9.5
    assert config.time_box_seconds == 321


def test_research_flow_loads_intermediate_checkpoint_artifacts_in_flow_scope(
    monkeypatch,
) -> None:
    module = _load_research_flow_module()

    classification = types.SimpleNamespace(
        recommended_tier=module.Tier.STANDARD,
        needs_clarification=False,
        clarification_question=None,
    )
    plan = types.SimpleNamespace(goal="goal", subtopics=["topic"], key_questions=["q"])
    normalized_candidates = [types.SimpleNamespace(key="k")]
    relevance_result = types.SimpleNamespace(
        candidates=[],
        budget=types.SimpleNamespace(estimated_cost_usd=0.0),
    )
    selection = SelectionGraph(items=[])

    loaded = {
        "classification": 0,
        "plan": 0,
        "normalize": 0,
        "score": 0,
        "merge": 0,
        "coverage": 0,
        "selection": 0,
        "reading": 0,
        "backing": 0,
    }

    class FakeArtifact:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def load(self):
            loaded[self.key] += 1
            return self.value

    monkeypatch.setattr(
        module,
        "classify_request",
        lambda brief, config: FakeArtifact("classification", classification),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(
        module,
        "build_plan",
        lambda brief, classification_value, tier: FakeArtifact("plan", plan),
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
        lambda raw_results: FakeArtifact("normalize", normalized_candidates),
    )
    monkeypatch.setattr(
        module,
        "score_relevance",
        lambda candidates, plan_value, config: FakeArtifact("score", relevance_result),
    )
    monkeypatch.setattr(
        module,
        "merge_evidence",
        lambda scored, ledger: FakeArtifact("merge", ledger),
    )
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        lambda ledger, plan_value: FakeArtifact(
            "coverage", types.SimpleNamespace(total=1.0)
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
        "build_selection_graph",
        lambda ledger, plan_value: FakeArtifact("selection", selection),
    )
    monkeypatch.setattr(
        module,
        "render_reading_path",
        lambda selection_value: FakeArtifact(
            "reading", types.SimpleNamespace(name="reading_path")
        ),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        lambda selection_value, ledger, plan_value: FakeArtifact(
            "backing", types.SimpleNamespace(name="backing_report")
        ),
    )
    monkeypatch.setattr(module, "assemble_package", lambda **kwargs: kwargs)

    package = module.research_flow.run("brief")

    assert package["research_plan"].value is plan
    assert package["selection_graph"].value is selection
    assert [render.key for render in package["renders"]] == ["reading", "backing"]
    assert loaded == {
        "classification": 1,
        "plan": 1,
        "normalize": 1,
        "score": 1,
        "merge": 1,
        "coverage": 1,
        "selection": 0,
        "reading": 0,
        "backing": 0,
    }


def test_research_flow_preserves_terminal_checkpoint_dependencies(monkeypatch) -> None:
    module = _load_research_flow_module()
    package_value = {"package": True}
    selection = SelectionGraph(items=[])
    loaded = {"selection": 0, "reading": 0, "backing": 0, "package": 0}

    class FakeArtifact:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def load(self):
            loaded[self.key] += 1
            return self.value

    selection_artifact = FakeArtifact("selection", selection)
    reading_artifact = FakeArtifact(
        "reading", types.SimpleNamespace(name="reading_path")
    )
    backing_artifact = FakeArtifact(
        "backing", types.SimpleNamespace(name="backing_report")
    )
    package_artifact = FakeArtifact("package", package_value)
    captured = {}

    def fake_render_reading_path(selection_value):
        captured["reading_input"] = selection_value
        return reading_artifact

    def fake_render_backing_report(selection_value, ledger, plan):
        captured["backing_input"] = selection_value
        return backing_artifact

    def fake_assemble_package(**kwargs):
        captured["assemble_kwargs"] = kwargs
        return package_artifact

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
        lambda brief, classification, tier: types.SimpleNamespace(goal="goal"),
    )
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        lambda ledger, plan: selection_artifact,
    )
    monkeypatch.setattr(
        module,
        "render_reading_path",
        fake_render_reading_path,
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        fake_render_backing_report,
    )
    monkeypatch.setattr(
        module,
        "assemble_package",
        fake_assemble_package,
    )

    package = module.research_flow.run("brief")

    assert package is package_artifact
    assert captured["reading_input"] is selection_artifact
    assert captured["backing_input"] is selection_artifact
    assert captured["assemble_kwargs"]["selection_graph"] is selection_artifact
    assert captured["assemble_kwargs"]["renders"] == [
        reading_artifact,
        backing_artifact,
    ]
    assert loaded == {"selection": 0, "reading": 0, "backing": 0, "package": 0}
