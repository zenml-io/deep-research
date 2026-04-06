import importlib
import sys
import types

from deep_research.models import SelectionGraph


def _as_checkpoint(func):
    """Add .submit() to a plain function so it behaves like a Kitaru checkpoint."""

    def submit(*args, after=None, id=None, **kwargs):
        result = func(*args, **kwargs)
        return types.SimpleNamespace(load=lambda: result)

    func.submit = submit
    return func


def _load_research_flow_module():
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


def test_run_iteration_uses_council_when_enabled(monkeypatch) -> None:
    module = _load_research_flow_module()
    called = {"council": False}

    result_obj = type(
        "SupervisorCheckpointResult",
        (),
        {
            "raw_results": [],
            "budget": type("Budget", (), {"estimated_cost_usd": 0.0})(),
        },
    )()

    def fake_council_generator(*args, **kwargs):
        called["council"] = True
        return result_obj

    monkeypatch.setattr(
        module, "run_council_generator", _as_checkpoint(fake_council_generator)
    )
    monkeypatch.setattr(
        module, "aggregate_council_results", lambda results: result_obj
    )

    module._run_iteration(
        plan=None,
        ledger=None,
        iteration=0,
        config=type("C", (), {"council_mode": True})(),
        council_models=["m1", "m2", "m3"],
    )

    assert called["council"] is True


def test_run_iteration_uses_supervisor_when_council_disabled(monkeypatch) -> None:
    module = _load_research_flow_module()
    called = {"supervisor": False}

    result_obj = type(
        "SupervisorCheckpointResult",
        (),
        {
            "raw_results": [],
            "budget": type("Budget", (), {"estimated_cost_usd": 0.0})(),
        },
    )()

    def fake_supervisor(*args, **kwargs):
        called["supervisor"] = True
        return result_obj

    monkeypatch.setattr(module, "run_supervisor", _as_checkpoint(fake_supervisor))

    result = module._run_iteration(
        plan=None,
        ledger=None,
        iteration=0,
        config=type("C", (), {"council_mode": False})(),
        council_models=["m1"],
    )

    assert called["supervisor"] is True
    assert result is result_obj


def test_run_iteration_returns_supervisor_result(monkeypatch) -> None:
    module = _load_research_flow_module()
    payload = type(
        "SupervisorCheckpointResult",
        (),
        {
            "raw_results": [],
            "budget": type("Budget", (), {"estimated_cost_usd": 0.0})(),
        },
    )()

    monkeypatch.setattr(
        module,
        "run_supervisor",
        _as_checkpoint(lambda *args, **kwargs: payload),
    )

    result = module._run_iteration(
        plan=None,
        ledger=None,
        iteration=0,
        config=type("C", (), {"council_mode": False})(),
        council_models=["m1"],
    )

    assert result is payload


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
        "_run_iteration",
        lambda *args, **kwargs: types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
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
        module, "merge_evidence", _as_checkpoint(lambda scored, ledger: ledger)
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
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
            lambda brief, classification, tier: types.SimpleNamespace(goal="goal")
        ),
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
        module, "merge_evidence", _as_checkpoint(lambda scored, ledger: ledger)
    )
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        _as_checkpoint(lambda ledger, plan: types.SimpleNamespace(total=0.1)),
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
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
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

    def fake_run_iteration(plan, ledger, iteration, config, council_models, **kwargs):
        observed_configs.append(config)
        return types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        )

    monkeypatch.setattr(module, "_run_iteration", fake_run_iteration)
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
        module, "merge_evidence", _as_checkpoint(lambda scored, ledger: ledger)
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        _as_checkpoint(lambda ledger, plan, config=None: SelectionGraph(items=[])),
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
    )
    plan = types.SimpleNamespace(goal="goal", subtopics=["topic"], key_questions=["q"])

    monkeypatch.setattr(module, "classify_request", track("classify", classification))
    monkeypatch.setattr(module, "wait", lambda **kwargs: True)
    monkeypatch.setattr(module, "build_plan", track("plan", plan))
    monkeypatch.setattr(
        module,
        "_run_iteration",
        lambda *args, **kwargs: types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
        ),
    )
    monkeypatch.setattr(module, "normalize_evidence", track("normalize", []))
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
        module, "merge_evidence", track("merge", module.EvidenceLedger())
    )
    monkeypatch.setattr(
        module,
        "evaluate_coverage",
        track("coverage", types.SimpleNamespace(total=1.0)),
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
        track("selection", SelectionGraph(items=[])),
    )
    monkeypatch.setattr(
        module,
        "render_reading_path",
        track("reading", types.SimpleNamespace(name="reading_path")),
    )
    monkeypatch.setattr(
        module,
        "render_backing_report",
        track("backing", types.SimpleNamespace(name="backing_report")),
    )
    monkeypatch.setattr(module, "assemble_package", lambda **kwargs: kwargs)

    module.research_flow.run("brief")

    assert called == {
        "classify",
        "plan",
        "normalize",
        "score",
        "merge",
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
    captured = {}

    def fake_render_reading(sel):
        captured["reading_selection"] = sel
        return types.SimpleNamespace(name="reading_path")

    def fake_render_backing(sel, ledger, plan):
        captured["backing_selection"] = sel
        return types.SimpleNamespace(name="backing_report")

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
        "_run_iteration",
        lambda *args, **kwargs: types.SimpleNamespace(
            raw_results=[],
            budget=types.SimpleNamespace(estimated_cost_usd=0.0),
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
        module, "merge_evidence", _as_checkpoint(lambda scored, ledger: ledger)
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
    monkeypatch.setattr(module, "log", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "build_selection_graph",
        _as_checkpoint(lambda ledger, plan, config=None: selection),
    )
    monkeypatch.setattr(
        module, "render_reading_path", _as_checkpoint(fake_render_reading)
    )
    monkeypatch.setattr(
        module, "render_backing_report", _as_checkpoint(fake_render_backing)
    )
    monkeypatch.setattr(module, "assemble_package", fake_assemble)

    module.research_flow.run("brief")

    assert captured["reading_selection"] is selection
    assert captured["backing_selection"] is selection
    assert captured["assemble_kwargs"]["selection_graph"] is selection
    assert len(captured["assemble_kwargs"]["renders"]) == 2
