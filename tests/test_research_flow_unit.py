import importlib
import sys
import types


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
        module, "build_selection_graph", lambda ledger, plan: {"items": []}
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
        module, "build_selection_graph", lambda ledger, plan: {"items": []}
    )
    monkeypatch.setattr(module, "assemble_package", lambda **kwargs: kwargs)

    package = module.research_flow.run("brief")

    assert package["run_summary"].stop_reason is module.StopReason.TIME_EXHAUSTED
