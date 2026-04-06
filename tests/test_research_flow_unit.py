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
