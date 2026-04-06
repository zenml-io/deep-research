import importlib
import sys
import types


def _install_agent_stubs(monkeypatch):
    wrap_calls = []
    prompt_calls = []

    allowed_result_types = {
        "RequestClassification",
        "ResearchPlan",
        "SupervisorCheckpointResult",
        "RelevanceCheckpointResult",
        "SelectionGraph",
        "RenderPayload",
        "InvestigationPackage",
    }

    class FakeAgent:
        def __init__(self, model_name, **kwargs):
            allowed_keys = {
                "name",
                "result_type",
                "system_prompt",
                "toolsets",
                "tools",
            }
            unexpected_keys = set(kwargs) - allowed_keys
            assert not unexpected_keys, f"unexpected Agent kwargs: {unexpected_keys}"
            assert "name" in kwargs
            assert "system_prompt" in kwargs
            if "result_type" in kwargs:
                assert kwargs["result_type"].__name__ in allowed_result_types
            self.model_name = model_name
            self.kwargs = kwargs

    def wrap(agent, tool_capture_config=None):
        wrapped = {
            "agent": agent,
            "tool_capture_config": tool_capture_config,
        }
        wrap_calls.append(wrapped)
        return wrapped

    def load_prompt(name: str) -> str:
        prompt_calls.append(name)
        return f"prompt:{name}"

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
        types.SimpleNamespace(load_prompt=load_prompt),
    )

    return wrap_calls, prompt_calls


def _load_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_build_classifier_agent_returns_wrapped_agent(monkeypatch) -> None:
    wrap_calls, prompt_calls = _install_agent_stubs(monkeypatch)
    module = _load_module("deep_research.agents.classifier")

    agent = module.build_classifier_agent(model_name="test")

    assert agent is not None
    assert len(wrap_calls) == 1
    assert prompt_calls == ["classifier"]
    assert wrap_calls[0]["agent"].kwargs == {
        "name": "classifier",
        "result_type": module.RequestClassification,
        "system_prompt": "prompt:classifier",
    }


def test_agent_factories_build_expected_wrapped_agents(monkeypatch) -> None:
    wrap_calls, prompt_calls = _install_agent_stubs(monkeypatch)

    classifier = _load_module("deep_research.agents.classifier")
    planner = _load_module("deep_research.agents.planner")
    supervisor = _load_module("deep_research.agents.supervisor")
    relevance_scorer = _load_module("deep_research.agents.relevance_scorer")
    curator = _load_module("deep_research.agents.curator")
    writer = _load_module("deep_research.agents.writer")
    aggregator = _load_module("deep_research.agents.aggregator")

    wrapped_agents = [
        classifier.build_classifier_agent(model_name="test-model"),
        planner.build_planner_agent(model_name="test-model"),
        supervisor.build_supervisor_agent(
            model_name="test-model",
            toolsets=[object()],
            tools=[object()],
        ),
        relevance_scorer.build_relevance_scorer_agent(model_name="test-model"),
        curator.build_curator_agent(model_name="test-model"),
        writer.build_writer_agent(model_name="test-model"),
        aggregator.build_aggregator_agent(model_name="test-model"),
    ]

    assert len(wrapped_agents) == 7
    assert len(wrap_calls) == 7
    assert prompt_calls == [
        "classifier",
        "planner",
        "supervisor",
        "relevance_scorer",
        "curator",
        "writer",
        "aggregator",
    ]

    expected = [
        ("classifier", "classifier", "RequestClassification", None),
        ("planner", "planner", "ResearchPlan", None),
        (
            "supervisor",
            "supervisor",
            "SupervisorCheckpointResult",
            {"mode": "full"},
        ),
        (
            "relevance_scorer",
            "relevance_scorer",
            "RelevanceCheckpointResult",
            None,
        ),
        ("curator", "curator", "SelectionGraph", None),
        ("writer", "writer", "RenderPayload", None),
        ("aggregator", "aggregator", "InvestigationPackage", None),
    ]

    for wrapped, (
        prompt_name,
        agent_name,
        result_type_name,
        tool_capture_config,
    ) in zip(wrapped_agents, expected, strict=True):
        agent = wrapped["agent"]
        assert wrapped["tool_capture_config"] == tool_capture_config
        assert agent.model_name == "test-model"
        assert agent.kwargs["name"] == agent_name
        assert agent.kwargs["system_prompt"] == f"prompt:{prompt_name}"
        assert agent.kwargs["result_type"].__name__ == result_type_name

    supervisor_agent = wrapped_agents[2]["agent"]
    assert len(supervisor_agent.kwargs["toolsets"]) == 1
    assert len(supervisor_agent.kwargs["tools"]) == 1
