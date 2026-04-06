import importlib
import sys
import types


def _install_agent_stubs(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    def wrap(agent, tool_capture_config=None):
        wrapped = {
            "agent": agent,
            "tool_capture_config": tool_capture_config,
        }
        calls.append(wrapped)
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

    return calls


def _load_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_build_classifier_agent_returns_wrapped_agent(monkeypatch) -> None:
    calls = _install_agent_stubs(monkeypatch)
    module = _load_module("deep_research.agents.classifier")

    agent = module.build_classifier_agent(model_name="test")

    assert agent is not None
    assert calls[0]["agent"].kwargs["name"] == "classifier"


def test_agent_factories_build_expected_wrapped_agents(monkeypatch) -> None:
    calls = _install_agent_stubs(monkeypatch)

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

    expected = [
        ("classifier", "classifier", "RequestClassification", None),
        ("planner", "planner", "ResearchPlan", None),
        ("supervisor", "supervisor", None, {"mode": "full"}),
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
        assert agent.kwargs["system_prompt"]
        if result_type_name is None:
            assert "result_type" not in agent.kwargs
        else:
            assert agent.kwargs["result_type"].__name__ == result_type_name

    supervisor_agent = wrapped_agents[2]["agent"]
    assert len(supervisor_agent.kwargs["toolsets"]) == 1
    assert len(supervisor_agent.kwargs["tools"]) == 1
