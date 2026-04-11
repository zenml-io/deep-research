import importlib
import sys
import types
from contextlib import contextmanager


@contextmanager
def _preserve_modules(*names: str):
    """Temporarily save and restore selected modules while operator stubs are injected.

    The helper prevents test-time stub installation from leaking into the wider process
    by restoring each named module to its original object after the import completes.
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
    """Import the flow module under minimal stubs needed for operator-name contract tests.

    The test only needs checkpoint, flow, and adapter symbols to exist, so this helper
    loads the module under lightweight stand-ins instead of the full runtime stack.
    """

    def checkpoint(func=None, *, type=None, retries=0, runtime=None):
        """Stub matching kitaru's overloaded ``@checkpoint`` / ``@checkpoint(...)``."""

        def _wrap(target):
            target._checkpoint_type = type
            target.submit = lambda *args, **kwargs: types.SimpleNamespace(
                load=lambda: target(*args, **kwargs)
            )
            return target

        if func is not None:
            return _wrap(func)
        return _wrap

    def flow(func):
        def run(*args, **kwargs):
            return func(*args, **kwargs)

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
        return importlib.import_module("deep_research.flow.research_flow")


def test_stable_operator_names_are_exported() -> None:
    module = _load_research_flow_module()

    assert module.APPROVE_PLAN_WAIT_NAME == "approve_plan"
    assert module.CLARIFY_BRIEF_WAIT_NAME == "clarify_brief"
    assert module.CLASSIFY_CHECKPOINT_NAME == "classify_request"
    assert module.PLAN_CHECKPOINT_NAME == "build_plan"
    assert module.SUPERVISOR_CHECKPOINT_NAME == "run_supervisor"
    assert module.COUNCIL_GENERATOR_CHECKPOINT_NAME == "run_council_generator"
