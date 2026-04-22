import importlib
import sys
import types

from evals.settings import EvalSettings


class _BlockLogfireImport:
    """A ``sys.meta_path`` finder that raises ImportError for ``logfire``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "logfire" or fullname.startswith("logfire."):
            raise ImportError("logfire blocked for test")
        return None


def _reload_eval_logfire(monkeypatch, logfire_module):
    if logfire_module is None:
        monkeypatch.delitem(sys.modules, "logfire", raising=False)
        blocker = _BlockLogfireImport()
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    else:
        monkeypatch.setitem(sys.modules, "logfire", logfire_module)
    # The evals shim caches a process-wide bootstrap flag; reload to get a clean slate.
    sys.modules.pop("evals.logfire", None)
    return importlib.import_module("evals.logfire")


def test_eval_logfire_bootstrap_returns_false_when_sdk_missing(monkeypatch) -> None:
    module = _reload_eval_logfire(monkeypatch, None)

    assert module.bootstrap_logfire(EvalSettings()) is False


def test_eval_logfire_bootstrap_configures_sdk(monkeypatch) -> None:
    captured = {}

    class FakeScrubbingOptions:
        def __init__(self, *, extra_patterns):
            captured["extra_patterns"] = list(extra_patterns)

    fake_logfire = types.SimpleNamespace(
        ScrubbingOptions=FakeScrubbingOptions,
        configure=lambda **kwargs: captured.setdefault("configure", kwargs),
        instrument_pydantic_ai=lambda **kwargs: captured.setdefault("instrument", kwargs),
        instrument_httpx=lambda: captured.setdefault("httpx", True),
        instrument_mcp=lambda: captured.setdefault("mcp", True),
    )

    module = _reload_eval_logfire(monkeypatch, fake_logfire)

    settings = EvalSettings(
        enable_logfire=True,
        logfire_service_name="deep-research-evals",
        logfire_environment="staging",
    )
    assert module.bootstrap_logfire(settings) is True
    assert captured["configure"]["send_to_logfire"] == "if-token-present"
    assert captured["configure"]["service_name"] == "deep-research-evals"
    assert captured["configure"]["environment"] == "staging"
    # Evals should now instrument pydantic-ai (the previous implementation forgot).
    assert "instrument" in captured
    assert "bearer" in captured["extra_patterns"]
