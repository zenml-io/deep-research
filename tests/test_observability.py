import importlib
import sys
import types


class _BlockLogfireImport:
    """A ``sys.meta_path`` finder that raises ImportError for ``logfire``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "logfire" or fullname.startswith("logfire."):
            raise ImportError("logfire blocked for test")
        return None


def _reload_observability_module(monkeypatch, logfire_module):
    if logfire_module is None:
        monkeypatch.delitem(sys.modules, "logfire", raising=False)
        blocker = _BlockLogfireImport()
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    else:
        monkeypatch.setitem(sys.modules, "logfire", logfire_module)
    sys.modules.pop("deep_research.observability", None)
    return importlib.import_module("deep_research.observability")


def test_bootstrap_logfire_returns_false_when_sdk_missing(monkeypatch) -> None:
    module = _reload_observability_module(monkeypatch, None)

    assert module.bootstrap_logfire() is False


def test_bootstrap_logfire_configures_sdk_and_pydantic_ai(monkeypatch) -> None:
    captured = {}

    class FakeScrubbingOptions:
        def __init__(self, *, extra_patterns):
            captured["extra_patterns"] = list(extra_patterns)

    def _fake_instrument_httpx():
        captured["instrument_httpx_called"] = True

    def _fake_instrument_mcp():
        captured["instrument_mcp_called"] = True

    fake_logfire = types.SimpleNamespace(
        ScrubbingOptions=FakeScrubbingOptions,
        configure=lambda **kwargs: captured.setdefault("configure", kwargs),
        instrument_pydantic_ai=lambda **kwargs: captured.setdefault(
            "instrument", kwargs
        ),
        instrument_httpx=_fake_instrument_httpx,
        instrument_mcp=_fake_instrument_mcp,
        span=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )

    monkeypatch.delenv("DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT", raising=False)
    monkeypatch.setenv("DEEP_RESEARCH_ENV", "dev")
    module = _reload_observability_module(monkeypatch, fake_logfire)
    monkeypatch.setattr(module, "_package_version", lambda: "0.1.0")

    assert module.bootstrap_logfire() is True
    assert captured["configure"]["send_to_logfire"] == "if-token-present"
    assert captured["configure"]["service_name"] == "deep-research"
    assert captured["configure"]["service_version"] == "0.1.0"
    assert captured["configure"]["environment"] == "dev"
    # include_content is gated off by default for safety.
    assert captured["instrument"] == {"include_content": False}
    assert captured.get("instrument_httpx_called") is True
    assert captured.get("instrument_mcp_called") is True
    assert "bearer" in captured["extra_patterns"]


def test_metric_emits_info_with_metric_attributes(monkeypatch) -> None:
    """``metric()`` should delegate to logfire.info with metric_name/metric_value."""
    emitted: list[tuple] = []

    fake_logfire = types.SimpleNamespace(
        ScrubbingOptions=lambda *, extra_patterns: None,
        configure=lambda **kwargs: None,
        instrument_pydantic_ai=lambda **kwargs: None,
        instrument_httpx=lambda: None,
        instrument_mcp=lambda: None,
        span=lambda *args, **kwargs: None,
        info=lambda msg, **kw: emitted.append((msg, kw)),
        warning=lambda *args, **kwargs: None,
    )

    monkeypatch.delenv("DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT", raising=False)
    module = _reload_observability_module(monkeypatch, fake_logfire)

    module.metric("iteration_coverage", 0.85, iteration=2)

    assert len(emitted) == 1
    msg, attrs = emitted[0]
    assert msg == "metric:iteration_coverage"
    assert attrs["metric_name"] == "iteration_coverage"
    assert attrs["metric_value"] == 0.85
    assert attrs["iteration"] == 2


def test_metric_falls_back_to_logging_when_logfire_missing(monkeypatch) -> None:
    """``metric()`` should log via stdlib when logfire is unavailable."""
    module = _reload_observability_module(monkeypatch, None)

    # Should not raise; falls back to stdlib logger
    module.metric("test_metric", 42.0, extra="val")
