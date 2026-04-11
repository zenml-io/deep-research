from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_BOOTSTRAP_LOCK = Lock()
_BOOTSTRAP_ATTEMPTED = False
_LOGFIRE_ENABLED = False

_DEFAULT_SCRUB_PATTERNS: tuple[str, ...] = (
    r"bearer",
    r"access[._ -]?token",
    r"refresh[._ -]?token",
    r"set-cookie",
    r"x-api-key",
)

# Back-compat alias: some tests/imports reference this module-level list directly.
_EXTRA_SCRUB_PATTERNS = list(_DEFAULT_SCRUB_PATTERNS)

_LEVEL_MAP: dict[str, int] = {
    "info": logging.INFO,
    "warning": logging.WARNING,
}


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using a permissive truthy set."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logfire(
    *,
    service_name: str = "deep-research",
    service_version: str | None = None,
    environment: str | None = None,
    extra_scrub_patterns: Sequence[str] | None = None,
    instrument_pydantic_ai: bool = True,
) -> bool:
    """Bootstrap Logfire with the provided overrides.

    This is the shared helper used by both the core app and the eval harness.
    It is safe to call multiple times — subsequent calls short-circuit after
    the first successful (or unsuccessful) attempt.

    The ``include_content`` flag for pydantic-ai instrumentation is gated on
    the ``DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT`` environment variable
    (default false) so LLM payloads are not shipped to Logfire by accident.
    """
    global _BOOTSTRAP_ATTEMPTED, _LOGFIRE_ENABLED
    if _BOOTSTRAP_ATTEMPTED:
        return _LOGFIRE_ENABLED

    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_ATTEMPTED:
            return _LOGFIRE_ENABLED
        _BOOTSTRAP_ATTEMPTED = True

        try:
            import logfire
        except ImportError:
            logger.debug("logfire is not installed; observability bootstrap skipped")
            return False

        scrub_patterns = list(_DEFAULT_SCRUB_PATTERNS)
        if extra_scrub_patterns:
            scrub_patterns.extend(extra_scrub_patterns)

        resolved_environment = environment or os.getenv("DEEP_RESEARCH_ENV", "dev")
        resolved_version = (
            service_version if service_version is not None else _package_version()
        )

        configure_kwargs: dict[str, Any] = {
            "send_to_logfire": "if-token-present",
            "service_name": service_name,
            "environment": resolved_environment,
            "scrubbing": logfire.ScrubbingOptions(extra_patterns=scrub_patterns),
        }
        if resolved_version is not None:
            configure_kwargs["service_version"] = resolved_version

        try:
            logfire.configure(**configure_kwargs)
        except Exception:
            logger.warning(
                "logfire bootstrap failed; continuing without Logfire",
                exc_info=True,
            )
            return False

        if instrument_pydantic_ai:
            include_content = _env_flag(
                "DEEP_RESEARCH_LOGFIRE_INCLUDE_CONTENT", default=False
            )
            try:
                logfire.instrument_pydantic_ai(include_content=include_content)
            except Exception:
                logger.warning(
                    "logfire.instrument_pydantic_ai failed; continuing",
                    exc_info=True,
                )

        # httpx and mcp instrumentation are optional: the underlying SDK calls
        # may raise ImportError if the target library is missing. Guard each
        # one independently so a single missing dep cannot disable the rest.
        try:
            logfire.instrument_httpx()
        except ImportError:
            logger.debug("httpx not installed; skipping logfire.instrument_httpx")
        except Exception:
            logger.debug("logfire.instrument_httpx failed", exc_info=True)

        try:
            logfire.instrument_mcp()
        except ImportError:
            logger.debug("mcp not installed; skipping logfire.instrument_mcp")
        except Exception:
            logger.debug("logfire.instrument_mcp failed", exc_info=True)

        _LOGFIRE_ENABLED = True
        return True


def bootstrap_logfire() -> bool:
    """Configure Logfire once per process using core-app defaults.

    Kept as a zero-arg public function for backwards compatibility; delegates
    to :func:`configure_logfire` with sane defaults.
    """
    return configure_logfire()


def span(message: str, **attributes: object):
    """Return a Logfire span when available, otherwise a no-op context manager."""
    logfire = _logfire_module()
    if logfire is None:
        return nullcontext()
    return logfire.span(message, **attributes)


def info(message: str, **attributes: object) -> None:
    """Emit a structured informational observability event."""
    _emit("info", message, **attributes)


def warning(message: str, **attributes: object) -> None:
    """Emit a structured warning event to Logfire when configured."""
    _emit("warning", message, **attributes)


def metric(name: str, value: float, **attributes: object) -> None:
    """Record a numeric metric to Logfire when configured."""
    _emit("info", f"metric:{name}", metric_name=name, metric_value=value, **attributes)


def _emit(level: str, message: str, **attributes: object) -> None:
    logfire = _logfire_module()
    if logfire is not None:
        getattr(logfire, level)(message, **attributes)
        return
    level_int = _LEVEL_MAP.get(level, logging.INFO)
    logger.log(level_int, message, extra=attributes)


def _logfire_module():
    if not bootstrap_logfire():
        return None
    try:
        import logfire
    except ImportError:  # pragma: no cover - defensive after bootstrap race/failure
        return None
    return logfire


def _package_version() -> str | None:
    try:
        return version("deep-research")
    except PackageNotFoundError:
        return None
