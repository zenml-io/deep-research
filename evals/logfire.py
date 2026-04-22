"""Optional Logfire bootstrap for eval runs.

Configures Logfire for the eval harness, always enabling pydantic-ai
instrumentation. Returns False when the Logfire SDK is not installed.
"""

from __future__ import annotations

import logging

from evals.settings import EvalSettings

logger = logging.getLogger(__name__)


def bootstrap_logfire(settings: EvalSettings) -> bool:
    """Configure Logfire once per process for eval runs.

    Returns ``True`` when the Logfire SDK is installed and configuration
    succeeded, ``False`` when Logfire is not installed.
    """
    try:
        import logfire
    except ImportError:
        logger.debug("Logfire SDK not installed; skipping eval bootstrap")
        return False

    scrubbing = logfire.ScrubbingOptions(extra_patterns=["bearer"])
    logfire.configure(
        send_to_logfire="if-token-present",
        service_name=settings.logfire_service_name,
        environment=settings.logfire_environment,
        scrubbing=scrubbing,
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()
    logfire.instrument_mcp()
    return True
