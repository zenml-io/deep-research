"""Optional Logfire bootstrap for eval runs.

Delegates to :func:`deep_research.observability.configure_logfire` so the
eval harness and the core app share one bootstrap path. Evals always enable
pydantic-ai instrumentation — the previous implementation forgot to do so.
"""

from __future__ import annotations

import logging

from deep_research.observability import configure_logfire
from evals.settings import EvalSettings

logger = logging.getLogger(__name__)


def bootstrap_logfire(settings: EvalSettings) -> bool:
    """Configure Logfire once per process for eval runs.

    Uses the shared helper from :mod:`deep_research.observability` with
    eval-specific service name and environment. Returns ``True`` when the
    Logfire SDK is installed and configuration succeeded.
    """
    return configure_logfire(
        service_name=settings.logfire_service_name,
        environment=settings.logfire_environment,
        instrument_pydantic_ai=True,
    )
