"""Shared async httpx client factory with retry and exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configuration for retry behaviour with exponential backoff."""

    max_retries: int = 3
    backoff_base_seconds: float = 0.5
    backoff_factor: float = 2.0
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_base_seconds < 0.0:
            raise ValueError("backoff_base_seconds must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be at least 1.0")


DEFAULT_RETRY_POLICY = RetryPolicy()


def build_async_client(timeout: int = 20) -> httpx.AsyncClient:
    """Build an async httpx client with sensible defaults."""
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True)


def _backoff_delay(policy: RetryPolicy, attempt: int) -> float:
    """Calculate backoff delay for the given attempt number."""
    return policy.backoff_base_seconds * (policy.backoff_factor**attempt)


def _is_retryable_http_error(
    exc: Exception,
    policy: RetryPolicy,
) -> bool:
    """Check whether an exception is retryable under the given policy."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in policy.retryable_status_codes
    return isinstance(exc, httpx.RequestError)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    **kwargs: object,
) -> httpx.Response:
    """Execute an HTTP request with automatic retry on transient errors.

    Retries on 429/5xx status codes and connection errors with
    exponential backoff.
    """
    last_exc: Exception | None = None
    for attempt in range(retry_policy.max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= retry_policy.max_retries or not _is_retryable_http_error(
                exc, retry_policy
            ):
                raise
            delay = _backoff_delay(retry_policy, attempt)
            logger.debug(
                "Retrying %s %s (attempt %d/%d) after %.1fs: %s",
                method,
                url,
                attempt + 1,
                retry_policy.max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    # Defensive — should not be reached.
    if last_exc is not None:  # pragma: no cover
        raise last_exc
    raise RuntimeError(
        "retry loop exited without returning or raising"
    )  # pragma: no cover
