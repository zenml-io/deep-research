from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, TypeVar

import httpx


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
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


def build_client(timeout: int = 20) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)


def is_retryable_http_error(exc: Exception) -> bool:
    return _is_retryable_http_error(exc, DEFAULT_RETRY_POLICY)


def call_with_retry(
    fn: Callable[[], T],
    *,
    retry_policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
) -> T:
    last_exc: Exception | None = None
    for attempt in range(retry_policy.max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - exercised via callers
            last_exc = exc
            if attempt >= retry_policy.max_retries or not is_retryable(exc):
                raise
            time.sleep(_backoff_delay(retry_policy, attempt))
    if last_exc is not None:  # pragma: no cover - defensive
        raise last_exc
    raise RuntimeError("retry loop exited without returning or raising")


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    **kwargs: object,
) -> httpx.Response:
    def _request() -> httpx.Response:
        request_method = getattr(client, method.lower())
        response = request_method(url, **kwargs)
        response.raise_for_status()
        return response

    return call_with_retry(
        _request,
        retry_policy=retry_policy,
        is_retryable=lambda exc: _is_retryable_http_error(exc, retry_policy),
    )


def _backoff_delay(retry_policy: RetryPolicy, attempt: int) -> float:
    return retry_policy.backoff_base_seconds * (
        retry_policy.backoff_factor**attempt
    )


def _is_retryable_http_error(
    exc: Exception,
    retry_policy: RetryPolicy,
) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code in retry_policy.retryable_status_codes
    return isinstance(exc, httpx.RequestError)
