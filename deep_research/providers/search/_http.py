from __future__ import annotations

import httpx


def build_client(timeout: int = 20) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)
