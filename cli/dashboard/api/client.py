"""
api/client.py — Base async HTTP client.

Security posture:
- HTTPS enforced in production (warn on http:// for non-localhost)
- JWT token sent only in Authorization header, never in URLs
- connect/read timeouts to avoid hanging the TUI
- Token value is never logged
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)
_MAX_RETRIES = 2


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


class BaseClient:
    """Thin async wrapper around httpx with auth and retry logic."""

    def __init__(self, base_url: str, headers: dict[str, str] | None = None) -> None:
        if base_url.startswith("http://") and not _is_localhost(base_url):
            log.warning(
                "Connecting to %s over plain HTTP — use HTTPS in production",
                base_url,
            )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        return await self._request("POST", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> Any:
        return await self._request("DELETE", path, **kwargs)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1 + _MAX_RETRIES):
            try:
                resp = await self._client.request(method, path, **kwargs)
                if resp.status_code >= 400:
                    _raise(resp)
                return resp.json()
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    log.debug("Retry %d for %s %s", attempt + 1, method, path)
        raise ApiError(0, f"Connection failed: {last_exc}")

    async def close(self) -> None:
        await self._client.aclose()


def _raise(resp: httpx.Response) -> None:
    try:
        body = resp.json()
        msg = (
            body.get("message")
            or body.get("error")
            or body.get("title")
            or str(body)
        )
    except Exception:
        msg = resp.text[:200]
    raise ApiError(resp.status_code, msg)


def _is_localhost(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "::1"))
