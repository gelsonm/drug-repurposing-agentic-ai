"""
Shared HTTP client utilities with retry logic and rate limiting.
All API clients should use these base functions.
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class APIError(Exception):
    """Raised when an API call fails after all retries."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _is_retryable(exc: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError))


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=False,
)
def _get_with_retry(client: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
    """Perform a GET request with exponential backoff retry."""
    response = client.get(url, **kwargs)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 5))
        logger.warning(f"Rate limited by {url}. Waiting {retry_after}s...")
        time.sleep(retry_after)
        response.raise_for_status()
    response.raise_for_status()
    return response


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> Any:
    """
    Perform a GET request and return parsed JSON.
    Handles retries, timeouts, and rate limiting automatically.
    """
    default_headers = {
        "Accept": "application/json",
        "User-Agent": "DrugRepurposingAgents/1.0 (research; contact: hackathon@example.com)",
    }
    if headers:
        default_headers.update(headers)

    with httpx.Client(timeout=timeout) as client:
        try:
            response = _get_with_retry(client, url, params=params, headers=default_headers)
            return response.json()
        except RetryError as e:
            raise APIError(f"Request failed after retries: {url}") from e
        except httpx.HTTPStatusError as e:
            raise APIError(
                f"HTTP {e.response.status_code} for {url}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise APIError(f"Unexpected error for {url}: {e}") from e


def get_text(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    """Perform a GET request and return raw text."""
    default_headers = {
        "User-Agent": "DrugRepurposingAgents/1.0 (research; contact: hackathon@example.com)",
    }
    if headers:
        default_headers.update(headers)

    with httpx.Client(timeout=timeout) as client:
        try:
            response = _get_with_retry(client, url, params=params, headers=default_headers)
            return response.text
        except RetryError as e:
            raise APIError(f"Request failed after retries: {url}") from e
        except httpx.HTTPStatusError as e:
            raise APIError(
                f"HTTP {e.response.status_code} for {url}",
                status_code=e.response.status_code,
            ) from e
