"""
Shared HTTP utility with retry, backoff, and rate-limit helpers.
Reused from drug_repurposing v1 (identical).
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger


class APIError(Exception):
    """Raised when an API call fails after retries."""


def get_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 3,
) -> Any:
    """GET request returning parsed JSON, with exponential backoff."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, params=params, headers=headers)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited on {url}, waiting {wait}s")
                time.sleep(wait)
            elif attempt == retries - 1:
                raise APIError(f"HTTP {e.response.status_code} on {url}") from e
        except Exception as e:
            if attempt == retries - 1:
                raise APIError(f"Request failed: {e}") from e
            time.sleep(1)
    raise APIError(f"All {retries} retries failed for {url}")


def post_json(
    url: str,
    json: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 3,
) -> Any:
    """POST request returning parsed JSON."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=json, headers=headers)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited on {url}, waiting {wait}s")
                time.sleep(wait)
            elif attempt == retries - 1:
                raise APIError(f"HTTP {e.response.status_code} on {url}") from e
        except Exception as e:
            if attempt == retries - 1:
                raise APIError(f"POST failed: {e}") from e
            time.sleep(1)
    raise APIError(f"All {retries} retries failed for {url}")


def get_text(url: str, params: dict | None = None, timeout: int = 30) -> str:
    """GET request returning raw text."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.text
    except httpx.HTTPStatusError as e:
        raise APIError(f"HTTP {e.response.status_code} on {url}") from e
    except Exception as e:
        raise APIError(f"Request failed: {e}") from e
