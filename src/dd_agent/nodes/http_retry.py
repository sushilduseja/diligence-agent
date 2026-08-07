"""Shared HTTP helper: GET with retry and exponential backoff on transport errors."""

import time

import httpx

RETRIES = 3
BASE_BACKOFF = 0.2


def get_with_retry(client, url, params, retries: int = RETRIES, backoff: float = BASE_BACKOFF) -> httpx.Response:
    """GET with exponential backoff. Only transport errors are retried; HTTP status codes are returned as-is."""
    last_error: httpx.HTTPError | None = None
    for attempt in range(retries):
        try:
            return client.get(url, params=params)
        except httpx.HTTPError as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_error
