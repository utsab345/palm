from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from google.genai import errors
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_STATUS_CODES
    return False


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
) -> T:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(1, max_attempts)),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    ):
        with attempt:
            return await operation()
    raise RuntimeError("retry loop exited without a result")
