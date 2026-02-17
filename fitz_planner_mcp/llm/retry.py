# fitz_planner_mcp/llm/retry.py
"""Retry logic for Ollama API calls with exponential backoff."""

import logging

from ollama import ResponseError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def is_retryable(exception: BaseException) -> bool:
    """
    Returns True if the exception should be retried.

    Retryable conditions:
    - ConnectionError (server unavailable)
    - ResponseError with status in (408, 429, 500, 502, 503, 504)
      BUT NOT status 500 with "requires more system memory" (that's OOM, handled by fallback)
    """
    if isinstance(exception, ConnectionError):
        return True

    if isinstance(exception, ResponseError):
        status = exception.status_code
        # Retryable HTTP status codes (transient errors)
        retryable_statuses = {408, 429, 500, 502, 503, 504}

        if status not in retryable_statuses:
            return False

        # Special case: 500 with OOM message should NOT be retried (fallback handles it)
        if status == 500:
            error_msg = str(exception).lower()
            if "requires more system memory" in error_msg:
                return False

        return True

    return False


# Tenacity retry decorator for Ollama API calls
ollama_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception(is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
