"""
retry_utils.py
Shared retry helper for all Gemini API calls.

Usage:
    from retry_utils import call_with_gemini_retry
    result = call_with_gemini_retry(client.models.generate_content, model=..., contents=...)
"""
import re
import time
import random

from google.genai.errors import ClientError, ServerError

MAX_RETRIES = 5


def call_with_gemini_retry(fn, *args, **kwargs):
    """
    Call any Gemini API function with automatic retry logic.

    Handles three error classes:
      1. Daily quota (PerDay)  → raise immediately, no retry possible.
      2. Per-minute rate limit → wait using the API's suggested delay (or
                                 exponential backoff) and retry.
      3. Server / 503 error   → exponential backoff + jitter and retry.
      4. Anything else        → re-raise immediately.

    Args:
        fn:        The callable to invoke (e.g. client.models.generate_content).
        *args:     Positional arguments forwarded to fn.
        **kwargs:  Keyword arguments forwarded to fn.

    Returns:
        The return value of fn on success.

    Raises:
        The original exception if all retries are exhausted or a
        non-recoverable error is detected.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)

        except (ClientError, ServerError) as e:
            err_str = str(e)

            # ── 1. Rate limit (429 / RESOURCE_EXHAUSTED) ───────────────────
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:

                # Daily quota cannot recover until tomorrow — bail immediately.
                if "PerDay" in err_str or "per day" in err_str.lower() or "Daily" in err_str:
                    print("❌ Daily API quota exhausted. Cannot retry until tomorrow.")
                    raise

                if attempt < MAX_RETRIES - 1:
                    # Prefer the API's own "retry in Xs" hint when present.
                    match = re.search(r"retry in (\d+\.?\d*)s", err_str)
                    if match:
                        wait_time = float(match.group(1)) + random.uniform(0.5, 1.5)
                    else:
                        wait_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                    print(
                        f"⚠️  Google API rate limit hit "
                        f"({attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print("❌ Final attempt failed. Google rate limits persist.")
                    raise

            # ── 2. Server / high-demand error (503 / UNAVAILABLE) ──────────
            elif isinstance(e, ServerError) or "503" in err_str or "UNAVAILABLE" in err_str:
                if attempt < MAX_RETRIES - 1:
                    wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(
                        f"⚠️  Google Server Error/High Demand "
                        f"({attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print("❌ Final attempt failed. Google server error persists.")
                    raise

            # ── 3. Unrecognised error — do not retry ───────────────────────
            else:
                raise
