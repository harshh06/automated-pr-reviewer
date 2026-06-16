"""
test_retry_utils.py
Unit tests for the shared Gemini API retry helper.

Run with:
    backend/venv/bin/python -m pytest backend/test_retry_utils.py -v
OR:
    backend/venv/bin/python backend/test_retry_utils.py
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call

# Make sure the backend package is on the path when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from retry_utils import call_with_gemini_retry, MAX_RETRIES
from google.genai.errors import ClientError, ServerError


# ── helpers ──────────────────────────────────────────────────────────────────

def make_client_error(msg: str) -> ClientError:
    """Construct a ClientError whose str() contains the given message."""
    err = ClientError.__new__(ClientError)
    err.args = (msg,)
    return err


def make_server_error(msg: str) -> ServerError:
    err = ServerError.__new__(ServerError)
    err.args = (msg,)
    return err


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCallWithGeminiRetry(unittest.TestCase):

    # ── success path ─────────────────────────────────────────────────────────

    def test_success_on_first_attempt(self):
        """Should return the function's value immediately on success."""
        mock_fn = MagicMock(return_value="ok")
        result = call_with_gemini_retry(mock_fn, "arg1", key="val")
        self.assertEqual(result, "ok")
        mock_fn.assert_called_once_with("arg1", key="val")

    # ── daily quota ──────────────────────────────────────────────────────────

    @patch("retry_utils.time.sleep")
    def test_daily_quota_raises_immediately(self, mock_sleep):
        """A PerDay error must raise without any retry or sleep."""
        daily_err = make_client_error(
            "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        mock_fn = MagicMock(side_effect=daily_err)

        with self.assertRaises(ClientError):
            call_with_gemini_retry(mock_fn)

        mock_fn.assert_called_once()          # only one attempt
        mock_sleep.assert_not_called()        # no sleeping

    @patch("retry_utils.time.sleep")
    def test_daily_quota_keyword_variations(self, mock_sleep):
        """All three daily-quota keyword patterns must trigger fast-fail."""
        patterns = [
            "429 quota: PerDay limit exceeded",
            "429 RESOURCE_EXHAUSTED: per day limit",
            "429 Daily limit reached",
        ]
        for msg in patterns:
            with self.subTest(msg=msg):
                mock_fn = MagicMock(side_effect=make_client_error(msg))
                with self.assertRaises(ClientError):
                    call_with_gemini_retry(mock_fn)
                mock_sleep.assert_not_called()

    # ── per-minute rate limit ────────────────────────────────────────────────

    @patch("retry_utils.random.uniform", return_value=1.0)
    @patch("retry_utils.time.sleep")
    def test_rate_limit_retries_and_succeeds(self, mock_sleep, _mock_rand):
        """Should retry after a per-minute 429 and return successfully."""
        rate_err = make_client_error("429 RESOURCE_EXHAUSTED: quota per minute")
        mock_fn = MagicMock(side_effect=[rate_err, "success"])

        result = call_with_gemini_retry(mock_fn)

        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 2)
        mock_sleep.assert_called_once()       # slept once between attempts

    @patch("retry_utils.random.uniform", return_value=1.0)
    @patch("retry_utils.time.sleep")
    def test_rate_limit_uses_retry_hint_from_api(self, mock_sleep, _mock_rand):
        """Should use the 'retry in Xs' hint from the error body when present."""
        rate_err = make_client_error(
            "429 RESOURCE_EXHAUSTED: quota per minute. retry in 5.0s"
        )
        mock_fn = MagicMock(side_effect=[rate_err, "ok"])

        call_with_gemini_retry(mock_fn)

        # sleep arg should be 5.0 (hint) + 1.0 (mocked uniform jitter)
        mock_sleep.assert_called_once_with(6.0)

    @patch("retry_utils.time.sleep")
    def test_rate_limit_raises_after_max_retries(self, _mock_sleep):
        """Should raise after MAX_RETRIES exhausted on per-minute limits."""
        rate_err = make_client_error("429 RESOURCE_EXHAUSTED: quota per minute")
        mock_fn = MagicMock(side_effect=rate_err)

        with self.assertRaises(ClientError):
            call_with_gemini_retry(mock_fn)

        self.assertEqual(mock_fn.call_count, MAX_RETRIES)

    # ── server / 503 errors ──────────────────────────────────────────────────

    @patch("retry_utils.random.uniform", return_value=0.5)
    @patch("retry_utils.time.sleep")
    def test_server_error_retries_with_backoff(self, mock_sleep, _mock_rand):
        """Should retry on 503 ServerError with exponential backoff."""
        server_err = make_server_error("503 UNAVAILABLE: model overloaded")
        mock_fn = MagicMock(side_effect=[server_err, server_err, "ok"])

        result = call_with_gemini_retry(mock_fn)

        self.assertEqual(result, "ok")
        self.assertEqual(mock_fn.call_count, 3)
        # Backoff: attempt 0 → 2^0+0.5=1.5, attempt 1 → 2^1+0.5=2.5
        mock_sleep.assert_has_calls([call(1.5), call(2.5)])

    @patch("retry_utils.time.sleep")
    def test_server_error_raises_after_max_retries(self, _mock_sleep):
        """Should raise after MAX_RETRIES exhausted on server errors."""
        mock_fn = MagicMock(side_effect=make_server_error("503 UNAVAILABLE"))

        with self.assertRaises(ServerError):
            call_with_gemini_retry(mock_fn)

        self.assertEqual(mock_fn.call_count, MAX_RETRIES)

    # ── unrelated errors ─────────────────────────────────────────────────────

    @patch("retry_utils.time.sleep")
    def test_unknown_error_raises_immediately(self, mock_sleep):
        """Non-rate-limit, non-server errors must propagate without retry."""
        unknown_err = make_client_error("400 INVALID_ARGUMENT: bad request")
        mock_fn = MagicMock(side_effect=unknown_err)

        with self.assertRaises(ClientError):
            call_with_gemini_retry(mock_fn)

        mock_fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("retry_utils.time.sleep")
    def test_non_api_exception_propagates_immediately(self, mock_sleep):
        """A plain Python exception should never be retried."""
        mock_fn = MagicMock(side_effect=ValueError("something wrong"))

        with self.assertRaises(ValueError):
            call_with_gemini_retry(mock_fn)

        mock_fn.assert_called_once()
        mock_sleep.assert_not_called()

    # ── success after transient failures ─────────────────────────────────────

    @patch("retry_utils.time.sleep")
    def test_succeeds_on_last_allowed_attempt(self, _mock_sleep):
        """Should succeed on the final attempt before giving up."""
        err = make_server_error("503 UNAVAILABLE")
        # Fail MAX_RETRIES-1 times, then succeed
        mock_fn = MagicMock(
            side_effect=[err] * (MAX_RETRIES - 1) + ["final_ok"]
        )
        result = call_with_gemini_retry(mock_fn)
        self.assertEqual(result, "final_ok")
        self.assertEqual(mock_fn.call_count, MAX_RETRIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
