"""Which download failures are retried, and which must not be.

A presigned URL that resets mid-stream is transient and used to cost the clip
its transcript. An expired signature is not transient: retrying a 403 only
delays the error the caller has to see.

    python tests/test_download_retry.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

import asr_core  # noqa: E402

asr_core.DOWNLOAD_TRIES = 3
asr_core._assert_public_host = lambda url: None
asr_core.time.sleep = lambda s: None  # no real backoff in a test


def http_error(status):
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status} error", response=resp)


def attempts(failures, dest="/dev/null"):
    """Run _download against a sequence of outcomes; return how many were used."""
    calls = {"n": 0}

    def fake_once(url, d):
        calls["n"] += 1
        outcome = failures[min(calls["n"] - 1, len(failures) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    asr_core._download_once = fake_once
    try:
        written = asr_core._download("https://example.test/a.wav", dest)
    except Exception as e:
        return calls["n"], e
    return calls["n"], written


def test_reset_then_success():
    n, result = attempts([requests.ConnectionError("reset by peer"), 1234])
    assert n == 2, f"expected 2 attempts, got {n}"
    assert result == 1234


def test_timeout_is_retried_to_the_limit():
    n, result = attempts([requests.Timeout("read timed out")])
    assert n == 3, f"expected 3 attempts, got {n}"
    assert isinstance(result, RuntimeError) and "after 3 attempts" in str(result)


def test_5xx_and_429_are_retried():
    for status in (500, 503, 429):
        n, _ = attempts([http_error(status)])
        assert n == 3, f"{status} should be retried, got {n} attempt(s)"


def test_403_is_not_retried():
    # An expired or revoked presigned signature. Every attempt returns the same.
    n, result = attempts([http_error(403)])
    assert n == 1, f"403 must not be retried, got {n} attempts"
    assert isinstance(result, requests.HTTPError)


def test_404_is_not_retried():
    n, result = attempts([http_error(404)])
    assert n == 1, f"404 must not be retried, got {n} attempts"
    assert isinstance(result, requests.HTTPError)


def test_size_cap_is_not_retried():
    # The audio will not get smaller on a second attempt.
    n, result = attempts([asr_core.InputError("Audio is 999999999 bytes, over the limit.")])
    assert n == 1, f"a size cap must not be retried, got {n} attempts"
    assert isinstance(result, asr_core.InputError)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
