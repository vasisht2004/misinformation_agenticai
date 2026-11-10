"""Box LLM gateway, with no network and no key.

set_fake stands in for Gemini. The returns still pass through Pydantic and
the retry loop, which is what these tests are checking. A real call is used
only for the missing-key case, and the key is blanked first.
"""

import json
import os
import subprocess
import sys

import pytest
from pydantic import BaseModel

import misinfo_agent.config as config
import misinfo_agent.llm as llm
from misinfo_agent.llm import (
    LLMError,
    LLMQuotaError,
    call_structured,
    calls_today,
    clear_fake,
    get_usage,
    reset_usage,
    set_fake,
)


class Note(BaseModel):
    label: str
    score: int


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    # Cache files must not land in data/cache during a test, and a fake left
    # by one test must not answer the next.
    monkeypatch.setattr(config, "LLM_CACHE", False)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    clear_fake()
    reset_usage()
    yield
    clear_fake()
    reset_usage()


def test_valid_fake_output_returns_a_schema_instance():
    set_fake(lambda prompt, schema, images: {"label": "ok", "score": 2})
    result = call_structured("a claim", Note, purpose="extract")
    assert isinstance(result, Note)
    assert result.label == "ok"
    assert result.score == 2


def test_invalid_output_is_retried_then_raises_llmerror_with_purpose():
    calls = []
    prompt = "UNIQUE_PROMPT_TEXT_SHOULD_NOT_LEAK"

    def fake(prompt, schema, images):
        calls.append(prompt)
        return {"label": "missing the score"}

    set_fake(fake)
    with pytest.raises(LLMError) as caught:
        call_structured(prompt, Note, purpose="extract")
    message = str(caught.value)
    # First try plus two retries, then stop. A fourth call would mean the cap
    # is not applied.
    assert len(calls) == 3
    assert "extract" in message
    assert prompt not in message
    assert "output did not match the schema" in message


def test_invalid_then_valid_succeeds_with_one_retry():
    state = {"n": 0}

    def fake(prompt, schema, images):
        state["n"] += 1
        if state["n"] == 1:
            return {"label": "not yet"}
        return '{"label": "ok", "score": 1}'

    set_fake(fake)
    result = call_structured("a claim", Note, purpose="extract")
    assert result == Note(label="ok", score=1)
    rows = get_usage()
    assert len(rows) == 1
    assert rows[0]["retries"] == 1
    assert rows[0]["success"] is True
    assert rows[0]["purpose"] == "extract"


def test_usage_records_purpose_and_cached_flag_and_reset_clears():
    set_fake(lambda prompt, schema, images: {"label": "ok", "score": 1})
    call_structured("a claim", Note, purpose="assess", prompt_version="v4")
    rows = get_usage()
    assert len(rows) == 1
    assert rows[0]["purpose"] == "assess"
    assert rows[0]["cached"] is False
    assert rows[0]["prompt_version"] == "v4"
    assert rows[0]["success"] is True
    reset_usage()
    assert get_usage() == []


def test_cache_hit_does_not_call_the_fake_again(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    calls = []

    def fake(prompt, schema, images):
        calls.append((prompt, images))
        return {"label": "ok", "score": 4}

    set_fake(fake)
    first = call_structured("same words", Note, purpose="extract")
    second = call_structured("same words", Note, purpose="extract")
    assert calls == [("same words", None)]
    assert second == first
    rows = get_usage()
    assert [row["cached"] for row in rows] == [False, True]
    # A different picture is a different call. The image hash is part of the
    # cache key, otherwise extract would reuse a text answer for a new photo.
    call_structured("same words", Note, purpose="extract", images=[b"\xff\xd8\xff"])
    assert len(calls) == 2


def test_real_call_without_a_key_raises_a_clear_error(monkeypatch):
    secret = "not-a-real-key-xyz"
    # A blank value is "not set". It must not be replaced by whatever .env
    # holds, and the secret below must not appear in the message.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("UNRELATED_SECRET", secret)
    clear_fake()
    with pytest.raises(LLMError) as caught:
        call_structured("a claim", Note, purpose="extract")
    message = str(caught.value)
    assert message == "extract: GEMINI_API_KEY is not set in .env"
    assert secret not in message
    assert os.environ["UNRELATED_SECRET"] not in message


def test_daily_cap_raises_and_does_not_call(monkeypatch):
    today = llm._pacific_today()
    llm.DAILY_COUNT_PATH.write_text(
        json.dumps({today: config.LLM_DAILY_MAX}),
        encoding="utf-8",
    )
    created = []

    def _refuse(*args, **kwargs):
        created.append(1)
        raise AssertionError("daily cap still built a client")

    monkeypatch.setattr(llm, "_client", _refuse)
    with pytest.raises(LLMQuotaError, match="daily LLM cap reached") as caught:
        call_structured("a claim", Note, purpose="cap")
    message = str(caught.value)
    assert f"daily LLM cap reached ({config.LLM_DAILY_MAX} calls); resets at midnight Pacific" in message
    assert created == []
    assert calls_today() == config.LLM_DAILY_MAX


def test_cached_calls_do_not_count_toward_the_daily_cap(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    set_fake(lambda prompt, schema, images: {"label": "ok", "score": 3})
    call_structured("same words", Note, purpose="extract")
    assert calls_today() == 0
    clear_fake()
    # A dummy value, not a real key. The cache hit must return before the
    # client exists; if it does not, this raises and the count moves.
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-offline-key")

    def _refuse(*args, **kwargs):
        raise AssertionError("cached call reached the client")

    monkeypatch.setattr(llm, "_client", _refuse)
    again = call_structured("same words", Note, purpose="extract")
    assert again.score == 3
    assert calls_today() == 0
    assert get_usage()[-1]["cached"] is True
    assert get_usage()[-1]["quota_wait_s"] == 0


def test_rate_limiter_waits(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-offline-key")
    clock = {"now": 1000.0}
    slept = []

    def monotonic():
        return clock["now"]

    def sleep(seconds):
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(llm.time, "monotonic", monotonic)
    monkeypatch.setattr(llm.time, "sleep", sleep)
    llm._RECENT_REAL_CALLS[:] = [1000.0] * config.LLM_RPM

    class _Response:
        text = '{"label": "ok", "score": 1}'
        usage_metadata = None

    calls = []

    class _Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            return _Response()

    class _Client:
        def __init__(self):
            self.models = _Models()

    monkeypatch.setattr(llm, "_client", lambda *args, **kwargs: _Client())
    result = call_structured("pace", Note, purpose="pace")
    assert result.score == 1
    assert slept == [60.0]
    assert calls == [1]
    assert get_usage()[-1]["quota_wait_s"] == 60.0
    assert calls_today() == 1


def test_daily_429_is_not_retried(monkeypatch):
    from google.genai.errors import APIError

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-offline-key")
    calls = []

    class _Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            raise APIError(
                429,
                {
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": (
                            "Quota exceeded for generate_content_free_tier_requests "
                            "per day. Daily limit reached."
                        ),
                    }
                },
            )

    class _Client:
        def __init__(self):
            self.models = _Models()

    monkeypatch.setattr(llm, "_client", lambda *args, **kwargs: _Client())
    with pytest.raises(LLMQuotaError, match="API daily quota exhausted"):
        call_structured("a claim", Note, purpose="quota")
    assert calls == [1]


def test_importing_llm_does_not_require_a_key():
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    script = (
        "import sys\n"
        "import misinfo_agent.llm\n"
        "assert 'google.genai' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


class Ping(BaseModel):
    ok: bool


@pytest.mark.live
def test_live_smoke_one_tiny_schema_call():
    """One real call. The key is whatever llm.py loaded; only the last 4 are printed."""
    before = calls_today()
    result = call_structured("Set ok to true.", Ping, purpose="live_smoke")
    row = get_usage()[-1]
    key = os.environ.get("GEMINI_API_KEY", "")
    last4 = key[-4:] if len(key) >= 4 else "none"
    print(
        f"key_last4={last4} model={row['model']} latency_s={row['latency_s']} "
        f"prompt_tokens={row['prompt_tokens']} output_tokens={row['output_tokens']} "
        f"total_tokens={row['total_tokens']} quota_wait_s={row['quota_wait_s']} "
        f"calls_today_before={before} calls_today_after={calls_today()}"
    )
    assert result.ok is True
    assert row["cached"] is False
    assert row["success"] is True
