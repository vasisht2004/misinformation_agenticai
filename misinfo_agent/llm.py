"""The only module that calls a language model.

Every other box imports this module instead of a provider SDK. The provider
is misinfo_agent.config.LLM_PROVIDER, so a swap does not touch the boxes.
Two reasons the call lives here and nowhere else:

- The key stays in one place. A box that built its own client would end up
  printing it in a traceback.
- The output is a Pydantic instance before any box reads it. A model that
  returns a missing field fails here, instead of becoming a verdict.

The Gemini client is built on the first real call. Importing this module
does not read the key and does not open a socket, so tests and the other
boxes can import it with an empty .env.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

import misinfo_agent.config as config

T = TypeVar("T", bound=BaseModel)

# One try plus this many retries. A third identical failure is the caller's
# problem; looping longer would stall the graph on a prompt the model cannot
# satisfy.
_MAX_RETRIES = 2

# Seconds before a retry. Short on purpose: a schema miss is the model being
# sloppy, not a rate limit, and box 7 may already be on a 60s budget.
_BACKOFF_S = 0.05

# 429 is not in this set. A quota error is not a blip: retrying it on the
# generic loop is how a free tier gets exhausted. Quota is handled on its own.
_RETRYABLE_STATUS = {408, 500, 502, 503, 504}

_LOCK = threading.Lock()
_QUOTA_LOCK = threading.Lock()
_USAGE: list[dict] = []
_FAKE = None
_CLIENT = None
# Timestamps of real calls in this process, for the per-minute cap.
# Not written to disk: a new process may use its first window immediately,
# which is still under LLM_RPM.
_RECENT_REAL_CALLS: list[float] = []
# Sibling of the response cache, not inside it. data/ is gitignored.
DAILY_COUNT_PATH = config.DATA_DIR / "cache" / "llm_daily_count.json"
_MINUTE = 60.0
# A retry hint longer than this is the day resetting, not the minute window.
_MINUTE_RETRY_MAX_S = 120.0


class LLMError(Exception):
    """A call failed in a way a box can show.

    The message is the purpose plus a short reason. It is not the prompt and
    it is not the API key: both would land in the audit row and in logs.
    """

    def __init__(self, purpose: str, reason: str):
        self.purpose = purpose
        self.reason = reason
        super().__init__(f"{purpose}: {reason}")


class LLMQuotaError(LLMError):
    """The day's quota is gone. Callers stop the run instead of retrying.

    Raised for the local daily cap and for a daily 429 from Gemini.
    A per-minute 429 is waited out once inside the gateway and is not
    this error unless that single retry is also refused.
    """


class _Retryable(Exception):
    """A failure that may succeed on the next attempt. Not part of the public API."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def call_structured(
    prompt: str,
    schema: type[T],
    *,
    purpose: str,
    images: list[bytes] | None = None,
    temperature: float = 0.0,
    prompt_version: str = "v1",
) -> T:
    """One structured call. Returns an instance of schema.

    purpose is the audit label ("prior_match_veto", "extract", ...), not
    shown to the model. images is for extract's vision prompt; text calls
    leave it empty. temperature defaults to 0 because a fact-check label
    should not change between two runs of the same prompt.
    """
    started = time.perf_counter()
    model = _model_name()
    # Sleep spent on the per-minute cap or a single 429 wait. Schema retries
    # use a different, short backoff and are not quota time.
    meter = {"quota_wait_s": 0.0}
    cache_path = _cache_path(model, prompt_version, prompt, schema, images)
    cached = _read_cache(cache_path, schema) if cache_path is not None else None
    if cached is not None:
        # A hit is still a call, from the audit's point of view. cached=True
        # tells box 16 it did not spend tokens again. Token fields stay empty
        # so summing them does not bill the replay. It does not count toward
        # the daily cap either: the answer was already paid for.
        _record(
            purpose=purpose,
            model=model,
            prompt_version=prompt_version,
            started=started,
            success=True,
            retries=0,
            cached=True,
            tokens=None,
            quota_wait_s=0.0,
        )
        return cached

    reason = "the model call failed"
    retries = 0
    tokens = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            raw, tokens = _produce(
                prompt, schema, images, purpose, model, temperature, meter
            )
            instance = _parse(schema, raw)
        except LLMError:
            # Missing key, the daily cap, or a non-retryable API error.
            # Already safe to show. A quota error is not retried here.
            _record(
                purpose=purpose,
                model=model,
                prompt_version=prompt_version,
                started=started,
                success=False,
                retries=retries,
                cached=False,
                tokens=tokens,
                quota_wait_s=meter["quota_wait_s"],
            )
            raise
        except _Retryable as exc:
            reason = exc.reason
        except ValidationError:
            # The message on ValidationError quotes the model output. Drop it.
            # `from None` below keeps that quote out of the traceback too.
            reason = "output did not match the schema"
        else:
            if cache_path is not None:
                _write_cache(cache_path, instance)
            _record(
                purpose=purpose,
                model=model,
                prompt_version=prompt_version,
                started=started,
                success=True,
                retries=retries,
                cached=False,
                tokens=tokens,
                quota_wait_s=meter["quota_wait_s"],
            )
            return instance
        if attempt == _MAX_RETRIES:
            break
        retries += 1
        time.sleep(_BACKOFF_S * retries)

    _record(
        purpose=purpose,
        model=model,
        prompt_version=prompt_version,
        started=started,
        success=False,
        retries=retries,
        cached=False,
        tokens=tokens,
        quota_wait_s=meter["quota_wait_s"],
    )
    raise LLMError(purpose, reason)


def get_usage() -> list[dict]:
    """A copy of every call since the last reset.

    A copy, not the live list: box 16 and the eval both read this, and one
    of them appending a field must not change what the other sees.
    """
    with _LOCK:
        return [dict(row) for row in _USAGE]


def reset_usage() -> None:
    """Clear the counter. Tests and a new claim both start from zero."""
    with _LOCK:
        _USAGE.clear()


def set_fake(fn) -> None:
    """Send calls to fn instead of Gemini.

    fn(prompt, schema, images) returns a dict or a JSON string. The return
    value still goes through Pydantic and the retry loop, and it is still
    recorded in usage. While a fake is set, this module does not build a
    client and does not read the API key.
    """
    global _FAKE
    _FAKE = fn


def clear_fake() -> None:
    """Return to the real provider."""
    global _FAKE
    _FAKE = None


def _produce(prompt, schema, images, purpose, model, temperature, meter):
    # A fake never reaches the cap or the client. Tests stay offline, and a
    # fake answer must not burn a slot of the daily quota.
    if _FAKE is not None:
        return _FAKE(prompt, schema, images), None
    if config.LLM_PROVIDER != "gemini":
        raise LLMError(purpose, f"unsupported provider {config.LLM_PROVIDER}")
    # Cap first, then the key, then the slot. A missing key must not count
    # as a call, and a full day must not build a client.
    _refuse_if_daily_cap(purpose)
    key = _require_key(purpose)
    _take_real_slot(purpose, meter)
    text, tokens = _gemini(
        prompt, schema, images, purpose, model, temperature, key, meter
    )
    return text, tokens


def _parse(schema: type[T], raw) -> T:
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, str):
        return schema.model_validate_json(raw)
    if isinstance(raw, dict):
        return schema.model_validate(raw)
    # A fake that returns a list, or a model that returns prose, is the same
    # kind of failure as a schema miss: retry, then tell the caller.
    raise _Retryable("output did not match the schema")


def _model_name() -> str:
    _load_env()
    chosen = os.environ.get("GEMINI_MODEL", "").strip()
    # An empty GEMINI_MODEL= in .env means "use the default", not "call a
    # model whose name is the empty string".
    return chosen or config.GEMINI_MODEL


def _require_key(purpose: str) -> str:
    _load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMError(purpose, "GEMINI_API_KEY is not set in .env")
    return key


def _load_env() -> None:
    # override=False: a variable already in the process (a test, a shell)
    # wins over the file. A blank GEMINI_API_KEY in the process stays blank,
    # which is how the tests prove a missing key cannot fall through to .env.
    load_dotenv(config.ENV_PATH, override=False)


def calls_today() -> int:
    """Real Gemini calls already made on today's Pacific date.

    Scripts print this before and after a batch. Cached and fake calls
    are not in the number. A missing file, or a file from an earlier
    Pacific day, is zero: the quota day has reset.
    """
    with _QUOTA_LOCK:
        return _count_today_unlocked()


def _refuse_if_daily_cap(purpose: str) -> None:
    """Stop before a real call when today's count is already at the cap.

    Does not increment. The refused call is not a call.
    """
    with _QUOTA_LOCK:
        count = _count_today_unlocked()
        if count >= config.LLM_DAILY_MAX:
            raise _daily_cap(purpose, count)


def _take_real_slot(purpose: str, meter: dict) -> None:
    """Sleep to stay under LLM_RPM, then count this real call."""
    with _QUOTA_LOCK:
        count = _count_today_unlocked()
        if count >= config.LLM_DAILY_MAX:
            raise _daily_cap(purpose, count)
        meter["quota_wait_s"] += _pace_unlocked()
        _set_today_unlocked(count + 1)


def _daily_cap(purpose: str, count: int) -> LLMQuotaError:
    return LLMQuotaError(
        purpose,
        f"daily LLM cap reached ({count} calls); resets at midnight Pacific",
    )


def _increment_real_call(purpose: str) -> None:
    """Count the single extra attempt after a per-minute 429.

    No second pace: the caller already slept for the delay Gemini asked
    for. If that extra attempt would cross the daily cap, stop.
    """
    with _QUOTA_LOCK:
        count = _count_today_unlocked()
        if count >= config.LLM_DAILY_MAX:
            raise _daily_cap(purpose, count)
        _set_today_unlocked(count + 1)
        _RECENT_REAL_CALLS.append(time.monotonic())


def _pacific_today() -> str:
    # Gemini's free-tier day ends at midnight Pacific, not local time.
    # An India-time date would keep calling for hours after the reset,
    # or stop while the quota was already fresh.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def _count_today_unlocked() -> int:
    return int(_read_counts().get(_pacific_today(), 0))


def _set_today_unlocked(count: int) -> None:
    # Only today's key is kept. Yesterday's count must not survive the
    # Pacific midnight and make the new day look already full.
    _write_counts({_pacific_today(): count})


def _read_counts() -> dict:
    path = DAILY_COUNT_PATH
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    counts = {}
    for key, value in payload.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def _write_counts(counts: dict) -> None:
    path = DAILY_COUNT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(counts), encoding="utf-8")
    temporary.replace(path)


def _pace_unlocked() -> float:
    """Sleep so this process stays under LLM_RPM real calls per minute.

    Returns the seconds slept. The oldest call in the window is what we
    wait on: when it ages out, one new call fits.
    """
    limit = int(config.LLM_RPM)
    now = time.monotonic()
    fresh = [stamp for stamp in _RECENT_REAL_CALLS if now - stamp < _MINUTE]
    _RECENT_REAL_CALLS[:] = fresh
    waited = 0.0
    if limit > 0 and len(_RECENT_REAL_CALLS) >= limit:
        wait_for = _MINUTE - (now - _RECENT_REAL_CALLS[0])
        if wait_for > 0:
            time.sleep(wait_for)
            waited = wait_for
            now = time.monotonic()
            _RECENT_REAL_CALLS[:] = [
                stamp for stamp in _RECENT_REAL_CALLS if now - stamp < _MINUTE
            ]
    _RECENT_REAL_CALLS.append(time.monotonic())
    return waited


def _generate(client, model, contents, gen_config):
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=gen_config,
    )


def _after_api_error(exc, client, model, contents, gen_config, purpose, prompt, key, meter):
    """One chance on a per-minute 429. A daily 429 stops immediately.

    The generic retry loop must not see these. Two or three repeats of a
    quota error are still the same exhausted window.
    """
    if _is_quota(exc):
        if _quota_kind(exc) == "daily":
            raise LLMQuotaError(
                purpose,
                "API daily quota exhausted; resets at midnight Pacific",
            ) from None
        delay = _retry_after_seconds(exc)
        if delay is None:
            delay = _MINUTE / max(int(config.LLM_RPM), 1)
        time.sleep(delay)
        meter["quota_wait_s"] += delay
        _increment_real_call(purpose)
        try:
            return _generate(client, model, contents, gen_config)
        except Exception as again:
            if _is_quota(again):
                raise LLMQuotaError(
                    purpose,
                    "per-minute LLM quota persisted after one wait",
                ) from None
            return _raise_api(again, purpose, prompt, key)
    return _raise_api(exc, purpose, prompt, key)


def _raise_api(exc, purpose: str, prompt: str, key: str | None):
    reason = _reason_from_api(exc, prompt, key)
    if _is_transient(exc):
        raise _Retryable(reason) from None
    raise LLMError(purpose, reason) from None


def _is_quota(exc: BaseException) -> bool:
    try:
        from google.genai.errors import APIError
    except Exception:
        return False
    if not isinstance(exc, APIError):
        return False
    if getattr(exc, "code", None) == 429:
        return True
    blob = _quota_blob(exc).upper()
    return "RESOURCE_EXHAUSTED" in blob


def _quota_kind(exc: BaseException) -> str:
    """'daily' or 'minute'.

    Gemini uses 429 and RESOURCE_EXHAUSTED for both windows. A per-day
    message, or a retry measured in many minutes, will not succeed if we
    try again now. A short 'retry in Ns' with no daily wording is the
    minute window. Anything we cannot tell apart is treated as daily:
    one extra call cannot help, and a retry loop spends the free tier.
    """
    blob = _quota_blob(exc).lower()
    delay = _retry_after_seconds(exc)
    daily = re.search(r"per[-_ ]?day|daily|requests_per_day", blob) is not None
    minute = re.search(r"per[-_ ]?minute|rate limit|rate_limit", blob) is not None
    if daily and not minute:
        return "daily"
    if minute and not daily:
        return "minute"
    if delay is not None and delay < _MINUTE_RETRY_MAX_S:
        return "minute"
    return "daily"


def _retry_after_seconds(exc: BaseException) -> float | None:
    blob = _quota_blob(exc)
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s", blob, re.IGNORECASE)
    if match is None:
        match = re.search(
            r"retryDelay['\"]?\s*[:=]\s*['\"]([0-9]+(?:\.[0-9]+)?)s",
            blob,
            re.IGNORECASE,
        )
    if match is None:
        return None
    return float(match.group(1))


def _quota_blob(exc: BaseException) -> str:
    parts = [str(exc)]
    details = getattr(exc, "details", None)
    if details is not None:
        try:
            parts.append(json.dumps(details))
        except TypeError:
            parts.append(str(details))
    message = getattr(exc, "message", None)
    if message:
        parts.append(str(message))
    return " ".join(parts)


def _client(key: str, purpose: str, prompt: str):
    """Build the Gemini client once. The key is an argument, never a log line."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        # Imported here so `import misinfo_agent.llm` does not need the SDK
        # to be configured, and does not read the key at import time.
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(config.LLM_TIMEOUT_S * 1000)),
        )
    except Exception as exc:
        raise LLMError(purpose, _reason_from_api(exc, prompt, key)) from None
    _CLIENT = client
    return client


def _gemini(prompt, schema, images, purpose, model, temperature, key, meter):
    from google.genai import types

    client = _client(key, purpose, prompt)
    # Text is the whole request unless extract attached pictures. The bytes
    # go as parts because the model has to see the image, not a description
    # of it. The mime type is guessed; a screenshot with no header is sent
    # as JPEG, which is what phone forwards usually are.
    if images:
        contents = [types.Part.from_text(text=prompt)]
        for blob in images:
            contents.append(types.Part.from_bytes(data=blob, mime_type=_mime(blob)))
    else:
        contents = prompt
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        # Native schema, then our own Pydantic check. The SDK's parsed object
        # is not trusted: a field the schema requires can still arrive empty.
        response_json_schema=schema.model_json_schema(),
    )
    try:
        response = _generate(client, model, contents, gen_config)
    except Exception as exc:
        response = _after_api_error(
            exc, client, model, contents, gen_config, purpose, prompt, key, meter
        )
    return _response_text(response), _token_counts(response)


def _mime(blob: bytes) -> str:
    import filetype

    kind = filetype.guess(blob)
    if kind is not None and kind.mime.startswith("image/"):
        return kind.mime
    return "image/jpeg"


def _response_text(response) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Some responses fill .parsed and leave .text empty. Re-serialise so the
    # same Pydantic check runs either way.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()
    if isinstance(parsed, dict):
        return json.dumps(parsed)
    return text or ""


def _token_counts(response) -> dict:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {"prompt_tokens": None, "output_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
        "total_tokens": getattr(meta, "total_token_count", None),
    }


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    try:
        import httpx
    except Exception:
        httpx = None
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return True
    try:
        from google.genai.errors import APIError
    except Exception:
        return False
    if isinstance(exc, APIError):
        return getattr(exc, "code", None) in _RETRYABLE_STATUS
    return False


def _reason_from_api(exc: BaseException, prompt: str, key: str | None) -> str:
    text = str(exc)
    if key:
        text = text.replace(key, "[redacted]")
    if prompt:
        text = text.replace(prompt, "[prompt]")
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:180] + "..."
    return text or "the model call failed"


def _cache_path(model, prompt_version, prompt, schema, images):
    # Off unless a run opts in. A default of on would replay yesterday's
    # answer after the prompt changed, and the eval would not notice.
    if not config.LLM_CACHE:
        return None
    schema_json = json.dumps(
        schema.model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    # Newlines, not raw concatenation: "ab"+"c" and "a"+"bc" must not share
    # a cache entry. Image bytes are hashed first so the key stays short and
    # the picture itself is not written into the filename.
    image_part = "".join(hashlib.sha256(blob).hexdigest() for blob in (images or []))
    raw = f"{model}\n{prompt_version}\n{prompt}\n{schema_json}\n{image_part}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return config.LLM_CACHE_DIR / f"{digest}.json"


def _read_cache(path, schema: type[T]) -> T | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Validate again. A half-written or hand-edited file is a miss, not
        # a verdict. The caller then makes a real call.
        return schema.model_validate(payload["output"])
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError):
        return None


def _write_cache(path, instance: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"output": instance.model_dump(mode="json")}, ensure_ascii=False)
    # Write aside and replace, so a crash cannot leave a truncated JSON file
    # that the next call would trust.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def _record(
    *,
    purpose,
    model,
    prompt_version,
    started,
    success,
    retries,
    cached,
    tokens,
    quota_wait_s,
) -> None:
    counts = tokens or {}
    row = {
        "purpose": purpose,
        "model": model,
        "prompt_version": prompt_version,
        "latency_s": round(time.perf_counter() - started, 6),
        "success": success,
        "retries": retries,
        "cached": cached,
        "quota_wait_s": round(quota_wait_s, 6),
        "prompt_tokens": counts.get("prompt_tokens"),
        "output_tokens": counts.get("output_tokens"),
        "total_tokens": counts.get("total_tokens"),
    }
    with _LOCK:
        _USAGE.append(row)
