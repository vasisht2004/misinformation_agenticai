"""The only module that calls Google Cloud Vision.

Reverse image search for an event photo lives here. A box that posted
the picture itself would end up with the key in a traceback, and it
would not see the monthly cap. Importing this module does not read the
key and does not open a socket. The request is built on the first real
call.

The feature is WEB_DETECTION only. Face, celebrity, and landmark
features are not requested: those name a person, and a name is not
evidence of when a photograph was taken. webEntities are ignored for
the same reason. bestGuessLabels are generic words ("flood", "stadium").
They are kept so a reader can see what Vision thought the picture was,
and they never become a verdict.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import misinfo_agent.config as config

_LOCK = threading.Lock()
_FAKE = None

# Sibling of the response cache, not inside it. Wiping cached detections
# must not reset the month. data/ is gitignored.
MONTHLY_COUNT_PATH = config.DATA_DIR / "cache" / "vision_monthly_count.json"

_MATCH = {"full", "partial", "page"}


class VisionError(Exception):
    """A detection failed in a way a caller can record.

    The message is a short reason. It is not the image and it is not the
    API key: both would land in a log. detect() does not raise this for a
    missing key or a full month. Those are statuses, because a photo check
    must still finish and say what it could not do.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def set_fake(fn) -> None:
    """Send detections to fn instead of Vision.

    fn(image_bytes) returns an annotate response, a webDetection object,
    or {"pages", "best_guess"}. The return still goes through the parser
    when it looks like an API body, so a test that plants webEntities
    proves they are dropped. A fake does not read the key, does not open
    a socket, and does not count against the monthly cap.
    """
    global _FAKE
    _FAKE = fn


def clear_fake() -> None:
    """Return to the real service."""
    global _FAKE
    _FAKE = None


def detect(image_path: str, *, cache_only: bool = False) -> dict:
    """WEB_DETECTION for the JPEG at image_path.

    Always returns a dict. vision_status is ran, cached, monthly_cap,
    no_key, error, not_image, or cache_only_miss. A missing key and a
    full month are statuses, not exceptions: the rest of the claim still
    runs. pages are {url, title, match} with match full, partial, or page.
    best_guess is the bestGuessLabels only.

    cache_only is the reuse path. The verdict is already decided, and the
    card only needs the pages we have already paid for. A miss returns
    cache_only_miss and does not open a socket or move the counter.
    """
    data = _read_jpeg(image_path)
    if data is None:
        return _empty("not_image")
    digest = hashlib.sha256(data).hexdigest()
    cached = _read_cache(digest)
    if cached is not None:
        # Already paid for. Reading it must not move the monthly counter.
        cached["vision_status"] = "cached"
        return cached
    if cache_only:
        return _empty("cache_only_miss")
    if _FAKE is not None:
        parsed = _from_any(_FAKE(data))
        if parsed is None:
            return _empty("error")
        parsed["vision_status"] = "ran"
        return parsed
    key = _key()
    if not key:
        return _empty("no_key")
    if calls_this_month() >= config.VISION_MONTHLY_MAX:
        # Refuse before the socket. The refused call is not a call.
        return _empty("monthly_cap")
    try:
        body = _post(data, key)
    except Exception:
        # The reason stays inside this function. A message from the HTTP
        # library can contain the request URL, and the key is on that URL.
        return _empty("error")
    parsed = _parse_api(body)
    if parsed is None:
        return _empty("error")
    _increment()
    _write_cache(digest, parsed)
    parsed["vision_status"] = "ran"
    return parsed


def calls_this_month() -> int:
    """Successful Vision responses already received this UTC month.

    A response counts when the JSON parsed. A refused call (no key, cap,
    HTTP error) does not: it did not spend the budget we track. Cached
    and fake calls are not in the number. A missing file, or a file from
    an earlier month, is zero.
    """
    with _LOCK:
        return _count_unlocked()


def request_body(image: bytes) -> dict:
    """The JSON body. Tests check this without opening a socket.

    One feature, WEB_DETECTION. maxResults comes from config so a test
    can see the number without a socket. Face, celebrity, and landmark
    features are absent on purpose.
    """
    return {
        "requests": [
            {
                "image": {"content": base64.b64encode(image).decode("ascii")},
                "features": [{"type": "WEB_DETECTION", "maxResults": int(config.VISION_MAX_RESULTS)}],
            }
        ]
    }


def _empty(status: str) -> dict:
    return {"vision_status": status, "best_guess": [], "pages": []}


def _read_jpeg(image_path: str) -> bytes | None:
    """The stored JPEG, or None when this is not that file.

    Ingest writes a downscaled, EXIF-free JPEG. Anything else (a missing
    path, a text file, a PNG) is not_image rather than a guess sent to
    Vision. The original upload is not read: it may still hold GPS.
    """
    text = str(image_path or "").strip()
    if not text:
        return None
    path = Path(text)
    try:
        if not path.is_file():
            return None
        data = path.read_bytes()
    except OSError:
        return None
    # JPEG start marker. A renamed text file must not be posted.
    if not data.startswith(b"\xff\xd8"):
        return None
    return data


def _from_any(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    if "responses" in raw:
        return _parse_api(raw)
    if "webDetection" in raw:
        return _parse_api({"responses": [{"webDetection": raw["webDetection"]}]})
    pages = []
    for row in raw.get("pages") or []:
        page = _page(row if isinstance(row, dict) else {})
        if page is not None:
            pages.append(page)
    labels = []
    for label in raw.get("best_guess") or []:
        text = " ".join(str(label).split())
        if text and text not in labels:
            labels.append(text)
    return {"best_guess": labels, "pages": pages}


def _parse_api(body) -> dict | None:
    """Pages and labels. webEntities are not read.

    The key is never copied onto the result. An error object on the
    response is a failure, not an empty detection: an empty detection is
    a real answer that the picture was not found.
    """
    if not isinstance(body, dict):
        return None
    responses = body.get("responses")
    if not isinstance(responses, list) or not responses:
        return None
    first = responses[0] if isinstance(responses[0], dict) else None
    if first is None:
        return None
    if first.get("error"):
        return None
    web = first.get("webDetection")
    if not isinstance(web, dict):
        web = {}
    # webEntities can name a person. Do not bind that list, do not copy
    # it into the cache, and do not put a description into best_guess.
    pages = []
    seen = set()
    for row in web.get("pagesWithMatchingImages") or []:
        if not isinstance(row, dict):
            continue
        page = _page_from_api(row)
        if page is None or page["url"] in seen:
            continue
        seen.add(page["url"])
        pages.append(page)
    labels = []
    for item in web.get("bestGuessLabels") or []:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("label") or "")
        else:
            continue
        text = " ".join(text.split())
        if text and text not in labels:
            labels.append(text)
    return {"best_guess": labels, "pages": pages}


def _page_from_api(row: dict) -> dict | None:
    """One pagesWithMatchingImages entry.

    full wins over partial when the page lists both: a full match is the
    same file, and that is the one the gate can later treat as support.
    A page with neither list is still a page that contained the picture
    somewhere; it is match "page" and cannot by itself make a claim True.
    """
    url = " ".join(str(row.get("url") or "").split())
    if not url:
        return None
    if row.get("fullMatchingImages"):
        kind = "full"
    elif row.get("partialMatchingImages"):
        kind = "partial"
    else:
        kind = "page"
    title = " ".join(str(row.get("pageTitle") or "").split())
    return {"url": url, "title": title, "match": kind}


def _page(row: dict) -> dict | None:
    url = " ".join(str(row.get("url") or "").split())
    kind = str(row.get("match") or "page")
    if not url or kind not in _MATCH:
        return None
    title = " ".join(str(row.get("title") or row.get("pageTitle") or "").split())
    return {"url": url, "title": title, "match": kind}


def _post(image: bytes, key: str) -> dict:
    """One annotate call. The key is a query parameter, never a log line."""
    import requests

    try:
        response = requests.post(
            "https://vision.googleapis.com/v1/images:annotate",
            params={"key": key},
            json=request_body(image),
            timeout=30,
        )
    except Exception as exc:
        raise VisionError(_redact(exc, key)) from None
    if response.status_code != 200:
        # The body can echo the request. Do not attach it.
        raise VisionError(f"vision http {response.status_code}")
    try:
        payload = response.json()
    except Exception as exc:
        raise VisionError(_redact(exc, key)) from None
    if not isinstance(payload, dict):
        raise VisionError("vision response was not an object")
    return payload


def _redact(exc: BaseException, key: str) -> str:
    text = str(exc)
    if key:
        text = text.replace(key, "[redacted]")
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:180] + "..."
    return text or "the vision call failed"


def _key() -> str:
    _load_env()
    return os.environ.get("GOOGLE_VISION_API_KEY", "").strip()


def _load_env() -> None:
    # override=False: a blank key already in the process stays blank,
    # which is how the tests prove a missing key cannot fall through to .env.
    load_dotenv(config.ENV_PATH, override=False)


def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _count_unlocked() -> int:
    try:
        return int(_read_counts().get(_month_key(), 0))
    except (TypeError, ValueError):
        return 0


def _increment() -> None:
    with _LOCK:
        count = _count_unlocked()
        _write_counts({_month_key(): count + 1})


def _read_counts() -> dict:
    path = MONTHLY_COUNT_PATH
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
    path = MONTHLY_COUNT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(counts), encoding="utf-8")
    temporary.replace(path)


def _cache_path(digest: str):
    if not config.VISION_CACHE:
        return None
    # The page limit is part of the name. A file saved when the request
    # asked for 20 pages must not answer a request that asks for 50:
    # those extra pages were never returned, so the photograph is asked
    # again once and the new file is what later runs read.
    limit = int(config.VISION_MAX_RESULTS)
    return config.VISION_CACHE_DIR / f"{digest}.r{limit}.json"


def _read_cache(digest: str) -> dict | None:
    path = _cache_path(digest)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    parsed = _from_any(payload if isinstance(payload, dict) else {})
    if parsed is None:
        return None
    # A cached file must not smuggle a person-name field back in.
    parsed.pop("webEntities", None)
    return parsed


def _write_cache(digest: str, parsed: dict) -> None:
    path = _cache_path(digest)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"best_guess": list(parsed.get("best_guess") or []), "pages": list(parsed.get("pages") or [])}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body), encoding="utf-8")
    temporary.replace(path)
