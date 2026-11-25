"""Earliest public capture of a URL Vision already named.

A page can show a photograph and still have no publication day: a photo
album, a social post, a file on a CDN. The Internet Archive's CDX index
records when that URL was first saved with HTTP 200. That day is an
existed_by date. The URL was on the public web by then. It is not the
day the article was written, and it must not be treated as one when the
page already has an earlier trusted day.

The request is the CDX API, not the archived page:

    https://web.archive.org/cdx/search/cdx
        ?url=<url>&output=json&limit=1&fl=timestamp&filter=statuscode:200

CDX returns the oldest row first. limit=1 is therefore the earliest
capture, not the latest. The call goes through fetch._http_get after the
same public-address check a page download uses, with that module's six
second timeout. This file does not open its own socket. Tests replace
_read_public, or replace fetch._http_get, and the default test run
refuses both.

Answers are cached under data/cache/wayback, including a real "no
capture". A timeout or a body that is not a CDX table is not cached:
the next check can try again. A private target is not sent to the
archive at all. Asking about 127.0.0.1 would still be a lookup of an
address this process is not allowed to touch.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from urllib.parse import urlencode, urljoin

import misinfo_agent.config as config

# data/ is gitignored. The key is the URL Vision returned, so two images
# that name the same page share one capture.
CACHE_DIR = config.DATA_DIR / "cache" / "wayback"

# Off. Dating partials called no gold-True image false on its own, but
# dating full matches did: img04 and img06, both gold True, became
# Misleading from a PIB album captured before the day in the caption.
# A gallery page's earliest capture is not the day this file was taken.
# ENABLED off means no CDX query. DATE_PARTIALS stays off so turning
# the lookup back on does not date lookalikes first. The re-run is in
# docs/eval_diagnosis.md.
ENABLED = False
DATE_PARTIALS = False

_CDX = "https://web.archive.org/cdx/search/cdx"
# Cache miss. A cached None is a real empty CDX table and must not be
# fetched again. The two have to be different objects.
_ABSENT = object()


def cdx_url(url: str) -> str:
    """The one CDX query. The target is a parameter, not the host we open.

    The host we open is web.archive.org. The SSRF check runs on that
    host, and again on every redirect. The target URL is only looked up
    in the index.
    """
    query = urlencode(
        {
            "url": url,
            "output": "json",
            "limit": "1",
            "fl": "timestamp",
            "filter": "statuscode:200",
        }
    )
    return f"{_CDX}?{query}"


def existed_by(url: str) -> date | None:
    """The earliest HTTP 200 capture, as a calendar day, or None.

    None means there is no usable capture: the feature is off, the URL
    is not http(s), Wayback has no 200, or this attempt failed. A
    failure is not stored. An empty table is stored, because the archive
    answered.
    """
    if not ENABLED:
        return None
    text = " ".join(str(url or "").split())
    if not text.lower().startswith(("http://", "https://")):
        return None
    cached = _read_cache(text)
    if cached is not _ABSENT:
        return cached
    try:
        # A target this process would refuse to fetch is not asked about.
        # The CDX host is public; the parameter would still name the
        # private address.
        from misinfo_agent.fetch import _refuse_unless_public

        _refuse_unless_public(text)
        body = _read_public(cdx_url(text))
        day = parse_cdx(body)
    except Exception:
        return None
    _write_cache(text, day)
    return day


# The function object. Tests replace the name `existed_by` and put this
# back when they want the real lookup with a fake CDX body.
_existed_by = existed_by


def earliest(urls: list[str]) -> date | None:
    """The earliest existed_by among these URLs. A miss is skipped.

    One dead URL must not throw away a capture we already have for
    another. The caller compares this day with the page's own date and
    is the place that refuses to move that date later.
    """
    found: date | None = None
    seen: set[str] = set()
    for url in urls:
        text = str(url or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        day = existed_by(text)
        if day is None:
            continue
        if found is None or day < found:
            found = day
    return found


def parse_cdx(body: bytes) -> date | None:
    """The earliest timestamp in a CDX JSON table, or None when it is empty.

    The first row is the field name. Later rows are captures. limit=1
    leaves at most one, and the default order is oldest first. The
    minimum is taken anyway: a row out of order must not become a later
    day. A body that is not that table raises ValueError so the caller
    does not cache a block page or an error string as "no capture".
    """
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ValueError("cdx body was not json") from exc
    if payload == []:
        return None
    if not isinstance(payload, list):
        raise ValueError("cdx body was not a table")
    stamps: list[str] = []
    for index, row in enumerate(payload):
        if not isinstance(row, list) or not row:
            raise ValueError("cdx row was not a list")
        cell = str(row[0] or "").strip()
        # The header. An empty table after it is a real miss.
        if index == 0 and cell == "timestamp":
            continue
        if not cell:
            continue
        stamps.append(cell)
    if not stamps:
        return None
    days = []
    for stamp in stamps:
        day = _stamp_to_date(stamp)
        if day is not None:
            days.append(day)
    if not days:
        raise ValueError("cdx timestamp did not contain a day")
    return min(days)


def _stamp_to_date(stamp: str) -> date | None:
    """YYYYMMDDhhmmss, which is the only shape CDX writes for fl=timestamp.

    The time of day is dropped. The rest of the pipeline compares calendar
    days. A capture at 00:01 and a capture at 23:59 are the same existed_by
    day: the URL was online on that date.
    """
    text = stamp.strip()
    if len(text) < 8 or not text[:8].isdigit():
        return None
    try:
        return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def _read_public(url: str) -> bytes:
    """One SSRF-checked GET. Tests replace this, or replace fetch._http_get.

    JSON is accepted. The page downloader refuses anything that is not
    HTML or PDF, and a CDX body is JSON, so this cannot call that
    downloader. The guard, the redirect loop, the six second clock, and
    the HTTP client are the same ones.
    """
    import time

    from misinfo_agent.fetch import (
        FetchError,
        _MAX_REDIRECTS,
        _REDIRECTS,
        _check_deadline,
        _http_get,
        _read_body,
        _refuse_unless_public,
    )

    deadline = time.monotonic() + config.FETCH_TIMEOUT_S
    current = url
    for hop in range(_MAX_REDIRECTS + 1):
        _check_deadline(deadline)
        _refuse_unless_public(current)
        try:
            response = _http_get(current)
        except Exception as exc:
            raise FetchError("failed") from exc
        try:
            status = int(getattr(response, "status_code", 0))
            headers = getattr(response, "headers", {}) or {}
            if status in _REDIRECTS:
                if hop == _MAX_REDIRECTS:
                    raise FetchError("status 310")
                location = headers.get("Location") or headers.get("location")
                if not location:
                    raise FetchError("status 302")
                current = urljoin(current, str(location))
                continue
            if status != 200:
                raise FetchError(f"status {status}")
            # A capture list is a few dozen bytes. A larger body is not
            # the table we asked for.
            return _read_body(response, 65_536, deadline)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    raise FetchError("status 310")


def _cache_path(url: str):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _read_cache(url: str):
    path = _cache_path(url)
    if not path.is_file():
        return _ABSENT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _ABSENT
    if not isinstance(payload, dict) or payload.get("v") != 1:
        return _ABSENT
    raw = payload.get("existed_by")
    if raw is None:
        return None
    text = str(raw)
    if len(text) < 10:
        return _ABSENT
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return _ABSENT


def _write_cache(url: str, day: date | None) -> None:
    path = _cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"v": 1, "existed_by": day.isoformat() if day is not None else None}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)
