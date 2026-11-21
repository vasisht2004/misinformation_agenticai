"""Fetch the pages box 7 already found. No stance and no model call.

Tavily's snippet is a few lines. The passage scorer needs the article.
This module is the only place those pages are downloaded. A social post
or a stock photo shows that the claim is circulating; it is not evidence,
and it is not fetched. A private address is not a page either: it is how
a cloud VM reads its own credentials (169.254.169.254). Both are refused
before a socket is opened to them.

When the search already returned the page text, that text is used and the
URL is not opened. A failed fetch does not drop the hit. The snippet is
what we still have, and fetched=False tells the passage step it is a
snippet, not an article.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Literal
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, Field

import misinfo_agent.config as config
from misinfo_agent.search import Hit, registrable_domain, site_host

_log = logging.getLogger(__name__)
# URLs opened over HTTP in this process, not served from the page cache
# and not taken from Tavily's raw text. The eval prints the list so a
# cache-only run can see which pages were new. Tavily is not involved.
_FRESH_FETCHES: list[str] = []


def fresh_page_fetches() -> list[str]:
    """Pages downloaded since the last reset. A copy, not the live list."""
    return list(_FRESH_FETCHES)


def reset_fresh_page_fetches() -> None:
    _FRESH_FETCHES.clear()


def _note_fresh_fetch(url: str) -> None:
    _FRESH_FETCHES.append(url)
    _log.info("fresh page fetch %s", url)

# HTML bodies. A circular over this is not an article we can hold in memory
# for one claim. PDFs are larger because a gazette is a scan of many pages;
# the cap for those is _MAX_PDF_BYTES, and only the first pages are read.
_MAX_HTML_BYTES = 2 * 1024 * 1024
_MAX_PDF_BYTES = 5 * 1024 * 1024
_PDF_PAGES = 10
_WORKERS = 8
_REDIRECTS = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 3

# A paywall or an empty shell. Below this the extract is the chrome, and
# the snippet is the only sentence the search actually gave us.
_MIN_ARTICLE_WORDS = 80

# Tavily's raw content is sometimes the whole site, millions of characters.
# Scoring every window of that stalled a claim for many minutes. The lede
# and the first part of the body are where the claim is discussed.
_MAX_ARTICLE_CHARS = 20_000
_MAX_ARTICLE_WORDS = 2500

# htmldate's own default ceiling is about today, which would swallow a
# future date and look like "no date". A far ceiling lets the date out so
# the untrusted rule below can label it instead of hiding it.
_HTMLDATE_MAX = "2100-12-31"

# /2016/11/08/ and /2016-11-08/. The slashes (or the ends of the path) keep
# a year that is just an id, such as /note-2016/, from becoming a date.
_URL_YMD = re.compile(r"(?:^|/)(\d{4})/(\d{2})/(\d{2})(?:/|$)")
_URL_ISO = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})(?:/|$)")
# The head of the article. A date in a "related stories" block further
# down is a different piece, and a claim's own date ("8 November 2016")
# is not the day the page was published.
_TEXT_DATE_WINDOW = 2500
_MONTH_NUMBER = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MDY = r"([A-Za-z]{3,9}\.?)\s+(\d{1,2}),?\s+(\d{4})"
_ISO_DATE = r"(\d{4})-(\d{2})-(\d{2})"
# "Published On Jul 15, 2025" and "This Article is From Jul 15, 2025".
# "published from a syndicated feed" does not match: no date follows.
_PUBLISHED_LINE = re.compile(
    rf"(?:published(?:\s+on)?|this article is from)\s*[:\-]?\s*(?:{_MDY}|{_ISO_DATE})",
    re.IGNORECASE,
)
# A byline date, which is the only date moneycontrol's extracted text
# keeps: "— PIB Fact Check (@PIBFactCheck) July 15, 2025". A sentence
# that merely mentions an event date has no byline mark, so it stays out.
_BYLINE_LINE = re.compile(
    rf"(?:—|–|--|\bBy\b)[^\n]{{0,80}}?{_MDY}",
    re.IGNORECASE,
)

# "text" is a publication line in the article we already have. Used when
# the host refuses the date-only fetch (moneycontrol and ndtv return 403)
# and the page cache was stored from Tavily text with no HTML date.
_DateSource = Literal["htmldate", "url", "trafilatura", "tavily", "untrusted", "page", "text"]
_Status = Literal["ok", "failed", "snippet", "circulation", "non_english"]
_BodyFrom = Literal["http", "pdf", "tavily", "snippet", "none"]


class FetchError(Exception):
    """The page was not an article we can read. The caller keeps the snippet.

    reason is one of the labels the run reports: a status code, timeout,
    non-HTML, too large, or blocked-IP. It is not the URL and not a body.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class Page(BaseModel):
    """One candidate after the download attempt.

    fetched is True only when the article text came from the page or from
    Tavily's raw content. fetch_status separates a download that worked
    ("ok") from a social host we refused ("circulation"), a hit past the
    fetch cap ("snippet"), a page in another language ("non_english"), and
    a try that failed ("failed"). failure_reason is set on the last of those.
    date_source names which clue survived. "untrusted" means the only dates
    we found were today or in the future, so published_date is empty.
    """

    url: str
    title: str = ""
    text: str = ""
    sitename: str | None = None
    published_date: str | None = None
    date_source: _DateSource | None = None
    fetched: bool = False
    fetch_status: _Status = "failed"
    failure_reason: str | None = None
    is_pdf: bool = False
    body_from: _BodyFrom = "none"
    query_kinds: list[str] = Field(default_factory=list)


def fetch_pages(hits: list[Hit]) -> list[Page]:
    """Download the hits that can be articles. Order matches the input.

    The input is reserved-slot order, so the official page is near the
    front. Only the first FETCH_MAX of those are opened. A circulation
    host does not take one of those slots: there is nothing to extract,
    and spending the slot on it would push a ministry page into the tail.
    Eight workers: more would look like a scrape, and fewer would make
    one slow host block the rest of the claim.
    """
    if not hits:
        return []
    modes = _fetch_modes(hits)
    pages: list[Page | None] = [None] * len(hits)
    to_fetch = []
    for index, (hit, mode) in enumerate(zip(hits, modes)):
        if mode == "circulation":
            pages[index] = _circulation(hit)
        elif mode == "snippet":
            pages[index] = _snippet(hit, "snippet", None)
        else:
            to_fetch.append(index)
    if to_fetch:
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            fetched = list(pool.map(lambda index: fetch_page(hits[index]), to_fetch))
        for index, page in zip(to_fetch, fetched):
            pages[index] = page
    return [page for page in pages if page is not None]


def fetch_page(hit: Hit) -> Page:
    """One hit. A skip or a failure returns the Tavily snippet, or nothing.

    Circulation hosts return no article text: a post that repeats the
    rumour is a sighting, and the passage step must not quote it.
    """
    if is_circulation(hit.url):
        return _circulation(hit)
    cached = _read_cache(hit.url)
    if cached is not None:
        return _from_cache(hit, cached)
    # The first slice, before split(). A multi-megabyte body is the whole
    # site, and splitting it is the slow part. Twenty thousand characters
    # is the head of the article; the word cap below trims it further.
    raw = hit.raw_content or ""
    if len(raw) > 20_000:
        raw = raw[:20_000]
    raw = raw.strip()
    # Long enough to be the article. A short raw body is a stub, and the
    # URL is still worth opening. This text is not HTML, so the date can
    # only come from the URL or from Tavily.
    if len(raw.split()) >= _MIN_ARTICLE_WORDS:
        page = _from_text(hit, raw, html=None, meta_date=None, is_pdf=False, body_from="tavily")
        if page.fetch_status in {"ok", "non_english"}:
            _write_cache(hit.url, page)
        return page
    try:
        # Abandoned at eight seconds even if the socket is still open.
        # requests will try every address getaddrinfo returned, and one
        # dead address used to hold the whole claim.
        _note_fresh_fetch(hit.url)
        final_url, body, content_type = _bounded(_download, hit.url)
    except FetchError as exc:
        return _snippet(hit, "failed", exc.reason)
    except Exception as exc:
        return _snippet(hit, "failed", _reason_from_exception(exc))
    if _is_pdf(content_type, final_url):
        try:
            text = _pdf_text(body)
        except FetchError as exc:
            return _snippet(hit, "failed", exc.reason)
        page = _from_text(
            hit, text, html=None, meta_date=None, is_pdf=True, body_from="pdf",
            final_url=final_url,
        )
    else:
        html = body.decode(_charset(content_type) or "utf-8", errors="replace")
        # The head and the article are at the front. Trafilatura on a
        # two-megabyte file is most of the wait, and the date tags are
        # in the first part of the document.
        if len(html) > 400_000:
            html = html[:400_000]
        try:
            extracted = _extract(html, final_url)
        except FetchError as exc:
            return _snippet(hit, "failed", exc.reason)
        page = _from_text(
            hit,
            extracted["text"],
            html=html,
            meta_date=extracted.get("date"),
            is_pdf=False,
            body_from="http",
            final_url=final_url,
            title=(extracted.get("title") or "").strip(),
            sitename=_blank(extracted.get("sitename")),
        )
    if page.fetch_status in {"ok", "non_english"}:
        _write_cache(hit.url, page)
    return page


def is_circulation(url: str) -> bool:
    """True when the host is a post or a stock library, including subdomains."""
    host = registrable_domain(site_host(url) or url)
    return host in config.NON_EVIDENCE_HOSTS


def date_missing(urls: list[str], *, deadline: float) -> dict[str, tuple[str, str]]:
    """A trusted publication date for URLs that did not have one.

    The path is free and is tried first: /2016/11/08/ is the date the
    site already published. Only a URL with no trusted path date is
    opened, and only for the head of the HTML, and only while the claim
    still has time left. The article text is not extracted and nothing
    is written to the page cache. A failure leaves that URL out of the
    dict. The caller keeps the passage undated.
    """
    found: dict[str, tuple[str, str]] = {}
    pending: list[str] = []
    for url in urls:
        path_date = _trusted_date(_url_date(url))
        if path_date:
            found[url] = (path_date, "url")
            continue
        pending.append(url)
    if not pending or time.monotonic() >= deadline:
        return found
    remaining = deadline - time.monotonic()
    if remaining < 0.5:
        return found
    workers = min(_WORKERS, len(pending))
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for url in pending:
            futures[pool.submit(_date_html, url)] = url
        from concurrent.futures import TimeoutError as FuturesTimeout
        from concurrent.futures import as_completed

        try:
            for future in as_completed(futures, timeout=remaining):
                url = futures[future]
                try:
                    html = future.result()
                except Exception:
                    continue
                dated = _trusted_date(_htmldate(html)) if html else None
                if dated:
                    found[url] = (dated, "htmldate")
        except FuturesTimeout:
            # The claim budget ran out. Dates already found stay.
            pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return found


def _trusted_date(raw) -> str | None:
    """A date strictly before today. Today and the future are stamps, not publication."""
    parsed = _parse_date(raw)
    if parsed is None or parsed >= date.today():
        return None
    return parsed.isoformat()


def _date_html(url: str) -> str | None:
    """The first slice of HTML, or None. Never raises: a date miss is not a failed claim."""
    from concurrent.futures import ThreadPoolExecutor as _Pool
    from concurrent.futures import TimeoutError as FuturesTimeout

    pool = _Pool(max_workers=1)
    try:
        future = pool.submit(_download_for_date, url)
        try:
            return future.result(timeout=config.DATE_FETCH_TIMEOUT_S)
        except (FuturesTimeout, Exception):
            return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _download_for_date(url: str) -> str | None:
    """Open the URL only far enough to read a publication date.

    Four seconds for every hop and every chunk. The body stops at
    DATE_FETCH_BYTES even when the page is larger: the meta tags are at
    the top, and reading the rest would repeat the article download.
    """
    deadline = time.monotonic() + config.DATE_FETCH_TIMEOUT_S
    current = url
    for hop in range(_MAX_REDIRECTS + 1):
        _check_deadline(deadline)
        lookup = max(0.2, deadline - time.monotonic())
        _refuse_unless_public(current, lookup_timeout=lookup)
        try:
            response = _http_get(current)
        except Exception:
            return None
        try:
            status = int(getattr(response, "status_code", 0))
            headers = getattr(response, "headers", {}) or {}
            if status in _REDIRECTS:
                if hop == _MAX_REDIRECTS:
                    return None
                location = headers.get("Location") or headers.get("location")
                if not location:
                    return None
                current = urljoin(current, str(location))
                continue
            if status != 200:
                return None
            ctype = str(headers.get("Content-Type") or headers.get("content-type") or "")
            if _body_kind(ctype, current) != "html" and ctype.strip():
                return None
            raw = _read_prefix(response, int(config.DATE_FETCH_BYTES), deadline)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
        if not raw:
            return None
        encoding = _charset(ctype) or "utf-8"
        try:
            return raw.decode(encoding, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")
    return None


def _read_prefix(response, limit: int, deadline: float) -> bytes:
    """The start of the body. A long page is cut, not rejected."""
    chunks = []
    total = 0
    iterator = getattr(response, "iter_content", None)
    if iterator is None:
        raw = getattr(response, "content", b"") or b""
        return raw[:limit]
    for chunk in iterator(chunk_size=65536):
        _check_deadline(deadline)
        if not chunk:
            continue
        room = limit - total
        if room <= 0:
            break
        piece = chunk[:room]
        chunks.append(piece)
        total += len(piece)
        if total >= limit:
            break
    return b"".join(chunks)


def _fetch_modes(hits: list[Hit]) -> list[str]:
    modes = []
    budget = int(config.FETCH_MAX)
    for hit in hits:
        if is_circulation(hit.url):
            modes.append("circulation")
            continue
        if budget > 0:
            modes.append("fetch")
            budget -= 1
            continue
        modes.append("snippet")
    return modes


def _from_text(
    hit: Hit,
    text: str,
    *,
    html: str | None,
    meta_date,
    is_pdf: bool,
    body_from: _BodyFrom,
    final_url: str | None = None,
    title: str = "",
    sitename: str | None = None,
) -> Page:
    """Apply the length and language rules, then attach a trusted date.

    A short extract is the cookie bar or a paywall, so the snippet replaces
    it. A page that is not English is counted and then dropped: translating
    it would be a model call, and a Hindi page is not evidence we can read.
    The date is resolved even then, so the stat can say the hit had one.
    """
    url_for_date = final_url or hit.url
    published, source = _resolve_date(html, url_for_date, meta_date, hit.published_date)
    # Tavily's extract is not HTML, so htmldate never sees it. The
    # publication line is often still in that text, and a later HTML
    # fetch may be refused. Read it here, before the page is cached.
    if published is None:
        from_text = publication_date_in_text(text)
        if from_text:
            published, source = from_text, "text"
    words = (text or "").split()
    if len(words) > _MAX_ARTICLE_WORDS:
        text = " ".join(words[:_MAX_ARTICLE_WORDS])
        words = words[:_MAX_ARTICLE_WORDS]
    if len(words) < _MIN_ARTICLE_WORDS:
        page = _snippet(hit, "failed", "short-text")
        # The page's date is still the page's date. The snippet only
        # replaces the text we could not read.
        page.published_date = published
        page.date_source = source
        page.is_pdf = is_pdf
        return page
    if not _is_english(text):
        return Page(
            url=hit.url,
            title=title or hit.title,
            text="",
            sitename=sitename,
            published_date=published,
            date_source=source,
            fetched=False,
            fetch_status="non_english",
            failure_reason="non-English",
            is_pdf=is_pdf,
            body_from=body_from,
            query_kinds=list(hit.query_kinds),
        )
    return Page(
        url=hit.url,
        title=title or hit.title,
        text=text.strip(),
        sitename=sitename,
        published_date=published,
        date_source=source,
        fetched=True,
        fetch_status="ok",
        is_pdf=is_pdf,
        body_from=body_from,
        query_kinds=list(hit.query_kinds),
    )


def _circulation(hit: Hit) -> Page:
    published, source = _resolve_date(None, hit.url, None, hit.published_date)
    # No text. A later step that forgot to skip this host must still have
    # nothing to quote. The sighting is the URL, counted by the caller.
    return Page(
        url=hit.url,
        title=hit.title,
        text="",
        published_date=published,
        date_source=source,
        fetched=False,
        fetch_status="circulation",
        body_from="none",
        query_kinds=list(hit.query_kinds),
    )


def _snippet(hit: Hit, status: Literal["failed", "snippet"], reason: str | None) -> Page:
    published, source = _resolve_date(None, hit.url, None, hit.published_date)
    return Page(
        url=hit.url,
        title=hit.title,
        text=hit.content,
        published_date=published,
        date_source=source,
        fetched=False,
        fetch_status=status,
        failure_reason=reason,
        body_from="snippet",
        query_kinds=list(hit.query_kinds),
    )


def _resolve_date(html, url: str, meta_date, tavily_date):
    """First trusted date, in a fixed order.

    htmldate on the HTML, then a date written in the URL path, then
    trafilatura's metadata, then Tavily. A date that is today or later is
    not a publication date: ministry pages stamp "last updated" with the
    day we fetched them, and a future date is a placeholder. Those are
    skipped. If every candidate is that kind of date, the source is
    "untrusted" and the value is empty.
    """
    today = date.today()
    saw_untrusted = False
    clues = []
    if html:
        clues.append(("htmldate", _htmldate(html)))
    clues.append(("url", _url_date(url)))
    clues.append(("trafilatura", _date_text(meta_date)))
    clues.append(("tavily", _date_text(tavily_date)))
    for source, raw in clues:
        parsed = _parse_date(raw)
        if parsed is None:
            continue
        if parsed >= today:
            saw_untrusted = True
            continue
        return parsed.isoformat(), source
    if saw_untrusted:
        return None, "untrusted"
    return None, None


def _htmldate(html: str) -> str | None:
    from htmldate import find_date

    # No url argument. Passing the URL makes htmldate read the path itself
    # and skip the order this function is here to keep. extensive_search
    # walks every word of the page looking for a date; on a long page that
    # is the slow part, and original_date already reads the meta tags.
    found = find_date(
        html,
        original_date=True,
        extensive_search=False,
        max_date=_HTMLDATE_MAX,
        verbose=False,
    )
    return _date_text(found)


def _url_date(url: str) -> str | None:
    path = urlsplit(url or "").path
    match = _URL_YMD.search(path) or _URL_ISO.search(path)
    if match is None:
        return None
    if match.lastindex == 1:
        text = match.group(1)
    else:
        text = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    parsed = _parse_date(text)
    if parsed is None:
        return None
    return parsed.isoformat()


def _parse_date(value) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _date_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _blank(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_english(text: str) -> bool:
    """True when the start of the article is English.

    langdetect is seeded so the same page does not change its mind between
    runs. A sample it cannot classify is kept: dropping it would throw away
    a short English circular because the detector abstained.
    """
    sample = " ".join(text.split())[:500]
    if not sample:
        return False
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException

        DetectorFactory.seed = 0
        return detect(sample) == "en"
    except LangDetectException:
        return True
    except Exception:
        return True


def _clip_article(text: str) -> str:
    """The head of the page. A cached body can be several megabytes."""
    if not text or len(text) <= _MAX_ARTICLE_CHARS:
        return text or ""
    return text[:_MAX_ARTICLE_CHARS]


def _bounded(fn, url: str):
    """Run a download on another thread and stop waiting at the deadline.

    The thread may keep the socket a little longer. The claim does not.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn, url)
        try:
            return future.result(timeout=config.FETCH_TIMEOUT_S + 2)
        except FuturesTimeout as exc:
            raise FetchError("timeout") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _download(url: str) -> tuple[str, bytes, str]:
    # One clock for every hop and every chunk. The socket timeout alone
    # resets each time a byte arrives, so a slow body never ends.
    deadline = time.monotonic() + config.FETCH_TIMEOUT_S
    current = url
    for hop in range(_MAX_REDIRECTS + 1):
        _check_deadline(deadline)
        # Every hop, including the first. A redirect to a private address
        # is the usual way around a check that only looked at the start.
        _refuse_unless_public(current)
        try:
            response = _http_get(current)
        except Exception as exc:
            raise FetchError(_reason_from_exception(exc)) from None
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
            ctype = str(headers.get("Content-Type") or headers.get("content-type") or "")
            kind = _body_kind(ctype, current)
            if kind == "other":
                raise FetchError("non-HTML")
            limit = _MAX_PDF_BYTES if kind == "pdf" else _MAX_HTML_BYTES
            return current, _read_body(response, limit, deadline), ctype
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    raise FetchError("status 310")


def _body_kind(content_type: str, url: str) -> str:
    folded = (content_type or "").casefold()
    if "pdf" in folded or urlsplit(url).path.lower().endswith(".pdf"):
        return "pdf"
    if "html" in folded:
        return "html"
    return "other"


def _is_pdf(content_type: str, url: str) -> bool:
    return _body_kind(content_type, url) == "pdf"


def _refuse_unless_public(url: str, lookup_timeout: float | None = None) -> None:
    parts = urlsplit((url or "").strip())
    if parts.scheme not in {"http", "https"}:
        raise FetchError("blocked-IP")
    host = (parts.hostname or "").strip(".").lower()
    if not host:
        raise FetchError("blocked-IP")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addresses = _addresses(host, port, timeout=lookup_timeout)
    if not addresses:
        raise FetchError("blocked-IP")
    for ip in addresses:
        if not _is_public(ip):
            raise FetchError("blocked-IP")


def _addresses(host: str, port: int, timeout: float | None = None) -> list[str]:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return [host]
    # getaddrinfo has no timeout of its own. A dead resolver would hold
    # every worker, so the wait is capped at the same six seconds as the
    # download. The lookup thread may outlive that wait; the claim does not.
    try:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
            infos = future.result(
                timeout=config.FETCH_TIMEOUT_S if timeout is None else timeout
            )
        except FuturesTimeout as exc:
            raise FetchError("timeout") from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    except socket.gaierror:
        return []
    found = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            found.append(sockaddr[0])
    return found


def _is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    # is_private covers 10/8 and the other RFC1918 ranges. is_link_local
    # covers 169.254.0.0/16, including the cloud metadata address.
    # is_reserved covers ranges that are not routable on the public internet.
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _http_get(url: str):
    """One request, with redirects left to _download so each hop is checked.

    The body is _request so a test can put this function back after the
    offline guard has replaced _http_get and still see the headers.
    """
    return _request(url)


def _request(url: str):
    import requests

    return requests.get(
        url,
        timeout=config.FETCH_TIMEOUT_S,
        allow_redirects=False,
        stream=True,
        headers={"User-Agent": config.FETCH_USER_AGENT},
    )


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise FetchError("timeout")


def _read_body(response, limit: int, deadline: float) -> bytes:
    headers = getattr(response, "headers", {}) or {}
    length = headers.get("Content-Length") or headers.get("content-length")
    if length:
        try:
            if int(length) > limit:
                raise FetchError("too large")
        except ValueError:
            pass
    chunks = []
    total = 0
    iterator = getattr(response, "iter_content", None)
    if iterator is None:
        raw = getattr(response, "content", b"") or b""
        if len(raw) > limit:
            raise FetchError("too large")
        return raw
    for chunk in iterator(chunk_size=65536):
        # Checked between chunks. The read in progress still stops at the
        # socket timeout, so the overrun is one read, not the whole body.
        _check_deadline(deadline)
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise FetchError("too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _charset(content_type: str) -> str | None:
    for part in content_type.split(";"):
        piece = part.strip()
        if piece.lower().startswith("charset="):
            return piece.split("=", 1)[1].strip().strip('"') or None
    return None


def _reason_from_exception(exc: BaseException) -> str:
    if isinstance(exc, FetchError):
        return exc.reason
    name = type(exc).__name__
    if isinstance(exc, (TimeoutError, socket.timeout)) or "Timeout" in name:
        return "timeout"
    return "failed"


def _pdf_text(body: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(body))
    except Exception as exc:
        raise FetchError("non-HTML") from exc
    parts = []
    # The first ten pages. A gazette's ruling is at the front; the rest is
    # the annex, and reading it all is how one PDF stalls the claim.
    for page in list(reader.pages)[:_PDF_PAGES]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def _extract(html: str, url: str) -> dict:
    from trafilatura import bare_extraction

    # with_metadata so the date and the site name come back with the text.
    # The default extract drops them. The date order still prefers htmldate
    # and the URL; this date is the third clue.
    document = bare_extraction(html, url=url, with_metadata=True)
    if document is None:
        raise FetchError("short-text")
    if hasattr(document, "as_dict"):
        data = document.as_dict()
    elif isinstance(document, dict):
        data = document
    else:
        raise FetchError("short-text")
    text = (data.get("text") or "").strip()
    if not text:
        raise FetchError("short-text")
    return data


def _cache_path(url: str):
    # Imported here. investigate imports this module, and it owns
    # canonical_url. A module-level import would be a cycle.
    from misinfo_agent.investigate import canonical_url

    key = canonical_url(url) or url
    # The suffix is the date rule. A cache written before it would replay
    # "today" as the publication date of a 2016 circular.
    digest = hashlib.sha256((key + "|dates-v2").encode("utf-8")).hexdigest()
    return config.PAGE_CACHE_DIR / f"{digest}.json"


def _read_cache(url: str) -> dict | None:
    path = _cache_path(url)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("v") != 2:
        return None
    if payload.get("fetch_status") == "ok" and not payload.get("text"):
        return None
    return payload


def publication_date_in_text(text: str) -> str | None:
    """A publication date written in the article, or None.

    Only a line that says the page was published, or a byline date.
    An event the article mentions ("demonetised on November 8, 2016")
    is not the page's date. Today and the future are stamps, not publication.
    """
    sample = (text or "")[:_TEXT_DATE_WINDOW]
    for pattern in (_PUBLISHED_LINE, _BYLINE_LINE):
        match = pattern.search(sample)
        if match is None:
            continue
        found = _trusted_date(_date_from_match(match))
        if found:
            return found
    return None


def remember_publication(url: str, published: str, source: str) -> None:
    """Write a date onto a cached page that was stored without one.

    The date-only fetch used to attach the date to the passage and leave
    the cache empty. The next run then read the cache, the host returned
    403, and the date was gone. Filling the cache is what makes the date
    survive that refusal.
    """
    path = _cache_path(url)
    if not path.is_file() or not published:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or payload.get("published_date"):
        return
    payload["published_date"] = published
    payload["date_source"] = source
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _date_from_match(match: re.Match) -> str | None:
    groups = match.groups()
    # Published lines have either a month name or an ISO date. Byline
    # lines have only the month name, so the ISO groups are absent.
    month, day, year = groups[0], groups[1], groups[2]
    if month and day and year and not month[0].isdigit():
        number = _MONTH_NUMBER.get(month.rstrip(".").lower())
        if number is None:
            return None
        try:
            return date(int(year), number, int(day)).isoformat()
        except ValueError:
            return None
    if len(groups) >= 6 and groups[3] and groups[4] and groups[5]:
        try:
            return date(int(groups[3]), int(groups[4]), int(groups[5])).isoformat()
        except ValueError:
            return None
    return None


def _from_cache(hit: Hit, payload: dict) -> Page:
    status = payload.get("fetch_status") or "ok"
    if status not in {"ok", "non_english"}:
        return _snippet(hit, "failed", "failed")
    published = payload.get("published_date")
    source = payload.get("date_source")
    # The cached body is the text we already paid Tavily for. A missing
    # date is looked up there before anyone opens the URL again.
    if not published:
        found = publication_date_in_text(str(payload.get("text") or ""))
        if found:
            published, source = found, "text"
            remember_publication(hit.url, published, source)
    return Page(
        url=hit.url,
        title=(payload.get("title") or "").strip() or hit.title,
        text=_clip_article(str(payload.get("text") or "")),
        sitename=_blank(payload.get("sitename")),
        published_date=published,
        date_source=source,
        fetched=bool(payload.get("fetched")),
        fetch_status=status,
        failure_reason=payload.get("failure_reason"),
        is_pdf=bool(payload.get("is_pdf")),
        body_from=payload.get("body_from") or "http",
        query_kinds=list(hit.query_kinds),
    )


def _write_cache(url: str, page: Page) -> None:
    path = _cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "v": 2,
            "title": page.title,
            "text": page.text,
            "sitename": page.sitename or "",
            "published_date": page.published_date,
            "date_source": page.date_source,
            "fetched": page.fetched,
            "fetch_status": page.fetch_status,
            "failure_reason": page.failure_reason,
            "is_pdf": page.is_pdf,
            "body_from": page.body_from,
        },
        ensure_ascii=False,
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)
