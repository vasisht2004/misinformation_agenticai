"""The only module that calls a web search.

Every other box imports this module instead of the Tavily SDK. Two reasons
the call lives here and nowhere else:

- The key stays in one place. A box that built its own client would end up
  printing it in a traceback.
- The caps live here. A box that searched on its own would not see the
  monthly bill or the per-claim budget.

Importing this module does not read the key and does not open a socket.
The client is built on the first real call, so tests can import it with
an empty .env.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import unicodedata
from datetime import datetime, timezone
from typing import Sequence
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

import misinfo_agent.config as config

# One try plus one retry. A second identical failure is the caller's
# problem; looping longer would stall box 7 on a query the engine cannot
# answer.
_BACKOFF_S = 0.5

_USAGE_LOCK = threading.Lock()
_QUOTA_LOCK = threading.Lock()
_USAGE: list[dict] = []
_FAKE = None
_CLIENT = None
# Real searches started for the claim that is open in this process.
# Not written to disk: the monthly file is the bill, and this counter only
# stops one claim from using it up. The caller sets it back to zero when a
# new claim starts.
_CLAIM_SEARCHES = 0

# Sibling of the response cache, not inside it. Wiping cached pages must
# not reset the month. data/ is gitignored.
MONTHLY_COUNT_PATH = config.DATA_DIR / "cache" / "search_monthly_count.json"


class SearchError(Exception):
    """A search failed in a way a box can show.

    The message is the purpose plus a short reason. It is not the query and
    it is not the API key: both would land in the audit row and in logs.
    """

    def __init__(self, purpose: str, reason: str):
        self.purpose = purpose
        self.reason = reason
        super().__init__(f"{purpose}: {reason}")


class SearchQuotaError(SearchError):
    """A cap was hit. Callers stop searching instead of trying the next query.

    Raised for the local monthly cap, the per-claim cap, and a quota error
    from Tavily. None of those is a blip, so none of them is retried.
    cap is "per_claim" or "monthly" when this process enforced the limit,
    and None when the provider reported the quota itself.
    """

    def __init__(self, purpose: str, reason: str, cap: str | None = None):
        super().__init__(purpose, reason)
        if cap in {"per_claim", "monthly"}:
            self.cap = cap
        elif "monthly" in (reason or ""):
            self.cap = "monthly"
        elif "per-claim" in (reason or "") or "per_claim" in (reason or ""):
            self.cap = "per_claim"
        else:
            self.cap = None


class Hit(BaseModel):
    """One result, after it has been checked for shape.

    query_kinds is empty on the way out of search(). gather_candidates fills
    it, because the gateway does not know which line of the plan asked.
    A URL found by two lines keeps both, in plan order.
    """

    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0
    published_date: str | None = None
    query_kinds: list[str] = Field(default_factory=list)
    # Page text Tavily already extracted for this hit. Empty when the
    # search did not return any. Box 7 uses it before opening the URL.
    # It is not a second billed extract: see _tavily_search.
    raw_content: str = ""

    @field_validator("title", "content", "raw_content", mode="before")
    @classmethod
    def _blank_text(cls, value):
        if value is None:
            return ""
        return str(value)

    @field_validator("score", mode="before")
    @classmethod
    def _blank_score(cls, value):
        # A result with no score is still a page. Sorting it last is safer
        # than dropping it: the snippet may be the official notice.
        if value is None or value == "":
            return 0.0
        return value

    @field_validator("published_date", mode="before")
    @classmethod
    def _blank_date(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, (int, float)):
            return str(value)
        return None


def search(
    query: str,
    *,
    purpose: str,
    max_results: int = 5,
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
    search_depth: str = "basic",
    end_date: str | None = None,
) -> list[Hit]:
    """One web search. Returns hits, never the raw provider payload.

    purpose is the audit label ("investigate_neutral", ...), not sent to
    Tavily. include_domains restricts the search; exclude_domains removes
    hosts. When fact-check blocking is on, those hosts are added to the
    exclude list and any hit that still comes back on one of them is
    dropped. A cached or fake answer does not count against either cap.

    end_date is YYYY-MM-DD. Tavily returns results published before that
    day and still returns results with no date. It does not add a credit:
    a basic search stays 1 credit (docs.tavily.com/documentation/api-credits,
    checked 12 Nov 2026). None leaves the date open.
    """
    started = time.perf_counter()
    include, exclude = _effective_domains(include_domains, exclude_domains)
    try:
        hits, from_cache, cache_miss = _run(
            query, purpose, max_results, include, exclude, search_depth, end_date
        )
    except SearchError as exc:
        # The status is recorded here, at the raise, so a later box does
        # not have to guess whether the empty list was a cap or a miss.
        _record(
            purpose, query, 0, started, cached=False, success=False,
            status=failure_status(exc),
        )
        raise
    except Exception as exc:
        # A bug or a library error can quote the request, and the request
        # can quote the key. Wrap it before it reaches the audit.
        _record(
            purpose, query, 0, started, cached=False, success=False, status="error"
        )
        raise SearchError(purpose, _safe(exc, None, query)) from None
    sent = normalise_query_text(query)
    if cache_miss:
        # An empty list, not a cached empty search. Writing it would make
        # the next run think Tavily returned nothing. cache_only_miss is
        # the card status: nothing was read, so the card must not say a
        # source addressed the claim.
        _record(
            purpose, sent or query, 0, started, cached=False, success=True,
            cache_miss=True, status="cache_only_miss",
        )
        return []
    # Belt and braces. The exclude list is already on the request; a hit
    # that ignores it must still not reach the evidence table. Done before
    # the cache write so a replay cannot bring the page back.
    kept = _without_blocked(hits)
    if not from_cache and config.SEARCH_CACHE:
        _write_cache(sent or query, max_results, include, exclude, search_depth, end_date, kept)
    _record(
        purpose,
        sent or query,
        len(kept),
        started,
        cached=from_cache,
        success=True,
        status="ran",
    )
    return kept


def get_search_usage() -> list[dict]:
    """A copy of every search since the last reset.

    A copy, not the live list: the audit and a test both read this, and
    one of them adding a field must not change what the other sees.
    """
    with _USAGE_LOCK:
        return [dict(row) for row in _USAGE]


def reset_search_usage() -> None:
    """Clear the usage rows.

    Does not change the monthly count or the per-claim counter. Those are
    budgets, not logs. Resetting the log must not grant more searches.
    """
    with _USAGE_LOCK:
        _USAGE.clear()


def reset_claim_searches() -> None:
    """Start a claim's search budget over.

    Call this once when a claim starts, before gather_candidates. Do not
    call it between search rounds of the same claim: the cap is for the
    whole claim, including a later search_more.
    """
    global _CLAIM_SEARCHES
    with _QUOTA_LOCK:
        _CLAIM_SEARCHES = 0


def searches_this_month() -> int:
    """Real Tavily calls already made in the current UTC month.

    Cached and fake calls are not in the number. A missing file, or a file
    from an earlier month, is zero: that month is over.
    """
    with _QUOTA_LOCK:
        return _count_unlocked()


def set_fake_search(fn) -> None:
    """Send searches to fn instead of Tavily.

    fn(query, *, max_results, include_domains, exclude_domains, search_depth,
    end_date) returns a list of hit dicts, or a payload shaped like
    Tavily's {"results": [...]}. The return value still goes through Hit
    and is still recorded in usage. While a fake is set, this module does
    not build a client and does not read the API key. A fake does not count
    against either cap. The domains it receives already include the
    fact-check block, when that block is on. end_date is None when the
    caller did not ask for one.
    """
    global _FAKE
    _FAKE = fn


def clear_fake_search() -> None:
    """Return to the real provider."""
    global _FAKE
    _FAKE = None


def site_host(url: str) -> str:
    """Lowercase host without a leading www.

    Fact-check blocking and URL canonicalisation have to agree on what
    'the same site' means. www.boomlive.in and boomlive.in are one
    publication in the archive; the stored list is the bare host.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlsplit(raw).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _run(query, purpose, max_results, include, exclude, search_depth, end_date):
    # Tavily sees the cleaned text, not a lowercased copy. The cache key
    # is the part that folds case. An empty string after stripping is not
    # a search: there is nothing to ask and nothing to store.
    cleaned = normalise_query_text(query)
    if not cleaned:
        return [], False, True
    cached = _read_cache(query, max_results, include, exclude, search_depth, end_date)
    if cached is not None:
        # A hit is still a search, from the audit's point of view.
        # cached=True tells box 16 it did not spend a credit again.
        # A legacy-file hit is included. It is not copied onto the new key.
        return cached, True, False
    # Cache-only is a miss with no provider call. Checked before the fake
    # and before _real so a test can prove Tavily was not asked.
    if cache_only_enabled():
        return [], False, True
    # A fake never reaches the cap or the client. Tests stay offline, and
    # a fake answer must not burn a slot of the monthly quota.
    if _FAKE is not None:
        raw = _FAKE(
            cleaned,
            max_results=max_results,
            include_domains=include or None,
            exclude_domains=exclude or None,
            search_depth=search_depth,
            end_date=end_date,
        )
        return _coerce_hits(raw, purpose), False, False
    return _real(
        cleaned, purpose, max_results, include, exclude, search_depth, end_date
    ), False, False


def _real(query, purpose, max_results, include, exclude, search_depth, end_date):
    # Cap first, then the key, then the client. A full month must not build
    # a client, and a missing key must not count as a call.
    _refuse_if_capped(purpose)
    key = _require_key(purpose)
    client = _client(key)
    last_reason = "the search failed"
    for attempt in (1, 2):
        # Each attempt is its own HTTP call, and Tavily bills each one.
        # Checked again so a retry cannot be the call that goes over the cap.
        _acquire_real_call(purpose)
        try:
            payload = _tavily_search(
                client,
                query,
                max_results=max_results,
                include_domains=include or None,
                exclude_domains=exclude or None,
                search_depth=search_depth,
                end_date=end_date,
            )
        except SearchQuotaError:
            raise
        except SearchError:
            raise
        except _WallTimeout:
            # The raw-content download kept the socket open past the wall
            # clock. Retrying it spends another credit on the same wait.
            raise SearchError(purpose, "timeout") from None
        except Exception as exc:
            if _is_provider_quota(exc):
                raise SearchQuotaError(purpose, _safe(exc, key, query)) from None
            reason = _safe(exc, key, query)
            # Auth is not a blip, and a 400 will not succeed by being asked
            # again. Only a dropped connection or a 5xx gets the one retry.
            if _is_auth(exc) or not _is_transient(exc) or attempt == 2:
                raise SearchError(purpose, reason) from None
            last_reason = reason
            time.sleep(_BACKOFF_S)
            continue
        return _coerce_hits(payload, purpose)
    raise SearchError(purpose, last_reason)


class _WallTimeout(Exception):
    """The provider did not answer before the wall clock, not just a socket blip."""


def _tavily_search(
    client,
    query,
    *,
    max_results,
    include_domains,
    exclude_domains,
    search_depth,
    end_date,
):
    # include_answer would be Tavily's own model writing a conclusion, so
    # it stays off. include_raw_content="text" is the cleaned page text.
    # Checked against https://docs.tavily.com/documentation/api-credits on
    # 11 Nov 2026: a search costs 1 credit at basic depth and 2 at advanced,
    # and that table does not add a charge for raw content. The separate
    # Extract endpoint does cost credits, and this call does not use it.
    # Box 7 downloads a URL only when this text is missing or too short.
    # The SDK timeout resets while bytes are arriving, so a multi-megabyte
    # raw body can run for minutes. Twenty seconds is the wall clock for
    # one attempt. Past that, the query is a miss and the snippet step
    # still has whatever other queries returned.
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    def call():
        return client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=False,
            include_raw_content="text",
            end_date=end_date,
            timeout=20,
        )

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(call)
        try:
            return future.result(timeout=20)
        except FuturesTimeout as exc:
            raise _WallTimeout() from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _coerce_hits(payload, purpose: str) -> list[Hit]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("results", [])
    else:
        raise SearchError(purpose, "search result was not a list")
    if not isinstance(rows, list):
        raise SearchError(purpose, "search result was not a list")
    hits = []
    for item in rows:
        hit = _one_hit(item)
        if hit is not None:
            hits.append(hit)
    return hits


def _one_hit(item) -> Hit | None:
    if isinstance(item, Hit):
        return item
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    if not isinstance(url, str) or not url.strip():
        # A row with no URL cannot be fetched later and cannot be cited.
        # Drop it. The rest of the list may still be usable.
        return None
    try:
        return Hit(
            url=url.strip(),
            title=item.get("title") or "",
            content=item.get("content") or "",
            score=0.0 if item.get("score") is None else item.get("score"),
            published_date=item.get("published_date"),
            raw_content=item.get("raw_content") or "",
        )
    except ValidationError:
        return None


def _effective_domains(include_domains, exclude_domains) -> tuple[list[str], list[str]]:
    include = _domain_list(include_domains)
    exclude = _domain_list(exclude_domains)
    if config.BLOCK_FACTCHECK_SITES:
        # Merged before the cache key is built. A blocked search and an
        # open search of the same words are different requests. The suffixes
        # are what we ask Tavily to skip; the post-filter below is the net
        # for a hit that ignores the request.
        exclude = _domain_list([*exclude, *block_suffixes()])
    return include, exclude


def _domain_list(values) -> list[str]:
    if not values:
        return []
    seen = set()
    cleaned = []
    for value in values:
        host = _clean_domain(value)
        if host and host not in seen:
            seen.add(host)
            cleaned.append(host)
    # Sorted so the same hosts in a different order share a cache file.
    # Tavily does not care about the order.
    cleaned.sort()
    return cleaned


def _clean_domain(value) -> str:
    text = str(value).strip().lower()
    if "://" in text or text.startswith("www."):
        text = site_host(text)
    if text.startswith("www."):
        text = text[4:]
    return text.strip(".")


def block_suffixes() -> list[str]:
    """Hosts to exclude while the fact-check block is on.

    Archive hosts expand to the registrable domain, so a language
    subdomain is the same publication. Extra hosts stay as written:
    factcheck.afp.com must not become afp.com, or the news wire would
    be dropped along with its fact-check desk.
    """
    suffixes = set()
    for host in config.FACTCHECK_DOMAINS:
        suffixes.add(_registrable(host))
    for host in config.EXTRA_FACTCHECK_DOMAINS:
        suffixes.add(_clean_domain(host))
    suffixes.discard("")
    return sorted(suffixes)


def registrable_domain(host: str) -> str:
    """The site, not the subdomain. m.facebook.com and facebook.com match.

    The same helper the fact-check block uses, so "registrable domain"
    means one thing in the block list and in the passage cap.
    """
    return _registrable(host)


def _registrable(host: str) -> str:
    cleaned = _clean_domain(host)
    if not cleaned:
        return ""
    try:
        from tld import get_fld

        # fix_protocol lets a bare host through. get_fld is the registrable
        # domain (youturn.in), not the subdomain the archive happened to store.
        found = get_fld(cleaned, fix_protocol=True)
    except Exception:
        return cleaned
    return (found or cleaned).lower()


def looks_like_factcheck(url: str, title: str = "") -> bool:
    """True when this page is a fact-check, whether or not the block is on.

    Five independent reasons, any one is enough: the host is a fact-check
    publication or its subdomain, the registrable name itself says
    fact-check, the path is a fact-check section, the title says it is a
    fact check, or the page is PIB Fact Check on any host. A normal
    pib.gov.in press release matches none of these. The eval uses this
    while fact-check sites are allowed, to count those pages without
    dropping them.
    """
    host = site_host(url)
    if _host_is_suffix(host, block_suffixes()):
        return True
    # factcheckhub.com is not on the archive list. The name is the desk.
    # Hyphens are removed so fact-check.example matches the same way.
    compact = registrable_domain(host).replace("-", "")
    if "factcheck" in compact or "factchecker" in compact:
        return True
    path = _url_path(url)
    # webqoof is The Quint's fact-check desk. The whole host is already
    # blocked; the path still drops a copy of that section elsewhere.
    if any(token in path for token in ("fact-check", "factcheck", "fact_check", "webqoof")):
        return True
    folded = (title or "").casefold()
    if "fact check" in folded or "fact-check" in folded:
        return True
    return _is_pib_factcheck(host, path, folded)


def factcheck_blocked(url: str, title: str = "") -> bool:
    """True when a blocked-mode search must not keep this hit.

    When the block is off this is always false: a demo, and the allowed
    eval, may read a fact-check on purpose. The page test itself is
    looks_like_factcheck, so the two cannot drift.
    """
    if not config.BLOCK_FACTCHECK_SITES:
        return False
    return looks_like_factcheck(url, title)


def _host_is_suffix(host: str, suffixes) -> bool:
    if not host:
        return False
    for suffix in suffixes:
        # The dot stops notboomlive.in matching boomlive.in.
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _url_path(url: str) -> str:
    raw = (url or "").strip()
    if raw and "://" not in raw:
        raw = "https://" + raw
    return (urlsplit(raw).path or "").casefold()


def _is_pib_factcheck(host: str, path: str, title: str) -> bool:
    # The desk's handle and its name. "pib.gov.in" alone is the press
    # office, and a release that does not call itself a fact check stays.
    compact_host = host.replace("-", "")
    compact_path = path.replace("-", "").replace("_", "")
    compact_title = title.replace(" ", "").replace("-", "")
    return (
        "pibfactcheck" in compact_host
        or "pibfactcheck" in compact_path
        or "pibfactcheck" in compact_title
    )


def _without_blocked(hits: list[Hit]) -> list[Hit]:
    if not config.BLOCK_FACTCHECK_SITES:
        return hits
    return [hit for hit in hits if not factcheck_blocked(hit.url, hit.title)]


def _refuse_if_capped(purpose: str) -> None:
    """Stop before a real call when a cap is already reached.

    Does not increment. The refused call is not a call.
    """
    with _QUOTA_LOCK:
        _raise_if_capped(purpose, _count_unlocked(), _CLAIM_SEARCHES)


def _acquire_real_call(purpose: str) -> None:
    """Count this HTTP attempt, or refuse it if either cap is full."""
    global _CLAIM_SEARCHES
    with _QUOTA_LOCK:
        count = _count_unlocked()
        _raise_if_capped(purpose, count, _CLAIM_SEARCHES)
        _set_unlocked(count + 1)
        _CLAIM_SEARCHES += 1


def _raise_if_capped(purpose: str, count: int, claim_n: int) -> None:
    if count >= config.TAVILY_MONTHLY_MAX:
        raise SearchQuotaError(
            purpose,
            f"monthly search cap reached ({count} calls)",
            cap="monthly",
        )
    if claim_n >= config.MAX_SEARCHES_PER_CLAIM:
        raise SearchQuotaError(
            purpose,
            f"per-claim search cap reached ({claim_n} calls)",
            cap="per_claim",
        )


def cache_only_enabled() -> bool:
    """True when SEARCH_CACHE_ONLY says a miss must not call Tavily.

    The eval sets this so a re-run can replay cached searches. A miss is
    an empty result, not a billed call. Page fetches are a different
    budget and are not gated here.
    """
    raw = os.environ.get("SEARCH_CACHE_ONLY", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _month_key() -> str:
    # Tavily does not document a reset timezone. UTC keeps the bucket the
    # same on every machine. A local month would change the count for a
    # process that moves between IST and UTC near midnight.
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _count_unlocked() -> int:
    try:
        return int(_read_counts().get(_month_key(), 0))
    except (TypeError, ValueError):
        return 0


def _set_unlocked(count: int) -> None:
    # Only this month's key is kept. Last month's count must not survive
    # and make the new month look already full.
    _write_counts({_month_key(): count})


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


def _require_key(purpose: str) -> str:
    _load_env()
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        raise SearchError(purpose, "TAVILY_API_KEY is not set in .env")
    return key


def _load_env() -> None:
    # override=False: a variable already in the process (a test, a shell)
    # wins over the file. A blank TAVILY_API_KEY in the process stays blank,
    # which is how the tests prove a missing key cannot fall through to .env.
    load_dotenv(config.ENV_PATH, override=False)


def _client(key: str):
    """Build the Tavily client once. The key is an argument, never a log line."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    from tavily import TavilyClient

    _CLIENT = TavilyClient(api_key=key)
    return _CLIENT


def _is_provider_quota(exc: BaseException) -> bool:
    try:
        from tavily.errors import UsageLimitExceededError
    except Exception:
        return False
    return isinstance(exc, UsageLimitExceededError)


def _is_auth(exc: BaseException) -> bool:
    try:
        from tavily.errors import ForbiddenError, InvalidAPIKeyError, MissingAPIKeyError
    except Exception:
        return False
    return isinstance(exc, (InvalidAPIKeyError, MissingAPIKeyError, ForbiddenError))


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    try:
        from tavily.errors import TimeoutError as TavilyTimeoutError
    except Exception:
        TavilyTimeoutError = ()
    if isinstance(exc, TavilyTimeoutError):
        return True
    try:
        import requests
    except Exception:
        return False
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
        return code in {408, 500, 502, 503, 504}
    return False


def _safe(exc: BaseException, key: str | None, query: str) -> str:
    text = str(exc)
    if key:
        text = text.replace(key, "[redacted]")
    if query:
        text = text.replace(query, "[query]")
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:180] + "..."
    return text or "the search failed"


# Edge marks only. A comma inside ₹2,000 is the amount, and a period
# before "fake OR" is part of the sentence. Quotes are the straight and
# curly ones people paste from a phone.
_QUERY_EDGE = frozenset(".,!?;:\"'“”‘’«»")


def normalise_query_text(text: str) -> str:
    """The words Tavily is sent. Case stays; the cache key folds it later.

    NFC so the same accented letter is not two strings. Whitespace
    collapses so a line break is not a different search. Leading and
    trailing punctuation is dropped so "total." and "total" are one
    query. Numbers, ₹, and punctuation in the middle stay: "₹2,000" is
    an amount, not a broken token.
    """
    cleaned = unicodedata.normalize("NFC", str(text or ""))
    cleaned = " ".join(cleaned.split())
    while cleaned and cleaned[0] in _QUERY_EDGE:
        cleaned = cleaned[1:].lstrip()
    while cleaned and cleaned[-1] in _QUERY_EDGE:
        cleaned = cleaned[:-1].rstrip()
    return " ".join(cleaned.split())


def _cache_query_key(query: str) -> str:
    """Lowercased cleaned text. Case is not part of the file name.

    Two spellings of one claim must share a cache entry. The provider
    still receives normalise_query_text, which keeps the caller's case.
    """
    return normalise_query_text(query).lower()


def _legacy_query_strings(original: str, cleaned: str) -> list[str]:
    """Strings whose old hashes may already be on disk.

    Files written before this key existed hashed the query exactly as it
    was sent, including a capital and a trailing period. The new name is
    the lowercased cleaned text, so those files would otherwise miss.
    The first-letter capital and a trailing period are the shapes the
    neutral line used. This list is read-only: a hit is not copied onto
    the new key, or a guessed form would become the stored search.
    """
    found: list[str] = []

    def add(text: str) -> None:
        if text and text not in found:
            found.append(text)

    add(original)
    add(cleaned)
    if cleaned:
        capped = cleaned[0].upper() + cleaned[1:]
        add(capped)
        if not cleaned.endswith("."):
            add(cleaned + ".")
            add(capped + ".")
    stored = cleaned.lower()
    return [text for text in found if text != stored]


def _cache_path(query, max_results, include, exclude, search_depth, end_date=None):
    if not config.SEARCH_CACHE:
        return None
    # Separators and sort_keys so two param lists cannot collide by
    # concatenation, and so the key does not depend on dict order.
    # Purpose is not in here: it is an audit label, and the same query
    # asked for two reasons is still the same search.
    # `query` is already the exact string that belongs in the hash. The
    # new key lowercases before it gets here. A legacy read passes the
    # old string through unchanged.
    body = {
        "query": query,
        "max_results": max_results,
        "include_domains": include,
        "exclude_domains": exclude,
        "search_depth": search_depth,
        # Part of the key so a snippet-only cache from before raw
        # content cannot answer a search that asked for the page text.
        "raw_content": "text",
    }
    # Only when a date was asked for. An open search has to keep hitting
    # the caches written before end_date existed. A dated search is a
    # different request: the result set is not the same pages.
    if end_date:
        body["end_date"] = end_date
    raw = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return config.SEARCH_CACHE_DIR / f"{digest}.json"


def _load_cache_file(path):
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["hits"]
        if not isinstance(rows, list):
            return None
        # Validate again. A half-written or hand-edited file is a miss,
        # not a source. The caller then makes a real call.
        return [Hit.model_validate(row) for row in rows]
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError):
        return None


def _read_cache(query, max_results, include, exclude, search_depth, end_date=None):
    cleaned = normalise_query_text(query)
    if not cleaned:
        return None
    # The new key first. A legacy hit must not be written back here:
    # copying it would turn a guess about the old spelling into the
    # stored result for every later case.
    found = _load_cache_file(
        _cache_path(cleaned.lower(), max_results, include, exclude, search_depth, end_date)
    )
    if found is not None:
        return found
    for legacy in _legacy_query_strings(query, cleaned):
        found = _load_cache_file(
            _cache_path(legacy, max_results, include, exclude, search_depth, end_date)
        )
        if found is not None:
            return found
    return None


def _write_cache(query, max_results, include, exclude, search_depth, end_date, hits) -> None:
    stored = _cache_query_key(query)
    if not stored:
        return
    path = _cache_path(stored, max_results, include, exclude, search_depth, end_date)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {"hits": [hit.model_dump(mode="json") for hit in hits]},
        ensure_ascii=False,
    )
    # Write aside and replace, so a crash cannot leave a truncated JSON
    # file that the next call would trust.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


# One label for the whole claim. A single query can be "ran" while a
# later query hits a cap; the rollup below keeps the cap, because that is
# what the card has to explain.
_SEARCH_STATUSES = frozenset({
    "ran",
    "cache_only_miss",
    "monthly_cap",
    "per_claim_cap",
    "error",
    "not_needed",
})


def failure_status(exc: BaseException) -> str:
    """The card status for a search that raised.

    A provider quota has cap None: Tavily reported the bill, which is the
    same monthly limit this process enforces itself. A per-claim stop is
    the only other cap. Anything else is a failed call, not a limit.
    """
    if isinstance(exc, SearchQuotaError):
        if getattr(exc, "cap", None) == "per_claim":
            return "per_claim_cap"
        return "monthly_cap"
    return "error"


def claim_search_status(rows, *, needed: bool = True) -> dict:
    """Roll one claim's usage rows into the status the card shows.

    A cap outranks a query that already returned: the card has to say the
    limit, not only that some page arrived. A cache-only miss is the status
    only when nothing came back from the cache or from Tavily. not_needed
    is for a claim that never asked, and only when this slice is empty of
    real outcomes.
    """
    live = 0
    cached_n = 0
    saw: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        saw.append(status)
        if row.get("cached"):
            cached_n += 1
        elif status == "ran":
            live += 1
    if "monthly_cap" in saw:
        status = "monthly_cap"
    elif "per_claim_cap" in saw:
        status = "per_claim_cap"
    elif "error" in saw and "ran" not in saw:
        status = "error"
    elif "cache_only_miss" in saw and "ran" not in saw:
        status = "cache_only_miss"
    elif "ran" in saw:
        status = "ran"
    elif not needed:
        status = "not_needed"
    else:
        # No row. Queries skipped because they already ran, or a round
        # that had nothing new to ask. That is not a miss and not a cap.
        status = "ran"
    return {
        "search_status": status,
        "searches_live": live,
        "searches_cached": cached_n,
    }


def search_not_needed(state: dict | None) -> bool:
    """True when this claim was never going to call Tavily.

    reuse_prior copies the archive. Other is not a checkable claim. Both
    are not_needed, which the card treats as a finished step rather than
    a search that failed.
    """
    if not isinstance(state, dict):
        return False
    if state.get("checkable") is False:
        return True
    if state.get("route") == "reuse_prior":
        return True
    if state.get("verdict") == "Other":
        return True
    return False


def _stored_counts(state: dict, stats: dict) -> tuple[int, int]:
    def one(value) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    live = state.get("searches_live", None)
    cached_n = state.get("searches_cached", None)
    if live is None:
        live = stats.get("searches_live", 0)
    if cached_n is None:
        cached_n = stats.get("searches_cached", 0)
    return one(live), one(cached_n)


def resolve_search_status(state: dict | None) -> dict:
    """The claim's status. Usage rows win; a saved card is the fallback.

    output() runs after the searches, in the same process, so the usage
    log still has this run. A fixture has no log. It keeps the status that
    was stored with the card, and a reused prior with no log is not_needed.
    """
    state = state if isinstance(state, dict) else {}
    needed = not search_not_needed(state)
    marks = state.get("run_marks") if isinstance(state.get("run_marks"), dict) else {}
    try:
        start = int(marks.get("searches") or 0)
    except (TypeError, ValueError):
        start = 0
    if start < 0:
        start = 0
    rows = get_search_usage()[start:]
    if rows:
        return claim_search_status(rows, needed=needed)
    stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    existing = state.get("search_status")
    if existing not in _SEARCH_STATUSES:
        existing = stats.get("search_status")
    if existing in _SEARCH_STATUSES:
        live, cached_n = _stored_counts(state, stats)
        return {
            "search_status": existing,
            "searches_live": live,
            "searches_cached": cached_n,
        }
    if not needed:
        return {"search_status": "not_needed", "searches_live": 0, "searches_cached": 0}
    capped = state.get("search_capped") or stats.get("search_capped")
    if capped == "monthly":
        return {"search_status": "monthly_cap", "searches_live": 0, "searches_cached": 0}
    if capped == "per_claim":
        return {"search_status": "per_claim_cap", "searches_live": 0, "searches_cached": 0}
    if stats or state.get("evidence"):
        try:
            live = max(0, int(stats.get("searches") or 0))
        except (TypeError, ValueError):
            live = 0
        return {"search_status": "ran", "searches_live": live, "searches_cached": 0}
    return {"search_status": "not_needed", "searches_live": 0, "searches_cached": 0}


def _record(purpose, query, n_results, started, *, cached, success, cache_miss=False, status="ran") -> None:
    row = {
        "purpose": purpose,
        "query": query,
        "n_results": n_results,
        "latency_s": round(time.perf_counter() - started, 6),
        "cached": cached,
        "success": success,
        # A cache-only miss. cached stays false: nothing was stored.
        "cache_miss": bool(cache_miss),
        # ran, cache_only_miss, monthly_cap, per_claim_cap, or error.
        # Set at the return or the raise, not inferred later.
        "status": status if status in _SEARCH_STATUSES else "error",
    }
    with _USAGE_LOCK:
        _USAGE.append(row)
