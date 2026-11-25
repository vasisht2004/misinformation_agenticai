"""Where an event photograph appeared before this claim.

Called from investigate for a photograph, not for a screenshot of a
forward. Vision (vision.py) says which pages contain the file. This
module fetches those pages with the same guard, cache, and date logic
as a text source, then asks once whether each page presents the file
as the same event or a different one. The gate reads the result. This
module does not pick the verdict.

A screenshot of a message stays on the OCR path. Forty words of OCR
and no caption is a paragraph of a forward, not a sign in a photograph,
so web detection is not spent on a picture of letters.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import misinfo_agent.config as config
from misinfo_agent.llm import LLMError, LLMQuotaError, call_structured
from misinfo_agent.search import Hit, factcheck_blocked, site_host
from misinfo_agent.vision import detect

# A forwarded message, not a placard. At this length the OCR is the claim
# and the existing text path is the one that checks it.
_SCREENSHOT_WORDS = 40

# The same sources fetch.py keeps when a publication day survived.
# "untrusted" is a stamp of today, not a day the page was published.
# linkedin_id is the activity clock, decoded without opening the post.
# wayback is an existed_by day from the earliest public capture, not a
# byline. It is trusted because the archive recorded the URL, and the
# caller only keeps it when the page has no trusted day or the capture
# is earlier. It is never allowed to move a day forward.
_TRUSTED_DATE = {"htmldate", "url", "trafilatura", "tavily", "page", "text", "linkedin_id", "wayback"}
# A partial is only "this file, long ago" when the date came from the
# page itself or from that capture. A missing source is not that.
# trafilatura, the page, and the article text are the metadata the
# comparison calls "metadata".
_PARTIAL_TRUSTED = {"htmldate", "url", "trafilatura", "page", "text", "linkedin_id", "wayback"}
_MONTH_YEAR = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_YEAR_MONTH = re.compile(
    r"\b((?:19|20)\d{2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_MONTH_NUMBER = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


class _Mismatch(BaseModel):
    detail: str = ""
    observed: str = ""
    expected: str = ""


class ImageCheck(BaseModel):
    """What the photograph shows, compared with the caption. Not a verdict."""

    consistent: Literal["yes", "no", "unclear"] = "unclear"
    mismatches: list[_Mismatch] = Field(default_factory=list)
    place_clues: list[str] = Field(default_factory=list)
    time_clues: list[str] = Field(default_factory=list)


class _Judgement(BaseModel):
    passage_id: str
    stance: Literal["supports", "refutes", "neutral"] = "neutral"
    quote: str = ""


class ProvenanceBatch(BaseModel):
    """One stance for every page. A second call would bill the same picture twice."""

    judgements: list[_Judgement] = Field(default_factory=list)


def is_event_photo(state: dict) -> bool:
    """True when this image is a photograph to reverse-search.

    input_type image is not enough. A screenshot of a WhatsApp forward
    has a paragraph of OCR and no caption: that text is the claim, and
    Vision would search a picture of letters. A photograph has a caption,
    or almost no readable text. The OCR itself still ran in ingest.
    """
    if str(state.get("input_type") or "") != "image":
        return False
    if not str(state.get("image_path") or "").strip():
        return False
    caption = str(state.get("caption") or "").strip()
    words = str(state.get("ocr_text") or "").split()
    if len(words) >= _SCREENSHOT_WORDS and not caption:
        return False
    return True


def names_checkable_event(state: dict) -> bool:
    """The caption names something a text search could look up.

    Extract already decided checkable. An empty claim, or a photo with
    no words, is not an event: Tavily is not asked, and provenance still
    runs so first_seen can be filled.
    """
    if state.get("checkable") is False:
        return False
    return bool(str(state.get("claim") or "").strip())


def picture_search_plan(state: dict) -> str:
    """search, not_event, capped, or off. Only a photograph can skip Tavily.

    A text claim always searches. An image does not, when IMAGE_TEXT_SEARCH
    is off: that is the eval switch that keeps the monthly budget for
    provenance. A photograph otherwise searches only when the claim names
    a checkable event and the monthly cap still has room. The cap is read,
    not spent: a skip does not increment the counter.
    """
    if str(state.get("input_type") or "") == "image" and not config.IMAGE_TEXT_SEARCH:
        return "off"
    if not is_event_photo(state):
        return "search"
    if not names_checkable_event(state):
        return "not_event"
    from misinfo_agent.search import searches_this_month

    if searches_this_month() >= config.TAVILY_MONTHLY_MAX:
        return "capped"
    return "search"


def investigate_picture(state: dict, *, id_start: int = 1) -> dict:
    """Provenance, consistency, and evidence rows a draft may cite as [E#].

    Does not raise for a missing Vision key, a full Vision month, or a
    failed observation. A Gemini quota error still propagates: retrying
    it is how the free tier gets exhausted. Text search is the caller's
    decision; this function never calls Tavily.
    """
    found = detect(str(state.get("image_path") or ""))
    provenance, evidence = _provenance(state, found, id_start=id_start)
    consistency = check_consistency(state)
    return {
        "image_provenance": provenance,
        "image_consistency": consistency,
        "evidence": evidence,
    }


def check_consistency(state: dict) -> dict | None:
    """One Gemini call. None when there is no caption and no claim.

    The observation is stored. It does not choose a verdict. Assess adds
    the caveat and the review flag when consistent is no.
    """
    if not is_event_photo(state):
        return None
    caption = str(state.get("caption") or "").strip()
    claim = str(state.get("claim") or "").strip()
    # Nothing to be inconsistent with. Calling the model would spend a
    # quota slot to learn that the caption is empty.
    if not caption and not claim:
        return None
    path = str(state.get("image_path") or "").strip()
    try:
        image = Path(path).read_bytes() if path else b""
    except OSError:
        image = b""
    if not image:
        return None
    template = (config.PROMPTS_DIR / "image_check_v1.txt").read_text(encoding="utf-8")
    prompt = (
        template.replace("__CAPTION__", caption)
        .replace("__VISUAL__", str(state.get("visual_description") or ""))
        .replace("__CLAIM__", claim)
    )
    try:
        parsed = call_structured(
            prompt,
            ImageCheck,
            purpose="image_check",
            images=[image],
            prompt_version="image_check_v1",
        )
    except LLMQuotaError:
        raise
    except LLMError:
        # A failed observation is unclear, not a contradiction, and not
        # a reason to stop the claim.
        return _consistency_dict(ImageCheck())
    return _consistency_dict(parsed)


def _consistency_dict(parsed: ImageCheck) -> dict:
    mismatches = []
    for item in parsed.mismatches:
        mismatches.append(
            {
                "detail": " ".join(item.detail.split()),
                "observed": " ".join(item.observed.split()),
                "expected": " ".join(item.expected.split()),
            }
        )
    return {
        "consistent": parsed.consistent,
        "mismatches": mismatches,
        "place_clues": [_phrase(item) for item in parsed.place_clues if _phrase(item)],
        "time_clues": [_phrase(item) for item in parsed.time_clues if _phrase(item)],
    }


def _phrase(text: str) -> str:
    return " ".join(str(text).split())


def attach_cached_provenance(state: dict) -> dict:
    """The photo's history when image memory reused the verdict.

    Reuse skips investigate, which is the only box that used to call
    Vision, so the card lost the timeline and the "online since" line.
    The cache already holds the pages. Reading it does not spend a call.
    A miss stays empty: this path must not open a socket to Vision.
    Stance is not asked either. The verdict is the stored one, and a
    new reading of the pages must not reopen it. Dates still come from
    the page cache, and the date rule still labels a full match.
    """
    if str(state.get("input_type") or "") != "image":
        return {}
    path = str(state.get("image_path") or "").strip()
    if not path:
        return {}
    found = detect(path, cache_only=True)
    status = str(found.get("vision_status") or "cache_only_miss")
    best_guess = [str(item) for item in (found.get("best_guess") or []) if str(item).strip()]
    if status != "cached":
        return {"image_provenance": _provenance_dict(status, [], None, best_guess)}
    provenance, _evidence = _provenance(state, found, id_start=1, label_stance=False)
    return {"image_provenance": provenance}


def _provenance(state: dict, found: dict, *, id_start: int, label_stance: bool = True) -> tuple[dict, list]:
    status = str(found.get("vision_status") or "error")
    best_guess = [str(item) for item in (found.get("best_guess") or []) if str(item).strip()]
    if status not in {"ran", "cached"}:
        return _provenance_dict(status, [], None, best_guess), []
    chosen = _choose_pages(found.get("pages") or [])
    fetched = _fetch_pages(chosen)
    # The capture can only move a day backward, and only before the
    # stance call. _mark_earlier then reads the day that survived.
    apply_wayback(fetched)
    # An earlier trusted date is the refute. The stance call is not asked
    # about those pages: the model cannot talk a date forward.
    _mark_earlier(state, fetched)
    if label_stance:
        labelled = _stance(state, fetched)
    else:
        for page in fetched:
            page.setdefault("stance", "neutral")
            page.setdefault("quote", "")
        labelled = fetched
    matches, evidence = _rows(state, labelled, id_start=id_start)
    return _provenance_dict(status, matches, _first_seen(matches), best_guess), evidence


def _provenance_dict(status: str, matches: list, first_seen, best_guess: list) -> dict:
    return {
        "vision_status": status,
        "matches": matches,
        "first_seen": first_seen,
        "best_guess": best_guess,
    }


def _choose_pages(pages: list) -> list[dict]:
    """Up to 12 pages, full matches first. Fact-check hosts drop when the block is on.

    The block is the same one text search uses, so an eval that must not
    read the fact-check it is scored against does not read it as the
    place the photograph came from either.
    """
    rank = {"full": 0, "partial": 1, "page": 2}
    ordered = sorted(
        (row for row in pages if isinstance(row, dict) and row.get("url")),
        key=lambda row: (rank.get(str(row.get("match") or "page"), 2), str(row.get("url"))),
    )
    kept = []
    seen = set()
    from misinfo_agent.investigate import canonical_url

    for row in ordered:
        url = str(row.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        title = str(row.get("title") or "")
        if factcheck_blocked(url, title):
            continue
        key = canonical_url(url)
        if key in seen:
            continue
        seen.add(key)
        kind = str(row.get("match") or "page")
        if kind not in rank:
            kind = "page"
        # File URLs travel with a full match only. apply_wayback asks
        # the archive about them. A partial's file is a different photo.
        images = []
        if kind == "full":
            for item in row.get("image_urls") or []:
                text = str(item or "").strip()
                if text.lower().startswith(("http://", "https://")) and text not in images:
                    images.append(text)
        kept.append({"url": url, "title": title, "match": kind, "image_urls": images})
        if len(kept) >= 12:
            break
    return kept


def _fetch_pages(pages: list[dict]) -> list[dict]:
    """fetch_page, so the SSRF guard, the page cache, and the date logic apply.

    A private address is refused inside fetch and comes back as blocked-IP.
    It is dropped here: a URL that was never fetched is not a source.
    The lead of the article is the passage. Trafilatura does not keep
    where the image sat in the HTML, so the description a reader sees
    with the photo is the opening of the article.
    """
    from misinfo_agent.fetch import fetch_page

    rows = []
    for page in pages:
        fetched = fetch_page(Hit(url=page["url"], title=page["title"], content=""))
        if str(getattr(fetched, "failure_reason", "") or "") == "blocked-IP":
            continue
        text = _lead(getattr(fetched, "text", "") or "")
        published = getattr(fetched, "published_date", None) or None
        rows.append(
            {
                "url": page["url"],
                "title": page["title"] or (getattr(fetched, "title", "") or ""),
                "match": page["match"],
                "text": text,
                "date": published,
                "date_source": getattr(fetched, "date_source", None),
                "host": site_host(page["url"]),
                "image_urls": list(page.get("image_urls") or []),
            }
        )
    return rows


def _lead(text: str, limit: int = 120) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    return " ".join(words[:limit])


def _stance(state: dict, pages: list[dict]) -> list[dict]:
    """One call for every page that has a passage. No claim means no call.

    Without a claim there is nothing for "same event" to mean. The pages
    stay neutral and their dates can still fill first_seen.
    """
    claim = str(state.get("claim") or state.get("caption") or "").strip()
    # dated_earlier pages already have their stance. Sending them would
    # spend a call whose answer this function is not allowed to keep.
    usable = [page for page in pages if page.get("text") and not page.get("dated_earlier")]
    judgements = {}
    if claim and usable:
        from misinfo_agent.investigate import Passage, _quote_in_passage

        passages = []
        for index, page in enumerate(usable, start=1):
            page["passage_id"] = f"P{index}"
            passages.append(
                Passage(
                    passage_id=page["passage_id"],
                    url=page["url"],
                    netloc=page["host"],
                    title=page["title"],
                    text=page["text"],
                    cos=0.0,
                    score=0.0,
                    published_date=page["date"],
                    fetched=True,
                )
            )
        try:
            batch = _judge(state, passages)
        except LLMQuotaError:
            raise
        except LLMError:
            batch = None
        if batch is not None:
            by_id = {item.passage_id: item for item in batch.judgements}
            for page in usable:
                item = by_id.get(page["passage_id"])
                stance = item.stance if item is not None else "neutral"
                quote = " ".join((item.quote if item is not None else "").split())
                # A quote that is not in the passage was not copied. It
                # cannot carry a stance: the label falls back to neutral.
                if stance in {"supports", "refutes"} and quote and _quote_in_passage(quote, page["text"]):
                    page["stance"] = stance
                    page["quote"] = quote
                else:
                    page["stance"] = "neutral"
                    page["quote"] = ""
                judgements[page["url"]] = True
    for page in pages:
        if page.get("dated_earlier"):
            continue
        page.setdefault("stance", "neutral")
        page.setdefault("quote", "")
    return pages


def _judge(state: dict, passages: list) -> ProvenanceBatch:
    template = (config.PROMPTS_DIR / "provenance_v1.txt").read_text(encoding="utf-8")
    blocks = []
    for passage in passages:
        blocks.append(
            f"{passage.passage_id}  url={passage.url}  date={passage.published_date or '-'}\n{passage.text}"
        )
    prompt = (
        template.replace("__CLAIM__", str(state.get("claim") or ""))
        .replace("__CAPTION__", str(state.get("caption") or ""))
        .replace("__CLAIM_DATE__", str(state.get("claim_date") or ""))
        .replace("__PASSAGES__", "\n\n".join(blocks))
    )
    return call_structured(
        prompt,
        ProvenanceBatch,
        purpose="provenance",
        prompt_version="provenance_v1",
    )


def _rows(state: dict, pages: list[dict], *, id_start: int) -> tuple[list, list]:
    """Tiers and MinHash groups the same way a text passage gets them.

    Supports and refutes become evidence rows so a draft can cite them
    as [E#]. They are marked provenance so the text gate does not count
    them as a news search. Neutrals stay on the provenance list, because
    the card shows every page, and they are not evidence rows.
    """
    from misinfo_agent.investigate import Passage, assign_groups, credibility_tier

    passages = []
    for index, page in enumerate(pages, start=1):
        page["passage_id"] = page.get("passage_id") or f"P{index}"
        passages.append(
            Passage(
                passage_id=page["passage_id"],
                url=page["url"],
                netloc=page["host"],
                title=page["title"],
                text=page["text"] or page["title"] or page["url"],
                cos=0.0,
                score=0.0,
                published_date=page["date"],
                fetched=bool(page["text"]),
            )
        )
    groups = assign_groups(passages) if passages else {}
    matches = []
    evidence = []
    number = id_start
    for page in pages:
        tier = credibility_tier(page["url"])
        group_id = groups.get(page["passage_id"], "")
        match = {
            "url": page["url"],
            "host": page["host"],
            "tier": tier,
            "date": page["date"] or None,
            "match": page["match"],
            "stance": page["stance"],
            "quote": page["quote"],
            "group_id": group_id,
            "date_source": page.get("date_source"),
        }
        matches.append(match)
        # A partial or a page hit is a similar picture, or a page that
        # contained one. It stays on the card. It is not the same file,
        # so it is not a source the gate can cite.
        if page["match"] != "full":
            continue
        if page["stance"] not in {"supports", "refutes"}:
            continue
        evidence.append(
            {
                "id": f"E{number}",
                "url": page["url"],
                "netloc": page["host"],
                "title": page["title"],
                "tier": tier,
                "group_id": group_id,
                "stance": page["stance"],
                "quote": page["quote"],
                "display_quote": page["quote"],
                "published_date": page["date"],
                "date_source": page.get("date_source"),
                "fetched": bool(page["text"]),
                "snippet_only": not bool(page["text"]),
                "atomic_id": "A1",
                # The text gate skips these. The photo rules read
                # image_provenance, where the match type and the date live.
                "provenance": True,
                "score": 0.0,
                "cos": 0.0,
            }
        )
        number += 1
    return matches, evidence


def claim_event_date(state: dict) -> date | None:
    """The first day of the event the caption names, else claim_date.

    The caption is the claim about when the photograph was taken. "August
    2026" is that month, "yesterday" is the day before claim_date, and
    "24 November 2026" is that day. Today is not a stand-in for any of
    those: a check run in November must not treat a September page as
    older than an August flood. claim_date is used only when the caption
    and the claim text name no time at all. The returned day is the start
    of the range. Earlier means before this start, minus the margin.
    """
    window = claim_event_window(state)
    if window is None:
        return None
    return window[0]


def claim_event_window(state: dict) -> tuple[date, date] | None:
    """(start, end) inclusive, from the caption, then the claim, then claim_date."""
    for key in ("caption", "claim"):
        window = _window_in_text(str(state.get(key) or ""), state.get("claim_date"))
        if window is not None:
            return window
    day = _iso(state.get("claim_date"))
    if day is None:
        return None
    return day, day


def _window_in_text(text: str, claim_date) -> tuple[date, date] | None:
    """A full day, else a month, else yesterday. None when the text has no time."""
    days = _complete_dates(text)
    if days:
        day = max(days)
        return day, day
    months = _month_spans(text)
    if months:
        start = min(item[0] for item in months)
        end = max(item[1] for item in months)
        return start, end
    if _YESTERDAY.search(text or ""):
        anchor = _iso(claim_date)
        if anchor is not None:
            day = anchor - timedelta(days=1)
            return day, day
    return None


def _month_spans(text: str) -> list[tuple[date, date]]:
    """'August 2026' is the 1st through the last day of that month."""
    found = []
    seen = set()
    for match in list(_MONTH_YEAR.finditer(text or "")) + list(_YEAR_MONTH.finditer(text or "")):
        if match.re is _MONTH_YEAR:
            name, year_text = match.group(1), match.group(2)
        else:
            year_text, name = match.group(1), match.group(2)
        month = _MONTH_NUMBER.get(name.lower())
        if month is None:
            continue
        year = int(year_text)
        key = (year, month)
        if key in seen:
            continue
        seen.add(key)
        last = calendar.monthrange(year, month)[1]
        try:
            found.append((date(year, month, 1), date(year, month, last)))
        except ValueError:
            continue
    return found


def _extracted_event_date(state: dict) -> date | None:
    entities = state.get("entities") if isinstance(state.get("entities"), dict) else {}
    pieces = [str(item).strip() for item in (entities.get("dates") or []) if str(item).strip()]
    clues = str(state.get("time_clues") or "").strip()
    if clues:
        pieces.append(clues)
    if not pieces:
        return None
    extracted: list[date] = []
    for piece in pieces:
        extracted.extend(_complete_dates(piece))
    if not extracted:
        return None
    stated: list[date] = []
    for key in ("caption", "claim"):
        stated.extend(_complete_dates(str(state.get(key) or "")))
    agreed = [day for day in stated if day in set(extracted)]
    if agreed:
        return max(agreed)
    # Extract found a date and the caption states one, but not the same
    # string. The caption is the event the forward names.
    if stated:
        return max(stated)
    return max(extracted)


def _complete_dates(text: str) -> list[date]:
    """Full days only. A bare year is not an event we can place a page before."""
    if not text.strip():
        return []
    from misinfo_agent.prior_match import _extract_dates

    found, _rest = _extract_dates(text)
    days = []
    for item in found:
        year, month, day = item.get("year"), item.get("month"), item.get("day")
        if not year or not month or not day:
            continue
        try:
            days.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return days


def date_is_trusted(source, value) -> bool:
    """True when this day can date the file.

    A missing source with an ISO day is a date a person put on the row.
    "untrusted" is today or the future. Any other named source has to be
    one fetch.py actually keeps.
    """
    if source == "untrusted":
        return False
    if source and source not in _TRUSTED_DATE:
        return False
    return _iso(value) is not None


def is_dated_earlier(match: dict, event: date | None) -> bool:
    """A match whose trusted day is before the event start, under PHOTO_RULE.

    event is the start of the caption's range. The margin keeps a story
    from the day before the event in this event. A page match never
    dates the file. v0 ignores partials. v1 counts a partial the same
    way as a full match. v2 and v3 count a partial only when its date is
    trusted and at least a year before the event starts: a gallery page
    stamped this year is not an earlier copy of the photograph.
    """
    if event is None or not isinstance(match, dict):
        return False
    kind = str(match.get("match") or "")
    if kind == "page":
        return False
    if kind == "partial" and not _partial_may_date(match, event):
        return False
    if kind not in {"full", "partial"}:
        return False
    if kind == "full" and not date_is_trusted(match.get("date_source"), match.get("date")):
        return False
    parsed = _iso(match.get("date"))
    if parsed is None:
        return False
    if kind == "partial" and config.PHOTO_RULE in {"v2", "v3"}:
        return (event - parsed).days >= config.PHOTO_PARTIAL_MIN_DAYS
    return parsed < event - timedelta(days=config.PHOTO_EARLIER_DAYS)


def photo_match_counts(match: dict, event: date | None) -> bool:
    """True when this page may vote on the photograph under PHOTO_RULE.

    A page hit never votes. v0 is full matches only. v1 lets every
    partial vote. v2 and v3 let a partial vote only when it is old
    enough and the date is trusted, so a stance on a lookalike from
    this year cannot call the caption out of context.
    """
    if not isinstance(match, dict):
        return False
    kind = str(match.get("match") or "")
    if kind == "full":
        return True
    if kind != "partial":
        return False
    rule = str(config.PHOTO_RULE or "v0")
    if rule == "v0":
        return False
    if rule == "v1":
        return True
    if event is None:
        return False
    return _partial_may_date(match, event)


def _partial_may_date(match: dict, event: date) -> bool:
    """Whether this partial is allowed to carry a date under the active rule."""
    rule = str(config.PHOTO_RULE or "v0")
    if rule == "v0":
        return False
    if rule == "v1":
        return date_is_trusted(match.get("date_source"), match.get("date"))
    source = match.get("date_source")
    if source not in _PARTIAL_TRUSTED:
        return False
    parsed = _iso(match.get("date"))
    if parsed is None:
        return False
    return (event - parsed).days >= config.PHOTO_PARTIAL_MIN_DAYS


def earlier_quote(page_date: date, event: date) -> str:
    return (
        f"Image on this page dated {page_date.isoformat()}, "
        f"before the claimed event ({event.isoformat()})."
    )


def apply_wayback(pages: list[dict]) -> None:
    """Set an existed_by day from the earliest Wayback capture.

    Full and partial page URLs are asked. A full match also asks about
    the image-file URLs Vision listed on that page: the file can have
    been saved before the article. A page-level hit is not asked. It is
    not this photograph.

    The capture replaces the page's day only when that day is missing or
    untrusted, or when the capture is strictly earlier. A later capture
    is ignored. Wayback must not make a known day later, and an equal
    day keeps the page's own source: the capture did not teach an
    earlier date.
    """
    from misinfo_agent import wayback

    if not wayback.ENABLED:
        return
    for page in pages:
        if not isinstance(page, dict):
            continue
        kind = str(page.get("match") or "")
        if kind not in {"full", "partial"}:
            continue
        # Partials are the lookalikes. The re-run turns this off when a
        # gold-True photograph was called false on the strength of one.
        if kind == "partial" and not wayback.DATE_PARTIALS:
            continue
        urls = []
        page_url = str(page.get("url") or "").strip()
        if page_url:
            urls.append(page_url)
        if kind == "full":
            for item in page.get("image_urls") or []:
                text = str(item or "").strip()
                if text and text not in urls:
                    urls.append(text)
        found = wayback.earliest(urls)
        if found is None:
            continue
        own = None
        if date_is_trusted(page.get("date_source"), page.get("date")):
            own = _iso(page.get("date"))
        if own is not None and found >= own:
            continue
        page["date"] = found.isoformat()
        page["date_source"] = "wayback"


def _mark_earlier(state: dict, pages: list[dict]) -> None:
    """Set the refute on pages the date already decides.

    No stance call. The quote is the two dates, not a sentence copied
    from the article, so the quote-in-passage check does not apply.
    """
    event = claim_event_date(state)
    for page in pages:
        if not is_dated_earlier(page, event):
            continue
        parsed = _iso(page.get("date"))
        page["dated_earlier"] = True
        page["stance"] = "refutes"
        page["quote"] = earlier_quote(parsed, event)


def _first_seen(matches: list[dict]) -> dict | None:
    """Earliest trusted date among full matches.

    A partial match is a similar picture. A page match only says the
    URL contained a picture. Neither dates this file. The earliest
    trusted day on a full match is when we can first place it.
    """
    dated = []
    for match in matches:
        if match.get("match") != "full":
            continue
        if not date_is_trusted(match.get("date_source"), match.get("date")):
            continue
        parsed = _iso(match.get("date"))
        if parsed is None:
            continue
        dated.append((parsed, match))
    if not dated:
        return None
    dated.sort(key=lambda item: (item[0], int(item[1].get("tier") or 4), item[1].get("url") or ""))
    parsed, match = dated[0]
    return {
        "date": parsed.isoformat(),
        "url": match.get("url") or "",
        "host": match.get("host") or "",
        "tier": match.get("tier"),
        # wayback when the capture is what placed the file. The card's
        # year is this date. The source is how a reader tells a byline
        # from an archive capture.
        "date_source": match.get("date_source"),
    }


def _iso(value) -> date | None:
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
