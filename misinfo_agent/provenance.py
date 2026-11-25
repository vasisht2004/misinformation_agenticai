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
_TRUSTED_DATE = {"htmldate", "url", "trafilatura", "tavily", "page", "text", "linkedin_id"}


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


def _provenance(state: dict, found: dict, *, id_start: int) -> tuple[dict, list]:
    status = str(found.get("vision_status") or "error")
    best_guess = [str(item) for item in (found.get("best_guess") or []) if str(item).strip()]
    if status not in {"ran", "cached"}:
        return _provenance_dict(status, [], None, best_guess), []
    chosen = _choose_pages(found.get("pages") or [])
    fetched = _fetch_pages(chosen)
    # An earlier trusted date is the refute. The stance call is not asked
    # about those pages: the model cannot talk a date forward.
    _mark_earlier(state, fetched)
    labelled = _stance(state, fetched)
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
        kept.append({"url": url, "title": title, "match": kind})
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
    """The day the caption says the photograph is of, else claim_date.

    Extract copies dates into entities.dates, and for an image into
    time_clues. That is "extract found one". The day used is one stated
    in the caption or the claim, because a date the model added is not
    the event. When extract found none, the forward's claim_date is the
    day, even if the caption happens to contain a date we did not record.
    """
    stated = _extracted_event_date(state)
    if stated is not None:
        return stated
    return _iso(state.get("claim_date"))


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
    """A full or partial file whose trusted day is before the event.

    A "page" match only says the URL contained the picture somewhere.
    It does not date the file. The margin is the same one the gate uses,
    so a story that ran the day before the forward is still this event.
    """
    if event is None or not isinstance(match, dict):
        return False
    if str(match.get("match") or "") not in {"full", "partial"}:
        return False
    if not date_is_trusted(match.get("date_source"), match.get("date")):
        return False
    parsed = _iso(match.get("date"))
    if parsed is None:
        return False
    return parsed < event - timedelta(days=config.PHOTO_EARLIER_DAYS)


def earlier_quote(page_date: date, event: date) -> str:
    return (
        f"Image on this page dated {page_date.isoformat()}, "
        f"before the claimed event ({event.isoformat()})."
    )


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
    """Earliest trusted date among full and partial matches.

    A "page" match only says the file appeared on that URL. It is not
    enough to date the photograph. A full or partial match is the file
    itself, and the earliest such date is when we can first place it.
    """
    dated = []
    for match in matches:
        if match.get("match") not in {"full", "partial"}:
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
    }


def _iso(value) -> date | None:
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
