"""Box 6: decide whether a past fact-check can be reused.

Thresholds propose exact / similar / none from the best dense cosine.
They do not get the last word. A ₹500 claim must not reuse a ₹2000
fact-check just because the sentences are otherwise the same, and a status
("approved", "banned", "invalid") goes stale. The model is not asked.
llm_context_check is the hook for a later call, and it is not called:
a model opinion here would be an LLM use outside llm.py, and the route
would no longer be a rule.
"""

from __future__ import annotations

import pickle
import re
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel

from misinfo_agent.config import (
    BM25_PATH,
    COMMUNITY_TERMS,
    LLM_CONTEXT_CHECK,
    STATUS_KEYWORDS,
    STALE_DAYS,
    TAU_HIGH,
    TAU_LOW,
)
from misinfo_agent.llm import call_structured

# Word-similarity floor for the paired-term rule. A missing key term whose
# nearest word in the other claim is below this is a different subject.
# 0.60 does not separate "cells" from "kills" (cosine 0.62), so the lemon
# sentence would reuse a cancer fact-check. Stage D3 grids 0.40–0.70 and
# the operating value is the best floor that still blocks that pair.
# It is not TAU_HIGH. Callers pass another floor without editing this line.
KEYTERM_SIM = 0.65
from misinfo_agent.memory import bm25_token_pairs, strip_rupee_amounts

# Months are capitalised and are not names. Leaving "December" in the name
# set makes "1 December" and "30 September" look like a place clash on top
# of the date clash the number rule already reports. "Forward" and "Video"
# are how a WhatsApp message starts, not a person.
_NOT_A_NAME = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "forward", "please", "share", "video", "image", "photo", "watch",
        "breaking", "shocking",
    }
)
_CAPITAL = re.compile(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]{2,}\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
# The optional fraction is part of the value. "9.2" is one number; a bare
# \d pattern would store 9 and 2 and call them a clash.
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
# "9.2%" and "2 percent" are one percent, not the digits inside them.
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_STATUS = re.compile(
    r"\b(?:"
    + "|".join(re.escape(word) for word in sorted(STATUS_KEYWORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# Not subjects. "Forward to all groups" and a bare "can" are how a message
# is packaged. They stay in the BM25 index (changing that would rebuild it)
# but they are not the key term box 6 quotes in a reason.
_NOT_A_SUBJECT = frozenset(
    {
        "forward", "please", "share", "group", "whatsapp", "urgent",
        "immediately", "can", "may", "just", "also",
    }
)

# There is no image memory yet, so a text hit cannot prove two pictures are
# the same file. "video … shows" and "photo of" are the claim talking about
# a picture. An old flood photo must not be reused for a new flood.
_PICTURE_CLAIM = re.compile(
    r"\b(photo|image|picture|video|clip|footage)\b.*\b(shows?|of)\b",
    re.IGNORECASE,
)
_MEDIA_CAP = (
    "The text matches an earlier fact-check, but we can't confirm it is the same picture; "
    "an old flood photo can't be reused for a new one."
)

_MONTH_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_DAYS = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
_MONTH_NAME = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
# "1 December", "1 December 2025", "1st December". Year is optional so a
# day-and-month with no year can still be compared.
_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME})(?:\s*,?\s*((?:19|20)\d{{2}}))?\b",
    re.IGNORECASE,
)
# "Dec 1", "September 30, 2025".
_MONTH_DAY = re.compile(
    rf"\b({_MONTH_NAME})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,?\s*((?:19|20)\d{{2}}))?\b",
    re.IGNORECASE,
)
# Indian fact-checks write the day first. 01/12/2025 is 1 December, not
# 12 January. A month above 12 is rejected rather than swapped.
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-]((?:19|20)\d{2})\b")

_IDF: dict[str, float] | None = None


class ContextVerdict(BaseModel):
    """The only thing the model is allowed to say about a pair.

    same_claim false is a veto. It cannot create an exact match. differing_detail
    is the one difference a reader would name, or null when the claims match.
    """

    same_claim: bool
    differing_detail: str | None = None


def llm_context_check(claim: str, match: dict) -> ContextVerdict:
    """One structured call, and only through llm.py.

    The question is whether the two sentences are the same checkable claim:
    same subject, person, place, amount, date, and community. The provider
    call lives in llm.call_structured. This function only writes the prompt
    and checks the shape.
    """
    other = str((match or {}).get("primary_claim") or "")
    prompt = (
        "Are these the same checkable claim "
        "(same subject, person, place, amount, date, community)?\n\n"
        f"Claim: {claim}\n"
        f"Fact-check: {other}\n\n"
        "same_claim is true only when a reader would be checking the same allegation. "
        "If they differ, differing_detail is that one difference, in a short phrase. "
        "If they are the same, differing_detail is null."
    )
    return ContextVerdict.model_validate(call_structured(prompt, ContextVerdict))


def decide_prior(
    claim: str,
    results: list,
    now=None,
    entities=None,
    form: str | None = None,
    tau_high: float | None = None,
    tau_low: float | None = None,
    keyterm_sim: float | None = None,
    rules: str = "new",
) -> dict:
    """exact / similar / none, and whether the old verdict can be reused.

    best_match is the hit with the highest dense cosine, not the reciprocal-
    rank winner. The thresholds are cosines. Leads stay in the order search
    returned them: that is the list a person should read, and it is what
    investigation receives when the verdict is not reused.

    tau_high and tau_low default to config and are closed bounds: a cosine
    equal to TAU_HIGH is an exact candidate, not a near miss. The sweep in
    the eval script passes other values. It does not write them back.

    rules="old" is the stage D3 comparison: difference-of-sets details and
    the IDF key-term list, with no community check. The default is the
    contradiction rule, the paired-term check, and the community list.
    """
    today = _as_date(now)
    high = TAU_HIGH if tau_high is None else float(tau_high)
    low = TAU_LOW if tau_low is None else float(tau_low)
    new_rules = rules != "old"
    sim_floor = KEYTERM_SIM if keyterm_sim is None else float(keyterm_sim)
    leads = list(results[:5])
    best = _best_match(results)
    if best is None:
        return _decision(
            "none",
            "investigate",
            None,
            [],
            False,
            ["Nothing similar is in the claim memory, so this has to be checked from scratch."],
        )

    cosine = float(best.get("dense_cos") or 0.0)
    # >=, not >. The configured cutoff itself counts. 0.850 is exact.
    if cosine >= high:
        match_type = "exact"
    elif cosine >= low:
        match_type = "similar"
    else:
        match_type = "none"

    notes: list[str] = []
    if match_type in {"exact", "similar"}:
        # Concrete clashes (the date, the amount) go on the card ahead of
        # the bag of leftover words. The card keeps the first two sentences.
        matched = str(best.get("primary_claim") or "")
        notes.extend(
            _detail_notes(
                claim,
                matched,
                entities,
                contradiction=new_rules,
                normalise=new_rules,
            )
        )
        if new_rules:
            community = _community_note(claim, matched)
            if community:
                notes.append(community)
            subject = _paired_term_note(claim, matched, sim_floor)
        else:
            subject = _subject_note(claim, matched)
        if subject:
            notes.append(subject)
    # Only an exact candidate can be downgraded by a detail. A merely
    # similar hit is already going to investigation; the notes explain why
    # it is not the same claim, they do not push it down to "none".
    downgraded = match_type == "exact" and bool(notes)
    if downgraded:
        match_type = "similar"

    # Text cannot identify a picture. Cap after the detail rule so an amount
    # clash and a missing picture are both explained, and neither one can
    # leave the label at "exact".
    media_note = None
    if _claim_is_picture(claim, form) and match_type in {"exact", "similar"}:
        match_type = "similar"
        media_note = _MEDIA_CAP

    # The model is a veto, not a second threshold. It runs only when every
    # rule has already left the pair exact, and only when the flag is on.
    # A similar or none result never reaches it, so the model cannot promote.
    veto_note = None
    if match_type == "exact" and LLM_CONTEXT_CHECK:
        verdict = llm_context_check(claim, best)
        if not verdict.same_claim:
            match_type = "similar"
            downgraded = True
            detail = (verdict.differing_detail or "").strip() or "the wording is not the same claim"
            veto_note = f"A careful read found a difference: {detail}."

    recheck, recheck_reason = _needs_recheck(claim, best, today)
    reasons = [_closeness_sentence(best, cosine, match_type, downgraded)]
    reasons.extend(notes)
    if veto_note:
        reasons.append(veto_note)
    if media_note:
        reasons.append(media_note)
    if recheck_reason:
        reasons.append(recheck_reason)

    route = "reuse_prior" if match_type == "exact" and not recheck else "investigate"
    if route == "reuse_prior":
        reasons.append(
            "The earlier fact-check can be reused. It is the same claim, "
            "and it is not a status that goes out of date."
        )
    else:
        reasons.append("This needs a fresh check. The memory hits are leads, not a verdict.")

    return _decision(match_type, route, best, leads, recheck, reasons)


def _decision(match_type, route, best, leads, recheck, reasons: list[str]) -> dict:
    """match_reasons is the full list. match_summary is the card: the first two.

    `reasons` is the same list. Callers and tests from before the card
    field still read it. The state key is match_reasons.
    """
    return {
        "match_type": match_type,
        "route": route,
        "best_match": best,
        "leads": leads,
        "needs_recheck": recheck,
        "match_reasons": reasons,
        "match_summary": reasons[:2],
        "reasons": reasons,
    }


def _claim_is_picture(claim: str, form: str | None) -> bool:
    """True when the claim is about a picture and we have no image hash.

    form comes from the claim state when extract or ingest has set it.
    The regex covers a WhatsApp line that never went through that step.
    There is no image memory to confirm the file, so this is always a cap,
    not a lookup.
    """
    if form and str(form).casefold() in {"image", "video"}:
        return True
    return _PICTURE_CLAIM.search(claim or "") is not None


def _best_match(results: list) -> dict | None:
    if not results:
        return None
    # Earlier row wins a tie. max() would otherwise pick whichever dict
    # happens to compare greater, and the choice would depend on key order.
    return max(
        enumerate(results),
        key=lambda pair: (float(pair[1].get("dense_cos") or 0.0), -pair[0]),
    )[1]


def _closeness_sentence(best: dict, cosine: float, match_type: str, downgraded: bool) -> str:
    source = _source(best)
    score = f"{cosine:.3f}"
    if match_type == "exact":
        return f"This is the same claim as {source} (similarity {score})."
    if downgraded:
        return (
            f"This is very close to {source} (similarity {score}), "
            "but a detail does not match, so it is not the same claim."
        )
    if match_type == "similar":
        return f"This is related to {source}, not the same claim (similarity {score})."
    return (
        f"The closest fact-check is {source}, and it is only weakly related "
        f"(similarity {score}), so memory cannot settle this."
    )


def _source(best: dict) -> str:
    site = best.get("site")
    when = _long_date(best.get("date"))
    dated = f" from {when}" if when else ""
    if site:
        return f"the {site} fact-check{dated}"
    if when:
        return f"a fact-check{dated}"
    return "a fact-check"


def _detail_notes(
    claim: str,
    matched: str,
    entities,
    *,
    contradiction: bool = True,
    normalise: bool = True,
) -> list[str]:
    """Dates, amounts, years, numbers, and rough names.

    A clash is a contradiction: both sides state that kind, and each side
    has a value the other does not. A fact-check that only adds a figure
    ("also 45 del", "and new ₹1000 notes") is more specific, not a different
    claim. The old comparison treated any unequal set as a clash; rules="old"
    still does, so the D3 table can score that behaviour.

    Dates are read after number words have been turned into digits, and the
    date span is lifted out before plain numbers. Otherwise "1 December"
    and "30 September" become the numbers 1 and 30. Box 3's entities, when
    passed, replace the regex on the claim side. A kind that appears on
    only one side is not a disagreement.
    """
    claim_bits = _parse_details(claim, normalise=normalise)
    if entities:
        claim_bits = _prefer_entities(claim_bits, entities)
    match_bits = _parse_details(matched, normalise=normalise)
    notes = []
    date_note = _date_clash(claim_bits["dates"], match_bits["dates"], contradiction=contradiction)
    if date_note:
        notes.append(date_note)
    amount = _numeric_clash(
        claim_bits["amounts"], match_bits["amounts"], "amount", "₹", contradiction=contradiction
    )
    if amount:
        notes.append(amount)
    percent = _numeric_clash(
        claim_bits["percents"],
        match_bits["percents"],
        "percent",
        "",
        suffix="%",
        contradiction=contradiction,
    )
    if percent:
        notes.append(percent)
    year = _numeric_clash(
        claim_bits["years"], match_bits["years"], "year", "", contradiction=contradiction
    )
    if year:
        notes.append(year)
    number = _numeric_clash(
        claim_bits["numbers"], match_bits["numbers"], "number", "", contradiction=contradiction
    )
    if number:
        notes.append(number)
    names = _name_clash(claim_bits["names"], match_bits["names"])
    if names:
        notes.append(names)
    return notes


def _extract_dates(text: str) -> tuple[list[dict], str]:
    """Date spans, and the text with those spans blanked.

    Blanking, not deleting, keeps the rest of the string in place. The year
    inside "30 September 2025" must not also be read as a bare year, and the
    1 and the 30 must not be read as plain numbers.
    """
    if not text:
        return [], ""
    found = []
    for pattern, kind in ((_NUMERIC_DATE, "numeric"), (_DAY_MONTH, "day_month"), (_MONTH_DAY, "month_day")):
        for match in pattern.finditer(text):
            parsed = _interpret_date(kind, match)
            if parsed is not None:
                found.append(parsed)
    # Longest span wins, so "30 September 2025" is one date, not a shorter
    # "30 September" plus a leftover year.
    found.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"]))
    chosen = []
    occupied: list[tuple[int, int]] = []
    for item in found:
        if any(item["start"] < end and item["end"] > start for start, end in occupied):
            continue
        chosen.append(item)
        occupied.append((item["start"], item["end"]))
    chosen.sort(key=lambda item: item["start"])
    chars = list(text)
    for item in chosen:
        for index in range(item["start"], item["end"]):
            chars[index] = " "
    return chosen, "".join(chars)


def _interpret_date(kind: str, match: re.Match) -> dict | None:
    if kind == "numeric":
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    elif kind == "day_month":
        day = int(match.group(1))
        month = _MONTH_NUM[match.group(2).lower()]
        year = int(match.group(3)) if match.group(3) else None
    else:
        month = _MONTH_NUM[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
    if not _real_day(day, month):
        return None
    return {
        "start": match.start(),
        "end": match.end(),
        "surface": " ".join(match.group(0).split()),
        "day": day,
        "month": month,
        "year": year,
    }


def _real_day(day: int, month: int) -> bool:
    if month < 1 or month > 12:
        return False
    return 1 <= day <= _MONTH_DAYS[month]


def _date_clash(left: list[dict], right: list[dict], *, contradiction: bool = True) -> str | None:
    """Same day and month agree. A missing year is not itself a clash.

    "1 December" and "1 December 2025" are the same day stated with less
    detail. "1 December" and "30 September 2025" are not. Both years have
    to be present, and different, before the year splits them.
    """
    if not left or not right:
        return None

    def compatible(one: dict, other: dict) -> bool:
        if one["day"] != other["day"] or one["month"] != other["month"]:
            return False
        if one["year"] and other["year"] and one["year"] != other["year"]:
            return False
        return True

    def uncovered(src, dst):
        return [item for item in src if not any(compatible(item, other) for other in dst)]

    missing_left = uncovered(left, right)
    missing_right = uncovered(right, left)
    # Contradiction: each side has a date the other cannot account for.
    # One extra date is a more specific telling of the same day.
    if contradiction:
        if not missing_left or not missing_right:
            return None
    elif not missing_left and not missing_right:
        return None
    sample_left = (missing_left or left)[0]["surface"]
    sample_right = (missing_right or right)[0]["surface"]
    return f"The date differs: {sample_left} vs {sample_right}."


_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE = {
    "hundred": 100, "thousand": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000,
}
# Tens before the units alternative, so "twenty five" is 25 and not the
# word "five" left beside an untouched "twenty".
_NUMBER_WORD = re.compile(
    r"\b(?:(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s](one|two|three|four|five|six|seven|eight|nine))?|"
    r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))\b",
    re.IGNORECASE,
)
_A_SCALE = re.compile(
    r"\ba\s+(hundred|thousand|lakh|lakhs|lac|lacs|crore|crores)\b",
    re.IGNORECASE,
)
_SCALED = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+(hundred|thousand|lakh|lakhs|lac|lacs|crore|crores)\b",
    re.IGNORECASE,
)
# Rs. / Rs / INR before a number, and "rupees" after one. The detail rule
# then sees one amount, not a bare 500 beside a separate word "rupees".
_RS_PREFIX = re.compile(
    r"(?<![A-Za-z])(?:₹|rs\.?|inr)\s*(\d[\d,]*)(?:\.\d+)?",
    re.IGNORECASE,
)
_RS_WORD = re.compile(
    r"(?<![A-Za-z\d])(\d[\d,]*)(?:\.\d+)?\s*rupees?\b",
    re.IGNORECASE,
)


def _normalise_quantities(text: str) -> str:
    """Number words, lakh/crore, and rupee spellings, before the detail rule.

    "five lakh" and "5 lakh" have to land on the same integer, or the card
    says the number differs when the writer only spelled it out. "Rs." and
    "rupees" have to land on ₹ for the same reason. The BM25 index is left
    alone: rebuilding it would change every search, and this comparison only
    needs the two claims box 6 is holding.
    """
    if not text:
        return ""

    def words(match: re.Match) -> str:
        if match.group(1):
            value = _TENS[match.group(1).lower()]
            if match.group(2):
                value += _ONES[match.group(2).lower()]
            return str(value)
        return str(_ONES[match.group(3).lower()])

    text = _NUMBER_WORD.sub(words, text)
    text = _A_SCALE.sub(lambda match: f"1 {match.group(1)}", text)

    def scaled(match: re.Match) -> str:
        value = float(match.group(1)) * _SCALE[match.group(2).lower()]
        if value.is_integer():
            return str(int(value))
        return str(value)

    # "5 hundred thousand" needs a second pass. Two is enough for lakh/crore
    # stacked on hundred; a third would still be the same string.
    for _ in range(2):
        text = _SCALED.sub(scaled, text)
    text = _RS_PREFIX.sub(lambda match: "₹" + match.group(1).replace(",", ""), text)
    text = _RS_WORD.sub(lambda match: "₹" + match.group(1).replace(",", ""), text)
    return text


def _parse_details(text: str, *, normalise: bool = True) -> dict:
    # Names stay on the original capitals. Number words are rewritten only
    # on the copy the amounts and the dates are read from.
    raw = text or ""
    working = _normalise_quantities(raw) if normalise else raw
    amounts = set()
    for _surface, stem in bm25_token_pairs(working):
        if stem.startswith("inr") and stem[3:].isdigit():
            amounts.add(stem[3:])
    # Rupee amounts are lifted before dates, then dates before years and
    # bare numbers. ₹500 must not also be the number 500, and the 1 in
    # "1 December" must not also be the number 1.
    remainder = strip_rupee_amounts(working)
    dates, remainder = _extract_dates(remainder)
    # Percents come out before plain numbers. Otherwise "9.2%" is the
    # numbers 9 and 2, and "2 percent" looks like the same digit.
    percents, remainder = _extract_percents(remainder)
    years = {match.group(0) for match in _YEAR.finditer(remainder)}
    numbers = set()
    for match in _NUMBER.finditer(remainder):
        digits = _canon_number(match.group(0))
        if digits in years:
            continue
        numbers.add(digits)
    names = {}
    # The first word of a sentence is capitalised because it is first, not
    # because it is a name. "Drinking hot water…" versus "Inhaling hot
    # water…" would otherwise look like a place clash. An acronym (RBI, WHO)
    # still counts at the start: the capitals are the name, not the sentence.
    raw = text or ""
    origin = len(raw) - len(raw.lstrip())
    for match in _CAPITAL.finditer(raw):
        word = match.group(0)
        if word.casefold() in _NOT_A_NAME:
            continue
        if match.start() == origin and not word.isupper():
            continue
        names.setdefault(word.casefold(), word)
    return {
        "amounts": amounts,
        "years": years,
        "numbers": numbers,
        "percents": percents,
        "names": names,
        "dates": dates,
    }


def _extract_percents(text: str) -> tuple[set[str], str]:
    """Percent values, and the text with those spans blanked.

    Blanking keeps "9.2%" from also being read as the plain numbers 9 and 2.
    "2 percent" is the same kind of value as "2%".
    """
    if not text:
        return set(), ""
    found = set()
    chars = list(text)
    for match in _PERCENT.finditer(text):
        found.add(_canon_number(match.group(1)))
        for index in range(match.start(), match.end()):
            chars[index] = " "
    return found, "".join(chars)


def _canon_number(text: str) -> str:
    """'9.20' and '9.2' are one value. '18.0' is 18, so a subset test can see it."""
    value = Decimal(text.replace(",", ""))
    if value == value.to_integral():
        return str(value.to_integral())
    return format(value.normalize(), "f")


def _prefer_entities(parsed: dict, entities: dict) -> dict:
    """Box 3 already pulled the amounts and names. Use those for the claim.

    The regex is a fallback for when extract has not run. Preferring the
    entities means a later, better parser wins without this module growing
    a second copy of it. A key that is present, even as an empty list, is
    what box 3 decided; a missing key falls back to the regex.
    """
    updated = dict(parsed)
    if "amounts" in entities:
        updated["amounts"] = _entity_amounts(entities.get("amounts") or [])
    if "years" in entities:
        updated["years"] = {str(year) for year in entities.get("years") or []}
    if "dates" in entities:
        found = []
        for item in entities.get("dates") or []:
            parsed, _rest = _extract_dates(str(item))
            found.extend(parsed)
        updated["dates"] = found
    if "names" in entities or "places" in entities:
        names = {}
        for word in list(entities.get("names") or []) + list(entities.get("places") or []):
            text = str(word).strip()
            if text:
                names.setdefault(text.casefold(), text)
        updated["names"] = names
    return updated


def _entity_amounts(values) -> set[str]:
    amounts = set()
    for value in values:
        taken = False
        for _surface, stem in bm25_token_pairs(str(value)):
            if stem.startswith("inr") and stem[3:].isdigit():
                amounts.add(stem[3:])
                taken = True
        if not taken:
            digits = re.sub(r"[^\d]", "", str(value))
            if digits:
                amounts.add(digits)
    return amounts


def _numeric_clash(
    left: set[str],
    right: set[str],
    kind: str,
    prefix: str,
    *,
    suffix: str = "",
    contradiction: bool = True,
) -> str | None:
    if not left or not right:
        return None
    # {2000} against {2000, 1000} is the same note plus an extra figure.
    # {500} against {2000} is a contradiction: neither set contains the other.
    if contradiction:
        if left <= right or right <= left:
            return None
    elif left == right:
        return None
    sample_left = _sample(left, right)
    sample_right = _sample(right, left)
    return (
        f"The {kind} differs: {prefix}{sample_left}{suffix} "
        f"vs {prefix}{sample_right}{suffix}."
    )


def _sample(have: set[str], other: set[str]) -> str:
    only = sorted(have - other, key=Decimal)
    if only:
        return only[0]
    return sorted(have, key=Decimal)[0]


def _name_clash(left: dict, right: dict) -> str | None:
    # One side with no proper noun is not a clash. "Notes are invalid" and
    # "RBI says notes are invalid" are the same claim with an added name.
    # Jaccard 0 for empty-against-something would downgrade that pair.
    if not left or not right:
        return None
    shared = set(left) & set(right)
    union = set(left) | set(right)
    if len(shared) / len(union) >= 0.2:
        return None
    return (
        "The names and places barely overlap: "
        f"{_quoted(list(left.values()))} vs {_quoted(list(right.values()))}."
    )


def _subject_note(claim: str, matched: str) -> str | None:
    """Key terms are the claim's 3 highest-IDF content tokens.

    Fewer than 2 of those 3 in the fact-check means the rare words are not
    shared, so an exact match is downgraded.

    That test misses a one-word subject swap when the swapped word is common
    in this archive. "Hot lemon water kills coronavirus" versus "…kills cancer
    cells" shares its three rarest tokens (hot, lemon, drink: IDF about 6.6,
    6.2 and 5.0). "Coronavirus" is common here, because the memory is full of
    covid fact-checks, so IDF does not rank it as a key term, and the number
    and capital-word rules never see a lowercase noun. If each side has one
    or two content words the other lacks, that swap is the subject, and it
    blocks reuse the same way. An added "forward this" does not: the extra
    word is only on one side.
    """
    claim_pairs = bm25_token_pairs(claim)
    match_pairs = bm25_token_pairs(matched)
    if not claim_pairs or not match_pairs:
        return None
    claim_stems = [stem for _surface, stem in claim_pairs]
    match_stems = {stem for _surface, stem in match_pairs}
    claim_set = set(claim_stems)
    surface = {}
    for word, stem in claim_pairs + match_pairs:
        surface.setdefault(stem, word)

    idf = _idf_table()
    ceiling = max(idf.values(), default=0.0) + 1.0

    def rarity(stem: str) -> float:
        # A word the archive has never indexed is the most distinctive
        # token we can have. Treating it as 0 would hide a novel subject.
        return idf.get(stem, ceiling)

    unique_claim = list(dict.fromkeys(claim_stems))
    # Years and rupee amounts have their own sentences. Ranking them as key
    # terms made "2016" and "forward" look like the subject of a ₹500 claim.
    candidates = [stem for stem in unique_claim if _content_stem(stem)]
    keys = sorted(candidates, key=lambda stem: (-rarity(stem), stem))[:3]
    present = sum(1 for stem in keys if stem in match_stems)
    need = 2 if len(keys) >= 2 else len(keys)

    only_claim = [stem for stem in unique_claim if stem not in match_stems and _content_stem(stem)]
    only_match = [
        stem
        for stem in dict.fromkeys(stem for _word, stem in match_pairs)
        if stem not in claim_set and _content_stem(stem)
    ]
    if need and present < need:
        missing = [stem for stem in keys if stem not in match_stems] or only_claim
        # A one- or two-word swap is "coronavirus vs cancer". A long mismatch
        # is not that: pairing the rarest missing words with the first words
        # of the fact-check turned "invalid" into a clash with "Reserve Bank".
        if 1 <= len(only_claim) <= 2 and 1 <= len(only_match) <= 2:
            return _subject_sentence(only_claim, only_match, surface)
        words = [surface.get(stem, stem) for stem in missing]
        verb = "is" if len(words) == 1 else "are"
        return (
            f"The distinctive words do not match: {_quoted(words)} "
            f"{verb} not in the fact-check."
        )
    # A small substitution, not a wholesale rewrite. Capping at two words a
    # side is what keeps "forward this" (extra words, one side only) and a
    # long paraphrase from being described as a single swapped subject.
    if 1 <= len(only_claim) <= 2 and 1 <= len(only_match) <= 2:
        return _subject_sentence(only_claim, only_match, surface)
    return None


def _content_stem(stem: str) -> bool:
    # Rupee amounts and bare numbers have their own sentences. Quoting them
    # again as a "subject" produces "inr500 vs inr2000" beside "₹500 vs ₹2000".
    if stem.startswith("inr") or stem.isdigit() or stem in _NOT_A_SUBJECT or stem in _NOT_A_NAME:
        return False
    return True


def _subject_sentence(claim_stems: list[str], match_stems: list[str], surface: dict) -> str:
    left = _quoted([surface.get(stem, stem) for stem in claim_stems])
    right = _quoted([surface.get(stem, stem) for stem in match_stems])
    if claim_stems and match_stems:
        return f"The key subject differs: {left} vs {right}."
    if claim_stems:
        return f"The key subject differs: {left} is not in the earlier fact-check."
    return f"The key subject differs: the fact-check is about {right}."


_COMMUNITY_RE = re.compile(
    r"\b("
    + "|".join(re.escape(term) for term in sorted(COMMUNITY_TERMS, key=len, reverse=True))
    + r")s?\b",
    re.IGNORECASE,
)
_WORD_CACHE: dict[str, object] = {}
_WORD_ENCODER = None


def _community_terms(text: str) -> list[str]:
    """Community words in reading order. Plurals collapse: Hindus -> hindu.

    The optional s is only the plural. The alternation is longest-first, so
    "hinduism" is itself and not the word "hindu" plus leftover letters.
    """
    found = []
    seen = set()
    for match in _COMMUNITY_RE.finditer(text or ""):
        term = match.group(1).lower()
        if term not in seen:
            seen.add(term)
            found.append(term)
    return found


def _community_note(claim: str, matched: str) -> str | None:
    """A community named on only one side, or a different community.

    Row 17 is the case this exists for: "a Hindu manager" and "an Indian
    manager" describe the same restaurant story with the communal label
    added. Indian is a nationality and is not in the list. Hindu is.
    """
    left = _community_terms(claim)
    right = set(_community_terms(matched))
    if set(left) == right:
        return None
    extra = [term for term in left if term not in right]
    if extra:
        shown = extra[0].capitalize()
        return (
            f"The claim names a community ('{shown}') that the fact-check does not; "
            "communal framing is often the false part."
        )
    only_match = [term for term in _community_terms(matched) if term not in set(left)]
    shown = only_match[0].capitalize()
    return (
        f"The fact-check names a community ('{shown}') that the claim does not; "
        "communal framing is often the false part."
    )


def _paired_term_note(claim: str, matched: str, sim_floor: float) -> str | None:
    """Downgrade when a key term is not a near-synonym of any word on the other side.

    Key terms are the top 3 IDF content stems of each claim, after number
    words have been folded into digits and stopwords dropped. The old test
    asked only whether those stems occurred. "Excess" and "additional" are
    different strings and the same idea, so a missing stem is now encoded
    as a single word and compared with the other claim's content words.

    Both directions are checked. On this archive "coronavirus" is common, so
    it is not a top-IDF term of "hot lemon water kills coronavirus", and the
    swapped word "cancer" lives only on the other sentence. Checking the
    incoming claim alone would leave that pair exact.
    """
    left = _content_tokens(claim)
    right = _content_tokens(matched)
    if not left or not right:
        return None
    failures = []
    failures.extend(_unmatched_keys(left, right, sim_floor))
    failures.extend(_unmatched_keys(right, left, sim_floor))
    if not failures:
        return None
    missing, nearest, score = min(failures, key=lambda item: item[2])
    return (
        f"The key subject differs: '{missing}' vs '{nearest}' "
        f"(word similarity {score:.2f})."
    )


def _content_tokens(text: str) -> list[tuple[str, str]]:
    """(surface, stem) content words, numbers already folded, one row per stem."""
    seen = set()
    tokens = []
    for surface, stem in bm25_token_pairs(_normalise_quantities(text)):
        if stem in seen or not _content_stem(stem):
            continue
        seen.add(stem)
        tokens.append((surface, stem))
    return tokens


def _top_keys(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    idf = _idf_table()
    ceiling = max(idf.values(), default=0.0) + 1.0

    def rarity(stem: str) -> float:
        return idf.get(stem, ceiling)

    return sorted(tokens, key=lambda item: (-rarity(item[1]), item[1]))[:3]


def _unmatched_keys(
    source: list[tuple[str, str]],
    other: list[tuple[str, str]],
    sim_floor: float,
) -> list[tuple[str, str, float]]:
    other_stems = {stem for _surface, stem in other}
    missing = [item for item in _top_keys(source) if item[1] not in other_stems]
    if not missing:
        return []
    words = [surface for surface, _stem in missing] + [surface for surface, _stem in other]
    vectors = _word_vectors(words)
    failures = []
    for surface, _stem in missing:
        nearest, score = max(
            ((word, float(vectors[surface] @ vectors[word])) for word, _other in other),
            key=lambda item: item[1],
        )
        if score < sim_floor:
            failures.append((surface, nearest, score))
    return failures


def _word_vectors(words: list[str]) -> dict:
    """Unit vectors for single words. The sentence cosine is already the threshold.

    Encoding the whole claim again would repeat box 5. A word vector asks a
    smaller question: is this missing key term a paraphrase of a word that
    is actually in the other claim. The cache is the grid. Every TAU_HIGH
    rescores the same words, and loading BGE-M3 once per word would dominate.
    """
    missing = [word for word in dict.fromkeys(words) if word not in _WORD_CACHE]
    if missing:
        encoder = _word_encoder()
        import numpy as np

        matrix = encoder.encode(
            missing,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        matrix = np.asarray(matrix, dtype=np.float32)
        for word, vector in zip(missing, matrix):
            _WORD_CACHE[word] = vector
    return {word: _WORD_CACHE[word] for word in words}


def _word_encoder():
    global _WORD_ENCODER
    if _WORD_ENCODER is None:
        from sentence_transformers import SentenceTransformer

        from misinfo_agent.config import MAX_SEQ_LEN, MODEL_NAME
        from misinfo_agent.memory import _device

        encoder = SentenceTransformer(MODEL_NAME, device=_device())
        # Same window as the memory. A single word is well inside it; leaving
        # the library default would still be a different tower than E.
        encoder.max_seq_length = MAX_SEQ_LEN
        _WORD_ENCODER = encoder
    return _WORD_ENCODER


def _quoted(words: list[str]) -> str:
    shown = [word for word in words if word][:3]
    if not shown:
        return "something else"
    if len(shown) == 1:
        return f"'{shown[0]}'"
    if len(shown) == 2:
        return f"'{shown[0]}' and '{shown[1]}'"
    return f"'{shown[0]}', '{shown[1]}' and '{shown[2]}'"


def _needs_recheck(claim: str, match: dict, today: date) -> tuple[bool, str | None]:
    """Status claims expire. An old photo's origin does not.

    Covaxin was "not WHO-approved" until November 2021 and approved after
    that. A fact-check of a status word (approved, banned, price, tax, …)
    older than a year cannot be reused: the world may have moved.
    An image or video that is out of context, or edited, is a claim about
    where a file came from. That origin does not change the way an approval
    does, so age alone does not force a new check. The corpus spells the
    edited class "Edited or doctored".
    """
    words = _status_words(claim, match.get("primary_claim"))
    if not words:
        return False, None
    published = _parse_date(match.get("date"))
    stale = published is None or (today - published).days > STALE_DAYS
    if not stale:
        return False, None
    if _origin_does_not_expire(match):
        return False, (
            "The earlier fact-check is an old image or video that was out of "
            "context or edited. Where that picture came from does not change, "
            "so its age does not force a new check."
        )
    listed = _quoted(words)
    if published is None:
        return True, (
            f"The claim is about a status that can change ({listed}), and the "
            "fact-check has no date, so it cannot be reused as current."
        )
    return True, (
        f"The fact-check is from {_long_date(published)}, more than a year ago, "
        f"and the claim is about a status that can change ({listed}). "
        "That has to be checked again."
    )


def _origin_does_not_expire(match: dict) -> bool:
    form = match.get("form") or ""
    kind = match.get("manipulation_type") or ""
    if form not in {"Image", "Video"}:
        return False
    return kind == "Out of context" or kind == "Edited" or kind.startswith("Edited")


def _status_words(*texts) -> list[str]:
    found = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for match in _STATUS.finditer(str(text)):
            word = match.group(0).lower()
            if word not in seen:
                seen.add(word)
                found.append(word)
    return found


def _idf_table() -> dict[str, float]:
    """IDF from the memory's BM25 file, not a second tokeniser.

    Loading the embedding matrix here would be a few hundred megabytes for
    a dictionary of word weights. The pickle is the vocabulary box 5 built.
    """
    global _IDF
    if _IDF is None:
        # The class has to be importable before unpickle, or the BM25 object
        # fails to reconstruct. This file is local build output, not a claim.
        import rank_bm25  # noqa: F401

        with BM25_PATH.open("rb") as handle:
            model = pickle.load(handle)
        _IDF = dict(model.idf)
    return _IDF


def _as_date(value) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _parse_date(value) -> date | None:
    if value is None or value == "":
        return None
    try:
        return _as_date(value)
    except ValueError:
        return None


def _long_date(value) -> str:
    if value is None or value == "":
        return ""
    try:
        parsed = value if isinstance(value, date) and not isinstance(value, datetime) else _parse_date(value)
    except ValueError:
        return ""
    if parsed is None:
        return ""
    return f"{parsed.day} {parsed.strftime('%B %Y')}"
