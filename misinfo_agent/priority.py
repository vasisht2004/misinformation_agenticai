"""Box 11: how soon a person should look at this result.

Rules only. The verdict is already decided by the time this runs, or it
is not this box's job. Style signals — caps, a forward label, a
"breaking" banner — move the score and never the verdict. A shouted
claim can be true, and a quiet one can be false.

The score is a weighted sum of four components, each in [0, 1]:

  harm      what the domain can do to someone, plus a few red flags
  novelty   1 minus the closest memory cosine
  reach     how far the forward has already travelled, on a log scale
  urgency   deadline language and a request to pass it on

Weights are hand-set in config. scripts/priority_sensitivity.py is the
check that a 20% change does not silently re-rank the fixtures.
"""

from __future__ import annotations

import math
import re
from datetime import date

import misinfo_agent.config as config

# Violent acts, not mere mention. "A Hindu temple was inaugurated" names
# a community and is not targeting. The archive has no label for this;
# the four-flag list in config accepts the words when a box writes them,
# and this pattern is how a claim earns the flag when it did not.
_VIOLENCE = re.compile(
    r"\b(?:attack\w*|kill\w*|lynch\w*|riot\w*|burn\w*|ston\w*|"
    r"raped?|slaughter\w*|drove\s+out|thrown\s+out|threw)\b",
    re.IGNORECASE,
)

# Deadline language that ingest did not already file under urgency_words.
# "immediately" and "last date" live there. Counting them again here
# would make one banner count as two.
_DEADLINE = re.compile(
    r"\b(?:deadline|last\s+day|expir(?:y|es|e)|today|tonight|tomorrow|"
    r"right\s+now|this\s+week|within\s+24\s+hours?)\b",
    re.IGNORECASE,
)

_MONTHS = {
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
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_DM = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})\b",
    re.IGNORECASE,
)
_DATE_MD = re.compile(
    rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def priority(state: dict, weights: dict | None = None) -> dict:
    """Score the claim and map it to High, Medium, or Low.

    `weights` is for the sensitivity script. The pipeline call leaves it
    empty and uses config.PRIORITY_WEIGHTS. Passing a dict does not
    change the four components, only how they are mixed, so a perturbed
    weight cannot quietly change what "harm" means.
    """
    parts = components(state)
    used = dict(config.PRIORITY_WEIGHTS if weights is None else weights)
    score = 0.0
    for name, weight in used.items():
        score += float(weight) * float(parts[name])
    # Components and the hand-set weights are each in range, and the
    # weights sum to 1. The clamp is for a sensitivity run that moves
    # one weight and does not re-balance the others.
    score = _unit(score)
    return {
        "priority": level(score),
        "priority_score": score,
        "priority_reasons": list(parts["reasons"]),
    }


def components(state: dict) -> dict:
    """The four numbers, plus the sentences that explain them.

    Kept separate from the weights so the sensitivity script can reuse
    the same harm, novelty, reach and urgency the box itself uses.
    """
    harm_s, harm_line = _harm(state)
    novel_s, novel_line = _novelty(state)
    reach_s, reach_line = _reach(state)
    urgent_s, urgent_line = _urgency(state)
    return {
        "harm": harm_s,
        "novelty": novel_s,
        "reach": reach_s,
        "urgency": urgent_s,
        "reasons": [harm_line, novel_line, reach_line, urgent_line],
    }


def level(score: float) -> str:
    """High at 0.60, Medium at 0.35, otherwise Low.

    The two cuts are far enough apart that a 20% weight wobble on a
    clearly high or clearly low claim usually stays in its band. A
    claim that sits on a cut is what the sensitivity script is for.
    """
    if score >= config.PRIORITY_HIGH:
        return "High"
    if score >= config.PRIORITY_MEDIUM:
        return "Medium"
    return "Low"


def harm(state: dict) -> float:
    """The H that box 12 puts in the confidence bar.

    This is the harm component only. Reach and urgency raise how fast
    a person should look; they do not lower the bar for calling a claim
    false. A viral sports rumour is still a sports rumour.
    """
    score, _line = _harm(state)
    return score


def _harm(state: dict) -> tuple[float, str]:
    domain = str(state.get("domain") or "")
    if domain in config.DOMAIN_HARM:
        base = float(config.DOMAIN_HARM[domain])
        domain_line = f"{domain} is the harm for this claim ({base:.2f})."
    else:
        # An empty or unknown domain is not "no harm". It is the middle
        # of the scale, so a missing label cannot hide a claim or panic one.
        base = 0.45
        domain_line = "The domain is not one of the 11, so harm stays in the middle (0.45)."
    flags = _harm_flags(state)
    bumped = _unit(base + config.HARM_FLAG_BUMP * len(flags))
    if flags:
        # One flag takes "raises". Two or more take "raise". The list is
        # the config order, so the card does not reshuffle between runs.
        if len(flags) == 1:
            names = flags[0]
            verb = "raises"
        else:
            names = ", ".join(flags[:-1]) + " and " + flags[-1]
            verb = "raise"
        line = (
            f"{domain_line} {names} {verb} it to {bumped:.2f}, "
            "because those flags make the content more dangerous rather than merely louder."
        )
    else:
        line = domain_line
    return bumped, line


def _harm_flags(state: dict) -> list[str]:
    """The harmful subset, in the config order, each at most once.

    Style flags that did not match a label stay out. A conspiracy
    phrase is recorded for the card and does not add harm on its own:
    "they are hiding it" is not yet a crowd, a cure, or a named attack.
    """
    found = []
    raw = state.get("red_flags") or []
    if not isinstance(raw, list):
        raw = []
    pool = {str(item) for item in raw}
    if _targets_a_community(state):
        pool.add("communal targeting")
    for name in config.HARM_RED_FLAGS:
        if name in pool:
            found.append(name)
    return found


def _targets_a_community(state: dict) -> bool:
    text = " ".join(
        str(state.get(key) or "")
        for key in ("claim", "text", "raw_text")
    )
    if _VIOLENCE.search(text) is None:
        return False
    for term in config.COMMUNITY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return True
    return False


def _novelty(state: dict) -> tuple[float, str]:
    """1 minus the closest memory cosine.

    Box 5 writes dense_cos on each hit. No hit means the archive has
    nothing to subtract, so the claim is new. A cosine above 1 is a
    float artefact and is treated as 1, which makes novelty 0: an exact
    echo is not new. A missing matches key is the same as an empty list.
    Both mean there is no cosine, not that the search was skipped in a
    way this box can see.
    """
    best = 0.0
    hits = state.get("matches")
    if isinstance(hits, list):
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            try:
                cosine = float(hit.get("dense_cos") or 0.0)
            except (TypeError, ValueError):
                cosine = 0.0
            if cosine > best:
                best = cosine
    best = _unit(best)
    score = _unit(1.0 - best)
    if best <= 0.0:
        line = "Nothing close is in memory, so this reads as new."
    else:
        line = (
            f"The closest memory hit has cosine {best:.2f}, "
            f"so novelty is {score:.2f}."
        )
    return score, line


def _reach(state: dict) -> tuple[float, str]:
    """How far this has already travelled, squashed with a log.

    The first stored copy is the submission itself, so submission_count
    contributes only the repeats. also_debunked_by is taken from the
    closest memory hit: a loosely related fact-check's outlet list is a
    different claim and would inflate the crowd. Circulation sightings
    are social posts box 7 refused to treat as evidence; they still show
    that the sentence is moving.

    log1p keeps the second and third repost visible and stops the 50th
    from adding as much as the first. The cap is "already everywhere".
    """
    try:
        submissions = int(state.get("submission_count") or 1)
    except (TypeError, ValueError):
        submissions = 1
    repeats = max(0, submissions - 1)
    debunked = _debunked(state)
    stats = state.get("investigate_stats")
    sightings = 0
    if isinstance(stats, dict):
        try:
            sightings = max(0, int(stats.get("circulation_sightings") or 0))
        except (TypeError, ValueError):
            sightings = 0
    label = ""
    signals = state.get("style_signals")
    if isinstance(signals, dict) and signals.get("forwarded_label"):
        label = str(signals.get("forwarded_label"))
    if label == "Forwarded many times":
        forward = config.REACH_FORWARD_MANY
        forward_line = "WhatsApp marks it as forwarded many times."
    elif label == "Forwarded as received":
        forward = config.REACH_FORWARD_RECEIVED
        forward_line = "It is marked forwarded as received, which counts less than forwarded many times."
    else:
        forward = 0
        forward_line = "It has no WhatsApp forwarded label."
    raw = repeats + debunked + sightings + forward
    score = _unit(math.log1p(raw) / math.log1p(config.REACH_LOG_CAP))
    line = (
        f"Reach is {score:.2f} from {_plural(repeats, 'repeat submission')}, "
        f"{_plural(debunked, 'other debunk')}, and {_plural(sightings, 'circulation sighting')}. "
        f"{forward_line}"
    )
    return score, line


def _plural(count: int, singular: str) -> str:
    word = singular if count == 1 else singular + "s"
    return f"{count} {word}"


def _debunked(state: dict) -> int:
    hits = state.get("matches")
    if not isinstance(hits, list) or not hits:
        return 0
    best = None
    best_cos = -1.0
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        try:
            cosine = float(hit.get("dense_cos") or 0.0)
        except (TypeError, ValueError):
            cosine = 0.0
        if best is None or cosine > best_cos:
            best = hit
            best_cos = cosine
    others = best.get("also_debunked_by") if isinstance(best, dict) else None
    if not isinstance(others, list):
        return 0
    return len(others)


def _urgency(state: dict) -> tuple[float, str]:
    """Time pressure. This is packaging, and it stops at this box.

    Urgency words and a share request come from ingest's style_signals.
    A near date or a deadline word is read from the claim because those
    are about when the alleged event happens, which the style counter
    does not parse. The three pieces share one cap so a message that
    says all of them is urgent once, not three times.
    """
    signals = state.get("style_signals")
    if not isinstance(signals, dict):
        signals = {}
    words = signals.get("urgency_words") or []
    if not isinstance(words, list):
        words = []
    word_points = min(config.URGENCY_WORD_CAP, config.URGENCY_PER_WORD * len(words))
    share = config.URGENCY_SHARE if signals.get("share_request") else 0.0
    deadline = config.URGENCY_DEADLINE if _deadline(state) else 0.0
    score = _unit(word_points + share + deadline)
    bits = []
    if words:
        bits.append("urgency words: " + ", ".join(str(word) for word in words))
    if share:
        bits.append("a request to forward it")
    if deadline:
        bits.append("a near date or a deadline")
    if not bits:
        line = "Nothing in the wording is time-critical, so urgency is 0."
    else:
        line = f"Urgency is {score:.2f} from " + " and ".join(bits) + "."
    return score, line


def _deadline(state: dict) -> bool:
    text = " ".join(
        str(state.get(key) or "")
        for key in ("claim", "text", "raw_text")
    )
    if _DEADLINE.search(text):
        return True
    return _near_date(state, text)


def _near_date(state: dict, text: str) -> bool:
    """A date in the claim that falls within URGENCY_NEAR_DAYS of claim_date.

    claim_date is when the rumour was circulating, not today. The eval
    sets it to the fact-check's day, and a deadline the following week
    is then urgent in a way a date from 2016 is not when the claim is
    checked in 2026. With no claim_date there is nothing to be near.
    """
    anchor = _parse_iso(state.get("claim_date"))
    if anchor is None or not text:
        return False
    window = config.URGENCY_NEAR_DAYS
    for match in _ISO.finditer(text):
        found = _make_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if found is not None and abs((found - anchor).days) <= window:
            return True
    for match in _DATE_DM.finditer(text):
        if _close_month_day(anchor, int(match.group(1)), _MONTHS[match.group(2).lower()], window):
            return True
    for match in _DATE_MD.finditer(text):
        if _close_month_day(anchor, int(match.group(2)), _MONTHS[match.group(1).lower()], window):
            return True
    return False


def _close_month_day(anchor: date, day: int, month: int, window: int) -> bool:
    """Try this year and the two neighbours.

    "1 December" has no year. The reading that sits closest to the day
    the claim circulated is the one the writer meant. A December date
    checked in January is the December just past or the one ahead, not
    a date centuries off.
    """
    best = None
    for year in (anchor.year - 1, anchor.year, anchor.year + 1):
        found = _make_date(year, month, day)
        if found is None:
            continue
        gap = abs((found - anchor).days)
        if best is None or gap < best:
            best = gap
    return best is not None and best <= window


def _parse_iso(value) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    return _make_date_from_iso(value[:10])


def _make_date_from_iso(text: str) -> date | None:
    try:
        year, month, day = (int(part) for part in text.split("-"))
    except ValueError:
        return None
    return _make_date(year, month, day)


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
