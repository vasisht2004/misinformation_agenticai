"""Boxes 1–2: turn a submission into clean text and a duplicate id.

No model call lives here. The verdict is decided later, from evidence.
What this box records about shouting, forwarding, and named authorities
is a warning sign for the result card and for priority (box 11). It is
not a reason to call the claim false.

Scope of this module: English text, a URL, or an image. Video and audio
are out of scope and are rejected rather than treated as a paragraph.

The submissions database stores a hash and a MinHash of the cleaned
text. It does not store the message, a name, or a phone number. The
raw string stays in the run's memory (raw_text / raw_input) because
later boxes have to read it. It is not written to SQLite.
"""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
import threading
import unicodedata
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

import misinfo_agent.config as config


def _seed_langdetect() -> None:
    """Make langdetect deterministic.

    Without a seed the library samples, and the same paragraph can come
    back as a different language on the next call. Seed 0 is set at
    import, before any detect() in this process, and again just before
    each call in case another module cleared it.
    """
    try:
        from langdetect import DetectorFactory

        DetectorFactory.seed = 0
    except Exception:
        return


_seed_langdetect()

# One writer at a time. Two identical forwards arriving together must
# share one row; a second insert would look like two rumours.
_DB_LOCK = threading.Lock()

# datasketch default. Fixed so a signature written today still matches
# a forward submitted next week. 128 permutations is the brief: enough
# that a Jaccard estimate near 0.8 is not noise from a tiny sketch.
_MINHASH_PERM = 128
_MINHASH_SEED = 1
# Character shingles, not word shingles. A short claim has too few words
# for a 5-word window, and an inserted "please" should not look like a
# different forward. Five characters is the width the brief asks for.
_SHINGLE_CHARS = 5
# Estimated Jaccard at or above this is the same forward. Exact copies
# score 1. A reworded rumour scores lower and stays a new row.
_JACCARD_SAME = 0.8

_TOO_SHORT = "too short to check"
_OUT_OF_SCOPE = "out of scope (not English)"
# Below this the detector is guessing, and a greeting would be labelled
# a foreign language. Length is the reason the brief gives for that case.
_MIN_CONTENT_WORDS = 5

# Bodies Indian forwards name instead of linking a notice. WHO is
# capitals only: the lowercase word "who" is ordinary English, and
# counting it would mark half of all sentences as an authority claim.
# The match is dropped when a URL sits next to the name. A bare name
# is the warning sign; a link is something the reader can open.
_AUTHORITIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AIIMS", re.compile(r"\bAIIMS\b", re.IGNORECASE)),
    ("RBI", re.compile(r"\bRBI\b", re.IGNORECASE)),
    ("WHO", re.compile(r"\bWHO\b")),
    ("PIB", re.compile(r"\bPIB\b", re.IGNORECASE)),
    ("police", re.compile(r"\bpolice\b", re.IGNORECASE)),
    ("ICMR", re.compile(r"\bICMR\b", re.IGNORECASE)),
    ("ISRO", re.compile(r"\bISRO\b", re.IGNORECASE)),
    ("NASA", re.compile(r"\bNASA\b", re.IGNORECASE)),
    ("SEBI", re.compile(r"\bSEBI\b", re.IGNORECASE)),
    ("CBI", re.compile(r"\bCBI\b")),
    ("NIA", re.compile(r"\bNIA\b")),
    ("UNESCO", re.compile(r"\bUNESCO\b", re.IGNORECASE)),
    ("UNICEF", re.compile(r"\bUNICEF\b", re.IGNORECASE)),
    ("IMF", re.compile(r"\bIMF\b")),
    ("CDC", re.compile(r"\bCDC\b")),
    ("Supreme Court", re.compile(r"\bSupreme\s+Court\b", re.IGNORECASE)),
    ("Election Commission", re.compile(r"\bElection\s+Commission\b", re.IGNORECASE)),
    ("World Bank", re.compile(r"\bWorld\s+Bank\b", re.IGNORECASE)),
)

# How close a URL has to be before we treat the authority as cited.
# The window is characters, not words: forwards paste the link on the
# same line as the name.
_AUTHORITY_URL_WINDOW = 40
_URL_IN_TEXT = re.compile(r"https?://|www\.", re.IGNORECASE)

# Shouting words. Counted so priority and the card can show them.
# They are not removed from the claim except where the brief says so
# (the forward label and the share line are removed; "BREAKING" stays
# in the text and is also listed here).
_URGENCY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("breaking", re.compile(r"\bbreaking\b", re.IGNORECASE)),
    ("urgent", re.compile(r"\burgent\b", re.IGNORECASE)),
    ("alert", re.compile(r"\balert\b", re.IGNORECASE)),
    ("immediately", re.compile(r"\bimmediately\b", re.IGNORECASE)),
    ("last date", re.compile(r"\blast\s+date\b", re.IGNORECASE)),
)

# "Forwarded many times" and "Forwarded as received" are WhatsApp's own
# labels. They describe the message's travels, not its claim. The value
# stored is one of those two spellings, or None.
_FORWARDED_MANY = re.compile(r"\bforwarded\s+many\s+times\b", re.IGNORECASE)
_FORWARDED_RECEIVED = re.compile(r"\bforwarded\s+as\s+received\b", re.IGNORECASE)
_LABEL_MANY = "Forwarded many times"
_LABEL_RECEIVED = "Forwarded as received"

# A request to pass the message on, and only that. "HDFC Bank share
# price fell" is a sentence about a stock. Deleting the word "share"
# would invent a different claim. The pattern needs the verb
# (forward / share / send) and a target (groups, family, friends).
# Two copies of one rumour, one of which says "send to 10 groups",
# then clean to the same match_text.
_SHARE_REQUESTS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bforward(?:ed)?\s+to\s+(?:all\s+)?(?:your\s+)?(?:\d+\s+)?"
        r"(?:groups|group|friends|friend|family|contacts|everyone|everybody|relatives)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bforward\s+to\s+all\b", re.IGNORECASE),
    re.compile(
        r"\bshare\s+(?:this\s+|it\s+)?(?:with|to)\s+(?:all\s+|your\s+)?"
        r"(?:groups|group|friends|friend|family|contacts|everyone|everybody|relatives)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsend\s+(?:this\s+|it\s+)?to\s+\d+\s+groups\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsend\s+to\s+\d+\s+groups\b", re.IGNORECASE),
)

_CONSPIRACY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hiding", re.compile(r"\bhiding\b", re.IGNORECASE)),
    (
        "they don't want you to know",
        re.compile(r"\bthey\s+don'?t\s+want\s+you\s+to\s+know\b", re.IGNORECASE),
    ),
    (
        "media won't show",
        re.compile(r"\bmedia\s+won'?t\s+show\b", re.IGNORECASE),
    ),
)

_CERTAINTY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("100%", re.compile(r"\b100\s*(?:%|percent)\b", re.IGNORECASE)),
    ("guaranteed", re.compile(r"\bguaranteed\b", re.IGNORECASE)),
    ("confirmed", re.compile(r"\bconfirmed\b", re.IGNORECASE)),
    ("cures", re.compile(r"\bcures?\b", re.IGNORECASE)),
)

# Pictographs only. Variation selectors and the double-exclamation mark
# are decoration, counted under exclaim_count, not as emoji. A skin-tone
# modifier is not a second emoji.
_EMOJI_CHAR = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]"
)
_ORNAMENT = frozenset("‼⁉")
# A rule of stars or dashes is a banner, not a sentence.
_BANNER_LINE = re.compile(r"[_*=~#\-–—•]{3,}")

# Hindi words written in Latin letters. Short English homographs are
# absent on purpose: "main", "log", and "par" would flag ordinary
# English. One distinctive hit is enough. Hinglish is still checked;
# the flag only tells the card the text mixes the two.
_HINGLISH_WORDS = frozenset(
    {
        "hai",
        "hain",
        "nahi",
        "nahin",
        "kya",
        "kyun",
        "kyu",
        "kaun",
        "kahan",
        "yeh",
        "aap",
        "tum",
        "hum",
        "bahut",
        "bohot",
        "abhi",
        "jaldi",
        "bhai",
        "yaar",
        "accha",
        "achha",
        "theek",
        "thik",
        "haan",
        "bilkul",
        "chahiye",
        "chahie",
        "karo",
        "kiya",
        "kiye",
        "hoga",
        "hogi",
        "dekho",
        "suno",
        "sarkar",
        "wala",
        "wali",
        "wale",
        "waala",
        "lekin",
        "magar",
        "namaste",
        "rupaye",
        "paisa",
        "mein",
        "bhi",
        "naya",
        "kal",
        "diya",
        # Function words. langdetect calls
        # "Modi ji ne bola ki 500 ke note band ho jayenge ..." Slovenian.
        # These tokens are how that sentence is recognised as Hinglish
        # before the detector's label is allowed to reject it.
        # "band" and "note" are ordinary English and are not in this set.
        "ji",
        "ne",
        "ki",
        "ke",
        "ka",
        "ko",
        "ho",
        "bola",
        "bol",
        "jayenge",
        "jayega",
        "gaye",
        "gaya",
        "gayi",
        "toh",
        "matlab",
        "aur",
    }
)
_LATIN_WORD = re.compile(r"[A-Za-z]+")
# A "content word" is a run of letters. Digits stay in the text (amounts
# and dates are the claim) but they are not words, and a 2-digit token
# must not push a greeting over the length bar. Two letters so "OK" counts
# and a stray "a" does not.
_CONTENT_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    sha256 TEXT PRIMARY KEY,
    minhash BLOB NOT NULL,
    first_seen TEXT NOT NULL,
    count INTEGER NOT NULL
)
"""
# Same file as submissions, different rows. An image is identified by
# its pixels, not by the caption, so it does not share the text table.
# A caption that happens to match some forward must not increment that
# forward's count.
_IMAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    sha256 TEXT PRIMARY KEY,
    phash TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    count INTEGER NOT NULL
)
"""

# The part of an article that becomes `text`. The full body stays in
# `body`. 1200 characters is the lede plus the first section, which is
# where a news claim is stated. The rest is background.
_BODY_CHARS = 1200

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGE_SIDE = 4000
# What the later model call is allowed to see. A 4000px screenshot is
# accepted, then reduced. OCR runs on this copy too, so the words and
# the pixels the model sees are the same image.
_LLM_IMAGE_SIDE = 1600
# imagehash.phash bits. A distance at or under this is the same picture
# with a recompress or a small crop, not a different photograph.
_PHASH_SAME = 6

# Posts. The page around them is a login wall and a comment thread.
# og:title and og:description are the words the person shared.
_SOCIAL_HOSTS = frozenset(
    {
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
        "youtube.com",
        "youtu.be",
        "t.me",
        "telegram.me",
        "reddit.com",
        "linkedin.com",
        "whatsapp.com",
        "threads.net",
        "tiktok.com",
    }
)
_FETCH_FAILED = "could not fetch the page"
_NO_ARTICLE = "no article text"


def ingest(
    raw: str | bytes,
    input_type: str = "text",
    *,
    db_path: Path | str | None = None,
    caption: str | None = None,
) -> dict:
    """Box 1–2 for one submission.

    db_path defaults to config.SUBMISSIONS_DB. Tests override that path
    in conftest, so a forgotten argument still cannot write the file
    under data/. When config.RECORD_SUBMISSIONS is False the database
    is not opened at all.

    Returns the keys this box writes. raw_text is the in-memory copy of
    a text submission or of a URL. An image's raw_text is the caption.
    The database row is a hash, never the message.
    """
    if input_type == "url":
        if not isinstance(raw, str):
            raise TypeError("url ingest expects a str")
        return _ingest_url(raw, db_path=db_path)
    if input_type == "image":
        return _ingest_image(raw, caption=caption, db_path=db_path)
    if input_type != "text":
        raise ValueError(f"input_type must be 'text', 'url', or 'image', got {input_type!r}")
    if not isinstance(raw, str):
        raise TypeError("text ingest expects a str")
    return _assemble(
        input_type="text",
        raw_text=raw,
        style_source=raw,
        to_clean=raw,
        db_path=db_path,
    )


def _assemble(
    *,
    input_type: str,
    raw_text: str,
    style_source: str,
    to_clean: str,
    db_path: Path | str | None,
    headline: str = "",
    body: str = "",
    article_date: str = "",
    forced_reason: str | None = None,
    store: bool = True,
    image_key: tuple[str, str] | None = None,
    fields: dict | None = None,
    count_emoji: bool = True,
) -> dict:
    """One shape for text, a page, and an image.

    style_source is the raw characters the warning signs are counted on.
    to_clean is what becomes `text`. For a forward they are the same
    string. For a URL, style sees the headline and the first two
    paragraphs, and text is the headline plus the first 1200 characters.
    Cleaning runs after the
    signals, because cleaning deletes the marks the signals count.

    store is False when there is nothing to remember (the page did not
    load). An empty match_text must not become a shared row.
    """
    signals = _style_signals(style_source, count_emoji=count_emoji)
    cleaned = _clean(to_clean)
    match = _match_text(cleaned)
    language, hinglish, scope_reason = _language(cleaned)
    words = _CONTENT_WORD.findall(cleaned)
    # A non-English script is out of scope even when it is also short:
    # the project does not check Hindi, and "too short" would hide that.
    # A short Latin greeting is "too short". A forced reason is a fetch
    # that failed before there was text to judge.
    if forced_reason:
        checkable = False
        reason = forced_reason
    elif scope_reason:
        checkable = False
        reason = scope_reason
    elif len(words) < _MIN_CONTENT_WORDS:
        checkable = False
        reason = _TOO_SHORT
    else:
        checkable = True
        reason = ""
    if not store:
        submission_hash, submission_count, method = "", 0, "skipped"
    elif image_key is not None:
        submission_hash, submission_count, method = _record_image(
            image_key[0], image_key[1], db_path
        )
    else:
        submission_hash, submission_count, method = _record(match, db_path)
    flags = _flags(signals, hinglish=hinglish, checkable=checkable)
    result = {
        "input_type": input_type,
        "raw_text": raw_text,
        "raw_input": raw_text,
        "style_signals": signals,
        "text": cleaned,
        "match_text": match,
        "headline": headline,
        "body": body,
        "article_date": article_date,
        "language": language,
        "lang": language,
        "hinglish": hinglish,
        "checkable_hint": checkable,
        "checkable_reason": reason,
        "claim_date": date.today().isoformat(),
        "submission_hash": submission_hash,
        "submission_count": submission_count,
        "flags": flags,
        "quality": {
            "ok": checkable,
            "chars": len(cleaned),
            "words": len(words),
            "reason": reason,
        },
        "dedupe": _dedupe(submission_count, method, submission_hash),
        # No pixels. A text or URL run leaves these empty. An image run
        # overwrites them with the path and the OCR. The bytes stay in
        # the file named by image_sha256.
        "image_path": "",
        "image_sha256": "",
        "image_phash": "",
        "ocr_text": "",
        "ocr_mean_conf": None,
        "caption": "",
    }
    if fields:
        result.update(fields)
    return result


def _dedupe(count: int, method: str, digest: str) -> dict:
    # count 0 means nothing was written. seen_count is how many times
    # this forward was stored before this one, so a skip stays at 0
    # rather than becoming -1.
    recorded = count > 0
    return {
        "is_duplicate": recorded and count > 1,
        "seen_count": count - 1 if recorded else 0,
        "method": method,
        "submission_hash": digest,
    }


def _style_signals(raw: str, *, count_emoji: bool = True) -> dict:
    """Warning signs. Never an input to the verdict.

    count_emoji is false for a URL. A press-release page is full of
    share icons, and those are not the author shouting.
    """
    urgency = [label for label, pattern in _URGENCY if pattern.search(raw)]
    if _FORWARDED_MANY.search(raw):
        # "many times" wins when both labels were pasted. It is the
        # stronger of WhatsApp's two wordings.
        label = _LABEL_MANY
    elif _FORWARDED_RECEIVED.search(raw):
        label = _LABEL_RECEIVED
    else:
        label = None
    caps_ratio = _caps_ratio(raw)
    # U+203C is one character and two exclamation marks. Count it as two
    # so a banner of ‼ and a banner of !! weigh the same.
    exclaim = raw.count("!") + 2 * raw.count("‼") + raw.count("❗") + raw.count("⁉")
    return {
        "urgency_words": urgency,
        "share_request": any(pattern.search(raw) for pattern in _SHARE_REQUESTS),
        "forwarded_label": label,
        "caps_ratio": caps_ratio,
        "exclaim_count": exclaim,
        "emoji_count": len(_EMOJI_CHAR.findall(raw)) if count_emoji else 0,
        "authority_mentions": _authorities(raw),
        "conspiracy_phrases": [name for name, pattern in _CONSPIRACY if pattern.search(raw)],
        "certainty_words": [name for name, pattern in _CERTAINTY if pattern.search(raw)],
    }


# An all-caps token this short is a name (HSBC, RBI, AIIMS, PIB), not
# someone shouting. Longer banners (BREAKING) still count.
_ACRONYM_CAP = 5
# Alarm is the chip a reader sees. Above 0.30 of the remaining letters,
# or three exclamation marks, or three emoji. Two of either is just emphasis.
_ALARM_CAPS = 0.30
_ALARM_EXCLAIM = 3
_ALARM_EMOJI = 3


def alarm_style(signals: dict) -> bool:
    """True when the wording is a banner, not a normal sentence.

    caps_ratio already ignores short acronyms, so this bar is about the
    rest of the letters. A fact name like HSBC does not light it. The
    garlic forward still does, because of the exclamation marks.
    """
    signals = signals or {}
    caps = float(signals.get("caps_ratio") or 0)
    exclaim = int(signals.get("exclaim_count") or 0)
    emoji = int(signals.get("emoji_count") or 0)
    return caps > _ALARM_CAPS or exclaim >= _ALARM_EXCLAIM or emoji >= _ALARM_EMOJI


def _caps_ratio(raw: str) -> float:
    """Share of letters in capitals, with short acronyms left out.

    Both the capitals and the letters of HSBC / RBI / AIIMS / PIB are
    skipped. Counting them made a plain sentence about HSBC look shouted.
    """
    upper = 0
    total = 0
    for token in re.findall(r"[A-Za-z]+", raw or ""):
        if token.isupper() and len(token) <= _ACRONYM_CAP:
            continue
        total += len(token)
        upper += sum(ch.isupper() for ch in token)
    if not total:
        return 0.0
    return round(upper / total, 4)


def _authorities(raw: str) -> list[str]:
    found: list[str] = []
    for name, pattern in _AUTHORITIES:
        match = pattern.search(raw)
        if match is None:
            continue
        start = max(0, match.start() - _AUTHORITY_URL_WINDOW)
        end = min(len(raw), match.end() + _AUTHORITY_URL_WINDOW)
        if _URL_IN_TEXT.search(raw[start:end]):
            continue
        found.append(name)
    return found


def _clean(raw: str) -> str:
    """Readable text. Rupees, digits, and dates stay. The trailer does not.

    NFKC first so a fullwidth "ＢＲＥＡＫＩＮＧ" and a composed accent fold to
    the same characters as a normal keyboard. Style signals already ran
    on the original string, so this fold cannot hide a banner from them.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _FORWARDED_MANY.sub(" ", text)
    text = _FORWARDED_RECEIVED.sub(" ", text)
    for pattern in _SHARE_REQUESTS:
        text = pattern.sub(" ", text)
    text = _BANNER_LINE.sub(" ", text)
    kept: list[str] = []
    for ch in text:
        if _drop_char(ch):
            continue
        kept.append(ch)
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def _drop_char(ch: str) -> bool:
    """Emoji, variation selectors, and ornamental symbols.

    Currency (₹ is Sc) is kept. Punctuation is kept in the readable
    text; match_text is what drops it. Accents (Mn) are kept too:
    only the variation-selector block is a decoration.
    """
    code = ord(ch)
    if 0xFE00 <= code <= 0xFE0F or code == 0x200D:
        return True
    if 0x1F3FB <= code <= 0x1F3FF:
        return True
    if ch in _ORNAMENT:
        return True
    if _EMOJI_CHAR.match(ch):
        return True
    category = unicodedata.category(ch)
    # So/Sk are symbols (emoji already caught, plus stars and dingbats).
    # Cf is zero-width and bidi marks. A claim does not need them.
    return category in {"So", "Sk", "Cf"}


def _match_text(cleaned: str) -> str:
    """Lowercased, punctuation replaced with spaces, whitespace collapsed.

    Replacing rather than deleting keeps "2016" from gluing onto the next
    word. The rupee sign is a currency symbol, not punctuation, so ₹500
    survives and two notes about different amounts do not share a hash.
    """
    parts: list[str] = []
    for ch in cleaned:
        if unicodedata.category(ch).startswith("P"):
            parts.append(" ")
        else:
            parts.append(ch.lower())
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _language(cleaned: str) -> tuple[str, bool, str | None]:
    """Return (language code, hinglish, scope reason or None).

    The script check comes before langdetect. A Devanagari forward is
    out of scope even if the detector shrugs, and a Latin forward is
    not Hindi. Hinglish is decided by the word list, and that decision
    wins over langdetect. The detector labels
    "Modi ji ne bola ki 500 ke note band ho jayenge from 1 December"
    as Slovenian. The words are Hindi written in Latin letters, and
    the run continues.
    """
    latin, devanagari, other = _script_counts(cleaned)
    letters = latin + devanagari + other
    if letters == 0:
        return "unknown", False, None
    if devanagari + other > latin:
        code = _detect_lang(cleaned) or "unknown"
        if code == "en":
            # The letters disagree with the detector. Trust the letters.
            code = "hi" if devanagari >= other else "unknown"
        return code, False, _OUT_OF_SCOPE
    hinglish = _is_hinglish(cleaned)
    if hinglish:
        # Still checked. language stays "en" so a downstream "not English"
        # branch does not drop a Hinglish forward the heuristic accepted.
        return "en", True, None
    code = _detect_lang(cleaned)
    if code is None or code == "en":
        # A detector failure is not evidence of another language.
        return "en", False, None
    return code, False, _OUT_OF_SCOPE


def _script_counts(text: str) -> tuple[int, int, int]:
    latin = devanagari = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        # Latin including Latin-1 and Latin Extended. Devanagari is the
        # Hindi block. Everything else (Tamil, Arabic, CJK) is out of scope.
        if code <= 0x024F or 0x1E00 <= code <= 0x1EFF:
            latin += 1
        elif 0x0900 <= code <= 0x097F:
            devanagari += 1
        else:
            other += 1
    return latin, devanagari, other


def _is_hinglish(text: str) -> bool:
    for word in _LATIN_WORD.findall(text):
        if word.lower() in _HINGLISH_WORDS:
            return True
    return False


def _detect_lang(text: str) -> str | None:
    """langdetect, seeded.

    The library draws a random sample unless the seed is set, so the
    same English paragraph can come back as another language on the
    next call. Seed 0 makes a rerun of this box return the same code.
    """
    try:
        from langdetect import DetectorFactory, detect

        # Again, in case another import reset the factory after ours.
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return None


def _flags(signals: dict, *, hinglish: bool, checkable: bool) -> list[str]:
    flags: list[str] = []
    if hinglish:
        flags.append("hinglish")
    if signals["share_request"]:
        flags.append("call_to_forward")
    if signals["forwarded_label"]:
        flags.append("forwarded_label")
    if signals["urgency_words"]:
        flags.append("urgency")
    if not checkable:
        flags.append("not_checkable")
    return flags


def _record(match_text: str, db_path: Path | str | None) -> tuple[str, int, str]:
    """Store the hash. Return (hash, count including this copy, method).

    Exact sha256 is checked first. It is the same forward with the same
    words. MinHash is the near copy: Jaccard >= 0.8 joins this text to
    the older row and increments that row, so the count is "how many
    times have we seen this rumour", not "how many spellings".

    The row written here is sha256, minhash bytes, first_seen, count.
    match_text itself is not inserted.
    """
    digest = hashlib.sha256(match_text.encode("utf-8")).hexdigest()
    if not config.RECORD_SUBMISSIONS:
        # Not opened, not created. An evaluation run must not add rows
        # or bump a count that box 11 will later read as reach.
        return digest, 0, "skipped"
    signature = _minhash(match_text)
    blob = np.asarray(signature.hashvalues, dtype=np.uint64).tobytes()
    seen = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = _resolve_db(db_path)
    with _DB_LOCK:
        # isolation_level None is autocommit, so BEGIN is the only
        # transaction. The lock above is the process; BEGIN IMMEDIATE
        # is the file, for a second process using the same database.
        conn = sqlite3.connect(path, isolation_level=None, timeout=5)
        try:
            conn.execute(_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT count FROM submissions WHERE sha256 = ?",
                    (digest,),
                ).fetchone()
                if row is not None:
                    count = int(row[0]) + 1
                    conn.execute(
                        "UPDATE submissions SET count = ? WHERE sha256 = ?",
                        (count, digest),
                    )
                    conn.execute("COMMIT")
                    return digest, count, "sha256"
                best_hash: str | None = None
                best_jaccard = 0.0
                best_count = 0
                for stored_hash, stored_blob, stored_count in conn.execute(
                    "SELECT sha256, minhash, count FROM submissions"
                ):
                    try:
                        other = _minhash_from_blob(stored_blob)
                        score = signature.jaccard(other)
                    except Exception:
                        # A row this version cannot read must not block
                        # the new forward. It is skipped, not deleted.
                        continue
                    if score > best_jaccard:
                        best_jaccard = score
                        best_hash = stored_hash
                        best_count = int(stored_count)
                if best_hash is not None and best_jaccard >= _JACCARD_SAME:
                    count = best_count + 1
                    conn.execute(
                        "UPDATE submissions SET count = ? WHERE sha256 = ?",
                        (count, best_hash),
                    )
                    conn.execute("COMMIT")
                    return best_hash, count, "minhash"
                conn.execute(
                    "INSERT INTO submissions (sha256, minhash, first_seen, count) "
                    "VALUES (?, ?, ?, 1)",
                    (digest, blob, seen),
                )
                conn.execute("COMMIT")
                return digest, 1, "minhash"
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()


def _resolve_db(db_path: Path | str | None) -> Path:
    # config.SUBMISSIONS_DB, not a path baked in at import. conftest
    # points that attribute at a temp file for the duration of a test.
    path = Path(db_path) if db_path is not None else Path(config.SUBMISSIONS_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _minhash(text: str):
    from datasketch import MinHash

    sketch = MinHash(num_perm=_MINHASH_PERM, seed=_MINHASH_SEED)
    for shingle in _shingles(text):
        sketch.update(shingle.encode("utf-8"))
    return sketch


def _minhash_from_blob(blob: bytes):
    from datasketch import MinHash

    values = np.frombuffer(blob, dtype=np.uint64)
    if values.size != _MINHASH_PERM:
        raise ValueError("minhash width")
    # frombuffer is a view of bytes that must not be written. Copy
    # before handing the array to MinHash.
    return MinHash(num_perm=_MINHASH_PERM, seed=_MINHASH_SEED, hashvalues=values.copy())


def _shingles(text: str) -> list[str]:
    if len(text) < _SHINGLE_CHARS:
        # One shingle so an empty or tiny string still has a signature.
        # Two empty match_texts then compare as equal, which they are.
        return [text or " "]
    return [text[i : i + _SHINGLE_CHARS] for i in range(len(text) - _SHINGLE_CHARS + 1)]


def _record_image(file_sha: str, phash: str, db_path: Path | str | None) -> tuple[str, int, str]:
    """Same rules as _record, for pixels.

    Exact sha256 of the file first. phash Hamming distance at or under
    6 joins a recompressed copy to the older row. The row is the hash,
    the phash, first_seen, and count. The pixels are not inserted.
    """
    if not config.RECORD_SUBMISSIONS:
        return file_sha, 0, "skipped"
    seen = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = _resolve_db(db_path)
    with _DB_LOCK:
        conn = sqlite3.connect(path, isolation_level=None, timeout=5)
        try:
            conn.execute(_IMAGE_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT count FROM images WHERE sha256 = ?",
                    (file_sha,),
                ).fetchone()
                if row is not None:
                    count = int(row[0]) + 1
                    conn.execute(
                        "UPDATE images SET count = ? WHERE sha256 = ?",
                        (count, file_sha),
                    )
                    conn.execute("COMMIT")
                    return file_sha, count, "sha256"
                best_hash: str | None = None
                best_distance = _PHASH_SAME + 1
                best_count = 0
                for stored_hash, stored_phash, stored_count in conn.execute(
                    "SELECT sha256, phash, count FROM images"
                ):
                    try:
                        distance = _phash_distance(phash, stored_phash)
                    except Exception:
                        continue
                    if distance < best_distance:
                        best_distance = distance
                        best_hash = stored_hash
                        best_count = int(stored_count)
                if best_hash is not None and best_distance <= _PHASH_SAME:
                    count = best_count + 1
                    conn.execute(
                        "UPDATE images SET count = ? WHERE sha256 = ?",
                        (count, best_hash),
                    )
                    conn.execute("COMMIT")
                    return best_hash, count, "phash"
                conn.execute(
                    "INSERT INTO images (sha256, phash, first_seen, count) VALUES (?, ?, ?, 1)",
                    (file_sha, phash, seen),
                )
                conn.execute("COMMIT")
                return file_sha, 1, "phash"
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()


def _phash_distance(left: str, right: str) -> int:
    import imagehash

    # imagehash defines subtraction as the Hamming distance of the two
    # bit strings. That is the comparison the duplicate rule uses.
    return int(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right))


def _ingest_url(raw: str, *, db_path: Path | str | None) -> dict:
    original = raw.strip()
    url = _strip_tracking(original)
    try:
        html = _fetch_html(url)
    except Exception as exc:
        from misinfo_agent.fetch import FetchError

        if not isinstance(exc, FetchError):
            raise
        return _assemble(
            input_type="url",
            raw_text=original,
            style_source="",
            to_clean="",
            db_path=db_path,
            forced_reason=_FETCH_FAILED,
            store=False,
        )
    if _is_social(url):
        # The comment thread under a post is not the claim. The title
        # and description the page offers to a preview are.
        headline, body = _open_graph(html)
        article_date = ""
    else:
        headline, body, article_date = _article_fields(html, url)
    if not headline and not body:
        return _assemble(
            input_type="url",
            raw_text=original,
            style_source="",
            to_clean="",
            db_path=db_path,
            forced_reason=_NO_ARTICLE,
            store=False,
        )
    # Signals see the headline and the first two paragraphs only. The
    # rest of a release is boilerplate and icon rows, and a warning sign
    # buried there is not how the page was shared. Emoji are not counted
    # at all on a URL: they are the site's icons. The hash is of the
    # clipped text, which is what the later boxes will read.
    clipped = _join_parts(headline, body[:_BODY_CHARS])
    return _assemble(
        input_type="url",
        raw_text=original,
        style_source=_join_parts(headline, _lead_paragraphs(body, 2)),
        to_clean=clipped,
        db_path=db_path,
        headline=headline,
        body=body,
        article_date=article_date,
        count_emoji=False,
    )


def _strip_tracking(url: str) -> str:
    """The same canonical form box 7 uses, so one article is one string.

    Host case, a leading www, the fragment, and utm_* / fbclid / gclid
    do not make a second page. Defined once, in investigate.canonical_url.
    """
    from misinfo_agent.investigate import canonical_url

    return canonical_url(url) or url.strip()


def _fetch_html(url: str) -> str:
    """Download through fetch.py. That module owns SSRF checks and the deadline.

    There is no second HTTP client here. A private address, a redirect
    onto one, a slow body, and an oversized body all fail inside
    _download before this function sees any bytes.
    """
    from misinfo_agent.fetch import FetchError, _charset, _download

    # final is the URL after redirects. raw_text stays the URL the
    # person pasted, so a hop does not replace what they submitted.
    _final, payload, ctype = _download(url)
    folded = (ctype or "").casefold()
    if "html" not in folded:
        raise FetchError("non-HTML")
    charset = _charset(ctype) or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _is_social(url: str) -> bool:
    from misinfo_agent.search import registrable_domain, site_host

    host = registrable_domain(site_host(url))
    return host in _SOCIAL_HOSTS


class _OpenGraph(HTMLParser):
    """og:title and og:description, attribute order aside.

    A social page's <title> is often "Log in". The preview tags are the
    words the share card showed. Nothing else from the page is kept.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.og: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "meta":
            return
        found = {key.lower(): (value or "") for key, value in attrs}
        name = (found.get("property") or found.get("name") or "").lower()
        if name in {"og:title", "og:description"}:
            self.og[name] = found.get("content", "").strip()


def _open_graph(html: str) -> tuple[str, str]:
    parser = _OpenGraph()
    try:
        parser.feed(html)
    except Exception:
        return "", ""
    return parser.og.get("og:title", ""), parser.og.get("og:description", "")


def _article_fields(html: str, url: str) -> tuple[str, str, str]:
    """trafilatura's headline, body, and date. Empty when it finds no article."""
    from trafilatura import bare_extraction

    try:
        document = bare_extraction(html, url=url, with_metadata=True)
    except Exception:
        return "", "", ""
    if document is None:
        return "", "", ""
    if hasattr(document, "as_dict"):
        data = document.as_dict()
    elif isinstance(document, dict):
        data = document
    else:
        return "", "", ""
    headline = str(data.get("title") or "").strip()
    # trafilatura's text starts with the title again. Stored body is the
    # article without that repeat, so `text` can be headline plus body
    # without saying the headline twice.
    body = str(data.get("text") or "").strip()
    lines = body.splitlines()
    if headline and lines and lines[0].strip().casefold() == headline.casefold():
        body = "\n".join(lines[1:]).strip()
    return headline, body, _iso_day(data.get("date"))


def _iso_day(value) -> str:
    if not value:
        return ""
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value))
    return match.group(0) if match else ""


def _lead_paragraphs(body: str, count: int) -> str:
    """The opening of the article, not the share buttons further down.

    trafilatura separates paragraphs with blank lines, or with single
    newlines when the page did not. Either way the first two blocks are
    the lede. A later "BREAKING" in a footer is not the article's tone.
    """
    text = (body or "").strip()
    if not text or count <= 0:
        return ""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        blocks = [line.strip() for line in text.splitlines() if line.strip()] or blocks
    return "\n\n".join(blocks[:count])


def _join_parts(headline: str, body: str) -> str:
    parts = [part.strip() for part in (headline, body) if part and part.strip()]
    return "\n".join(parts)


def _ingest_image(raw, *, caption: str | None, db_path: Path | str | None) -> dict:
    _original, upright = _open_image(raw)
    import imagehash

    # phash of the upright picture, before the 1600px shrink. The hash
    # resizes to 32px internally, so a smaller copy of the same photo
    # lands on the same bits. EXIF orientation is applied first, or a
    # phone's sideways flag would look like a different image.
    phash = str(imagehash.phash(upright))
    shown = _downscale(upright)
    # The file is the picture extract will load. Its sha256 is the name,
    # so a second submit of the same pixels does not write a second file.
    # The original upload's bytes are not kept: they may hold GPS.
    jpeg = _jpeg_bytes(shown)
    digest, stored = _save_upload(jpeg)
    ocr_text, mean_conf = _ocr(shown)
    caption_text = (caption or "").strip()
    # The caption is the only text the person typed. A filesystem path
    # is not stored on the state: a path under /Users contains a name.
    # image_path below is the path we just chose, under UPLOADS_DIR.
    combined = _join_parts(ocr_text, caption_text)
    return _assemble(
        input_type="image",
        raw_text=caption_text,
        style_source=combined,
        to_clean=combined,
        db_path=db_path,
        image_key=(digest, phash),
        fields={
            "image_path": str(stored),
            "image_sha256": digest,
            "image_phash": phash,
            "ocr_text": ocr_text,
            "ocr_mean_conf": mean_conf,
            "caption": caption_text,
        },
    )


def _open_image(raw) -> tuple[bytes, object]:
    """Bytes and an upright image, or ValueError.

    The limits are the brief: 10 MB, 4000 px on a side, and verify()
    before the pixels are trusted. A text file fails verify or is not
    a format Pillow knows, and is rejected. A decompression bomb is
    the same rejection: we do not decode it.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    data = _image_bytes(raw)
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("image larger than 10 MB")
    previous = Image.MAX_IMAGE_PIXELS
    # Just above a 4000×4000 image. A file that claims more pixels than
    # that is refused before Pillow allocates them.
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_SIDE * _MAX_IMAGE_SIDE
    try:
        try:
            probe = Image.open(io.BytesIO(data))
            probe.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("not an image") from exc
        try:
            opened = Image.open(io.BytesIO(data))
            opened.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("not an image") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    if max(opened.size) > _MAX_IMAGE_SIDE:
        raise ValueError("image side longer than 4000 px")
    upright = ImageOps.exif_transpose(opened) or opened
    if max(upright.size) > _MAX_IMAGE_SIDE:
        raise ValueError("image side longer than 4000 px")
    if upright.mode not in ("RGB", "L"):
        upright = upright.convert("RGB")
    return data, upright


def _image_bytes(raw) -> bytes:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, (str, Path)):
        path = Path(raw)
        if not path.is_file():
            raise ValueError("not an image")
        if path.stat().st_size > _MAX_IMAGE_BYTES:
            raise ValueError("image larger than 10 MB")
        return path.read_bytes()
    raise TypeError("image ingest expects bytes or a path")


def _downscale(image):
    view = image.copy()
    if max(view.size) > _LLM_IMAGE_SIDE:
        view.thumbnail((_LLM_IMAGE_SIDE, _LLM_IMAGE_SIDE), resample=_lanczos())
    return view


def _lanczos():
    from PIL import Image

    return Image.Resampling.LANCZOS


def _jpeg_bytes(image) -> bytes:
    """A new JPEG built from pixels only.

    Saving the Pillow object would keep the info dict, and that dict is
    where EXIF and GPS live. frombytes has no info dict to copy. JPEG,
    not PNG: the file extract loads is the one the brief names.
    """
    from PIL import Image

    rgb = image.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _save_upload(jpeg: bytes) -> tuple[str, Path]:
    """Write the JPEG once, under config.UPLOADS_DIR.

    The directory is read at call time so a test can point it at a temp
    folder. The name is the sha256 of these bytes. If that file is
    already there, it is the same picture and is not written again.
    """
    digest = hashlib.sha256(jpeg).hexdigest()
    directory = Path(config.UPLOADS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.jpg"
    if not path.exists():
        path.write_bytes(jpeg)
    return digest, path


def _ocr(image) -> tuple[str, float]:
    """Words Tesseract is at least 60% sure of, and their mean confidence.

    The picture is made grayscale first. Colour is not the letters, and
    it adds edges Tesseract treats as characters. A short side under
    1000 px is doubled with LANCZOS. A WhatsApp screenshot's 16px type
    is then about 32px, which is a size the recogniser can read. The
    upscaled copy is discarded. The file on disk stays at most 1600 px,
    which is what the later model call is allowed to see.

    image_to_data returns a confidence for every token, and -1 for the
    gaps. A low-confidence guess is how a photo of a crowd becomes a
    sentence the checker then treats as the claim. Those tokens are
    dropped. The mean is over the words that were kept.
    """
    import pytesseract
    from PIL import Image

    gray = image.convert("L")
    short = min(gray.size)
    if 0 < short < 1000:
        gray = gray.resize(
            (gray.size[0] * 2, gray.size[1] * 2),
            Image.Resampling.LANCZOS,
        )
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("tesseract is not installed") from exc
    words: list[str] = []
    scores: list[float] = []
    for word, conf in zip(data.get("text") or [], data.get("conf") or []):
        token = (word or "").strip()
        if not token:
            continue
        try:
            score = float(conf)
        except (TypeError, ValueError):
            continue
        if score >= 60:
            words.append(token)
            scores.append(score)
    mean = round(sum(scores) / len(scores), 2) if scores else 0.0
    return " ".join(words), mean
