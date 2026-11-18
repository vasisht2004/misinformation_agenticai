"""Boxes 3–4: read the cleaned text and write the checkable claim.

One model call, through llm.call_structured, checked with Pydantic.
The claim is written in the same style as primary_claim in
prompts/prompt_v4a.txt so box 5 compares like with like.

The model does not decide the verdict. The only verdict this box writes
is Other, and only when there is nothing to check (a greeting, a joke,
an opinion, a prediction). Later boxes skip that run.

Style signals are warning signs. They are merged into red_flags when a
label fits, and kept as style_flags when none does. They are not evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

import misinfo_agent.config as config
from misinfo_agent.llm import LLMQuotaError, call_structured

# Re-exported so a caller that hits the daily cap can stop without
# importing the gateway. extract does not catch it: a quota error is
# the end of the run, not a claim to retry.
__all__ = ["LLMQuotaError", "extract"]

# A share of letters in capitals, or a run of exclamation marks, is the
# "Sensational style" label. The RBI sample forward sits near 0.06 from
# the letters in "RBI" and "December", so the bar is above a normal
# sentence and an ordinary proper noun.
_CAPS_SENSATIONAL = 0.15
_EXCLAIM_SENSATIONAL = 2
_EMOJI_SENSATIONAL = 2
# Certainty words that the red-flag list already has a name for.
# "confirmed" is not one of them: it is recorded as a style flag.
_MIRACLE_WORDS = frozenset({"100%", "guaranteed", "cures"})
# A motive is not an event. The same phrases ingest already flags, plus
# the "to protect pharma" ending the garlic forward uses. Box 7 will
# not search an atomic claim this matches.
_MOTIVE = re.compile(
    r"\b(?:hiding|hid(?:e|den)|cover[- ]?up|they don'?t want you to know|"
    r"media won'?t show|to protect (?:pharma|pharmaceutical))\b",
    re.IGNORECASE,
)


class AtomicClaim(BaseModel):
    """One allegation. Exactly one claim in the list is the core.

    checkable is false for a motive or a conspiracy ("the government is
    hiding it"). That sentence is not an event a page can confirm, and
    box 7 does not spend a search on it. The default is true so an older
    payload that omitted the flag is still searched.
    """

    id: str
    text: str
    core: bool = False
    checkable: bool = True


class Entities(BaseModel):
    """Names and numbers copied from the submission, not inferred."""

    people: list[str] = Field(default_factory=list)
    orgs: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)


class ExtractResult(BaseModel):
    """What the model must return. A bad domain fails here and is retried.

    call_structured treats a ValidationError as a schema miss and asks
    again. That is the retry for a domain outside config.DOMAINS, a
    red-flag label outside config.RED_FLAGS, or a list that does not
    have exactly one core claim.
    """

    checkable: bool
    not_checkable_reason: str = ""
    claim: str = ""
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    entities: Entities = Field(default_factory=Entities)
    domain: str = ""
    red_flags: list[str] = Field(default_factory=list)
    headline_body_mismatch: str = ""
    visual_description: str = ""
    location_clues: str = ""
    time_clues: str = ""

    @model_validator(mode="after")
    def _checkable_shape(self):
        # A greeting has no domain and no atomic claims. Requiring them
        # would burn a retry on a submission we are about to call Other.
        if not self.checkable:
            return self
        if self.domain not in config.DOMAINS:
            raise ValueError("domain is not one of the 11 labels")
        if len(self.claim.split()) > 30 or not self.claim.strip():
            raise ValueError("claim must be one sentence of at most 30 words")
        if not 1 <= len(self.atomic_claims) <= 3:
            raise ValueError("atomic_claims must have 1 to 3 items")
        cores = [item for item in self.atomic_claims if item.core]
        if len(cores) != 1:
            raise ValueError("atomic_claims must have exactly one core")
        ids = [item.id for item in self.atomic_claims]
        if len(ids) != len(set(ids)) or any(item not in {"A1", "A2", "A3"} for item in ids):
            raise ValueError("atomic ids must be A1, A2, A3 without repeats")
        unknown = [flag for flag in self.red_flags if flag not in config.RED_FLAGS]
        if unknown:
            raise ValueError("red_flags has a label outside the list")
        return self


def extract(state: dict) -> dict:
    """Boxes 3–4. One structured call, then the code guards.

    checkable_hint False means ingest already knows this is too short or
    not English. That is not a model question. LLMQuotaError is not
    caught: the caller stops the run and reports how far it got.
    """
    if state.get("checkable_hint") is False:
        reason = state.get("checkable_reason") or "not checkable"
        return _other(reason, state)
    prompt = _prompt(state)
    images = _load_image(state)
    parsed = call_structured(
        prompt,
        ExtractResult,
        purpose="extract",
        prompt_version="extract_v1",
        images=images,
    )
    return _guard(state, parsed)


def _other(reason: str, state: dict) -> dict:
    """Nothing to look up. Later boxes see verdict Other and stop."""
    flags, style_flags = _merge_flags([], state.get("style_signals") or {})
    return {
        "checkable": False,
        "not_checkable_reason": reason,
        "verdict": "Other",
        "claim": "",
        "atomic_claims": [],
        "entities": _empty_entities(),
        "domain": "",
        "red_flags": flags,
        "style_flags": style_flags,
        "extract_notes": [],
        "headline_body_mismatch": "",
        "visual_description": "",
        "location_clues": "",
        "time_clues": "",
    }


def _guard(state: dict, parsed: ExtractResult) -> dict:
    flags, style_flags = _merge_flags(parsed.red_flags, state.get("style_signals") or {})
    if not parsed.checkable:
        result = _other(parsed.not_checkable_reason or "not checkable", state)
        # The model's own red flags stay. _other also folded the style
        # signals, so a shouted joke still shows its warning signs.
        result["red_flags"] = flags
        result["style_flags"] = style_flags
        return result
    entities, notes = _ground(parsed.entities, _sources(state))
    mismatch = ""
    visual = ""
    location = ""
    when = ""
    # These three fields are meaningful for one input type. A value on
    # the wrong type is the model filling a box it was told to leave
    # empty, and it is dropped rather than shown on the card.
    if state.get("input_type") == "url":
        mismatch = _one_line(parsed.headline_body_mismatch)
    elif parsed.headline_body_mismatch.strip():
        notes.append("headline_body_mismatch ignored: not a URL")
    if state.get("input_type") == "image":
        visual = parsed.visual_description.strip()
        location = parsed.location_clues.strip()
        when = parsed.time_clues.strip()
    elif any(
        part.strip()
        for part in (parsed.visual_description, parsed.location_clues, parsed.time_clues)
    ):
        notes.append("image clues ignored: not an image")
    return {
        "checkable": True,
        "not_checkable_reason": "",
        "claim": " ".join(parsed.claim.split()),
        "atomic_claims": _atomic_claims(parsed.atomic_claims, notes),
        "entities": entities,
        "domain": parsed.domain,
        "red_flags": flags,
        "style_flags": style_flags,
        "extract_notes": notes,
        "headline_body_mismatch": mismatch,
        "visual_description": visual,
        "location_clues": location,
        "time_clues": when,
    }


def _atomic_claims(items: list[AtomicClaim], notes: list[str]) -> list[dict]:
    """Write checkable the way box 7 reads it.

    The model is asked to mark a motive false. If it marks "the
    government is hiding it" true anyway, the code corrects that and
    records the correction. A real event is left as the model marked it.
    """
    written = []
    for item in items:
        checkable = bool(item.checkable)
        if _MOTIVE.search(item.text):
            if checkable:
                notes.append(
                    f"{item.id} marked not checkable: conspiracy or motive"
                )
            checkable = False
        payload = item.model_dump()
        payload["checkable"] = checkable
        payload["text"] = " ".join(item.text.split())
        written.append(payload)
    return written


def _sources(state: dict) -> str:
    """The text the entity is allowed to come from.

    text is the cleaned forward or the lede. ocr_text and headline are
    named because a URL's headline and a screenshot's words are easy
    for the model to cite. body and caption are included for the same
    reason: they were in the prompt, so an entity taken from them is
    grounded. Anything else was guessed.
    """
    parts = [
        state.get("text") or "",
        state.get("ocr_text") or "",
        state.get("headline") or "",
        state.get("body") or "",
        state.get("caption") or "",
    ]
    return "\n".join(part for part in parts if part)


def _ground(entities: Entities, sources: str) -> tuple[dict, list[str]]:
    haystack = sources.casefold()
    kept: dict[str, list[str]] = {}
    notes: list[str] = []
    for field in ("people", "orgs", "places", "dates", "amounts"):
        kept[field] = []
        for item in getattr(entities, field):
            token = str(item).strip()
            if not token:
                continue
            if token.casefold() in haystack:
                kept[field].append(token)
            else:
                notes.append(f"dropped {field} {token!r}: not in the text")
    return kept, notes


def _merge_flags(model_flags: list[str], signals: dict) -> tuple[list[str], list[str]]:
    """Union of the model's labels and the style signals that have a label.

    The order is config.RED_FLAGS, so the card does not reshuffle when
    the model lists them differently. A signal with no label (a
    conspiracy phrase, the word "confirmed") stays a style flag. Neither
    list is an input to the verdict.
    """
    found: list[str] = []
    style_flags: list[str] = []
    if signals.get("urgency_words"):
        found.append("Urgency")
    if signals.get("share_request"):
        found.append("Call to forward")
    if signals.get("forwarded_label"):
        found.append("Unverifiable source")
    if signals.get("authority_mentions"):
        found.append("Fake or vague authority")
    certainty = [str(word) for word in (signals.get("certainty_words") or [])]
    if any(word in _MIRACLE_WORDS for word in certainty):
        found.append("Miracle or absolute claim")
    for word in certainty:
        if word not in _MIRACLE_WORDS:
            style_flags.append(f"certainty: {word}")
    for phrase in signals.get("conspiracy_phrases") or []:
        style_flags.append(f"conspiracy: {phrase}")
    caps = float(signals.get("caps_ratio") or 0)
    exclaim = int(signals.get("exclaim_count") or 0)
    emoji = int(signals.get("emoji_count") or 0)
    if caps >= _CAPS_SENSATIONAL or exclaim >= _EXCLAIM_SENSATIONAL or emoji >= _EMOJI_SENSATIONAL:
        found.append("Sensational style")
    elif emoji:
        style_flags.append(f"emoji_count: {emoji}")
    chosen = []
    pool = set(model_flags) | set(found)
    for label in config.RED_FLAGS:
        if label in pool:
            chosen.append(label)
    return chosen, style_flags


def _one_line(text: str) -> str:
    line = " ".join(str(text).split())
    return line


def _empty_entities() -> dict:
    return {"people": [], "orgs": [], "places": [], "dates": [], "amounts": []}


def _prompt(state: dict) -> str:
    template = (config.PROMPTS_DIR / "extract_v1.txt").read_text(encoding="utf-8")
    return (
        template.replace("__DOMAINS__", " | ".join(config.DOMAINS))
        .replace("__RED_FLAGS__", " | ".join(config.RED_FLAGS))
        .replace("__INPUT_TYPE__", str(state.get("input_type") or "text"))
        .replace("__TEXT__", str(state.get("text") or ""))
        .replace("__HEADLINE__", str(state.get("headline") or ""))
        .replace("__BODY__", str(state.get("body") or ""))
        .replace("__OCR__", str(state.get("ocr_text") or ""))
        .replace("__CAPTION__", str(state.get("caption") or ""))
    )


def _load_image(state: dict) -> list[bytes] | None:
    """The JPEG ingest saved. Not the original upload, which may hold GPS.

    Only an image input attaches bytes. A missing file is an error
    before the call: the model would otherwise describe a picture it
    was not shown.
    """
    if state.get("input_type") != "image":
        return None
    path = str(state.get("image_path") or "").strip()
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"image_path does not exist: {path}")
    return [file.read_bytes()]
