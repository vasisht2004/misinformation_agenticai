"""ClaimState: the one dict every box reads and returns a partial update to.

Why a TypedDict with total=False: LangGraph merges the dict a node returns
into the running state. Early boxes have not filled later keys, so missing
keys are the normal case, not an error. Changing this file is a team contract
change (see CONTRIBUTING.md and docs/ARCHITECTURE.md §4).

evidence uses operator.add so a second investigate round appends rows. A plain
list would be replaced, and the gate would judge only the latest search.
"""

import operator
from typing import Annotated, TypedDict


class ClaimState(TypedDict, total=False):
    """Keys in pipeline order. Each comment names the box that writes the key."""

    # Box 1–2. Untouched user submission, before cleaning or OCR.
    raw_input: str
    # Box 1–2. "text", "url" or "image". Later boxes branch on this.
    input_type: str
    # Box 1–2. Cleaned text (boilerplate and OCR noise removed).
    text: str
    # Box 1–2. The string box 5 embeds. Claim-shaped, not the whole forward,
    # so a "forward to all groups" trailer does not dilute the cosine.
    match_text: str
    # Box 1–2. Where it arrived from: a URL, or a channel name such as "whatsapp".
    source: str
    # Box 1–2. Image path, perceptual hash and OCR payload. Absent for pure text.
    image: dict
    # Box 1–2. Detected language. English is in scope; Hinglish is flagged, not dropped.
    lang: str
    # Box 1–2. Ingest flags (hinglish, low_ocr, call_to_forward, ...).
    flags: list
    # Box 1–2. Quality check outcome. A failed check stops the run before any LLM call.
    quality: dict
    # Box 1–2. Duplicate-forward check (MinHash / exact hash), separate from box 5 memory.
    dedupe: dict

    # Box 3–4. One checkable claim, written in the same style as primary_claim
    # (prompt_v4a) so it is comparable with the memory.
    claim: str
    # Box 3–4. Atomic claims. One core claim is the common case; a forward
    # that bundles two rumours becomes two, and box 8 judges each.
    atomic_claims: list
    # Box 3–4. Numbers, years, rupee amounts, names, places. Box 6 prefers
    # these over re-parsing the string when they are present.
    entities: dict
    # Box 3–4. One of the 11 domains. Also a box 5 search filter.
    domain: str
    # Box 3–4. Warning signs from the claim as circulated (prompt_v4a list).
    red_flags: list

    # Box 5. Hybrid-search hits. Empty means the memory was searched and nothing
    # survived the filters, which is different from the key being absent.
    matches: list

    # Box 6. "exact", "similar" or "none". Thresholds alone do not decide this.
    # A disagreeing date, amount, year, number, name or subject downgrades
    # exact to similar, and a picture claim is capped at similar because
    # there is no image hash yet.
    match_type: str
    # Box 6. "reuse_prior" only when the match is exact and fresh. Otherwise
    # "investigate", and the hits are leads rather than a verdict.
    route: str
    # Box 6. Top matches handed to box 7 when the prior is not reused as-is.
    leads: list
    # Box 6. True when a status claim's best match is older than STALE_DAYS,
    # except an out-of-context or edited image/video whose origin cannot expire.
    needs_recheck: bool
    # Box 6. Plain-English sentences, in card order: how close the claim is,
    # then the mismatch, the picture cap, expiry, and the route. Shown in
    # the demo, so they are part of the product, not debug logs.
    match_reasons: list
    # Box 6. The first one or two of match_reasons. The result card shows
    # this; the full list stays in match_reasons.
    match_summary: list

    # Box 7. Evidence rows E1…En. The reducer appends across search rounds.
    evidence: Annotated[list, operator.add]
    # Box 9. How many investigate rounds have run. The gate stops at 2.
    rounds: int
    # Box 9. "sufficient", "insufficient", "conflicting" or "search_more".
    gate: str

    # Box 8+10. False / Misleading / True / Unverifiable / Other.
    # Computed from evidence by rule, not taken from the model's self-report.
    verdict: str
    # Box 8+10. Confidence from evidence features, not the LLM's own score.
    confidence: float
    # Box 8+10. Plain-language explanation, true version and the shareable
    # correction. Box 14–15 displays this; it does not rewrite it.
    explanation: str

    # Box 11. "High", "Medium" or "Low".
    priority: str
    # Box 12. "alert", "unsure" (human), "search_more" or "store".
    decision: str
    # Box 13. Reviewer action once a paused run resumes. Empty until then.
    human: dict

    # Box 16. One audit record: models, prompt version, call counts, latency.
    audit: dict
