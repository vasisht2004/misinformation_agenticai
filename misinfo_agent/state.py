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
    # The run holds this in memory. The submissions database does not.
    raw_input: str
    # Box 1–2. Same characters as raw_input. ingest() returns this name.
    # Kept beside raw_input so older readers and the ingest box agree.
    # Still memory only: the database stores a hash, never this string,
    # a person's name, or a phone number.
    raw_text: str
    # Box 1–2. "text", "url" or "image". Later boxes branch on this.
    input_type: str
    # Box 1–2. Cleaned text (boilerplate and OCR noise removed).
    text: str
    # Box 1–2. text, lowercased, with punctuation turned into spaces.
    # The duplicate store hashes this string, and box 5 embeds it. A
    # "forward to all groups" trailer is already gone, so it cannot
    # change the hash or dilute the cosine. Case and commas are gone
    # so two copies of one forward land on one row.
    match_text: str
    # Box 1–2. A URL's headline, or the og:title of a social post.
    # Empty for a plain text forward and for an image.
    headline: str
    # Box 1–2. The article body, or the og:description of a social post.
    # text is the headline plus about the first 1200 characters of this.
    # The rest stays here so a reader can see what was clipped.
    body: str
    # Box 1–2. Publication date the page stated, YYYY-MM-DD, or "".
    # claim_date stays the day it was submitted. The eval overwrites
    # claim_date; this one is only what the page said.
    article_date: str
    # Box 1–2. Where it arrived from: a URL, or a channel name such as "whatsapp".
    source: str
    # Box 1–2. Legacy image payload. Ingest no longer puts pixels here.
    # The checkpointer would store them on every run. See image_path.
    image: dict
    # Box 1–2. Where the downscaled, EXIF-free JPEG was written.
    # extract loads the bytes from this path when the input is an image.
    image_path: str
    # Box 1–2. sha256 of that JPEG. Also the file name under UPLOADS_DIR.
    image_sha256: str
    # Box 1–2. imagehash.phash hex of the upright picture. Hamming
    # distance 6 or less is the same image in the submissions database.
    image_phash: str
    # Box 1–2. Words Tesseract scored at 60 or above, after the grayscale
    # and the upscale. Empty when the input is not an image.
    ocr_text: str
    # Box 1–2. Mean of those kept confidences. None when OCR did not run.
    ocr_mean_conf: float
    # Box 1–2. The caption typed with an image. Empty for text and URLs.
    caption: str
    # Box 1–2. Detected language. English is in scope; Hinglish is flagged, not dropped.
    # ingest() writes the same code into language. This key stays so a
    # reader written against the original contract still finds it.
    lang: str
    # Box 1–2. langdetect's code for the cleaned text. "en" for English
    # and for Hinglish (see hinglish). Any other code is out of scope,
    # unless the Hinglish word list already matched: that list wins,
    # because the detector often calls Hinglish something else.
    language: str
    # Box 1–2. True when the letters are Latin and a common Hindi word
    # is present. The run continues. Hindi in Devanagari is not this flag.
    # Checked before langdetect's label is trusted.
    hinglish: bool
    # Box 1–2. Ingest flags (hinglish, low_ocr, call_to_forward, ...).
    flags: list
    # Box 1–2. Warning signs counted on the raw forward, before cleaning
    # removes them. Box 11 (priority) and the result card read this dict.
    # The verdict does not. A forward can shout and still be true, and a
    # quiet one can still be false. Shape: urgency_words, share_request,
    # forwarded_label, caps_ratio, exclaim_count, emoji_count,
    # authority_mentions, conspiracy_phrases, certainty_words.
    style_signals: dict
    # Box 1–2. Quality check outcome. A failed check stops the run before any LLM call.
    quality: dict
    # Box 1–2. Whether the text is worth a model call. False for a
    # greeting and for a language this project does not check. The reason
    # string is empty when the hint is true.
    checkable_hint: bool
    # Box 1–2. "too short to check", "out of scope (not English)", or "".
    checkable_reason: str
    # Box 1–2. Duplicate-forward check (MinHash / exact hash), separate from box 5 memory.
    dedupe: dict
    # Box 1–2. sha256 of match_text. When MinHash joins this forward to an
    # older one, this is the older row's hash, so both copies share an id.
    # It is not a hash of the raw message.
    submission_hash: str
    # Box 1–2. Times this forward has been stored, including this one.
    # The first copy is 1. A repeat is 2 or more.
    submission_count: int

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
    # Ingest's style signals are merged in here when they match a label.
    # Signals with no label stay in style_flags and do not change the verdict.
    red_flags: list
    # Box 3–4. Style signals that did not match a red_flags label
    # (a conspiracy phrase, a "confirmed" with no miracle wording).
    style_flags: list
    # Box 3–4. False for an opinion, a joke, a prediction, or a greeting.
    # The verdict is then Other and later boxes do not search.
    checkable: bool
    # Box 3–4. Why checkable is false. Empty when the claim can be checked.
    not_checkable_reason: str
    # Box 3–4. What the code changed after the model answered: an entity
    # that was not in the text, a mismatch cleared because this was not a URL.
    extract_notes: list
    # Box 3–4. One line, URLs only, when the headline says something the
    # body does not. Empty when they agree or the input is not a URL.
    headline_body_mismatch: str
    # Box 3–4. Images only. What is visibly in the picture. The model is
    # told not to guess a place or a date that is not shown.
    visual_description: str
    location_clues: str
    time_clues: str
    # Box 1–2, or the eval. YYYY-MM-DD the claim was circulating. Ingest
    # sets it from the message; the eval sets it to the fact-check article
    # date. Box 7 compares a page's date with this so a later write-up
    # cannot count as something the system could have read at the time.
    claim_date: str

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
    # Box 7. Windows picked from the fetched pages, before stance. Stance
    # is what writes evidence. These rows are the input to that step.
    passages: list
    # Box 7. Counts for this round: searches, the stance call, how the
    # passages split, independent groups on each side, circulation
    # sightings, seconds, and a note when a cap stopped the round short.
    investigate_stats: dict
    # Box 9. How many investigate rounds have already finished. The gate
    # and box 12 stop at config.MAX_SEARCH_ROUNDS. A search_more return
    # is the only write that advances this.
    rounds: int
    # Box 9. "sufficient", "insufficient", "conflicting", "search_more",
    # "error", or "skipped". error is a stance failure after the search
    # budget, never a lack of sources. skipped means box 6 reused an
    # exact fresh prior, or the submission is not checkable, so this
    # table was not judged. This is the gate_decision.
    gate: str
    # Box 9. Plain sentences for the card: one line per atomic claim,
    # then why the overall gate is what it is. A shout is not in here.
    gate_reasons: list
    # Box 9. Atomic-claim id -> status. not_checkable, conflicting,
    # refuted, supported, or insufficient. The overall gate follows
    # the core claim only.
    atomic_status: dict

    # Box 8+10. False / Misleading / True / Unverifiable / Other.
    # Computed from evidence by rule, not taken from the model's self-report.
    verdict: str
    # Box 8+10. Confidence from evidence features, not the LLM's own score.
    confidence: float
    # Box 8+10. Plain-language explanation, true version and the shareable
    # correction. Box 14–15 displays this; it does not rewrite it.
    explanation: str

    # Box 11. "High", "Medium" or "Low". This is the priority_level.
    # The score that produced it is priority_score. Style signals can
    # move this. They cannot move the verdict.
    priority: str
    # Box 11. Weighted harm, novelty, reach and urgency, in [0, 1].
    priority_score: float
    # Box 11. One plain sentence per component, in that order.
    priority_reasons: list
    # Box 12. "alert", "human", "search_more" or "store".
    # human is the pause. Box 13 records what the person did.
    # An alert still pauses there while ALERT_NEEDS_APPROVAL is on;
    # the decision stays "alert" so the card can say what was asked.
    decision: str
    # Box 12. Why `decision` was chosen. An alert says box 13 must
    # approve it before anything is sent.
    decision_reasons: list
    # Box 12. The confidence bar, a / (a + bH). Higher harm, lower bar.
    p_star: float
    # Box 13. Reviewer action once a paused run resumes. Empty until then.
    human: dict

    # Box 16. One audit record: models, prompt version, call counts, latency.
    audit: dict
