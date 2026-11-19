"""Box 7: find live pages for a claim the memory could not settle.

Owner: Vash. The plan is fixed, the pages are fetched, and the passages
closest to the claim are kept. One stance call then reads those passages.
Tiers, independence and freshness are rules. The model does not decide
the verdict; it only says what each passage does.

The plan is fixed on purpose. A model that writes its own queries can
search for the verdict it already expects. These five lines always ask
for support, for a refutation, for the office that would have issued the
order, and, when box 6 found a near miss, for the older wording. They
are built from the core atomic claim. Each further atomic claim is a
different event, so it gets one neutral line of its own; a single query
for "A and then B" returns pages about A and misses B. One top-up is
allowed after that, and only when fewer than two independent tier 1–2
groups support or refute the claim. The gate's second call skips queries
this box already ran.

The encoder is BGE-M3, the same model as the memory. It embeds text. It
is not a language-model call, and it is loaded once per process.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from pydantic import BaseModel, Field, field_validator

import misinfo_agent.config as config
from misinfo_agent.fetch import Page, date_missing, fetch_pages, remember_publication
from misinfo_agent.llm import LLMError, LLMQuotaError, call_structured, get_usage
from misinfo_agent.memory import get_encoder
from misinfo_agent.search import (
    Hit,
    SearchError,
    SearchQuotaError,
    factcheck_blocked,
    get_search_usage,
    registrable_domain,
    reset_claim_searches,
    search,
    site_host,
)

# A checkable claim is one sentence. Twenty words keeps a long one from
# becoming a query that only matches the forward itself. Most claims are
# shorter than this and pass through whole.
_NEUTRAL_WORDS = 20

# primary_claim is already a short sentence. The cap is only a guard
# against a lead that is the whole article.
_LEAD_WORDS = 30

# The entities already carry the names and the numbers. Eight more words
# is the subject, not a second copy of the neutral sentence.
_SUBJECT_WORDS = 8

# Five lines when the claim is one event. A later kind has to take a
# slot, not add a search. A checkable second or third atomic claim may
# add one neutral line. The plan then stops at six, so two of the eight
# searches allowed for the claim are left for the top-up.
_MAX_QUERIES = 5
_PLAN_CAP = 6

# Five snippets times five queries is 25, which is also the candidate cap.
# Asking for more would fetch pages the cap throws away.
_RESULTS_PER_QUERY = 5
_MAX_HITS = 25
# Each line of the plan keeps its best three before the overall cap.
# Five lines times three is 15, so the reserve fits inside 25 and the
# rest of the slots fill by score. Without this, one strong query fills
# the cap and the official page never gets fetched.
_PER_KIND = 3

# A ministry search box matches the nouns, not the sentence. Two to four
# of them: fewer is too vague when the claim has the words, more turns
# back into the sentence.
_CORE_MAX = 4

_WINDOW_SENTENCES = 3
_WINDOW_OVERLAP = 1
_MIN_WINDOW_WORDS = 20
_PER_PAGE = 2
# Two windows from one site is enough. A third is the same newsroom
# saying the same thing, and it would crowd out a second source.
_PER_DOMAIN = 2
_MAX_PASSAGES = 40
_ENCODE_BATCH = 32

# The words a debunking headline actually uses. The capital OR is the
# disjunction: a page that calls the claim any one of these is the page
# the refute line is looking for. Tavily may treat OR as a plain word;
# the four words are still the right ones to send.
_REFUTE_SUFFIX = "fake OR false OR hoax OR misleading"

# Function words. Negation is not in here: "not" and "never" change the
# claim, and dropping them would search for the opposite.
# Words that describe the post, not the event. "Video shows flood in
# Chennai" is a search for a video. The office that would confirm the
# flood files it under Chennai and flood.
_MEDIA = frozenset(
    {
        "video", "photo", "image", "clip", "viral", "shows", "claims",
        "says", "news", "true", "false",
    }
)

_STOP = frozenset(
    {
        "a", "an", "the", "of", "to", "and", "or", "is", "are", "was", "were",
        "be", "been", "being", "that", "this", "with", "for", "in", "on", "by",
        "as", "at", "it", "its", "their", "they", "them", "all", "into", "over",
        "under", "about", "between", "during", "without", "within", "than",
        "then", "also", "just", "can", "may", "will", "has", "have", "had",
        "from", "onto", "upon", "per", "via",
    }
)
_EDGE_PUNCT = ".,;:!?\"'()[]{}“”‘’"

# Click ids and campaign tags. They do not change which page it is, and
# leaving them on splits one source into two hits. Only these: a real
# query parameter such as an article id must survive.
_TRACKING = frozenset({"fbclid", "gclid"})

# Names and places first, then the figures. A stable order so the same
# claim always produces the same verify query.
_ENTITY_KEYS = ("names", "places", "amounts", "dates", "years")


class Query(BaseModel):
    """One line of the fixed plan.

    kind is why we search, not a rank. include_domains is set only for the
    official line. None means the open web; the fact-check block, when it
    is on, is applied inside search() and is not repeated here.
    """

    text: str
    kind: Literal["neutral", "verify", "refute", "official", "lead", "topup"]
    include_domains: list[str] | None = None


# The stance call reads this many passages. More than that is a second
# claim inside one prompt. A thin result may add one more stance call
# for the top-up pages, and that is the last model call this box makes.
_STANCE_LIMIT = 20
_TOPUP_QUERIES = 2
# A page a year older than the claim cannot show that a "now" claim is
# happening now. It can still refute the claim, which is why only a
# support is cleared. A refute is never rewritten.
_OLD_EVENT_DAYS = 365
# v8 is the stance prompt. Older versions stay on disk so a cached check
# can still be read. v8 is v7 plus one rule: a passage that quotes the
# claim as a myth, rumour, hoax, or something people believe is a refute
# (or neutral if it gives no verdict), never a support. v7 keeps every v6
# rule and adds the share date, cure/effect refutes, and the
# campus-versus-station example.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "stance_v8.txt"
_TOPUP_PATH = Path(__file__).resolve().parent / "prompts" / "topup_v1.txt"
_CURRENT_CLAIM = re.compile(
    r"\b(?:now|new|permanently)\b|\bfrom\s+(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)
_SHINGLE = 5
_MINHASH_PERM = 128
# Curly quotes and the guillemet. The model copies one, the page has the
# other, and the quote check would otherwise throw the row away.
_QUOTE_MARKS = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u00ab": '"', "\u00bb": '"',
})


class StanceJudgement(BaseModel):
    """One passage, as the model labelled it. The code still checks the quote."""

    passage_id: str
    stance: Literal["supports", "refutes", "neutral"]
    quote: str = ""
    reason: str = ""

    @field_validator("quote", "reason", "passage_id", mode="before")
    @classmethod
    def _text(cls, value):
        if value is None:
            return ""
        return str(value)

    @field_validator("reason")
    @classmethod
    def _short_reason(cls, value):
        # A long reason must not fail the call. Twenty words is the card,
        # and the rest is the model talking past it.
        return " ".join(value.split()[:20])


class StanceBatch(BaseModel):
    """The stance call returns an object, not a bare list.

    Gemini's structured output is an object schema. The list of labels
    is the judgements field. The prompt names that field.
    """

    judgements: list[StanceJudgement] = Field(default_factory=list)


class AtomicStanceJudgement(BaseModel):
    """One passage judged against one atomic claim.

    A separate model, not a field on StanceJudgement. One atomic claim
    (or none) must keep today's schema, so a cached single-claim response
    still validates and the model is not asked for an id it did not have.
    """

    passage_id: str
    atomic_id: str
    stance: Literal["supports", "refutes", "neutral"]
    quote: str = ""
    reason: str = ""

    @field_validator("quote", "reason", "passage_id", "atomic_id", mode="before")
    @classmethod
    def _text(cls, value):
        if value is None:
            return ""
        return str(value)

    @field_validator("reason")
    @classmethod
    def _short_reason(cls, value):
        return " ".join(value.split()[:20])

    @field_validator("atomic_id")
    @classmethod
    def _id(cls, value):
        text = " ".join(str(value).split())
        if not text:
            raise ValueError("atomic_id is required when there are several claims")
        return text


class AtomicStanceBatch(BaseModel):
    """Same call as StanceBatch, with one judgement per passage per claim."""

    judgements: list[AtomicStanceJudgement] = Field(default_factory=list)


class TopupBatch(BaseModel):
    """At most two new queries. The code drops anything past the second."""

    queries: list[str] = Field(default_factory=list)

    @field_validator("queries", mode="before")
    @classmethod
    def _lines(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        lines = []
        for item in value:
            text = " ".join(str(item).split())
            if text:
                # A paragraph is not a query. Twenty words is one search.
                lines.append(" ".join(text.split()[:20]))
        return lines[:_TOPUP_QUERIES]


def investigate(state: dict) -> dict:
    """Search, read, and label. Returns passages, evidence, and the counts.

    A new claim resets the per-claim search budget. A second call from the
    gate does not: that budget is for the whole claim, and the queries
    already recorded in investigate_stats are skipped. One stance call
    reads the best passages. If fewer than two independent tier 1–2 groups
    support or refute, one top-up proposes at most two new queries and one
    more stance call reads only the new passages. Tiers and groups are
    rules. A search quota or a model error returns whatever is already in
    hand and a note. It does not raise: the gate is the box that says the
    check was not enough.
    """
    started = time.perf_counter()
    search_mark = len(get_search_usage())
    llm_mark = len(get_usage())
    prior_queries = _prior_queries(state)
    prior_evidence = state.get("evidence") if isinstance(state.get("evidence"), list) else []
    # The in-memory cap survives for a second call in this process. A new
    # claim has no recorded queries, so it starts the budget over. Resetting
    # a gate round would double the eight searches the cap allows.
    if not prior_queries:
        reset_claim_searches()
    notes: list[str] = []
    passages: list[Passage] = []
    pages: list[Page] = []
    labelled: list[dict] = []
    unverified = 0
    ran: list[str] = []
    stance_failed = False
    search_failed = False
    try:
        bundle = gather_passages(state, skip_queries=prior_queries)
        pages = bundle["pages"]
        passages = list(bundle["passages"])
        ran.extend(bundle["queries"])
    except SearchQuotaError as exc:
        notes.append(str(exc))
        search_failed = True
    else:
        chosen = passages[:_STANCE_LIMIT]
        if chosen:
            try:
                batch = judge_stance(state, chosen)
            except LLMError as exc:
                # The day's quota is spent. Another search or a top-up would
                # only fail the same way, so the caller stops the run here.
                if isinstance(exc, LLMQuotaError):
                    raise
                notes.append(str(exc))
                # Box 9 must treat this as "system error -> retry / human",
                # never as "insufficient evidence". The passages stay so a
                # person can see what was fetched. An empty table here means
                # the stance call failed, not that no source was found.
                stance_failed = True
            else:
                labelled, unverified = apply_stance(
                    _claim_text(state),
                    chosen,
                    batch.judgements,
                    state.get("claim_date"),
                    _atomic_items(state),
                )
                # The rows the model did not see are not part of this round.
                # Keeping them would make the counts disagree with the call.
                passages = chosen
    evidence = build_evidence(
        state, labelled, id_start=_next_evidence_id(prior_evidence), prior=prior_evidence
    )
    topup_used = False
    topup_queries: list[str] = []
    # A failed stance, or a search cap already hit, is not "thin evidence".
    # Spending another call would only repeat the failure.
    if not stance_failed and not search_failed and _strong_groups(evidence) < 2:
        topup_used, extra_labelled, extra_passages, extra_pages, extra_queries, extra_unverified, topup_note = (
            _topup(state, labelled, passages, prior_queries + ran)
        )
        topup_queries = extra_queries
        ran.extend(extra_queries)
        if topup_note:
            notes.append(topup_note)
        if extra_passages:
            labelled = labelled + extra_labelled
            passages = passages + extra_passages
            pages = pages + extra_pages
            unverified += extra_unverified
            evidence = build_evidence(
                state, labelled, id_start=_next_evidence_id(prior_evidence), prior=prior_evidence
            )
    stats = _stats(
        passages=passages,
        labelled=labelled,
        evidence=evidence,
        pages=pages,
        searches=len(get_search_usage()) - search_mark,
        llm_calls=len(get_usage()) - llm_mark,
        usage=get_usage()[llm_mark:],
        unverified=unverified,
        notes=notes,
        started=started,
        queries=_merge_queries(prior_queries, ran),
        topup_used=topup_used,
        topup_queries=topup_queries,
        stance_failed=stance_failed,
        prior_evidence=prior_evidence,
    )
    return {
        "passages": [passage.model_dump(mode="json") for passage in passages],
        "evidence": evidence,
        "investigate_stats": stats,
    }


def _atomic_items(state: dict) -> list[dict]:
    """Extract's atomic claims as {id, text, core}, or an empty list.

    A string is the older shape (the whole claim repeated for the stance
    prompt). A dict is what extract writes. Empty text is skipped. When
    nothing is marked core, the first item is the core: a single claim
    has no other event to search.
    """
    atomic = state.get("atomic_claims")
    if not isinstance(atomic, list):
        return []
    items = []
    for item in atomic:
        if isinstance(item, dict):
            text = " ".join(str(item.get("text") or "").split())
            if not text:
                continue
            ident = " ".join(str(item.get("id") or "").split()) or f"A{len(items) + 1}"
            # Missing means checkable. Older callers and a one-line string
            # did not have the flag, and they were searched.
            items.append({
                "id": ident,
                "text": text,
                "core": bool(item.get("core")),
                "checkable": item.get("checkable") is not False,
            })
        else:
            text = " ".join(str(item).split())
            if not text:
                continue
            items.append({
                "id": f"A{len(items) + 1}",
                "text": text,
                "core": not items,
                "checkable": True,
            })
    if items and not any(item["core"] for item in items):
        items[0] = {**items[0], "core": True}
    return items


def _atomic_text(state: dict) -> str:
    """The atomic claims as lines the stance prompt can read.

    Stringifying a dict would put the braces into the prompt. The core
    mark tells the model which sentence the fixed plan was built from.
    """
    items = _atomic_items(state)
    if not items:
        return "(none)"
    lines = []
    for item in items:
        mark = "core" if item["core"] else "also"
        lines.append(f"{item['id']} ({mark}) {item['text']}")
    return "\n".join(lines)


def judge_stance(state: dict, passages: list[Passage]) -> StanceBatch | AtomicStanceBatch:
    """One call for the whole claim. The prompt file holds the rules.

    Two or three atomic claims are judged in that same call, one label
    per passage per claim. One claim, or none, keeps the original schema
    so the JSON shape does not change.
    """
    items = _atomic_items(state)
    multi = len(items) >= 2
    blocks = []
    for passage in passages:
        when = passage.published_date or "unknown"
        blocks.append(f"{passage.passage_id}\ndate: {when}\n{passage.text}")
    shared = state.get("claim_date")
    if not isinstance(shared, str) or not shared.strip():
        shared = "an unknown date"
    prompt = (
        _PROMPT_PATH.read_text(encoding="utf-8")
        .replace("__CLAIM_DATE__", shared.strip())
        .replace("__CLAIM__", _claim_text(state) or "(none)")
        .replace("__ATOMIC__", _atomic_text(state) or "(none)")
        .replace("__PASSAGES__", "\n\n".join(blocks))
    )
    schema = StanceBatch
    if multi:
        # The file already holds every v6 rule. This paragraph only tells
        # the model to apply that same text once per atomic claim. A
        # one-claim call does not get it, so its schema stays the old one.
        ids = ", ".join(item["id"] for item in items)
        prompt += (
            "\n\nThere is more than one atomic claim. Return one judgement "
            "for every passage and every atomic claim "
            f"({ids}), and set atomic_id to that claim's id. Apply every "
            "rule above to that atomic claim alone, including the share "
            "date, the cure rule, and the named-place rule. A passage may "
            "support one and be neutral about another. Do not require it "
            "to confirm the other atomic claims.\n"
        )
        schema = AtomicStanceBatch
    return call_structured(
        prompt,
        schema,
        purpose="stance",
        prompt_version="stance_v8",
    )


def apply_stance(
    claim: str,
    passages: list[Passage],
    judgements: list,
    claim_date: str | None = None,
    atomic_items: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Keep a label only when its quote is really in the passage.

    A quote that is not in the text, once whitespace and quote marks are
    flattened, did not come from the page. The label becomes neutral and
    the row is flagged. A passage that only repeats the claim is neutral
    even if the model calls it support: the rumour is not evidence for itself.

    One atomic claim, or none, is one label per passage and the row's
    atomic_id is A1, which is the shape this box had before a claim could
    split. Two or three claims are judged separately, and the v6 rules
    (the quote, a passage that only repeats the claim, an old page) use
    that claim's own sentence.
    """
    items = list(atomic_items or [])
    labelled: list[dict] = []
    unverified = 0
    if len(items) < 2:
        by_id = {item.passage_id: item for item in judgements}
        for passage in passages:
            row, flag = _judge_passage(claim, passage, by_id.get(passage.passage_id), claim_date)
            row["atomic_id"] = "A1"
            unverified += int(flag)
            labelled.append(row)
        return labelled, unverified
    by_key = {}
    for item in judgements:
        aid = " ".join(str(getattr(item, "atomic_id", "") or "").split())
        by_key[(item.passage_id, aid)] = item
    for passage in passages:
        for atomic in items:
            row, flag = _judge_passage(
                atomic["text"],
                passage,
                by_key.get((passage.passage_id, atomic["id"])),
                claim_date,
            )
            row["atomic_id"] = atomic["id"]
            unverified += int(flag)
            labelled.append(row)
    return labelled, unverified


def _judge_passage(claim: str, passage: Passage, judgement, claim_date) -> tuple[dict, bool]:
    """One passage against one sentence. The caller sets atomic_id."""
    stance = judgement.stance if judgement is not None else "neutral"
    quote = judgement.quote if judgement is not None else ""
    reason = judgement.reason if judgement is not None else "No judgement was returned."
    flag = False
    if quote and not _quote_in_passage(quote, passage.text):
        stance = "neutral"
        flag = True
    elif stance in {"supports", "refutes"} and not _quote_in_passage(quote, passage.text):
        # An empty quote is not a quote. Support without words is not evidence.
        stance = "neutral"
        flag = True
    if _only_repeats(claim, passage.text):
        stance = "neutral"
        reason = reason or "The passage only repeats the claim."
    # Only a support is cleared. An old page that names a different
    # date is a refutation of "from 1 December", and this line must
    # not turn that refutes into neutral.
    if stance == "supports" and _old_event(claim, claim_date, passage):
        stance = "neutral"
        reason = "old event"
    return (
        {
            "passage": passage,
            "stance": stance,
            "quote": quote,
            "reason": reason,
            "quote_unverified": flag,
        },
        flag,
    )


def _quote_in_passage(quote: str, passage: str) -> bool:
    """True when the quote's words appear in the passage.

    Straight, curly, single, and double quotes are the same mark. The
    model copies the words and swaps the marks. That is still the
    sentence on the page, so the row is not thrown away.
    """
    needle = _loose_quotes(_flat_quotes(quote))
    if not needle:
        return False
    return needle in _loose_quotes(_flat_quotes(passage))


def _loose_quotes(text: str) -> str:
    return " ".join(text.replace('"', " ").replace("'", " ").split())


def _flat_quotes(text: str) -> str:
    return " ".join((text or "").translate(_QUOTE_MARKS).split())


def _only_repeats(claim: str, text: str) -> bool:
    """True when the passage adds no content word the claim did not already use."""
    claim_words = set(_content_words(claim))
    words = _content_words(text)
    if not words or not claim_words:
        return False
    return all(word in claim_words for word in words)


def _content_words(text: str) -> list[str]:
    words = []
    for raw in _flat_quotes(text).casefold().split():
        word = raw.strip(_EDGE_PUNCT)
        # Media words stay. "false" and "video" are not the claim, so a
        # passage that adds them is not only a copy of the claim.
        if word and word not in _STOP:
            words.append(word)
    return words


def credibility_tier(url: str) -> int:
    """1 official, 2 newsroom or journal, 3 other organisation, 4 the rest.

    The tier is the source. A snippet from a ministry is still tier 1;
    snippet_only on the row says we did not open the page.
    """
    domain = _source_domain(url)
    if _is_tier1(domain):
        return 1
    # A blog is not promoted by also matching a news suffix we do not have.
    if _listed(domain, config.TIER4_HOSTS):
        return 4
    if _is_tier2(domain):
        return 2
    if _is_tier3(domain):
        return 3
    return 4


def _source_domain(url: str) -> str:
    """The registrable domain. www.bbc.co.uk and bbc.co.uk are one source."""
    host = site_host(url) or ""
    return registrable_domain(host) or host


def _is_tier1(domain: str) -> bool:
    if _suffix_matches(domain, config.TIER1_SUFFIXES):
        return True
    if _listed(domain, config.TIER1_HOSTS):
        return True
    offices = []
    for hosts in config.OFFICIAL_DOMAINS.values():
        offices.extend(hosts)
    return _listed(domain, offices)


def _is_tier2(domain: str) -> bool:
    if _listed(domain, config.TIER2_HOSTS):
        return True
    return _suffix_matches(domain, config.TIER2_SUFFIXES)


def _is_tier3(domain: str) -> bool:
    if _listed(domain, config.TIER3_HOSTS):
        return True
    # A hospital whose name is the host, not one of the four we listed.
    return "hospital" in domain


def _suffix_matches(domain: str, patterns) -> bool:
    """*.gov and *.gov.* on the registrable domain.

    A pattern that ends in a dot (".gov.") is an internal label: pib.gov.in
    contains ".gov.". A pattern that does not (".edu", ".gc.ca") is a
    suffix of the whole domain.
    """
    if not domain:
        return False
    padded = f".{domain}."
    for pattern in patterns:
        if pattern.startswith(".") and pattern.endswith(".") and pattern != ".":
            if pattern in padded:
                return True
            continue
        bare = pattern[1:] if pattern.startswith(".") else pattern
        if domain == bare or domain.endswith("." + bare):
            return True
    return False


def _listed(domain: str, names) -> bool:
    for name in names:
        if domain == name or domain.endswith("." + name):
            return True
    return False


def assign_groups(passages: list[Passage]) -> dict[str, str]:
    """One group per newsroom, and one group for a story copied across newsrooms.

    Two pages on bbc.com are the same outlet even when the words differ, so
    they start in one group. MinHash then joins a near-copy on another
    domain: five-word shingles, 128 permutations, and an LSH threshold of
    0.5. A syndicated article must not count as two independent sources.
    """
    parent = list(range(len(passages)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_domain: dict[str, int] = {}
    for index, passage in enumerate(passages):
        host = registrable_domain(site_host(passage.url) or passage.url)
        earlier = by_domain.get(host)
        if earlier is None:
            by_domain[host] = index
        else:
            union(earlier, index)
    hashes = [_minhash(passage.text) for passage in passages]
    present = [(index, digest) for index, digest in enumerate(hashes) if digest is not None]
    if present:
        from datasketch import MinHashLSH

        lsh = MinHashLSH(threshold=0.5, num_perm=_MINHASH_PERM)
        for index, digest in present:
            lsh.insert(str(index), digest)
        for index, digest in present:
            for other in lsh.query(digest):
                union(index, int(other))
    labels: dict[int, str] = {}
    groups: dict[str, str] = {}
    next_id = 1
    for index, passage in enumerate(passages):
        root = find(index)
        if root not in labels:
            labels[root] = f"G{next_id}"
            next_id += 1
        groups[passage.passage_id] = labels[root]
    return groups


def _minhash(text: str):
    words = _flat_quotes(text).split()
    if len(words) >= _SHINGLE:
        shingles = [" ".join(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)]
    elif words:
        shingles = [" ".join(words)]
    else:
        return None
    from datasketch import MinHash

    digest = MinHash(num_perm=_MINHASH_PERM, seed=1)
    for shingle in shingles:
        digest.update(shingle.encode("utf-8"))
    return digest


def build_evidence(
    state: dict, labelled: list[dict], *, id_start: int = 1, prior: list | None = None
) -> list[dict]:
    """Supports and refutes only, best tier first, then the higher score.

    Neutral passages stay in the counts. They are not rows: a row is
    something the gate can cite. Inside one group, a second row with the
    same quote is the same sentence and is dropped. A group that contains
    both a support and a refute is kept in full and marked group_conflict.
    Ids continue after the rows a previous round already returned, so a
    second call appends E5 rather than another E1.
    """
    if not labelled:
        return []
    # A top-up numbers its passages from P1 again. Grouping is keyed by
    # that id, so the second P1 would overwrite the first and one newsroom
    # would look like two groups. Renumber before grouping.
    labelled = [
        {**row, "passage": row["passage"].model_copy(update={"passage_id": f"P{index}"})}
        for index, row in enumerate(labelled, start=1)
    ]
    passages = [row["passage"] for row in labelled]
    groups = _groups_with_prior(passages, prior or [])
    claim_date = state.get("claim_date") if isinstance(state.get("claim_date"), str) else None
    rows = []
    for row in labelled:
        if row["stance"] not in {"supports", "refutes"}:
            continue
        passage = row["passage"]
        # The page the person submitted cannot be evidence for itself.
        # A search hit of that same URL is dropped earlier; this is the
        # backstop for a row that arrived another way.
        if _is_submitted_page(state, passage.url):
            continue
        rows.append(
            {
                "url": passage.url,
                "netloc": passage.netloc,
                "title": passage.title,
                "tier": credibility_tier(passage.url),
                "group_id": groups.get(passage.passage_id, ""),
                "stance": row["stance"],
                "quote": row["quote"],
                # Shown to a person. The quote field stays the model's words
                # so a later check can still find them in the passage.
                "display_quote": _display_quote(row["quote"], passage.text),
                "reason": row["reason"],
                "passage": passage.text,
                "score": passage.score,
                "cos": passage.cos,
                "published_date": passage.published_date,
                "date_source": passage.date_source,
                "age_days": _age_days(claim_date, passage.published_date),
                "after_claim": passage.after_claim,
                "fetched": passage.fetched,
                # The tier does not change when this is true. The gate can
                # still see that the row is a snippet.
                "snippet_only": not passage.fetched,
                # A1 when the claim was not split. Two events keep their
                # own id so a support of one is not a support of the other.
                "atomic_id": str(row.get("atomic_id") or "A1"),
            }
        )
    rows.sort(key=lambda item: (item["tier"], -item["score"], item["url"]))
    rows = _drop_repeated_quotes(rows, prior or [])
    # After the duplicate quote is gone. A repeated sentence is not a
    # second stance, and it must not invent a conflict by itself.
    _mark_group_conflicts(rows, prior or [])
    for offset, row in enumerate(rows):
        row["id"] = f"E{id_start + offset}"
    return rows


def _display_quote(quote: str, passage: str) -> str:
    """The quote, plus the sentence before it when the quote is a fragment.

    "This is a myth." is a real stance and a useless line on its own. The
    previous sentence says what the myth is. Eight words or more already
    carry that, so those quotes are shown as the model wrote them. This
    does not change the label or the prompt.
    """
    text = " ".join(str(quote or "").split())
    if len(text.split()) >= 8:
        return text
    needle = _flat_quotes(text)
    if not needle:
        return text
    sentences = _sentences(passage or "")
    for index, sentence in enumerate(sentences):
        if needle not in _flat_quotes(sentence):
            continue
        if index == 0:
            return text
        previous = " ".join(sentences[index - 1].split())
        return f"{previous} {text}"
    return text


def _age_days(claim_date, published) -> int | None:
    """How many days the page was old when the claim circulated.

    Positive means the page came first. None when either date is missing,
    so an undated page is not treated as fresh or as stale.
    """
    claim = _parse_date(claim_date)
    page = _parse_date(published)
    if claim is None or page is None:
        return None
    return (claim - page).days


def _stats(
    *,
    passages,
    labelled,
    evidence,
    pages,
    searches,
    llm_calls,
    usage,
    unverified,
    notes,
    started,
    queries,
    topup_used,
    topup_queries,
    stance_failed=False,
    prior_evidence=None,
) -> dict:
    supports = sum(1 for row in labelled if row["stance"] == "supports")
    refutes = sum(1 for row in labelled if row["stance"] == "refutes")
    neutral = sum(1 for row in labelled if row["stance"] == "neutral")
    support_groups = {row["group_id"] for row in evidence if row["stance"] == "supports"}
    refute_groups = {row["group_id"] for row in evidence if row["stance"] == "refutes"}
    sightings = sum(1 for page in pages if page.fetch_status == "circulation")
    prompt_tokens = 0
    output_tokens = 0
    for row in usage:
        prompt_tokens += int(row.get("prompt_tokens") or 0)
        output_tokens += int(row.get("output_tokens") or 0)
    return {
        "searches": searches,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "n_passages": len(passages),
        "n_supports": supports,
        "n_refutes": refutes,
        "n_neutral": neutral,
        "groups_supporting": len(support_groups),
        "groups_refuting": len(refute_groups),
        "circulation_sightings": sightings,
        "seconds": round(time.perf_counter() - started, 2),
        "notes": notes,
        "quote_unverified": unverified,
        # Every query this claim has run, including the previous round.
        # The next call skips these. topup_queries is only this call.
        "queries": queries,
        "topup_used": topup_used,
        "topup_queries": topup_queries,
        # True only when the stance call itself failed. Box 9 must read this
        # as "system error -> retry / human", never as "insufficient evidence".
        "stance_failed": bool(stance_failed),
        # Groups that argue both sides. Box 9 (the gate) must count each of
        # these on neither side and flag the claim. The rows stay in evidence.
        "group_conflicts": _conflicting_groups(evidence, prior_evidence or []),
        # The same totals, split by atomic claim. A passage can support A1
        # and say nothing about A2; the top-level counts would hide that.
        "by_atomic": _by_atomic(labelled, evidence),
        # Neutral rows are not evidence. They are kept here so a run with
        # an empty table can still show why each passage was not a hit.
        "stance_rows": _stance_rows(labelled),
    }


def _old_event(claim: str, claim_date, passage: Passage) -> bool:
    """True when a 'now' claim is being propped up by an old page."""
    if not isinstance(claim, str) or _CURRENT_CLAIM.search(claim) is None:
        return False
    if not isinstance(claim_date, str):
        return False
    age = _age_days(claim_date, passage.published_date)
    return age is not None and age >= _OLD_EVENT_DAYS


def _prior_queries(state: dict) -> list[str]:
    stats = state.get("investigate_stats")
    if not isinstance(stats, dict):
        return []
    found = []
    values = stats.get("queries")
    if isinstance(values, list):
        for item in values:
            text = " ".join(str(item).split())
            if text:
                found.append(text)
    return found


def _merge_queries(prior: list[str], ran: list[str]) -> list[str]:
    merged = []
    seen = set()
    for text in list(prior) + list(ran):
        key = _norm_query(text)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def _norm_query(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _next_evidence_id(prior: list) -> int:
    highest = 0
    for row in prior:
        if not isinstance(row, dict):
            continue
        label = str(row.get("id") or "")
        if label.startswith("E") and label[1:].isdigit():
            highest = max(highest, int(label[1:]))
    return highest + 1


def _by_atomic(labelled: list[dict], evidence: list[dict]) -> dict:
    """Supports, refutes, and independent groups for each atomic claim.

    A missing id is A1. That is the single-claim path, and it keeps the
    per-claim block equal to the top-level counts.
    """
    order: list[str] = []

    def _add(row: dict) -> None:
        aid = str(row.get("atomic_id") or "A1")
        if aid not in order:
            order.append(aid)

    for row in labelled:
        _add(row)
    for row in evidence:
        _add(row)
    if not order:
        order = ["A1"]
    blocks = {}
    for aid in order:
        lab = [row for row in labelled if str(row.get("atomic_id") or "A1") == aid]
        ev = [row for row in evidence if str(row.get("atomic_id") or "A1") == aid]
        blocks[aid] = {
            "n_supports": sum(1 for row in lab if row["stance"] == "supports"),
            "n_refutes": sum(1 for row in lab if row["stance"] == "refutes"),
            "n_neutral": sum(1 for row in lab if row["stance"] == "neutral"),
            "groups_supporting": len({row["group_id"] for row in ev if row["stance"] == "supports"}),
            "groups_refuting": len({row["group_id"] for row in ev if row["stance"] == "refutes"}),
        }
    return blocks


def _stance_rows(labelled: list[dict]) -> list[dict]:
    """One line per judgement, including neutrals, for the demo and the audit."""
    rows = []
    for row in labelled:
        passage = row.get("passage")
        if passage is None:
            continue
        rows.append({
            "passage_id": passage.passage_id,
            "url": passage.url,
            "netloc": passage.netloc,
            "atomic_id": str(row.get("atomic_id") or "A1"),
            "stance": row.get("stance") or "",
            "reason": row.get("reason") or "",
        })
    return rows


def _strong_groups(evidence: list[dict]) -> int:
    """Independent groups that support or refute and are tier 1 or 2."""
    return len(
        {
            row["group_id"]
            for row in evidence
            if row.get("tier", 4) <= 2 and row.get("stance") in {"supports", "refutes"}
        }
    )


def _topup(state, labelled, passages, already):
    """One query proposal, then fetch and stance only the new pages.

    Returns whether the model was asked, the new labels, passages, pages,
    the queries that were actually run, how many quotes failed the check,
    and a note when the cap or the model stopped the top-up.
    """
    note = ""
    try:
        proposed = propose_topup(state, labelled, already)
    except LLMError as exc:
        if isinstance(exc, LLMQuotaError):
            raise
        return True, [], [], [], [], 0, str(exc)
    fresh = []
    seen = {_norm_query(item) for item in already}
    for text in proposed.queries:
        key = _norm_query(text)
        if not key or key in seen:
            continue
        seen.add(key)
        fresh.append(text)
        if len(fresh) >= _TOPUP_QUERIES:
            break
    if not fresh:
        return True, [], [], [], [], 0, ""
    known = {canonical_url(passage.url) for passage in passages}
    try:
        bundle = gather_passages(state, extra_queries=fresh, known_urls=known)
    except SearchQuotaError as exc:
        return True, [], [], [], [], 0, str(exc)
    # Continue the ids. Reusing P1 makes the group map overwrite the first pass.
    chosen = _renumber(list(bundle["passages"])[:_STANCE_LIMIT], start=len(passages) + 1)
    ran = list(bundle["queries"])
    if not chosen:
        return True, [], [], bundle["pages"], ran, 0, ""
    try:
        batch = judge_stance(state, chosen)
    except LLMError as exc:
        if isinstance(exc, LLMQuotaError):
            raise
        return True, [], chosen, bundle["pages"], ran, 0, str(exc)
    extra_labelled, unverified = apply_stance(
        _claim_text(state),
        chosen,
        batch.judgements,
        state.get("claim_date"),
        _atomic_items(state),
    )
    return True, extra_labelled, chosen, bundle["pages"], ran, unverified, note


def propose_topup(state: dict, labelled: list[dict], already: list[str]) -> TopupBatch:
    """One call. The prompt file holds the rules; the code caps the list at two."""
    found = []
    for row in labelled:
        if row["stance"] == "neutral":
            continue
        passage = row["passage"]
        found.append(
            f"{passage.netloc} tier {credibility_tier(passage.url)} {row['stance']}: {row['quote']}"
        )
    found_text = "\n".join(found[:8]) if found else "Nothing supported or refuted the claim."
    searched = "\n".join(already) if already else "(none)"
    prompt = (
        _TOPUP_PATH.read_text(encoding="utf-8")
        .replace("__CLAIM__", _claim_text(state) or "(none)")
        .replace("__FOUND__", found_text)
        .replace("__SEARCHED__", searched)
    )
    return call_structured(
        prompt,
        TopupBatch,
        purpose="topup_queries",
        prompt_version="topup_v1",
    )


def _groups_with_prior(passages: list[Passage], prior: list) -> dict[str, str]:
    """Group labels that do not collide with a previous round.

    A newsroom already seen keeps that group id. New groups continue the
    numbering, so a second call's first new site is not another G1.
    """
    groups = assign_groups(passages)
    if not prior:
        return groups
    domain_group: dict[str, str] = {}
    highest = 0
    for row in prior:
        if not isinstance(row, dict):
            continue
        label = str(row.get("group_id") or "")
        if label.startswith("G") and label[1:].isdigit():
            highest = max(highest, int(label[1:]))
        url = str(row.get("url") or "")
        host = registrable_domain(site_host(url) or "")
        if host and label:
            domain_group.setdefault(host, label)
    remap: dict[str, str] = {}
    for passage in passages:
        host = registrable_domain(site_host(passage.url) or "")
        earlier = domain_group.get(host)
        label = groups.get(passage.passage_id)
        if earlier and label:
            remap[label] = earlier
    renumber: dict[str, str] = {}
    next_id = highest + 1
    for label in groups.values():
        if label in remap or label in renumber:
            continue
        renumber[label] = f"G{next_id}"
        next_id += 1
    return {
        passage_id: remap.get(label, renumber.get(label, label))
        for passage_id, label in groups.items()
    }


def _conflict_pairs(*tables: list) -> set[tuple[str, str]]:
    """(group, atomic_id) pairs that argue both sides of the same claim.

    A newsroom can support the inauguration and refute the campus visit.
    That is two claims, not one conflict. A conflict is both stances on
    one atomic claim. Rows with no id are A1, so an older round still
    matches a new one.
    """
    stances: dict[tuple[str, str], set[str]] = {}
    for table in tables:
        for row in table:
            if not isinstance(row, dict):
                continue
            group = str(row.get("group_id") or "")
            stance = row.get("stance")
            if not group or stance not in {"supports", "refutes"}:
                continue
            aid = str(row.get("atomic_id") or "A1")
            stances.setdefault((group, aid), set()).add(stance)
    return {
        key for key, sides in stances.items() if "supports" in sides and "refutes" in sides
    }


def _conflicting_groups(*tables: list) -> list[str]:
    """Group ids that contain both a support and a refute of one claim.

    One outlet cannot be an independent vote for the claim and against it.
    Box 9 (the gate) must count such a group on neither side and flag it.
    """
    return sorted({group for group, _aid in _conflict_pairs(*tables)})


def _mark_group_conflicts(rows: list[dict], prior: list) -> None:
    """Set group_conflict on every current row whose group argues both sides.

    The rows are not deleted. Dropping the support or the refute would hide
    the disagreement, and the gate needs both sentences. A group that only
    supports, or only refutes, is not a conflict. A previous round counts:
    a new refute of a group that already supports is the same disagreement.
    """
    conflicts = _conflict_pairs(rows, prior)
    for row in rows:
        row["group_conflict"] = (
            str(row.get("group_id") or ""),
            str(row.get("atomic_id") or "A1"),
        ) in conflicts


def _drop_repeated_quotes(rows: list[dict], prior: list) -> list[dict]:
    """One sentence, one row, inside a group. The earlier row is the better tier."""
    seen = set()
    for row in prior:
        if not isinstance(row, dict):
            continue
        seen.add(_quote_key(row))
    kept = []
    for row in rows:
        # The same sentence can be evidence for A1 and not for A2. The
        # atomic id keeps the second row.
        key = _quote_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _quote_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("group_id") or ""),
        str(row.get("atomic_id") or "A1"),
        _flat_quotes(str(row.get("quote") or "")).casefold(),
    )


def build_queries(state: dict) -> list[Query]:
    """The plan for one claim. Deterministic. No model call.

    neutral asks for the claim in its own words, which is how a straight
    report headlines it. verify asks whether the named things are true,
    which is how a careful page titles a check without being a fact-check
    site. refute asks for the pages that already call it false. official
    asks the office that would have issued the order. lead, only after a
    similar prior, asks for the older incident in the wording people used
    then, which is the history the new wording has to be compared with.

    Those lines use the core atomic claim when extract split the
    submission. Each other atomic claim adds one neutral line, because
    that event is not in the core sentence. With no atomic claims, or
    only one, the plan is the same five lines as before.
    """
    items = _atomic_items(state)
    extras: list[dict] = []
    if items:
        cores = [item for item in items if item["core"]]
        core = cores[0] if cores else items[0]
        claim = core["text"]
        if len(items) >= 2:
            # A motive ("the government is hiding it") is not an event.
            # Searching it spends a slot the top-up needs for the cure.
            extras = [
                item for item in items
                if item["id"] != core["id"] and item.get("checkable", True)
            ]
    else:
        claim = _claim_text(state)
    trimmed = _trim_words(claim, _NEUTRAL_WORDS)
    queries: list[Query] = []
    if trimmed:
        queries.append(Query(text=trimmed, kind="neutral"))
        # Names from the rest of a long article blow the query into a
        # list. Only entities that are actually in the core sentence,
        # and at most five of them, belong on the verify and official lines.
        scoped = _entities_in_claim(state.get("entities"), claim, limit=5)
        verify = _verify_text(claim, scoped)
        if verify:
            queries.append(Query(text=verify, kind="verify"))
        queries.append(
            Query(text=f"{trimmed} {_REFUTE_SUFFIX}", kind="refute")
        )
        official = _official_domains(state.get("domain"))
        if official:
            # Keywords, not the sentence. A circular is filed under the
            # names and the figures, and "true or false" is not how a
            # ministry page is written.
            keywords = _official_text(claim, scoped)
            if keywords:
                queries.append(
                    Query(text=keywords, kind="official", include_domains=official)
                )
    seen = {_norm_query(query.text) for query in queries}
    for item in extras:
        # The core sentence was already searched. A second copy of it
        # would spend a slot and return the same pages.
        text = _trim_words(item["text"], _NEUTRAL_WORDS)
        key = _norm_query(text)
        if not text or key in seen:
            continue
        seen.add(key)
        queries.append(Query(text=text, kind="neutral"))
    lead = _lead_text(state)
    if lead:
        queries.append(Query(text=lead, kind="lead"))
    # No split: five lines, as before. A split stops at six so the
    # top-up still has two searches inside the per-claim cap of eight.
    cap = _PLAN_CAP if extras else _MAX_QUERIES
    return queries[:cap]


def gather_candidates(
    state: dict,
    *,
    skip_queries: list[str] | None = None,
    extra_queries: list[str] | None = None,
    ran: list[str] | None = None,
) -> list[Hit]:
    """Run the plan and return the best unique pages.

    Does not reset the per-claim search counter. The caller does that once
    when the claim starts, so a second round shares the same budget.

    URLs are canonicalised before they are compared, so a tracking
    parameter cannot make one page look like two. The copy we keep is the
    one with the higher score. Before the cap of 25, the best three hits
    of each query kind are reserved, and the remaining slots fill by
    score. The list stays in that order so the fetch cap sees the
    reserved pages first.
    """
    found: dict[str, dict] = {}
    queries = build_queries(state)
    # The four or five lines are independent. Running them together is
    # what keeps one slow raw-content reply from stacking four waits.
    # map keeps plan order, so a URL found by two lines still records
    # verify before refute.
    # Tavily's end_date means "published before this day", so the day
    # after the claim is "on or before the claim". Sent only while the
    # leakage guard is on. Undated hits still come back: asking Tavily to
    # drop them would hide the pages the date-only read can still date.
    # time_range is relative to today, which is the wrong window for an
    # archive claim. Neither parameter adds a credit.
    end_date = None
    if config.EVAL_DROP_AFTER_CLAIM:
        end_date = _search_end_date(state.get("claim_date"))
    if extra_queries:
        # The top-up replaces the plan for this call. The plan already ran.
        queries = [
            Query(text=text, kind="topup")
            for text in extra_queries
            if _norm_query(text)
        ]
    else:
        skip = {_norm_query(item) for item in (skip_queries or [])}
        queries = [query for query in queries if _norm_query(query.text) not in skip]
    batches = _run_queries(queries, end_date, ran)
    submitted = _submitted_page(state)
    for query, hits in batches:
        for hit in hits:
            key = canonical_url(hit.url)
            if not key:
                continue
            # The article the person pasted is not a source for itself.
            if submitted and _same_page(key, submitted):
                continue
            slot = found.get(key)
            if slot is None:
                found[key] = {"hit": hit, "kinds": [query.kind], "score": hit.score}
                continue
            if query.kind not in slot["kinds"]:
                slot["kinds"].append(query.kind)
            # A tie keeps the earlier copy. The score did not improve,
            # and the earlier query is the one we would rather cite.
            if hit.score > slot["score"]:
                slot["hit"] = hit
                slot["score"] = hit.score
    ranked = sorted(found, key=lambda key: (-found[key]["score"], key))
    reserved = _reserved_keys(found)
    # Reserved keys first, in score order, then the cap fills from the
    # same ranking. This order is what the fetch cap walks. Re-sorting the
    # whole list by score would push a weak official page past the twelfth
    # slot, and that page would never be downloaded.
    chosen = [key for key in ranked if key in reserved][:_MAX_HITS]
    chosen_set = set(chosen)
    for key in ranked:
        if key in chosen_set:
            continue
        if len(chosen) >= _MAX_HITS:
            break
        chosen.append(key)
        chosen_set.add(key)
    kept = []
    for key in chosen:
        slot = found[key]
        kept.append(
            slot["hit"].model_copy(
                update={"url": key, "query_kinds": list(slot["kinds"])}
            )
        )
    return kept


def _search_end_date(claim_date) -> str | None:
    parsed = _parse_date(claim_date)
    if parsed is None:
        return None
    return (parsed + timedelta(days=1)).isoformat()


def _run_queries(
    queries: list[Query], end_date: str | None = None, ran: list[str] | None = None
) -> list[tuple[Query, list[Hit]]]:
    if not queries:
        return []

    def run(query: Query):
        # The kind is in the purpose because the usage row has no kind
        # field of its own, and box 16 already stores purpose.
        # A failed line is a miss. The other lines still stand. The quota
        # is not a miss: spending more after the cap is the thing it stops.
        # A refused query is not recorded, so the next round can still try it.
        try:
            hits = search(
                query.text,
                purpose=f"investigate_{query.kind}",
                max_results=_RESULTS_PER_QUERY,
                include_domains=query.include_domains,
                end_date=end_date,
            )
        except SearchQuotaError:
            raise
        except SearchError:
            if ran is not None:
                ran.append(query.text)
            return query, []
        if ran is not None:
            ran.append(query.text)
        return query, hits

    workers = min(4, len(queries))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, queries))


def _reserved_keys(found: dict[str, dict]) -> set[str]:
    by_kind: dict[str, list[str]] = {}
    for key, slot in found.items():
        for kind in slot["kinds"]:
            by_kind.setdefault(kind, []).append(key)
    reserved = set()
    for keys in by_kind.values():
        keys.sort(key=lambda key: (-found[key]["score"], key))
        reserved.update(keys[:_PER_KIND])
    return reserved


def canonical_url(url: str) -> str:
    """One page, one string.

    Lowercase host, no leading www, no fragment, no trailing slash, and no
    utm_*, fbclid or gclid. Remaining query parameters are sorted so
    ?b=1&a=2 and ?a=2&b=1 are the same page. The path keeps its case:
    paths can be case-sensitive, hosts cannot.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    parts = urlsplit(raw)
    host = site_host(raw)
    if not host:
        return ""
    scheme = (parts.scheme or "https").lower()
    port = parts.port
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    path = parts.path or ""
    if path == "/":
        path = ""
    elif path.endswith("/"):
        path = path.rstrip("/")
    kept = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if _is_tracking(key):
            continue
        kept.append((key, value))
    kept.sort()
    return urlunsplit((scheme, netloc, path, urlencode(kept), ""))


def _is_tracking(key: str) -> bool:
    name = key.casefold()
    return name.startswith("utm_") or name in _TRACKING


def _submitted_page(state: dict) -> str:
    """The canonical URL of a page the person submitted. Empty otherwise.

    A text forward and an image have no page to exclude. raw_text on a
    URL run is the address they pasted.
    """
    if state.get("input_type") != "url":
        return ""
    for key in ("raw_text", "raw_input"):
        value = state.get(key)
        if isinstance(value, str) and "://" in value:
            return canonical_url(value)
    return ""


def _same_page(left: str, right: str) -> bool:
    """True when two URLs are the same article, scheme aside.

    canonical_url keeps http and https distinct. A search result that
    flips the scheme is still the page the person already handed us.
    """
    a = canonical_url(left)
    b = canonical_url(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return a.split("://", 1)[-1] == b.split("://", 1)[-1]


def _is_submitted_page(state: dict, url: str) -> bool:
    submitted = _submitted_page(state)
    return bool(submitted) and _same_page(url, submitted)


def _entities_in_claim(entities, claim: str, *, limit: int) -> dict:
    """At most `limit` entity strings that actually occur in the claim.

    The order is the order _entity_bits already walks, so the verify
    line stays stable. An organisation named only in a later paragraph
    of a long article is not in the core sentence and is left out.
    """
    if not isinstance(entities, dict) or limit <= 0:
        return {}
    haystack = " ".join(str(claim).casefold().split())
    if not haystack:
        return {}
    keys = list(_ENTITY_KEYS) + sorted(
        key for key in entities if key not in _ENTITY_KEYS
    )
    kept: dict[str, list[str]] = {}
    count = 0
    for key in keys:
        value = entities.get(key)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            text = " ".join(str(item).split())
            if not text or text.casefold() not in haystack:
                continue
            if count >= limit:
                return kept
            kept.setdefault(key, []).append(text)
            count += 1
    return kept


def _claim_text(state: dict) -> str:
    # claim is the checkable sentence. match_text is what box 5 embedded.
    # text is the whole forward, which may end in "forward to all groups".
    for key in ("claim", "match_text", "text"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _trim_words(text: str, limit: int) -> str:
    return " ".join(text.split()[:limit])


def _verify_text(claim: str, entities) -> str | None:
    bits = _entity_bits(entities)
    covered = set()
    for bit in bits:
        covered.add(bit.casefold())
        for token in bit.split():
            key = token.strip(_EDGE_PUNCT).casefold()
            if key:
                covered.add(key)
    words = []
    for raw in claim.split():
        cleaned = raw.strip(_EDGE_PUNCT)
        key = cleaned.casefold()
        if not key or key in _STOP or key in covered:
            continue
        words.append(cleaned)
        covered.add(key)
        if len(words) >= _SUBJECT_WORDS:
            break
    parts = bits + words
    # "true or false" on its own matches every check on the web.
    if not parts:
        return None
    return " ".join(parts) + " true or false"


def _entity_bits(entities) -> list[str]:
    if not isinstance(entities, dict):
        return []
    bits = []
    seen = set()
    keys = list(_ENTITY_KEYS) + sorted(
        key for key in entities if key not in _ENTITY_KEYS
    )
    for key in keys:
        value = entities.get(key)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            text = " ".join(str(item).split())
            if not text or text.casefold() in seen:
                continue
            tokens = [
                token.strip(_EDGE_PUNCT).casefold()
                for token in text.split()
            ]
            tokens = [token for token in tokens if token]
            # "2024" on its own adds nothing once "1 December 2024" is in.
            if tokens and all(token in seen for token in tokens):
                continue
            seen.add(text.casefold())
            seen.update(tokens)
            bits.append(text)
    return bits


def _official_text(claim: str, entities) -> str | None:
    bits = []
    for bit in _entity_bits(entities):
        kept = _without_media(bit)
        if kept:
            bits.append(kept)
    covered = set()
    for bit in bits:
        covered.add(bit.casefold())
        for token in bit.split():
            key = token.strip(_EDGE_PUNCT).casefold()
            if key:
                covered.add(key)
    words = []
    for raw in claim.split():
        cleaned = raw.strip(_EDGE_PUNCT)
        key = cleaned.casefold()
        # Media words describe the forward ("video shows"), not the event
        # a ministry search box is indexed under.
        if not key or key in _STOP or key in _MEDIA or key in covered:
            continue
        words.append(cleaned)
    # No tagger on this path. The longest remaining words are the nouns
    # and verbs a circular is filed under. "hot" loses to "coronavirus".
    # A length tie keeps the earlier word so the query is stable.
    ranked = sorted(enumerate(words), key=lambda pair: (-len(pair[1]), pair[0]))
    chosen = [index for index, _word in ranked[:_CORE_MAX]]
    core = [words[index] for index in sorted(chosen)]
    parts = bits + core
    if not parts:
        return None
    return " ".join(parts)


def _without_media(text: str) -> str:
    kept = []
    for raw in text.split():
        cleaned = raw.strip(_EDGE_PUNCT)
        if cleaned and cleaned.casefold() not in _MEDIA:
            kept.append(cleaned)
    return " ".join(kept)


def _official_domains(domain) -> list[str] | None:
    if not isinstance(domain, str):
        return None
    mapping = config.OFFICIAL_DOMAINS.get(domain)
    if not mapping:
        return None
    return [str(host) for host in mapping]


def _lead_text(state: dict) -> str | None:
    # Only a similar prior. An exact lead is the same sentence the neutral
    # query already sends. A "none" has no history. The similar lead is a
    # related claim whose wording differs in the detail box 6 refused to
    # treat as the same event, and that wording is the history.
    if state.get("match_type") != "similar":
        return None
    leads = state.get("leads")
    if not isinstance(leads, list):
        return None
    for lead in leads:
        if not isinstance(lead, dict):
            continue
        text = lead.get("primary_claim")
        if isinstance(text, str) and text.strip():
            # First lead with text. Box 6 already chose the reading order;
            # re-ranking here would second-guess it.
            return _trim_words(" ".join(text.split()), _LEAD_WORDS)
    return None


class Passage(BaseModel):
    """One window of a page, close enough to the claim to keep.

    after_claim is True when the page date is later than the claim date,
    False when it is not, and None when either date is missing. The eval
    drops the True ones so a page published afterwards cannot leak.
    """

    passage_id: str
    url: str
    netloc: str
    title: str
    text: str
    cos: float
    # cos plus the entity nudge. Ranking uses this. cos stays the raw
    # similarity so a later reader can see the boost did not invent it.
    score: float = 0.0
    published_date: str | None = None
    date_source: str | None = None
    fetched: bool
    query_kinds: list[str] = Field(default_factory=list)
    after_claim: bool | None = None


def gather_passages(
    state: dict,
    *,
    skip_queries: list[str] | None = None,
    extra_queries: list[str] | None = None,
    known_urls: set[str] | None = None,
) -> dict:
    """Search, fetch, and pick passages for one claim.

    Does not call a language model. The cosine is BGE-M3, loaded once.
    After the passages are chosen, URLs that still have no trusted date
    are dated from the path or from the head of the HTML, inside whatever
    is left of the claim budget. Stance is not called.
    """
    started = time.perf_counter()
    deadline = time.monotonic() + config.CLAIM_BUDGET_S
    ran: list[str] = []
    hits = gather_candidates(
        state, skip_queries=skip_queries, extra_queries=extra_queries, ran=ran
    )
    pages = fetch_pages(hits)
    passages, cosines, dropped_after = select_passages(state, pages)
    finished = finish_passages(state, passages, deadline=deadline)
    kept = finished["passages"]
    if known_urls:
        # The top-up stances only pages this call has not already read.
        kept = _renumber(
            [passage for passage in kept if canonical_url(passage.url) not in known_urls]
        )
    return {
        "hits": hits,
        "pages": pages,
        "passages": kept,
        "queries": ran,
        "cosines": cosines,
        "dropped_after_claim": dropped_after + finished["dropped_after_claim"],
        "dropped_undated": finished["dropped_undated"],
        "passages_if_keep": finished["passages_if_keep"],
        "passages_if_drop": finished["passages_if_drop"],
        "seconds": time.perf_counter() - started,
    }


def select_passages(state: dict, pages: list[Page]) -> tuple[list[Passage], list[float], int]:
    """Top windows per page, then the best across the claim.

    The cosine list is every window that was long enough to score, before
    the floor and the caps, so a run can see whether 0.45 is cutting real
    text or only noise. Returned passages are best first: at most two per
    page, at most two per registrable domain, and forty per claim.

    The third number is how many of those per-page slots were removed
    because the page is dated after the claim. It is zero when that guard
    is off. Unknown dates are not in the count.
    """
    claim = _claim_text(state)
    claim_date = state.get("claim_date") if isinstance(state.get("claim_date"), str) else None
    terms = _key_terms(claim, state.get("entities"))
    windows: list[dict] = []
    for page in pages:
        if not _page_can_support(page):
            continue
        # A cached page can still be the whole site. Windows are cut from
        # the head only, which is the part that discusses the claim.
        source = page.text
        if len(source) > 20_000:
            source = source[:20_000]
        for text in passage_windows(source):
            if _junk_window(text) or _quotes_factcheck(text):
                continue
            windows.append({"page": page, "text": text})
    if not windows or not claim:
        return [], [], 0
    cosines = _cosines(claim, [item["text"] for item in windows])
    scored = []
    for item, cos in zip(windows, cosines):
        page = item["page"]
        cosine = float(cos)
        scored.append(
            {
                "page": page,
                "text": item["text"],
                "cos": cosine,
                "score": _boosted(cosine, item["text"], terms),
                "after_claim": _after_claim(page.published_date, claim_date),
            }
        )
    all_cos = [row["cos"] for row in scored]
    scored = _drop_duplicate_windows(scored)
    by_url: dict[str, list[dict]] = {}
    for row in scored:
        by_url.setdefault(row["page"].url, []).append(row)
    kept = []
    dropped_after = 0
    for rows in by_url.values():
        # Score, then cosine. The floor below is still the cosine: the
        # boost is a reorder, not a way under the floor.
        rows.sort(key=lambda row: (-row["score"], -row["cos"], row["text"]))
        above = [row for row in rows if row["cos"] >= config.PASSAGE_MIN_COS]
        if config.EVAL_DROP_AFTER_CLAIM:
            # Count the slots the guard takes, then backfill from the
            # older windows on the same page. Unknown dates stay: dropping
            # them would hide every undated page, which is most of the web.
            dropped_after += sum(
                1 for row in above[:_PER_PAGE] if row["after_claim"] is True
            )
            above = [row for row in above if row["after_claim"] is not True]
        kept.extend(above[:_PER_PAGE])
    kept.sort(key=lambda row: (-row["score"], -row["cos"], row["page"].url, row["text"]))
    kept = _cap_per_domain(kept)
    passages = []
    for index, row in enumerate(kept[:_MAX_PASSAGES], start=1):
        page = row["page"]
        passages.append(
            Passage(
                passage_id=f"P{index}",
                url=page.url,
                netloc=site_host(page.url),
                title=page.title,
                text=row["text"],
                cos=row["cos"],
                score=row["score"],
                published_date=page.published_date,
                date_source=page.date_source,
                fetched=page.fetched,
                query_kinds=list(page.query_kinds),
                after_claim=row["after_claim"],
            )
        )
    return passages, all_cos, dropped_after


def finish_passages(state: dict, passages: list[Passage], *, deadline: float) -> dict:
    """Date the kept passages that still have none, then apply the undated rule.

    Path dates and a short HTML read happen here, after ranking, so a page
    is not downloaded twice just to be thrown away by the cosine floor.
    A date that turns out to be after the claim is dropped when that guard
    is on. Undated passages are dropped only when both the guard and
    EVAL_UNDATED say so. passages_if_keep and passages_if_drop are the two
    counts after the after-claim drop, so the eval can report both without
    a second search.
    """
    claim_date = state.get("claim_date") if isinstance(state.get("claim_date"), str) else None
    urls: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        if passage.published_date or passage.url in seen:
            continue
        seen.add(passage.url)
        urls.append(passage.url)
    found = date_missing(urls, deadline=deadline) if urls else {}
    kept: list[Passage] = []
    dropped_after = 0
    for passage in passages:
        dated = found.get(passage.url)
        if dated and not passage.published_date:
            published, source = dated
            after = _after_claim(published, claim_date)
            if config.EVAL_DROP_AFTER_CLAIM and after is True:
                dropped_after += 1
                continue
            passage = passage.model_copy(
                update={
                    "published_date": published,
                    "date_source": source,
                    "after_claim": after,
                }
            )
            # The cache must keep this. A later run that cannot open the
            # host would otherwise lose the date again.
            remember_publication(passage.url, published, source or "htmldate")
        kept.append(passage)
    if_keep = len(kept)
    undated = sum(1 for passage in kept if not passage.published_date)
    dropped_undated = 0
    if config.EVAL_UNDATED == "drop" and config.EVAL_DROP_AFTER_CLAIM:
        before = len(kept)
        kept = [passage for passage in kept if passage.published_date]
        dropped_undated = before - len(kept)
    return {
        "passages": _renumber(kept),
        "dropped_after_claim": dropped_after,
        "dropped_undated": dropped_undated,
        "passages_if_keep": if_keep,
        "passages_if_drop": if_keep - undated,
    }


def _renumber(passages: list[Passage], start: int = 1) -> list[Passage]:
    return [
        passage.model_copy(update={"passage_id": f"P{index}"})
        for index, passage in enumerate(passages, start=start)
    ]


def _key_terms(claim: str, entities) -> list[str]:
    """Names, places, numbers, and amounts. Used only to nudge the rank.

    A capitalised word in the claim is a name or a place, except the first
    word of the sentence when the next word is not capitalised too: that
    first word is just how a sentence starts ("Drinking hot water"). An
    acronym in that position still counts ("HSBC left"). State entities
    are added even when the claim text did not capitalise them. A number
    or an amount in the claim counts on its own.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = " ".join(str(term).split()).strip(_EDGE_PUNCT)
        key = cleaned.casefold()
        if not key or key in seen or key in _STOP or key in _MEDIA:
            return
        seen.add(key)
        terms.append(cleaned)

    for bit in _entity_bits(entities):
        add(bit)
    for term in _capitalised_terms(claim):
        # Already covered by a longer entity ("Sri" inside "Sri Lanka").
        if any(term.casefold() in bit.casefold().split() for bit in terms):
            continue
        add(term)
    for term in _amounts(claim):
        if any(term.casefold() in bit.casefold().split() for bit in terms):
            continue
        add(term)
    return terms


def _capitalised_terms(claim: str) -> list[str]:
    words = claim.split()
    found = []
    for index, raw in enumerate(words):
        cleaned = raw.strip(_EDGE_PUNCT)
        if not _is_name_token(cleaned):
            continue
        if index == 0 and not _opens_as_name(cleaned, words):
            continue
        if cleaned.casefold() in _STOP or cleaned.casefold() in _MEDIA:
            continue
        found.append(cleaned)
    return found


def _is_name_token(token: str) -> bool:
    if not token or not any(char.isalpha() for char in token):
        return False
    return token[:1].isupper()


def _opens_as_name(token: str, words: list[str]) -> bool:
    """The first word counts when it is an acronym or the start of a name."""
    letters = [char for char in token if char.isalpha()]
    if len(letters) >= 2 and all(char.isupper() for char in letters):
        return True
    if len(words) < 2:
        return False
    nxt = words[1].strip(_EDGE_PUNCT)
    return _is_name_token(nxt)


def _amounts(claim: str) -> list[str]:
    # Currency first, so "₹500" is one term and not also the bare "500".
    pattern = re.compile(
        r"(?:₹|rs\.?)\s?\d[\d,]*(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b",
        re.IGNORECASE,
    )
    return [match.group(0) for match in pattern.finditer(claim)]


def _boosted(cosine: float, text: str, terms: list[str]) -> float:
    if not terms:
        return cosine
    present = sum(1 for term in terms if _contains_term(text, term))
    return cosine + float(config.ENTITY_BOOST) * (present / len(terms))


def _contains_term(text: str, term: str) -> bool:
    needle = " ".join(term.casefold().split())
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text.casefold()) is not None


def _page_can_support(page: Page) -> bool:
    # A sighting and a page we could not read in English have no passage.
    # Checked on the host as well as the status, so a snippet that was
    # kept for a social URL still cannot be quoted as evidence.
    if page.fetch_status in {"circulation", "non_english"}:
        return False
    if not (page.text or "").strip():
        return False
    # A fact-check that slipped past the search filter is still the answer
    # key when the eval is blocked. It does not become a passage.
    if factcheck_blocked(page.url, page.title):
        return False
    host = registrable_domain(site_host(page.url) or page.url)
    return host not in config.NON_EVIDENCE_HOSTS


def _drop_duplicate_windows(rows: list[dict]) -> list[dict]:
    """One copy of a window that was pasted onto two pages.

    The higher cosine wins. A tie keeps the earlier URL so the choice
    does not flicker. The text compared is the words, not the raw
    spacing: a newline does not make a second passage.
    """
    best: dict[str, dict] = {}
    for row in rows:
        key = " ".join(row["text"].split())
        current = best.get(key)
        if current is None or row["cos"] > current["cos"]:
            best[key] = row
            continue
        if row["cos"] == current["cos"] and row["page"].url < current["page"].url:
            best[key] = row
    return list(best.values())


def _cap_per_domain(rows: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    kept = []
    for row in rows:
        host = registrable_domain(site_host(row["page"].url) or row["page"].url)
        if counts.get(host, 0) >= _PER_DOMAIN:
            continue
        counts[host] = counts.get(host, 0) + 1
        kept.append(row)
    return kept


# A window that is mostly the cookie bar or the login wall. One mention of
# "subscribe" in an article is not this: the share of the words has to be
# the chrome, not the story.
_JUNK_WORDS = frozenset(
    {
        "login", "signin", "signup", "subscribe", "unsubscribe", "cookie",
        "cookies", "consent", "newsletter", "gdpr",
    }
)
_JUNK_PHRASES = (
    "log in", "sign in", "sign up", "accept all", "accept cookies",
    "cookie policy", "privacy policy", "manage preferences",
    "forgot password", "forgot account", "create an account",
    "subscribe to",
)


def _junk_window(text: str) -> bool:
    folded = " ".join(text.casefold().split())
    words = [word.strip(_EDGE_PUNCT) for word in folded.split()]
    words = [word for word in words if word]
    if not words:
        return True
    hits = sum(1 for word in words if word in _JUNK_WORDS)
    for phrase in _JUNK_PHRASES:
        if phrase in folded:
            hits += len(phrase.split())
    return hits / len(words) >= 0.4


# Phrases a fact-check uses to announce its verdict. In the blocked eval
# these sentences are the answer key. A news page that merely reports the
# event does not say them. Off when the block is off, so a demo can still
# read a fact-check it opened on purpose.
_FACTCHECK_PHRASES = (
    "pib fact check",
    "#pibfactcheck",
    "fact check",
    "fact-check",
    "fact-checked",
    "debunked by",
)


def _quotes_factcheck(text: str) -> bool:
    if not config.BLOCK_FACTCHECK_SITES:
        return False
    folded = text.casefold()
    return any(phrase in folded for phrase in _FACTCHECK_PHRASES)


def passage_windows(text: str) -> list[str]:
    """Windows of about three sentences, overlapping by one.

    A window under 20 words is a fragment (a caption, a "Rs." split) and
    is not a passage. A page with only one or two sentences is one window
    when it is long enough.
    """
    sentences = _sentences(text)
    if not sentences:
        return []
    if len(sentences) <= _WINDOW_SENTENCES:
        chunk = " ".join(sentences)
        return [chunk] if len(chunk.split()) >= _MIN_WINDOW_WORDS else []
    step = _WINDOW_SENTENCES - _WINDOW_OVERLAP
    windows = []
    index = 0
    while index < len(sentences):
        chunk = " ".join(sentences[index:index + _WINDOW_SENTENCES])
        if len(chunk.split()) >= _MIN_WINDOW_WORDS:
            windows.append(chunk)
        if index + _WINDOW_SENTENCES >= len(sentences):
            break
        index += step
    return windows


def _sentences(text: str) -> list[str]:
    raw = " ".join((text or "").split())
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+", raw)
    return [part.strip() for part in parts if part.strip()]


def _cosines(claim: str, windows: list[str]) -> list[float]:
    claim_vec = _encode([_flat(claim)])[0]
    rows = []
    flat = [_flat(window) for window in windows]
    for start in range(0, len(flat), _ENCODE_BATCH):
        rows.append(_encode(flat[start:start + _ENCODE_BATCH]))
    matrix = np.vstack(rows)
    return (matrix @ claim_vec).tolist()


def _encode(texts: list[str]) -> np.ndarray:
    vectors = get_encoder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _after_claim(published, claim_date) -> bool | None:
    page = _parse_date(published)
    claim = _parse_date(claim_date)
    if page is None or claim is None:
        return None
    return page > claim


def _parse_date(value) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    try:
        parsed = datetime.strptime(text[:10], "%Y/%m/%d")
        return parsed.date()
    except ValueError:
        return None
