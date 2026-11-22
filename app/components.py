"""HTML for the checker page. Builders return strings; they do not call Streamlit.

Every claim, passage, quote, URL, and model string is escaped before it
can reach unsafe_allow_html. A javascript: URL is not a link: only http
and https are written into href, and those links carry rel="noopener
noreferrer". The engine is adding card fields in parallel with this UI,
so each new field is read with .get() and a default. A card from before
that merge still renders.

The figures on How it works live in STATS and nowhere else.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

# The wordmark. The star is the one coral character; theme.py colours it.
APP_NAME = "FACT✦LENS"

# One source for every figure painted on the page. Do not add a number
# that is not in these files.
# docs/ARCHITECTURE.md — 52,099 fact-checks, 15 Indian fact-checkers, 16 boxes.
# docs/eval_end_to_end.md — recycled rumours 40/40 (section C), blocked
#   new-rumour correct 8/50 against Gemini-alone 13/50.
# docs/eval_diagnosis.md — exact McNemar p 0.301758 (shown as 0.30, not
#   significant). The assisted 18/50 figure is post-hoc and is not shown.
STATS = {
    "memory": 52099,
    "memory_display": "52,099",
    "checkers": 15,
    "steps": 16,
    "recycled_hit": 40,
    "recycled_n": 40,
    "blocked_system": "8/50",
    "blocked_gemini": "13/50",
    "mcnemar_p": "0.30",
    "sample_n": 50,
}

# Graph node order. Boxes 1–2, 3–4, 8+10 and 14–15 share a node, so the
# stepper is 16 boxes and 12 nodes. Gate is box 9 and runs before assess
# (boxes 8 and 10); the list follows the run, not the box numbers.
STEPS = (
    {"node": "ingest", "group": "Ingest", "box": "1", "title": "Read the submission"},
    {"node": "ingest", "group": "Ingest", "box": "2", "title": "Clean text and language"},
    {"node": "extract", "group": "Extract", "box": "3", "title": "Write the claim"},
    {"node": "extract", "group": "Extract", "box": "4", "title": "Entities, domain, red flags"},
    {"node": "memory", "group": "Memory", "box": "5", "title": "Search past fact-checks"},
    {"node": "prior_match", "group": "Prior", "box": "6", "title": "Match a prior check"},
    {"node": "investigate", "group": "Investigate", "box": "7", "title": "Search and read the web"},
    {"node": "gate", "group": "Gate", "box": "9", "title": "Judge if the evidence is enough"},
    {"node": "assess", "group": "Assess", "box": "8", "title": "Draft the explanation"},
    {"node": "assess", "group": "Assess", "box": "10", "title": "Verdict and confidence"},
    {"node": "priority", "group": "Priority", "box": "11", "title": "Set priority"},
    {"node": "decide", "group": "Decide", "box": "12", "title": "Alert, store, or ask a person"},
    {"node": "review", "group": "Review", "box": "13", "title": "Human review"},
    {"node": "output", "group": "Output", "box": "14", "title": "Build the card"},
    {"node": "output", "group": "Output", "box": "15", "title": "Correction a person can forward"},
    {"node": "audit", "group": "Audit", "box": "16", "title": "Write the audit row"},
)

# The eleven domains the memory was built with. The page passes
# config.DOMAINS when it can; this copy keeps the marquee working in a
# unit test that does not import the engine.
DOMAINS = (
    "Politics & elections",
    "Religion & communal",
    "Defence & international",
    "Crime & law",
    "Society & other",
    "Health",
    "Disasters & accidents",
    "Finance & scams",
    "Entertainment & celebrities",
    "Sports",
    "Science & technology",
)

_VERDICT_COLOR = {
    "false": "#E05A47",
    "misleading": "#E8A33D",
    "true": "#3FB27F",
    "unverifiable": "#7C8AA5",
    "other": "#5B6475",
    "withheld": "#5B6475",
}

# Codes the eval and the new card already use. Anything else is shown
# with the underscores turned into words, so an unfamiliar code still
# reads as a sentence instead of a raw token.
_UNVERIFIABLE_PLAIN = {
    "no_credible_record": "No two independent credible sources were found.",
    "no_hits": "The search did not find a page about this claim.",
    "single_credible": "Only one credible source was found. Deciding takes two independent ones.",
    "only_tier34": "The only sources are lower tier, which is not enough to decide.",
    "insufficient": "The evidence is not enough to decide.",
    "stale": "The only sources are too old for this claim.",
    "picture": "Text sources cannot confirm what a picture shows.",
}

_SEARCH_NODES = {"memory", "prior_match", "investigate", "gate"}
_MISSING = object()
_CITE = re.compile(r"\[([A-Za-z][A-Za-z0-9]{0,11})\]")
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# What each of the 16 boxes is for, in the order a claim meets them.
# The arrow list is the how-it-works page. Copy stays short on purpose:
# the architecture doc is the contract, this is the map.
BOXES = (
    ("1", "Ingest", "Read the submission", "Text, a link, or a screenshot. The raw message is not what later boxes search."),
    ("2", "Ingest", "Clean text and language", "English proceeds. Hinglish is flagged. Anything else stops before a model call."),
    ("3", "Extract", "Write the claim", "One sentence, the way a forward says it. The model does not add a verdict."),
    ("4", "Extract", "Entities, domain, red flags", "Amounts, dates, names, one of 11 domains, and the warning signs on the card."),
    ("5", "Memory", "Search past fact-checks", "Hybrid search over the archive. Near-duplicate outlets collapse into one hit."),
    ("6", "Prior", "Match a prior check", "Exact and fresh reuses the archive. Similar, stale, or a different detail is checked again."),
    ("7", "Investigate", "Search and read the web", "A fixed query plan, a public-address check on every fetch, then a stance on each passage."),
    ("9", "Gate", "Judge if the evidence is enough", "Two independent groups on the same side, at least one of them tier 1 or 2. A credible clash waits for a person."),
    ("8", "Assess", "Draft the explanation", "The model writes the sentences and the correction. It is told not to pick the verdict."),
    ("10", "Assess", "Verdict and confidence", "Code picks the verdict from the gate and scores confidence from the table, not from the model."),
    ("11", "Priority", "Set priority", "Harm, novelty, reach, and urgency. Style can move the priority. It cannot move the verdict."),
    ("12", "Decide", "Alert, store, or ask a person", "High-priority False or Misleading can become an alert. An alert still waits for a person."),
    ("13", "Review", "Human review", "Approve, edit, or reject. The run is paused until that answer, including after a crash in an earlier box."),
    ("14", "Output", "Build the card", "The card copies the verdict the rules stored. It does not score the claim again."),
    ("15", "Output", "Correction a person can forward", "A short message, plus what is actually true when the table supports a correction."),
    ("16", "Audit", "Write the audit row", "One row per finished run: the verdict, the evidence ids, the cost. Not the raw forward."),
)


def esc(value) -> str:
    """Text that will be concatenated into HTML. None is an empty string."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def safe_href(url) -> str | None:
    """An href that is safe to put in an anchor, or None.

    javascript:, data:, and any other scheme are dropped. A URL with a
    quote or an angle bracket is dropped too: escaping would make it
    safe, and dropping it means a hostile string cannot be a link even
    if a later edit forgets to escape the attribute.
    """
    text = "" if url is None else str(url).strip()
    if not text.lower().startswith(("http://", "https://")):
        return None
    if any(ch in text for ch in " \t\r\n\"'<>"):
        return None
    return text


def anchor(url, label) -> str:
    """A link when the URL is http(s), otherwise the label as text."""
    href = safe_href(url)
    shown = esc(label)
    if href is None:
        return shown
    return (
        f'<a href="{esc(href)}" target="_blank" rel="noopener noreferrer">{shown}</a>'
    )


# A search that did not read a page. "ran" and "not_needed" are quiet:
# the first did the search, the second was never asked to.
_SEARCH_BANNER = frozenset({
    "cache_only_miss",
    "monthly_cap",
    "per_claim_cap",
    "error",
})
_RING_VERDICTS = frozenset({"false", "misleading", "true"})
# The assess template. It is a lie when the search never opened a page.
_UNREAD = re.compile(r"no source(?: we read| that we read)? addressed", re.IGNORECASE)


def plain_unverifiable(code) -> str:
    """The reason code in a sentence. Empty when the card has no code."""
    text = "" if code is None else str(code).strip()
    if not text:
        return ""
    known = _UNVERIFIABLE_PLAIN.get(text)
    if known:
        return known
    words = text.replace("_", " ").strip()
    if not words:
        return ""
    return words[:1].upper() + words[1:] + "."


def _drop_unread(text: str, *, read: bool) -> str:
    """Remove 'no source addressed this' when nothing was read.

    The sentence is the template for an empty evidence table. On a
    cache miss or a cap it describes a search that did not happen.
    """
    raw = "" if text is None else str(text).strip()
    if read or not raw or not _UNREAD.search(raw):
        return raw
    parts = re.split(r"(?<=[.!?])\s+", raw)
    kept = [part for part in parts if part and not _UNREAD.search(part)]
    return " ".join(kept).strip()


def sources_were_read(model: dict) -> bool:
    """True when a page is on the card, or the search actually ran.

    An empty table after a real search can still mean the pages we
    opened did not address the claim. A cap or a cache miss cannot.
    """
    model = model or {}
    if model.get("evidence"):
        return True
    return str(model.get("search_status") or "") == "ran"


def search_banner_text(model: dict) -> str:
    """Plain words for a search that did not finish. Empty when it did.

    'Didn't run' is only honest when no live call and no cache hit
    came back. A cap that stopped a later query says 'stopped'.
    """
    model = model or {}
    status = str(model.get("search_status") or "")
    if status not in _SEARCH_BANNER:
        return ""
    try:
        live = int(model.get("searches_live") or 0)
    except (TypeError, ValueError):
        live = 0
    try:
        cached = int(model.get("searches_cached") or 0)
    except (TypeError, ValueError):
        cached = 0
    quiet = live + cached <= 0
    if status == "monthly_cap":
        if quiet:
            return (
                "Search didn't run: the monthly search limit is reached. "
                "This result is based on past fact-checks only."
            )
        return "Search stopped: the monthly search limit is reached."
    if status == "per_claim_cap":
        if quiet:
            return (
                "Search didn't run: this claim reached its search limit. "
                "This result is based on past fact-checks only."
            )
        return "Search stopped: this claim reached its search limit."
    if status == "cache_only_miss":
        return (
            "Search didn't run: this claim was not in the search cache. "
            "This result is based on past fact-checks only."
        )
    if quiet:
        return (
            "Search didn't run: the search failed. "
            "This result is based on past fact-checks only."
        )
    return "Search stopped: the search failed."


def search_banner_html(model: dict) -> str:
    text = search_banner_text(model)
    if not text:
        return ""
    return f"<div class='fl-search-banner'>{esc(text)}</div>"


def _states_a_correction(found: str) -> bool:
    """what_we_found names what the sources said, not the abstain template."""
    text = " ".join(str(found or "").split())
    if not text:
        return False
    folded = text.casefold()
    if _UNREAD.search(folded):
        return False
    if "not confirmed" in folded and "could not find two independent" in folded:
        return False
    return True


def correction_text(model: dict) -> str:
    """The shareable line the card shows.

    An Unverifiable correction that says 'not confirmed' contradicts a
    what_we_found that already states what the sources said. The card
    quotes that finding instead of denying it.
    """
    model = model or {}
    message = " ".join(str(model.get("correction_message") or "").split())
    found = " ".join(str(model.get("what_we_found") or "").split())
    verdict = str(model.get("verdict") or "").strip()
    if (
        verdict == "Unverifiable"
        and "not confirmed" in message.casefold()
        and _states_a_correction(found)
    ):
        return (
            "Not enough credible sources to decide. "
            f"What the sources we found say: {found}"
        )
    return message


def correction_html(model: dict) -> str:
    text = correction_text(model)
    if not text:
        return ""
    return (
        "<section class='fl-correction'>"
        f"{eyebrow('Correction')}"
        f"<p>{esc(text)}</p>"
        "</section>"
    )


def load_fixture(name: str) -> dict:
    """One recorded card. The name is a token, not a path.

    Fixture mode renders this file and does not call the pipeline. A
    name with a slash would walk out of app/fixtures, so anything that
    is not a plain token is refused before the filesystem is touched.
    """
    token = "" if name is None else str(name).strip()
    if not re.fullmatch(r"[a-z0-9_]+", token):
        raise FileNotFoundError(token or "fixture")
    path = _FIXTURE_DIR / f"{token}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(token)
    return payload


def _pick(result: dict, state: dict, key: str, default=None):
    """result wins when the card has the key. Otherwise the state. Else default.

    .get() cannot tell a stored null from a field the engine has not
    added yet. A conflicting run stores a null verdict on purpose, so
    a missing key and a present null take different paths. The sentinel
    is how both stay .get() calls.
    """
    for src in (result, state):
        if not isinstance(src, dict):
            continue
        value = src.get(key, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _norm_row(row: dict) -> dict:
    """One evidence row in the shape the cards draw.

    The live card uses site/link. The state uses netloc/url/group. Both
    are accepted so a fixture and a finished run look the same.
    """
    url = row.get("url") or row.get("link") or ""
    netloc = row.get("netloc") or row.get("site") or ""
    if not netloc and url:
        try:
            netloc = urlsplit(str(url)).netloc
        except ValueError:
            netloc = ""
    group = row.get("group_id") or row.get("group") or netloc or "ungrouped"
    return {
        "id": str(row.get("id") or ""),
        "url": "" if url is None else str(url),
        "netloc": "" if netloc is None else str(netloc),
        "tier": row.get("tier"),
        "stance": str(row.get("stance") or "").strip().lower(),
        "quote": str(row.get("quote") or row.get("display_quote") or ""),
        "passage": str(row.get("passage") or ""),
        "group_id": str(group),
        "date": str(row.get("date") or row.get("published_date") or ""),
    }


def _merge_evidence(state: dict, result: dict) -> list[dict]:
    """Card rows, with passage and group filled from the state when the card omitted them.

    Output drops the passage on purpose. The highlight still needs the
    passage, and the state is where investigate left it. Matching on id
    is what keeps a second round from pairing the wrong quote.
    """
    card_rows = result.get("evidence", []) if isinstance(result, dict) else []
    state_rows = state.get("evidence", []) if isinstance(state, dict) else []
    if not isinstance(card_rows, list):
        card_rows = []
    if not isinstance(state_rows, list):
        state_rows = []
    by_id = {}
    for row in state_rows:
        if isinstance(row, dict):
            by_id[str(row.get("id") or "")] = row
    source = card_rows or state_rows
    merged = []
    for row in source:
        if not isinstance(row, dict):
            continue
        prior = by_id.get(str(row.get("id") or ""), {})
        combined = dict(prior)
        for key, value in row.items():
            if value not in (None, ""):
                combined[key] = value
        if not combined.get("passage"):
            combined["passage"] = prior.get("passage") or ""
        merged.append(_norm_row(combined))
    return merged


def view_model(state: dict | None = None, result: dict | None = None) -> dict:
    """The flat card the HTML builders draw. Missing new fields stay at defaults."""
    state = state if isinstance(state, dict) else {}
    if result is None:
        inner = state.get("result", None)
        result = inner if isinstance(inner, dict) else {}
    elif not isinstance(result, dict):
        result = {}

    evidence = _merge_evidence(state, result)
    best = state.get("best_match", None)
    if not isinstance(best, dict):
        best = {}
    seen = _pick(result, state, "seen_before", None)
    if not isinstance(seen, dict):
        seen = {}
    score = seen.get("cosine", _MISSING)
    if score is _MISSING:
        score = seen.get("score", _MISSING)
    if score is _MISSING:
        score = best.get("dense_cos", None)
    match_type = _pick(result, state, "match_type", "") or seen.get("match_type", "") or ""
    route = state.get("route", "") or ""
    if not route:
        route = result.get("route", "") or ""

    lean = _pick(result, state, "model_lean", None)
    if not isinstance(lean, dict):
        lean = None
    core_ids = _pick(result, state, "core_detail_only_ids", []) or []
    if not isinstance(core_ids, (list, tuple)):
        core_ids = []

    flags = _pick(result, state, "warning_signs", None)
    if not flags:
        flags = _pick(result, state, "red_flags", []) or []
    if not isinstance(flags, (list, tuple)):
        flags = []

    stats = state.get("investigate_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    queries = _pick(result, state, "queries", None)
    if not isinstance(queries, list):
        queries = stats.get("queries", []) if isinstance(stats.get("queries", None), list) else []
    cost = result.get("cost", None)
    if not isinstance(cost, dict):
        cost = {}
    audit = state.get("audit", None)
    if not isinstance(audit, dict):
        audit = {}

    reason_code = _pick(result, state, "unverifiable_reason", "") or ""
    verdict = _pick(result, state, "verdict", None)
    status = _pick(result, state, "search_status", "") or ""
    explanation = _pick(result, state, "explanation", "") or ""
    found = _pick(result, state, "what_we_found", "") or ""
    # Scrub before the dict is built. Both the static card and the page
    # read this model, so the unread sentence has to be gone here.
    read = sources_were_read({"evidence": evidence, "search_status": status})
    explanation = _drop_unread(explanation, read=read)
    found = _drop_unread(found, read=read)
    caveats = _pick(result, state, "caveats", []) or []
    if isinstance(caveats, str):
        caveats = [caveats]
    if not isinstance(caveats, (list, tuple)):
        caveats = []
    return {
        "claim": _pick(result, state, "claim", "") or "",
        "verdict": verdict,
        "confidence": _pick(result, state, "confidence", None),
        "priority": _pick(result, state, "priority", "") or "",
        "explanation": explanation,
        "evidence": evidence,
        "red_flags": [str(flag) for flag in flags if str(flag or "").strip()],
        "true_version": _pick(result, state, "true_version", "") or "",
        "correction_message": _pick(result, state, "correction_message", "") or "",
        "what_we_found": found,
        "caveats": [str(item).strip() for item in caveats if str(item or "").strip()],
        "search_status": "" if status is None else str(status),
        "searches_live": _pick(result, state, "searches_live", 0) or 0,
        "searches_cached": _pick(result, state, "searches_cached", 0) or 0,
        "provisional": bool(_pick(result, state, "provisional", False)),
        "review_status": _pick(result, state, "review_status", "") or "",
        "review_reason": _pick(result, state, "review_reason", "") or _pick(result, state, "why", "") or "",
        "alert_pending_approval": bool(_pick(result, state, "alert_pending_approval", False)),
        "unverifiable_reason": "" if reason_code is None else str(reason_code),
        "unverifiable_plain": plain_unverifiable(reason_code),
        "model_lean": lean,
        "core_detail_only_ids": [str(item) for item in core_ids],
        "seen_site": seen.get("site") or best.get("site") or "",
        "seen_date": seen.get("date") or best.get("date") or "",
        "seen_link": seen.get("link") or seen.get("url") or best.get("url") or "",
        "seen_claim": seen.get("claim") or best.get("primary_claim") or "",
        "seen_score": score,
        "match_type": "" if match_type is None else str(match_type),
        "route": "" if route is None else str(route),
        "also": best.get("also_debunked_by") if isinstance(best.get("also_debunked_by"), list) else [],
        "queries": [str(item) for item in queries if str(item or "").strip()],
        "llm_calls": cost.get("llm_calls", audit.get("llm_calls", stats.get("llm_calls", None))),
        "searches": cost.get("searches", audit.get("search_calls", audit.get("searches", stats.get("searches", None)))),
        "latency_s": audit.get("latency_s", stats.get("seconds", None)),
        "gate": state.get("gate", "") or "",
        "decision": _pick(result, state, "decision", "") or "",
        "checkable": state.get("checkable", True),
    }


def wordmark_html(*, huge: bool = True) -> str:
    """FACT✦LENS. The star is its own span so the accent colour has one job."""
    parts = APP_NAME.split("✦")
    star = "<span class='fl-star'>✦</span>"
    inner = star.join(esc(part) for part in parts)
    kind = "huge" if huge else "small"
    return f"<div class='fl-wordmark {kind}'>{inner}</div>"


def utility_html() -> str:
    return (
        "<div class='fl-utility'><div class='fl-utility-inner'>"
        "English text, links, and screenshots · A cited verdict, or Unverifiable"
        "</div></div>"
    )


def tabs_html(current: str, pending: int, fixture: str = "") -> str:
    """Pill tabs. The review pill carries the queue count when there is one.

    Links keep the fixture parameter so a screenshot URL stays on the
    recorded card when the reader moves between pages. cite is dropped:
    a highlight belongs to the check page that set it.
    """
    pages = (
        ("check", "Check"),
        ("review", "Review queue"),
        ("history", "History"),
        ("how", "How it works"),
    )
    bits = ["<nav class='fl-tabs'>"]
    for key, label in pages:
        query = [("page", key)]
        if fixture:
            query.append(("fixture", fixture))
        href = "?" + "&amp;".join(f"{esc(k)}={esc(v)}" for k, v in query)
        on = " is-on" if key == current else ""
        badge = ""
        if key == "review" and pending > 0:
            badge = f"<span class='fl-badge'>{int(pending)}</span>"
        bits.append(
            f"<a class='fl-tab{on}' href='{href}'>{esc(label)}{badge}</a>"
        )
    bits.append("</nav>")
    return "".join(bits)


def hero_html() -> str:
    """The opening. The dotted line is the only place the archive size is repeated as prose."""
    memory = esc(STATS["memory_display"])
    checkers = esc(STATS["checkers"])
    steps = esc(STATS["steps"])
    dot = "<span class='fl-dot'>•</span>"
    return (
        "<section class='fl-hero'>"
        f"<p class='fl-kicker'>{dot} {memory} past fact-checks "
        f"{dot} {checkers} Indian fact-checkers "
        f"{dot} {steps}-step agent</p>"
        f"{wordmark_html(huge=True)}"
        "<p class='fl-lede'>Paste a forward. Get a cited verdict.</p>"
        "</section>"
    )


def marquee_html(domains=None) -> str:
    """The 11 domains, separated by a coral star. The track is doubled so the loop has no gap.

    The second copy is aria-hidden and is removed when the reader asks
    for less motion, so the list is not shown twice as a static wrap.
    """
    names = list(domains) if domains else list(DOMAINS)
    one = []
    for index, name in enumerate(names):
        if index:
            one.append("<span class='fl-star' aria-hidden='true'>✦</span>")
        one.append(f"<span class='fl-marquee-item'>{esc(name)}</span>")
    row = "".join(one)
    return (
        "<div class='fl-marquee'><div class='fl-marquee-track'>"
        f"<div class='fl-marquee-seq'>{row}</div>"
        f"<div class='fl-marquee-seq fl-marquee-copy' aria-hidden='true'>{row}</div>"
        "</div></div>"
    )


def quota_html(model_used: int, model_cap: int, search_used: int, search_cap: int) -> str:
    """The live counters. Caps come from config, not from STATS.

    The search chip turns amber at 90% of the cap and red when the cap
    is reached. The tooltip is the practical consequence: a new claim
    cannot open a search, and a cached claim still can.
    """
    try:
        used = int(search_used)
        cap = int(search_cap)
    except (TypeError, ValueError):
        used, cap = 0, 0
    klass = ""
    title = ""
    if cap > 0 and used >= cap:
        klass = " is-cap"
        title = "new claims can't search; cached claims still work"
    elif cap > 0 and used / cap >= 0.9:
        klass = " is-warn"
        title = "new claims can't search; cached claims still work"
    tip = f" title=\"{esc(title)}\"" if title else ""
    return (
        "<div class='fl-quota'>"
        f"<span class='fl-chip'>Model {int(model_used)}/{int(model_cap)} today</span>"
        f"<span class='fl-chip{klass}'{tip}>Search {used}/{cap} month</span>"
        "</div>"
    )


def _clock(seconds) -> str:
    if seconds is None:
        return ""
    try:
        number = float(seconds)
    except (TypeError, ValueError):
        return ""
    if number != number or number < 0:
        return ""
    # Under a twentieth of a second the one-decimal clock rounds to 0.0s,
    # which reads as a measurement of zero. A dash says the step finished
    # too fast to show.
    if number < 0.05:
        return "—"
    if number < 10:
        return f"{number:.1f}s"
    return f"{round(number)}s"


def _skip_reason(node: str, state: dict) -> str | None:
    """Why a node that did not run was right to stay dark.

    reuse_prior must not look like a failed search. A stored decision
    must not look like a review that is still waiting. The wording for
    the archive hit is the one the page is asked to show.
    """
    if state.get("checkable", True) is False and node in _SEARCH_NODES:
        return "not a checkable claim — no search"
    if state.get("route", "") == "reuse_prior" and node in {"investigate", "gate"}:
        return "past fact-check reused — no search needed"
    decision = state.get("decision", "") or ""
    if node == "review" and decision == "store":
        return "stored — a reviewer was not needed"
    if node == "review" and state.get("route", "") == "reuse_prior" and decision not in {"alert", "human"}:
        return "stored — a reviewer was not needed"
    return None


def _elapsed_maps(ran) -> tuple[dict, dict]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    timed: dict[str, bool] = {}
    for item in ran or []:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        name = str(item[0])
        counts[name] = counts.get(name, 0) + 1
        elapsed = item[1] if len(item) > 1 else None
        if elapsed is None:
            continue
        try:
            totals[name] = totals.get(name, 0.0) + float(elapsed)
            timed[name] = True
        except (TypeError, ValueError):
            continue
    return totals, counts


def stepper_html(ran=None, state=None, *, settled: bool = False, replay: bool = False) -> str:
    """The 16 boxes. A node that has not finished is waiting, running, or skipped.

    Elapsed seconds come from the caller. This function does not sleep
    and does not invent a duration. None means the step finished and
    nobody timed it, which is what the fallback path has to say.
    Two boxes that share a node show that node's clock on each of them:
    the graph ran once, and both boxes are that run.
    """
    state = state if isinstance(state, dict) else {}
    totals, counts = _elapsed_maps(ran)
    ran_nodes = set(counts)
    replay_class = " is-replay" if replay else ""
    bits = ["<div class='fl-stepper" + replay_class + "'>"]
    if replay:
        bits.append("<p class='fl-replay'>replay</p>")
    running_assigned = settled or replay
    glow_index = 0
    last_group = None
    for step in STEPS:
        node = step["node"]
        if step["group"] != last_group:
            last_group = step["group"]
            bits.append(f"<div class='fl-step-group'>{esc(step['group'])}</div>")
        reason = None if node in ran_nodes else _skip_reason(node, state)
        if node in ran_nodes:
            status = "is-done"
        elif reason and (settled or state.get("route") or state.get("checkable", True) is False or state.get("decision")):
            status = "is-skipped"
        elif not running_assigned:
            status = "is-running"
            running_assigned = True
        else:
            status = "is-wait"
            if settled and reason is None:
                reason = "not reached on this run"
                status = "is-skipped"
        style = ""
        if replay and status == "is-done":
            style = f" style='--i: {glow_index}'"
            glow_index += 1
        clock = _clock(totals.get(node)) if status == "is-done" else ""
        visits = counts.get(node, 0)
        if visits > 1 and clock:
            clock = f"{clock} ×{visits}"
        mark = "<span class='fl-dot-done'></span>" if status == "is-done" else ""
        bits.append(
            f"<div class='fl-step {status}'{style}>"
            f"<span class='fl-step-box'>{esc(step['box'])}</span>"
            f"<span class='fl-step-title'>{mark}{esc(step['title'])}</span>"
            f"<span class='fl-step-time'>{esc(clock)}</span>"
            + (f"<span class='fl-step-reason'>{esc(reason)}</span>" if reason else "")
            + "</div>"
        )
    bits.append("</div>")
    return "".join(bits)


def _verdict_kind(verdict, gate: str) -> tuple[str, str]:
    text = "" if verdict is None else str(verdict).strip()
    if not text:
        if str(gate or "").lower() == "conflicting":
            return "Withheld", "withheld"
        return "Undecided", "other"
    kind = text.casefold()
    if kind not in _VERDICT_COLOR:
        return text, "other"
    return text, kind


def _confidence_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    if number > 1:
        number = number / 100.0
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _ring(confidence, color: str) -> str:
    """SVG ring. The dash is evidence strength, not the model's own score."""
    radius = 36
    circ = 2 * 3.14159265 * radius
    number = _confidence_float(confidence)
    if number is None:
        dash = 0.0
        label = "—"
    else:
        dash = circ * number
        label = f"{number:.0%}"
    gap = circ - dash
    return (
        "<svg class='fl-ring' width='96' height='96' viewBox='0 0 96 96' role='img' "
        f"aria-label='{esc('evidence strength ' + label)}'>"
        "<circle cx='48' cy='48' r='36' fill='none' stroke='#1E2A45' stroke-width='8'/>"
        f"<circle cx='48' cy='48' r='36' fill='none' stroke='{esc(color)}' stroke-width='8' "
        f"stroke-linecap='round' stroke-dasharray='{dash:.2f} {gap:.2f}' "
        "transform='rotate(-90 48 48)'/>"
        f"<text x='48' y='52' text-anchor='middle' fill='#F2F0EA' "
        f"font-family='Barlow Condensed, Arial Narrow, sans-serif' font-size='18'>{esc(label)}</text>"
        "</svg>"
    )


def verdict_html(model: dict, *, compact: bool = False) -> str:
    """The verdict hero, the provisional band, and the claim. All of it escaped.

    compact is the queue. A full hero on every paused run would push the
    approve button off the screen, and the word still has to be the
    verdict colour so a draft is not a grey label.
    """
    model = model or {}
    label, kind = _verdict_kind(model.get("verdict", None), model.get("gate", ""))
    color = _VERDICT_COLOR.get(kind, _VERDICT_COLOR["other"])
    bits = []
    if model.get("provisional", False):
        priority = str(model.get("priority", "") or "").strip()
        parts = ["PROVISIONAL"]
        if priority:
            parts.append(f"{priority.upper()} PRIORITY")
        parts.append("AWAITING HUMAN VERIFICATION")
        reason = str(model.get("review_reason", "") or "").strip()
        reason_html = f"<p>{esc(reason)}</p>" if reason else ""
        bits.append(f"<div class='fl-band'>{esc(' · '.join(parts))}{reason_html}</div>")
    elif str(model.get("review_status", "") or "").strip():
        bits.append(
            "<p class='fl-plain-reason'>Review status: "
            f"{esc(model.get('review_status', ''))}</p>"
        )
    if model.get("alert_pending_approval", False):
        bits.append("<p class='fl-alert-held'>Alert held until a reviewer approves</p>")
    priority = str(model.get("priority", "") or "").strip()
    chip = ""
    if priority:
        chip = (
            f"<div class='fl-priority {esc(priority.casefold())}'>"
            f"{esc(priority)} priority</div>"
        )
    claim = str(model.get("claim", "") or "").strip()
    claim_html = f"<p class='fl-claim'>{esc(claim)}</p>" if claim else ""
    plain = str(model.get("unverifiable_plain", "") or "").strip()
    # The ring is evidence strength, and only for a verdict the evidence
    # rule actually chose. Unverifiable has no strength to draw; the
    # reason sentence stands where the ring would have been.
    ring = ""
    under = ""
    if kind == "unverifiable":
        if plain:
            under = f"<p class='fl-plain-reason'>{esc(plain)}</p>"
    elif not compact and kind in _RING_VERDICTS:
        ring = (
            "<div class='fl-strength'>"
            f"{_ring(model.get('confidence', None), color)}"
            "<div class='fl-strength-label'>evidence strength</div>"
            "</div>"
        )
    caveat_html = _caveat_html(model.get("caveats"))
    below = ""
    if kind != "unverifiable" and plain:
        below = f"<p class='fl-plain-reason'>{esc(plain)}</p>"
    size = "font-size: 2.6rem;" if compact else ""
    bits.append(
        "<div class='fl-verdict-row'>"
        f"<div><h2 class='fl-verdict-word' style='color: {esc(color)}; {size}'>{esc(label)}</h2>"
        f"{chip}{caveat_html}{under}</div>"
        f"{ring}"
        "</div>"
        f"{claim_html}{below}"
    )
    return "".join(bits)


def _caveat_html(caveats) -> str:
    """Amber, directly under the verdict word. Empty when the card has none."""
    if isinstance(caveats, str):
        caveats = [caveats]
    bits = []
    for item in caveats or []:
        text = str(item or "").strip()
        if text:
            bits.append(f"<p class='fl-caveat'>{esc(text)}</p>")
    return "".join(bits)


def evidence_lean_html(model: dict) -> str:
    """What the card's own evidence rows lean, with no model in it.

    Only an Unverifiable card. A decided verdict already is the lean.
    A group that argues both sides counts on neither, the same way the
    gate refuses to let it vote. The border is solid so it reads as
    stronger than the dashed model guess underneath.
    """
    model = model or {}
    if str(model.get("verdict") or "").strip() != "Unverifiable":
        return ""
    groups: dict[str, dict] = {}
    for row in model.get("evidence") or []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("group_id") or row.get("netloc") or "ungrouped")
        slot = groups.setdefault(key, {"stances": set(), "tiers": []})
        stance = str(row.get("stance") or "").strip().lower()
        if stance:
            slot["stances"].add(stance)
        try:
            tier = int(row.get("tier"))
        except (TypeError, ValueError):
            tier = None
        if tier is not None:
            slot["tiers"].append(tier)
    refute = 0
    support = 0
    tiers: list[int] = []
    for slot in groups.values():
        stances = slot["stances"]
        # Both sides in one group is not a vote. Counting it twice would
        # invent a lean the gate refused to take.
        if "refutes" in stances and "supports" in stances:
            continue
        if "refutes" in stances:
            refute += 1
            tiers.extend(slot["tiers"])
        elif "supports" in stances:
            support += 1
            tiers.extend(slot["tiers"])
    if refute == 0 and support == 0:
        return ""
    if refute > support:
        word = "FALSE"
    elif support > refute:
        word = "TRUE"
    else:
        word = "SPLIT"
    note = _tier_span(tiers)
    if refute == 1:
        refute_bit = "1 independent group refutes"
    else:
        refute_bit = f"{refute} independent groups refute"
    line = f"Sources lean {word}: {refute_bit}, {support} support ({note})"
    if word == "SPLIT":
        line = f"Sources are split: {refute_bit}, {support} support ({note})"
    return (
        "<aside class='fl-src-lean'>"
        f"<strong>{esc(line)}</strong>"
        "</aside>"
    )


def _tier_span(tiers: list[int]) -> str:
    if not tiers:
        return "tier unknown"
    lo, hi = min(tiers), max(tiers)
    if lo == hi:
        return f"all tier {lo}"
    return f"all tier {lo}-{hi}"


def lean_html(model: dict) -> str:
    """The model's guess, drawn so it cannot be read as the verdict.

    It sits in a dashed box, in the muted colour, with an explicit
    sentence that there are no sources behind it. The verdict above is
    the one the evidence rule produced.
    """
    lean = (model or {}).get("model_lean", None)
    if not isinstance(lean, dict):
        return ""
    label = str(lean.get("label", "") or "").strip()
    if not label:
        return ""
    number = _confidence_float(lean.get("confidence", None))
    score = "" if number is None else f" · {number:.0%}"
    return (
        "<aside class='fl-lean'>"
        "<strong>Model lean (not a verdict)</strong>"
        f"<p class='fl-lean-label'>{esc(label)}{esc(score)}</p>"
        "<p>Gemini's guess from memory. No sources.</p>"
        "</aside>"
    )


def _score_text(score) -> str:
    if score is None or score == "":
        return ""
    try:
        return f" · match score {float(score):.2f}"
    except (TypeError, ValueError):
        return ""


def seen_html(model: dict) -> str:
    """Exact is 'Seen before'. Similar is a collapsed related list. None is nothing.

    A score under 0.70 is match_type none. Showing it as 'Seen before'
    told a reader this claim had already been checked. The 0.68 youturn
    hit on the states claim was that mistake.
    """
    model = model or {}
    kind = str(model.get("match_type") or "").strip().lower()
    if kind == "exact":
        return _seen_block(model, collapsed=False)
    if kind == "similar":
        return _seen_block(model, collapsed=True)
    return ""


def _seen_block(model: dict, *, collapsed: bool) -> str:
    site = str(model.get("seen_site", "") or "").strip()
    link = str(model.get("seen_link", "") or "").strip()
    claim = str(model.get("seen_claim", "") or "").strip()
    if not (site or link or claim):
        return ""
    when = str(model.get("seen_date", "") or "").strip() or "date unknown"
    score_text = _score_text(model.get("seen_score", None))
    link_html = ""
    if safe_href(link):
        link_html = f"<div>{anchor(link, 'Read the fact-check')}</div>"
    claim_html = f"<p>{esc(claim)}</p>" if claim else ""
    extras = []
    for item in model.get("also") or []:
        if not isinstance(item, dict):
            continue
        other = str(item.get("site") or "").strip()
        other_url = item.get("url") or ""
        if other or other_url:
            extras.append(anchor(other_url, other or "Also debunked"))
    extra_html = ""
    if extras:
        extra_html = "<p class='fl-muted'>Also debunked by " + " · ".join(extras) + "</p>"
    body = (
        f"<div><strong>{esc(site or 'Fact-check')}</strong>"
        f" <span class='fl-muted'>{esc(when)}{esc(score_text)}</span></div>"
        f"{claim_html}{link_html}{extra_html}"
    )
    if collapsed:
        # Closed by default. The score is in the summary so a reader sees
        # how close the other claim was without opening it.
        summary = "Related past fact-checks" + score_text
        return (
            "<details class='fl-related'>"
            f"<summary>{esc(summary)}</summary>"
            f"{body}"
            "</details>"
        )
    return (
        "<section class='fl-seen' id='ev-C1'>"
        "<div class='fl-eyebrow'>— Seen before</div>"
        f"{body}"
        "</section>"
    )


def explanation_html(text, evidence_ids, cite_href=None) -> str:
    """The explanation, with [E2] turned into a chip that can open that card.

    A bracket that is not one of this card's evidence ids stays text.
    That keeps a model sentence from inventing a link. cite_href, when
    given, builds the in-page address; the chip text is still escaped.
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return ""
    known = {str(item) for item in (evidence_ids or []) if str(item or "").strip()}
    bits = ["<div class='fl-explain'>"]
    cursor = 0
    for match in _CITE.finditer(raw):
        bits.append(esc(raw[cursor:match.start()]))
        ident = match.group(1)
        if ident in known and cite_href is not None:
            href = str(cite_href(ident) or "")
            if href.startswith("?") or href.startswith("#"):
                bits.append(
                    f"<a class='fl-cite' href='{esc(href)}'>[{esc(ident)}]</a>"
                )
            else:
                bits.append(esc(match.group(0)))
        else:
            bits.append(esc(match.group(0)))
        cursor = match.end()
    bits.append(esc(raw[cursor:]))
    bits.append("</div>")
    return "".join(bits)


def _tier_badge(value) -> tuple[str, str]:
    try:
        tier = int(value)
    except (TypeError, ValueError):
        tier = 4
    if tier < 1 or tier > 4:
        tier = 4
    css = {1: "t1", 2: "t2", 3: "t3", 4: "t4"}[tier]
    return f"T{tier}", css


def _mark_quote(passage: str, quote: str) -> str:
    """Escape the passage, and wrap the quote in <mark> when it really occurs there.

    The mark tags are ours. The passage and the quote are escaped on
    each side of the match, so a <script> inside either one is text.
    """
    if not passage:
        return ""
    quote = (quote or "").strip()
    if not quote:
        return esc(passage)
    index = passage.find(quote)
    if index < 0:
        folded = passage.lower().find(quote.lower())
        if folded < 0:
            return esc(passage)
        index = folded
        quote = passage[index:index + len(quote)]
    end = index + len(quote)
    return f"{esc(passage[:index])}<mark>{esc(passage[index:end])}</mark>{esc(passage[end:])}"


def _row_visible(row: dict, stances, tiers) -> bool:
    if stances is not None:
        allowed = {str(item).strip().lower() for item in stances}
        if row.get("stance") not in allowed:
            return False
    if tiers is not None:
        label, _css = _tier_badge(row.get("tier"))
        allowed_tiers = set()
        for item in tiers:
            text = str(item).strip().lower().replace(" ", "")
            if text.startswith("tier"):
                text = "t" + text[-1:]
            allowed_tiers.add(text)
        if label.lower() not in allowed_tiers:
            return False
    return True


def evidence_html(
    rows,
    *,
    highlight: str = "",
    core_detail_only_ids=None,
    stances=None,
    tiers=None,
) -> str:
    """Evidence cards grouped by independence group.

    The gate counts groups, not pages. Two articles from one outlet are
    one vote, so they sit under one heading. A highlighted citation is
    kept even when a filter would hide it: the chip promised to open it.
    """
    core = {str(item) for item in (core_detail_only_ids or [])}
    hot = str(highlight or "")
    prepared = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        norm = row if "group_id" in row and "netloc" in row else _norm_row(row)
        if hot and norm.get("id") == hot:
            prepared.append(norm)
            continue
        if _row_visible(norm, stances, tiers):
            prepared.append(norm)
    if not prepared:
        return "<p class='fl-muted'>No evidence rows match these filters.</p>"
    groups: dict[str, list] = {}
    order = []
    for row in prepared:
        key = row.get("group_id") or "ungrouped"
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    bits = []
    for key in order:
        bits.append(f"<div class='fl-group'>Independent group · {esc(key)}</div>")
        for row in groups[key]:
            stance = row.get("stance") or "other"
            bar = stance if stance in {"supports", "refutes", "neutral"} else "other"
            tier_label, tier_css = _tier_badge(row.get("tier"))
            when = row.get("date") or "date unknown"
            host = row.get("netloc") or "unknown source"
            ident = row.get("id") or ""
            hot_class = " is-hot" if hot and ident == hot else ""
            detail = ""
            if ident and ident in core:
                detail = "<span class='fl-detail-tag'>confirms a detail, not the main claim</span>"
            passage = row.get("passage") or ""
            quote = row.get("quote") or ""
            body = passage or quote
            marked = _mark_quote(body, quote) if body else ""
            open_attr = " open" if hot_class else ""
            passage_block = ""
            if marked:
                passage_block = (
                    f"<details{open_attr}><summary>Passage</summary>"
                    f"<div class='fl-passage'>{marked}</div></details>"
                )
            anchor_id = "ev-" + re.sub(r"[^A-Za-z0-9_-]", "", ident or "row")
            bits.append(
                f"<article class='fl-ev{hot_class}' id='{esc(anchor_id)}'>"
                f"<div class='fl-bar {bar}'></div>"
                "<div class='fl-ev-body'>"
                "<div class='fl-ev-top'>"
                f"<strong>{esc(ident)}</strong> "
                f"{anchor(row.get('url'), host)}"
                f"<span class='fl-tier {tier_css}'>{esc(tier_label)}</span>"
                f"<span class='fl-muted'>{esc(stance or 'unstated')}</span>"
                f"<span class='fl-muted'>{esc(when)}</span>"
                f"{detail}"
                "</div>"
                f"{passage_block}"
                "</div></article>"
            )
    return "".join(bits)


def flags_html(flags) -> str:
    chips = []
    for flag in flags or []:
        text = str(flag or "").strip()
        if text:
            chips.append(f"<span class='fl-chip flag'>{esc(text)}</span>")
    if not chips:
        return "<p class='fl-muted'>No warning signs recorded. Warning signs don't decide whether a claim is true.</p>"
    return (
        "<div class='fl-flags'>" + "".join(chips) + "</div>"
        "<p class='fl-muted'>Warning signs don't decide whether a claim is true.</p>"
    )


def stance_tier_rows(rows) -> list[dict]:
    """Distinct independence groups per stance and tier, for the bar chart.

    A second page in a group that is already counted must not add a bar.
    That would show two votes where the gate saw one.
    """
    seen = set()
    counts: dict[tuple[str, str], int] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        norm = row if "group_id" in row else _norm_row(row)
        stance = norm.get("stance") or "unstated"
        tier, _css = _tier_badge(norm.get("tier"))
        key = (stance, tier, norm.get("group_id") or "")
        if key in seen:
            continue
        seen.add(key)
        bucket = (stance, tier)
        counts[bucket] = counts.get(bucket, 0) + 1
    return [
        {"stance": stance, "tier": tier, "groups": count}
        for (stance, tier), count in counts.items()
    ]


def card_html(model: dict, *, cite_href=None, highlight: str = "") -> str:
    """The static card: banner, verdict, leans, seen-before, evidence, correction.

    Charts stay in the page. Everything that is a string from the claim
    is in this HTML, including the correction the card is willing to show.
    """
    model = model or {}
    ids = [row.get("id") for row in model.get("evidence") or [] if isinstance(row, dict) and row.get("id")]
    # C1 is the reused fact-check. A similar or none hit is not that chip.
    if str(model.get("match_type") or "").lower() == "exact":
        ids.append("C1")
    return "".join((
        search_banner_html(model),
        verdict_html(model),
        evidence_lean_html(model),
        lean_html(model),
        seen_html(model),
        explanation_html(model.get("explanation", ""), ids, cite_href),
        evidence_html(
            model.get("evidence") or [],
            highlight=highlight,
            core_detail_only_ids=model.get("core_detail_only_ids") or [],
        ),
        flags_html(model.get("red_flags") or []),
        correction_html(model),
    ))


def how_it_works_html() -> str:
    """Sixteen boxes and the honest scores. The principle is the sentence the system is built on."""
    bits = [
        "<p class='fl-eyebrow'>— How it works</p>",
        "<p class='fl-principle'>The model reads and writes. Evidence and rules decide.</p>",
        "<div class='fl-flow'>",
    ]
    for index, (number, group, title, body) in enumerate(BOXES):
        if index:
            bits.append("<div class='fl-arrow' aria-hidden='true'>↓</div>")
        bits.append(
            "<article class='fl-flow-card'>"
            f"<div class='fl-flow-kicker'>{esc(group)} · box {esc(number)}</div>"
            f"<h3 class='fl-flow-title'>{esc(title)}</h3>"
            f"<p class='fl-muted'>{esc(body)}</p>"
            "</article>"
        )
    bits.append("</div>")
    bits.append("<p class='fl-eyebrow'>— What the tests actually showed</p>")
    recycled = f"{STATS['recycled_hit']}/{STATS['recycled_n']}"
    bits.append(
        "<article class='fl-honest'>"
        "<h3 class='fl-flow-title'>Recycled rumours</h3>"
        f"<p>{esc(recycled)} caught when the fact-check was already in memory. "
        "Median about 3 seconds, and no web search. "
        "That is the easy case: the debunk is something the archive already holds.</p>"
        "</article>"
        "<article class='fl-honest'>"
        "<h3 class='fl-flow-title'>New rumours, fact-check sites blocked</h3>"
        f"<p>On {esc(STATS['sample_n'])} claims, with fact-check sites blocked and the source article hidden, "
        f"the system was right on {esc(STATS['blocked_system'])}. "
        f"Gemini alone, with no sources in front of it, was right on {esc(STATS['blocked_gemini'])}. "
        f"McNemar p={esc(STATS['mcnemar_p'])}. That difference is not significant.</p>"
        "</article>"
    )
    return "".join(bits)


def eyebrow(text: str) -> str:
    return f"<p class='fl-eyebrow'>— {esc(text)}</p>"


def queue_summary_html(row: dict, evidence_count) -> str:
    """One paused run. The verdict is provisional unless the row says it is not.

    A queue row is a pause. Showing the bare word False here is how a
    draft gets read as the stored verdict, so a missing provisional
    flag still draws the band.
    """
    row = row if isinstance(row, dict) else {}
    verdict = row.get("provisional_verdict", row.get("verdict", None))
    model = {
        "verdict": verdict,
        "claim": row.get("claim", "") or "",
        "priority": row.get("priority", "") or "",
        "confidence": row.get("confidence", None),
        "provisional": bool(row.get("provisional", True)),
        "review_reason": row.get("review_reason", row.get("reason", "")) or "",
        "review_status": row.get("review_status", "") or "",
        "alert_pending_approval": bool(row.get("alert_pending_approval", False)),
        "gate": row.get("gate", "") or "",
        "unverifiable_plain": plain_unverifiable(row.get("unverifiable_reason", "")),
    }
    count = "—"
    if evidence_count is not None:
        try:
            count = str(int(evidence_count))
        except (TypeError, ValueError):
            count = "—"
    return (
        "<div class='fl-panel'>"
        + verdict_html(model, compact=True)
        + f"<p class='fl-muted'>Evidence rows: {esc(count)}</p>"
        + "</div>"
    )
