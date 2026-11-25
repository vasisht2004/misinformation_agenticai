"""Boxes 14–15: the result card.

Owner: Vash. This is the dict a person is shown. It copies the verdict
the rules already chose. It does not run them again, and it does not
call a model. Email and webhook delivery stay out of this box until a
real sink exists; the card is the product.
"""

from __future__ import annotations


def render(state: dict) -> dict:
    """Build state['result']. Do not re-decide the verdict.

    The search status is written here as well as on the card. The output
    node is the one that always runs, including when the search was not
    asked (a reused prior, or Other). Investigate has already copied the
    same rollup onto its own return when a search round happened.
    """
    from misinfo_agent.search import resolve_search_status

    report = resolve_search_status(state)
    return {
        "search_status": report["search_status"],
        "searches_live": report["searches_live"],
        "searches_cached": report["searches_cached"],
        "result": _card(state, report),
    }


def past_factcheck_heading(match_type: str) -> str:
    """The card heading for the archive hit box 6 kept.

    "Seen before" means this claim was checked. A similar match is a
    different claim that shares wording, so that sentence would overstate
    what the archive found. Exact matches, and a card with no hit, keep
    the original heading.
    """
    if match_type == "similar":
        return "Related past fact-check"
    return "Seen before"


def _card(state: dict, report: dict | None = None) -> dict:
    explanations, parts = _parts(state)
    best = state.get("best_match") if isinstance(state.get("best_match"), dict) else {}
    seen = None
    if best:
        # The archive link is the "seen before" line. An empty match
        # is not a fact-check, so the card leaves the field empty.
        seen = {
            "site": best.get("site") or "",
            "date": best.get("date") or "",
            "link": best.get("url") or "",
            "claim": best.get("primary_claim") or "",
            "cosine": best.get("dense_cos"),
            # Box 6's label. "exact" is this claim again. "similar" is a
            # near miss, and the page must not call that "seen before".
            "match_type": state.get("match_type") or "",
        }
    marks = state.get("run_marks") if isinstance(state.get("run_marks"), dict) else {}
    leans = _leans(state)
    explanation = state.get("explanation") or ""
    # A conflict has no single verdict. The evidence rows already carry
    # both stances; this sentence is what the hero shows when assess
    # left the explanation empty, so neither lean is hidden.
    if (
        not explanation
        and state.get("unverifiable_reason") == "conflicting_credible_sources"
        and leans["supports"]
        and leans["refutes"]
    ):
        explanation = (
            "Sources lean both ways. Supports: "
            + ", ".join(leans["supports"])
            + ". Refutes: "
            + ", ".join(leans["refutes"])
            + "."
        )
    return {
        "verdict": state.get("verdict"),
        "confidence": state.get("confidence"),
        "explanation": explanation,
        "leans": leans,
        "explanations": explanations,
        # One entry per part, with the core flag. The page badges the
        # main claim from this list. explanations stays the shorter
        # shape a caller that only wanted the sentences already used.
        "parts": parts,
        "true_version": state.get("true_version") or "",
        "correction_message": state.get("correction_message") or "",
        "what_we_found": state.get("what_we_found") or "",
        "citations": _citations(state),
        # Stance is what makes a row "for" or "against". Citations are
        # only the ids the text mentioned, so the columns cannot be
        # built from them. The page reads this card and does not open
        # the checkpoint to reconstruct the table.
        "evidence": _evidence(state),
        "seen_before": seen,
        "warning_signs": list(state.get("red_flags") or []),
        "priority": state.get("priority") or "",
        "decision": state.get("decision") or "",
        "human_action": state.get("human_action") or "",
        "human_edited": bool(state.get("human_edited")),
        "note": state.get("note") or "",
        # The UI reads the card, not the checkpoint. These are the review
        # and the lean. model_lean is never copied into verdict.
        "needs_human": bool(state.get("needs_human")),
        "provisional": bool(state.get("provisional")),
        "review_status": state.get("review_status") or "none",
        "review_reason": state.get("review_reason") or "",
        "alert_pending_approval": bool(state.get("alert_pending_approval")),
        "unverifiable_reason": state.get("unverifiable_reason") or "",
        "model_lean": state.get("model_lean"),
        # The page reads the card. A caveat that stays only on the state
        # never reaches the amber line under the verdict.
        "caveats": _caveats(state),
        "search_status": (report or {}).get("search_status") or "",
        "searches_live": (report or {}).get("searches_live") or 0,
        "searches_cached": (report or {}).get("searches_cached") or 0,
        "image_provenance": state.get("image_provenance"),
        "image_consistency": state.get("image_consistency"),
        # The UI draws the "online since" line from this. None on a text
        # claim: there is no photograph to date.
        "photo_lean": photo_lean(state),
        "image_memory_hit": state.get("image_memory_hit"),
        "core_detail_only_ids": list(state.get("core_detail_only_ids") or []),
        "queries": _queries(state),
        # Known here, one step before the audit row. The details section
        # is part of the card. It should still render if that write fails.
        "cost": _cost(state),
        "run_id": str(marks.get("run_id") or ""),
    }


def photo_lean(state: dict) -> dict | None:
    """How long this photograph has been online, from provenance only.

    first_seen_year is the earliest trusted full or partial match.
    n_sites is distinct registrable domains, so www and a subdomain are
    one site. n_dated_earlier counts full or partial matches whose trusted
    day is before the claimed event, which is the date rule's own test.
    A text claim has no photograph, so the card field stays empty.
    """
    if str(state.get("input_type") or "") != "image":
        return None
    from misinfo_agent.provenance import claim_event_date, is_dated_earlier
    from misinfo_agent.search import registrable_domain, site_host

    raw = state.get("image_provenance")
    provenance = raw if isinstance(raw, dict) else {}
    matches = [row for row in (provenance.get("matches") or []) if isinstance(row, dict)]
    seen = provenance.get("first_seen") if isinstance(provenance.get("first_seen"), dict) else {}
    event = claim_event_date(state)
    sites = set()
    earlier = 0
    for match in matches:
        host = str(match.get("host") or site_host(str(match.get("url") or "")) or "").strip()
        domain = registrable_domain(host) if host else ""
        if domain:
            sites.add(domain.casefold())
        if is_dated_earlier(match, event):
            earlier += 1
    year = _year(seen.get("date"))
    return {
        "first_seen_year": year,
        "n_sites": len(sites),
        "n_dated_earlier": earlier,
    }


def _year(value):
    text = str(value or "").strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:4])


def _leans(state: dict) -> dict:
    """Hosts on each side. The conflict card names both, not one."""
    support: list[str] = []
    refute: list[str] = []
    seen_support: set[str] = set()
    seen_refute: set[str] = set()
    for row in state.get("evidence") or []:
        if not isinstance(row, dict):
            continue
        host = str(row.get("netloc") or row.get("site") or "").strip()
        if not host:
            continue
        stance = str(row.get("stance") or "")
        if stance == "supports" and host not in seen_support:
            seen_support.add(host)
            support.append(host)
        elif stance == "refutes" and host not in seen_refute:
            seen_refute.add(host)
            refute.append(host)
    return {"supports": support, "refutes": refute}


def _parts(state: dict) -> tuple[list[dict], list[dict]]:
    claims = {}
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("id"):
            claims[str(item["id"])] = item
    explanations = []
    parts = []
    for item in state.get("atomic_results") or []:
        if not isinstance(item, dict):
            continue
        atomic_id = str(item.get("atomic_id") or "")
        explanation = item.get("explanation") or ""
        explanations.append({"atomic_id": atomic_id, "explanation": explanation})
        claim = claims.get(atomic_id) or {}
        parts.append({
            "atomic_id": atomic_id,
            "text": claim.get("text") or "",
            "core": bool(claim.get("core")),
            "explanation": explanation,
        })
    if parts:
        return explanations, parts
    # A reused prior, or a submission that was not checkable, may have
    # no per-part draft. The card still has the one explanation the
    # rules wrote, and that explanation is about the whole claim.
    text = state.get("claim") or ""
    for claim in claims.values():
        if claim.get("core") and claim.get("text"):
            text = claim.get("text") or text
            break
    explanation = state.get("explanation") or ""
    if not text and not explanation:
        return [], []
    if explanation:
        explanations.append({"atomic_id": "A1", "explanation": explanation})
    parts.append({
        "atomic_id": "A1",
        "text": text,
        "core": True,
        "explanation": explanation,
    })
    return explanations, parts


def _evidence(state: dict) -> list[dict]:
    """Display rows only. The passage text is not copied onto the card."""
    rows = []
    for row in state.get("evidence") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "id": str(row.get("id") or ""),
            "site": row.get("netloc") or row.get("site") or "",
            "tier": _tier(row.get("tier")),
            "stance": row.get("stance") or "",
            "date": row.get("published_date") or row.get("date") or "",
            "link": row.get("url") or "",
            "quote": " ".join(str(row.get("display_quote") or row.get("quote") or "").split()),
            "atomic_id": str(row.get("atomic_id") or "A1"),
        })
    return rows


def _caveats(state: dict) -> list[str]:
    """The strings the card already carries. Empty stays empty."""
    raw = state.get("caveats")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    found = []
    for item in raw:
        text = " ".join(str(item or "").split())
        if text:
            found.append(text)
    return found


def _tier(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _queries(state: dict) -> list[str]:
    stats = state.get("investigate_stats")
    if not isinstance(stats, dict):
        return []
    found = []
    for item in stats.get("queries") or []:
        text = " ".join(str(item).split())
        if text:
            found.append(text)
    return found


def _cost(state: dict) -> dict:
    """This run's calls and searches, not every earlier call in the process."""
    from misinfo_agent.llm import get_usage
    from misinfo_agent.search import get_search_usage

    marks = state.get("run_marks") if isinstance(state.get("run_marks"), dict) else {}
    return {
        "llm_calls": max(0, len(get_usage()) - _count(marks.get("llm_calls"))),
        "searches": max(0, len(get_search_usage()) - _count(marks.get("searches"))),
    }


def _count(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _citations(state: dict) -> list[dict]:
    """Each cited id, with the site and the link a reader can open.

    C1 is the archive fact-check, not a web row. The other ids are
    evidence rows. An id that is not in the table is dropped: the card
    does not invent a source for a citation the guard already lost.
    """
    by_id = {}
    for row in state.get("evidence") or []:
        if isinstance(row, dict) and row.get("id"):
            by_id[str(row["id"])] = row
    best = state.get("best_match") if isinstance(state.get("best_match"), dict) else {}
    cited = []
    seen = set()
    for raw in state.get("citations") or []:
        ident = str(raw)
        if ident in seen:
            continue
        seen.add(ident)
        if ident == "C1" and best:
            cited.append({
                "id": "C1",
                "site": best.get("site") or "",
                "tier": None,
                "date": best.get("date") or "",
                "link": best.get("url") or "",
            })
            continue
        row = by_id.get(ident)
        if row is None:
            continue
        cited.append({
            "id": ident,
            "site": row.get("netloc") or "",
            "tier": row.get("tier"),
            "date": row.get("published_date") or row.get("date") or "",
            "link": row.get("url") or "",
        })
    return cited
