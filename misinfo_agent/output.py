"""Boxes 14–15: the result card.

Owner: Vash. This is the dict a person is shown. It copies the verdict
the rules already chose. It does not run them again, and it does not
call a model. Email and webhook delivery stay out of this box until a
real sink exists; the card is the product.
"""

from __future__ import annotations


def render(state: dict) -> dict:
    """Build state['result']. Do not re-decide the verdict."""
    return {"result": _card(state)}


def _card(state: dict) -> dict:
    parts = []
    for item in state.get("atomic_results") or []:
        if not isinstance(item, dict):
            continue
        parts.append({
            "atomic_id": item.get("atomic_id") or "",
            "explanation": item.get("explanation") or "",
        })
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
        }
    return {
        "verdict": state.get("verdict"),
        "confidence": state.get("confidence"),
        "explanation": state.get("explanation") or "",
        "explanations": parts,
        "true_version": state.get("true_version") or "",
        "correction_message": state.get("correction_message") or "",
        "what_we_found": state.get("what_we_found") or "",
        "citations": _citations(state),
        "seen_before": seen,
        "warning_signs": list(state.get("red_flags") or []),
        "priority": state.get("priority") or "",
        "decision": state.get("decision") or "",
        "human_action": state.get("human_action") or "",
        "human_edited": bool(state.get("human_edited")),
        "note": state.get("note") or "",
    }


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
