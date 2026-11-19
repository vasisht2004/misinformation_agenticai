"""Box 13: pause for a person when the rules will not decide alone.

Owner: Vash. LangGraph interrupt() pauses the run before a verdict is
shown as final. The resume value is a Command(resume=...) payload:
approve, edit, reject, or more_evidence. This box does not search and
does not call a model. It only records what the person decided.

A failed node is routed here on purpose. A silent store after an
exception would look like a verdict the evidence never earned.
"""

from __future__ import annotations

from langgraph.types import interrupt


def review(state: dict) -> dict:
    """Pause, then apply the reviewer's action.

    approve keeps the verdict. edit replaces the verdict and the
    correction and marks the run human_edited. reject clears the
    verdict. more_evidence does not change the verdict; the graph
    sends the claim back to investigate while a round is still allowed.
    """
    summary = _summary(state)
    # interrupt() returns whatever the caller passed to Command(resume=).
    # Until that resume arrives, the process is parked on the checkpoint.
    answer = interrupt(summary)
    payload = answer if isinstance(answer, dict) else {}
    action = str(payload.get("action") or "").strip().lower()
    note = " ".join(str(payload.get("note") or "").split())
    # An unrecognised action is not an approval. Approving by default
    # would publish a verdict the person did not accept.
    if action not in {"approve", "edit", "reject", "more_evidence"}:
        action = "reject"
        note = note or "unrecognised review action"
    human = {
        "status": "done",
        "reviewer": "terminal",
        "action": action,
        "note": note,
    }
    out = {
        "human": human,
        "human_action": action,
        "note": note,
        "human_edited": False,
    }
    if action == "approve":
        return out
    if action == "edit":
        # Only the fields the reviewer sent are replaced. A missing
        # verdict keeps the one the evidence rule wrote.
        out["human_edited"] = True
        verdict = payload.get("verdict")
        if isinstance(verdict, str) and verdict.strip():
            out["verdict"] = verdict.strip()
        correction = payload.get("correction")
        if isinstance(correction, str):
            out["correction_message"] = " ".join(correction.split())
        return out
    if action == "more_evidence":
        return out
    # reject, and the unrecognised action above.
    out["verdict"] = None
    out["note"] = note or "rejected by reviewer"
    out["human"] = {**human, "note": out["note"]}
    reasons = list(state.get("decision_reasons") or [])
    reasons.append("rejected by reviewer")
    out["decision_reasons"] = reasons
    return out


def _summary(state: dict) -> dict:
    """What the reviewer sees. No raw submission: the claim is enough."""
    rows = []
    for row in state.get("evidence") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "id": row.get("id"),
            "site": row.get("netloc") or row.get("site") or "",
            "tier": row.get("tier"),
            "stance": row.get("stance"),
            "date": row.get("published_date") or row.get("date") or "",
            "quote": " ".join(str(row.get("quote") or "").split()),
            "link": row.get("url") or "",
        })
    errors = list(state.get("errors") or [])
    why = _why(state, errors)
    return {
        "claim": state.get("claim") or "",
        "verdict": state.get("verdict"),
        "confidence": state.get("confidence"),
        "evidence": rows,
        "reasons": list(state.get("gate_reasons") or []) + list(state.get("decision_reasons") or []),
        "why": why,
        "decision": state.get("decision") or "",
        "priority": state.get("priority") or "",
        "errors": errors,
    }


def _why(state: dict, errors: list) -> str:
    """One sentence for the queue. The evidence table is the rest."""
    if errors:
        last = errors[-1] if isinstance(errors[-1], dict) else {}
        node = last.get("node") or "a node"
        return f"{node} failed, so a person has to see the run before it ends."
    decision = state.get("decision") or ""
    if decision == "alert":
        return "The verdict is ready to send as an alert, and an alert waits for approval."
    if state.get("gate") == "conflicting":
        return "The sources disagree, so a person has to judge the claim."
    if state.get("gate") == "error":
        return "The stance check failed, so a person has to look before any action."
    if decision == "human":
        return "The rules asked for a person rather than storing the claim."
    return "The run was sent to review."
