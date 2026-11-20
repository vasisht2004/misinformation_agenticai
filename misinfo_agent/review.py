"""Box 13: pause for a person when the rules will not decide alone.

Owner: Vash. LangGraph interrupt() pauses the run before a verdict is
shown as final. The resume value is a Command(resume=...) payload:
approve, edit, reject, or more_evidence. This box does not search and
does not call a model. It only records what the person decided.

A failed node is routed here on purpose. A silent store after an
exception would look like a verdict the evidence never earned.

The checkpointer can resume a thread id. It cannot list the threads
that are waiting. pending_reviews, in the same SQLite file as the
finished runs, is that list. The row is written before interrupt()
parks the process, and removed once a person has answered. A finished
decision must not stay in the queue.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.types import interrupt

import misinfo_agent.config as config

# The queue columns. provisional_verdict is the label the rules wrote,
# which a person has not accepted yet. The raw forward is not a column:
# the claim is what the reviewer judges, and the submission itself stays
# out of this file the same way the audit row keeps only a hash.
_PENDING_COLUMNS = (
    "thread_id",
    "created_at",
    "claim",
    "provisional_verdict",
    "confidence",
    "priority",
    "reason",
)

_PENDING_ORDER = """
ORDER BY CASE priority
    WHEN 'High' THEN 0
    WHEN 'Medium' THEN 1
    WHEN 'Low' THEN 2
    ELSE 3
END,
created_at DESC,
rowid DESC
"""


def review(state: dict, *, db_path: Path | str | None = None, thread_id: str | None = None) -> dict:
    """Pause, then apply the reviewer's action.

    approve keeps the verdict. edit replaces the verdict and the
    correction and marks the run human_edited. reject clears the
    verdict. more_evidence does not change the verdict; the graph
    sends the claim back to investigate while a round is still allowed.
    """
    summary = _summary(state)
    thread = _thread(state, thread_id)
    # Written before the pause, and committed, because interrupt()
    # raises and does not return until a person resumes. On that resume
    # this function starts over, so the row is written again and then
    # removed below, once the answer is in hand. The queue shows the
    # run only while it is parked.
    _write_pending(db_path, thread, summary)
    # interrupt() returns whatever the caller passed to Command(resume=).
    # Until that resume arrives, the process is parked on the checkpoint.
    answer = interrupt(summary)
    _delete_pending(db_path, thread)
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
            "quote": " ".join(str(row.get("quote") or row.get("display_quote") or "").split()),
            "link": row.get("url") or "",
            # Which part of the claim this row is about. The page groups
            # for and against under that part. A row from before atomic
            # ids existed is the core claim.
            "atomic_id": str(row.get("atomic_id") or "A1"),
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
        # The edit box starts from this draft. Sending it back unchanged
        # keeps the correction the evidence rule wrote. An empty box
        # would wipe it, because an edit replaces the correction when
        # the key is present.
        "correction_message": state.get("correction_message") or "",
        "true_version": state.get("true_version") or "",
        "what_we_found": state.get("what_we_found") or "",
        "warning_signs": list(state.get("red_flags") or []),
        # Output has not run, so there is no result card yet. The paused
        # page still has a details section, and these are that section.
        "queries": _queries(state),
        "cost": _cost(state),
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


def list_pending(db_path: Path | str | None = None) -> list[dict]:
    """Paused runs, High priority first.

    Within one priority, the newest pause is first: the run just sent
    here is the one a person is about to open. Opening the queue does
    not create the database. A check that never paused has nothing to list.
    """
    path = _audit_path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_reviews'"
        ).fetchone()
        if not found:
            return []
        conn.row_factory = sqlite3.Row
        queried = conn.execute(f"SELECT {', '.join(_PENDING_COLUMNS)} FROM pending_reviews {_PENDING_ORDER}")
        return [dict(row) for row in queried.fetchall()]


def _thread(state: dict, thread_id: str | None) -> str:
    """The checkpointer's thread id.

    run_pipeline stores that id on run_marks before the first node.
    The queue, the audit row, and the resume call have to name the
    same run. An empty id is refused by the writer: a blank primary
    key would be shared by every run that forgot one, and a delete
    would not know which pause it was clearing.
    """
    if thread_id and str(thread_id).strip():
        return str(thread_id).strip()
    marks = state.get("run_marks")
    if isinstance(marks, dict) and marks.get("run_id"):
        return str(marks["run_id"]).strip()
    return ""


def _audit_path(db_path: Path | str | None) -> Path:
    return Path(db_path) if db_path else config.DATA_DIR / "audit.db"


def _write_pending(db_path: Path | str | None, thread: str, summary: dict) -> None:
    if not thread:
        return
    path = _audit_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict = summary.get("verdict")
    if verdict is None:
        stored_verdict = None
    else:
        stored_verdict = str(verdict).strip() or None
    confidence = _confidence(summary.get("confidence"))
    priority = str(summary.get("priority") or "").strip()
    reason = " ".join(str(summary.get("why") or "").split())[:500]
    claim = " ".join(str(summary.get("claim") or "").split())[:2000]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(path) as conn:
        _ensure_pending(conn)
        # created_at stays on conflict. Resume re-enters this function
        # before interrupt() returns, and that second write must not
        # look like a new pause that just arrived.
        conn.execute(
            """
            INSERT INTO pending_reviews (
                thread_id, created_at, claim, provisional_verdict, confidence, priority, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                claim=excluded.claim,
                provisional_verdict=excluded.provisional_verdict,
                confidence=excluded.confidence,
                priority=excluded.priority,
                reason=excluded.reason
            """,
            (thread, created, claim, stored_verdict, confidence, priority, reason),
        )


def _delete_pending(db_path: Path | str | None, thread: str) -> None:
    if not thread:
        return
    path = _audit_path(db_path)
    if not path.exists():
        return
    with sqlite3.connect(path) as conn:
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_reviews'"
        ).fetchone()
        if not found:
            return
        conn.execute("DELETE FROM pending_reviews WHERE thread_id = ?", (thread,))


def _ensure_pending(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_reviews (
            thread_id TEXT PRIMARY KEY,
            created_at TEXT,
            claim TEXT,
            provisional_verdict TEXT,
            confidence REAL,
            priority TEXT,
            reason TEXT
        )
        """
    )


def _confidence(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


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
    """Calls and searches since this run started. Not a model call.

    The same slice the audit row uses. Output has not built the card
    yet, and the paused page still shows a cost.
    """
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
