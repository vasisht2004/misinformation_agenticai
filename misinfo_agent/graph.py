"""LangGraph wiring: one node per box, conditional edges at 6, 9, 12 and 13.

Owner: Vash. The nodes call the box functions and return their partial
updates. They do not reimplement a verdict. The checkpointer is SQLite
so a review pause (box 13) can resume after the process that asked the
question has exited. One thread_id is one run.

A node that raises, other than a quota error, is recorded on
state["errors"] and sent to review. A quota error is not caught: the
day is spent, and another node would fail the same way. A failed node
is never turned into a stored verdict.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, StateGraph
from langgraph.types import Command

import misinfo_agent.config as config
from misinfo_agent.llm import LLMQuotaError, get_usage
from misinfo_agent.search import get_search_usage
from misinfo_agent.state import ClaimState

# Kept so the SQLite connection is not closed while a graph built from
# this process still has a paused thread. A closed connection would
# make the next resume look like a brand-new run.
_SAVERS: dict[str, tuple[sqlite3.Connection, SqliteSaver]] = {}

_RECURSION_LIMIT = 40


def build_graph(
    checkpoint_path: Path | str | None = None,
    audit_path: Path | str | None = None,
    additions_path: Path | str | None = None,
):
    """Compile the StateGraph. The checkpointer file is data/checkpoints.db."""
    checkpoint = Path(checkpoint_path) if checkpoint_path else config.DATA_DIR / "checkpoints.db"
    audit_db = Path(audit_path) if audit_path else config.DATA_DIR / "audit.db"
    additions = Path(additions_path) if additions_path else config.DATA_DIR / "memory_additions.parquet"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saver = _saver(checkpoint)

    builder = StateGraph(ClaimState)
    builder.add_node("ingest", _guard("ingest", _ingest))
    builder.add_node("extract", _guard("extract", _extract))
    builder.add_node("memory", _guard("memory", _memory))
    builder.add_node("prior_match", _guard("prior_match", _prior))
    builder.add_node("investigate", _guard("investigate", _investigate))
    builder.add_node("gate", _guard("gate", _gate))
    builder.add_node("assess", _guard("assess", _assess))
    builder.add_node("priority", _guard("priority", _priority))
    builder.add_node("decide", _guard("decide", _decide))

    def _review_node(state: dict) -> dict:
        # The queue lives in the audit file. Tests pass a temp path into
        # this builder. review() must use that path: its own default is
        # data/audit.db, and a test would then write the real queue.
        from misinfo_agent.review import review

        return review(state, db_path=audit_db)

    builder.add_node("review", _guard("review", _review_node))
    builder.add_node("output", _guard("output", _output))
    builder.add_node("audit", _guard("audit", lambda state: _audit(state, audit_db, additions)))

    builder.set_entry_point("ingest")
    builder.add_conditional_edges("ingest", _after_ingest, ["extract", "review"])
    builder.add_conditional_edges("extract", _after_extract, ["assess", "memory", "review"])
    builder.add_conditional_edges("memory", _after_memory, ["prior_match", "review"])
    builder.add_conditional_edges("prior_match", _after_prior, ["assess", "investigate", "review"])
    builder.add_conditional_edges("investigate", _after_investigate, ["gate", "review"])
    # search_more is the gate authorising another round. The counter is
    # already the next round, so this edge does not also test rounds < MAX.
    # That test would skip the round the gate just allowed.
    builder.add_conditional_edges(
        "gate",
        _after_gate,
        ["investigate", "assess", "review"],
    )
    builder.add_conditional_edges("assess", _after_assess, ["priority", "review"])
    builder.add_conditional_edges("priority", _after_priority, ["decide", "review"])
    builder.add_conditional_edges(
        "decide",
        _after_decide,
        ["investigate", "review", "output"],
    )
    builder.add_conditional_edges("review", _after_review, ["investigate", "output"])
    builder.add_conditional_edges("output", _after_output, ["audit", "review"])
    # An audit failure still ends. Sending it back to review would loop
    # on the same exception. The error is already on the checkpoint.
    builder.add_conditional_edges("audit", _after_audit, [END, "output"])

    return builder.compile(checkpointer=saver)


def run_pipeline(
    raw: str | bytes | None = None,
    input_type: str = "text",
    thread_id: str | None = None,
    caption: str | None = None,
    *,
    resume: dict | None = None,
    checkpoint_path: Path | str | None = None,
    audit_path: Path | str | None = None,
    additions_path: Path | str | None = None,
    eval_claim_date: str | None = None,
    eval_exclude_ids: list[str] | None = None,
    eval_exclude_urls: list[str] | None = None,
    eval_sites: str | None = None,
) -> dict:
    """Run one claim, or resume a paused review on the same thread_id.

    A quota error propagates. An interrupt returns status "interrupt"
    and the review summary. A finished run returns status "done" and
    the result card. The raw submission is not in that return value.

    eval_claim_date, eval_exclude_ids and eval_exclude_urls are for the
    evaluation harness. A normal check leaves them empty. The date
    replaces the submission day after ingest. The ids are left out of
    the memory search. The urls are the source article: search results,
    fetched pages and evidence all drop that one page. They ride on
    run_marks, which is already part of the state, so the state
    contract does not grow a field for one script.
    """
    thread = thread_id or uuid.uuid4().hex
    graph = build_graph(checkpoint_path, audit_path, additions_path)
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": _RECURSION_LIMIT}
    if resume is None:
        if isinstance(raw, bytes):
            # The checkpointer would store the pixels on every step.
            # Ingest reads a path. A bytes submission is not a path.
            raise TypeError("image input must be a file path, not raw bytes")
        marks = {
            "run_id": thread,
            "llm_calls": len(get_usage()),
            "searches": len(get_search_usage()),
            "started": time.perf_counter(),
        }
        # Absent on a normal check. Present only when the eval asked.
        if eval_claim_date:
            marks["eval_claim_date"] = eval_claim_date
        if eval_exclude_ids:
            marks["eval_exclude_ids"] = [str(item) for item in eval_exclude_ids]
        if eval_exclude_urls:
            marks["eval_exclude_urls"] = [str(item) for item in eval_exclude_urls if str(item).strip()]
        if eval_sites:
            # Which eval file this thread belongs to. A later read of the
            # checkpoint has to tell a blocked run from an allowed one.
            marks["eval_sites"] = str(eval_sites)
        initial = {
            "raw_input": raw or "",
            "input_type": input_type,
            "caption": caption or "",
            "run_marks": marks,
        }
        out = graph.invoke(initial, cfg)
    else:
        out = graph.invoke(Command(resume=resume), cfg)
    if isinstance(out, dict) and out.get("__interrupt__"):
        pending = out["__interrupt__"][0]
        summary = pending.value if hasattr(pending, "value") else pending
        return {"status": "interrupt", "thread_id": thread, "review": summary}
    values = graph.get_state(cfg).values
    return {
        "status": "done",
        "thread_id": thread,
        "result": values.get("result") or {},
        "audit": values.get("audit") or {},
    }


def paused_review(
    thread_id: str,
    *,
    checkpoint_path: Path | str | None = None,
    audit_path: Path | str | None = None,
) -> dict:
    """The interrupt summary for one paused thread. Empty if it is not waiting.

    The queue table names who is waiting. It does not copy every quote.
    The evidence a reviewer reads is this summary, which the checkpointer
    kept when interrupt() parked the run. A missing thread is an empty
    dict, not an error: the queue row can outlive a checkpoint that was
    deleted by hand, and the page still has the claim and the reason.
    """
    graph = build_graph(checkpoint_path, audit_path)
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    for item in getattr(snap, "interrupts", ()) or ():
        value = getattr(item, "value", None)
        if isinstance(value, dict):
            return value
    return {}


def _saver(path: Path) -> SqliteSaver:
    key = str(path.resolve())
    found = _SAVERS.get(key)
    if found is not None:
        return found[1]
    conn = sqlite3.connect(key, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    _SAVERS[key] = (conn, saver)
    return saver


def _guard(name: str, fn):
    """Catch a node failure into state['errors']. Let a quota error out.

    The routers send a state whose latest error names this node to
    review. Swallowing the exception here is what stops LangGraph from
    aborting the thread before that route can run.
    """
    def node(state: dict) -> dict:
        try:
            update = fn(state)
        except (GraphBubbleUp, LLMQuotaError):
            # interrupt() raises GraphInterrupt, and that class is an
            # Exception. Catching it here would turn a review pause into
            # a recorded failure and the run would store a card nobody
            # had approved. A quota error is the other one that must
            # leave the node: the day is spent, and the next node would
            # fail the same way.
            raise
        except Exception as exc:
            # The message is clipped. A traceback can quote the prompt,
            # and the prompt quotes the submission. The node name and
            # the exception type are what the audit row needs.
            detail = f"{type(exc).__name__}: {exc}"
            return {"errors": [{"node": name, "error": detail[:300]}]}
        if not isinstance(update, dict):
            return {}
        return update
    return node


def _failed(state: dict, name: str) -> bool:
    errors = state.get("errors") or []
    if not errors or not isinstance(errors[-1], dict):
        return False
    return errors[-1].get("node") == name


def _ingest(state: dict) -> dict:
    from misinfo_agent.ingest import ingest
    update = ingest(
        state.get("raw_input") or "",
        input_type=state.get("input_type") or "text",
        caption=state.get("caption") or None,
    )
    override = _mark(state, "eval_claim_date")
    if isinstance(override, str) and override.strip():
        # Ingest dates the message as today. An eval claim was circulating
        # on the fact-check's date, and the memory filter and the page-date
        # guard both read claim_date. The first ten characters are the day.
        update["claim_date"] = override.strip()[:10]
    return update


def _mark(state: dict, key: str):
    """One value off run_marks, or None when this run did not set it."""
    marks = state.get("run_marks")
    if isinstance(marks, dict):
        return marks.get(key)
    return None


def _extract(state: dict) -> dict:
    from misinfo_agent.extract import extract
    return extract(state)


def _memory(state: dict) -> dict:
    from misinfo_agent.memory import get_memory
    claim = state.get("claim") or state.get("match_text") or ""
    domain = state.get("domain") or None
    exclude = _mark(state, "eval_exclude_ids")
    # date_before is strict, so the source article's own day is already
    # out. The harness decides which ids to add. The blocked eval passes
    # the article plus its near-duplicate copies. The allowed eval passes
    # only the article, so an earlier fact-check of the same rumour can
    # still be retrieved. A normal check leaves the list unset.
    hits = get_memory().search(
        claim,
        date_before=state.get("claim_date") or None,
        domains=[domain] if domain else None,
        exclude_ids=list(exclude) if exclude else None,
    )
    return {"matches": hits}


def _prior(state: dict) -> dict:
    from misinfo_agent.prior_match import decide_prior
    decision = decide_prior(
        state.get("claim") or "",
        list(state.get("matches") or []),
        now=state.get("claim_date") or None,
        entities=state.get("entities"),
        form=state.get("input_type"),
    )
    # The state key is a dict. None would erase a previous match on a
    # merge that treats None as a value, and the card would crash on it.
    if decision.get("best_match") is None:
        decision["best_match"] = {}
    return decision


def _investigate(state: dict) -> dict:
    """One search round. New evidence rows are appended by the reducer.

    investigate() returns only the rows it just labelled. Returning
    those, and not the rows already on the state, is what makes the
    reducer append instead of duplicating round 1. Passages are not a
    reducer, so this node concatenates them itself. rounds is set to 1
    after the first round so the gate's "another search" test sees a
    finished round. A later round leaves the number the gate wrote.
    """
    from misinfo_agent.investigate import investigate
    result = investigate(state)
    fresh = list(result.get("evidence") or [])
    passages = list(state.get("passages") or []) + list(result.get("passages") or [])
    stats = _merge_stats(state.get("investigate_stats"), result.get("investigate_stats") or {})
    out = {
        "evidence": fresh,
        "passages": passages,
        "investigate_stats": stats,
    }
    # The cap is a property of the claim, not of the last round. A later
    # round that searched cleanly must not wipe a cap the first round hit.
    cap = _prefer_search_cap(state.get("search_capped"), stats.get("search_capped"))
    if cap:
        out["search_capped"] = cap
    if int(state.get("rounds") or 0) < 1:
        out["rounds"] = 1
    return out


def _merge_stats(old, new: dict) -> dict:
    """Keep this round's report, and add the counts a later box still needs.

    Same reason as the demo runner: round 2's dict would otherwise
    forget round 1's unverified quotes, and confidence would look cleaner
    than the quotes were.
    """
    if not isinstance(old, dict) or not old:
        return dict(new)
    merged = dict(new)
    merged["quote_unverified"] = int(old.get("quote_unverified") or 0) + int(new.get("quote_unverified") or 0)
    merged["topup_used"] = bool(old.get("topup_used")) or bool(new.get("topup_used"))
    merged["stance_failed"] = bool(old.get("stance_failed")) or bool(new.get("stance_failed"))
    merged["repaired"] = int(old.get("repaired") or 0) + int(new.get("repaired") or 0)
    # The new round's dict would otherwise replace the first round's
    # mismatches, and the eval would under-count a claim that searched twice.
    merged["reason_label_mismatch"] = (
        int(old.get("reason_label_mismatch") or 0) + int(new.get("reason_label_mismatch") or 0)
    )
    merged["search_capped"] = _prefer_search_cap(old.get("search_capped"), new.get("search_capped"))
    return merged


def _prefer_search_cap(left, right) -> str | None:
    found = [item for item in (left, right) if item in {"per_claim", "monthly"}]
    if "monthly" in found:
        return "monthly"
    if "per_claim" in found:
        return "per_claim"
    return None


def _gate(state: dict) -> dict:
    from misinfo_agent.gate import gate
    return gate(state)


def _assess(state: dict) -> dict:
    from misinfo_agent.assess import assess
    return assess(state)


def _priority(state: dict) -> dict:
    from misinfo_agent.priority import priority
    return priority(state)


def _decide(state: dict) -> dict:
    from misinfo_agent.decide import decide
    return decide(state)


def _output(state: dict) -> dict:
    from misinfo_agent.output import render
    return render(state)


def _audit(state: dict, db_path: Path, additions_path: Path) -> dict:
    from misinfo_agent.audit import audit
    return audit(state, db_path=db_path, additions_path=additions_path)


def _after_ingest(state: dict) -> str:
    if _failed(state, "ingest"):
        return "review"
    return "extract"


def _after_extract(state: dict) -> str:
    if _failed(state, "extract"):
        return "review"
    # Nothing to look up. Assess writes Other and the rest of the
    # pipeline stores it. Memory would embed an empty claim.
    if state.get("checkable") is False:
        return "assess"
    return "memory"


def _after_memory(state: dict) -> str:
    if _failed(state, "memory"):
        return "review"
    return "prior_match"


def _after_prior(state: dict) -> str:
    if _failed(state, "prior_match"):
        return "review"
    # An exact fresh fact-check is the verdict. Searching would let a
    # web page outvote the archive row box 6 already accepted.
    if state.get("route") == "reuse_prior":
        return "assess"
    return "investigate"


def _after_investigate(state: dict) -> str:
    if _failed(state, "investigate"):
        return "review"
    return "gate"


def _after_gate(state: dict) -> str:
    if _failed(state, "gate"):
        return "review"
    gate = state.get("gate")
    if gate == "search_more":
        return "investigate"
    if gate in {"sufficient", "insufficient", "conflicting", "error", "skipped"}:
        return "assess"
    return "review"


def _after_assess(state: dict) -> str:
    if _failed(state, "assess"):
        return "review"
    return "priority"


def _after_priority(state: dict) -> str:
    if _failed(state, "priority"):
        return "review"
    return "decide"


def _after_decide(state: dict) -> str:
    if _failed(state, "decide"):
        return "review"
    decision = state.get("decision")
    rounds = int(state.get("rounds") or 0)
    # The decider increments rounds when it asks for another search.
    # Once that number has reached the cap, the authorised round is
    # already counted and a further investigate is not started. Review
    # sees it instead of a silent store.
    if decision == "search_more" and rounds < config.MAX_SEARCH_ROUNDS:
        return "investigate"
    if decision in {"alert", "human"}:
        return "review"
    if decision == "store":
        return "output"
    return "review"


def _after_review(state: dict) -> str:
    if _failed(state, "review"):
        return "output"
    rounds = int(state.get("rounds") or 0)
    if state.get("human_action") == "more_evidence" and rounds < config.MAX_SEARCH_ROUNDS:
        return "investigate"
    return "output"


def _after_output(state: dict) -> str:
    if _failed(state, "output"):
        return "review"
    return "audit"


def _after_audit(state: dict) -> str:
    return END
