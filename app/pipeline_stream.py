"""Per-node clocks for one check.

run_pipeline calls graph.invoke, which returns only the end of the run.
The page wants each node as it finishes, with the seconds that node
actually took. stream_mode="updates" is that stream. The config, the
thread id and the recursion limit are the ones run_pipeline uses, so a
pause still resumes on the same checkpoint.

If the compiled graph cannot be streamed, we call run_pipeline and then
mark the nodes the final state shows were reached. Those clocks stay
empty except where the state itself recorded a duration. This module
does not sleep, and it does not invent a pace.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

# Append-only keys. A later investigate round must add rows, not replace
# the first round, or the live stepper would forget evidence it already showed.
_APPEND = {"evidence", "errors"}


def stream_pipeline(
    raw: str | None = None,
    *,
    input_type: str = "text",
    caption: str | None = None,
    thread_id: str | None = None,
    outcome: dict | None = None,
) -> Iterator[tuple[str, float | None]]:
    """Yield (node_name, elapsed_s) as each node finishes.

    outcome is filled in as the run goes: status, thread_id, result,
    review, audit, state, mode. The caller paints from the yields. The
    dict is the same object, so a step can read the state merged so far
    and grey out a search that the prior-match node already skipped.
    """
    sink = outcome if isinstance(outcome, dict) else {}
    sink.setdefault("state", {})
    sink.setdefault("ran", [])
    try:
        prepared = _prepare(raw, input_type=input_type, caption=caption, thread_id=thread_id)
    except Exception:
        yield from _fallback(raw, input_type=input_type, caption=caption, thread_id=thread_id, outcome=sink)
        return
    graph, cfg, initial, limit_ok = prepared
    if not limit_ok:
        yield from _fallback(raw, input_type=input_type, caption=caption, thread_id=thread_id, outcome=sink)
        return
    try:
        iterator = graph.stream(initial, cfg, stream_mode="updates")
    except TypeError:
        # An older graph without stream_mode must not be followed by a
        # second invoke. The fallback is the one run.
        yield from _fallback(raw, input_type=input_type, caption=caption, thread_id=thread_id, outcome=sink)
        return
    sink["mode"] = "stream"
    sink["thread_id"] = cfg["configurable"]["thread_id"]
    started = time.perf_counter()
    try:
        for chunk in iterator:
            now = time.perf_counter()
            elapsed = now - started
            started = now
            interrupted = _take_interrupt(chunk, sink)
            for name, update in _nodes(chunk):
                _merge(sink["state"], update)
                sink["ran"].append(name)
                yield name, elapsed
            if interrupted:
                sink["status"] = "interrupt"
    except _interrupt_types() as exc:
        # interrupt() raises out of the node. That is a pause, not a
        # failed check. The checkpoint holds the summary.
        sink["status"] = "interrupt"
        sink["interrupt_error"] = type(exc).__name__
    except Exception:
        # A quota error, or a node that escaped the graph's own guard,
        # still leaves a checkpoint. Read it, then let the page show
        # the failure. Do not start a second run to "finish" the clock.
        try:
            _finish(graph, cfg, sink)
        finally:
            raise
    _finish(graph, cfg, sink)


def _prepare(raw, *, input_type: str, caption: str | None, thread_id: str | None):
    """The same graph, config and starting state run_pipeline builds.

    The recursion limit is read off the graph module when it publishes
    one, so a change there is the change here. 40 is the value the
    wiring was written with, used only when the name is absent.
    limit_ok is False when we could not see a stream method at all.
    """
    import misinfo_agent.graph as graph_mod
    from misinfo_agent.graph import build_graph
    from misinfo_agent.llm import get_usage
    from misinfo_agent.search import get_search_usage

    if not hasattr(build_graph, "__call__"):
        return None, None, None, False
    graph = build_graph()
    if not hasattr(graph, "stream"):
        return None, None, None, False
    thread = thread_id or uuid.uuid4().hex
    limit = getattr(graph_mod, "_RECURSION_LIMIT", 40)
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": limit}
    marks = {
        "run_id": thread,
        "llm_calls": len(get_usage()),
        "searches": len(get_search_usage()),
        "started": time.perf_counter(),
    }
    initial = {
        "raw_input": raw or "",
        "input_type": input_type,
        "caption": caption or "",
        "run_marks": marks,
    }
    return graph, cfg, initial, True


def _fallback(raw, *, input_type: str, caption: str | None, thread_id: str | None, outcome: dict):
    """One invoke, then the nodes the final state proves ran.

    Per-step seconds are not in that return value. Investigate records
    its own seconds; everything else is None, which the stepper shows
    as finished without a clock. None is not zero: zero would look like
    a measurement.
    """
    from misinfo_agent.graph import build_graph, run_pipeline

    outcome["mode"] = "fallback"
    out = run_pipeline(raw, input_type=input_type, caption=caption, thread_id=thread_id)
    outcome["status"] = out.get("status")
    outcome["thread_id"] = out.get("thread_id") or thread_id or ""
    outcome["result"] = out.get("result") or {}
    outcome["review"] = out.get("review") or {}
    outcome["audit"] = out.get("audit") or {}
    state = {}
    try:
        graph = build_graph()
        snap = graph.get_state({"configurable": {"thread_id": outcome["thread_id"]}})
        values = getattr(snap, "values", None) or {}
        if isinstance(values, dict):
            state = dict(values)
    except Exception:
        state = {}
    outcome["state"] = state
    investigate_s = None
    stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    if stats.get("seconds") is not None:
        try:
            investigate_s = float(stats.get("seconds"))
        except (TypeError, ValueError):
            investigate_s = None
    for name in _nodes_present(state, outcome):
        elapsed = investigate_s if name == "investigate" else None
        outcome["ran"].append(name)
        yield name, elapsed


def _nodes_present(state: dict, outcome: dict) -> list[str]:
    """Nodes we can defend from the checkpoint. Order is the pipeline order."""
    found = []
    if state.get("text") or state.get("language") or state.get("input_type"):
        found.append("ingest")
    if state.get("claim") or state.get("atomic_claims") or state.get("checkable") is False:
        found.append("extract")
    if "matches" in state:
        found.append("memory")
    if state.get("route") or state.get("match_type"):
        found.append("prior_match")
    if state.get("evidence") or state.get("investigate_stats"):
        found.append("investigate")
    if state.get("gate"):
        found.append("gate")
    if state.get("explanation") or state.get("confidence") is not None or "verdict" in state:
        found.append("assess")
    if state.get("priority") or state.get("priority_score") is not None:
        found.append("priority")
    if state.get("decision"):
        found.append("decide")
    if state.get("human_action") or outcome.get("status") == "interrupt":
        found.append("review")
    if outcome.get("status") == "done" or state.get("result"):
        found.append("output")
    if state.get("audit") or outcome.get("audit"):
        found.append("audit")
    if found:
        return found
    # No checkpoint to read. Say only what the return value itself proves.
    if outcome.get("status") == "interrupt":
        return ["review"]
    if outcome.get("status") == "done":
        return ["output", "audit"]
    return []


def _nodes(chunk):
    if not isinstance(chunk, dict):
        return
    for name, update in chunk.items():
        if str(name).startswith("__"):
            continue
        yield str(name), update if isinstance(update, dict) else {}


def _take_interrupt(chunk, outcome: dict) -> bool:
    if not isinstance(chunk, dict) or "__interrupt__" not in chunk:
        return False
    payload = chunk.get("__interrupt__")
    review = {}
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        value = getattr(first, "value", first)
        if isinstance(value, dict):
            review = value
    elif isinstance(payload, dict):
        review = payload
    outcome["review"] = review
    return True


def _merge(state: dict, update: dict) -> None:
    if not isinstance(update, dict):
        return
    for key, value in update.items():
        if key in _APPEND and isinstance(value, list):
            state[key] = list(state.get(key) or []) + value
        else:
            state[key] = value


def _interrupt_types() -> tuple[type[BaseException], ...]:
    try:
        from langgraph.errors import GraphBubbleUp
    except Exception:
        return ()
    return (GraphBubbleUp,)


def _finish(graph, cfg, outcome: dict) -> None:
    """Read the checkpoint the stream just wrote. Same end state invoke would return."""
    try:
        snap = graph.get_state(cfg)
    except Exception:
        outcome.setdefault("status", "done")
        return
    values = getattr(snap, "values", None) or {}
    if isinstance(values, dict) and values:
        outcome["state"] = dict(values)
    review = {}
    for item in getattr(snap, "interrupts", ()) or ():
        value = getattr(item, "value", None)
        if isinstance(value, dict):
            review = value
            break
    if review or outcome.get("status") == "interrupt":
        outcome["status"] = "interrupt"
        if review:
            outcome["review"] = review
        return
    outcome["status"] = "done"
    if isinstance(values, dict):
        outcome["result"] = values.get("result") or outcome.get("result") or {}
        outcome["audit"] = values.get("audit") or outcome.get("audit") or {}
