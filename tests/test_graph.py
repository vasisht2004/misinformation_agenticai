"""The graph's routes, with a fake model and a fake search.

Memory is a stub. The test is about which box runs next, and loading
the embedding index would not change an edge. Temp files stand in for
the checkpoint, the audit database, and the reviewer parquet. The live
memory index is not opened.
"""

import json
import re
import sqlite3
from datetime import date

import pytest

import misinfo_agent.config as config
from misinfo_agent.graph import run_pipeline
from misinfo_agent.llm import set_fake
from misinfo_agent.search import set_fake_search


# The leading CLAIM token is what the fake encoder treats as the claim
# vector. The rest is a sentence with no relative-time word, so an old
# page is still allowed to refute it.
CLAIM = "CLAIM the minister visited the city museum"
SECRET = "ZZZGRAPHSECRET"


class _Memory:
    def __init__(self, hits):
        self.hits = hits

    def search(self, claim, **kwargs):
        return list(self.hits)


class _Encoder:
    """First token is the cosine, except the claim word CLAIM which is 1."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np

        rows = []
        for text in texts:
            token = str(text).split()[0]
            cos = 1.0 if token == "CLAIM" else float(token)
            rest = (1.0 - cos * cos) ** 0.5
            rows.append([cos, rest])
        return np.asarray(rows, dtype=np.float32)


def _paths(tmp_path):
    return {
        "checkpoint_path": tmp_path / "checkpoints.db",
        "audit_path": tmp_path / "audit.db",
        "additions_path": tmp_path / "memory_additions.parquet",
    }


def _page(url, marker, stance_word, salt=None):
    # Long enough that fetch keeps the Tavily body and does not open the URL.
    # The first token is the cosine the fake encoder reads. The salt is
    # different per page: two windows with the same words are one window,
    # and a refuted claim would then have only one group.
    filler = " ".join(f"{(salt or marker).lower()}word{i}" for i in range(30))
    body = f"0.90 {marker} {stance_word} the claim in full detail here. {filler}"
    return {
        "url": url,
        "title": marker,
        "content": body,
        "score": 0.9,
        "published_date": "2024-06-01",
        "raw_content": body,
    }


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PAGE_CACHE_DIR", tmp_path / "pages")
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _Encoder())
    yield
    set_fake(None)
    set_fake_search(None)


def _install(monkeypatch, *, hits=None, pages=None, extract=None, on_search=None):
    import misinfo_agent.memory as memory

    monkeypatch.setattr(memory, "get_memory", lambda: _Memory(hits or []))
    pages = list(pages or [])

    def fake_search(query, **kwargs):
        if on_search is not None:
            on_search(query)
        return pages

    def fake_llm(prompt, schema, images):
        name = schema.__name__
        if name == "ExtractResult":
            return extract or _checkable("Society & other")
        if name == "ContextVerdict":
            return {"same_claim": True, "differing_detail": None}
        if name == "StanceBatch":
            return {"judgements": _stance(prompt)}
        if name == "AtomicStanceBatch":
            return {"judgements": []}
        if name == "RepairBatch":
            return {"judgements": []}
        if name == "TopupBatch":
            return {"queries": []}
        if name == "_Draft":
            return {
                "atomic": [{
                    "atomic_id": "A1",
                    "explanation": "The page addresses the claim [E1].",
                    "refute_kind": "entirely_false",
                }],
                "true_version": "The visit did not happen as described [E1].",
                "correction_message": "The minister did not make that visit.",
            }
        raise AssertionError(name)

    set_fake_search(fake_search)
    set_fake(fake_llm)


def _checkable(domain, checkable=True):
    if not checkable:
        return {"checkable": False, "not_checkable_reason": "a greeting", "claim": "", "domain": ""}
    return {
        "checkable": True,
        "claim": CLAIM,
        "domain": domain,
        "atomic_claims": [{"id": "A1", "text": CLAIM, "core": True, "checkable": True}],
        "red_flags": [],
    }


def _stance(prompt):
    # The id is its own line. The first passage is not always its own
    # paragraph: the prompt file puts it on the line after a heading, so
    # splitting on a blank line drops P1 and the claim looks one-sourced.
    marks = list(re.finditer(r"(?m)^(P\d+)$", prompt))
    rows = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(prompt)
        text = prompt[mark.end():end]
        passage_id = mark.group(1)
        if "ALPHA" in text:
            rows.append({
                "passage_id": passage_id,
                "stance": "supports",
                "quote": "ALPHA supports the claim in full detail here",
                "reason": "The page agrees",
            })
        elif "BETA" in text:
            rows.append({
                "passage_id": passage_id,
                "stance": "refutes",
                "quote": "BETA refutes the claim in full detail here",
                "reason": "The page disagrees",
            })
    return rows


def _run(tmp_path, raw, **kwargs):
    return run_pipeline(raw, thread_id=kwargs.pop("thread_id", None), **_paths(tmp_path), **kwargs)


def _audit_rows(tmp_path):
    conn = sqlite3.connect(tmp_path / "audit.db")
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM runs")]
    conn.close()
    return rows


def _refute_pages():
    return [
        _page("https://www.bbc.com/news/one", "BETA", "refutes", salt="bbc"),
        _page("https://www.reuters.com/world/two", "BETA", "refutes", salt="reuters"),
    ]


def test_reuse_prior_skips_search(offline, monkeypatch, tmp_path):
    today = date.today().isoformat()
    hit = {
        "primary_claim": CLAIM,
        "dense_cos": 0.99,
        "date": today,
        "site": "boom",
        "url": "https://www.boomlive.in/example",
        "verdict_norm": "False",
        "true_version": "The minister did not visit the museum.",
        "article_id": "abc",
    }
    searched = []
    _install(monkeypatch, hits=[hit], on_search=searched.append)
    out = _run(tmp_path, f"{SECRET} {CLAIM}", thread_id="reuse")
    assert out["status"] == "done"
    assert out["result"]["verdict"] == "False"
    assert out["result"]["decision"] == "store"
    assert out["result"]["human_action"] == ""
    assert searched == []
    assert out["result"]["seen_before"]["link"] == "https://www.boomlive.in/example"
    rows = _audit_rows(tmp_path)
    assert len(rows) == 1
    blob = json.dumps(rows[0])
    assert SECRET not in blob
    assert rows[0]["claim"] == CLAIM


def test_refuted_claim_is_stored(offline, monkeypatch, tmp_path):
    _install(monkeypatch, pages=_refute_pages())
    out = _run(tmp_path, CLAIM, thread_id="store")
    assert out["status"] == "done"
    assert out["result"]["verdict"] == "False"
    assert out["result"]["decision"] == "store"
    assert out["result"]["human_action"] == ""
    assert out["result"]["run_id"] == "store"
    assert "llm_calls" in out["result"]["cost"]
    assert "searches" in out["result"]["cost"]
    assert isinstance(out["result"]["queries"], list)
    assert any(row.get("stance") == "refutes" for row in out["result"]["evidence"])
    assert any(part.get("core") for part in out["result"]["parts"])
    # A stored run never paused, so it must not sit in the review queue.
    from misinfo_agent.review import list_pending
    assert list_pending(tmp_path / "audit.db") == []
    assert len(_audit_rows(tmp_path)) == 1


def test_search_more_stops_at_the_round_cap(offline, monkeypatch, tmp_path):
    import misinfo_agent.investigate as investigate

    calls = {"n": 0}
    real = investigate.investigate

    def wrapped(state):
        calls["n"] += 1
        return real(state)

    monkeypatch.setattr(investigate, "investigate", wrapped)
    _install(monkeypatch, pages=[])
    out = _run(tmp_path, CLAIM, thread_id="thin")
    assert out["status"] == "done"
    assert out["result"]["verdict"] == "Unverifiable"
    assert out["result"]["decision"] == "store"
    # The first round sets rounds to 1. The gate may ask once more.
    # The third call is the one the cap refuses.
    assert calls["n"] == config.MAX_SEARCH_ROUNDS


def test_conflicting_review_can_be_approved(offline, monkeypatch, tmp_path):
    pages = [
        _page("https://www.rbi.org.in/notice", "ALPHA", "supports"),
        _page("https://pib.gov.in/press", "BETA", "refutes"),
    ]
    _install(monkeypatch, pages=pages)
    paused = _run(tmp_path, CLAIM, thread_id="conflict")
    assert paused["status"] == "interrupt"
    assert paused["review"]["why"]
    done = run_pipeline(
        resume={"action": "approve", "note": "looks right"},
        thread_id="conflict",
        **_paths(tmp_path),
    )
    assert done["status"] == "done"
    assert done["result"]["human_action"] == "approve"
    assert done["result"]["human_edited"] is False
    assert done["result"]["note"] == "looks right"
    # A withheld verdict is not a fact-check to learn from.
    assert not (tmp_path / "memory_additions.parquet").exists()
    assert SECRET not in json.dumps(_audit_rows(tmp_path)[0])


def test_alert_can_be_edited(offline, monkeypatch, tmp_path):
    _install(monkeypatch, pages=_refute_pages(), extract=_checkable("Health"))
    raw = f"BREAKING send to 10 groups {SECRET} {CLAIM}"
    paused = _run(tmp_path, raw, thread_id="alert")
    assert paused["status"] == "interrupt"
    assert paused["review"]["decision"] == "alert"
    done = run_pipeline(
        resume={"action": "edit", "verdict": "Misleading", "correction": "Only part of this is right."},
        thread_id="alert",
        **_paths(tmp_path),
    )
    assert done["result"]["verdict"] == "Misleading"
    assert done["result"]["correction_message"] == "Only part of this is right."
    assert done["result"]["human_edited"] is True
    assert done["result"]["human_action"] == "edit"
    frame = __import__("pandas").read_parquet(tmp_path / "memory_additions.parquet")
    assert list(frame["source"]) == ["reviewer"]
    assert list(frame["verdict"]) == ["Misleading"]
    blob = json.dumps(_audit_rows(tmp_path)[0])
    assert SECRET not in blob


def test_reject_clears_the_verdict(offline, monkeypatch, tmp_path):
    pages = [
        _page("https://www.rbi.org.in/notice", "ALPHA", "supports"),
        _page("https://pib.gov.in/press", "BETA", "refutes"),
    ]
    _install(monkeypatch, pages=pages)
    paused = _run(tmp_path, CLAIM, thread_id="reject")
    assert paused["status"] == "interrupt"
    done = run_pipeline(resume={"action": "reject"}, thread_id="reject", **_paths(tmp_path))
    assert done["result"]["verdict"] is None
    assert done["result"]["human_action"] == "reject"
    assert "rejected by reviewer" in (done["result"]["note"] or "")
    assert not (tmp_path / "memory_additions.parquet").exists()


def test_more_evidence_searches_once_more(offline, monkeypatch, tmp_path):
    import misinfo_agent.investigate as investigate

    calls = {"n": 0}
    real = investigate.investigate

    def wrapped(state):
        calls["n"] += 1
        return real(state)

    monkeypatch.setattr(investigate, "investigate", wrapped)
    _install(monkeypatch, pages=_refute_pages(), extract=_checkable("Health"))
    raw = f"BREAKING send to 10 groups {CLAIM}"
    paused = _run(tmp_path, raw, thread_id="more")
    assert paused["status"] == "interrupt"
    assert calls["n"] == 1
    again = run_pipeline(resume={"action": "more_evidence"}, thread_id="more", **_paths(tmp_path))
    assert again["status"] == "interrupt"
    assert calls["n"] == 2
    done = run_pipeline(resume={"action": "approve"}, thread_id="more", **_paths(tmp_path))
    assert done["status"] == "done"
    assert calls["n"] == 2
    assert done["result"]["human_action"] == "approve"


def test_not_checkable_is_other(offline, monkeypatch, tmp_path):
    searched = []
    _install(monkeypatch, extract=_checkable("Society & other", checkable=False), on_search=searched.append)
    out = _run(tmp_path, "hello there friend", thread_id="other")
    assert out["status"] == "done"
    assert out["result"]["verdict"] == "Other"
    assert out["result"]["decision"] == "store"
    assert searched == []


def test_a_node_exception_goes_to_review(offline, monkeypatch, tmp_path):
    import misinfo_agent.extract as extract

    def boom(state):
        raise RuntimeError("extract blew up")

    monkeypatch.setattr(extract, "extract", boom)
    _install(monkeypatch)
    paused = _run(tmp_path, f"{SECRET} hello", thread_id="boom")
    assert paused["status"] == "interrupt"
    assert "extract" in paused["review"]["why"]
    done = run_pipeline(resume={"action": "reject"}, thread_id="boom", **_paths(tmp_path))
    assert done["status"] == "done"
    rows = _audit_rows(tmp_path)
    assert len(rows) == 1
    assert "extract" in rows[0]["errors"]
    assert SECRET not in json.dumps(rows[0])
    assert "hello" not in rows[0]["input_sha256"]


def test_checkpoint_resumes_on_a_new_graph(offline, monkeypatch, tmp_path):
    pages = [
        _page("https://www.rbi.org.in/notice", "ALPHA", "supports"),
        _page("https://pib.gov.in/press", "BETA", "refutes"),
    ]
    _install(monkeypatch, pages=pages)
    paused = _run(tmp_path, CLAIM, thread_id="restart")
    assert paused["status"] == "interrupt"
    # run_pipeline builds a new compiled graph. The thread id and the
    # checkpoint file are what join it to the paused run.
    done = run_pipeline(resume={"action": "approve"}, thread_id="restart", **_paths(tmp_path))
    assert done["status"] == "done"
    assert done["result"]["human_action"] == "approve"
    assert len(_audit_rows(tmp_path)) == 1


def test_pending_review_is_written_on_interrupt_and_removed_on_resume(offline, monkeypatch, tmp_path):
    """The queue is the pause. Resume takes the row out before the card is stored."""
    pages = [
        _page("https://www.rbi.org.in/notice", "ALPHA", "supports"),
        _page("https://pib.gov.in/press", "BETA", "refutes"),
    ]
    _install(monkeypatch, pages=pages)
    paused = _run(tmp_path, f"{SECRET} {CLAIM}", thread_id="queue")
    assert paused["status"] == "interrupt"
    from misinfo_agent.graph import paused_review
    from misinfo_agent.review import list_pending

    rows = list_pending(tmp_path / "audit.db")
    assert len(rows) == 1
    row = rows[0]
    assert row["thread_id"] == "queue"
    assert row["claim"] == CLAIM
    assert row["reason"]
    assert row["priority"] in {"High", "Medium", "Low"}
    assert row["provisional_verdict"] is None
    assert isinstance(row["confidence"], float)
    assert SECRET not in json.dumps(row)
    payload = paused_review(
        "queue",
        checkpoint_path=tmp_path / "checkpoints.db",
        audit_path=tmp_path / "audit.db",
    )
    assert payload.get("claim") == CLAIM
    assert payload.get("evidence")
    done = run_pipeline(resume={"action": "approve"}, thread_id="queue", **_paths(tmp_path))
    assert done["status"] == "done"
    assert done["result"]["human_action"] == "approve"
    assert list_pending(tmp_path / "audit.db") == []
    assert _audit_rows(tmp_path)[0]["human_action"] == "approve"
