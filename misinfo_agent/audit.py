"""Box 16: one SQLite row per finished run.

Owner: Vash. The row is how a wrong verdict is traced back to the
evidence, the model, and the prompt version that produced it. The raw
submission is not stored. A hash is enough to find the same input
again, and it does not keep a person's message in the audit file.

Approved and edited verdicts are appended to
data/memory_additions.parquet for the next memory rebuild. The live
index is not modified. Rebuilding is a separate step; writing into the
index from a review would change what the next claim can match before
anyone has checked the addition.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import misinfo_agent.config as config
from misinfo_agent.llm import get_usage
from misinfo_agent.search import get_search_usage

_COLUMNS = (
    "run_id",
    "created_at",
    "input_type",
    "input_sha256",
    "claim",
    "domain",
    "verdict",
    "confidence",
    "decision",
    "human_action",
    "evidence_ids",
    "llm_calls",
    "searches",
    "latency_s",
    "models",
    "prompt_versions",
    "errors",
)


def audit(state: dict, *, db_path: Path | str | None = None, additions_path: Path | str | None = None) -> dict:
    """Write one runs row and, when a person accepted a verdict, queue it.

    Returns the record that was stored, under the audit key. The live
    memory index is not opened.
    """
    db = Path(db_path) if db_path else config.DATA_DIR / "audit.db"
    additions = Path(additions_path) if additions_path else config.DATA_DIR / "memory_additions.parquet"
    record = _record(state)
    _insert(db, record)
    if _accepted(state):
        # Queued for the next rebuild. The index box 5 searches is a
        # different file, and this function does not write it.
        _append_addition(additions, state)
    return {"audit": record}


def _record(state: dict) -> dict:
    marks = state.get("run_marks") if isinstance(state.get("run_marks"), dict) else {}
    usage = get_usage()
    searches = get_search_usage()
    start_llm = int(marks.get("llm_calls") or 0)
    start_search = int(marks.get("searches") or 0)
    started = marks.get("started")
    # started is a perf_counter stamp, not a duration. The duration is
    # now minus that stamp. A missing stamp is a run that did not go
    # through run_pipeline; the row still exists, with no latency.
    latency = None
    if isinstance(started, (int, float)):
        import time
        latency = round(time.perf_counter() - float(started), 2)
    slice_usage = usage[start_llm:]
    models = sorted({str(row.get("model_used") or row.get("model") or "") for row in slice_usage if row.get("model") or row.get("model_used")})
    versions = sorted({str(row.get("prompt_version") or "") for row in slice_usage if row.get("prompt_version")})
    evidence_ids = [
        str(row.get("id"))
        for row in (state.get("evidence") or [])
        if isinstance(row, dict) and row.get("id")
    ]
    verdict = state.get("verdict")
    confidence = state.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return {
        "run_id": str(marks.get("run_id") or state.get("thread_id") or ""),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_type": state.get("input_type") or "",
        "input_sha256": _input_hash(state),
        "claim": state.get("claim") or "",
        "domain": state.get("domain") or "",
        "verdict": "" if verdict is None else str(verdict),
        "confidence": confidence,
        "decision": state.get("decision") or "",
        "human_action": state.get("human_action") or "",
        "evidence_ids": evidence_ids,
        "llm_calls": max(0, len(usage) - start_llm),
        "searches": max(0, len(searches) - start_search),
        "latency_s": latency,
        "models": models,
        "prompt_versions": versions,
        "errors": list(state.get("errors") or []),
    }


def _input_hash(state: dict) -> str:
    """sha256 of the submission. The text itself is not returned."""
    raw = state.get("raw_text")
    if not isinstance(raw, str) or not raw:
        raw = state.get("raw_input") if isinstance(state.get("raw_input"), str) else ""
    if not raw and state.get("image_sha256"):
        return str(state.get("image_sha256"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _accepted(state: dict) -> bool:
    """A reviewer kept or rewrote a verdict. A rejection is not training data."""
    action = state.get("human_action")
    verdict = state.get("verdict")
    if action not in {"approve", "edit"}:
        return False
    return isinstance(verdict, str) and bool(verdict.strip())


def _append_addition(path: Path, state: dict) -> None:
    """One row on the reviewer queue. The live index is not modified."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "claim": state.get("claim") or "",
        "verdict": state.get("verdict") or "",
        "true_version": state.get("true_version") or "",
        "date": state.get("claim_date") or "",
        "source": "reviewer",
    }])
    if path.exists():
        # columns= so a future extra column cannot pull a body in.
        old = pd.read_parquet(path, columns=["claim", "verdict", "true_version", "date", "source"])
        row = pd.concat([old, row], ignore_index=True)
    row.to_parquet(path, index=False)


def _insert(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                input_type TEXT,
                input_sha256 TEXT,
                claim TEXT,
                domain TEXT,
                verdict TEXT,
                confidence REAL,
                decision TEXT,
                human_action TEXT,
                evidence_ids TEXT,
                llm_calls INTEGER,
                searches INTEGER,
                latency_s REAL,
                models TEXT,
                prompt_versions TEXT,
                errors TEXT
            )
            """
        )
        payload = dict(record)
        for key in ("evidence_ids", "models", "prompt_versions", "errors"):
            payload[key] = json.dumps(payload.get(key) or [], ensure_ascii=False)
        conn.execute(
            f"INSERT OR REPLACE INTO runs ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})",
            [payload.get(column) for column in _COLUMNS],
        )
