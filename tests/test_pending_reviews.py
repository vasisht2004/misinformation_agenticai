"""The review queue and the history list, without a model or a search.

The graph test is the one that pauses a real run. These check the
order and the empty file, which do not need a checkpoint.
"""

from misinfo_agent.audit import audit, list_runs
from misinfo_agent.review import _delete_pending, _write_pending, list_pending


def test_missing_audit_file_lists_nothing(tmp_path):
    missing = tmp_path / "no-such.db"
    assert list_pending(missing) == []
    assert list_runs(missing) == []


def test_blank_thread_is_not_queued(tmp_path):
    # An empty primary key would be shared by every run that forgot its
    # id, and the next delete would clear the wrong pause.
    db = tmp_path / "audit.db"
    _write_pending(db, "", {"claim": "a claim", "why": "because", "priority": "High"})
    assert not db.exists()
    assert list_pending(db) == []


def test_high_priority_is_listed_first(tmp_path):
    db = tmp_path / "audit.db"
    _write_pending(
        db,
        "low",
        {"claim": "the quiet claim", "why": "a person should see it", "priority": "Low", "verdict": "False", "confidence": 0.4},
    )
    _write_pending(
        db,
        "high",
        {"claim": "the harmful claim", "why": "an alert waits for approval", "priority": "High", "verdict": "False", "confidence": 0.8},
    )
    _write_pending(
        db,
        "medium",
        {"claim": "the middle claim", "why": "the rules asked for a person", "priority": "Medium", "verdict": None, "confidence": None},
    )
    rows = list_pending(db)
    assert [row["thread_id"] for row in rows] == ["high", "medium", "low"]
    assert rows[0]["provisional_verdict"] == "False"
    assert rows[2]["confidence"] == 0.4
    assert rows[1]["provisional_verdict"] is None
    assert rows[1]["confidence"] is None
    _delete_pending(db, "high")
    _delete_pending(db, "not-a-run")
    assert [row["thread_id"] for row in list_pending(db)] == ["medium", "low"]


def test_history_is_newest_first(tmp_path):
    db = tmp_path / "audit.db"
    additions = tmp_path / "additions.parquet"
    audit(
        {"claim": "first", "verdict": "False", "human_action": "", "run_marks": {"run_id": "1"}},
        db_path=db,
        additions_path=additions,
    )
    audit(
        {"claim": "second", "verdict": "True", "human_action": "approve", "run_marks": {"run_id": "2"}},
        db_path=db,
        additions_path=additions,
    )
    rows = list_runs(db)
    # The two inserts can share a timestamp. rowid is the later write.
    assert [row["claim"] for row in rows] == ["second", "first"]
    assert rows[0]["human_action"] == "approve"
    assert rows[0]["verdict"] == "True"
