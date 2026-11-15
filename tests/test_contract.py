"""The team contract: one ClaimState shape, and every box module imports.

Why this exists before any box does real work: a renamed key or a forgotten
module fails here, on an empty stub, instead of inside a later branch's tests.
The fixture is the complete example docs/ARCHITECTURE.md §4 points the UI at.
"""

import importlib
import json
from pathlib import Path

from misinfo_agent.state import ClaimState

FIXTURE = Path(__file__).parent / "fixtures" / "sample_state.json"

# Every box module, plus the shared modules they all import. llm.py is the
# only legal place for a model call, search.py the only legal place for a
# web search. A missing module would let a later box import a provider SDK
# directly.
MODULES = [
    "misinfo_agent.config",
    "misinfo_agent.state",
    "misinfo_agent.ingest",
    "misinfo_agent.extract",
    "misinfo_agent.memory",
    "misinfo_agent.prior_match",
    "misinfo_agent.investigate",
    "misinfo_agent.gate",
    "misinfo_agent.assess",
    "misinfo_agent.priority",
    "misinfo_agent.decide",
    "misinfo_agent.review",
    "misinfo_agent.output",
    "misinfo_agent.audit",
    "misinfo_agent.llm",
    "misinfo_agent.search",
    "misinfo_agent.fetch",
    "misinfo_agent.graph",
]


def test_fixture_covers_every_claim_state_key():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = set(ClaimState.__annotations__)
    # A complete fixture has each key present, even when the value is null.
    # Extra keys would be a second, undocumented contract.
    assert expected <= set(payload), sorted(expected - set(payload))
    assert set(payload) <= expected, sorted(set(payload) - expected)
    assert len(payload["evidence"]) == 3
    assert [row["id"] for row in payload["evidence"]] == ["E1", "E2", "E3"]
    for row in payload["evidence"]:
        assert {"url", "stance", "tier", "group", "date", "quote"} <= set(row)


def test_every_box_module_imports():
    for name in MODULES:
        importlib.import_module(name)
