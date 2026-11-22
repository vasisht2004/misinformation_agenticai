"""Box 8+10. The model is faked. No network and no quota.

The verdict comes from the gate's status plus refute_kind. These tests
build short evidence rows, run the real gate, and check that a draft
cannot outvote it. Confidence is compared on the same features so a
weight change that flattens four groups into two, or ignores a named
dissenter, fails here.
"""

import pytest

import misinfo_agent.config as config
from misinfo_agent.assess import assess
from misinfo_agent.gate import gate
from misinfo_agent.llm import LLMError, LLMQuotaError, clear_fake, set_fake


@pytest.fixture(autouse=True)
def _old_gate(monkeypatch):
    # These rows were written for the older count. The new core-match
    # and single-denial rules have their own tests. Assess is what this
    # file locks, and it needs the table the draft was written against.
    monkeypatch.setattr(config, "TRUE_NEEDS_CORE_MATCH", False)
    monkeypatch.setattr(config, "TIER1_SINGLE_REFUTE", False)


@pytest.fixture(autouse=True)
def _reset_fake():
    clear_fake()
    yield
    clear_fake()


def _row(n, group, stance, tier, *, atomic="A1", quote=None):
    return {
        "id": f"E{n}",
        "url": f"https://www.{group}/story-{n}",
        "netloc": group,
        "stance": stance,
        "tier": tier,
        "group_id": group,
        "atomic_id": atomic,
        "quote": quote or f"{stance} from {group}",
        "published_date": "2024-06-01",
        "date_source": "htmldate",
    }


def _claim(aid, text, *, core=False, checkable=True):
    return {"id": aid, "text": text, "core": core, "checkable": checkable}


def _ready(evidence, atomics, *, rounds=1, claim="The claim under test"):
    """Run the real gate so assess sees the statuses it is supposed to trust."""
    state = {
        "claim": claim,
        "claim_date": "2024-06-01",
        "checkable": True,
        "route": "investigate",
        "input_type": "text",
        "match_type": "none",
        "matches": [],
        "best_match": {},
        "atomic_claims": atomics,
        "evidence": evidence,
        "rounds": rounds,
        "investigate_stats": {"quote_unverified": 0, "topup_used": False},
    }
    # The gate advances rounds only when it asks for another search.
    # Setting the budget first is what turns a thin table into
    # insufficient instead of search_more.
    state.update(gate(state))
    if "rounds" not in state:
        state["rounds"] = rounds
    return state


def _draft(items, *, true_version="What the rows say [E1].", correction="Do not forward this."):
    def fake(prompt, schema, images):
        return {
            "atomic": items,
            "true_version": true_version,
            "correction_message": correction,
        }

    set_fake(fake)


def _refutes(n, group, tier, **kwargs):
    return _row(n, group, "refutes", tier, **kwargs)


def _supports(n, group, tier, **kwargs):
    return _row(n, group, "supports", tier, **kwargs)


def test_core_refuted_entirely_false_is_false():
    state = _ready(
        [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        [_claim("A1", "The cure works.", core=True)],
    )
    assert state["atomic_status"]["A1"] == "refuted"
    seen = {}

    def fake(prompt, schema, images):
        seen["prompt"] = prompt
        return {
            "atomic": [
                {
                    "atomic_id": "A1",
                    "explanation": "The sources say this cure does not work [E1].",
                    "refute_kind": "entirely_false",
                }
            ],
            "true_version": "The cure does not work [E1].",
            "correction_message": "This cure does not work. Do not forward it.",
        }

    set_fake(fake)
    out = assess(state)
    assert out["verdict"] == "False"
    assert out["needs_human"] is False
    assert "E1" in out["explanation"]
    assert out["caveats"] == []
    # The prompt is the one that forbids the model from picking the label.
    assert "ONLY the evidence" in seen["prompt"]
    assert "not choosing a verdict" in seen["prompt"]


def test_core_refuted_wrong_detail_is_misleading():
    state = _ready(
        [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        [_claim("A1", "The line opened in 2019.", core=True)],
    )
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "The line opened, but not in that year [E1].",
                "refute_kind": "wrong_detail",
            }
        ]
    )
    out = assess(state)
    assert out["verdict"] == "Misleading"
    assert out["needs_human"] is False


def test_refute_kind_maps_false_and_misleading():
    # The main assertion being one wrong fact is False. A side detail,
    # an old truth, a partial truth, and a quote pulled from somewhere
    # else are Misleading. Each kind is its own draft.
    state = _ready(
        [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        [_claim("A1", "The bank opened in 1947.", core=True)],
    )
    expected = {
        "entirely_false": "False",
        "single_fact_wrong": "False",
        "wrong_detail": "Misleading",
        "outdated": "Misleading",
        "partly_true": "Misleading",
        "out_of_context": "Misleading",
    }
    for kind, verdict in expected.items():
        _draft(
            [
                {
                    "atomic_id": "A1",
                    "explanation": "The rows disagree with the claim [E1].",
                    "refute_kind": kind,
                }
            ]
        )
        out = assess(state)
        assert out["verdict"] == verdict, kind


def test_supported_core_and_supported_rest_is_true():
    evidence = [
        _supports(1, "a.com", 2, atomic="A1"),
        _supports(2, "b.com", 2, atomic="A1"),
        _supports(3, "c.com", 2, atomic="A2"),
        _supports(4, "d.com", 2, atomic="A2"),
    ]
    state = _ready(
        evidence,
        [
            _claim("A1", "The metro line opened.", core=True),
            _claim("A2", "The minister attended.", core=False),
        ],
    )
    assert state["atomic_status"] == {"A1": "supported", "A2": "supported"}
    _draft(
        [
            {"atomic_id": "A1", "explanation": "Two newsrooms report the opening [E1].", "refute_kind": None},
            {"atomic_id": "A2", "explanation": "Two newsrooms report the visit [E3].", "refute_kind": None},
        ],
        true_version="The line opened and the minister attended [E1].",
    )
    out = assess(state)
    assert out["verdict"] == "True"
    assert out["caveats"] == []


def test_supported_core_and_refuted_part_is_misleading():
    evidence = [
        _supports(1, "a.com", 2, atomic="A1"),
        _supports(2, "b.com", 2, atomic="A1"),
        _refutes(3, "c.com", 1, atomic="A2"),
        _refutes(4, "d.com", 2, atomic="A2"),
    ]
    state = _ready(
        evidence,
        [
            _claim("A1", "The metro line opened.", core=True),
            _claim("A2", "The campus was closed that day.", core=False),
        ],
    )
    assert state["atomic_status"]["A1"] == "supported"
    assert state["atomic_status"]["A2"] == "refuted"
    _draft(
        [
            {"atomic_id": "A1", "explanation": "The opening is reported [E1].", "refute_kind": None},
            {"atomic_id": "A2", "explanation": "The campus stayed open [E3].", "refute_kind": "entirely_false"},
        ]
    )
    out = assess(state)
    assert out["verdict"] == "Misleading"


def test_supported_core_and_unconfirmed_part_is_true_with_caveats():
    evidence = [
        _supports(1, "a.com", 1, atomic="A1"),
        _supports(2, "b.com", 2, atomic="A1"),
        _supports(3, "blog.example", 4, atomic="A2"),
    ]
    campus = "He then visited the IIM Bangalore campus."
    state = _ready(
        evidence,
        [
            _claim("A1", "The Pink Line was inaugurated.", core=True),
            _claim("A2", campus, core=False),
        ],
    )
    assert state["atomic_status"]["A1"] == "supported"
    assert state["atomic_status"]["A2"] == "insufficient"
    _draft(
        [
            {"atomic_id": "A1", "explanation": "Official pages report the inauguration [E1].", "refute_kind": None},
            {"atomic_id": "A2", "explanation": "Only one blog mentions the campus [E3].", "refute_kind": None},
        ]
    )
    out = assess(state)
    assert out["verdict"] == "True"
    assert campus in out["caveats"]


def test_core_insufficient_stays_unverifiable_even_if_the_draft_says_false():
    state = _ready(
        [_refutes(1, "a.com", 1)],
        [_claim("A1", "One source is not enough.", core=True)],
        rounds=2,
    )
    assert state["gate"] == "insufficient"
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "This one source rejects it [E1].",
                "refute_kind": "entirely_false",
            }
        ]
    )
    out = assess(state)
    assert out["verdict"] == "Unverifiable"
    # The score is confidence in the abstention, and it has to be a
    # probability rather than a leftover from the draft.
    assert 0.0 < out["confidence"] < 1.0


def test_conflicting_gate_withholds_the_verdict():
    evidence = [
        _supports(1, "ministry.gov.in", 1),
        _refutes(2, "paper.com", 2),
        _refutes(3, "other.com", 2),
    ]
    state = _ready(evidence, [_claim("A1", "Both sides are credible.", core=True)])
    assert state["gate"] == "conflicting"
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "The rows disagree [E1].",
                "refute_kind": "entirely_false",
            }
        ]
    )
    out = assess(state)
    assert out["verdict"] is None
    assert out["needs_human"] is True


def test_error_gate_withholds_the_verdict():
    state = _ready(
        [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        [_claim("A1", "The stance call crashed.", core=True)],
        rounds=2,
    )
    state["investigate_stats"] = {"stance_failed": True, "quote_unverified": 0, "topup_used": False}
    state.update(gate(state))
    assert state["gate"] == "error"
    _draft(
        [{"atomic_id": "A1", "explanation": "Looks refuted [E1].", "refute_kind": "entirely_false"}]
    )
    out = assess(state)
    assert out["verdict"] is None
    assert out["needs_human"] is True


def test_not_checkable_is_other_and_skips_the_model():
    calls = []

    def fake(prompt, schema, images):
        calls.append(prompt)
        raise AssertionError("assess called the model for a non-claim")

    set_fake(fake)
    out = assess(
        {
            "checkable": False,
            "route": "investigate",
            "claim": "What a lovely morning.",
            "evidence": [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        }
    )
    assert calls == []
    assert out["verdict"] == "Other"
    assert out["confidence"] == 0.0


def test_invalid_citation_is_dropped():
    evidence = [
        _refutes(1, "a.com", 1, atomic="A1"),
        _refutes(2, "b.com", 2, atomic="A1"),
        _refutes(3, "c.com", 2, atomic="A2"),
    ]
    state = _ready(
        evidence,
        [
            _claim("A1", "The core event.", core=True),
            _claim("A2", "A side event.", core=False),
        ],
    )
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "The bank denied it [E1]. A different claim's row was cited [E3]. A missing row was cited [E99].",
                "refute_kind": "entirely_false",
            },
            {
                "atomic_id": "A2",
                "explanation": "This side has one source [E3].",
                "refute_kind": None,
            },
        ]
    )
    out = assess(state)
    core = out["atomic_results"][0]["explanation"]
    assert "[E1]" in core
    assert "[E3]" not in core
    assert "[E99]" not in core
    dropped = " ".join(out["assess_notes"])
    assert "E3" in dropped
    assert "E99" in dropped


def test_empty_explanation_uses_the_evidence_template():
    state = _ready(
        [_refutes(1, "rbi.org.in", 1, quote="No such circular was issued."), _refutes(2, "thehindu.com", 2)],
        [_claim("A1", "The notice is real.", core=True)],
    )
    _draft(
        [{"atomic_id": "A1", "explanation": "   ", "refute_kind": "entirely_false"}],
        true_version="",
        correction="",
    )
    out = assess(state)
    assert out["verdict"] == "False"
    assert "[E1]" in out["explanation"]
    assert "No such circular was issued." in out["explanation"]
    assert "rbi.org.in" in out["explanation"]
    assert any("no explanation" in note for note in out["assess_notes"])
    # The shareable line is the template, and it does not carry an id.
    assert "[E" not in out["correction_message"]
    assert out["correction_message"]


def test_llm_failure_uses_the_template():
    state = _ready(
        [_refutes(1, "a.com", 1, quote="The claim is false."), _refutes(2, "b.com", 2)],
        [_claim("A1", "A cure claim.", core=True)],
    )

    def fake(prompt, schema, images):
        raise LLMError("assess", "model unavailable")

    set_fake(fake)
    out = assess(state)
    assert "[E1]" in out["explanation"]
    assert "The claim is false." in out["explanation"]
    assert any("assess call failed" in note for note in out["assess_notes"])
    # False needs the model to say entirely_false. A failed call did not
    # say it, so the harsher label is not assumed.
    assert out["verdict"] == "Misleading"


def test_reuse_prior_does_not_call_the_model():
    calls = []

    def fake(prompt, schema, images):
        calls.append(prompt)
        raise AssertionError("reuse_prior called the model")

    set_fake(fake)
    out = assess(
        {
            "route": "reuse_prior",
            "checkable": True,
            "claim": "HSBC Bank is completely withdrawing from Sri Lanka.",
            "atomic_claims": [
                _claim("A1", "HSBC Bank is completely withdrawing from Sri Lanka.", core=True)
            ],
            "best_match": {
                "verdict_norm": "Misleading",
                "true_version": "HSBC is selling only its retail banking operations.",
                "primary_claim": "HSBC Bank is completely withdrawing from Sri Lanka.",
                "dense_cos": 0.96,
            },
        }
    )
    assert calls == []
    assert out["verdict"] == "Misleading"
    assert out["explanation"] == "HSBC is selling only its retail banking operations. [C1]"
    assert out["true_version"] == out["explanation"]
    assert "[C1]" not in out["correction_message"]
    assert out["citations"] == ["C1"]
    assert out["what_we_found"] == ""
    assert out["needs_human"] is False
    assert any("No assess model call" in note for note in out["assess_notes"])
    weaker = assess(
        {
            "route": "reuse_prior",
            "checkable": True,
            "claim": "same claim",
            "best_match": {
                "verdict_norm": "Misleading",
                "true_version": "Only retail.",
                "dense_cos": 0.85,
            },
        }
    )
    assert weaker["confidence"] < out["confidence"]


def test_four_credible_groups_outrank_two():
    def rows(names):
        return [_refutes(i, name, 2) for i, name in enumerate(names, start=1)]

    wide = _ready(rows(["a.com", "b.com", "c.com", "d.com"]), [_claim("A1", "Wide.", core=True)])
    narrow = _ready(rows(["a.com", "b.com"]), [_claim("A1", "Narrow.", core=True)])
    assert wide["atomic_status"]["A1"] == "refuted"
    assert narrow["atomic_status"]["A1"] == "refuted"
    _draft(
        [{"atomic_id": "A1", "explanation": "The rows refute it [E1].", "refute_kind": "entirely_false"}]
    )
    assert assess(wide)["confidence"] > assess(narrow)["confidence"]


def test_a_named_dissenter_lowers_confidence():
    base = [_refutes(i, f"p{i}.com", 2) for i in range(1, 5)]
    clean = _ready(base, [_claim("A1", "Four sources.", core=True)])
    mixed_rows = base + [_supports(5, "dissenter.com", 2)]
    mixed = _ready(mixed_rows, [_claim("A1", "Four sources and one dissent.", core=True)])
    # The stray is named by the gate and does not flip the status. The
    # confidence drop is that same named source, not a second verdict.
    assert mixed["atomic_status"]["A1"] == "refuted"
    assert any("1 source disagrees" in line for line in mixed["gate_reasons"])
    _draft(
        [{"atomic_id": "A1", "explanation": "The rows refute it [E1].", "refute_kind": "entirely_false"}]
    )
    assert assess(mixed)["confidence"] < assess(clean)["confidence"]


def test_quota_error_is_not_turned_into_a_template():
    state = _ready(
        [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        [_claim("A1", "Stop the run.", core=True)],
    )

    def fake(prompt, schema, images):
        raise LLMQuotaError("assess", "daily cap")

    set_fake(fake)
    with pytest.raises(LLMQuotaError):
        assess(state)


def test_unverifiable_moves_the_summary_and_drops_the_kind():
    state = _ready(
        [_refutes(1, "a.com", 1)],
        [_claim("A1", "One source is not enough.", core=True)],
        rounds=2,
    )
    assert state["gate"] == "insufficient"
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "This one source rejects it [E1].",
                "refute_kind": "entirely_false",
            }
        ],
        true_version="No official holiday was declared [E1].",
        correction="Schools were not closed.",
    )
    out = assess(state)
    assert out["verdict"] == "Unverifiable"
    assert out["true_version"] == ""
    assert out["correction_message"] == (
        "We could not find two independent credible sources, so this is not confirmed."
    )
    assert out["what_we_found"] == "No official holiday was declared [E1]."
    assert "Schools were not closed." not in out["correction_message"]
    assert out["atomic_results"][0]["refute_kind"] is None


def test_a_refuted_claim_keeps_its_kind_and_its_correction():
    state = _ready(
        [_refutes(1, "a.com", 1), _refutes(2, "b.com", 2)],
        [_claim("A1", "The cure works.", core=True)],
    )
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "The sources say this cure does not work [E1].",
                "refute_kind": "entirely_false",
            }
        ],
        true_version="The cure does not work [E1].",
        correction="This cure does not work. Do not forward it.",
    )
    out = assess(state)
    assert out["verdict"] == "False"
    assert out["true_version"] == "The cure does not work [E1]."
    assert out["correction_message"] == "This cure does not work. Do not forward it."
    assert out["what_we_found"] == ""
    assert out["atomic_results"][0]["refute_kind"] == "entirely_false"


def test_a_withheld_verdict_uses_the_same_template():
    evidence = [
        _supports(1, "ministry.gov.in", 1),
        _refutes(2, "paper.com", 2),
        _refutes(3, "other.com", 2),
    ]
    state = _ready(evidence, [_claim("A1", "Both sides are credible.", core=True)])
    assert state["gate"] == "conflicting"
    _draft(
        [
            {
                "atomic_id": "A1",
                "explanation": "The rows disagree [E1].",
                "refute_kind": "entirely_false",
            }
        ],
        true_version="The ministry and the papers disagree [E1].",
        correction="Do not treat this as settled.",
    )
    out = assess(state)
    assert out["verdict"] is None
    assert out["needs_human"] is True
    assert out["true_version"] == ""
    assert out["correction_message"] == (
        "We could not find two independent credible sources, so this is not confirmed."
    )
    assert out["what_we_found"] == "The ministry and the papers disagree [E1]."
    assert out["atomic_results"][0]["refute_kind"] is None
