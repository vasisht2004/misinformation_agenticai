"""Box 9 on short evidence rows. No model and no network.

Each case is a claim the investigate eval already met, reduced to the
rows the rule actually counts: group, tier, stance, and which atomic
claim the row is about. The gate must not open a client to judge them.
"""

import json
from pathlib import Path

import pytest

import misinfo_agent.config as config
from misinfo_agent.gate import gate

FIXTURE = Path(__file__).parent / "fixtures" / "sample_state.json"


def _row(
    n,
    group,
    stance,
    tier,
    *,
    atomic="A1",
    conflict=False,
    after=False,
    host=None,
    quote=None,
):
    # The host is only so a reader can see which outlet the group is.
    # The gate counts group_id and the tier already written on the row.
    host = host or group
    return {
        "id": f"E{n}",
        "url": f"https://{host}/story-{n}",
        "netloc": host,
        "stance": stance,
        "tier": tier,
        "group_id": group,
        "atomic_id": atomic,
        "group_conflict": conflict,
        "after_claim": after,
        "quote": f"{stance} {n}" if quote is None else quote,
    }


def _state(**overrides):
    state = {
        "claim": "A checkable claim.",
        "checkable": True,
        "route": "investigate",
        "input_type": "text",
        "rounds": 1,
        "atomic_claims": [
            {"id": "A1", "text": "A checkable claim.", "core": True, "checkable": True}
        ],
        "evidence": [],
        "investigate_stats": {"stance_failed": False},
    }
    state.update(overrides)
    return state


def _joined(result) -> str:
    return " ".join(result["gate_reasons"])


def test_lemon_five_refuting_groups_is_refuted():
    # National Academies, Global News, UQ, Euronews, plus a blog.
    # Four of the five refutes are tier 2. The support is a tier-4 blog
    # and must not become "1 source disagrees": that phrase is for a
    # credible stray, and a blog is not one.
    evidence = [
        _row(1, "nationalacademies.org", "refutes", 2),
        _row(2, "globalnews.ca", "refutes", 2),
        _row(3, "uq.edu.au", "refutes", 2),
        _row(4, "euronews.com", "refutes", 2),
        _row(5, "medium.com", "refutes", 4),
        _row(6, "blogspot.com", "supports", 4),
    ]
    out = gate(_state(
        claim="Drinking hot lemon water kills coronavirus",
        evidence=evidence,
        atomic_claims=[],
    ))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert "1 source disagrees" not in _joined(out)


def test_garlic_springer_and_hk_health_bureau_is_refuted():
    evidence = [
        _row(1, "springer.com", "refutes", 2, host="link.springer.com"),
        _row(2, "chp.gov.hk", "refutes", 1),
    ]
    out = gate(_state(
        claim="Drinking raw garlic juice every morning cures Type 2 diabetes in 30 days.",
        domain="Health",
        evidence=evidence,
    ))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"


def test_rupee_500_rbi_pib_bbc_is_refuted():
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1),
        _row(2, "pib.gov.in", "refutes", 1),
        _row(3, "bbc.com", "refutes", 2),
    ]
    out = gate(_state(
        claim="RBI has announced that all ₹500 notes printed before 2016 will be invalid from 1 December.",
        domain="Finance & scams",
        evidence=evidence,
    ))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"


def test_pink_line_core_supported_and_side_claim_insufficient(monkeypatch):
    # A1 has five supporting groups, four of them credible. A2 is a
    # second allegation that only a blog repeats. The core is enough,
    # so the overall gate is sufficient and the thin side claim does
    # not send the whole forward back to search.
    # Every quote is the same sentence so the side claim is what this
    # case is about. The shipped shared-source flag would collapse that
    # repetition into one outlet; the copy rule has its own test.
    monkeypatch.setattr(config, "GATE_SHARED_SOURCE", False)
    a1 = "PM Narendra Modi inaugurated the Pink Line of Namma Metro in Bengaluru."
    a2 = "The Pink Line opened with a fare of ₹10 for the full route."
    evidence = [
        _row(1, "pib.gov.in", "supports", 1, atomic="A1", quote=a1),
        _row(2, "ndtv.com", "supports", 2, atomic="A1", quote=a1),
        _row(3, "thehindu.com", "supports", 2, atomic="A1", quote=a1),
        _row(4, "indianexpress.com", "supports", 2, atomic="A1", quote=a1),
        _row(5, "medium.com", "supports", 4, atomic="A1", quote=a1),
        _row(6, "blogspot.com", "supports", 4, atomic="A2", quote=a2),
    ]
    out = gate(_state(
        claim=a1,
        evidence=evidence,
        atomic_claims=[
            {"id": "A1", "text": a1, "core": True, "checkable": True},
            {"id": "A2", "text": a2, "core": False, "checkable": True},
        ],
    ))
    assert out["atomic_status"]["A1"] == "supported"
    assert out["atomic_status"]["A2"] == "insufficient"
    assert out["gate"] == "sufficient"
    assert "rounds" not in out


def test_saudi_one_tier4_support_searches_then_is_insufficient():
    evidence = [_row(1, "bronx.com", "supports", 4)]
    claim = "Women in Saudi Arabia may choose what they wear."
    first = gate(_state(claim=claim, evidence=evidence, rounds=1))
    assert first["atomic_status"]["A1"] == "insufficient"
    assert first["gate"] == "search_more"
    assert first["rounds"] == 2
    second = gate(_state(claim=claim, evidence=evidence, rounds=2))
    assert second["atomic_status"]["A1"] == "insufficient"
    assert second["gate"] == "insufficient"
    assert "rounds" not in second


def test_schools_image_with_no_evidence_is_insufficient():
    claim = "All Bengaluru schools are closed till Monday due to heavy rain."
    base = dict(
        claim=claim,
        input_type="image",
        evidence=[],
        atomic_claims=[{"id": "A1", "text": claim, "core": True, "checkable": True}],
    )
    # One finished round still has a search left. An empty table is not
    # yet the final word, because a later round might refute the notice.
    early = gate(_state(**base, rounds=1))
    assert early["atomic_status"]["A1"] == "insufficient"
    assert early["gate"] == "search_more"
    done = gate(_state(**base, rounds=2))
    assert done["atomic_status"]["A1"] == "insufficient"
    assert done["gate"] == "insufficient"


def test_chennai_picture_is_never_supported_by_text():
    claim = "Video shows flood in Chennai"
    evidence = [
        _row(1, "aljazeera.com", "supports", 2, quote=claim),
        _row(2, "ndtv.com", "supports", 2, quote=claim),
    ]
    # The claim is text. Box 6's picture detector is what blocks it,
    # not input_type. Two tier-2 supports would otherwise be enough.
    out = gate(_state(claim=claim, evidence=evidence, rounds=2))
    assert out["atomic_status"]["A1"] == "insufficient"
    assert out["atomic_status"]["A1"] != "supported"
    assert out["gate"] == "insufficient"
    assert "Text sources cannot confirm what a picture shows." in _joined(out)
    early = gate(_state(claim=claim, evidence=evidence, rounds=1))
    assert early["atomic_status"]["A1"] != "supported"
    assert early["gate"] == "search_more"
    assert "Text sources cannot confirm what a picture shows." in _joined(early)


def test_text_can_refute_a_picture_claim():
    claim = "Video shows flood in Chennai"
    evidence = [
        _row(1, "aljazeera.com", "refutes", 2),
        _row(2, "ndtv.com", "refutes", 2),
    ]
    out = gate(_state(claim=claim, evidence=evidence))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert "Text sources cannot confirm what a picture shows." not in _joined(out)


def test_tier1_support_against_tier2_refutes_is_conflicting():
    # One tier-1 support and two tier-2 refutes. The support is a third
    # of the credible groups, so it blocks even before the tier-1 rule.
    evidence = [
        _row(1, "who.int", "supports", 1, quote="A checkable claim."),
        _row(2, "bbc.com", "refutes", 2),
        _row(3, "thehindu.com", "refutes", 2),
    ]
    out = gate(_state(evidence=evidence, rounds=1))
    assert out["atomic_status"]["A1"] == "conflicting"
    assert out["gate"] == "conflicting"
    assert "rounds" not in out

    # The fraction alone would not block 1 against 4. Tier 1 still does:
    # a ministry is not a stray source.
    many = [
        _row(1, "who.int", "supports", 1, quote="A checkable claim."),
        _row(2, "bbc.com", "refutes", 2),
        _row(3, "thehindu.com", "refutes", 2),
        _row(4, "ndtv.com", "refutes", 2),
        _row(5, "reuters.com", "refutes", 2),
    ]
    blocked = gate(_state(evidence=many, rounds=2))
    assert blocked["atomic_status"]["A1"] == "conflicting"
    assert blocked["gate"] == "conflicting"


def test_one_credible_support_against_four_refutes_does_not_block():
    evidence = [
        _row(1, "bbc.com", "refutes", 2),
        _row(2, "thehindu.com", "refutes", 2),
        _row(3, "ndtv.com", "refutes", 2),
        _row(4, "reuters.com", "refutes", 2),
        _row(5, "aljazeera.com", "supports", 2, quote="A checkable claim."),
    ]
    out = gate(_state(evidence=evidence))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert "1 source disagrees" in _joined(out)


def test_group_conflict_counts_on_neither_side():
    conflicted = [
        _row(1, "thehindu.com", "supports", 2, conflict=True, quote="A checkable claim."),
        _row(2, "thehindu.com", "refutes", 2, conflict=True),
    ]
    alone = gate(_state(evidence=conflicted, rounds=2))
    assert alone["atomic_status"]["A1"] == "insufficient"
    assert "thehindu.com argues both sides" in _joined(alone)
    assert "1 source disagrees" not in _joined(alone)

    # Two real refutes still refute. The conflicted outlet is named and
    # is not a third vote, which would have made the sentence say 3.
    with_votes = conflicted + [
        _row(3, "bbc.com", "refutes", 2),
        _row(4, "reuters.com", "refutes", 2),
    ]
    out = gate(_state(evidence=with_votes))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert "2 independent groups refute it" in _joined(out)
    assert "thehindu.com argues both sides" in _joined(out)


def test_stance_failed_searches_then_errors():
    # A full refute table must not become a verdict when the stance
    # call itself failed. The rows may be a partial batch.
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1),
        _row(2, "pib.gov.in", "refutes", 1),
        _row(3, "bbc.com", "refutes", 2),
    ]
    failed = {"investigate_stats": {"stance_failed": True}, "evidence": evidence}
    first = gate(_state(**failed, rounds=1))
    assert first["gate"] == "search_more"
    assert first["rounds"] == 2
    assert first["atomic_status"] == {}
    second = gate(_state(**failed, rounds=2))
    assert second["gate"] == "error"
    assert second["gate"] != "insufficient"
    assert "rounds" not in second
    assert "system failure" in _joined(second)


def test_reuse_prior_and_not_checkable_skip_the_gate():
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1),
        _row(2, "bbc.com", "refutes", 2),
    ]
    reused = gate(_state(route="reuse_prior", evidence=evidence, rounds=1))
    assert reused["gate"] == "skipped"
    assert reused["atomic_status"] == {}
    assert "exact, fresh fact-check" in _joined(reused)

    unchecked = gate(_state(checkable=False, evidence=[], rounds=0))
    assert unchecked["gate"] == "skipped"
    assert unchecked["gate"] != "search_more"
    assert "not checkable" in _joined(unchecked)


def test_not_checkable_atomic_is_recorded_and_skipped():
    evidence = [
        _row(1, "springer.com", "refutes", 2, atomic="A1"),
        _row(2, "chp.gov.hk", "refutes", 1, atomic="A1"),
        _row(3, "medium.com", "supports", 4, atomic="A2"),
    ]
    out = gate(_state(
        evidence=evidence,
        atomic_claims=[
            {
                "id": "A1",
                "text": "Drinking raw garlic juice cures Type 2 diabetes in 30 days.",
                "core": True,
                "checkable": True,
            },
            {
                "id": "A2",
                "text": "The government is hiding it to protect pharma companies.",
                "core": False,
                "checkable": False,
            },
        ],
    ))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["atomic_status"]["A2"] == "not_checkable"
    assert out["gate"] == "sufficient"


def test_after_claim_rows_are_ignored_only_when_the_guard_is_on(monkeypatch):
    evidence = [
        _row(1, "bbc.com", "refutes", 2, after=False),
        _row(2, "thehindu.com", "refutes", 2, after=True),
    ]
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)
    kept = gate(_state(evidence=evidence, rounds=2))
    assert kept["atomic_status"]["A1"] == "refuted"
    assert kept["gate"] == "sufficient"

    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", True)
    dropped = gate(_state(evidence=evidence, rounds=2))
    assert dropped["atomic_status"]["A1"] == "insufficient"
    assert dropped["gate"] == "insufficient"


def test_style_does_not_change_the_gate():
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1),
        _row(2, "pib.gov.in", "refutes", 1),
        _row(3, "bbc.com", "refutes", 2),
    ]
    quiet = gate(_state(evidence=evidence, style_signals={}))
    shouted = gate(_state(
        evidence=evidence,
        style_signals={
            "urgency_words": ["breaking", "urgent"],
            "share_request": True,
            "forwarded_label": "Forwarded many times",
        },
        red_flags=["Fear or outrage", "Call to forward"],
    ))
    assert quiet["gate"] == shouted["gate"] == "sufficient"
    assert quiet["atomic_status"] == shouted["atomic_status"]


def test_sample_fixture_core_is_refuted():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    out = gate(payload)
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"


def test_core_match_drops_a_detail_and_needs_two_tier12_groups(monkeypatch):
    # The fare is true and the inauguration is the claim. Counting the
    # fare as support was a false True. Flags on is the default.
    claim = "PM Narendra Modi inaugurated the Pink Line in Bengaluru."
    detail = "The Pink Line fare is ₹10."
    evidence = [
        _row(1, "pib.gov.in", "supports", 1, quote=detail),
        _row(2, "ndtv.com", "supports", 2, quote=claim),
        _row(3, "thehindu.com", "supports", 2, quote=claim),
    ]
    out = gate(_state(claim=claim, evidence=evidence, rounds=2, atomic_claims=[
        {"id": "A1", "text": claim, "core": True, "checkable": True},
    ]))
    assert out["atomic_status"]["A1"] == "supported"
    assert out["core_detail_only_ids"] == ["E1"]

    thin = [
        _row(1, "pib.gov.in", "supports", 1, quote=claim),
        _row(2, "blogspot.com", "supports", 4, quote=claim),
    ]
    strict = gate(_state(claim=claim, evidence=thin, rounds=2, atomic_claims=[
        {"id": "A1", "text": claim, "core": True, "checkable": True},
    ]))
    assert strict["atomic_status"]["A1"] == "insufficient"
    monkeypatch.setattr(config, "TRUE_NEEDS_CORE_MATCH", False)
    monkeypatch.setattr(config, "TIER1_SINGLE_REFUTE", False)
    old = gate(_state(claim=claim, evidence=thin, rounds=2, atomic_claims=[
        {"id": "A1", "text": claim, "core": True, "checkable": True},
    ]))
    assert old["atomic_status"]["A1"] == "supported"
    assert old["core_detail_only_ids"] == []
    assert "2 independent groups support it, and 1 of them are tier 1 or 2." in _joined(old)


def test_relaxed_paraphrase_matches_and_strict_does_not(monkeypatch):
    from misinfo_agent.prior_match import core_support_matches

    core = "Zohran Mamdani won the Democratic primary"
    monkeypatch.setattr(config, "CORE_MATCH_MODE", "relaxed")
    assert core_support_matches("Mamdani won", core)
    assert core_support_matches("Zohran Mamdani's win", core)
    assert core_support_matches("Mamdani defeated", core)
    # The sentence around the quote can carry the verb. The next sentence
    # is not read.
    assert core_support_matches(
        "Mamdani",
        core,
        passage="City desk. Mamdani won the Democratic primary last night. Turnout was high.",
    )
    assert core_support_matches("Rahul Gandhi won the primary", core) is False
    money = "The ₹500 note is invalid."
    assert core_support_matches("The ₹2,000 note is invalid.", money) is False
    assert core_support_matches("The ₹2,000 note is invalid.", money, mode="strict") is False

    evidence = [
        _row(1, "bbc.com", "supports", 2, quote="Mamdani won"),
        _row(2, "reuters.com", "supports", 2, quote="Zohran Mamdani's win"),
        _row(3, "ndtv.com", "supports", 2, quote="Mamdani defeated Cuomo"),
    ]
    state = _state(claim=core, evidence=evidence, rounds=2, atomic_claims=[
        {"id": "A1", "text": core, "core": True, "checkable": True},
    ])
    relaxed = gate(state)
    assert relaxed["atomic_status"]["A1"] == "supported"
    assert relaxed["core_detail_only_ids"] == []

    # Strict is the previous rule: the quote must contain the full name
    # and the exact word "won".
    monkeypatch.setattr(config, "CORE_MATCH_MODE", "strict")
    assert core_support_matches("Mamdani won", core) is False
    assert core_support_matches(core, core) is True
    strict = gate(state)
    assert strict["atomic_status"]["A1"] == "insufficient"
    assert set(strict["core_detail_only_ids"]) == {"E1", "E2", "E3"}


def test_relaxed_dates_numbers_and_rupees_match_exactly(monkeypatch):
    # The forms are the same fact. A different year or a different count
    # is a different fact, including after the words are turned into digits.
    from misinfo_agent.prior_match import core_support_matches

    monkeypatch.setattr(config, "CORE_MATCH_MODE", "relaxed")
    claim = "The Constitution of India came into effect on 26 January 1950."
    for quote in (
        "The Constitution of India came into effect on 26 January 1950.",
        "The Constitution of India came into effect on January 26, 1950.",
        "The Constitution of India came into effect on 26 Jan 1950.",
        "The Constitution of India came into effect on 26-01-1950.",
        "The Constitution of India came into effect on 26/01/1950.",
        "The Constitution of India came into effect on 1950-01-26.",
    ):
        assert core_support_matches(quote, claim) is True, quote
    # A year-only claim is covered by a full date in that year.
    assert core_support_matches(
        "The bank was established on 1 April 1935.",
        "The bank was established in 1935.",
    ) is True
    # The other direction is not a match: a year does not supply the day.
    assert core_support_matches(
        "The Constitution of India came into effect in 1950.",
        claim,
    ) is False
    assert core_support_matches(
        "The bank was established in 1947.",
        "The bank was established in 1950.",
    ) is False
    assert core_support_matches(
        "The Constitution of India came into effect on 15 August 1947.",
        claim,
    ) is False
    for quote in (
        "An Aadhaar number has 12 digits.",
        "An Aadhaar number has 12-digit length.",
        "An Aadhaar number has twelve digits.",
    ):
        assert core_support_matches(quote, "An Aadhaar number has 12 digits.") is True
    assert core_support_matches(
        "An Aadhaar number has 16 digits.",
        "An Aadhaar number has 12 digits.",
    ) is False
    assert core_support_matches(
        "An Aadhaar number has sixteen digits.",
        "An Aadhaar number has twelve digits.",
    ) is False
    rupee = "The ₹2,000 note was withdrawn."
    for quote in (
        "The ₹2,000 note was withdrawn.",
        "The Rs 2000 note was withdrawn.",
        "The 2000-rupee note was withdrawn.",
        "The two thousand rupee note was withdrawn.",
    ):
        assert core_support_matches(quote, rupee) is True
    assert core_support_matches("The ₹500 note was withdrawn.", rupee) is False


def test_one_tier1_denial_refutes_only_when_the_flag_is_on(monkeypatch):
    claim = "RBI has banned ₹500 notes."
    evidence = [_row(1, "rbi.org.in", "refutes", 1, host="rbi.org.in")]
    out = gate(_state(claim=claim, evidence=evidence, rounds=2))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert "single official denial: rbi.org.in" in _joined(out)

    monkeypatch.setattr(config, "TIER1_SINGLE_REFUTE", False)
    monkeypatch.setattr(config, "TRUE_NEEDS_CORE_MATCH", False)
    old = gate(_state(claim=claim, evidence=evidence, rounds=2))
    assert old["atomic_status"]["A1"] == "insufficient"
    assert "single official denial" not in _joined(old)


def test_tier1_denial_does_not_fire_when_a_tier12_source_supports():
    claim = "RBI has banned ₹500 notes."
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1, host="rbi.org.in"),
        _row(2, "bbc.com", "supports", 2, quote=claim),
    ]
    out = gate(_state(claim=claim, evidence=evidence, rounds=2))
    assert out["atomic_status"]["A1"] == "conflicting"
    assert "single official denial" not in _joined(out)


def test_unverifiable_reason_no_credible_record_needs_the_official_query():
    from misinfo_agent.gate import unverifiable_reason

    claim = "A checkable claim."
    evidence = [_row(1, "blogspot.com", "supports", 4, quote=claim)]
    state = _state(claim=claim, evidence=evidence, rounds=2)
    judged = {**state, **gate(state)}
    assert judged["gate"] == "insufficient"
    assert unverifiable_reason(judged) == "only_tier34"
    official = {
        **judged,
        "investigate_stats": {"queries": [{"kind": "official", "text": "A checkable claim"}]},
    }
    assert unverifiable_reason(official) == "no_credible_record"
    empty = {**state, "evidence": [], "gate": "insufficient", "rounds": 2}
    assert unverifiable_reason(empty) == "no_hits"


def test_product_path_includes_factcheck_sites_and_memory():
    # The graph default leaves the sites in. Only the eval's blocked
    # condition turns the filter on. Box 5 is a node, not an optional hop.
    assert config.BLOCK_FACTCHECK_SITES is False
    from misinfo_agent.graph import build_graph

    compiled = build_graph.__wrapped__ if hasattr(build_graph, "__wrapped__") else None
    source = Path(__file__).resolve().parents[1] / "misinfo_agent" / "graph.py"
    text = source.read_text(encoding="utf-8")
    assert 'add_node("memory"' in text
    assert compiled is None


def _dated(row, cos, day, quote):
    row = dict(row)
    row["cos"] = cos
    row["published_date"] = day
    row["quote"] = quote
    return row


def test_weak_refute_needs_three_groups_and_no_support(monkeypatch):
    # The motivating shape: three independent lower-tier denials, no page
    # on the other side, and each denial is about the claim. The number in
    # the quote is the correction, not a copy of the false figure.
    claim = "India has 29 states in total."
    evidence = [
        _row(1, "en.wikipedia.org", "refutes", 3, quote="India is a federal union comprising 28 states and 8 union territories"),
        _row(2, "testbook.com", "refutes", 4, quote="At present, there are 28 states in the Indian union."),
        _row(3, "sarkarilist.in", "refutes", 4, quote="List of 28 States of India"),
    ]
    state = _state(claim=claim, evidence=evidence, rounds=2, atomic_claims=[
        {"id": "A1", "text": claim, "core": True, "checkable": True},
    ])
    monkeypatch.setattr(config, "GATE_WEAK_REFUTE", False)
    assert gate(state)["atomic_status"]["A1"] == "insufficient"
    monkeypatch.setattr(config, "GATE_WEAK_REFUTE", True)
    out = gate(state)
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert config.WEAK_REFUTE_CAVEAT in _joined(out)
    # One supporting page, at any tier, blocks the path.
    blocked = evidence + [_row(4, "example.com", "supports", 4, quote="India has 29 states")]
    held = gate(_state(claim=claim, evidence=blocked, rounds=2, atomic_claims=state["atomic_claims"]))
    assert held["atomic_status"]["A1"] == "insufficient"
    # Two denials are not three.
    short = gate(_state(claim=claim, evidence=evidence[:2], rounds=2, atomic_claims=state["atomic_claims"]))
    assert short["atomic_status"]["A1"] == "insufficient"


def test_weak_refute_does_not_support_and_drops_an_off_topic_denial(monkeypatch):
    monkeypatch.setattr(config, "GATE_WEAK_REFUTE", True)
    claim = "India has 29 states in total."
    supports = [
        _row(1, "a.example", "supports", 4, quote="India has 29 states in total."),
        _row(2, "b.example", "supports", 4, quote="India has 29 states in total."),
        _row(3, "c.example", "supports", 4, quote="India has 29 states in total."),
    ]
    out = gate(_state(claim=claim, evidence=supports, rounds=2))
    assert out["atomic_status"]["A1"] == "insufficient"
    # A denial that never mentions the claim is not one of the three.
    off = [
        _row(1, "en.wikipedia.org", "refutes", 3, quote="India has 28 states"),
        _row(2, "testbook.com", "refutes", 4, quote="India has 28 states"),
        _row(3, "other.example", "refutes", 4, quote="The minister resigned yesterday"),
    ]
    missed = gate(_state(claim=claim, evidence=off, rounds=2))
    assert missed["atomic_status"]["A1"] == "insufficient"


def test_weak_refute_caps_confidence_and_maps_a_missing_kind(monkeypatch):
    from misinfo_agent.assess import assess
    from misinfo_agent.llm import clear_fake, set_fake

    monkeypatch.setattr(config, "GATE_WEAK_REFUTE", True)
    claim = "India has 29 states in total."
    evidence = [
        _row(1, "en.wikipedia.org", "refutes", 3, quote="India is a federal union comprising 28 states and 8 union territories"),
        _row(2, "testbook.com", "refutes", 4, quote="At present, there are 28 states in the Indian union."),
        _row(3, "sarkarilist.in", "refutes", 4, quote="List of 28 States of India"),
    ]
    state = _state(
        claim=claim,
        evidence=evidence,
        rounds=2,
        atomic_claims=[{"id": "A1", "text": claim, "core": True, "checkable": True}],
    )
    state["claim_date"] = "2026-11-22"
    state["investigate_stats"] = {"quote_unverified": 0, "topup_used": False, "stance_failed": False}
    state.update(gate(state))
    set_fake(lambda prompt, schema, images: {
        "atomic": [{"atomic_id": "A1", "explanation": "The pages say 28 states [E1].", "refute_kind": None}],
        "true_version": "India has 28 states [E1].",
        "correction_message": "Do not forward this.",
    })
    try:
        out = assess(state)
    finally:
        clear_fake()
    # Missing refute_kind stays Misleading. The live rule does not invent False.
    assert out["verdict"] == "Misleading"
    assert config.WEAK_REFUTE_CAVEAT in out["caveats"]
    assert out["confidence"] <= config.WEAK_REFUTE_CONFIDENCE_CAP


def test_ordinary_refute_is_not_capped_as_weak(monkeypatch):
    monkeypatch.setattr(config, "GATE_WEAK_REFUTE", True)
    claim = "RBI banned the note."
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1, quote=claim),
        _row(2, "bbc.com", "refutes", 2, quote=claim),
    ]
    out = gate(_state(claim=claim, evidence=evidence, rounds=2))
    assert out["atomic_status"]["A1"] == "refuted"
    assert config.WEAK_REFUTE_CAVEAT not in _joined(out)


def _about(claim, evidence, **extra):
    return _state(
        claim=claim,
        evidence=evidence,
        rounds=2,
        atomic_claims=[{"id": "A1", "text": claim, "core": True, "checkable": True}],
        **extra,
    )


def test_primary_source_can_support_or_refute_alone(monkeypatch):
    claim = "RBI approved the new note."
    quote = "RBI approved the new note."
    one = [_row(1, "rbi.org.in", "supports", 1, host="rbi.org.in", quote=quote)]
    state = _about(claim, one)
    monkeypatch.setattr(config, "GATE_PRIMARY_SOURCE", False)
    assert gate(state)["atomic_status"]["A1"] == "insufficient"
    monkeypatch.setattr(config, "GATE_PRIMARY_SOURCE", True)
    out = gate(state)
    assert out["atomic_status"]["A1"] == "supported"
    assert "rbi.org.in" in _joined(out)
    # A tier-1 page that is not the body named in the claim does not count.
    fare = "The fare is ten rupees."
    other = [_row(1, "pib.gov.in", "supports", 1, host="pib.gov.in", quote=fare)]
    assert gate(_about(fare, other))["atomic_status"]["A1"] == "insufficient"
    # A tier 1-2 group on the other side blocks the single page.
    opposed = one + [_row(2, "bbc.com", "refutes", 2, quote=quote)]
    blocked = gate(_about(claim, opposed))
    assert blocked["atomic_status"]["A1"] != "supported"
    denial = [_row(1, "rbi.org.in", "refutes", 1, host="rbi.org.in", quote="RBI did not approve the new note.")]
    # The single-denial flag is a different rule. Turn it off so this
    # assertion is about the body-page path.
    monkeypatch.setattr(config, "TIER1_SINGLE_REFUTE", False)
    refuted = gate(_about(claim, denial))
    assert refuted["atomic_status"]["A1"] == "refuted"


def test_shared_source_merges_a_copied_sentence_a_wire_and_a_release(monkeypatch):
    claim = "RBI banned the note."
    long_quote = "The reserve bank banned the old note from circulation on Monday morning"
    monkeypatch.setattr(config, "GATE_SHARED_SOURCE", False)
    pair = [
        _row(1, "bbc.com", "supports", 2, quote=long_quote),
        _row(2, "reuters.com", "supports", 2, quote=long_quote),
    ]
    # Twelve shared words are one copied sentence, so two newsrooms are one group.
    open_gate = gate(_about(long_quote, pair))
    assert open_gate["atomic_status"]["A1"] == "supported"
    monkeypatch.setattr(config, "GATE_SHARED_SOURCE", True)
    merged = gate(_about(long_quote, pair))
    assert merged["atomic_status"]["A1"] == "insufficient"

    wire = [
        _row(1, "bbc.com", "supports", 2, quote=claim),
        _row(2, "reuters.com", "supports", 2, quote=claim),
    ]
    wire[0]["passage"] = "PTI reported the order."
    wire[1]["passage"] = "The copy was credited to PTI."
    monkeypatch.setattr(config, "GATE_SHARED_SOURCE", False)
    assert gate(_about(claim, wire))["atomic_status"]["A1"] == "supported"
    monkeypatch.setattr(config, "GATE_SHARED_SOURCE", True)
    assert gate(_about(claim, wire))["atomic_status"]["A1"] == "insufficient"

    release = [
        _row(1, "bbc.com", "supports", 2, quote=claim),
        _row(2, "reuters.com", "supports", 2, quote=claim),
    ]
    url = "https://pib.gov.in/PressReleasePage.aspx?PRID=999"
    release[0]["passage"] = f"The text is at {url}."
    release[1]["passage"] = f"See {url} for the note."
    assert gate(_about(claim, release))["atomic_status"]["A1"] == "insufficient"


def test_evidence_score_is_asymmetric_and_uses_the_threshold(monkeypatch):
    from misinfo_agent.gate import evidence_score

    claim = "RBI banned the note."
    day = "2024-06-01"
    # The shipped weak-refute flag would decide three tier-3 denials
    # before theta is read. This case is the score, so that path stays off.
    monkeypatch.setattr(config, "GATE_WEAK_REFUTE", False)
    monkeypatch.setattr(config, "GATE_EVIDENCE_SCORE", True)
    monkeypatch.setattr(config, "GATE_EVIDENCE_THETA", 0.8)
    lower = [
        _dated(_row(1, "a.example", "refutes", 3, quote=claim), 1.0, day, claim),
        _dated(_row(2, "b.example", "refutes", 3, quote=claim), 1.0, day, claim),
        _dated(_row(3, "c.example", "refutes", 3, quote=claim), 1.0, day, claim),
    ]
    state = _about(claim, lower)
    state["claim_date"] = day
    score = evidence_score(state, lower, claim)
    assert score == pytest.approx(-1.05, abs=1e-9)
    assert gate(state)["atomic_status"]["A1"] == "refuted"
    monkeypatch.setattr(config, "GATE_EVIDENCE_THETA", 2.0)
    assert gate(state)["atomic_status"]["A1"] == "insufficient"
    # Three tier-3 supports clear the number and still cannot confirm.
    monkeypatch.setattr(config, "GATE_EVIDENCE_THETA", 0.8)
    supports = [
        _dated(_row(1, "a.example", "supports", 3, quote=claim), 1.0, day, claim),
        _dated(_row(2, "b.example", "supports", 3, quote=claim), 1.0, day, claim),
        _dated(_row(3, "c.example", "supports", 3, quote=claim), 1.0, day, claim),
    ]
    held = _about(claim, supports)
    held["claim_date"] = day
    assert gate(held)["atomic_status"]["A1"] == "insufficient"
    supports.append(_dated(_row(4, "bbc.com", "supports", 2, quote=claim), 1.0, day, claim))
    opened = _about(claim, supports)
    opened["claim_date"] = day
    assert gate(opened)["atomic_status"]["A1"] == "supported"


def test_gate_does_not_import_a_client(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("gate tried to build a client")

    monkeypatch.setattr("misinfo_agent.llm._client", _boom)
    monkeypatch.setattr("misinfo_agent.search._client", _boom)
    out = gate(_state(evidence=[_row(1, "bbc.com", "refutes", 2)]))
    assert out["gate"] in {"search_more", "insufficient", "sufficient", "conflicting", "error", "skipped"}


def test_any_support_blocks_a_single_tier1_denial():
    # A blog is support. The single-office rule does not get to ignore it.
    claim = "RBI has banned ₹500 notes."
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1, host="rbi.org.in"),
        _row(2, "blogspot.com", "supports", 4, quote=claim),
    ]
    out = gate(_about(claim, evidence))
    assert out["atomic_status"]["A1"] != "refuted"
    assert "single official denial" not in _joined(out)


def test_primary_refute_stops_when_any_page_supports(monkeypatch):
    monkeypatch.setattr(config, "TIER1_SINGLE_REFUTE", False)
    claim = "RBI approved the new note."
    evidence = [
        _row(1, "rbi.org.in", "refutes", 1, host="rbi.org.in", quote="RBI did not approve the new note."),
        _row(2, "blogspot.com", "supports", 4, quote=claim),
    ]
    out = gate(_about(claim, evidence))
    assert out["atomic_status"]["A1"] != "refuted"


def test_dated_recent_tier1_denial_with_no_support_still_refutes():
    claim = "India has 29 states in total."
    row = _row(1, "mha.gov.in", "refutes", 1, host="mha.gov.in", quote="India has 28 states.")
    row["published_date"] = "2026-11-23"
    out = gate(_about(claim, [row]))
    assert out["atomic_status"]["A1"] == "refuted"
    assert out["gate"] == "sufficient"
    assert "single official denial: mha.gov.in" in _joined(out)
    assert "outdated" not in _joined(out)


def test_undated_count_page_cannot_decide_alone():
    claim = "India has 29 states in total."
    row = _row(1, "mha.gov.in", "refutes", 1, host="mha.gov.in", quote="India has 28 states.")
    out = gate(_about(claim, [row]))
    assert out["atomic_status"]["A1"] != "refuted"
    assert "official page may be outdated: mha.gov.in, undated" in _joined(out)


def test_older_official_page_does_not_outvote_a_newer_row():
    claim = "India has 29 states in total."
    denial = _row(1, "mha.gov.in", "refutes", 1, host="mha.gov.in", quote="India has 28 states.")
    denial["published_date"] = "2019-01-01"
    blog = _row(2, "blogspot.com", "supports", 4, quote=claim)
    blog["published_date"] = "2026-11-01"
    out = gate(_about(claim, [denial, blog]))
    assert out["atomic_status"]["A1"] != "refuted"
    assert "official page may be outdated: mha.gov.in, 2019-01-01" in _joined(out)


def test_stale_office_against_newsrooms_conflicts_and_the_card_shows_both_leans():
    from misinfo_agent.output import render

    claim = "India has 29 states in total."
    denial = _row(1, "mha.gov.in", "refutes", 1, host="mha.gov.in", quote="India has 28 states.")
    bbc = _row(2, "bbc.com", "supports", 2, quote=claim)
    bbc["published_date"] = "2026-11-01"
    reuters = _row(3, "reuters.com", "supports", 2, quote=claim)
    reuters["published_date"] = "2026-11-01"
    state = _about(claim, [denial, bbc, reuters])
    out = gate(state)
    assert out["atomic_status"]["A1"] == "conflicting"
    assert "official page may be outdated: mha.gov.in, undated" in _joined(out)
    card = render({
        **state,
        **out,
        "verdict": "Unverifiable",
        "unverifiable_reason": "conflicting_credible_sources",
        "explanation": "",
    })["result"]
    assert "mha.gov.in" in card["explanation"]
    assert "bbc.com" in card["explanation"]
    assert "reuters.com" in card["explanation"]
    assert card["leans"]["refutes"] == ["mha.gov.in"]
    assert card["leans"]["supports"] == ["bbc.com", "reuters.com"]


def test_stored_f01_is_not_false_and_f02_stays_refuted():
    path = Path(__file__).resolve().parents[1] / "data" / "eval" / "results_facts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    f01 = gate(by_id["f01"]["state"])
    assert f01["atomic_status"]["A1"] != "refuted"
    assert "official page may be outdated: interstatecouncil.mha.gov.in, undated" in _joined(f01)
    f02 = gate(by_id["f02"]["state"])
    assert f02["atomic_status"]["A1"] == "refuted"
