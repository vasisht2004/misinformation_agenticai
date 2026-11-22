"""The facts-set rule and the budget. No search and no model call.

Loading the runner does not start a claim. The live path is not what
these tests exercise.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "run_facts"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "run_facts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


facts = _load()


def _metric(name, order, correct2, decided, wrong3=0):
    return {
        "name": name,
        "order": order,
        "correct2": correct2,
        "decided": decided,
        "correct3": decided - wrong3,
        "wrong": wrong3,
    }


def test_registered_file_has_the_sixteen_claims_dated_today():
    rows = facts.load_facts()
    assert [row["id"] for row in rows] == [f"f{i:02d}" for i in range(1, 17)]
    assert rows[0]["label"] == "True"
    assert rows[1]["label"] == "Misleading"
    assert rows[1]["claim"] == "India has 29 states in total."
    assert all(row["claim_date"] == date.today().isoformat() for row in rows)
    labels = {row["id"]: row["label"] for row in rows}
    assert labels["f04"] == "False"
    assert labels["f16"] == "False"
    assert labels["f15"] == "True"


def test_rule_text_is_on_disk():
    text = (ROOT / "docs" / "eval_diagnosis.md").read_text(encoding="utf-8")
    assert "## ESTABLISHED-FACTS SET" in text
    assert facts.RULE in text
    assert "f02" in text and "Misleading" in text


def test_budget_stops_when_the_counter_is_already_at_the_cap():
    assert facts.FACTS_SEARCH_BUDGET == 990
    assert facts.claim_may_start(989) is True
    assert facts.claim_may_start(990) is False
    assert facts.claim_may_start(991) is False


def test_winner_needs_more_correct_and_no_extra_wrong():
    g0 = _metric("G0", 0, correct2=4, decided=5, wrong3=1)
    better = _metric("G2R", 2, correct2=6, decided=7, wrong3=1)
    extra_wrong = _metric("G1", 1, correct2=6, decided=8, wrong3=2)
    same = _metric("G3", 3, correct2=4, decided=4, wrong3=0)
    assert facts.choose_facts_winner([g0, extra_wrong, better, same], set()) == "G2R"
    # Same correct count is a tie even if it makes fewer mistakes.
    assert facts.choose_facts_winner([g0, same], set()) is None
    # More correct, and one more 2-class wrong, does not turn a flag on.
    assert facts.choose_facts_winner([g0, extra_wrong], set()) is None


def test_tie_between_variants_keeps_the_earlier_one():
    g0 = _metric("G0", 0, correct2=4, decided=5, wrong3=1)
    first = _metric("G2R", 2, correct2=6, decided=7, wrong3=1)
    combo = _metric("G2R+G3", 5, correct2=6, decided=7, wrong3=1)
    assert facts.choose_facts_winner([g0, combo, first], set()) == "G2R"


def test_tier34_support_blocks_an_otherwise_better_variant():
    g0 = _metric("G0", 0, correct2=4, decided=5, wrong3=1)
    better = _metric("G2R", 2, correct2=6, decided=7, wrong3=1)
    assert facts.choose_facts_winner([g0, better], {"G2R"}) is None


def test_tier34_helper_flags_a_support_with_no_credible_group(monkeypatch):
    import misinfo_agent.config as config

    # The helper is about the tier, not about whether the quote repeats
    # the core's words. Leave those rows in the count.
    monkeypatch.setattr(config, "TRUE_NEEDS_CORE_MATCH", False)
    monkeypatch.setattr(config, "GATE_SHARED_SOURCE", False)
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)
    state = {
        "route": "investigate",
        "claim": "Example happened in 1935.",
        "claim_date": "2026-11-23",
        "atomic_claims": [{
            "id": "A1",
            "text": "Example happened in 1935.",
            "core": True,
            "checkable": True,
        }],
        "evidence": [
            {
                "id": "E1",
                "atomic_id": "A1",
                "stance": "supports",
                "tier": 4,
                "group_id": "G1",
                "quote": "Example happened in 1935.",
                "url": "https://a.example/x",
            },
            {
                "id": "E2",
                "atomic_id": "A1",
                "stance": "supports",
                "tier": 3,
                "group_id": "G2",
                "quote": "Example happened in 1935.",
                "url": "https://b.example/y",
            },
        ],
    }
    gated = {"gate": "sufficient", "atomic_status": {"A1": "supported"}}
    assert facts.support_from_tier34_only(state, gated) is True
    state["evidence"][0]["tier"] = 1
    assert facts.support_from_tier34_only(state, gated) is False
    state["route"] = "reuse_prior"
    state["evidence"][0]["tier"] = 4
    assert facts.support_from_tier34_only(state, gated) is False


def test_product_settings_leave_the_gate_flags_off(monkeypatch):
    import misinfo_agent.config as config

    # Save the process defaults first. apply_product writes over them,
    # and a later test in this process still expects the file's values.
    for name in (
        "GATE_PRIMARY_SOURCE",
        "GATE_WEAK_REFUTE",
        "GATE_SHARED_SOURCE",
        "GATE_EVIDENCE_SCORE",
        "BLOCK_FACTCHECK_SITES",
        "EVAL_DROP_AFTER_CLAIM",
        "RECORD_SUBMISSIONS",
        "TAVILY_MONTHLY_MAX",
        "LLM_CACHE",
    ):
        monkeypatch.setattr(config, name, getattr(config, name))
    config.GATE_PRIMARY_SOURCE = True
    config.GATE_WEAK_REFUTE = True
    config.GATE_SHARED_SOURCE = True
    config.GATE_EVIDENCE_SCORE = True
    config.BLOCK_FACTCHECK_SITES = True
    config.EVAL_DROP_AFTER_CLAIM = True
    facts.apply_product()
    assert config.BLOCK_FACTCHECK_SITES is False
    assert config.EVAL_DROP_AFTER_CLAIM is False
    assert config.GATE_PRIMARY_SOURCE is False
    assert config.GATE_WEAK_REFUTE is False
    assert config.GATE_SHARED_SOURCE is False
    assert config.GATE_EVIDENCE_SCORE is False
    assert config.TAVILY_MONTHLY_MAX == 990
    assert config.RECORD_SUBMISSIONS is False


def test_official_hosts_are_tier_1():
    # These offices are not all under a .gov suffix. rbi.org.in and
    # npci.org.in were the ones a stored page left at tier 4. Each host
    # the facts diagnosis named has to resolve to tier 1 on its own URL.
    from misinfo_agent.investigate import credibility_tier

    hosts = [
        "rbi.org.in",
        "npci.org.in",
        "uidai.gov.in",
        "isro.gov.in",
        "rajbhasha.gov.in",
        "legislative.gov.in",
        "indiacode.nic.in",
        "eci.gov.in",
        "mea.gov.in",
        "sebi.gov.in",
        "sansad.in",
    ]
    for host in hosts:
        assert credibility_tier(f"https://{host}/about") == 1, host
