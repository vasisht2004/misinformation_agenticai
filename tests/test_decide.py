"""Box 12 on states whose verdict and confidence are already set.

No model and no network. The branches are the rule, not a sample of
evidence: assess is what writes the verdict, and this box only reads it.
"""

import pytest

import misinfo_agent.config as config
from misinfo_agent.decide import _p_star, decide


def _state(**overrides):
    state = {
        "domain": "Health",
        "claim": "A health claim.",
        "red_flags": [],
        "style_signals": {},
        "verdict": "False",
        "confidence": 0.9,
        "priority": "High",
        "gate": "sufficient",
        "rounds": 1,
        "matches": [],
    }
    state.update(overrides)
    return state


def _joined(result) -> str:
    return " ".join(result["decision_reasons"])


def test_p_star_falls_as_harm_rises():
    # a/(a+bH). At 0 the bar is 1. At 1 it is 0.25. The sports domain
    # sits under health, so its bar is higher: a sports rumour has to
    # be more certain before anyone acts.
    assert _p_star(0.0) == pytest.approx(1.0)
    assert _p_star(1.0) == pytest.approx(1.0 / (1.0 + 3.0))
    assert _p_star(1.0) < _p_star(0.5) < _p_star(0.0)
    health = decide(_state(domain="Health", verdict="True", priority="Low"))
    sports = decide(_state(domain="Sports", verdict="True", priority="Low"))
    assert health["p_star"] < sports["p_star"]


def test_gate_error_and_conflict_go_to_a_human():
    # A confident false verdict must not alert when the gate already
    # refused to trust the table.
    for gate in ("error", "conflicting"):
        out = decide(_state(gate=gate, confidence=0.99))
        assert out["decision"] == "human"
        assert "rounds" not in out


def test_false_and_misleading_high_priority_above_the_bar_alert():
    for verdict in ("False", "Misleading"):
        out = decide(_state(verdict=verdict, confidence=0.9, priority="High"))
        assert out["decision"] == "alert"
        assert "box 13" in _joined(out)
        assert "approve" in _joined(out)


def test_alert_pause_follows_the_config_flag(monkeypatch):
    monkeypatch.setattr(config, "ALERT_NEEDS_APPROVAL", False)
    out = decide(_state())
    assert out["decision"] == "alert"
    assert "box 13" not in _joined(out)


def test_high_harm_just_under_the_bar_searches_again():
    # Health with no flag is 0.95, so the bar is about 0.26. 0.20 is
    # under it and inside the 0.15 margin. One round is still left.
    out = decide(_state(confidence=0.20, priority="High", rounds=1))
    assert out["decision"] == "search_more"
    assert out["rounds"] == 2
    assert out["p_star"] > 0.20


def test_high_harm_far_under_the_bar_is_human():
    out = decide(_state(confidence=0.05, priority="High", rounds=1))
    assert out["decision"] == "human"
    assert "rounds" not in out


def test_high_harm_under_the_bar_at_the_search_limit_is_human():
    out = decide(_state(confidence=0.20, priority="High", rounds=config.MAX_SEARCH_ROUNDS))
    assert out["decision"] == "human"
    assert "rounds" not in out


def test_unverifiable_high_harm_is_human_and_low_harm_is_stored():
    high = decide(_state(verdict="Unverifiable", domain="Health", priority="Low"))
    assert high["decision"] == "human"
    low = decide(_state(verdict="Unverifiable", domain="Sports", priority="Low", red_flags=[]))
    assert low["decision"] == "store"


def test_true_and_other_are_stored():
    true = decide(_state(verdict="True", confidence=0.99, priority="High"))
    other = decide(_state(verdict="Other", confidence=0.99, priority="High"))
    assert true["decision"] == "store"
    assert other["decision"] == "store"


def test_false_below_high_priority_is_stored_even_when_confident():
    out = decide(_state(priority="Medium", confidence=0.9))
    assert out["decision"] == "store"


def test_style_does_not_change_the_decision_once_priority_is_set():
    # The level is already on the state. A banner word must not reopen
    # it. Style had its chance in box 11.
    quiet = decide(_state(style_signals={"urgency_words": [], "share_request": False}))
    loud = decide(_state(style_signals={
        "urgency_words": ["breaking", "urgent"],
        "share_request": True,
        "forwarded_label": "Forwarded many times",
    }))
    assert quiet["decision"] == loud["decision"] == "alert"
    assert quiet["p_star"] == loud["p_star"]
