"""The assisted lean is a field. It is not a verdict."""

import pytest

import misinfo_agent.config as config
from misinfo_agent.lean import assisted_lean
from misinfo_agent.llm import LLMQuotaError, set_fake
from misinfo_agent.output import render


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_fake(None)


def _fake(label, confidence=0.4):
    def answer(prompt, schema, images):
        assert schema.__name__ == "BaselineVerdict"
        assert "Judge this claim." in prompt
        return {"verdict": label, "confidence": confidence}

    set_fake(answer)


def test_lean_does_not_change_an_unverifiable_decision(monkeypatch):
    monkeypatch.setattr(config, "ASSISTED_LEAN", True)
    _fake("False", 0.2)
    state = {
        "claim": "The minister banned the notes.",
        "verdict": "Unverifiable",
        "confidence": 0.55,
        "priority": "High",
        "decision": "human",
        "unverifiable_reason": "conflicting_credible_sources",
    }
    extra = assisted_lean(state)
    merged = {**state, **extra}
    card = render(merged)["result"]
    assert extra["model_lean"] == {"label": "False", "confidence": 0.2}
    assert merged["verdict"] == "Unverifiable"
    assert merged["confidence"] == 0.55
    assert merged["priority"] == "High"
    assert merged["decision"] == "human"
    assert card["verdict"] == "Unverifiable"
    assert card["model_lean"]["label"] == "False"
    assert card["confidence"] == 0.55


def test_lean_is_absent_when_the_verdict_is_decided_or_other(monkeypatch):
    monkeypatch.setattr(config, "ASSISTED_LEAN", True)
    called = {"n": 0}

    def boom(prompt, schema, images):
        called["n"] += 1
        raise AssertionError("lean asked for a decided claim")

    set_fake(boom)
    for verdict in ("False", "Misleading", "True", "Other"):
        assert assisted_lean({"claim": "A claim.", "verdict": verdict}) == {}
    assert called["n"] == 0


def test_an_unverifiable_lean_is_dropped(monkeypatch):
    monkeypatch.setattr(config, "ASSISTED_LEAN", True)
    _fake("Unverifiable", 0.3)
    extra = assisted_lean({"claim": "A claim.", "verdict": "Unverifiable"})
    assert extra == {"model_lean": None}


def test_quota_errors_propagate(monkeypatch):
    monkeypatch.setattr(config, "ASSISTED_LEAN", True)

    def capped(prompt, schema, images):
        raise LLMQuotaError("eval_baseline", "daily cap")

    set_fake(capped)
    with pytest.raises(LLMQuotaError):
        assisted_lean({"claim": "A claim.", "verdict": "Unverifiable"})
