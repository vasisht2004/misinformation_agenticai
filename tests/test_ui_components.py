"""The card HTML stays safe when the text is hostile or the new fields are missing.

The page renders with unsafe_allow_html. A passage is evidence, so a
tag inside it has to become text. A javascript: URL must not become an
href. The engine is adding card fields in parallel; a card from before
that merge still has to render.
"""

from __future__ import annotations

from app.components import (
    APP_NAME,
    card_html,
    evidence_html,
    load_fixture,
    safe_href,
    stepper_html,
    view_model,
)

_HOSTILE_ROW = {
    "id": "E1",
    "url": "https://example.com/a",
    "netloc": "example.com",
    "tier": 2,
    "stance": "refutes",
    "quote": "quoted",
    "passage": "See <script>alert(1)</script> now",
    "group_id": "example.com",
    "date": "2024-01-02",
}


def test_script_in_passage_is_escaped():
    html = evidence_html([_HOSTILE_ROW])
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert 'rel="noopener noreferrer"' in html


def test_javascript_url_is_dropped():
    html = evidence_html([{
        "id": "E2",
        "url": "javascript:alert(1)",
        "netloc": "bad.example",
        "tier": 4,
        "stance": "neutral",
        "quote": "x",
        "passage": "x is here",
        "group_id": "g",
        "date": "",
    }])
    assert "javascript:" not in html.lower()
    assert "href=" not in html
    assert safe_href("javascript:alert(1)") is None
    assert safe_href("JavaScript:alert(1)") is None
    assert safe_href("https://example.com/a") == "https://example.com/a"


def test_missing_new_fields_do_not_crash():
    model = view_model({
        "verdict": "Misleading",
        "claim": "A claim <b>x</b>",
        "confidence": 0.4,
    })
    html = card_html(model)
    assert "Misleading" in html
    assert "<b>" not in html
    assert "&lt;b&gt;" in html
    assert model["provisional"] is False
    assert model["model_lean"] is None
    assert model["alert_pending_approval"] is False
    assert model["core_detail_only_ids"] == []
    assert model["unverifiable_reason"] == ""
    assert model["review_status"] == ""
    assert model["review_reason"] == ""


def test_fixtures_cover_the_named_cards():
    names = (
        "false_alert",
        "misleading",
        "true",
        "unverifiable_lean",
        "seen_before",
        "conflicting",
        "pipeline",
    )
    for name in names:
        payload = load_fixture(name)
        assert isinstance(payload, dict)
        if payload.get("kind") == "replay":
            html = stepper_html(
                [(item["node"], item["elapsed_s"]) for item in payload["nodes"]],
                payload,
                settled=True,
                replay=True,
            )
            assert "replay" in html
            assert "past fact-check reused — no search needed" in html
            continue
        model = view_model(payload)
        html = card_html(model)
        assert html
        assert "<script>" not in html
    assert APP_NAME == "FACT✦LENS"
    alert = load_fixture("false_alert")
    assert alert.get("priority") == "High"
    assert alert.get("provisional") is True
    assert alert.get("alert_pending_approval") is True
    rendered = card_html(view_model(alert))
    assert "PROVISIONAL · HIGH PRIORITY · AWAITING HUMAN VERIFICATION" in rendered
    assert "Alert held until a reviewer approves" in rendered
    lean = load_fixture("unverifiable_lean")
    assert lean.get("unverifiable_reason") == "no_credible_record"
    assert isinstance(lean.get("model_lean"), dict)
    lean_html = card_html(view_model(lean))
    assert "Model lean (not a verdict)" in lean_html
    assert "Gemini's guess from memory. No sources." in lean_html
    assert "No two independent credible sources were found." in lean_html
    seen = load_fixture("seen_before")
    assert seen.get("route") == "reuse_prior"
    assert "Seen before" in card_html(view_model(seen))
    conflict = view_model(load_fixture("conflicting"))
    assert "Withheld" in card_html(conflict)
