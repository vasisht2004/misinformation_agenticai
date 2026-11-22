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
    esc,
    evidence_html,
    load_fixture,
    quota_html,
    safe_href,
    search_banner_text,
    stepper_html,
    view_model,
)
from app.theme import CSS

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


def test_search_banner_and_no_unread_sentence():
    quiet = {
        "verdict": "Unverifiable",
        "claim": "A claim nobody searched",
        "searches_live": 0,
        "searches_cached": 0,
        "explanation": "No source we read addressed this part of the claim.",
        "what_we_found": "No source we read addressed this part of the claim.",
        "evidence": [],
    }
    monthly_model = view_model({**quiet, "search_status": "monthly_cap"})
    monthly = card_html(monthly_model)
    banner = (
        "Search didn't run: the monthly search limit is reached. "
        "This result is based on past fact-checks only."
    )
    assert search_banner_text(monthly_model) == banner
    assert esc(banner) in monthly
    assert "addressed this" not in monthly.lower()
    missed = card_html(view_model({**quiet, "search_status": "cache_only_miss"}))
    assert "this claim was not in the search cache" in missed
    assert "addressed this" not in missed.lower()
    capped = card_html(view_model({**quiet, "search_status": "per_claim_cap"}))
    assert "this claim reached its search limit" in capped
    failed = card_html(view_model({**quiet, "search_status": "error"}))
    assert "the search failed" in failed
    ran = card_html(view_model({**quiet, "search_status": "ran", "verdict": "False", "confidence": 0.4}))
    assert "Search didn't run" not in ran
    skipped = card_html(view_model({**quiet, "search_status": "not_needed", "verdict": "Other"}))
    assert "Search didn't run" not in skipped


def test_seen_before_only_when_exact():
    exact = card_html(view_model(load_fixture("seen_before")))
    assert "Seen before" in exact
    similar = card_html(view_model({
        "verdict": "Misleading",
        "confidence": 0.7,
        "match_type": "similar",
        "route": "investigate",
        "best_match": {
            "site": "Alt News",
            "url": "https://www.altnews.in/example",
            "primary_claim": "A different ₹500 notice.",
            "dense_cos": 0.91,
            "date": "2016-11-15",
        },
    }))
    assert "Seen before" not in similar
    assert "<details class='fl-related'>" in similar
    assert " open" not in similar.split("fl-related")[1][:80]
    assert "Related past fact-checks" in similar
    assert "0.91" in similar
    none = card_html(view_model({
        "verdict": "Unverifiable",
        "match_type": "none",
        "best_match": {
            "site": "youturn",
            "url": "https://en.youturn.in/factcheck/universities",
            "primary_claim": "India has 5288 universities.",
            "dense_cos": 0.68,
        },
        "seen_before": {
            "site": "youturn",
            "link": "https://en.youturn.in/factcheck/universities",
            "claim": "India has 5288 universities.",
            "cosine": 0.68,
            "match_type": "none",
        },
    }))
    assert "Seen before" not in none
    assert "Related past fact-checks" not in none
    assert "youturn" not in none


def test_unverifiable_has_reason_instead_of_a_ring():
    html = card_html(view_model({
        "verdict": "Unverifiable",
        "confidence": 0.94,
        "unverifiable_reason": "no_credible_record",
    }))
    assert "fl-ring" not in html
    assert "evidence strength" not in html
    assert "No two independent credible sources were found." in html
    for verdict in ("False", "Misleading", "True"):
        decided = card_html(view_model({"verdict": verdict, "confidence": 0.8}))
        assert "fl-ring" in decided
        assert "evidence strength" in decided


def test_evidence_lean_sits_above_the_model_lean():
    rows = [
        {"id": "E1", "url": "https://a.example/1", "netloc": "a.example", "tier": 3, "stance": "refutes", "group_id": "G1"},
        {"id": "E2", "url": "https://a.example/2", "netloc": "a.example", "tier": 3, "stance": "refutes", "group_id": "G1"},
        {"id": "E3", "url": "https://b.example/1", "netloc": "b.example", "tier": 4, "stance": "refutes", "group_id": "G2"},
        {"id": "E4", "url": "https://c.example/1", "netloc": "c.example", "tier": 4, "stance": "refutes", "group_id": "G3"},
    ]
    html = card_html(view_model({
        "verdict": "Unverifiable",
        "evidence": rows,
        "model_lean": {"label": "False", "confidence": 0.4},
    }))
    assert "Sources lean FALSE: 3 independent groups refute, 0 support (all tier 3-4)" in html
    assert html.index("fl-src-lean") < html.index("fl-lean")
    assert "Model lean (not a verdict)" in html
    assert "border: 1px solid" in CSS.split(".fl-src-lean")[1].split("}")[0]
    assert "dashed" in CSS.split(".fl-lean")[1].split("}")[0]
    decided = card_html(view_model({"verdict": "False", "confidence": 0.8, "evidence": rows}))
    assert "Sources lean" not in decided


def test_unverifiable_correction_quotes_what_we_found():
    found = "India is a federal union comprising 28 states and 8 union territories [E1]."
    html = card_html(view_model({
        "verdict": "Unverifiable",
        "explanation": "India actually comprises 28 states rather than 29 states.",
        "what_we_found": found,
        "correction_message": "We could not find two independent credible sources, so this is not confirmed.",
        "evidence": [{
            "id": "E1",
            "url": "https://en.wikipedia.org/wiki/States_and_union_territories_of_India",
            "netloc": "en.wikipedia.org",
            "tier": 3,
            "stance": "refutes",
            "group_id": "wiki",
        }],
    }))
    assert "not confirmed" not in html.lower()
    assert (
        "Not enough credible sources to decide. What the sources we found say: " + found
    ) in html
    assert "28 states" in html


def test_caveat_sits_under_the_verdict_in_amber():
    text = "Lower-tier sources only: no tier 1-2 source was found."
    html = card_html(view_model({
        "verdict": "Unverifiable",
        "unverifiable_reason": "only_tier34",
        "caveats": [text],
    }))
    assert "fl-caveat" in html
    assert text in html
    assert html.index("Unverifiable") < html.index("fl-caveat")
    assert "#E8A33D" in CSS.split(".fl-caveat")[1].split("}")[0]


def test_fast_step_is_a_dash_and_search_chip_warns():
    fast = stepper_html([("priority", 0.02), ("decide", 0.2)], {}, settled=True)
    assert "0.0s" not in fast
    assert "—" in fast
    assert "0.2s" in fast
    warn = quota_html(1, 10, 900, 1000)
    assert "is-warn" in warn
    assert "new claims can&#x27;t search; cached claims still work" in warn
    full = quota_html(1, 10, 1000, 1000)
    assert "is-cap" in full
    assert "is-warn" not in full
    quiet = quota_html(1, 10, 100, 1000)
    assert "is-warn" not in quiet
    assert "is-cap" not in quiet


def test_states_29_card_hides_the_weak_match_and_does_not_contradict_itself():
    html = card_html(view_model(load_fixture("states_29")))
    assert "India has 29 states in total." in html
    assert "Seen before" not in html
    assert "Related past fact-checks" not in html
    assert "youturn" not in html
    assert "fl-ring" not in html
    assert "evidence strength" not in html
    assert "No two independent credible sources were found." in html
    assert "Sources lean FALSE: 3 independent groups refute, 0 support (all tier 3-4)" in html
    assert html.index("fl-src-lean") < html.index("fl-lean")
    assert "not confirmed" not in html.lower()
    assert (
        "Not enough credible sources to decide. What the sources we found say: "
        "India is a federal union comprising 28 states and 8 union territories [E1]."
    ) in html
    assert "https://en.wikipedia.org/wiki/States_and_union_territories_of_India" in html
