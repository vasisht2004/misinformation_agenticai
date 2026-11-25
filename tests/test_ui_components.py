"""The card HTML stays safe when the text is hostile or the new fields are missing.

The page renders with unsafe_allow_html. A passage is evidence, so a
tag inside it has to become text. A javascript: URL must not become an
href. The engine is adding card fields in parallel; a card from before
that merge still has to render.
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from app.components import (
    APP_NAME,
    IMAGE_BYTE_LIMIT,
    card_html,
    esc,
    evidence_html,
    image_prompt_html,
    image_too_large,
    load_fixture,
    quota_html,
    safe_href,
    search_banner_text,
    stepper_html,
    view_model,
    vision_banner_text,
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
    assert model["input_type"] == ""
    assert model["caption"] == ""
    assert model["image_path"] == ""
    assert model["refute_kind"] == ""
    assert model["image_provenance"] is None
    assert model["image_consistency"] is None
    assert model["image_memory_hit"] is None
    assert "Image check" not in html


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


def _card(name: str) -> str:
    return card_html(view_model(load_fixture(name)))


def _timeline(html: str) -> str:
    chunk = html.split("<div class='fl-timeline'>", 1)[1]
    # Undated rows and similar images sit under the dated line.
    for marker in ("<div class='fl-undated'>", "<section class='fl-similar'>"):
        if marker in chunk:
            chunk = chunk.split(marker, 1)[0]
    return chunk


def test_placeholder_jpeg_is_generated_and_has_no_exif():
    path = Path("app/fixtures/placeholder.jpg")
    raw = path.read_bytes()
    assert raw.startswith(b"\xff\xd8\xff")
    image = Image.open(path)
    assert image.size == (960, 600)
    # JPEG quantises the flat navy. The corner stays that field, not a photo.
    red, green, blue = image.getpixel((0, 0))[:3]
    assert abs(red - 15) <= 2
    assert abs(green - 27) <= 2
    assert abs(blue - 54) <= 2
    assert not image.getexif()


def test_image_out_of_context_card():
    html = _card("image_out_of_context")
    assert "fl-verdict-word is-context" in html
    assert "MISLEADING · OUT OF CONTEXT" in html
    assert "color: #E8A33D" in html
    assert "fl-ring" in html
    assert "Image check" in html
    assert "data:image/jpeg;base64," in html
    encoded = base64.b64encode(Path("app/fixtures/placeholder.jpg").read_bytes()).decode("ascii")
    assert encoded in html
    assert "Flooding in Mumbai this morning" in html
    assert "What the AI sees" in html
    assert "hill town" in html
    assert "snow on roofs" in html
    assert "winter light" in html
    assert "Snow on a hill town" in html
    assert "fl-cap-no" in html
    assert "The caption does not match the image." in html
    assert "Place: a snowy hill town vs flooded streets in Mumbai" in html
    assert "Time: winter light and bare trees vs November 2026 monsoon flooding" in html
    assert "AI observation — not a verdict" in html
    assert "First seen online" in html
    assert "12 Mar 2019" in html
    assert "hillwire.example" in html
    assert "T3" in html
    assert "https://hillwire.example/2019/03/snow-on-the-ridge" in html
    assert 'rel="noopener noreferrer"' in html
    timeline = _timeline(html)
    assert "12 Mar 2019" in timeline
    assert "2 Nov 2024" not in timeline
    assert "12 Nov 2026" not in timeline
    assert timeline.count("is-first") == 1
    assert "fl-tl-point is-neutral is-first" in timeline
    assert "Full match" in timeline
    assert "Partial match" not in timeline
    similar = html.split("<section class='fl-similar'>", 1)[1]
    assert "Similar images" in similar
    assert similar.index("2 Nov 2024") < similar.index("12 Nov 2026")
    assert "is-refutes" in similar
    assert "is-supports" in similar
    assert "Partial match" in similar
    assert "fl-undated" in html
    assert "bad.example" in html
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "javascript:" not in html.lower()
    seen = html.split("fl-image-seen", 1)[1][:500]
    assert ">Misleading<" in seen
    assert "12 Mar 2019" in seen
    assert "0.07" in seen
    assert "img-thread-2019" not in html
    first_rule = CSS.split(".fl-tl-point.is-first .fl-tl-dot")[1].split("}")[0]
    refute_rule = CSS.split(".fl-tl-point.is-refutes .fl-tl-dot")[1].split("}")[0]
    support_rule = CSS.split(".fl-tl-point.is-supports .fl-tl-dot")[1].split("}")[0]
    assert "#E05A47" in first_rule
    assert "18px" in first_rule
    assert "#E05A47" in refute_rule
    assert "#3FB27F" in support_rule
    assert CSS.index(".fl-tl-point.is-supports .fl-tl-dot") < CSS.index(".fl-tl-point.is-first .fl-tl-dot")


def test_image_genuine_card():
    html = _card("image_genuine")
    assert ">True</h2>" in html
    assert "OUT OF CONTEXT" not in html
    assert "fl-cap-yes" in html
    assert "fl-cap-no" not in html
    assert "The caption matches what the image shows." in html
    assert "AI observation — not a verdict" in html
    assert "river front" in html
    assert "June light" in html
    assert "A glass building beside a river" in html
    assert "First seen online" in html
    assert "4 Jun 2024" in html
    assert "alderport.example" in html
    assert "T1" in html
    assert "https://alderport.example/news/library-opens" in html
    timeline = _timeline(html)
    assert timeline.index("4 Jun 2024") < timeline.index("5 Jun 2024")
    assert "fl-tl-point is-supports is-first" in timeline
    assert timeline.count("is-first") == 1
    assert "Full match" in timeline
    assert "fl-image-seen" not in html
    assert "data:image/jpeg;base64," in html
    assert "Image check" in html


def test_image_no_caption_card():
    html = _card("image_no_caption")
    assert ">Other</h2>" in html
    assert "Image check" in html
    assert "fl-thumb-missing" in html
    assert "Image not stored" in html
    assert "data:image" not in html
    assert "No caption supplied." in html
    assert "fl-cap-unclear" in html
    assert "It is unclear whether the caption matches the image." in html
    assert "AI observation — not a verdict" in html
    assert "indoor hall" in html
    assert "First seen online" in html
    assert "19 Aug 2021" in html
    assert "archive.example" in html
    assert "https://archive.example/2021/08/hall-photo" in html
    assert "T4" in html
    assert "Undated matches" in html
    assert "http://notes.example/same-hall" in html
    assert "Same hall, no date on the page." in html
    assert "Partial match" in html
    timeline = _timeline(html)
    assert "19 Aug 2021" in timeline
    assert "notes.example" not in timeline


def test_image_no_key_and_vision_banners():
    html = _card("image_no_key")
    banner = "Reverse image search isn't set up"
    assert vision_banner_text(view_model(load_fixture("image_no_key"))["image_provenance"]) == banner
    assert esc(banner) in html
    assert "AIzaSyFAKEKEYSHOULDNOTRENDER" not in html
    assert "api_key" not in html
    assert "no_key" not in html
    assert "First seen online" not in html
    assert "Should not list" not in html
    assert "hidden observed" not in html
    assert "fl-cap-unclear" in html
    assert "A bridge in daylight" in html
    assert "a bridge" in html
    assert "daytime" in html
    assert "Text sources cannot confirm what a picture shows." in html
    assert "data:image/jpeg;base64," in html
    cases = {
        "no_key": "Reverse image search isn't set up",
        "monthly_cap": "Reverse image search limit reached this month",
        "error": "Reverse image search failed; the result uses other evidence only",
    }
    for status, sentence in cases.items():
        rendered = card_html(view_model({
            "input_type": "image",
            "caption": "A caption",
            "image_provenance": {
                "vision_status": status,
                "api_key": "SECRET-KEY-99",
                "token": "SECRET-KEY-99",
            },
        }))
        assert vision_banner_text({"vision_status": status}) == sentence
        assert esc(sentence) in rendered
        assert "SECRET-KEY-99" not in rendered
    stuffed = card_html(view_model({
        "input_type": "image",
        "image_provenance": {"vision_status": "error api_key=SECRET-KEY-99"},
    }))
    assert esc(cases["error"]) in stuffed
    assert "SECRET-KEY-99" not in stuffed
    assert "api_key" not in stuffed
    quiet = card_html(view_model({
        "input_type": "image",
        "image_provenance": {"vision_status": "ok", "matches": [], "first_seen": None},
    }))
    assert "Reverse image search" not in quiet
    assert "No earlier copies were found online." in quiet
    assert vision_banner_text({"vision_status": "ok"}) == ""
    assert vision_banner_text(None) == ""


def test_image_section_escapes_and_stays_off_text_cards():
    plain = _card("true")
    assert "Image check" not in plain
    assert "First seen online" not in plain
    misleading = _card("misleading")
    assert "OUT OF CONTEXT" not in misleading
    assert ">Misleading</h2>" in misleading
    kind_only = card_html(view_model({
        "input_type": "text",
        "verdict": "Misleading",
        "confidence": 0.4,
        "atomic_claims": [{"id": "A1", "core": True}],
        "atomic_results": [{"atomic_id": "A1", "refute_kind": "out_of_context"}],
    }))
    assert "MISLEADING · OUT OF CONTEXT" in kind_only
    assert "Image check" not in kind_only
    side = card_html(view_model({
        "verdict": "True",
        "confidence": 0.8,
        "atomic_claims": [{"id": "A1", "core": True}, {"id": "A2", "core": False}],
        "atomic_results": [
            {"atomic_id": "A1", "refute_kind": None},
            {"atomic_id": "A2", "refute_kind": "out_of_context"},
        ],
    }))
    assert "OUT OF CONTEXT" not in side
    assert ">True</h2>" in side
    kept = card_html(view_model({
        "verdict": "True",
        "confidence": 0.8,
        "refute_kind": "out_of_context",
    }))
    assert "OUT OF CONTEXT" not in kept
    hostile = card_html(view_model({
        "input_type": "image",
        "caption": "<b>caption</b>",
        "image_path": "javascript:alert(1)",
        "image_consistency": {
            "consistent": "no",
            "mismatches": [{
                "detail": "<i>place</i>",
                "observed": "<script>",
                "expected": "x",
            }],
            "place_clues": ["<img>"],
        },
        "image_provenance": {
            "vision_status": "ok",
            "matches": [{
                "url": "data:text/html,<script>alert(1)</script>",
                "host": "x.example",
                "quote": "<b>no</b>",
                "date": "2020-01-01",
                "stance": "neutral",
                "match": "full",
                "tier": 4,
            }],
            "first_seen": {
                "date": "2020-01-01",
                "url": "javascript:alert(1)",
                "host": "x.example",
                "tier": 1,
            },
        },
    }))
    assert "<script>" not in hostile
    assert "<b>" not in hostile
    assert "<i>" not in hostile
    assert "<img>" not in hostile
    assert "javascript:" not in hostile.lower()
    assert "data:text/html" not in hostile
    assert "fl-thumb-missing" in hostile
    assert "&lt;script&gt;" in hostile
    assert "&lt;b&gt;caption&lt;/b&gt;" in hostile
    assert "x.example" in hostile
    first = hostile.split("fl-first", 1)[1].split("fl-timeline", 1)[0]
    assert "href=" not in first
    assert "View page" not in first
    sparse = card_html(view_model({
        "input_type": "image",
        "image_provenance": {"matches": [{"stance": "refutes"}]},
    }))
    assert "unknown source" in sparse
    assert "Undated matches" in sparse
    assert "Image check" in card_html(view_model({"input_type": "image", "verdict": "Other"}))


def test_image_tab_copy_keeps_the_ten_megabyte_limit():
    page = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "image_prompt_html" in page
    assert "image_too_large" in page
    assert "image_check_html" in page
    html = image_prompt_html()
    sentence = (
        "Upload a photo of an event (optional caption). "
        "We check where it first appeared online."
    )
    assert sentence in html
    assert "10 MB" in html
    assert image_too_large(IMAGE_BYTE_LIMIT) is False
    assert image_too_large(IMAGE_BYTE_LIMIT + 1) is True
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert "maxUploadSize = 10" in config
