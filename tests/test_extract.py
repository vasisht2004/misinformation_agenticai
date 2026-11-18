"""Boxes 3–4 with a fake model. No Gemini call.

The fake's return still goes through Pydantic, so a bad domain is a
schema miss and call_structured asks again. The guards after that
(grounding, style flags, Other) are this box, not the model.
"""

from misinfo_agent.extract import extract
from misinfo_agent.llm import LLMQuotaError, clear_fake, set_fake

_TEXT = (
    "RBI has announced that all ₹500 notes printed before 2016 "
    "will be invalid from 1 December."
)


def _claim(**overrides):
    payload = {
        "checkable": True,
        "not_checkable_reason": "",
        "claim": "RBI has announced that all ₹500 notes printed before 2016 will be invalid from 1 December.",
        "atomic_claims": [
            {
                "id": "A1",
                "text": "RBI has announced that all ₹500 notes printed before 2016 will be invalid from 1 December.",
                "core": True,
            }
        ],
        "entities": {
            "people": [],
            "orgs": ["RBI"],
            "places": [],
            "dates": ["1 December"],
            "amounts": ["₹500"],
        },
        "domain": "Finance & scams",
        "red_flags": ["Fake or vague authority"],
        "headline_body_mismatch": "",
        "visual_description": "",
        "location_clues": "",
        "time_clues": "",
    }
    payload.update(overrides)
    return payload


def _state(**overrides):
    state = {
        "input_type": "text",
        "text": _TEXT,
        "checkable_hint": True,
        "style_signals": {
            "urgency_words": [],
            "share_request": False,
            "forwarded_label": None,
            "caps_ratio": 0.06,
            "exclaim_count": 0,
            "emoji_count": 0,
            "authority_mentions": ["RBI"],
            "conspiracy_phrases": [],
            "certainty_words": [],
        },
    }
    state.update(overrides)
    return state


def test_schema_accepted_and_one_core():
    set_fake(lambda prompt, schema, images: _claim())
    try:
        out = extract(_state())
    finally:
        clear_fake()
    assert out["checkable"] is True
    assert out["domain"] == "Finance & scams"
    assert out["claim"].startswith("RBI has announced")
    cores = [item for item in out["atomic_claims"] if item["core"]]
    assert len(out["atomic_claims"]) == 1
    assert len(cores) == 1
    assert cores[0]["id"] == "A1"
    assert "Fake or vague authority" in out["red_flags"]


def test_bad_domain_is_retried():
    calls = []

    def fake(prompt, schema, images):
        calls.append(schema.__name__)
        if len(calls) == 1:
            return _claim(domain="Medicine")
        return _claim()

    set_fake(fake)
    try:
        out = extract(_state())
    finally:
        clear_fake()
    assert len(calls) == 2
    assert out["domain"] == "Finance & scams"


def test_two_cores_are_retried_then_one_remains():
    calls = []

    def fake(prompt, schema, images):
        calls.append(1)
        if len(calls) == 1:
            return _claim(
                atomic_claims=[
                    {"id": "A1", "text": "RBI announced the deadline.", "core": True},
                    {"id": "A2", "text": "Notes printed before 2016 are named.", "core": True},
                ]
            )
        return _claim(
            atomic_claims=[
                {
                    "id": "A1",
                    "text": "RBI has announced that ₹500 notes will be invalid from 1 December.",
                    "core": True,
                },
                {
                    "id": "A2",
                    "text": "The notes were printed before 2016.",
                    "core": False,
                },
            ]
        )

    set_fake(fake)
    try:
        out = extract(_state())
    finally:
        clear_fake()
    assert len(calls) == 2
    assert sum(1 for item in out["atomic_claims"] if item["core"]) == 1
    assert [item["id"] for item in out["atomic_claims"]] == ["A1", "A2"]


def test_a_motive_atomic_claim_is_not_checkable():
    # The model marked the conspiracy true. The code corrects it, because
    # "hiding it" is not an event box 7 can search.
    set_fake(lambda prompt, schema, images: _claim(
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
                "checkable": True,
            },
        ],
        claim="Drinking raw garlic juice cures Type 2 diabetes in 30 days.",
        domain="Health",
    ))
    try:
        out = extract(_state(text=(
            "AIIMS Delhi doctors have confirmed that drinking raw garlic juice "
            "cures Type 2 diabetes in 30 days. The government is hiding it to "
            "protect pharma companies."
        )))
    finally:
        clear_fake()
    by_id = {item["id"]: item for item in out["atomic_claims"]}
    assert by_id["A1"]["checkable"] is True
    assert by_id["A2"]["checkable"] is False
    assert any("A2" in note and "motive" in note for note in out["extract_notes"])


def test_ungrounded_entity_is_dropped():
    set_fake(
        lambda prompt, schema, images: _claim(
            entities={
                "people": ["Zelda"],
                "orgs": ["RBI"],
                "places": ["Narnia"],
                "dates": ["1 December"],
                "amounts": ["₹500"],
            }
        )
    )
    try:
        out = extract(_state())
    finally:
        clear_fake()
    assert out["entities"]["orgs"] == ["RBI"]
    assert out["entities"]["people"] == []
    assert out["entities"]["places"] == []
    assert out["entities"]["amounts"] == ["₹500"]
    notes = " ".join(out["extract_notes"])
    assert "Zelda" in notes
    assert "Narnia" in notes


def test_style_signals_merge_into_red_flags():
    set_fake(lambda prompt, schema, images: _claim(red_flags=["Fear or outrage"]))
    try:
        out = extract(
            _state(
                style_signals={
                    "urgency_words": ["breaking"],
                    "share_request": True,
                    "forwarded_label": "Forwarded as received",
                    "caps_ratio": 0.4,
                    "exclaim_count": 2,
                    "emoji_count": 0,
                    "authority_mentions": ["AIIMS"],
                    "conspiracy_phrases": ["hiding"],
                    "certainty_words": ["confirmed", "cures"],
                }
            )
        )
    finally:
        clear_fake()
    assert out["red_flags"] == [
        "Urgency",
        "Fear or outrage",
        "Fake or vague authority",
        "Miracle or absolute claim",
        "Call to forward",
        "Sensational style",
        "Unverifiable source",
    ]
    assert "conspiracy: hiding" in out["style_flags"]
    assert "certainty: confirmed" in out["style_flags"]


def test_non_checkable_is_other_and_skips_the_model_when_ingest_already_said_so():
    set_fake(
        lambda prompt, schema, images: {
            "checkable": False,
            "not_checkable_reason": "joke",
        }
    )
    try:
        out = extract(_state(text="Why did the chicken cross the road this morning in Delhi?"))
    finally:
        clear_fake()
    assert out["verdict"] == "Other"
    assert out["checkable"] is False
    assert out["not_checkable_reason"] == "joke"
    assert out["atomic_claims"] == []

    calls = []

    def fake(prompt, schema, images):
        calls.append(prompt)
        return _claim()

    set_fake(fake)
    try:
        skipped = extract(
            {
                "checkable_hint": False,
                "checkable_reason": "too short to check",
                "text": "Good morning",
                "style_signals": {},
            }
        )
    finally:
        clear_fake()
    assert calls == []
    assert skipped["verdict"] == "Other"
    assert skipped["not_checkable_reason"] == "too short to check"


def test_image_bytes_are_loaded_from_the_path(tmp_path):
    jpeg = tmp_path / "shot.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9not-a-real-photo-but-the-bytes")
    seen = {}

    def fake(prompt, schema, images):
        seen["images"] = images
        seen["prompt"] = prompt
        return _claim(
            domain="Society & other",
            claim="All Bengaluru schools are closed till Monday due to heavy rain.",
            atomic_claims=[
                {
                    "id": "A1",
                    "text": "All Bengaluru schools are closed till Monday due to heavy rain.",
                    "core": True,
                }
            ],
            entities={"people": [], "orgs": [], "places": ["Bengaluru"], "dates": ["Monday"], "amounts": []},
            visual_description="A white notice with black text.",
            location_clues="no place visible",
            time_clues="no date visible",
        )

    set_fake(fake)
    try:
        out = extract(
            {
                "input_type": "image",
                "text": "All Bengaluru schools closed till Monday due to heavy rain",
                "ocr_text": "All Bengaluru schools closed till Monday due to heavy rain",
                "image_path": str(jpeg),
                "checkable_hint": True,
                "style_signals": {},
            }
        )
    finally:
        clear_fake()
    assert seen["images"] == [jpeg.read_bytes()]
    assert "image" in seen["prompt"].lower()
    assert out["visual_description"] == "A white notice with black text."
    assert out["location_clues"] == "no place visible"
    assert out["time_clues"] == "no date visible"
    assert out["entities"]["places"] == ["Bengaluru"]


def test_quota_error_propagates():
    def fake(prompt, schema, images):
        raise LLMQuotaError("extract", "daily cap")

    set_fake(fake)
    try:
        try:
            extract(_state())
            raised = False
        except LLMQuotaError as exc:
            raised = True
            assert exc.purpose == "extract"
        assert raised
    finally:
        clear_fake()
