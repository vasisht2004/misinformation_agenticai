"""Offline checks for photograph provenance. No Vision call and no Tavily call."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import imagehash
import numpy as np

import misinfo_agent.config as config
from misinfo_agent.assess import CAPTION_MISMATCH, CAPTION_MISMATCH_FLAG, assess
from misinfo_agent.fetch import Page
from misinfo_agent.gate import gate
from misinfo_agent.image_memory import apply_image_reuse, lookup, remember
from misinfo_agent.investigate import investigate
from misinfo_agent.llm import clear_fake as clear_fake_llm
from misinfo_agent.llm import set_fake as set_fake_llm
from misinfo_agent.provenance import investigate_picture
from misinfo_agent.search import clear_fake_search, set_fake_search
from misinfo_agent.vision import calls_this_month, clear_fake, detect, request_body, set_fake


def _jpeg(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), "blue").save(path, format="JPEG")
    return path


def _claim(text: str, **extra) -> dict:
    state = {
        "claim": text,
        "input_type": "image",
        "checkable": True,
        "rounds": 2,
        "claim_date": "2026-03-10",
        "atomic_claims": [{"id": "A1", "text": text, "core": True, "checkable": True}],
        "evidence": [],
    }
    state.update(extra)
    return state


def _match(**kwargs) -> dict:
    row = {
        "url": "https://example.com/a",
        "host": "example.com",
        "tier": 4,
        "date": "2020-01-01",
        "match": "full",
        "stance": "refutes",
        "quote": "This photograph is from an earlier flood.",
        "group_id": "G1",
    }
    row.update(kwargs)
    return row


def test_vision_without_a_key_does_not_raise(tmp_path):
    found = detect(str(_jpeg(tmp_path)))
    assert found["vision_status"] == "no_key"
    assert found["pages"] == []
    assert calls_this_month() == 0


def test_vision_cap_does_not_call(tmp_path, monkeypatch):
    from misinfo_agent import vision

    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key")
    vision._write_counts({vision._month_key(): config.VISION_MONTHLY_MAX})

    def _post(*args, **kwargs):
        raise AssertionError("the monthly cap must not call Vision")

    monkeypatch.setattr(vision, "_post", _post)
    found = detect(str(_jpeg(tmp_path)))
    assert found["vision_status"] == "monthly_cap"
    assert calls_this_month() == config.VISION_MONTHLY_MAX


def test_vision_cache_hit_is_not_counted(tmp_path, monkeypatch):
    from misinfo_agent import vision

    jpeg = _jpeg(tmp_path)
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key")
    calls = {"n": 0}

    def _post(image, key):
        calls["n"] += 1
        return {
            "responses": [
                {
                    "webDetection": {
                        "bestGuessLabels": [{"label": "flood"}],
                        "pagesWithMatchingImages": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(vision, "_post", _post)
    before = calls_this_month()
    first = detect(str(jpeg))
    assert first["vision_status"] == "ran"
    assert first["best_guess"] == ["flood"]
    assert calls_this_month() == before + 1
    second = detect(str(jpeg))
    assert second["vision_status"] == "cached"
    assert calls_this_month() == before + 1
    assert calls["n"] == 1


def test_web_entities_are_ignored(tmp_path):
    set_fake(
        lambda image: {
            "responses": [
                {
                    "webDetection": {
                        "webEntities": [{"description": "Narendra Modi", "score": 0.9}],
                        "bestGuessLabels": [{"label": "flood"}],
                        "fullMatchingImages": [{"url": "https://cdn.example/a.jpg"}],
                        "pagesWithMatchingImages": [
                            {
                                "url": "https://www.bbc.com/news/kerala-flood",
                                "pageTitle": "Kerala floods",
                                "fullMatchingImages": [{"url": "https://cdn.example/a.jpg"}],
                            }
                        ],
                    }
                }
            ]
        }
    )
    try:
        found = detect(str(_jpeg(tmp_path)))
    finally:
        clear_fake()
    assert found["vision_status"] == "ran"
    assert found["best_guess"] == ["flood"]
    assert found["pages"] == [
        {
            "url": "https://www.bbc.com/news/kerala-flood",
            "title": "Kerala floods",
            "match": "full",
            "image_urls": ["https://cdn.example/a.jpg"],
        }
    ]
    blob = str(found)
    assert "Narendra Modi" not in blob
    assert "webEntities" not in found
    assert calls_this_month() == 0


def test_partial_file_urls_are_not_kept(tmp_path):
    """A partial is a different photograph. Its file URL must not be dated as this one."""
    set_fake(
        lambda image: {
            "responses": [
                {
                    "webDetection": {
                        "pagesWithMatchingImages": [
                            {
                                "url": "https://www.bbc.com/news/kerala-flood",
                                "pageTitle": "Kerala floods",
                                "fullMatchingImages": [{"url": "https://cdn.example/a.jpg"}],
                                "partialMatchingImages": [{"url": "https://cdn.example/other.jpg"}],
                            },
                            {
                                "url": "https://blog.example/lookalike",
                                "pageTitle": "Similar",
                                "partialMatchingImages": [{"url": "https://cdn.example/other.jpg"}],
                            },
                        ]
                    }
                }
            ]
        }
    )
    try:
        found = detect(str(_jpeg(tmp_path)))
    finally:
        clear_fake()
    assert found["pages"][0]["image_urls"] == ["https://cdn.example/a.jpg"]
    assert "image_urls" not in found["pages"][1]
    assert "other.jpg" not in str(found["pages"])


def test_request_asks_only_for_web_detection():
    body = request_body(b"\xff\xd8\xff")
    features = body["requests"][0]["features"]
    assert features == [{"type": "WEB_DETECTION", "maxResults": config.VISION_MAX_RESULTS}]
    assert config.VISION_MAX_RESULTS == 50
    folded = str(body).casefold()
    assert "face" not in folded
    assert "landmark" not in folded
    assert "celebrity" not in folded


def test_out_of_context_with_two_groups_is_misleading():
    state = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": ["flood"],
            "first_seen": {"date": "2019-08-01", "url": "https://example.com/a", "host": "example.com", "tier": 4},
            "matches": [
                _match(group_id="G1", url="https://example.com/a"),
                _match(
                    group_id="G2",
                    url="https://blog.example.net/b",
                    host="blog.example.net",
                    date="2019-05-01",
                ),
            ],
        },
    )
    judged = gate(state)
    assert judged["atomic_status"]["A1"] == "refuted"
    assert judged["gate"] == "sufficient"
    result = assess({**state, **judged})
    assert result["verdict"] == "Misleading"
    assert result["atomic_results"][0]["refute_kind"] == "out_of_context"


def test_one_tier4_match_is_not_decided():
    state = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": None,
            "matches": [_match()],
        },
    )
    judged = gate(state)
    assert judged["atomic_status"]["A1"] == "insufficient"
    assert judged["gate"] == "insufficient"
    assert judged["atomic_status"]["A1"] not in {"refuted", "supported"}


def test_true_needs_two_tier12_full_matches():
    claim = "This photo shows the Chennai flood on 10 March 2026."
    one = _claim(
        claim,
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": {"date": "2026-03-10", "url": "https://www.bbc.com/a", "host": "bbc.com", "tier": 2},
            "matches": [
                _match(
                    url="https://www.bbc.com/a",
                    host="bbc.com",
                    tier=2,
                    date="2026-03-10",
                    match="full",
                    stance="supports",
                    quote="The picture shows the Chennai flood.",
                    group_id="G1",
                )
            ],
        },
    )
    alone = gate(one)
    assert alone["atomic_status"]["A1"] != "supported"
    two = _claim(
        claim,
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": {"date": "2026-03-09", "url": "https://www.bbc.com/a", "host": "bbc.com", "tier": 2},
            "matches": [
                one["image_provenance"]["matches"][0],
                _match(
                    url="https://www.reuters.com/b",
                    host="reuters.com",
                    tier=2,
                    date="2026-03-09",
                    match="full",
                    stance="supports",
                    quote="The same photograph shows the Chennai flood.",
                    group_id="G2",
                ),
            ],
        },
    )
    judged = gate(two)
    assert judged["atomic_status"]["A1"] == "supported"
    result = assess({**two, **judged})
    assert result["verdict"] == "True"


def test_consistency_no_does_not_decide():
    state = _claim(
        "This photo shows a flood in Chennai today.",
        image_consistency={
            "consistent": "no",
            "mismatches": [{"detail": "place", "observed": "a dry street", "expected": "a flood"}],
            "place_clues": [],
            "time_clues": [],
        },
    )
    judged = gate(state)
    assert judged["atomic_status"]["A1"] == "insufficient"
    result = assess({**state, **judged})
    assert result["verdict"] == "Unverifiable"
    assert result["verdict"] not in {"False", "Misleading", "True"}
    assert CAPTION_MISMATCH in result["caveats"]
    assert result["review_status"] == "pending"
    assert CAPTION_MISMATCH_FLAG in result["red_flags"]


def test_no_caption_photo_is_other_and_keeps_first_seen(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    set_fake(
        lambda image: {
            "best_guess": ["flood"],
            "pages": [{"url": "https://www.bbc.com/old-flood", "title": "2019 floods", "match": "full"}],
        }
    )
    set_fake_search(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Tavily must not run")))

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text="The picture was taken during the 2019 Kerala floods.",
            published_date="2019-08-01",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)
    state = {
        "input_type": "image",
        "image_path": str(jpeg),
        "ocr_text": "",
        "caption": "",
        "claim": "",
        "checkable": False,
        "claim_date": "2026-03-10",
    }
    try:
        result = investigate(state)
    finally:
        clear_fake()
        clear_fake_search()
    assert result["search_status"] == "not_needed"
    assert result["image_provenance"]["first_seen"]["date"] == "2019-08-01"
    assert result["image_provenance"]["first_seen"]["host"] == "bbc.com"
    other = assess({**state, **result})
    assert other["verdict"] == "Other"


def _near_hashes():
    """Two pHashes within Hamming 6: a recompress, not a different photograph."""
    left = imagehash.hex_to_hash("f" * 16)
    right = imagehash.hex_to_hash("f" * 16)
    right.hash.flat[:3] = ~right.hash.flat[:3]
    assert 0 < (left - right) <= 6
    return str(left), str(right)


def _claim_encoder(monkeypatch):
    """Bag of words, unit length. Digits are left out on purpose.

    A date is a few digits. If those digits moved the vector, a caption
    that only changed the day would miss TAU_HIGH and the date rule would
    never be what blocked reuse. The real encoder often leaves two such
    sentences this close. Place names stay in the vector, so a swapped
    city has to be blocked by the place rule while the cosine is still high.
    """
    from misinfo_agent import prior_match

    class _Encoder:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, **kwargs):
            return np.stack([self._one(text) for text in texts])

        def _one(self, text):
            vec = np.zeros(1024, dtype=np.float32)
            for raw in str(text).split():
                token = "".join(ch for ch in raw.casefold() if ch.isalnum())
                if not token or token.isdigit():
                    continue
                bucket = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 1024
                vec[bucket] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm:
                vec /= norm
            return vec

    encoder = _Encoder()
    monkeypatch.setattr("misinfo_agent.memory.get_encoder", lambda: encoder)
    monkeypatch.setattr("misinfo_agent.prior_match.get_encoder", lambda: encoder)
    monkeypatch.setattr("misinfo_agent.investigate.get_encoder", lambda: encoder)
    # A word vector cached here would be this fake, and a later test of the
    # real key-term rule would score against it.
    monkeypatch.setattr(prior_match, "_WORD_CACHE", {})
    return encoder


def _cosine(encoder, left: str, right: str) -> float:
    a = encoder.encode([left])[0]
    b = encoder.encode([right])[0]
    return float(np.dot(a, b))


def _store_photo(phash: str, claim: str, caption: str | None = None, **extra) -> None:
    state = {
        "image_phash": phash,
        "verdict": "Misleading",
        "claim": claim,
        "caption": claim if caption is None else caption,
        "gate_reasons": [
            "The photo is from a different event or an earlier time, so the caption is out of context."
        ],
        "run_marks": {"run_id": "thread-old"},
        "image_provenance": {
            "first_seen": {
                "date": "2019-08-01",
                "url": "https://www.bbc.com/old",
                "host": "bbc.com",
                "tier": 2,
            }
        },
    }
    state.update(extra)
    remember(state)


def _age_memory(stored_on: str) -> None:
    with sqlite3.connect(config.IMAGE_MEMORY_DB) as conn:
        conn.execute("UPDATE images SET stored_on = ?", (stored_on,))


def _open_decision() -> dict:
    return {"route": "investigate", "match_type": "none", "best_match": {}, "match_reasons": []}


def test_same_photo_same_caption_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_MEMORY_DB", tmp_path / "image_memory.db")
    _claim_encoder(monkeypatch)
    left, right = _near_hashes()
    claim = "This photo shows the Chennai flood today."
    _store_photo(left, claim)
    # Out of context does not expire. An old stored_on must still reuse
    # when the caption is the same claim.
    _age_memory("2020-01-01")
    hit = lookup(right)
    assert hit["verdict"] == "Misleading"
    assert hit["distance"] <= 6
    assert hit["thread_id"] == "thread-old"
    assert hit["first_seen"]["date"] == "2019-08-01"
    decision = apply_image_reuse(
        {"image_phash": right, "claim": claim, "caption": claim},
        _open_decision(),
    )
    assert decision["route"] == "reuse_prior"
    assert decision["best_match"]["verdict"] == "Misleading"
    assert decision["best_match"]["refute_kind"] == "out_of_context"
    assert "different caption" not in " ".join(decision["match_reasons"]).casefold()


def test_same_photo_different_caption_is_investigated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_MEMORY_DB", tmp_path / "image_memory.db")
    encoder = _claim_encoder(monkeypatch)
    left, right = _near_hashes()
    stored = "This photo shows the Chennai flood today."
    fresh = "This photo shows the new Parliament building in New Delhi."
    assert _cosine(encoder, stored, fresh) < config.TAU_HIGH
    _store_photo(left, stored)
    jpeg = _jpeg(tmp_path)
    state = {
        "input_type": "image",
        "image_path": str(jpeg),
        "image_phash": right,
        "ocr_text": "",
        "caption": fresh,
        "claim": fresh,
        "checkable": True,
        "claim_date": "2026-03-10",
        "visual_description": "A large building.",
        "atomic_claims": [{"id": "A1", "text": fresh, "core": True, "checkable": True}],
    }
    decision = apply_image_reuse(state, _open_decision())
    assert decision["route"] == "investigate"
    assert decision["best_match"] == {}
    assert decision["image_memory_hit"]["verdict"] == "Misleading"
    assert (
        decision["image_memory_hit"]["note"]
        == "This photo was checked before with a different caption."
    )
    searched = {"n": 0}

    def _search(*args, **kwargs):
        searched["n"] += 1
        return []

    set_fake(lambda image: {"best_guess": ["building"], "pages": []})
    set_fake_search(_search)
    set_fake_llm(
        lambda prompt, schema, images: {
            "consistent": "unclear",
            "mismatches": [],
            "place_clues": [],
            "time_clues": [],
            "judgements": [],
            "queries": [],
        }
    )
    try:
        result = investigate(state)
    finally:
        clear_fake()
        clear_fake_search()
        clear_fake_llm()
    assert searched["n"] >= 1
    assert result["image_provenance"]["vision_status"] == "ran"


def test_same_photo_paraphrased_caption_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_MEMORY_DB", tmp_path / "image_memory.db")
    encoder = _claim_encoder(monkeypatch)
    left, right = _near_hashes()
    stored = "This photo shows the Chennai flood today."
    fresh = "Today this photo shows the Chennai flood."
    assert _cosine(encoder, stored, fresh) >= config.TAU_HIGH
    _store_photo(left, stored)
    decision = apply_image_reuse(
        {"image_phash": right, "claim": fresh, "caption": fresh},
        _open_decision(),
    )
    assert decision["route"] == "reuse_prior"
    assert decision["best_match"]["verdict"] == "Misleading"


def test_same_photo_different_date_or_place_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_MEMORY_DB", tmp_path / "image_memory.db")
    encoder = _claim_encoder(monkeypatch)
    left, right = _near_hashes()
    dated = "This photo shows the Chennai flood on 10 March 2026."
    other_day = "This photo shows the Chennai flood on 15 August 2019."
    placed = "This photo shows the Chennai flood today."
    other_place = "This photo shows the Mumbai flood today."
    assert _cosine(encoder, dated, other_day) >= config.TAU_HIGH
    assert _cosine(encoder, placed, other_place) >= config.TAU_HIGH
    _store_photo(left, dated)
    by_date = apply_image_reuse(
        {"image_phash": right, "claim": other_day, "caption": other_day},
        _open_decision(),
    )
    assert by_date["route"] == "investigate"
    assert (
        by_date["image_memory_hit"]["note"]
        == "This photo was checked before with a different caption."
    )
    _store_photo(left, placed)
    by_place = apply_image_reuse(
        {"image_phash": right, "claim": other_place, "caption": other_place},
        _open_decision(),
    )
    assert by_place["route"] == "investigate"
    assert (
        by_place["image_memory_hit"]["note"]
        == "This photo was checked before with a different caption."
    )


def test_image_text_search_off_is_provenance_only(tmp_path, monkeypatch):
    from misinfo_agent.provenance import picture_search_plan

    monkeypatch.setattr(config, "IMAGE_TEXT_SEARCH", False)
    jpeg = _jpeg(tmp_path)
    photo = {
        "input_type": "image",
        "image_path": str(jpeg),
        "ocr_text": "",
        "caption": "Chennai flood today",
        "claim": "This photo shows the Chennai flood today.",
        "checkable": True,
        "claim_date": "2026-03-10",
        "visual_description": "A flooded street.",
        "atomic_claims": [
            {
                "id": "A1",
                "text": "This photo shows the Chennai flood today.",
                "core": True,
                "checkable": True,
            }
        ],
    }
    assert picture_search_plan(photo) == "off"
    assert picture_search_plan({"input_type": "text", "claim": "India has 28 states."}) == "search"
    set_fake(
        lambda image: {
            "best_guess": ["flood"],
            "pages": [{"url": "https://www.bbc.com/old", "title": "Old", "match": "full"}],
        }
    )
    set_fake_search(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Tavily must not run")))
    set_fake_llm(
        lambda prompt, schema, images: {
            "consistent": "unclear",
            "mismatches": [],
            "place_clues": [],
            "time_clues": [],
            "judgements": [{"passage_id": "P1", "stance": "neutral", "quote": ""}],
        }
    )

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text="The photograph shows an older flood.",
            published_date="2019-08-01",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)
    try:
        result = investigate(photo)
    finally:
        clear_fake()
        clear_fake_search()
        clear_fake_llm()
    assert result["search_status"] == "not_needed"
    assert result["picture_search"] == "off"
    assert result["image_provenance"]["vision_status"] == "ran"
    assert result["image_provenance"]["first_seen"]["date"] == "2019-08-01"


def test_run_images_no_text_search_flag(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setattr(config, "IMAGE_TEXT_SEARCH", True)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_images_under_test", root / "scripts" / "run_images.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_images_under_test"] = module
    spec.loader.exec_module(module)
    assert module.build_parser().parse_args([]).no_text_search is False
    assert module.build_parser().parse_args(["--no-text-search"]).no_text_search is True
    module.apply_run_options(no_text_search=True)
    assert config.IMAGE_TEXT_SEARCH is False


def test_run_images_only_flag(tmp_path, monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setattr(config, "IMAGE_TEXT_SEARCH", True)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_images_only_under_test", root / "scripts" / "run_images.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_images_only_under_test"] = module
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        ["--only", "data/eval/images/smoke.jpg", "--limit", "2", "--no-text-search"]
    )
    assert args.only == "data/eval/images/smoke.jpg"
    assert args.limit == 2
    assert args.no_text_search is True
    assert args.sites == "both"
    assert module.modes_for("both") == ("blocked", "allowed")
    assert module.modes_for("blocked") == ("blocked",)
    module.apply_run_options(no_text_search=args.no_text_search)
    assert config.IMAGE_TEXT_SEARCH is False
    rows = [
        {"file": "data/eval/images/img01.jpg", "caption": "flood", "label": "Misleading", "claim_date": "2026-09-04"},
        {"file": "data/eval/images/img02.jpg", "caption": "meeting", "label": "False", "claim_date": "2026-08-10"},
    ]
    # --only is applied before --limit, so a later row is not dropped by the cap.
    assert [row["file"] for row in module.prepare_rows(rows, "data/eval/images/img02.jpg", 1)] == [
        "data/eval/images/img02.jpg"
    ]
    assert [row["file"] for row in module.prepare_rows(rows, "", 1)] == ["data/eval/images/img01.jpg"]
    assert module.prepare_rows(rows, "data/eval/images/missing.jpg", 0) == []
    photo = tmp_path / "img01.jpg"
    smoke = tmp_path / "smoke.jpg"
    photo.write_bytes(b"\xff\xd8same-photo")
    smoke.write_bytes(b"\xff\xd8same-photo")
    copied = module.select_rows(
        [{"file": str(photo), "caption": "Nepal floods", "label": "Misleading", "claim_date": "2026-09-04"}],
        str(smoke),
    )
    assert Path(copied[0]["file"]).resolve() == smoke.resolve()
    assert copied[0]["caption"] == "Nepal floods"
    assert copied[0]["claim_date"] == "2026-09-04"
    other = tmp_path / "other.jpg"
    other.write_bytes(b"\xff\xd8different")
    blank = module.select_rows(
        [{"file": str(photo), "caption": "Nepal floods", "label": "Misleading", "claim_date": "2026-09-04"}],
        str(other),
    )
    assert blank[0]["caption"] == ""
    assert blank[0]["claim_date"] == ""


def test_provenance_fetch_refuses_a_private_address(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [{"url": "http://127.0.0.1/secret", "title": "local", "match": "full"}],
        }
    )
    called = []

    def _spy(url):
        called.append(url)
        raise RuntimeError("offline test tried to fetch a page")

    monkeypatch.setattr("misinfo_agent.fetch._http_get", _spy)
    try:
        result = investigate_picture(
            {
                "input_type": "image",
                "image_path": str(jpeg),
                "caption": "",
                "claim": "",
                "checkable": False,
            }
        )
    finally:
        clear_fake()
    assert called == []
    assert result["image_provenance"]["matches"] == []


def test_screenshot_ocr_does_not_call_vision(tmp_path):
    from misinfo_agent.provenance import is_event_photo, picture_search_plan

    screenshot = {
        "input_type": "image",
        "image_path": str(_jpeg(tmp_path)),
        "ocr_text": " ".join(["word"] * 40),
        "caption": "",
        "claim": "A forwarded notice about a school closure.",
        "checkable": True,
    }
    # A paragraph of OCR and no caption is the forward. Vision is not asked.
    # The text search plan stays "search", which is the OCR path as it was.
    assert is_event_photo(screenshot) is False
    assert picture_search_plan(screenshot) == "search"
    photo = {
        "input_type": "image",
        "image_path": str(_jpeg(tmp_path)),
        "ocr_text": "",
        "caption": "",
        "claim": "",
        "checkable": False,
    }
    assert is_event_photo(photo) is True
    assert picture_search_plan(photo) == "not_event"


def test_capped_caption_search_still_runs_provenance(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    monkeypatch.setattr(
        "misinfo_agent.search.searches_this_month",
        lambda: config.TAVILY_MONTHLY_MAX,
    )
    set_fake(
        lambda image: {
            "best_guess": ["flood"],
            "pages": [{"url": "https://www.bbc.com/old", "title": "Old", "match": "full"}],
        }
    )
    set_fake_search(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Tavily must not run")))
    set_fake_llm(
        lambda prompt, schema, images: {
            "consistent": "unclear",
            "mismatches": [],
            "place_clues": [],
            "time_clues": [],
            "judgements": [{"passage_id": "P1", "stance": "neutral", "quote": ""}],
        }
    )

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text="The photograph shows an older flood.",
            published_date="2019-08-01",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)
    try:
        result = investigate(
            {
                "input_type": "image",
                "image_path": str(jpeg),
                "ocr_text": "",
                "caption": "Chennai flood today",
                "claim": "This photo shows the Chennai flood today.",
                "checkable": True,
                "claim_date": "2026-03-10",
                "visual_description": "A flooded street.",
                "atomic_claims": [
                    {
                        "id": "A1",
                        "text": "This photo shows the Chennai flood today.",
                        "core": True,
                        "checkable": True,
                    }
                ],
            }
        )
    finally:
        clear_fake()
        clear_fake_search()
        clear_fake_llm()
    assert result["search_status"] == "monthly_cap"
    assert result["image_provenance"]["vision_status"] == "ran"
    assert result["image_provenance"]["first_seen"]["date"] == "2019-08-01"


def test_factcheck_pages_are_blocked_for_provenance(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [
                {"url": "https://www.boomlive.in/fact-check/old", "title": "Fact check", "match": "full"},
                {"url": "https://www.bbc.com/news/old", "title": "News", "match": "full"},
            ],
        }
    )
    fetched = []

    def _fetch(hit):
        fetched.append(hit.url)
        return Page(
            url=hit.url,
            title=hit.title,
            text="The picture is from the 2019 Kerala floods.",
            # On the claim day, so the earlier-date rule does not label
            # this page. The test is that the model's quote is kept.
            published_date="2026-03-10",
            date_source="htmldate",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)
    set_fake_llm(
        lambda prompt, schema, images: {
            "judgements": [
                {
                    "passage_id": "P1",
                    "stance": "refutes",
                    "quote": "The picture is from the 2019 Kerala floods.",
                }
            ]
        }
    )
    try:
        result = investigate_picture(
            {
                "input_type": "image",
                "image_path": str(jpeg),
                "caption": "Chennai today",
                "claim": "This photo shows the Chennai flood today.",
                "checkable": True,
                "claim_date": "2026-03-10",
                "visual_description": "",
            }
        )
    finally:
        clear_fake()
        clear_fake_llm()
    assert fetched == ["https://www.bbc.com/news/old"]
    urls = [row["url"] for row in result["image_provenance"]["matches"]]
    assert "https://www.bbc.com/news/old" in urls
    assert all("boomlive" not in url for url in urls)
    assert result["image_provenance"]["matches"][0]["stance"] == "refutes"
    assert result["image_provenance"]["matches"][0]["quote"] == "The picture is from the 2019 Kerala floods."
    assert result["evidence"][0]["id"].startswith("E")
    assert result["evidence"][0]["provenance"] is True


def test_cache_key_includes_max_results(tmp_path, monkeypatch):
    from misinfo_agent import vision

    jpeg = _jpeg(tmp_path)
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "test-key")
    calls = {"n": 0}

    def _post(image, key):
        calls["n"] += 1
        return {"responses": [{"webDetection": {"bestGuessLabels": [], "pagesWithMatchingImages": []}}]}

    monkeypatch.setattr(vision, "_post", _post)
    # A file saved under the image hash alone is the old 20-page cache.
    # It must not answer a request that asks for a different number.
    digest = hashlib.sha256(jpeg.read_bytes()).hexdigest()
    old = config.VISION_CACHE_DIR / f"{digest}.json"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text('{"best_guess": ["stale"], "pages": []}', encoding="utf-8")
    found = detect(str(jpeg))
    assert found["vision_status"] == "ran"
    assert found["best_guess"] == []
    assert calls["n"] == 1
    assert (config.VISION_CACHE_DIR / f"{digest}.r{config.VISION_MAX_RESULTS}.json").is_file()
    again = detect(str(jpeg))
    assert again["vision_status"] == "cached"
    assert calls["n"] == 1
    monkeypatch.setattr(config, "VISION_MAX_RESULTS", 20)
    third = detect(str(jpeg))
    assert third["vision_status"] == "ran"
    assert calls["n"] == 2


def test_earlier_trusted_date_refutes_without_a_stance_call(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    pages = {
        "https://www.bbc.com/a": "The first newsroom describes a flooded street from years ago in detail.",
        "https://www.bbc.com/b": "The same newsroom later wrote a different account of boats on a river.",
        "https://www.reuters.com/c": "A second newsroom covered rain and rescue work in another city then.",
    }
    set_fake(
        lambda image: {
            "best_guess": ["flood"],
            "pages": [
                {"url": url, "title": "Old flood", "match": "full"} for url in pages
            ],
        }
    )

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text=pages[hit.url],
            published_date="2019-08-01",
            date_source="htmldate",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)

    def _llm(prompt, schema, images):
        if schema.__name__ == "ProvenanceBatch":
            raise AssertionError("an earlier trusted date must not ask for a stance")
        return {"consistent": "unclear", "mismatches": [], "place_clues": [], "time_clues": []}

    set_fake_llm(_llm)
    try:
        result = investigate_picture(
            _claim(
                "This photo shows the Chennai flood today.",
                image_path=str(jpeg),
                caption="Chennai flood today",
                claim_date="2026-03-10",
            )
        )
    finally:
        clear_fake()
        clear_fake_llm()
    matches = result["image_provenance"]["matches"]
    by_url = {row["url"]: row for row in matches}
    quote = "Image on this page dated 2019-08-01, before the claimed event (2026-03-10)."
    assert by_url["https://www.bbc.com/a"]["group_id"] == by_url["https://www.bbc.com/b"]["group_id"]
    assert by_url["https://www.reuters.com/c"]["group_id"] != by_url["https://www.bbc.com/a"]["group_id"]
    for row in matches:
        assert row["stance"] == "refutes"
        assert row["quote"] == quote
    assert len(result["evidence"]) == 3
    assert all(row["provenance"] is True and row["stance"] == "refutes" for row in result["evidence"])


def test_page_match_and_untrusted_date_still_ask_for_stance(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [
                {"url": "https://www.bbc.com/page", "title": "Page", "match": "page"},
                {"url": "https://www.reuters.com/old", "title": "Old", "match": "full"},
            ],
        }
    )
    asked = {"n": 0}

    def _fetch(hit):
        source = "untrusted" if "reuters" in hit.url else "htmldate"
        return Page(
            url=hit.url,
            title=hit.title,
            text="The photograph shows a flooded street from an earlier year entirely.",
            published_date="2019-08-01",
            date_source=source,
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)

    def _llm(prompt, schema, images):
        if schema.__name__ == "ProvenanceBatch":
            asked["n"] += 1
            return {"judgements": [{"passage_id": "P1", "stance": "neutral", "quote": ""}]}
        return {"consistent": "unclear", "mismatches": [], "place_clues": [], "time_clues": []}

    set_fake_llm(_llm)
    try:
        result = investigate_picture(
            _claim(
                "This photo shows the Chennai flood today.",
                image_path=str(jpeg),
                caption="Chennai flood today",
                claim_date="2026-03-10",
            )
        )
    finally:
        clear_fake()
        clear_fake_llm()
    assert asked["n"] == 1
    by_url = {row["url"]: row for row in result["image_provenance"]["matches"]}
    assert by_url["https://www.bbc.com/page"]["stance"] == "neutral"
    assert by_url["https://www.reuters.com/old"]["stance"] == "neutral"
    assert "before the claimed event" not in by_url["https://www.reuters.com/old"]["quote"]


def test_extract_date_is_the_event_and_claim_date_is_the_fallback(tmp_path, monkeypatch):
    jpeg = _jpeg(tmp_path)
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [
                {"url": "https://www.bbc.com/august", "title": "August", "match": "full"},
                {"url": "https://www.reuters.com/january", "title": "January", "match": "partial"},
            ],
        }
    )

    def _fetch(hit):
        published = "2026-08-01" if "august" in hit.url else "2026-01-01"
        return Page(
            url=hit.url,
            title=hit.title,
            text="The photograph was published with a dated account of the flood.",
            published_date=published,
            date_source="htmldate",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)
    calls = {"n": 0}

    def _llm(prompt, schema, images):
        if schema.__name__ == "ProvenanceBatch":
            calls["n"] += 1
            return {"judgements": [{"passage_id": "P1", "stance": "neutral", "quote": ""}]}
        return {"consistent": "unclear", "mismatches": [], "place_clues": [], "time_clues": []}

    set_fake_llm(_llm)
    state = _claim(
        "This photo shows the Chennai flood on 10 March 2026.",
        image_path=str(jpeg),
        caption="This photo shows the Chennai flood on 10 March 2026.",
        claim_date="2026-09-01",
        entities={"dates": ["10 March 2026"]},
    )
    try:
        found = investigate_picture(state)
    finally:
        clear_fake_llm()
    by_url = {row["url"]: row for row in found["image_provenance"]["matches"]}
    # August is a full match after the March event, so the model is asked.
    # January is earlier, but it is only a partial, so the date rule
    # does not label it and it is not evidence.
    assert calls["n"] == 1
    assert by_url["https://www.bbc.com/august"]["stance"] == "neutral"
    assert by_url["https://www.reuters.com/january"]["stance"] == "neutral"
    assert "before the claimed event" not in by_url["https://www.reuters.com/january"]["quote"]
    assert found["image_provenance"]["first_seen"]["date"] == "2026-08-01"
    assert all(row["url"] != "https://www.reuters.com/january" for row in found["evidence"])
    # The caption's 10 March is the event even when extract recorded nothing.
    # The August page is after that day, so it is not an earlier copy.
    clear_fake()
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [{"url": "https://www.bbc.com/august", "title": "August", "match": "full"}],
        }
    )
    calls["n"] = 0
    set_fake_llm(_llm)
    bare = _claim(
        "This photo shows the Chennai flood on 10 March 2026.",
        image_path=str(jpeg),
        caption="This photo shows the Chennai flood on 10 March 2026.",
        claim_date="2026-09-01",
    )
    try:
        missed = investigate_picture(bare)
    finally:
        clear_fake()
        clear_fake_llm()
    assert calls["n"] == 1
    assert "before the claimed event" not in missed["image_provenance"]["matches"][0]["quote"]


def test_linkedin_id_dates_a_photo_without_fetching(tmp_path):
    from datetime import datetime, timezone

    from misinfo_agent.fetch import linkedin_post_date

    when = datetime(2020, 6, 15, 12, 0, tzinfo=timezone.utc)
    activity = (int(when.timestamp() * 1000) << 22) | 0x2A
    url = f"https://www.linkedin.com/feed/update/urn:li:activity:{activity}"
    assert linkedin_post_date(url) == "2020-06-15"
    jpeg = _jpeg(tmp_path)
    set_fake(lambda image: {"best_guess": [], "pages": [{"url": url, "title": "Post", "match": "full"}]})

    def _llm(prompt, schema, images):
        if schema.__name__ == "ProvenanceBatch":
            raise AssertionError("the activity id already dated the post")
        return {"consistent": "unclear", "mismatches": [], "place_clues": [], "time_clues": []}

    set_fake_llm(_llm)
    try:
        result = investigate_picture(
            _claim(
                "This photo shows the Chennai flood today.",
                image_path=str(jpeg),
                caption="Chennai flood today",
                claim_date="2026-03-10",
            )
        )
    finally:
        clear_fake()
        clear_fake_llm()
    match = result["image_provenance"]["matches"][0]
    assert match["date"] == "2020-06-15"
    assert match["date_source"] == "linkedin_id"
    assert match["stance"] == "refutes"
    assert match["quote"] == "Image on this page dated 2020-06-15, before the claimed event (2026-03-10)."


def test_photo_weak_refute_caps_confidence_and_keeps_the_tier12_rule(monkeypatch):
    monkeypatch.setattr("misinfo_agent.assess._confidence", lambda *args, **kwargs: 0.9)
    weak = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": {"date": "2019-08-01", "url": "https://a.example/a", "host": "a.example", "tier": 4},
            "matches": [
                _match(group_id="G1", url="https://a.example/a", host="a.example"),
                _match(group_id="G2", url="https://b.example/b", host="b.example", date="2019-05-01"),
            ],
        },
    )
    judged = gate(weak)
    assert judged["atomic_status"]["A1"] == "refuted"
    assert config.PHOTO_WEAK_REFUTE_CAVEAT in judged["gate_reasons"]
    set_fake_llm(
        lambda prompt, schema, images: {
            "atomic": [{"atomic_id": "A1", "explanation": "The pages are older.", "refute_kind": None}],
            "true_version": "",
            "correction_message": "Do not forward this.",
        }
    )
    try:
        result = assess({**weak, **judged})
    finally:
        clear_fake_llm()
    assert result["verdict"] == "Misleading"
    assert result["atomic_results"][0]["refute_kind"] == "out_of_context"
    assert config.PHOTO_WEAK_REFUTE_CAVEAT in result["caveats"]
    assert result["confidence"] == config.PHOTO_WEAK_REFUTE_CONFIDENCE_CAP

    credible = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": None,
            "matches": [
                _match(
                    url="https://www.bbc.com/a",
                    host="bbc.com",
                    tier=2,
                    date="2026-03-10",
                    stance="refutes",
                    quote="The photograph is from a different flood.",
                    group_id="G1",
                )
            ],
        },
    )
    tiered = gate(credible)
    assert tiered["atomic_status"]["A1"] == "refuted"
    assert config.PHOTO_WEAK_REFUTE_CAVEAT not in tiered["gate_reasons"]
    set_fake_llm(
        lambda prompt, schema, images: {
            "atomic": [{"atomic_id": "A1", "explanation": "The newsroom disagrees.", "refute_kind": "entirely_false"}],
            "true_version": "",
            "correction_message": "Do not forward this.",
        }
    )
    try:
        strong = assess({**credible, **tiered})
    finally:
        clear_fake_llm()
    assert strong["confidence"] == 0.9
    assert config.PHOTO_WEAK_REFUTE_CAVEAT not in strong["caveats"]
    assert strong["atomic_results"][0]["refute_kind"] == "out_of_context"

    monkeypatch.setattr(config, "GATE_PHOTO_WEAK_REFUTE", False)
    assert gate(weak)["atomic_status"]["A1"] == "insufficient"
    assert gate(credible)["atomic_status"]["A1"] == "refuted"


def test_photo_weak_refute_stops_when_any_group_supports():
    blocked = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": None,
            "matches": [
                _match(group_id="G1", url="https://a.example/a", host="a.example"),
                _match(group_id="G2", url="https://b.example/b", host="b.example"),
                _match(
                    group_id="G3",
                    url="https://c.example/c",
                    host="c.example",
                    tier=4,
                    date="2026-03-10",
                    stance="supports",
                    quote="The picture shows the Chennai flood.",
                ),
            ],
        },
    )
    assert gate(blocked)["atomic_status"]["A1"] == "insufficient"
    # A lower-tier support does not cancel the tier 1-2 rule.
    kept = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": None,
            "matches": [
                _match(
                    group_id="G1",
                    url="https://www.bbc.com/a",
                    host="bbc.com",
                    tier=2,
                    date="2026-03-10",
                    stance="refutes",
                    quote="The photograph is from a different flood.",
                ),
                blocked["image_provenance"]["matches"][2],
            ],
        },
    )
    judged = gate(kept)
    assert judged["atomic_status"]["A1"] == "refuted"
    assert config.PHOTO_WEAK_REFUTE_CAVEAT not in judged["gate_reasons"]


def test_photo_lean_line_is_on_every_image_card():
    from app.components import card_html, view_model
    from misinfo_agent.output import render

    state = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": {"date": "2019-08-01", "url": "https://example.com/a", "host": "example.com", "tier": 4},
            "matches": [
                _match(group_id="G1", url="https://example.com/a", host="example.com", date="2019-08-01"),
                _match(
                    group_id="G2",
                    url="https://blog.example.net/b",
                    host="blog.example.net",
                    date="2019-05-01",
                ),
                _match(
                    group_id="G3",
                    url="https://notes.example.org/c",
                    host="notes.example.org",
                    match="page",
                    date="2018-01-01",
                    stance="neutral",
                    quote="",
                ),
            ],
        },
    )
    card = render(state)["result"]
    # The page match is not this file, so it is not one of the sites.
    assert card["photo_lean"] == {
        "first_seen_year": 2019,
        "n_sites": 2,
        "n_dated_earlier": 2,
        "similar_since": None,
    }
    html = card_html(view_model(result=card))
    assert "This photo has been online since at least 2019 (2 sites)" in html
    decided = dict(card)
    decided["verdict"] = "Misleading"
    assert "This photo has been online since at least 2019 (2 sites)" in card_html(view_model(result=decided))
    text = render({"input_type": "text", "claim": "India has 28 states.", "verdict": "Unverifiable"})["result"]
    assert text["photo_lean"] is None
    assert "online since" not in card_html(view_model(result=text))


def test_cache_only_does_not_call_vision(tmp_path):
    jpeg = _jpeg(tmp_path)
    missed = detect(str(jpeg), cache_only=True)
    assert missed["vision_status"] == "cache_only_miss"
    assert missed["pages"] == []
    assert calls_this_month() == 0


def test_partial_from_2019_never_refutes_and_two_full_matches_do(tmp_path, monkeypatch):
    # This is the v0 contract: a lookalike does not vote. v1 and v2 are
    # covered by their own tests, so a later default does not rewrite this one.
    monkeypatch.setattr(config, "PHOTO_RULE", "v0")
    jpeg = _jpeg(tmp_path)
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [{"url": "https://www.bbc.com/old", "title": "Old", "match": "partial"}],
        }
    )

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text="A similar frame from an earlier flood.",
            published_date="2019-08-01",
            date_source="htmldate",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)

    def _llm(prompt, schema, images):
        if schema.__name__ == "ProvenanceBatch":
            return {"judgements": [{"passage_id": "P1", "stance": "refutes", "quote": "earlier flood"}]}
        return {"consistent": "unclear", "mismatches": [], "place_clues": [], "time_clues": []}

    set_fake_llm(_llm)
    try:
        found = investigate_picture(
            _claim(
                "This photo shows the Chennai flood today.",
                image_path=str(jpeg),
                caption="Chennai flood today",
                claim_date="2026-03-10",
            )
        )
    finally:
        clear_fake()
        clear_fake_llm()
    match = found["image_provenance"]["matches"][0]
    # The model may call a lookalike a refute. It is still not this file,
    # so it is not evidence and it does not date the photograph.
    assert match["match"] == "partial"
    assert found["evidence"] == []
    assert found["image_provenance"]["first_seen"] is None
    partial = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance=found["image_provenance"],
    )
    judged = gate(partial)
    assert judged["atomic_status"]["A1"] != "refuted"
    assert "out of context" not in " ".join(judged["gate_reasons"]).casefold()

    two = _claim(
        "This photo shows the Chennai flood today.",
        image_provenance={
            "vision_status": "ran",
            "best_guess": [],
            "first_seen": {"date": "2019-08-01", "url": "https://a.example/a", "host": "a.example", "tier": 4},
            "matches": [
                _match(group_id="G1", url="https://a.example/a", host="a.example", date="2019-08-01"),
                _match(group_id="G2", url="https://b.example/b", host="b.example", date="2019-05-01"),
                _match(
                    group_id="G3",
                    url="https://c.example/c",
                    host="c.example",
                    match="partial",
                    date="2018-01-01",
                ),
            ],
        },
    )
    refuted = gate(two)
    assert refuted["atomic_status"]["A1"] == "refuted"
    assert "out of context" in " ".join(refuted["gate_reasons"]).casefold()


def test_genuine_photo_with_only_partial_earlier_matches_is_not_out_of_context(monkeypatch):
    monkeypatch.setattr(config, "PHOTO_RULE", "v0")
    from app.components import card_html, view_model
    from misinfo_agent.output import render

    state = _claim(
        "This photo shows the library opening on 10 March 2026.",
        image_provenance={
            "vision_status": "cached",
            "best_guess": [],
            "first_seen": None,
            "matches": [
                _match(
                    group_id="G1",
                    url="https://a.example/old",
                    host="a.example",
                    match="partial",
                    date="2019-08-01",
                    date_source="htmldate",
                    stance="refutes",
                ),
                _match(
                    group_id="G2",
                    url="https://b.example/older",
                    host="b.example",
                    match="partial",
                    date="2018-01-01",
                    date_source="htmldate",
                    stance="refutes",
                ),
            ],
        },
    )
    judged = gate(state)
    assert judged["atomic_status"]["A1"] != "refuted"
    assert "out of context" not in " ".join(judged["gate_reasons"]).casefold()
    card = render({**state, **judged})["result"]
    assert card["photo_lean"]["first_seen_year"] is None
    assert card["photo_lean"]["similar_since"] == 2018
    assert card["photo_lean"]["n_dated_earlier"] == 0
    html = card_html(view_model(result=card))
    assert "Similar images seen online since 2018" in html
    assert "This photo has been online since" not in html


def test_reused_verdict_keeps_provenance_from_the_vision_cache(tmp_path, monkeypatch):
    from misinfo_agent.graph import _prior
    from misinfo_agent.output import photo_lean
    from misinfo_agent.vision import _write_cache

    monkeypatch.setattr(config, "IMAGE_MEMORY_DB", tmp_path / "image_memory.db")
    _claim_encoder(monkeypatch)
    jpeg = _jpeg(tmp_path)
    digest = hashlib.sha256(jpeg.read_bytes()).hexdigest()
    _write_cache(
        digest,
        {
            "best_guess": [],
            "pages": [{"url": "https://www.bbc.com/old", "title": "Old flood", "match": "full"}],
        },
    )

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text="The photograph is from the 2019 flood.",
            published_date="2019-08-01",
            date_source="htmldate",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)
    monkeypatch.setattr(
        "misinfo_agent.prior_match.decide_prior",
        lambda *args, **kwargs: {
            "route": "investigate",
            "match_type": "none",
            "best_match": {},
            "match_reasons": [],
        },
    )
    claim = "This photo shows the Chennai flood today."
    phash, _near = _near_hashes()
    _store_photo(phash, claim)
    before = calls_this_month()
    state = _claim(
        claim,
        image_path=str(jpeg),
        image_phash=phash,
        caption=claim,
        claim_date="2026-03-10",
    )
    decision = _prior(state)
    assert calls_this_month() == before
    assert decision["route"] == "reuse_prior"
    provenance = decision["image_provenance"]
    assert provenance["vision_status"] == "cached"
    assert provenance["first_seen"]["date"] == "2019-08-01"
    assert provenance["matches"][0]["stance"] == "refutes"
    lean = photo_lean({**state, **decision})
    assert lean["first_seen_year"] == 2019
    assert lean["n_dated_earlier"] == 1
    assert "This photo has been online since at least 2019" in (
        f"This photo has been online since at least {lean['first_seen_year']} ({lean['n_sites']} sites)"
    )


def test_run_images_uses_a_fresh_image_memory(monkeypatch):
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_images_memory_under_test", root / "scripts" / "run_images.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_images_memory_under_test"] = module
    spec.loader.exec_module(module)
    real = config.DATA_DIR / "image_memory.db"
    path = module.use_fresh_image_memory()
    assert path != real
    assert path == config.IMAGE_MEMORY_DB
    assert path.name == "image_memory.db"
    assert "image-memory-eval-" in str(path.parent)


def test_september_page_does_not_refute_an_august_claim(monkeypatch):
    """The event is the caption's month, not the day the check is run."""
    monkeypatch.setattr(config, "PHOTO_RULE", "v0")
    state = _claim(
        "This photo shows the Nepal floods in August 2026.",
        caption="This photo shows the Nepal floods in August 2026.",
        claim_date="2026-11-25",
        image_provenance={
            "vision_status": "cached",
            "best_guess": [],
            "first_seen": {"date": "2026-09-07", "url": "https://news.example/a", "host": "news.example", "tier": 2},
            "matches": [
                _match(
                    group_id="G1",
                    url="https://news.example/a",
                    host="news.example",
                    tier=2,
                    date="2026-09-07",
                    date_source="htmldate",
                    stance="neutral",
                    quote="",
                ),
                _match(
                    group_id="G2",
                    url="https://other.example/b",
                    host="other.example",
                    tier=2,
                    date="2026-09-12",
                    date_source="htmldate",
                    stance="neutral",
                    quote="",
                ),
            ],
        },
    )
    from misinfo_agent.provenance import claim_event_date, is_dated_earlier

    assert claim_event_date(state).isoformat() == "2026-08-01"
    assert is_dated_earlier(state["image_provenance"]["matches"][0], claim_event_date(state)) is False
    judged = gate(state)
    assert judged["atomic_status"]["A1"] != "refuted"
    assert "out of context" not in " ".join(judged["gate_reasons"]).casefold()


def test_yesterday_and_an_explicit_day_are_the_event():
    from datetime import date

    from misinfo_agent.provenance import claim_event_window

    yesterday = _claim(
        "This photo was taken yesterday.",
        caption="This photo was taken yesterday.",
        claim_date="2026-11-25",
    )
    assert claim_event_window(yesterday) == (date(2026, 11, 24), date(2026, 11, 24))
    explicit = _claim(
        "The flood on 10 March 2026.",
        caption="The flood on 10 March 2026.",
        claim_date="2026-11-25",
    )
    assert claim_event_window(explicit) == (date(2026, 3, 10), date(2026, 3, 10))
    month = _claim("Floods in August 2026.", caption="Floods in August 2026.", claim_date="2026-11-25")
    assert claim_event_window(month) == (date(2026, 8, 1), date(2026, 8, 31))


def test_photo_rule_variants(monkeypatch):
    """v1 counts lookalikes, v2 only a year-old trusted one, v3 a same-event full match blocks."""
    from misinfo_agent.provenance import claim_event_date, is_dated_earlier

    def _state(matches, caption="This photo shows the library opening on 10 March 2026."):
        return _claim(caption, caption=caption, claim_date="2026-11-25", image_provenance={
            "vision_status": "cached",
            "best_guess": [],
            "first_seen": None,
            "matches": matches,
        })

    recent = [
        _match(group_id="G1", url="https://a.example/a", host="a.example", match="partial",
               date="2026-01-01", date_source="trafilatura", stance="refutes", tier=1),
        _match(group_id="G2", url="https://b.example/b", host="b.example", match="partial",
               date="2026-02-01", date_source="htmldate", stance="refutes", tier=4),
    ]
    monkeypatch.setattr(config, "PHOTO_RULE", "v1")
    state = _state(recent)
    event = claim_event_date(state)
    assert is_dated_earlier(recent[0], event) is True
    assert gate(state)["atomic_status"]["A1"] == "refuted"

    monkeypatch.setattr(config, "PHOTO_RULE", "v2")
    # January is the same year as March, so a lookalike does not count.
    assert is_dated_earlier(recent[0], event) is False
    assert gate(_state(recent))["atomic_status"]["A1"] != "refuted"
    old = [
        _match(group_id="G1", url="https://a.example/a", host="a.example", match="partial",
               date="2019-08-01", date_source="htmldate", stance="refutes", tier=2),
        _match(group_id="G2", url="https://b.example/b", host="b.example", match="partial",
               date="2018-01-01", date_source="url", stance="refutes", tier=4),
    ]
    assert is_dated_earlier(old[0], event) is True
    assert gate(_state(old))["atomic_status"]["A1"] == "refuted"
    untrusted = [
        _match(group_id="G1", url="https://a.example/a", host="a.example", match="partial",
               date="2015-05-06", date_source="none", stance="refutes", tier=2),
        _match(group_id="G2", url="https://b.example/b", host="b.example", match="partial",
               date="2014-01-01", date_source=None, stance="refutes", tier=2),
    ]
    assert gate(_state(untrusted))["atomic_status"]["A1"] != "refuted"

    monkeypatch.setattr(config, "PHOTO_RULE", "v3")
    blocked = _state(old + [
        _match(group_id="G3", url="https://pib.gov.in/photo", host="pib.gov.in", match="full",
               date="2026-03-10", date_source="htmldate", stance="supports", tier=1, quote="The opening."),
    ])
    judged = gate(blocked)
    assert judged["atomic_status"]["A1"] != "refuted"
    assert "out of context" not in " ".join(judged["gate_reasons"]).casefold()
