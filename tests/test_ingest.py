"""Box 1–2. No model and no network.

The garlic forward is the brief's own example: every warning sign has
to fire on the raw string, and the cleaned text has to lose the trailer.
URL tests serve saved HTML through the safe fetcher's HTTP seam. The
database assertions read the file. A hash of the message is allowed.
The message itself is not.
"""

import hashlib
import io
import re
import socket
import sqlite3
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

import misinfo_agent.config as config
import misinfo_agent.fetch as fetch
from misinfo_agent.ingest import alarm_style, ingest

FIXTURES = Path(__file__).parent / "fixtures"
_FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")

@pytest.fixture(autouse=True)
def _record_in_these_tests(monkeypatch, request):
    """The product default is off so a demo does not inflate reach.

    These tests are about the counter itself, so they turn recording on.
    The test of the off switch sets the flag itself, after this fixture.
    """
    if request.node.name != "test_record_submissions_false_writes_nothing":
        monkeypatch.setattr(config, "RECORD_SUBMISSIONS", True)


GARLIC = (
    "‼️BREAKING‼️ Forwarded as received — AIIMS Delhi doctors have CONFIRMED "
    "that drinking 2 spoons of raw garlic juice every morning cures Type 2 "
    "diabetes in 30 days. Govt is hiding this to protect pharma companies!! "
    "Send to 10 groups 🙏🙏"
)
HINDI = (
    "दिल्ली एम्स के डॉक्टरों ने दावा किया है कि कच्चा लहसुन खाने से "
    "मधुमेह तीस दिन में ठीक हो जाता है"
)


def test_garlic_forward_signals_and_clean_text(tmp_path):
    out = ingest(GARLIC, input_type="text", db_path=tmp_path / "submissions.db")
    signals = out["style_signals"]
    assert "breaking" in signals["urgency_words"]
    assert signals["share_request"] is True
    assert signals["forwarded_label"] == "Forwarded as received"
    assert "AIIMS" in signals["authority_mentions"]
    assert "hiding" in signals["conspiracy_phrases"]
    assert "confirmed" in signals["certainty_words"]
    assert "cures" in signals["certainty_words"]
    assert signals["caps_ratio"] == _caps_ratio(GARLIC)
    # AIIMS is five capitals. Leaving it out is why this is not the
    # ratio of every letter in the forward.
    every_letter = [ch for ch in GARLIC if ch.isalpha()]
    assert signals["caps_ratio"] < round(
        sum(ch.isupper() for ch in every_letter) / len(every_letter), 4
    )
    assert signals["exclaim_count"] == GARLIC.count("!") + 2 * GARLIC.count("‼")
    assert signals["emoji_count"] == 2

    text = out["text"]
    assert "garlic" in text.lower()
    assert "2" in text and "30" in text
    assert "Forwarded as received" not in text
    assert "Send to 10 groups" not in text
    assert "🙏" not in text
    assert "‼" not in text
    # The readable text keeps the sentence. The match string is what the
    # hash is of: lower case, punctuation gone, rupee sign would stay.
    assert out["match_text"] == out["match_text"].lower()
    assert "!" not in out["match_text"]
    assert "garlic" in out["match_text"]
    assert "forwarded as received" not in out["match_text"]
    assert "send to 10 groups" not in out["match_text"]

    assert out["input_type"] == "text"
    assert out["raw_text"] == GARLIC
    assert out["language"] == "en"
    assert out["hinglish"] is False
    assert out["checkable_hint"] is True
    assert out["checkable_reason"] == ""
    assert out["claim_date"] == date.today().isoformat()
    assert out["submission_count"] == 1
    assert out["submission_hash"] == hashlib.sha256(
        out["match_text"].encode("utf-8")
    ).hexdigest()


def test_greeting_is_too_short(tmp_path):
    out = ingest("Good morning 🙏", db_path=tmp_path / "submissions.db")
    assert out["checkable_hint"] is False
    assert out["checkable_reason"] == "too short to check"
    assert "🙏" not in out["text"]


def test_hindi_is_out_of_scope(tmp_path):
    out = ingest(HINDI, db_path=tmp_path / "submissions.db")
    assert out["checkable_hint"] is False
    assert out["checkable_reason"] == "out of scope (not English)"
    assert out["hinglish"] is False


def test_hinglish_continues(tmp_path):
    # Latin letters plus Hindi words. The run must not stop: Hinglish
    # is in scope, and the flag is how the card says so.
    text = (
        "Modi ji ne yeh naya health scheme kal launch kar diya hai "
        "officially across India"
    )
    out = ingest(text, db_path=tmp_path / "submissions.db")
    assert out["hinglish"] is True
    assert out["checkable_hint"] is True
    assert out["language"] == "en"
    assert "hinglish" in out["flags"]


def test_same_forward_different_emojis_counts_twice(tmp_path):
    db = tmp_path / "submissions.db"
    first = ingest(GARLIC, db_path=db)
    second = ingest(GARLIC.replace("🙏🙏", "🚨🚨"), db_path=db)
    assert first["submission_count"] == 1
    assert second["submission_count"] == 2
    assert first["submission_hash"] == second["submission_hash"]
    assert first["match_text"] == second["match_text"]
    assert second["dedupe"]["is_duplicate"] is True
    assert second["dedupe"]["seen_count"] == 1
    assert second["dedupe"]["method"] == "sha256"


def test_near_copy_joins_the_same_row(tmp_path):
    # One extra short word. Character shingles of a long sentence still
    # clear 0.8, so this is one rumour with a trailer, not two.
    db = tmp_path / "submissions.db"
    base = (
        "The reserve bank has announced that all five hundred rupee notes "
        "printed before twenty sixteen will be invalid from the first of "
        "December this year according to a message circulating in the city"
    )
    first = ingest(base, db_path=db)
    second = ingest(base + " today", db_path=db)
    assert first["submission_hash"] == second["submission_hash"]
    assert second["submission_count"] == 2
    assert second["dedupe"]["method"] == "minhash"


def test_different_forward_is_a_new_row(tmp_path):
    db = tmp_path / "submissions.db"
    first = ingest(GARLIC, db_path=db)
    other = ingest(
        "The cricket board banned the player for two years after the "
        "match fixing inquiry ended in Mumbai yesterday evening",
        db_path=db,
    )
    assert other["submission_count"] == 1
    assert other["submission_hash"] != first["submission_hash"]


def test_rupee_and_numbers_survive_cleaning(tmp_path):
    out = ingest(
        "RBI says ₹500 notes stay valid until 31 December 2026 according "
        "to the latest circular sent from the Mumbai office.",
        db_path=tmp_path / "submissions.db",
    )
    assert "₹500" in out["text"]
    assert "31" in out["text"] and "2026" in out["text"]
    assert "₹500" in out["match_text"]
    assert "rbi" in out["match_text"]
    assert "RBI" in out["style_signals"]["authority_mentions"]


def test_authority_next_to_a_link_is_not_a_bare_mention(tmp_path):
    # A name with a URL beside it is a citation. The warning is the name
    # standing alone. WHO in lowercase is the English word, not the body.
    linked = ingest(
        "AIIMS published the note at https://www.aiims.edu/garlic-note and "
        "doctors there rejected the diabetes claim in writing yesterday.",
        db_path=tmp_path / "submissions.db",
    )
    assert "AIIMS" not in linked["style_signals"]["authority_mentions"]
    ordinary = ingest(
        "People who drink garlic juice every morning still need a doctor "
        "and the claim about a thirty day cure is the thing being checked.",
        db_path=tmp_path / "one.db",
    )
    assert "WHO" not in ordinary["style_signals"]["authority_mentions"]


def test_database_has_no_raw_text_name_or_phone(tmp_path):
    db = tmp_path / "submissions.db"
    ingest(GARLIC, db_path=db)
    personal = (
        "Ramesh Kumar says call 9876543210 because the reserve bank has "
        "announced that all five hundred rupee notes printed before 2016 "
        "will be invalid from 1 December this year"
    )
    ingest(personal, db_path=db)
    blob = db.read_bytes()
    assert GARLIC.encode("utf-8") not in blob
    assert "Forwarded as received".encode("utf-8") not in blob
    assert b"Ramesh" not in blob
    assert b"9876543210" not in blob
    assert b"Kumar" not in blob
    with sqlite3.connect(db) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(submissions)")]
        assert columns == ["sha256", "minhash", "first_seen", "count"]
        stored = list(conn.execute("SELECT sha256, minhash, first_seen, count FROM submissions"))
    assert stored
    for sha, signature, first_seen, count in stored:
        assert isinstance(sha, str) and len(sha) == 64
        assert isinstance(signature, bytes) and len(signature) == 128 * 8
        assert "garlic" not in sha
        assert "Ramesh" not in first_seen
        assert "98765" not in first_seen
        assert count >= 1
        # The signature is hashes, not the shingles they came from.
        assert b"garlic" not in signature
        assert b"Ramesh" not in signature
        assert b"9876543210" not in signature


def test_hinglish_wins_when_langdetect_disagrees(tmp_path):
    # This sentence is the case the detector gets wrong. The word list
    # has to accept it, and the run has to stay checkable.
    from langdetect import DetectorFactory, detect

    sentence = "Modi ji ne bola ki 500 ke note band ho jayenge from 1 December"
    assert DetectorFactory.seed == 0
    assert detect(sentence) == detect(sentence)
    assert detect(sentence) != "en"
    out = ingest(sentence, db_path=tmp_path / "submissions.db")
    assert out["hinglish"] is True
    assert out["checkable_hint"] is True
    assert out["checkable_reason"] == ""
    assert out["language"] == "en"


def test_langdetect_seed_is_zero():
    from langdetect import DetectorFactory

    assert DetectorFactory.seed == 0


def test_share_price_is_not_a_share_request(tmp_path):
    out = ingest(
        "HDFC Bank share price fell 10% today",
        db_path=tmp_path / "submissions.db",
    )
    assert "share price" in out["text"]
    assert "10%" in out["text"]
    assert out["style_signals"]["share_request"] is False

    mixed = ingest(
        "Please share this with your family. HDFC Bank share price fell "
        "10% today after the quarterly results were announced in Mumbai.",
        db_path=tmp_path / "other.db",
    )
    assert "share price" in mixed["text"]
    assert "with your family" not in mixed["text"].lower()
    assert mixed["style_signals"]["share_request"] is True


def test_tests_do_not_write_the_repo_submissions_db():
    # conftest points SUBMISSIONS_DB at a temp file. Calling ingest with
    # no db_path must follow that, and must leave the repo file alone.
    real = config.DATA_DIR / "submissions.db"
    assert Path(config.SUBMISSIONS_DB) != real
    before = real.exists()
    ingest(
        "The reserve bank has announced that old five hundred rupee notes remain valid today"
    )
    assert Path(config.SUBMISSIONS_DB).is_file()
    assert real.exists() is before


def test_record_submissions_false_writes_nothing(monkeypatch, tmp_path):
    target = tmp_path / "eval" / "submissions.db"
    monkeypatch.setattr(config, "SUBMISSIONS_DB", target)
    monkeypatch.setattr(config, "RECORD_SUBMISSIONS", False)
    out = ingest(GARLIC)
    assert out["submission_count"] == 0
    assert out["dedupe"]["seen_count"] == 0
    assert out["dedupe"]["method"] == "skipped"
    assert out["submission_hash"]
    assert not target.exists()
    blank = Image.new("RGB", (40, 40), "white")
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")
    image_out = ingest(buffer.getvalue(), input_type="image")
    assert image_out["submission_count"] == 0
    assert not target.exists()


class _Response:
    def __init__(self, status, headers, body=b""):
        self.status_code = status
        self.headers = dict(headers)
        self._body = body

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        return None


def _public_dns(monkeypatch):
    def getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]

    monkeypatch.setattr(fetch.socket, "getaddrinfo", getaddrinfo)


def test_article_url_uses_fixture_and_strips_tracking(monkeypatch, tmp_path):
    html = (FIXTURES / "article.html").read_bytes()
    seen = []

    def http(url):
        seen.append(url)
        return _Response(200, {"Content-Type": "text/html; charset=utf-8"}, html)

    monkeypatch.setattr(fetch, "_http_get", http)
    _public_dns(monkeypatch)
    db = tmp_path / "submissions.db"
    out = ingest(
        "https://www.example.com/news/rbi-notes?utm_source=whatsapp&utm_medium=share&id=99",
        input_type="url",
        db_path=db,
    )
    assert seen == ["https://example.com/news/rbi-notes?id=99"]
    assert out["headline"] == "BREAKING: RBI says old notes stay valid"
    assert out["article_date"] == "2024-03-15"
    assert "legal tender" in out["body"]
    assert "UNIQUE_TAIL_MARKER" in out["body"]
    assert out["text"].startswith("BREAKING: RBI says old notes stay valid")
    assert "legal tender" in out["text"]
    assert "UNIQUE_TAIL_MARKER" not in out["text"]
    assert "9876543210" not in out["text"]
    assert len(out["body"]) > 1200
    assert "breaking" in out["style_signals"]["urgency_words"]
    assert out["checkable_hint"] is True
    blob = db.read_bytes()
    assert b"9876543210" not in blob
    assert b"Ramesh" not in blob
    assert b"legal tender" not in blob
    assert b"UNIQUE_TAIL_MARKER" not in blob


def test_social_url_keeps_only_the_preview(monkeypatch, tmp_path):
    html = (FIXTURES / "social_post.html").read_bytes()

    def http(url):
        return _Response(200, {"Content-Type": "text/html; charset=utf-8"}, html)

    monkeypatch.setattr(fetch, "_http_get", http)
    _public_dns(monkeypatch)
    out = ingest(
        "https://www.facebook.com/story.php?utm_source=whatsapp&id=44",
        input_type="url",
        db_path=tmp_path / "submissions.db",
    )
    assert out["headline"] == "BREAKING schools closed"
    assert out["body"] == (
        "All Bengaluru schools closed till Monday due to heavy rain "
        "according to the message being shared"
    )
    assert "Log in to see this post" not in out["text"]
    assert "Log in to see this post" not in out["body"]
    assert "comment thread" not in out["text"]
    assert "schools closed" in out["text"].lower()
    assert "breaking" in out["style_signals"]["urgency_words"]
    assert out["article_date"] == ""


def test_url_style_uses_the_lede_and_ignores_icons(monkeypatch, tmp_path):
    html = """<!DOCTYPE html><html><head><title>Schools stay open</title></head><body><article>
<h1>Schools stay open</h1>
<p>The city said schools will stay open on Monday.</p>
<p>Parents should send children as usual.</p>
<p>BREAKING share this 🚨🚨🚨 and forward to ten groups.</p>
</article></body></html>""".encode()

    def http(url):
        return _Response(200, {"Content-Type": "text/html; charset=utf-8"}, html)

    monkeypatch.setattr(fetch, "_http_get", http)
    _public_dns(monkeypatch)
    out = ingest("https://example.com/schools", input_type="url", db_path=tmp_path / "submissions.db")
    signals = out["style_signals"]
    # The shout and the icons are in the third paragraph. Style sees the
    # headline and the first two only, and a URL never counts emoji.
    assert signals["emoji_count"] == 0
    assert signals["urgency_words"] == []
    assert signals["share_request"] is False


def test_private_url_is_not_fetched(monkeypatch, tmp_path):
    calls = []

    def http(url):
        calls.append(url)
        raise AssertionError(url)

    monkeypatch.setattr(fetch, "_http_get", http)
    out = ingest("http://127.0.0.1/secret", input_type="url", db_path=tmp_path / "submissions.db")
    assert calls == []
    assert out["checkable_hint"] is False
    assert out["checkable_reason"] == "could not fetch the page"
    assert not (tmp_path / "submissions.db").exists()


def _font(size: int):
    if _FONT.is_file():
        return ImageFont.truetype(str(_FONT), size)
    return ImageFont.load_default(size=size)


def _drawn(text: str, path: Path) -> None:
    image = Image.new("RGB", (1400, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 80), text, fill="black", font=_font(48))
    image.save(path, format="PNG")


def test_image_ocr_and_rejects_a_non_image(tmp_path):
    path = tmp_path / "notice.png"
    _drawn("Schools closed till Monday", path)
    out = ingest(str(path), input_type="image", caption="Shared in Bengaluru today", db_path=tmp_path / "submissions.db")
    assert "closed" in out["ocr_text"].lower()
    assert "monday" in out["text"].lower()
    assert "Bengaluru" in out["text"]
    assert "bytes" not in out
    assert "image" not in out
    stored = Path(out["image_path"])
    assert stored.is_file()
    assert stored.parent == Path(config.UPLOADS_DIR)
    assert stored.parent != config.DATA_DIR / "uploads"
    assert stored.suffix == ".jpg"
    assert out["image_sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()
    reloaded = Image.open(stored)
    assert max(reloaded.size) <= 1600
    assert not reloaded.getexif()
    assert out["checkable_hint"] is True
    note = tmp_path / "notes.txt"
    note.write_text("this is not an image, it is a paragraph of text", encoding="utf-8")
    with pytest.raises(ValueError, match="not an image"):
        ingest(str(note), input_type="image")


def test_image_strips_exif_and_near_duplicates(tmp_path):
    path = tmp_path / "tagged.jpg"
    image = Image.new("RGB", (900, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 90), "Schools closed till Monday", fill="black", font=_font(42))
    exif = image.getexif()
    exif[0x0131] = "SECRET-GPS-12.3456"
    image.save(path, format="JPEG", exif=exif)
    db = tmp_path / "submissions.db"
    first = ingest(path.read_bytes(), input_type="image", db_path=db)
    saved = Path(first["image_path"]).read_bytes()
    assert b"SECRET-GPS-12.3456" not in saved
    assert "bytes" not in first
    # One pixel changes the file hash. The picture is the same, so phash
    # has to join it to the row the first file already opened.
    edited = path.read_bytes()
    twin = Image.open(io.BytesIO(edited))
    twin.putpixel((0, 0), (255, 0, 0))
    buffer = io.BytesIO()
    twin.save(buffer, format="PNG")
    second = ingest(buffer.getvalue(), input_type="image", db_path=db)
    assert first["submission_hash"] == second["submission_hash"]
    assert second["submission_count"] == 2
    assert second["dedupe"]["method"] == "phash"
    blob = db.read_bytes()
    assert b"SECRET-GPS" not in blob
    assert b"Schools closed" not in blob
    with sqlite3.connect(db) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(images)")]
    assert columns == ["sha256", "phash", "first_seen", "count"]


_NOTICE = "All Bengaluru schools closed till Monday due to heavy rain"


def _notice(font_size: int, width: int, height: int, y: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, y), _NOTICE, fill="black", font=_font(font_size))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_ocr_reads_a_normal_font_and_a_small_one(tmp_path):
    # The exact strings are asserted, not just a substring, so a change
    # in the recogniser shows up as the text it actually returned.
    normal = ingest(
        _notice(48, 1600, 220, 80),
        input_type="image",
        db_path=tmp_path / "normal.db",
    )
    small = ingest(
        _notice(16, 1100, 80, 28),
        input_type="image",
        db_path=tmp_path / "small.db",
    )
    assert "Bengaluru" in normal["ocr_text"]
    assert "schools closed" in normal["ocr_text"]
    assert "Bengaluru" in small["ocr_text"]
    assert "schools closed" in small["ocr_text"]
    assert normal["ocr_text"] == _NOTICE
    assert small["ocr_text"] == _NOTICE
    assert normal["ocr_mean_conf"] >= 60
    assert small["ocr_mean_conf"] >= 60
    # A live check may already have created data/uploads. This test must
    # still write only under the temporary UPLOADS_DIR, never into that tree.
    uploads = config.DATA_DIR / "uploads"
    assert Path(normal["image_path"]).parent == Path(config.UPLOADS_DIR)
    assert Path(small["image_path"]).parent == Path(config.UPLOADS_DIR)
    assert uploads not in Path(normal["image_path"]).parents
    assert uploads not in Path(small["image_path"]).parents


def test_image_over_the_side_limit_is_rejected():
    image = Image.new("RGB", (4001, 8), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with pytest.raises(ValueError, match="4000"):
        ingest(buffer.getvalue(), input_type="image")


def _caps_ratio(raw: str) -> float:
    """The spec: all-caps tokens of 5 letters or fewer are not shouting."""
    upper = 0
    total = 0
    for token in re.findall(r"[A-Za-z]+", raw):
        if token.isupper() and len(token) <= 5:
            continue
        total += len(token)
        upper += sum(ch.isupper() for ch in token)
    return round(upper / total, 4) if total else 0.0


def test_hsbc_sentence_is_not_alarm_and_garlic_still_is(tmp_path):
    hsbc = ingest(
        "HSBC Bank is completely withdrawing from Sri Lanka",
        db_path=tmp_path / "hsbc.db",
    )
    assert alarm_style(hsbc["style_signals"]) is False
    garlic = ingest(GARLIC, db_path=tmp_path / "garlic.db")
    assert alarm_style(garlic["style_signals"]) is True


def test_image_over_ten_megabytes_is_rejected():
    payload = b"\x89PNG" + b"x" * (10 * 1024 * 1024)
    with pytest.raises(ValueError, match="10 MB"):
        ingest(payload, input_type="image")


def test_sample_rbi_forward_matches_the_fixture_hash(tmp_path):
    # The fixture's match_text and submission_hash are this box's output
    # for the sample forward, so the contract example is not a second dialect.
    raw = (
        "RBI has announced that all ₹500 notes printed before 2016 will be "
        "invalid from 1 December. Forward to all groups"
    )
    out = ingest(raw, db_path=tmp_path / "submissions.db")
    assert out["text"] == (
        "RBI has announced that all ₹500 notes printed before 2016 will be "
        "invalid from 1 December."
    )
    assert out["match_text"] == (
        "rbi has announced that all ₹500 notes printed before 2016 will be "
        "invalid from 1 december"
    )
    assert out["style_signals"]["share_request"] is True
    assert out["style_signals"]["authority_mentions"] == ["RBI"]
    # RBI is three capitals. The old ratio, 0.0602, counted them.
    assert out["style_signals"]["caps_ratio"] == _caps_ratio(raw)
    assert out["style_signals"]["caps_ratio"] < 0.0602
    assert out["submission_hash"] == (
        "c9c4b414c2739182479b83b46a5fff47007d8393071d10ecf8b421aee70d6d98"
    )
    assert out["checkable_hint"] is True
