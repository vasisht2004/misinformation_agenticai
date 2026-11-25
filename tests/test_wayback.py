"""Wayback existed_by dates. No live archive.org call.

The CDX body is a fake response. DNS is a stub. The default fixture
replaces existed_by, and one test shows that replacement never opens a
socket. The tests that parse a capture put the real lookup back and
stub fetch._http_get, which is the client wayback is required to use.
"""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

import misinfo_agent.fetch as fetch
import misinfo_agent.wayback as wayback
from misinfo_agent.fetch import Page
from misinfo_agent.llm import clear_fake as clear_fake_llm
from misinfo_agent.llm import set_fake as set_fake_llm
from misinfo_agent.output import photo_lean
from misinfo_agent.provenance import apply_wayback, investigate_picture
from misinfo_agent.vision import clear_fake, set_fake


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


def _restore_reader(monkeypatch):
    """Put the real lookup back. The fixture replaces both names per test."""
    monkeypatch.setattr(wayback, "existed_by", wayback._existed_by)
    monkeypatch.setattr(wayback, "_read_public", _READ_PUBLIC)


# Bound at import, before a test replaces the name. conftest's fixture
# runs per test, after this module is imported, so this is the real reader.
_READ_PUBLIC = wayback._read_public


@pytest.fixture(autouse=True)
def _feature_on(monkeypatch):
    """The product default may be off after the gold-True check.

    These tests are the mechanism. They turn the lookup on, and the two
    tests that cover the switches turn it off themselves.
    """
    monkeypatch.setattr(wayback, "ENABLED", True)
    monkeypatch.setattr(wayback, "DATE_PARTIALS", True)


def test_the_default_test_run_does_not_open_a_socket(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(fetch, "_request", boom)
    assert wayback.existed_by("https://example.com/a") is None


def test_earliest_capture_is_an_existed_by_day(monkeypatch):
    _restore_reader(monkeypatch)
    _public_dns(monkeypatch)
    seen = []

    def http(url):
        seen.append(url)
        body = json.dumps([["timestamp"], ["20150315010101"]]).encode()
        return _Response(200, {"Content-Type": "application/json"}, body)

    monkeypatch.setattr(fetch, "_http_get", http)
    assert wayback.existed_by("https://example.com/a") == date(2015, 3, 15)
    parts = urlsplit(seen[0])
    query = parse_qs(parts.query)
    assert parts.scheme == "https"
    assert parts.netloc == "web.archive.org"
    assert parts.path == "/cdx/search/cdx"
    assert query["url"] == ["https://example.com/a"]
    assert query["output"] == ["json"]
    assert query["limit"] == ["1"]
    assert query["fl"] == ["timestamp"]
    assert query["filter"] == ["statuscode:200"]
    # The day is cached. A second check must not open the client again.
    assert wayback.existed_by("https://example.com/a") == date(2015, 3, 15)
    assert len(seen) == 1


def test_an_empty_table_is_cached_and_a_bad_body_is_not(monkeypatch):
    _restore_reader(monkeypatch)
    _public_dns(monkeypatch)
    calls = {"n": 0}

    def http(url):
        calls["n"] += 1
        if "empty" in url:
            return _Response(200, {"Content-Type": "application/json"}, b"[]")
        return _Response(200, {"Content-Type": "text/html"}, b"<html>nope</html>")

    monkeypatch.setattr(fetch, "_http_get", http)
    assert wayback.existed_by("https://example.com/empty") is None
    assert wayback.existed_by("https://example.com/empty") is None
    assert calls["n"] == 1
    assert wayback.existed_by("https://example.com/bad") is None
    assert wayback.existed_by("https://example.com/bad") is None
    assert calls["n"] == 3


def test_a_refused_client_is_not_a_date(monkeypatch):
    _restore_reader(monkeypatch)
    _public_dns(monkeypatch)

    def http(url):
        raise RuntimeError("offline test tried to fetch a page")

    monkeypatch.setattr(fetch, "_http_get", http)
    assert wayback.existed_by("https://example.com/a") is None
    # Not cached: the next call still tries, and still does not invent a day.
    assert wayback.existed_by("https://example.com/a") is None


def test_a_redirect_to_a_private_address_is_not_followed(monkeypatch):
    _restore_reader(monkeypatch)
    _public_dns(monkeypatch)
    seen = []

    def http(url):
        seen.append(url)
        if "127.0.0.1" in url:
            raise AssertionError("private host was requested")
        return _Response(302, {"Location": "http://127.0.0.1/secret"}, b"")

    monkeypatch.setattr(fetch, "_http_get", http)
    assert wayback.existed_by("https://example.com/a") is None
    assert seen
    assert all("127.0.0.1" not in url for url in seen)


def test_wayback_never_makes_a_date_later(monkeypatch):
    def fake(url):
        if "cdn" in url:
            return date(2014, 1, 1)
        if "partial" in url:
            return date(2015, 3, 15)
        if "same" in url:
            return date(2018, 1, 1)
        if "undated" in url:
            return date(2016, 5, 1)
        return date(2019, 6, 1)

    monkeypatch.setattr(wayback, "existed_by", fake)
    pages = [
        {
            "url": "https://news.example/later",
            "match": "full",
            "date": "2018-01-01",
            "date_source": "htmldate",
            "image_urls": [],
        },
        {
            "url": "https://news.example/same",
            "match": "full",
            "date": "2018-01-01",
            "date_source": "htmldate",
            "image_urls": [],
        },
        {
            "url": "https://news.example/undated",
            "match": "full",
            "date": None,
            "date_source": None,
            "image_urls": ["https://cdn.example/a.jpg"],
        },
        {
            "url": "https://news.example/partial",
            "match": "partial",
            "date": None,
            "date_source": None,
            "image_urls": ["https://cdn.example/lookalike.jpg"],
        },
        {
            "url": "https://news.example/page",
            "match": "page",
            "date": None,
            "date_source": None,
            "image_urls": ["https://cdn.example/a.jpg"],
        },
        {
            "url": "https://news.example/untrusted",
            "match": "full",
            "date": "2019-08-01",
            "date_source": "untrusted",
            "image_urls": [],
        },
    ]
    apply_wayback(pages)
    by_url = {row["url"]: row for row in pages}
    # 2019 is after the page's own 2018. The byline stays.
    assert by_url["https://news.example/later"]["date"] == "2018-01-01"
    assert by_url["https://news.example/later"]["date_source"] == "htmldate"
    # The same day is not earlier.
    assert by_url["https://news.example/same"]["date_source"] == "htmldate"
    # No trusted day, and the file was captured before the page would have been.
    assert by_url["https://news.example/undated"]["date"] == "2014-01-01"
    assert by_url["https://news.example/undated"]["date_source"] == "wayback"
    # The partial page is dated. Its file URL is not this photograph, so
    # the lookalike capture must not be the day we keep. fake() would have
    # returned 2014 for a cdn URL. The page capture is 2015.
    assert by_url["https://news.example/partial"]["date"] == "2015-03-15"
    assert by_url["https://news.example/partial"]["date_source"] == "wayback"
    # A page hit is not asked, even though a file URL is sitting on the row.
    assert by_url["https://news.example/page"]["date"] is None
    assert by_url["https://news.example/page"]["date_source"] is None
    # "untrusted" is today, not a publication day. The capture may replace it.
    assert by_url["https://news.example/untrusted"]["date"] == "2019-06-01"
    assert by_url["https://news.example/untrusted"]["date_source"] == "wayback"


def test_partials_can_be_left_undated(monkeypatch):
    asked = []

    def fake(url):
        asked.append(url)
        return date(2015, 1, 1)

    monkeypatch.setattr(wayback, "existed_by", fake)
    monkeypatch.setattr(wayback, "DATE_PARTIALS", False)
    pages = [
        {"url": "https://news.example/full", "match": "full", "date": None, "date_source": None, "image_urls": []},
        {"url": "https://news.example/partial", "match": "partial", "date": None, "date_source": None, "image_urls": []},
    ]
    apply_wayback(pages)
    assert pages[0]["date_source"] == "wayback"
    assert pages[1]["date"] is None
    assert asked == ["https://news.example/full"]


def test_turned_off_dates_nothing(monkeypatch):
    def boom(url):
        raise AssertionError("wayback was asked")

    monkeypatch.setattr(wayback, "existed_by", boom)
    monkeypatch.setattr(wayback, "ENABLED", False)
    pages = [
        {"url": "https://news.example/full", "match": "full", "date": None, "date_source": None, "image_urls": []},
    ]
    apply_wayback(pages)
    assert pages[0]["date"] is None


def test_parse_takes_the_earliest_stamp():
    body = json.dumps([["timestamp"], ["20190601000000"], ["20150315010101"]]).encode()
    assert wayback.parse_cdx(body) == date(2015, 3, 15)
    assert wayback.parse_cdx(b"[]") is None
    assert wayback.parse_cdx(b'[["timestamp"]]') is None


def _jpeg(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), "blue").save(path, format="JPEG")
    return path


def test_provenance_uses_an_earlier_capture_and_the_lean_shows_the_year(tmp_path, monkeypatch):
    """The pipeline path, with a fake CDX body and no socket."""
    _restore_reader(monkeypatch)
    _public_dns(monkeypatch)
    jpeg = _jpeg(tmp_path)
    page_url = "https://www.bbc.com/old-flood"
    image_url = "https://cdn.example/flood.jpg"
    set_fake(
        lambda image: {
            "best_guess": [],
            "pages": [
                {
                    "url": page_url,
                    "title": "Old flood",
                    "match": "full",
                    "image_urls": [image_url],
                }
            ],
        }
    )

    def _fetch(hit):
        return Page(
            url=hit.url,
            title=hit.title,
            text="The photograph shows a flooded street from an earlier year entirely.",
            published_date="2019-08-01",
            date_source="htmldate",
            fetched=True,
            fetch_status="ok",
        )

    monkeypatch.setattr("misinfo_agent.fetch.fetch_page", _fetch)

    def http(url):
        # The file was archived before the article. That is the day we keep.
        if "cdn.example" in url:
            stamp = "20140101000000"
        else:
            stamp = "20190601000000"
        body = json.dumps([["timestamp"], [stamp]]).encode()
        return _Response(200, {"Content-Type": "application/json"}, body)

    monkeypatch.setattr(fetch, "_http_get", http)

    def _llm(prompt, schema, images):
        if schema.__name__ == "ProvenanceBatch":
            raise AssertionError("an earlier capture must not ask for a stance")
        return {"consistent": "unclear", "mismatches": [], "place_clues": [], "time_clues": []}

    set_fake_llm(_llm)
    try:
        result = investigate_picture(
            {
                "input_type": "image",
                "image_path": str(jpeg),
                "caption": "This photo shows the Chennai flood today.",
                "claim": "This photo shows the Chennai flood today.",
                "claim_date": "2026-03-10",
                "checkable": True,
            }
        )
    finally:
        clear_fake()
        clear_fake_llm()
    match = result["image_provenance"]["matches"][0]
    assert match["date"] == "2014-01-01"
    assert match["date_source"] == "wayback"
    assert match["stance"] == "refutes"
    seen = result["image_provenance"]["first_seen"]
    assert seen["date"] == "2014-01-01"
    assert seen["date_source"] == "wayback"
    lean = photo_lean(
        {
            "input_type": "image",
            "caption": "This photo shows the Chennai flood today.",
            "claim": "This photo shows the Chennai flood today.",
            "claim_date": "2026-03-10",
            "image_provenance": result["image_provenance"],
        }
    )
    assert lean["first_seen_year"] == 2014
    line = f"This photo has been online since at least {lean['first_seen_year']} ({lean['n_sites']} sites)"
    assert line == "This photo has been online since at least 2014 (1 sites)"
