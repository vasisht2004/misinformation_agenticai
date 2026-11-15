"""Page fetch, with no network. The HTTP client and DNS are stubs."""

import socket

import misinfo_agent.config as config
import misinfo_agent.fetch as fetch
from misinfo_agent.fetch import fetch_page
from misinfo_agent.search import Hit


_HTML = """<!DOCTYPE html><html><head>
<title>RBI says notes stay valid</title>
<meta property="og:site_name" content="Reserve Bank of India"/>
<meta property="article:published_time" content="2020-01-02T00:00:00Z"/>
</head><body><article>
<h1>RBI says notes stay valid</h1>
<time datetime="2020-01-02">2 January 2020</time>
<p>The Reserve Bank of India has not withdrawn five hundred rupee notes printed before 2016. The notes remain legal tender across the country and no such deadline has been notified by the bank. Holders can continue to use these notes for payments every day. Banks will keep accepting the notes at counters and cash machines, and the public does not need to exchange them for a new series by any date announced in a message. The notification people are forwarding is not on the bank website, and branches have been told to treat it as a rumour rather than a circular.</p>
</article></body></html>"""


class _Response:
    def __init__(self, status, headers, body=b""):
        self.status_code = status
        self.headers = dict(headers)
        self._body = body

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        return None


def _hit(url, content="tavily snippet about the claim", title="t", published=None):
    return Hit(url=url, title=title, content=content, score=0.5, published_date=published)


def _public_dns(monkeypatch):
    def getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]

    monkeypatch.setattr(fetch.socket, "getaddrinfo", getaddrinfo)


def test_ssrf_blocks_private_addresses_and_other_schemes(monkeypatch):
    calls = []

    def http(url):
        calls.append(url)
        raise AssertionError(url)

    monkeypatch.setattr(fetch, "_http_get", http)
    for url in (
        "http://127.0.0.1/secret",
        "http://10.0.0.5/secret",
        "http://169.254.169.254/latest/meta-data",
        "file:///etc/passwd",
    ):
        page = fetch_page(_hit(url))
        assert page.fetched is False
        assert page.fetch_status == "failed"
        assert page.failure_reason == "blocked-IP"
        assert page.text == "tavily snippet about the claim"
    assert calls == []


def test_ssrf_does_not_follow_a_redirect_to_a_private_address(monkeypatch):
    _public_dns(monkeypatch)
    calls = []

    def http(url):
        calls.append(url)
        return _Response(302, {"Location": "http://169.254.169.254/latest/meta-data"})

    monkeypatch.setattr(fetch, "_http_get", http)
    page = fetch_page(_hit("http://example.com/start"))
    assert calls == ["http://example.com/start"]
    assert page.fetched is False
    assert page.fetch_status == "failed"
    assert page.failure_reason == "blocked-IP"
    assert page.text == "tavily snippet about the claim"


def test_non_article_hosts_are_not_fetched(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "_http_get", lambda url: calls.append(url))
    for url in (
        "https://www.youtube.com/watch?v=abc",
        "https://m.facebook.com/posts/1",
        "https://x.com/someone/status/1",
        "https://twitter.com/someone/status/1",
        "https://www.instagram.com/p/abc",
        "https://t.me/channel/1",
        "https://web.whatsapp.com/",
        "https://media.gettyimages.com/photo/1",
        "https://www.shutterstock.com/image-photo/1",
        "https://in.pinterest.com/pin/1",
        "https://www.alamy.com/stock-photo-1",
        "https://www.reddit.com/r/india/comments/1",
        "https://www.linkedin.com/posts/1",
        "https://www.quora.com/What-happened",
    ):
        page = fetch_page(_hit(url, content="the snippet", published="2024-01-01"))
        assert page.fetch_status == "circulation"
        assert page.fetched is False
        assert page.text == ""
        assert page.published_date == "2024-01-01"
        assert page.date_source == "tavily"
    assert calls == []


def test_fetch_failure_falls_back_to_the_snippet(monkeypatch):
    _public_dns(monkeypatch)

    def http(url):
        raise TimeoutError("timed out")

    monkeypatch.setattr(fetch, "_http_get", http)
    page = fetch_page(_hit("https://example.com/story", content="only the snippet"))
    assert page.fetched is False
    assert page.fetch_status == "failed"
    assert page.failure_reason == "timeout"
    assert page.text == "only the snippet"


def test_successful_fetch_uses_the_page_date(monkeypatch):
    _public_dns(monkeypatch)
    body = _HTML.encode("utf-8")

    def http(url):
        return _Response(200, {"Content-Type": "text/html; charset=utf-8"}, body)

    monkeypatch.setattr(fetch, "_http_get", http)
    page = fetch_page(
        _hit(
            "https://www.rbi.org.in/note",
            content="snippet",
            title="snippet title",
            published="2019-05-01",
        )
    )
    assert page.fetched is True
    assert page.fetch_status == "ok"
    assert "legal tender" in page.text
    assert page.title == "RBI says notes stay valid"
    assert page.published_date == "2020-01-02"
    assert page.date_source == "htmldate"
    assert page.sitename == "Reserve Bank of India"


def test_missing_page_date_falls_back_to_tavily(monkeypatch):
    _public_dns(monkeypatch)
    monkeypatch.setattr(fetch, "_http_get", lambda url: _Response(200, {"Content-Type": "text/html"}, b"<html></html>"))
    monkeypatch.setattr(
        fetch,
        "_extract",
        lambda html, url: {
            "text": " ".join(["The bank notice is still in force and the notes remain usable."] * 12),
            "title": "From the page",
            "date": None,
            "sitename": None,
        },
    )
    page = fetch_page(_hit("https://example.com/undated", published="2019-05-01"))
    assert page.fetched is True
    assert page.published_date == "2019-05-01"
    assert page.date_source == "tavily"
    assert page.title == "From the page"


def _long_html(when: str) -> bytes:
    paragraph = (
        "The ministry published a notice about the scheme and the notes. "
        "People can keep using the existing notes for ordinary payments. "
        "No deadline in a forwarded message changes that position. "
        "Banks have been asked to ignore the rumour and to serve customers as usual. "
    ) * 4
    html = (
        "<!DOCTYPE html><html><head>"
        f'<meta property="article:published_time" content="{when}T00:00:00Z"/>'
        "</head><body><article>"
        f'<time datetime="{when}">{when}</time><p>'
        + paragraph
        + "</p></article></body></html>"
    )
    return html.encode("utf-8")


def test_url_date_wins_when_metadata_is_today(monkeypatch):
    from datetime import date

    _public_dns(monkeypatch)
    today = date.today().isoformat()
    body = _long_html(today)

    def http(url):
        return _Response(200, {"Content-Type": "text/html; charset=utf-8"}, body)

    monkeypatch.setattr(fetch, "_http_get", http)
    page = fetch_page(_hit("https://example.com/2016/11/08/note", published=today))
    assert page.fetch_status == "ok"
    assert page.published_date == "2016-11-08"
    assert page.date_source == "url"


def test_future_date_is_untrusted(monkeypatch):
    _public_dns(monkeypatch)
    body = _long_html("2099-01-01")
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url: _Response(200, {"Content-Type": "text/html"}, body),
    )
    page = fetch_page(_hit("https://example.com/note"))
    assert page.published_date is None
    assert page.date_source == "untrusted"


def test_short_extract_falls_back_to_the_snippet(monkeypatch):
    _public_dns(monkeypatch)
    html = b"<html><body><p>Too short.</p></body></html>"
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url: _Response(200, {"Content-Type": "text/html"}, html),
    )
    page = fetch_page(_hit("https://example.com/paywall", content="the search snippet is what we keep"))
    assert page.fetched is False
    assert page.failure_reason == "short-text"
    assert page.text == "the search snippet is what we keep"


def test_raw_content_skips_the_download(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "_http_get", lambda url: calls.append(url))
    words = " ".join(
        ["The official notice says the notes remain legal tender for payments."] * 16
    )
    hit = _hit("https://example.com/already", content="snippet")
    hit.raw_content = words
    page = fetch_page(hit)
    assert calls == []
    assert page.fetched is True
    assert page.body_from == "tavily"
    assert page.fetch_status == "ok"
    assert "legal tender" in page.text


def test_huge_raw_content_is_cut_to_the_article_head(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "_http_get", lambda url: calls.append(url))
    sentence = "The official notice says the notes remain legal tender."
    hit = _hit("https://example.com/huge", content="snippet")
    hit.raw_content = " ".join([sentence] * 2000)
    page = fetch_page(hit)
    assert calls == []
    assert page.fetch_status == "ok"
    assert len(page.text.split()) == 2500


def test_only_twelve_non_circulation_hits_are_fetched(monkeypatch):
    calls = []

    def fake_fetch(hit):
        calls.append(hit.url)
        return fetch.Page(url=hit.url, text="body", fetch_status="ok", fetched=True)

    monkeypatch.setattr(fetch, "fetch_page", fake_fetch)
    hits = [Hit(url="https://www.youtube.com/watch?v=1", content="a post")]
    hits.extend(Hit(url=f"https://news{i}.com/story", content="snippet") for i in range(13))
    pages = fetch.fetch_pages(hits)
    assert pages[0].fetch_status == "circulation"
    assert "https://www.youtube.com/watch?v=1" not in calls
    assert len(calls) == config.FETCH_MAX
    assert pages[-1].fetch_status == "snippet"
    assert pages[-1].text == "snippet"


def test_pdf_text_is_extracted(monkeypatch):
    import io

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    _public_dns(monkeypatch)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    words = " ".join(
        ["The government circular states that the existing notes remain legal tender."] * 10
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({words}) Tj ET".encode("latin-1"))
    writer.pages[0][NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    writer.pages[0][NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    buf = io.BytesIO()
    writer.write(buf)
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url: _Response(200, {"Content-Type": "application/pdf"}, buf.getvalue()),
    )
    page = fetch_page(_hit("https://example.com/circular.pdf"))
    assert page.fetch_status == "ok"
    assert page.is_pdf is True
    assert page.body_from == "pdf"
    assert "government circular" in page.text.casefold()


def test_non_english_page_is_skipped(monkeypatch):
    _public_dns(monkeypatch)
    french = (
        "Le ministere a publie un avis sur les billets et le public peut continuer "
        "a les utiliser pour les paiements ordinaires sans aucune date limite. "
        "Les banques ont recu la consigne de traiter le message transfere comme "
        "une rumeur et non comme une circulaire officielle du gouvernement. "
    ) * 4
    html = ("<html><body><article><p>" + french + "</p></article></body></html>").encode()
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url: _Response(200, {"Content-Type": "text/html"}, html),
    )
    page = fetch_page(_hit("https://example.com/fr"))
    assert page.fetch_status == "non_english"
    assert page.failure_reason == "non-English"
    assert page.text == ""


def test_a_cached_megabyte_page_is_clipped(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PAGE_CACHE_DIR", tmp_path)
    page = fetch.Page(
        url="https://example.com/huge-cache",
        text="word " * 200_000,
        fetched=True,
        fetch_status="ok",
        body_from="http",
    )
    fetch._write_cache(page.url, page)
    loaded = fetch_page(_hit(page.url, content="snippet"))
    assert loaded.fetch_status == "ok"
    assert len(loaded.text) == 20_000


def test_a_cached_megabyte_page_is_clipped(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PAGE_CACHE_DIR", tmp_path)
    page = fetch.Page(
        url="https://example.com/huge-cache",
        text="word " * 200_000,
        fetched=True,
        fetch_status="ok",
        body_from="http",
    )
    fetch._write_cache(page.url, page)
    loaded = fetch_page(_hit(page.url, content="snippet"))
    assert loaded.fetch_status == "ok"
    assert len(loaded.text) == 20_000


def test_a_trickling_body_stops_at_the_deadline(monkeypatch):
    import time

    _public_dns(monkeypatch)
    monkeypatch.setattr(config, "FETCH_TIMEOUT_S", 0.3)

    class _Slow(_Response):
        def iter_content(self, chunk_size=65536):
            for _ in range(30):
                time.sleep(0.15)
                yield b"word " * 20

    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url: _Slow(200, {"Content-Type": "text/html"}, b""),
    )
    started = time.monotonic()
    page = fetch_page(_hit("https://example.com/drip", content="the snippet"))
    assert page.failure_reason == "timeout"
    assert page.text == "the snippet"
    assert time.monotonic() - started < 2


def test_browser_user_agent_and_six_second_timeout(monkeypatch):
    _public_dns(monkeypatch)
    seen = {}

    def fake_get(url, timeout, allow_redirects, stream, headers):
        seen["timeout"] = timeout
        seen["headers"] = headers
        raise TimeoutError("slow")

    import requests

    monkeypatch.setattr(fetch, "_http_get", fetch._request)
    monkeypatch.setattr(requests, "get", fake_get)
    page = fetch_page(_hit("https://example.com/slow"))
    assert page.failure_reason == "timeout"
    assert seen["timeout"] == 6
    agent = seen["headers"]["User-Agent"]
    assert agent.startswith("Mozilla/5.0")
    assert "misinfo" not in agent.casefold()


def test_date_only_fetch_fills_a_missing_date(monkeypatch):
    import time

    import htmldate

    from misinfo_agent.investigate import Passage, finish_passages

    _public_dns(monkeypatch)
    seen = {}
    real = htmldate.find_date

    def wrapped(obj, *args, **kwargs):
        seen["length"] = len(obj) if isinstance(obj, str) else 0
        seen["original"] = kwargs.get("original_date")
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(htmldate, "find_date", wrapped)
    body = (_HTML + "<!-- " + ("x" * 200_000) + " -->").encode("utf-8")
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url: _Response(200, {"Content-Type": "text/html; charset=utf-8"}, body),
    )
    url = "https://example.com/notes"
    passage = Passage(
        passage_id="P0",
        url=url,
        netloc="example.com",
        title="t",
        text="word " * 30,
        cos=0.6,
        score=0.6,
        fetched=True,
    )
    done = finish_passages(
        {"claim_date": "2024-01-01"},
        [passage],
        deadline=time.monotonic() + 10,
    )
    assert done["passages"][0].published_date == "2020-01-02"
    assert done["passages"][0].date_source == "htmldate"
    assert seen["original"] is True
    # The pad is past the cap. Only the head is handed to htmldate.
    assert seen["length"] <= config.DATE_FETCH_BYTES
    assert seen["length"] < len(body)


def test_article_text_keeps_a_date_when_the_host_blocks_the_fetch():
    """Samosa's moneycontrol and ndtv pages. The HTML fetch now returns 403.

    Day 2 dated moneycontrol from the HTML and left that date off the page
    cache, which had been stored from Tavily text. Day 3 read the cache,
    the host refused the date-only fetch, and the date was unknown. The
    publication line is still in the text, and an event date in the same
    article is not the page's date.
    """
    from misinfo_agent.fetch import Page, _write_cache, publication_date_in_text

    assert publication_date_in_text(
        'This Article is From Jul 15, 2025 "Misleading": Health Ministry'
    ) == "2025-07-15"
    assert publication_date_in_text(
        "— PIB Fact Check (@PIBFactCheck) July 15, 2025\nThe ministry said no labels were ordered."
    ) == "2025-07-15"
    assert publication_date_in_text(
        "The notes were demonetised on November 8, 2016 and that is the event the article describes."
    ) is None

    ndtv = (
        '"Misleading": Health Ministry On Warning Labels For Samosas, Jalebis '
        "This Article is From Jul 15, 2025 The Health Ministry denied directing "
        "vendors to put warning labels on food items like samosas and jalebis."
    )
    page = Page(
        url="https://www.ndtv.com/india-news/samosas-8880316",
        title="Misleading",
        text=ndtv,
        published_date=None,
        fetched=True,
        fetch_status="ok",
        body_from="tavily",
    )
    _write_cache(page.url, page)
    loaded = fetch_page(Hit(url=page.url, title="Misleading", content="snippet", raw_content="short"))
    assert loaded.published_date == "2025-07-15"
    assert loaded.date_source == "text"
    # Written back, so the next read does not depend on the host answering.
    again = fetch_page(Hit(url=page.url, title="Misleading", content="snippet", raw_content="short"))
    assert again.published_date == "2025-07-15"
    assert again.date_source == "text"
