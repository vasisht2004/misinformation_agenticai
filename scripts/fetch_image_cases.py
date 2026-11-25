"""Download photograph candidates for the image eval. No model, no Vision, no Tavily.

The 25 rows come from scripts/find_image_cases.py. Each article is opened
with misinfo_agent/fetch.py, which is the only downloader that already
refuses a private address and stores the page text in the page cache.
That cache keeps the article words, not the tags, so the HTML is read
again with the same guard: og:image and the pictures in the body are in
the markup, and the cached text does not have them.

A photograph is not HTML. fetch._download refuses a body that is not a
page, so each image uses the same address check, the same GET, and the
same redirect rule, and then keeps a JPEG, PNG, or WebP. GIF and SVG are
dropped. Nothing here calls a search engine. The four genuine photos are
taken from the English PIB gallery page, which lists them.

Usage:
    python scripts/fetch_image_cases.py
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.fetch import (  # noqa: E402
    FetchError,
    _MAX_REDIRECTS,
    _REDIRECTS,
    _check_deadline,
    _download,
    _http_get,
    _read_body,
    _refuse_unless_public,
    fetch_page,
)
from misinfo_agent.search import Hit  # noqa: E402

# Six pictures is enough to see the claim and the distractors on one
# article. More than that is the site's chrome repeated. Five megabytes
# is a photograph, not a print-resolution scan we would re-encode anyway.
MAX_PER_ARTICLE = 6
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_SIDE = 300
# A strip this wide relative to its height is a row of pictures pasted
# together, or a banner. A normal frame, and the usual 1.9:1 social card,
# sit under it.
_COLLAGE_RATIO = 2.6
# Tries, not saves. A logo that only reveals its size after the download
# must not push a real photograph off the end of the list, and it must
# not turn into a crawl of every thumbnail on the page.
_MAX_TRIES = 10

CANDIDATE_DIR = config.EVAL_DIR / "images" / "candidates"
DRAFT_PATH = config.EVAL_DIR / "images_testset_draft.csv"
SHEET_PATH = CANDIDATE_DIR / "contact_sheet.jpg"
SHEET_CSV = CANDIDATE_DIR / "contact_sheet.csv"

# The English photo gallery linked as "Photos". lang=1 keeps the English
# caption; the default redirect follows the browser toward Hindi.
# Photogallery.aspx repeats one caption across every angle of an event.
# PhotogalleryNew.aspx is the page that gallery links to, and each frame
# there has its own line. This URL is the page, not a search.
PIB_GALLERY = "https://www.pib.gov.in/PhotogalleryNew.aspx?reg=3&lang=1"

# Tokens, not substrings. "ad" must not fire inside "upload", and "icon"
# must not fire inside a word that merely contains those letters.
_JUNK_TOKENS = frozenset({
    "logo", "logos", "icon", "icons", "avatar", "author", "authors",
    "byline", "headshot", "profile", "gravatar",
    "ad", "ads", "advert", "adverts", "advertisement", "sponsor", "sponsored",
    "doubleclick", "taboola", "outbrain", "googleads",
    "sprite", "emoji", "badge", "favicon", "pixel", "tracking", "spacer",
    "placeholder", "share", "social", "playbutton", "emoji", "badges",
})
# A platform name is not junk by itself. The picture a fact-check is about
# is often the file the forwarder saved, named WhatsApp-Image-….jpg. A share
# button still matches "share", "icon", or "logo", or sits in chrome.
_CHROME_TOKENS = frozenset({
    "sidebar", "related", "widget", "author", "comment", "comments",
    "footer", "nav", "menu", "share", "advert", "promo", "recommended",
    "popular", "trending", "newsletter", "cookie", "banner", "banners",
    "icon", "icons",
})
# The class or id of the element that holds the article, not the whole
# template. The first match with real text wins, so a "content" wrapper
# around the header does not beat the entry that holds the paragraphs.
_BODY_MARKS = (
    "entry-content", "post-content", "article-content", "article-body",
    "article__body", "article__content", "story-body", "story-content",
    "td-post-content", "single-content", "post-body", "content-body",
    "news-content", "articlebody", "main_article",
)
_COLLAGE_WORDS = (
    "collage", "composite", "side-by-side", "sidebyside",
    "before-after", "before_after", "comparison",
)
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_DAY = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_DAY_MONTH = re.compile(
    r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+")
_PX = re.compile(r"^\s*(\d+)")


def map_archive_label(verdict: str) -> str:
    """False or Misleading. These rows are out-of-context photographs.

    The archive already stores those two words. A longer site label that
    still says false or misleading lands on the same two, because the
    image eval only scores those.
    """
    text = " ".join(str(verdict or "").split()).casefold()
    if text == "misleading" or "mislead" in text:
        return "Misleading"
    if text == "false" or "false" in text or text in {"fake", "fabricated"}:
        return "False"
    return "Misleading"


def date_in_text(text: str) -> str:
    """YYYY-MM-DD written in a caption, or empty when the caption has no day."""
    match = _MONTH_DAY.search(text or "")
    if match:
        month = _MONTHS[match.group(1).casefold()]
        return _iso(int(match.group(3)), month, int(match.group(2)))
    match = _DAY_MONTH.search(text or "")
    if match:
        month = _MONTHS[match.group(2).casefold()]
        return _iso(int(match.group(3)), month, int(match.group(1)))
    return ""


def _iso(year: int, month: int, day: int) -> str:
    try:
        from datetime import date
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def _tokens(*parts: str) -> set[str]:
    found = set()
    for part in parts:
        found.update(_TOKEN.findall(str(part or "").casefold()))
    return found


def is_junk(*parts: str) -> bool:
    """True for a logo, an icon, an author headshot, an ad, a GIF, or an SVG.

    The decision is the tokens of the URL, the class, and the alt text.
    A photograph whose caption happens to mention a logo is not junk: the
    caller passes the address and the class, not the article.
    """
    blob = " ".join(str(part or "") for part in parts).casefold().split("?", 1)[0]
    if blob.startswith("data:image/gif") or blob.startswith("data:image/svg"):
        return True
    path = urlsplit(blob).path if "://" in blob or blob.startswith("/") else blob
    if path.endswith((".gif", ".svg", ".svgz")):
        return True
    if "<svg" in blob:
        return True
    return bool(_tokens(blob) & _JUNK_TOKENS)


def declared_too_small(width, height) -> bool:
    """True when the tag already says a side is under 300.

    A missing side is not a reason to drop the picture. Fact-check pages
    often omit the attributes and put the real size in the file. A tag
    that does say 110 by 96 is a thumbnail, and downloading it would only
    confirm that.
    """
    sides = []
    for value in (width, height):
        match = _PX.match(str(value or ""))
        if match:
            sides.append(int(match.group(1)))
    if not sides:
        return False
    return min(sides) < MIN_SIDE


def obvious_collage(url: str, alt: str, width: int, height: int) -> bool:
    """True when the picture is a strip of several photos, or says so.

    The words are the editor's own label. The ratio catches a banner that
    nobody labelled. A single frame is not wide enough to trip it.
    """
    blob = f"{url} {alt}".casefold()
    if any(word in blob for word in _COLLAGE_WORDS):
        return True
    if width > 0 and height > 0 and (width / height) >= _COLLAGE_RATIO:
        return True
    return False


def choose_suggestion(saved: list[dict]) -> dict | None:
    """og:image when it is one photograph, otherwise the first large body image.

    A collage is the editor's comparison, not the picture the claim was
    about. The body order is the order on the page, which is the order a
    reader meets the photograph.
    """
    for kind in ("og", "body"):
        for item in saved:
            if item.get("kind") == kind and not item.get("collage"):
                return item
    return saved[0] if saved else None


def distinct_entries(entries: list[dict]) -> list[dict]:
    """One frame per official caption.

    A PIB event is several angles with the same line. Four copies of that
    line are one release, not four.
    """
    seen = set()
    kept = []
    for entry in entries:
        caption = " ".join(str(entry.get("caption") or "").split())
        if not caption or caption in seen:
            continue
        seen.add(caption)
        kept.append(entry)
    return kept


def pib_full_url(url: str) -> str:
    """The gallery's T file is the thumbnail. H is the same frame, full size.

    PIB names them that way. Asking for the thumbnail and calling it the
    photograph would store a 770-pixel preview of a 2200-pixel release.
    """
    return re.sub(r"/T(\d+\.jpe?g)\b", r"/H\1", url, count=1, flags=re.IGNORECASE)


def _srcset_url(srcset: str, page_url: str) -> str:
    """The smallest srcset candidate that is still at least 300 wide.

    The largest file on a srcset is often the print original, which blows
    the 5 MB cap and then leaves us with nothing. The 300-wide file is the
    one the size rule already asked for.
    """
    options = []
    for part in (srcset or "").split(","):
        bits = part.strip().split()
        if not bits:
            continue
        url = urljoin(page_url, bits[0])
        width = 10**9
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                width = int(bits[1][:-1])
            except ValueError:
                width = 10**9
        options.append((width, url))
    if not options:
        return ""
    big = [item for item in options if item[0] >= MIN_SIDE]
    return sorted(big or options)[0][1]


def _direct_url(img, page_url: str) -> str:
    """data-src, then src. A gif or SVG stand-in is not the photograph."""
    for key in ("data-src", "data-lazy-src", "data-original", "src"):
        raw = (img.get(key) or "").strip()
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(page_url, raw)
        if url and not is_junk(url):
            return url
    return ""


def _both_sides_large(width, height) -> bool:
    """True only when the tag itself names both sides and both clear 300.

    A missing side is not 'large'. The srcset fallback still applies then.
    """
    sides = []
    for value in (width, height):
        match = _PX.match(str(value or ""))
        if not match:
            return False
        sides.append(int(match.group(1)))
    return min(sides) >= MIN_SIDE


def _image_url(img, page_url: str) -> str:
    """The address to download. A placeholder src is not that address.

    When the tag already says the frame is at least 300 on both sides, that
    src is the frame. The srcset's smallest 300-wide file is often a short
    preview that then fails the pixel check, so it is only the fallback for
    a tag that did not name its size. Lazy pages put the photograph in
    data-src and a 1-pixel gif, or an SVG stand-in, in src.
    """
    if _both_sides_large(img.get("width"), img.get("height")):
        direct = _direct_url(img, page_url)
        if direct:
            return direct
    srcset = img.get("srcset") or img.get("data-srcset") or ""
    sized = _srcset_url(srcset, page_url)
    if sized and not is_junk(sized):
        return sized
    return _direct_url(img, page_url)


def _in_chrome(img, stop=None) -> bool:
    """True when a related box, a share row, or an author card wraps the image.

    The walk stops at the article. A layout column named right-sidebar often
    wraps the whole story; that name is the theme, not a sidebar widget, and
    treating it as chrome drops every photograph in the article.
    """
    node = img
    while node is not None and node is not stop:
        blob = " ".join([
            node.get("class") or "",
            node.get("id") or "",
            node.tag or "",
        ])
        if _tokens(blob) & _CHROME_TOKENS:
            return True
        node = node.getparent()
    return False


def _article_root(doc):
    """The element that holds the article, or None when the page has no body.

    A class match with almost no text is the template. The longest match
    is the story. Without one, an <article> or <main> is the fallback, and
    a page that is only a script shell contributes no body images. og:image
    is collected separately and does not need this element.
    """
    best = None
    best_len = 0
    for el in doc.xpath("//*"):
        mark = f"{el.get('class') or ''} {el.get('id') or ''}".casefold()
        if not any(token in mark for token in _BODY_MARKS):
            continue
        length = len(" ".join(el.text_content().split()))
        if length > best_len:
            best = el
            best_len = length
    if best is not None and best_len >= 200:
        return best
    articles = doc.xpath("//article")
    if articles:
        return max(articles, key=lambda el: len(el.text_content()))
    mains = doc.xpath("//main")
    if mains:
        return mains[0]
    return None


def og_urls(doc, page_url: str) -> list[str]:
    """og:image only. og:image:width is a number, and it is not a picture."""
    found = []
    seen = set()
    for meta in doc.xpath("//meta[@property or @name]"):
        key = (meta.get("property") or meta.get("name") or "").strip().casefold()
        if key not in {"og:image", "og:image:url", "og:image:secure_url"}:
            continue
        raw = (meta.get("content") or "").strip()
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(page_url, raw)
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        if is_junk(url):
            continue
        seen.add(url)
        found.append(url)
    return found


def body_candidates(doc, page_url: str) -> list[dict]:
    """<img> inside the article whose tag does not already say it is small."""
    root = _article_root(doc)
    if root is None:
        return []
    found = []
    seen = set()
    for img in root.xpath(".//img"):
        if _in_chrome(img, root):
            continue
        url = _image_url(img, page_url)
        if not url or url in seen or is_junk(url, img.get("class"), img.get("alt"), img.get("id")):
            continue
        if declared_too_small(img.get("width"), img.get("height")):
            continue
        seen.add(url)
        found.append({
            "url": url,
            "kind": "body",
            "alt": " ".join((img.get("alt") or "").split()),
        })
    return found


def candidate_list(html: bytes, page_url: str) -> list[dict]:
    """og:image first, then body pictures in reading order."""
    from lxml import html as lhtml

    try:
        doc = lhtml.document_fromstring(html)
    except Exception:
        return []
    listed = [{"url": url, "kind": "og", "alt": ""} for url in og_urls(doc, page_url)]
    seen = {item["url"] for item in listed}
    for item in body_candidates(doc, page_url):
        if item["url"] not in seen:
            seen.add(item["url"])
            listed.append(item)
    return listed


def download_public(url: str, *, limit: int, images: bool) -> tuple[str, bytes, str]:
    """One URL, every redirect checked, body capped.

    images=False is the article, and it keeps the HTML rule fetch._download
    already applies by calling that function. images=True is a photograph:
    the same hops, and an image body instead of a page.
    """
    if not images:
        return _download(url)
    deadline = time.monotonic() + config.FETCH_TIMEOUT_S
    current = url
    for hop in range(_MAX_REDIRECTS + 1):
        _check_deadline(deadline)
        _refuse_unless_public(current)
        try:
            response = _http_get(current)
        except Exception as exc:
            raise FetchError(_reason(exc)) from None
        try:
            status = int(getattr(response, "status_code", 0))
            headers = getattr(response, "headers", {}) or {}
            if status in _REDIRECTS:
                if hop == _MAX_REDIRECTS:
                    raise FetchError("status 310")
                location = headers.get("Location") or headers.get("location")
                if not location:
                    raise FetchError("status 302")
                current = urljoin(current, str(location))
                continue
            if status != 200:
                raise FetchError(f"status {status}")
            ctype = str(headers.get("Content-Type") or headers.get("content-type") or "")
            body = _read_body(response, limit, deadline)
            if not _image_bytes(ctype, current, body):
                raise FetchError("not-image")
            return current, body, ctype
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    raise FetchError("status 310")


def _reason(exc: BaseException) -> str:
    if isinstance(exc, FetchError):
        return exc.reason
    name = type(exc).__name__
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    return "failed"


def _image_bytes(content_type: str, url: str, body: bytes) -> bool:
    """JPEG, PNG, or WebP. GIF and SVG are refused even if the header lies."""
    folded = (content_type or "").casefold()
    path = urlsplit(url).path.casefold()
    if "gif" in folded or path.endswith(".gif") or body.startswith((b"GIF87a", b"GIF89a")):
        return False
    if "svg" in folded or path.endswith((".svg", ".svgz")) or body.lstrip()[:200].lower().startswith((b"<svg", b"<?xml")):
        return False
    if body.startswith(b"\xff\xd8\xff") or body.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if body.startswith(b"RIFF") and body[8:12] == b"WEBP":
        return True
    return "jpeg" in folded or "jpg" in folded or "png" in folded or "webp" in folded


def _once(url: str, *, images: bool) -> tuple[str, bytes, str]:
    """One try, and a second only when the first was a timeout.

    A dead address used to fail the whole article. The second try is the
    same fetcher, not a different one, and a 404 is not asked again.
    """
    try:
        return download_public(url, limit=MAX_IMAGE_BYTES if images else 2 * 1024 * 1024, images=images)
    except FetchError as exc:
        if exc.reason != "timeout":
            raise
        return download_public(url, limit=MAX_IMAGE_BYTES if images else 2 * 1024 * 1024, images=images)


def to_jpeg(body: bytes) -> tuple[bytes, int, int]:
    """Re-encode and drop EXIF. Returns the JPEG and the pixel size.

    exif_transpose applies the orientation tag first. The save does not
    receive an exif payload, so the file on disk has no camera metadata
    and no GPS. A side under 300 is refused by the caller.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        opened = Image.open(BytesIO(body))
        opened.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise FetchError("not-image") from exc
    upright = ImageOps.exif_transpose(opened)
    if upright.mode in {"RGBA", "LA"} or (upright.mode == "P" and "transparency" in upright.info):
        rgba = upright.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        upright = flat
    else:
        upright = upright.convert("RGB")
    width, height = upright.size
    out = BytesIO()
    upright.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue(), width, height


def _load_cases():
    spec = importlib.util.spec_from_file_location(
        "find_image_cases", ROOT / "scripts" / "find_image_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.candidates(25)


def _open_article(url: str) -> tuple[bytes | None, str]:
    """Page cache via fetch_page, then the HTML the cache does not keep.

    fetch_page stores the extracted words. The pictures are in the tags,
    so the HTML is a second read through the same guard. A failure here
    is the reason printed for an article that yielded nothing.
    """
    try:
        fetch_page(Hit(url=url))
    except Exception as exc:
        # The text cache is a courtesy for the next run. A failure there
        # does not decide whether the photographs can be read.
        print(f"  page-cache note {type(exc).__name__}", flush=True)
    try:
        _final, body, _ctype = _once(url, images=False)
    except FetchError as exc:
        return None, exc.reason
    except Exception as exc:
        return None, _reason(exc)
    return body, ""


def _save_one(item: dict, dest: Path) -> dict | None:
    try:
        _final, body, _ctype = _once(item["url"], images=True)
        jpeg, width, height = to_jpeg(body)
    except FetchError as exc:
        print(f"    skip {exc.reason} {item['url'][:80]}", flush=True)
        return None
    except Exception as exc:
        print(f"    skip {_reason(exc)} {item['url'][:80]}", flush=True)
        return None
    if width < MIN_SIDE or height < MIN_SIDE:
        print(f"    skip {width}x{height} {item['url'][:80]}", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(jpeg)
    return {
        "url": item["url"],
        "kind": item["kind"],
        "alt": item.get("alt") or "",
        "path": dest,
        "width": width,
        "height": height,
        "collage": obvious_collage(item["url"], item.get("alt") or "", width, height),
    }


def fetch_article(row_number: int, row: dict) -> tuple[list[dict], str]:
    """Saved photographs for one archive article, and why the page failed.

    The file number counts keeps, so a logo that was downloaded and then
    rejected does not leave a hole in c01_1, c01_2. The walk still stops
    after a handful of tries: the rest of a news page is the template.
    """
    print(f"c{row_number:02d} {row['url']}", flush=True)
    html, reason = _open_article(row["url"])
    if html is None:
        print(f"  page {reason}", flush=True)
        return [], reason or "no page"
    saved = []
    for attempt, item in enumerate(candidate_list(html, row["url"]), start=1):
        if len(saved) >= MAX_PER_ARTICLE or attempt > _MAX_TRIES:
            break
        dest = CANDIDATE_DIR / f"c{row_number:02d}_{len(saved) + 1}.jpg"
        got = _save_one(item, dest)
        if got is None:
            continue
        saved.append(got)
        print(f"  {dest.name} {got['width']}x{got['height']} {got['kind']}", flush=True)
    return saved, ""


def _pib_entries(html: bytes, page_url: str) -> list[dict]:
    """Gallery frames with an official caption. The emblem and the share icons are not frames."""
    from lxml import html as lhtml

    try:
        doc = lhtml.document_fromstring(html)
    except Exception:
        return []
    found = []
    seen = set()
    # The older gallery marks frames with class photo-gallery. The Photos
    # page does not, but both put the frame under PhotoGallery in the URL.
    for img in doc.xpath("//img"):
        src = urljoin(page_url, (img.get("src") or "").strip())
        caption = " ".join((img.get("alt") or img.get("title") or "").split())
        if "photogallery" not in src.casefold() or not caption or src in seen:
            continue
        seen.add(src)
        found.append({"url": src, "caption": caption, "date": date_in_text(caption)})
    return found


def fetch_genuine(limit: int = 4) -> list[dict]:
    """Four recent PIB photographs. The gallery page is fetched, not searched.

    The caption on the English gallery is the official line, and the day
    in that line is the claim date. H is the full frame of the thumbnail
    the gallery shows. A thumbnail that is all we can get is kept when it
    is still at least 300 pixels on a side.
    """
    print(f"pib {PIB_GALLERY}", flush=True)
    try:
        fetch_page(Hit(url=PIB_GALLERY))
    except Exception as exc:
        print(f"  page-cache note {type(exc).__name__}", flush=True)
    try:
        final, html, _ctype = _once(PIB_GALLERY, images=False)
    except Exception as exc:
        print(f"  gallery {_reason(exc) if not isinstance(exc, FetchError) else exc.reason}", flush=True)
        return []
    entries = distinct_entries(
        [item for item in _pib_entries(html, final) if item["date"]]
    )
    saved = []
    for entry in entries:
        if len(saved) >= limit:
            break
        full = pib_full_url(entry["url"])
        urls = [full] if full != entry["url"] else []
        urls.append(entry["url"])
        got = None
        for url in urls:
            got = _save_one(
                {"url": url, "kind": "pib", "alt": entry["caption"]},
                CANDIDATE_DIR / f"g{len(saved) + 1}.jpg",
            )
            if got is not None:
                break
        if got is None:
            continue
        got["caption"] = entry["caption"]
        got["claim_date"] = entry["date"]
        got["source"] = final
        saved.append(got)
        print(f"  {got['path'].name} {got['width']}x{got['height']} {entry['date']}", flush=True)
    return saved


def write_sheet(records: list[dict], path: Path = SHEET_PATH) -> None:
    """One grid. Each tile is about 300 pixels, with the filename in large type.

    The filename is what a person types into the draft when the suggestion
    is the wrong frame. A sheet without that label is a page of anonymous
    pictures.
    """
    from PIL import Image, ImageDraw, ImageFont

    tile = 300
    label_h = 44
    gap = 8
    columns = 6
    font = _sheet_font(28)
    if not records:
        sheet = Image.new("RGB", (tile, tile), "white")
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, format="JPEG", quality=85)
        return
    rows = (len(records) + columns - 1) // columns
    width = columns * tile + (columns + 1) * gap
    height = rows * (tile + label_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        column = index % columns
        row = index // columns
        left = gap + column * (tile + gap)
        top = gap + row * (tile + label_h + gap)
        with Image.open(record["path"]) as photo:
            photo = photo.convert("RGB")
            photo.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x = left + (tile - photo.width) // 2
            y = top + (tile - photo.height) // 2
            sheet.paste(photo, (x, y))
        name = record["path"].name
        bar_top = top + tile
        draw.rectangle([left, bar_top, left + tile, bar_top + label_h], fill=(20, 20, 20))
        box = draw.textbbox((0, 0), name, font=font)
        text_w = box[2] - box[0]
        text_h = box[3] - box[1]
        draw.text(
            (left + max(0, (tile - text_w) // 2), bar_top + max(0, (label_h - text_h) // 2) - box[1]),
            name,
            fill="white",
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=85)


def _sheet_font(size: int):
    from PIL import ImageFont

    path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def write_sheet_csv(records: list[dict]) -> None:
    with SHEET_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "article URL", "primary_claim", "archive verdict", "image width x height"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "filename": record["path"].name,
                "article URL": record.get("article_url") or "",
                "primary_claim": record.get("claim") or "",
                "archive verdict": record.get("verdict") or "",
                "image width x height": f"{record['width']}x{record['height']}",
            })


def write_draft(rows: list[dict]) -> None:
    with DRAFT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "caption", "label", "claim_date", "source_article"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    cases = _load_cases()
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    # A previous run's leftover frames would sit beside this run's files
    # and look like candidates the sheet does not list.
    for stale in CANDIDATE_DIR.glob("*.jpg"):
        # contact_sheet.jpg starts with "c" too. Only the frames we numbered.
        if re.fullmatch(r"c\d+_\d+\.jpg", stale.name) or re.fullmatch(r"g\d+\.jpg", stale.name):
            stale.unlink()
    sheet_records = []
    draft = []
    counts = []
    empty = []
    for number, row in enumerate(cases, start=1):
        saved, reason = fetch_article(number, row)
        counts.append((number, len(saved), row["url"], reason))
        if not saved:
            empty.append((number, row["url"], reason))
            continue
        for item in saved:
            item["article_url"] = row["url"]
            item["claim"] = row["primary_claim"]
            item["verdict"] = row["verdict"]
            sheet_records.append(item)
        chosen = choose_suggestion(saved)
        if chosen is None:
            empty.append((number, row["url"], "no suggestion"))
            continue
        draft.append({
            "file": _rel(chosen["path"]),
            "caption": row["primary_claim"],
            "label": map_archive_label(row["verdict"]),
            "claim_date": str(row.get("date") or "")[:10],
            "source_article": row["url"],
        })
    genuine = fetch_genuine(4)
    for item in genuine:
        item["article_url"] = item.get("source") or PIB_GALLERY
        item["claim"] = item.get("caption") or ""
        item["verdict"] = "True"
        sheet_records.append(item)
        draft.append({
            "file": _rel(item["path"]),
            "caption": item.get("caption") or "",
            "label": "True",
            "claim_date": item.get("claim_date") or "",
            "source_article": item.get("source") or PIB_GALLERY,
        })
    write_sheet(sheet_records, SHEET_PATH)
    write_sheet_csv(sheet_records)
    write_draft(draft)
    print("")
    print("candidates per article")
    for number, count, url, reason in counts:
        tail = f"  ({reason})" if reason and count == 0 else ""
        print(f"  c{number:02d}  {count}  {url}{tail}")
    print("")
    if empty:
        print("yielded nothing")
        for number, url, reason in empty:
            print(f"  c{number:02d}  {reason}  {url}")
    else:
        print("yielded nothing")
        print("  (none)")
    print(f"genuine pib photos {len(genuine)}")
    print(SHEET_PATH)
    print(DRAFT_PATH)


if __name__ == "__main__":
    main()
