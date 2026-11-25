"""The image-set rules. No page is opened and no picture is downloaded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "fetch_image_cases", ROOT / "scripts" / "fetch_image_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load()


def test_junk_and_size_and_collage_rules():
    assert cases.is_junk("https://cdn.example/logo.png")
    assert cases.is_junk("https://cdn.example/author-avatar.jpg")
    assert cases.is_junk("/static/icon.svg")
    assert cases.is_junk("https://cdn.example/storage/badges/seal.png")
    assert cases.is_junk("data:image/gif;base64,AAAA")
    assert not cases.is_junk("https://cdn.example/nepal-flood.jpg", "children in a flood")
    # The viral file is often named for the app that forwarded it.
    assert not cases.is_junk(
        "https://www.vishvasnews.com/wp-content/uploads/2026/08/WhatsApp-Image-2026-08-10-at-5.08.33-PM.jpeg"
    )
    assert cases.declared_too_small("110", "96")
    assert not cases.declared_too_small("640", "480")
    assert not cases.declared_too_small(None, None)
    assert cases.obvious_collage("https://cdn.example/before-after.jpg", "", 800, 600)
    assert cases.obvious_collage("https://cdn.example/frame.jpg", "", 1800, 500)
    assert not cases.obvious_collage("https://cdn.example/frame.jpg", "a flood", 1200, 630)


def test_suggestion_prefers_a_single_og_image_over_a_collage():
    collage = {"kind": "og", "collage": True, "name": "og"}
    body = {"kind": "body", "collage": False, "name": "body"}
    plain = {"kind": "og", "collage": False, "name": "plain"}
    assert cases.choose_suggestion([plain, body]) is plain
    assert cases.choose_suggestion([collage, body]) is body
    assert cases.choose_suggestion([collage]) is collage


def test_body_keeps_the_large_frame_and_drops_the_logo():
    html = """
    <html><head>
      <meta property="og:image" content="https://cdn.example/og.jpg"/>
      <meta property="og:image:width" content="1403"/>
      <meta property="og:image" content="https://cdn.example/logo.png"/>
    </head><body>
      <img src="https://cdn.example/site-logo.png" width="400" height="400"/>
      <div class="entry-content">
        <p>The photograph shows the children after the flood in Nepal. The rest of this sentence is here so the body is long enough to count as the article rather than a short label on the template, and a reader would still recognise it as the story.</p>
        <img src="https://cdn.example/thumb.jpg" width="110" height="96"/>
        <img src="data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA=" width="600" height="400"/>
        <img src="https://cdn.example/photo.jpg" width="800" height="600" alt="children"/>
        <img src="/images/placeholder.svg" data-src="https://cdn.example/lazy.webp" width="900" height="600"/>
      </div>
    </body></html>
    """
    listed = cases.candidate_list(html.encode(), "https://news.example/story")
    urls = [item["url"] for item in listed]
    assert urls[0] == "https://cdn.example/og.jpg"
    assert "https://cdn.example/logo.png" not in urls
    assert "https://cdn.example/site-logo.png" not in urls
    assert "https://cdn.example/thumb.jpg" not in urls
    assert "https://cdn.example/photo.jpg" in urls
    assert "https://cdn.example/lazy.webp" in urls


def test_layout_column_named_sidebar_does_not_hide_the_story():
    html = """
    <html><body>
      <div class="right-sidebar">
        <div class="entry-content">
          <p>The photograph shows the children after the flood in Nepal. The rest of this sentence is here so the body is long enough to count as the article rather than a short label on the template, and a reader would still recognise it as the story.</p>
          <img src="https://cdn.example/full.jpg" srcset="https://cdn.example/mid.jpg 450w" width="846" height="1024" alt="children"/>
          <aside class="related"><img src="https://cdn.example/other.jpg" width="640" height="480"/></aside>
        </div>
      </div>
    </body></html>
    """
    urls = [item["url"] for item in cases.candidate_list(html.encode(), "https://news.example/story")]
    assert "https://cdn.example/full.jpg" in urls
    assert "https://cdn.example/mid.jpg" not in urls
    assert "https://cdn.example/other.jpg" not in urls


def test_one_frame_per_pib_caption():
    entries = [
        {"caption": "Same line.", "url": "a"},
        {"caption": "Same line.", "url": "b"},
        {"caption": "Another release on November 24, 2026.", "url": "c"},
    ]
    kept = cases.distinct_entries(entries)
    assert [item["url"] for item in kept] == ["a", "c"]


def test_pib_thumbnail_url_points_at_the_full_frame():
    thumb = "https://static.pib.gov.in/WriteReadData/Gallery/PhotoGallery/2026/Nov/T20261125221545.JPG"
    assert cases.pib_full_url(thumb).endswith("/H20261125221545.JPG")
    assert cases.date_in_text(
        "Devotees at Bangla Sahib Gurudwara, in New Delhi on November 24, 2026."
    ) == "2026-11-24"
    assert cases.map_archive_label("False") == "False"
    assert cases.map_archive_label("Misleading") == "Misleading"


def test_contact_sheet_labels_the_tile(tmp_path):
    from PIL import Image

    photo = tmp_path / "c01_1.jpg"
    Image.new("RGB", (40, 30), "blue").save(photo, format="JPEG")
    dest = tmp_path / "contact_sheet.jpg"
    cases.write_sheet([{"path": photo, "width": 40, "height": 30}], dest)
    sheet = Image.open(dest)
    assert sheet.width > 300
    # The label bar sits under the 300-pixel tile and is dark, not the blue photo.
    pixel = sheet.getpixel((20, 320))
    assert pixel[0] < 40 and pixel[1] < 40 and pixel[2] < 40
