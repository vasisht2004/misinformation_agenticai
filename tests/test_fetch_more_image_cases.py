"""The second-sheet filter. No page is opened and no picture is downloaded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "fetch_more_image_cases", ROOT / "scripts" / "fetch_more_image_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


more = _load()


def _row(**kwargs):
    base = {
        "url": "https://factly.in/story",
        "domain": "Politics & elections",
        "primary_claim": "This photo shows the minister at the rally.",
        "verdict": "False",
    }
    base.update(kwargs)
    return base


def test_health_stock_and_firstcheck_are_left_out():
    rows = [
        _row(url="https://firstcheck.in/a", domain="Health", primary_claim="Salt cures a headache."),
        _row(domain="Health", primary_claim="This photo shows a flood."),
        _row(primary_claim="Eating two kiwis before bed helps people sleep."),
        _row(primary_claim="This photo shows a man in the Assam floods."),
        _row(domain="Finance & scams", primary_claim="WhatsApp has 10 billion downloads."),
        _row(primary_claim="India's passport ranking has fallen."),
    ]
    chosen, notes = more.select_rows(rows, set())
    assert [item[0] for item in chosen] == [4]
    assert any("firstcheck" in line for line in notes)
    assert any("health domain" in line for line in notes)
    assert any("food, diet, sleep, or disease" in line for line in notes)


def test_a_repeated_story_is_kept_twice():
    claim = "Photo shows Education Minister Dharmendra Pradhan with Manisha Mandhare in the NEET case."
    rows = [
        _row(url="https://factly.in/a", primary_claim=claim),
        _row(url="https://www.boomlive.in/b", primary_claim=claim),
        _row(url="https://www.thequint.com/c", primary_claim=claim),
        _row(url="https://newschecker.in/d", primary_claim="Photograph shows an LPG tanker struck by a missile."),
    ]
    chosen, notes = more.select_rows(rows, set())
    assert [item[0] for item in chosen] == [1, 2, 4]
    assert any("same story" in line for line in notes)


def test_an_already_downloaded_url_is_skipped_and_dfrac_is_not():
    url = "https://www.thequint.com/news/old-photo"
    rows = [
        _row(url=url),
        _row(url="https://dfrac.org/en/2026/09/07/sonia"),
    ]
    chosen, notes = more.select_rows(rows, {url})
    assert [item[0] for item in chosen] == [2]
    assert any("already downloaded" in line for line in notes)


def test_stamp_words_match_the_whole_word():
    assert more.stamp_hit("FALSE") == "false"
    assert more.stamp_hit("This is FACT-CHECK copy") == "fact check"
    assert more.stamp_hit("WEBQOOF") == "webqoof"
    assert more.stamp_hit("Alt News") == "alt news"
    assert more.stamp_hit("The photo was falsely shared as Nepal") == ""
    assert more.stamp_hit("a crowd in a street") == ""


def test_pti_shell_has_no_image_tag():
    assert more._shell(b"<html><body>loading</body></html>")
    assert not more._shell(b"<html><body><img src='a.jpg'></body></html>")
