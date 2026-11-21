"""WP:RSP sits behind the curated lists. These check that order.

A missing perennial-source file is a warning, once, and the curated
lists still decide. The warning names the command that builds the file.


hollywoodreporter.com is not on the curated tier-2 list, so a tier of 2
there is the perennial-source file. thehindu.com and pib.gov.in are
curated, and a fake RSP row that calls them deprecated must not move them.
"""

from __future__ import annotations

import csv
import logging

import misinfo_agent.config as config
from misinfo_agent.investigate import credibility_tier, curated_tier


def test_rsp_and_curated_tiers():
    # Not curated. The 2 has to come from the RSP file.
    assert curated_tier("https://www.hollywoodreporter.com/news/story") is None
    assert credibility_tier("https://www.hollywoodreporter.com/news/story") == 2

    deprecated = _uncurated_deprecated()
    assert credibility_tier(f"https://{deprecated}/story") == 4

    assert credibility_tier("https://pib.gov.in/press") == 1
    assert curated_tier("https://www.thehindu.com/news/national") == 2
    assert credibility_tier("https://www.thehindu.com/news/national") == 2
    assert credibility_tier("https://some-unknown-health-blog.example/post") == 4


def test_curated_lists_win_over_rsp(monkeypatch):
    # The file on disk may agree with us. Force the disagreement so the
    # test is about the order, not about what Wikipedia currently says.
    monkeypatch.setattr(
        "misinfo_agent.investigate._rsp_table",
        lambda: {
            "thehindu.com": 4,
            "pib.gov.in": 4,
            "hollywoodreporter.com": 2,
        },
    )
    assert credibility_tier("https://pib.gov.in/press") == 1
    assert credibility_tier("https://www.thehindu.com/news") == 2
    assert credibility_tier("https://hollywoodreporter.com/news") == 2
    # Same URL either way. There is no snippet argument to flip the tier.
    assert credibility_tier("https://hollywoodreporter.com/news") == 2


def test_past_factcheck_heading():
    from misinfo_agent.output import past_factcheck_heading, render

    assert past_factcheck_heading("similar") == "Related past fact-check"
    assert past_factcheck_heading("exact") == "Seen before"
    assert past_factcheck_heading("") == "Seen before"
    card = render({
        "match_type": "similar",
        "best_match": {
            "site": "factly",
            "date": "2020-04-09",
            "url": "https://factly.in/example",
            "primary_claim": "A related claim.",
            "dense_cos": 0.8,
        },
    })["result"]["seen_before"]
    assert card["match_type"] == "similar"
    assert past_factcheck_heading(card["match_type"]) == "Related past fact-check"


def test_missing_rsp_file_warns_once_and_keeps_curated_tiers(monkeypatch, tmp_path, caplog):
    import misinfo_agent.investigate as investigate

    # The real table may already be cached. This test is about a missing
    # file, so the loader has to run again, and the warning flag too.
    monkeypatch.setattr(investigate, "_RSP_TABLE", None)
    monkeypatch.setattr(investigate, "_RSP_WARNED", False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    caplog.set_level(logging.WARNING, logger="misinfo_agent.investigate")

    assert credibility_tier("https://www.thehindu.com/news/story") == 2
    assert credibility_tier("https://pib.gov.in/press") == 1
    assert credibility_tier("https://some-unknown-health-blog.example/post") == 4
    # The second lookup must not warn again. The empty table is cached.
    assert credibility_tier("https://www.thehindu.com/news/another") == 2

    warnings = [record for record in caplog.records if "source_tiers_rsp.csv" in record.message]
    assert len(warnings) == 1
    assert "python scripts/build_source_tiers.py" in warnings[0].message
    assert "curated lists" in warnings[0].message


def _uncurated_deprecated() -> str:
    path = config.DATA_DIR / "source_tiers_rsp.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("rsp_status") != "deprecated":
                continue
            domain = row["domain"]
            if curated_tier(f"https://{domain}/") is None:
                return domain
    raise AssertionError("RSP file has no deprecated domain outside the curated lists")
