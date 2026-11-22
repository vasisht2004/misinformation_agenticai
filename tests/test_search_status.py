"""search_status is set where a search raises or misses, then copied onto the card.

One test per status. None of them calls Tavily. A fake, a cache-only miss,
and the local caps are enough: the status is recorded before a client exists.
"""

from __future__ import annotations

import json

import misinfo_agent.config as config
import misinfo_agent.search as search
from misinfo_agent.investigate import investigate
from misinfo_agent.output import render
from misinfo_agent.search import (
    SearchError,
    SearchQuotaError,
    claim_search_status,
    failure_status,
    get_search_usage,
    search as run_search,
    set_fake_search,
)


def _fill_month(n: int) -> None:
    path = search.MONTHLY_COUNT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({search._month_key(): n}), encoding="utf-8")


def test_status_ran_counts_a_live_call():
    set_fake_search(lambda query, **kwargs: [{
        "url": "https://example.com/a",
        "title": "t",
        "content": "c",
        "score": 0.2,
    }])
    hits = run_search("a live query", purpose="investigate_neutral")
    assert hits
    row = get_search_usage()[-1]
    assert row["status"] == "ran"
    assert row["cached"] is False
    report = claim_search_status(get_search_usage(), needed=True)
    assert report == {"search_status": "ran", "searches_live": 1, "searches_cached": 0}
    card = render({"route": "investigate", "verdict": "False", "run_marks": {"searches": 0}})
    assert card["search_status"] == "ran"
    assert card["result"]["search_status"] == "ran"
    assert card["result"]["searches_live"] == 1


def test_status_ran_counts_a_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SEARCH_CACHE", True)
    monkeypatch.setattr(config, "SEARCH_CACHE_DIR", tmp_path)
    set_fake_search(lambda query, **kwargs: [{
        "url": "https://example.com/cached",
        "title": "t",
        "content": "c",
        "score": 0.4,
    }])
    run_search("the same words", purpose="investigate_neutral")
    set_fake_search(None)
    second = run_search("the same words", purpose="investigate_neutral")
    assert second
    assert get_search_usage()[-1]["cached"] is True
    assert get_search_usage()[-1]["status"] == "ran"
    report = claim_search_status(get_search_usage(), needed=True)
    assert report["search_status"] == "ran"
    assert report["searches_live"] == 1
    assert report["searches_cached"] == 1


def test_status_cache_only_miss(monkeypatch):
    monkeypatch.setenv("SEARCH_CACHE_ONLY", "1")
    hits = run_search("not in the cache", purpose="cache_only_test")
    assert hits == []
    row = get_search_usage()[-1]
    assert row["status"] == "cache_only_miss"
    assert row["cache_miss"] is True
    assert row["cached"] is False
    report = claim_search_status(get_search_usage(), needed=True)
    assert report["search_status"] == "cache_only_miss"
    assert report["searches_live"] == 0
    assert report["searches_cached"] == 0
    card = render({"route": "investigate", "run_marks": {"searches": 0}})
    assert card["result"]["search_status"] == "cache_only_miss"


def test_status_monthly_cap():
    # The provider's own quota has no local cap label. It is still the bill.
    assert failure_status(SearchQuotaError("investigate_neutral", "usage limit")) == "monthly_cap"
    _fill_month(config.TAVILY_MONTHLY_MAX)
    try:
        run_search("over the month", purpose="investigate_neutral")
    except SearchQuotaError as exc:
        assert exc.cap == "monthly"
    else:
        raise AssertionError("monthly cap did not raise")
    assert get_search_usage()[-1]["status"] == "monthly_cap"
    assert claim_search_status(get_search_usage())["search_status"] == "monthly_cap"


def test_status_per_claim_cap(monkeypatch):
    monkeypatch.setattr(search, "_CLAIM_SEARCHES", config.MAX_SEARCHES_PER_CLAIM)
    try:
        run_search("over the claim", purpose="investigate_refute")
    except SearchQuotaError as exc:
        assert exc.cap == "per_claim"
    else:
        raise AssertionError("per-claim cap did not raise")
    assert get_search_usage()[-1]["status"] == "per_claim_cap"
    card = render({"route": "investigate", "run_marks": {"searches": 0}})
    assert card["search_status"] == "per_claim_cap"
    assert card["result"]["searches_live"] == 0


def test_status_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("socket down")

    monkeypatch.setattr(search, "_real", boom)
    try:
        run_search("a broken query", purpose="investigate_neutral")
    except SearchError as exc:
        assert not isinstance(exc, SearchQuotaError)
    else:
        raise AssertionError("search error did not raise")
    assert get_search_usage()[-1]["status"] == "error"
    assert claim_search_status(get_search_usage())["search_status"] == "error"


def test_status_not_needed_for_reuse_and_other():
    reuse = render({
        "route": "reuse_prior",
        "verdict": "False",
        "checkable": True,
        "match_type": "exact",
    })
    assert reuse["search_status"] == "not_needed"
    assert reuse["result"]["search_status"] == "not_needed"
    assert reuse["searches_live"] == 0
    other = render({"verdict": "Other", "checkable": False})
    assert other["search_status"] == "not_needed"
    assert other["result"]["searches_cached"] == 0


def test_investigate_keeps_a_monthly_cap_on_the_round():
    def capped(query, **kwargs):
        raise SearchQuotaError("investigate_neutral", "monthly search cap reached", cap="monthly")

    set_fake_search(capped)
    result = investigate({
        "claim": "The ministry banned a medicine on 1 December 2024",
        "domain": "Health",
        "match_type": "none",
        "leads": [],
        "entities": {"names": [], "places": [], "amounts": [], "dates": [], "years": []},
    })
    assert result["search_status"] == "monthly_cap"
    assert result["investigate_stats"]["search_status"] == "monthly_cap"
    assert result["searches_live"] == 0
