"""Box 7 day 1: the query plan and the search gateway, with no network.

set_fake_search stands in for Tavily. A real client is used only where the
test has to prove a cap or a retry, and even then the client is a stub.
The live test at the bottom is the one run that spends credits.
"""

import json
import time

import pytest

import misinfo_agent.config as config
import misinfo_agent.search as search
from misinfo_agent.fetch import Page
from misinfo_agent.investigate import (
    Passage,
    build_queries,
    canonical_url,
    finish_passages,
    gather_candidates,
    gather_factcheck_hits,
    investigate,
    passage_windows,
    select_passages,
)
from misinfo_agent.search import (
    SearchError,
    SearchQuotaError,
    factcheck_blocked,
    get_search_usage,
    registrable_domain,
    reset_search_usage,
    search as run_search,
    searches_this_month,
    set_fake_search,
)


def _state(domain, match_type="none", leads=None, claim=None, entities=None):
    return {
        "claim": claim or "The ministry banned a medicine on 1 December 2024",
        "domain": domain,
        "match_type": match_type,
        "leads": [] if leads is None else leads,
        "entities": {
            "names": [],
            "places": [],
            "amounts": [],
            "dates": ["1 December 2024"],
            "years": ["2024"],
        }
        if entities is None
        else entities,
    }


def _hit(url, title, score, content="c", published_date=None):
    return {
        "url": url,
        "title": title,
        "content": content,
        "score": score,
        "published_date": published_date,
    }


def _fill_month(n):
    path = search.MONTHLY_COUNT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({search._month_key(): n}), encoding="utf-8")


def _stub_tavily(monkeypatch, handler):
    # The offline fixture refuses the real client. A stub that never
    # touches the network is what a cap test and a retry test both need.
    monkeypatch.setenv("TAVILY_API_KEY", "k-test")
    monkeypatch.setattr(search, "_client", lambda key: object())
    monkeypatch.setattr(search, "_tavily_search", handler)


def test_factcheck_domains_are_fifteen_bare_hosts():
    hosts = list(config.FACTCHECK_DOMAINS)
    assert len(hosts) == 15
    assert len(set(hosts)) == 15
    for host in hosts:
        assert host == host.lower()
        assert not host.startswith("www.")
        assert "/" not in host


def test_query_kinds_for_every_domain():
    # Every domain has an official list, so the plan is four lines, and a
    # similar prior adds the fifth. An unmapped domain is tested separately.
    assert set(config.OFFICIAL_DOMAINS) == set(config.DOMAINS)
    lead = [{"primary_claim": "An older wording of this claim"}]
    for domain in config.DOMAINS:
        plain = build_queries(_state(domain))
        assert [query.kind for query in plain] == [
            "neutral",
            "verify",
            "refute",
            "official",
        ]
        official = plain[-1]
        domain_hosts = list(config.OFFICIAL_DOMAINS[domain])
        # The domain's own offices stay first. The national list is added
        # once, so a host already on the domain list is not repeated.
        assert official.include_domains[: len(domain_hosts)] == domain_hosts
        for host in config.NATIONAL_OFFICIAL_DOMAINS:
            assert host in official.include_domains
        assert len(official.include_domains) == len(set(official.include_domains))
        # Keywords, not the sentence, and not the verify line's question.
        assert official.text != plain[0].text
        assert "true or false" not in official.text.casefold()
        assert official.include_domains
        similar = build_queries(_state(domain, match_type="similar", leads=lead))
        assert [query.kind for query in similar] == [
            "neutral",
            "verify",
            "refute",
            "official",
            "lead",
        ]
        assert similar[-1].text == "An older wording of this claim"
        assert len(similar) <= 5


def test_one_atomic_claim_keeps_the_same_plan():
    plain = build_queries(_state("Health"))
    state = _state("Health")
    state["atomic_claims"] = [{"id": "A1", "text": state["claim"], "core": True}]
    assert [query.model_dump() for query in build_queries(state)] == [
        query.model_dump() for query in plain
    ]


def test_a_second_atomic_claim_adds_one_neutral_query():
    # The fixed lines search the core event. The campus visit is not in
    # that sentence, so it needs its own neutral line, and the older lead
    # still fits under the per-claim cap.
    state = _state(
        "Health",
        claim="CLAIM the ministry banned the medicine and then closed the clinics",
    )
    state["atomic_claims"] = [
        {"id": "A1", "text": "CLAIM the ministry banned the medicine", "core": True},
        {"id": "A2", "text": "The ministry closed the clinics", "core": False},
    ]
    state["match_type"] = "similar"
    state["leads"] = [{"primary_claim": "An older wording of this claim"}]
    queries = build_queries(state)
    texts = [query.text for query in queries]
    assert queries[0].text == "CLAIM the ministry banned the medicine"
    assert queries[0].kind == "neutral"
    assert "The ministry closed the clinics" in texts
    assert "An older wording of this claim" in texts
    assert len(queries) <= config.MAX_SEARCHES_PER_CLAIM
    assert len([query for query in queries if query.kind == "neutral"]) == 2


def test_a_motive_atomic_claim_adds_no_query_and_the_plan_stops_at_six():
    state = _state(
        "Health",
        claim="CLAIM garlic juice cures diabetes and the government is hiding it",
    )
    state["atomic_claims"] = [
        {"id": "A1", "text": "CLAIM garlic juice cures diabetes", "core": True, "checkable": True},
        {"id": "A2", "text": "AIIMS Delhi doctors confirmed the remedy", "core": False, "checkable": True},
        {"id": "A3", "text": "The government is hiding it", "core": False, "checkable": False},
    ]
    state["match_type"] = "similar"
    state["leads"] = [{"primary_claim": "An older wording of this claim"}]
    queries = build_queries(state)
    texts = [query.text for query in queries]
    assert "AIIMS Delhi doctors confirmed the remedy" in texts
    assert "The government is hiding it" not in texts
    # Four core lines, the one checkable extra, and the lead. Six is the
    # plan cap, so two searches remain for the top-up.
    assert len(queries) <= 6
    assert queries[-1].kind == "lead"


def test_two_checkable_extras_leave_the_lead_out_to_keep_two_searches():
    state = _state("Health", claim="CLAIM the ministry banned the medicine and then closed the clinics")
    state["atomic_claims"] = [
        {"id": "A1", "text": "CLAIM the ministry banned the medicine", "core": True, "checkable": True},
        {"id": "A2", "text": "The ministry closed the clinics", "core": False, "checkable": True},
        {"id": "A3", "text": "The ministry reopened them on Tuesday", "core": False, "checkable": True},
    ]
    state["match_type"] = "similar"
    state["leads"] = [{"primary_claim": "An older wording of this claim"}]
    queries = build_queries(state)
    texts = [query.text for query in queries]
    assert "The ministry closed the clinics" in texts
    assert "The ministry reopened them on Tuesday" in texts
    assert len(queries) == 6
    assert all(query.kind != "lead" for query in queries)


def test_verify_query_keeps_five_entities_from_the_core_claim():
    from misinfo_agent.investigate import _entities_in_claim

    names = ["Modi", "Rahul", "Kejriwal", "Stalin", "Mamata", "Yogi"]
    claim = "Modi Rahul Kejriwal Stalin Mamata Yogi met"
    entities = {"people": names, "places": ["IIM Bangalore"], "orgs": ["Namma Metro"]}
    kept = _entities_in_claim(entities, claim, limit=5)
    flat = [item for values in kept.values() for item in values]
    # Five names from the sentence. The sixth is past the cap, and a name
    # that is not in the sentence is not a candidate.
    assert flat == names[:5]
    state = _state("Politics & elections", claim=claim, entities=entities)
    verify = {query.kind: query for query in build_queries(state)}["verify"].text
    assert "IIM Bangalore" not in verify
    assert "Namma Metro" not in verify
    for name in names[:5]:
        assert name in verify


def test_health_official_query_includes_cure_sources():
    hosts = config.OFFICIAL_DOMAINS["Health"]
    for host in ("who.int", "nih.gov", "ncbi.nlm.nih.gov", "mohfw.gov.in", "icmr.gov.in", "diabetes.org", "idf.org"):
        assert host in hosts


def test_society_official_query_includes_national_sources():
    # The national list is part of every domain's official query, including
    # the catch-all. These hosts are how a "how many states" claim can reach
    # the census and Know India. A future search uses this list. The blocked
    # eval cache was stored before it, so that cache does not grow these hosts.
    national = (
        "india.gov.in",
        "knowindia.india.gov.in",
        "mha.gov.in",
        "censusindia.gov.in",
        "pib.gov.in",
        "mospi.gov.in",
    )
    official = {query.kind: query for query in build_queries(_state("Society & other"))}["official"]
    for host in national:
        assert host in official.include_domains
    assert official.include_domains.count("india.gov.in") == 1
    assert official.include_domains.count("pib.gov.in") == 1


def test_submitted_url_is_dropped_from_search_and_evidence():
    from misinfo_agent.investigate import build_evidence

    state = _state("Health", claim="CLAIM the ministry banned the medicine")
    state["input_type"] = "url"
    state["raw_text"] = "https://www.Example.com/story?utm_source=whatsapp"

    def fake(query, **kwargs):
        return [
            _hit("https://example.com/story", "the page itself", 0.99),
            _hit("https://www.bbc.com/news/other", "a newsroom", 0.4),
        ]

    set_fake_search(fake)
    try:
        hits = gather_candidates(state)
    finally:
        set_fake_search(None)
    urls = [hit.url for hit in hits]
    assert urls == ["https://bbc.com/news/other"]
    quote = "The ministry banned the medicine yesterday"
    rows = build_evidence(
        state,
        [
            {
                "passage": _stance_passage("P1", "https://example.com/story", quote, 0.9),
                "stance": "supports",
                "quote": quote,
                "reason": "It repeats the page",
                "quote_unverified": False,
                "atomic_id": "A1",
            },
            {
                "passage": _stance_passage("P2", "https://www.bbc.com/news/other", quote, 0.8),
                "stance": "supports",
                "quote": quote,
                "reason": "A newsroom confirms it",
                "quote_unverified": False,
                "atomic_id": "A1",
            },
        ],
    )
    assert [row["url"] for row in rows] == ["https://www.bbc.com/news/other"]


def test_lead_query_only_when_similar():
    state = _state("Health")
    assert "lead" not in [query.kind for query in build_queries(state)]

    # An exact prior can still carry leads, for the card. It is the same
    # sentence the neutral line already searches, so it is not a fifth query.
    state["match_type"] = "exact"
    state["leads"] = [{"primary_claim": "older wording"}]
    assert "lead" not in [query.kind for query in build_queries(state)]

    state["match_type"] = "none"
    assert "lead" not in [query.kind for query in build_queries(state)]

    state["match_type"] = "similar"
    state["leads"] = []
    assert "lead" not in [query.kind for query in build_queries(state)]

    state["leads"] = [{"dense_cos": 0.8}]
    assert "lead" not in [query.kind for query in build_queries(state)]

    # The first lead with no text is skipped. The next one is the history.
    state["leads"] = [
        {"primary_claim": "  "},
        {"primary_claim": "The older wording about the same medicine"},
    ]
    queries = build_queries(state)
    assert queries[-1].kind == "lead"
    assert queries[-1].text == "The older wording about the same medicine"
    assert queries[-1].include_domains is None


def test_neutral_query_trims_to_twenty_words():
    words = [f"w{i}" for i in range(25)]
    neutral = build_queries(_state("Health", claim=" ".join(words)))[0]
    assert neutral.kind == "neutral"
    assert neutral.text.split() == words[:20]


def test_verify_and_refute_use_the_planned_wording():
    state = _state(
        "Finance & scams",
        claim="₹500 notes printed before 2016 are invalid from 1 December",
        entities={
            "amounts": ["₹500"],
            "years": ["2016"],
            "dates": ["1 December"],
            "names": [],
            "places": [],
        },
    )
    by_kind = {query.kind: query for query in build_queries(state)}
    assert (
        by_kind["neutral"].text
        == "₹500 notes printed before 2016 are invalid from 1 December"
    )
    assert by_kind["verify"].text.endswith("true or false")
    assert "₹500" in by_kind["verify"].text
    assert "2016" in by_kind["verify"].text
    assert "1 December" in by_kind["verify"].text
    assert by_kind["refute"].text.endswith("fake OR false OR hoax OR misleading")
    finance_hosts = list(config.OFFICIAL_DOMAINS["Finance & scams"])
    assert by_kind["official"].include_domains[: len(finance_hosts)] == finance_hosts
    for host in config.NATIONAL_OFFICIAL_DOMAINS:
        assert host in by_kind["official"].include_domains
    # Entities, then the longest content words. Not the sentence, and not
    # "true or false", which is only the verify line.
    assert (
        by_kind["official"].text
        == "₹500 1 December 2016 notes printed before invalid"
    )
    health = _state(
        "Health",
        claim="Drinking hot water with lemon kills coronavirus",
        entities={"names": [], "places": [], "amounts": [], "dates": [], "years": []},
    )
    official = {query.kind: query for query in build_queries(health)}["official"]
    assert official.text == "Drinking water lemon coronavirus"
    assert len(official.text.split()) <= 4


def test_official_query_skipped_when_the_domain_has_no_mapping(monkeypatch):
    monkeypatch.setattr(config, "OFFICIAL_DOMAINS", {})
    kinds = [query.kind for query in build_queries(_state("Health"))]
    assert "official" not in kinds
    assert kinds == ["neutral", "verify", "refute"]


def test_canonical_url_strips_host_case_www_tracking_fragment_and_slash():
    raw = "HTTPS://WWW.Example.COM:443/Foo/?utm_source=a&id=2&fbclid=z&gclid=y#section"
    assert canonical_url(raw) == "https://example.com/Foo?id=2"
    assert canonical_url("https://www.example.com/foo/") == "https://example.com/foo"
    assert canonical_url("https://example.com/foo") == "https://example.com/foo"
    assert canonical_url("https://example.com/") == "https://example.com"
    # Param order is not part of the page.
    assert canonical_url("https://example.com/a?b=1&a=2") == canonical_url(
        "https://example.com/a?a=2&b=1"
    )
    # A non-default port is part of the page. The path's case is too.
    assert canonical_url("https://Example.com:8443/Foo/") == "https://example.com:8443/Foo"


def test_dedup_keeps_the_best_score_and_records_kinds():
    def fake(query, **kwargs):
        if query.endswith("true or false"):
            return [
                _hit(
                    "https://www.Example.com/a/?utm_source=x&fbclid=1",
                    "low",
                    0.2,
                )
            ]
        if "fake OR false" in query:
            return [
                _hit(
                    "https://Example.com/a/#frag",
                    "high",
                    0.9,
                    published_date="2024-05-01",
                ),
                _hit("https://rbi.org.in/note", "only refute", 0.4),
            ]
        return []

    set_fake_search(fake)
    hits = gather_candidates(_state("Finance & scams"))
    by_url = {hit.url: hit for hit in hits}
    assert set(by_url) == {"https://example.com/a", "https://rbi.org.in/note"}
    best = by_url["https://example.com/a"]
    assert best.score == 0.9
    assert best.title == "high"
    assert best.published_date == "2024-05-01"
    # Plan order, not score order: verify is asked before refute.
    assert best.query_kinds == ["verify", "refute"]
    assert by_url["https://rbi.org.in/note"].query_kinds == ["refute"]
    assert searches_this_month() == 0


def test_gather_caps_at_25_unique_hits():
    def fake(query, **kwargs):
        return [_hit(f"https://example.com/p{i}", str(i), float(i)) for i in range(30)]

    set_fake_search(fake)
    hits = gather_candidates(_state("Health"))
    assert len(hits) == 25
    assert [hit.score for hit in hits] == [float(i) for i in range(29, 4, -1)]


def test_factcheck_hits_dropped_and_excluded_when_blocking(monkeypatch):
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    assert "boomlive.in" in config.FACTCHECK_DOMAINS
    assert "english.factcrescendo.com" in config.FACTCHECK_DOMAINS
    seen = {}

    def fake(query, **kwargs):
        seen["exclude"] = kwargs["exclude_domains"]
        return [
            _hit("https://hindi.boomlive.in/story", "a report", 0.99),
            _hit("https://english.factcrescendo.com/story", "a report", 0.7),
            # The registrable domain, not only the subdomain the archive stored.
            _hit("https://factcrescendo.com/other", "a report", 0.5),
            # Same suffix would be wrong: the name only looks similar.
            _hit("https://notboomlive.in/a", "a report", 0.45),
            _hit("https://www.rbi.org.in/note", "primary", 0.4),
        ]

    set_fake_search(fake)
    hits = gather_candidates(_state("Finance & scams"))
    assert {hit.url for hit in hits} == {
        "https://notboomlive.in/a",
        "https://rbi.org.in/note",
    }
    exclude = set(seen["exclude"])
    assert "boomlive.in" in exclude
    assert "factcrescendo.com" in exclude
    assert "youturn.in" in exclude
    # The desk, not the whole news wire.
    assert "factcheck.afp.com" in exclude
    assert "afp.com" not in exclude
    assert searches_this_month() == 0


def test_factcheck_phrase_window_stays_when_blocking_is_off(monkeypatch):
    # The phrase rule is the blocked-mode answer key. A news page that
    # quotes a fact-check is dropped only while that block is on. With
    # the block off the same window is a passage.
    import misinfo_agent.investigate as investigate_module

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)
    page = _page(
        "https://www.thehindu.com/story",
        [
            _sentence("0.90"),
            "This fact check says the notice was never issued by the ministry yesterday.",
            _sentence("0.20"),
        ],
    )
    state = _state("Health", claim="CLAIM the ministry banned the medicine outright")
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    assert investigate_module._quotes_factcheck("This fact check says the notice was never issued.") is False
    kept, _cosines, _dropped = select_passages(state, [page])
    assert kept
    assert any("fact check" in passage.text.casefold() for passage in kept)
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    assert investigate_module._quotes_factcheck("This fact check says the notice was never issued.") is True
    dropped, _cosines, _dropped = select_passages(state, [page])
    assert dropped == []


def test_factcheck_budget_excludes_the_exact_url_and_caps_fetches(monkeypatch):
    # The source article is one URL. Another story on that host is still
    # a fact-check the allowed run may read. News hits are not part of
    # this budget, and the eighth fetch is the last one opened.
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)
    rows = [
        _hit("https://www.boomlive.in/own?utm_source=a", "own", 0.99),
        _hit("https://www.boomlive.in/other-story", "other", 0.98),
        _hit("https://www.thehindu.com/news/story", "news", 0.97),
    ]
    rows.extend(
        _hit(f"https://altnews.in/story-{index}", f"desk {index}", 0.50 - index * 0.01)
        for index in range(9)
    )
    set_fake_search(lambda query, **kwargs: rows)
    state = _state("Health")
    state["run_marks"] = {
        "eval_sites": "allowed",
        "eval_exclude_urls": ["http://boomlive.in/own"],
    }
    found = gather_factcheck_hits(state)
    urls = [hit.url for hit in found]
    assert len(urls) == 8
    assert "https://boomlive.in/own" not in urls
    assert "https://boomlive.in/other-story" in urls
    assert "https://altnews.in/story-0" in urls
    assert "https://altnews.in/story-8" not in urls
    assert all("thehindu.com" not in url for url in urls)


def test_allowed_evidence_keeps_every_blocked_row(monkeypatch):
    # Allowed evidence is the blocked rows plus a fact-check row. The
    # open-web response does not include the news URLs, so they survive
    # only because the news search still runs with the block on.
    import re

    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", True)
    # One sentence, so the fake encoder sees 0.90 as the first token of
    # the only window. A period would start a second window on "The".
    filler = (
        "and the branches were told to keep paying the same notes to customers "
        "who needed cash for the market the school fee and the train home while "
        "the clerk repeated that no order had arrived by post or by email"
    )

    def article(lead):
        return {
            "content": "snippet",
            "raw_content": f"0.90 {lead} {filler} {filler}",
        }

    rbi = {
        "url": "https://www.rbi.org.in/notice",
        "title": "RBI",
        "score": 0.9,
        "published_date": "2024-11-01",
        **article("The Reserve Bank said no circular has been issued declaring the old notes invalid"),
    }
    hindu = {
        "url": "https://www.thehindu.com/news/story",
        "title": "The Hindu",
        "score": 0.8,
        "published_date": "2024-11-02",
        **article("The ministry confirmed that no such order was sent to the banks or the branches"),
    }
    own = {
        "url": "https://www.boomlive.in/own?utm_source=a",
        "title": "own fact check",
        "score": 0.99,
        "published_date": "2024-11-03",
        **article("This fact check found that the notice was never issued by the ministry at all"),
    }
    other = {
        "url": "https://www.boomlive.in/other-story",
        "title": "other fact check",
        "score": 0.7,
        "published_date": "2024-11-04",
        **article("This fact check found that the notice was never issued by the ministry at all"),
    }
    later = {
        "url": "https://www.altnews.in/later",
        "title": "later fact check",
        "score": 0.6,
        "published_date": "2025-02-01",
        **article("A later fact check found that the notice was never issued by the ministry afterwards"),
    }

    def fake_search(query, **kwargs):
        exclude = kwargs.get("exclude_domains") or []
        if "boomlive.in" in exclude:
            return [rbi, hindu]
        return [own, other, later]

    def fake_llm(prompt, schema, images):
        if schema.__name__ == "TopupBatch":
            return {"queries": []}
        if schema.__name__ == "RepairBatch":
            return {"judgements": []}
        judgements = []
        for block in re.split(r"(?m)(?=^P\d+\b)", prompt):
            match = re.match(r"P\d+", block)
            if match is None:
                continue
            if "fact check found" in block.casefold():
                quote = "This fact check found that the notice was never issued"
            elif "ministry confirmed" in block.casefold():
                quote = "The ministry confirmed that no such order was sent"
            else:
                quote = "The Reserve Bank said no circular has been issued"
            judgements.append({
                "passage_id": match.group(0),
                "stance": "supports",
                "quote": quote,
                "reason": "The page states it.",
            })
        return {"judgements": judgements}

    set_fake_search(fake_search)
    set_fake(fake_llm)
    state = _state("Health", claim="CLAIM the ministry banned the medicine outright")
    state["claim_date"] = "2024-12-15"
    state["run_marks"] = {
        "eval_sites": "allowed",
        "eval_exclude_urls": ["http://boomlive.in/own"],
    }
    try:
        result = investigate(state)
    finally:
        set_fake(None)
    urls = {row["url"] for row in result["evidence"]}
    assert "https://rbi.org.in/notice" in urls
    assert "https://thehindu.com/news/story" in urls
    assert "https://boomlive.in/other-story" in urls
    assert "https://boomlive.in/own" not in urls
    assert "https://altnews.in/later" not in urls
    assert any("fact check" in passage["text"].casefold() for passage in result["passages"])
    assert result["investigate_stats"]["factcheck_after_claim"] >= 1
    assert config.BLOCK_FACTCHECK_SITES is False


def test_factcheck_hits_kept_when_blocking_is_off():
    set_fake_search(
        lambda query, **kwargs: [
            _hit("https://www.boomlive.in/story", "fact-check", 0.99),
        ]
        if kwargs["exclude_domains"] is None
        else []
    )
    hits = gather_candidates(_state("Health", claim="A short health claim"))
    assert any(hit.url == "https://boomlive.in/story" for hit in hits)


def test_block_flag_is_part_of_the_cache_key(monkeypatch, tmp_path):
    # An open-web cache must not answer the blocked run. Filtering the
    # cached rows would still hide the pages Tavily would have returned
    # once the fact-check sites were excluded.
    monkeypatch.setattr(config, "SEARCH_CACHE", True)
    monkeypatch.setattr(config, "SEARCH_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    page = [_hit("https://www.boomlive.in/x", "fc", 0.8)]
    set_fake_search(lambda query, **kwargs: page)
    first = run_search("the same words", purpose="investigate_neutral")
    assert len(first) == 1

    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    calls = {"n": 0}

    def fake(query, **kwargs):
        calls["n"] += 1
        return page + [_hit("https://rbi.org.in/note", "primary", 0.3)]

    set_fake_search(fake)
    second = run_search("the same words", purpose="investigate_neutral")
    assert calls["n"] == 1
    assert [hit.url for hit in second] == ["https://rbi.org.in/note"]


def test_monthly_cap_raises_with_no_call(monkeypatch):
    _fill_month(config.TAVILY_MONTHLY_MAX)
    calls = []
    monkeypatch.setattr(
        search, "_client", lambda key: calls.append("client") or object()
    )
    with pytest.raises(SearchQuotaError):
        run_search("a query", purpose="investigate_neutral")
    assert calls == []
    assert searches_this_month() == config.TAVILY_MONTHLY_MAX
    rows = get_search_usage()
    assert rows[-1]["success"] is False
    assert rows[-1]["cached"] is False
    assert rows[-1]["purpose"] == "investigate_neutral"


def test_per_claim_cap_raises_with_no_call(monkeypatch):
    calls = {"n": 0}

    def tavily(*args, **kwargs):
        calls["n"] += 1
        return {"results": [_hit("https://example.com/a", "t", 0.2)]}

    _stub_tavily(monkeypatch, tavily)
    # One under the cap, so the next real call is the last one allowed.
    monkeypatch.setattr(search, "_CLAIM_SEARCHES", config.MAX_SEARCHES_PER_CLAIM - 1)
    run_search("q-one", purpose="investigate_neutral")
    assert calls["n"] == 1
    assert searches_this_month() == 1
    with pytest.raises(SearchQuotaError):
        run_search("q-two", purpose="investigate_refute")
    assert calls["n"] == 1
    assert searches_this_month() == 1


def test_cached_and_fake_calls_do_not_count(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SEARCH_CACHE", True)
    monkeypatch.setattr(config, "SEARCH_CACHE_DIR", tmp_path / "cache")
    calls = {"n": 0}

    def tavily(*args, **kwargs):
        calls["n"] += 1
        return {"results": [_hit("https://example.com/a", "t", 0.2)]}

    _stub_tavily(monkeypatch, tavily)
    run_search("q", purpose="investigate_neutral")
    assert searches_this_month() == 1
    assert calls["n"] == 1

    # A different purpose is the same search. The label is for the audit.
    run_search("q", purpose="investigate_verify")
    assert calls["n"] == 1
    assert searches_this_month() == 1

    # And a full month does not stop a replay. The credit was already spent.
    _fill_month(config.TAVILY_MONTHLY_MAX)
    again = run_search("q", purpose="investigate_official")
    assert [hit.url for hit in again] == ["https://example.com/a"]
    assert calls["n"] == 1
    assert searches_this_month() == config.TAVILY_MONTHLY_MAX

    rows = get_search_usage()
    assert rows[0]["cached"] is False
    assert rows[1]["cached"] is True
    assert rows[2]["cached"] is True
    assert set(rows[0]) == {
        "purpose",
        "query",
        "n_results",
        "latency_s",
        "cached",
        "success",
        "cache_miss",
        # Set where the call returns or raises. The card rolls these up.
        "status",
    }
    reset_search_usage()
    assert get_search_usage() == []
    # The log is not the budget.
    assert searches_this_month() == config.TAVILY_MONTHLY_MAX

    # A different host list is a different request, even with the same words.
    def restricted(*args, **kwargs):
        calls["n"] += 1
        return {"results": [_hit("https://rbi.org.in/note", "rbi", 0.4)]}

    monkeypatch.setattr(search, "_tavily_search", restricted)
    # The month file is at the cap, so this would raise if it were not... it
    # IS at the cap, and a new request must raise rather than call. Reset
    # the month so the miss is allowed, and prove the key changed.
    _fill_month(0)
    run_search("q", purpose="investigate_official", include_domains=["rbi.org.in"])
    assert calls["n"] == 2
    assert searches_this_month() == 1


def test_transient_error_is_retried_once(monkeypatch):
    monkeypatch.setattr(search, "_BACKOFF_S", 0)
    calls = {"n": 0}

    def tavily(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        return {"results": [_hit("https://example.com/a", "t", 0.4)]}

    _stub_tavily(monkeypatch, tavily)
    hits = run_search("q", purpose="investigate_neutral")
    assert [hit.url for hit in hits] == ["https://example.com/a"]
    # Two HTTP attempts, one usage row: the retry is a billed call, and
    # the audit still sees one search.
    assert calls["n"] == 2
    assert searches_this_month() == 2
    assert len(get_search_usage()) == 1
    assert get_search_usage()[0]["success"] is True

    calls["n"] = 0

    def always(*args, **kwargs):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(search, "_tavily_search", always)
    with pytest.raises(SearchError):
        run_search("other", purpose="investigate_refute")
    assert calls["n"] == 2


def test_auth_and_provider_quota_are_not_retried(monkeypatch):
    from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

    monkeypatch.setattr(search, "_BACKOFF_S", 0)
    calls = {"n": 0}
    secret = "super-secret-key"
    monkeypatch.setenv("TAVILY_API_KEY", secret)
    monkeypatch.setattr(search, "_client", lambda key: object())

    cases = (
        (InvalidAPIKeyError(f"bad key {secret}"), False),
        (UsageLimitExceededError("monthly"), True),
    )
    for exc, is_quota in cases:
        before = searches_this_month()
        calls["n"] = 0

        def tavily(*args, **kwargs):
            calls["n"] += 1
            raise exc

        monkeypatch.setattr(search, "_tavily_search", tavily)
        with pytest.raises(SearchQuotaError if is_quota else SearchError) as caught:
            run_search("q", purpose="investigate_neutral")
        if not is_quota:
            assert not isinstance(caught.value, SearchQuotaError)
        assert calls["n"] == 1
        assert searches_this_month() == before + 1
        assert secret not in str(caught.value)
        assert "q" not in str(caught.value)


def test_each_factcheck_rule_drops_and_a_press_release_stays(monkeypatch):
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)

    def kept(url, title="a report"):
        set_fake_search(lambda query, **kwargs: [_hit(url, title, 0.5)])
        return run_search("q", purpose="p")

    # Subdomain of an archive publication, and the three path spellings.
    assert kept("https://hindi.boomlive.in/story") == []
    assert kept("https://example.com/world/fact-check/note") == []
    assert kept("https://example.com/world/factcheck/note") == []
    assert kept("https://example.com/world/fact_check/note") == []
    # Title, including a fact check that does not use those words in the path.
    assert kept("https://example.com/a", "Fact Check: the claim") == []
    assert kept("https://example.com/a", "FACT CHECK this rumour") == []
    assert kept("https://example.com/a", "A Fact-check of the video") == []
    # PIB Fact Check on a social host, and a fact-check path on pib.gov.in.
    assert kept("https://x.com/PIBFactCheck/status/1", "a post") == []
    assert kept("https://x.com/someone/status/2", "PIB Fact Check on X: hello") == []
    assert kept("https://pib.gov.in/fact-check/note", "a note") == []
    # The extra desk is blocked. Its parent news site is not.
    assert kept("https://factcheck.afp.com/story") == []
    assert kept("https://www.afp.com/story", "AFP news")
    assert kept("https://factcheckhub.com/saudi") == []
    assert kept("https://checkyourfact.com/story") == []
    assert kept("https://africacheck.org/fact-checks/story") == []
    assert kept("https://example.com/news/webqoof/story") == []
    # The Quint's newsroom is tier 2. WebQoof, on the same host, is the
    # fact-check desk and stays out of a blocked-mode search.
    assert kept("https://www.thequint.com/news/webqoof/garlic-cures-diabetes") == []
    # A normal press release is the page the official query is for.
    release = kept(
        "https://pib.gov.in/PressRelease.aspx",
        "Cabinet approves the scheme",
    )
    assert [hit.url for hit in release] == ["https://pib.gov.in/PressRelease.aspx"]
    assert factcheck_blocked(
        "https://pib.gov.in/PressRelease.aspx",
        "Cabinet approves the scheme",
    ) is False


def test_reserved_slots_keep_the_top_three_of_each_kind():
    def fake(query, **kwargs):
        if "fake OR false" in query:
            return [
                _hit(f"https://refute.example/r{i}", "r", score)
                for i, score in enumerate((0.04, 0.03, 0.02, 0.01))
            ]
        if "true or false" in query:
            return []
        # Neutral is the sentence. Official is the keyword line.
        if query.startswith("The ministry"):
            return [
                _hit(f"https://news.example/n{i}", "n", 100 - i) for i in range(30)
            ]
        return []

    set_fake_search(fake)
    hits = gather_candidates(_state("Health"))
    urls = {hit.url for hit in hits}
    assert len(hits) == 25
    # The three strongest refute hits survive a cap full of higher scores.
    # The fourth does not: the reserve is three, not the whole kind.
    assert "https://refute.example/r0" in urls
    assert "https://refute.example/r1" in urls
    assert "https://refute.example/r2" in urls
    assert "https://refute.example/r3" not in urls
    assert "https://news.example/n29" not in urls


def test_passage_windows_overlap_and_drop_short_fragments():
    sentence = " ".join(f"w{i}" for i in range(10)) + "."
    text = " ".join([sentence] * 4)
    windows = passage_windows(text)
    # Four sentences. The first window takes three, the next starts one back.
    assert len(windows) == 2
    assert len(windows[0].split()) == 30
    assert len(windows[1].split()) == 20
    assert passage_windows("Too short. Also short.") == []


def _sentence(first, n=9):
    return " ".join([first] + [f"w{i}" for i in range(n)]) + "."


def _page(url, sentences, published=None):
    return Page(
        url=url,
        title="t",
        text=" ".join(sentences),
        published_date=published,
        date_source="page" if published else None,
        fetched=True,
        fetch_status="ok",
    )


class _FakeEncoder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np

        rows = []
        for text in texts:
            token = text.split()[0]
            cos = 1.0 if token == "CLAIM" else float(token)
            rest = (1.0 - cos * cos) ** 0.5
            rows.append([cos, rest])
        return np.asarray(rows, dtype=np.float32)


def test_top_two_per_page_and_the_cosine_floor(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    # Seven sentences make three windows, headed by 0.90, 0.80 and 0.70.
    page_a = _page(
        "https://alpha.com/a",
        [
            _sentence("0.90"),
            _sentence("0.11"),
            _sentence("0.80"),
            _sentence("0.12"),
            _sentence("0.70"),
            _sentence("0.13"),
            _sentence("0.10"),
        ],
    )
    page_b = _page(
        "https://beta.com/b",
        [
            _sentence("0.50"),
            _sentence("0.15"),
            _sentence("0.20"),
            _sentence("0.16"),
        ],
    )
    passages, cosines, dropped = select_passages(
        {"claim": "CLAIM the medicine was banned"},
        [page_a, page_b],
    )
    assert dropped == 0
    assert [round(value, 2) for value in cosines] == [0.90, 0.80, 0.70, 0.50, 0.20]
    # 0.70 is above the floor but it is the third window on the page.
    # 0.20 is under the floor.
    assert [(item.url, round(item.cos, 2)) for item in passages] == [
        ("https://alpha.com/a", 0.90),
        ("https://alpha.com/a", 0.80),
        ("https://beta.com/b", 0.50),
    ]
    assert [item.passage_id for item in passages] == ["P1", "P2", "P3"]


def test_passage_cap_is_forty(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    pages = []
    for index in range(21):
        high = f"{0.90 - index * 0.01:.2f}"
        low = f"{0.89 - index * 0.01:.2f}"
        pages.append(
            _page(
                f"https://site{index}.com/p",
                [_sentence(high), _sentence("0.11"), _sentence(low), _sentence("0.12")],
            )
        )
    passages, _cosines, _dropped = select_passages({"claim": "CLAIM the medicine was banned"}, pages)
    assert len(passages) == 40
    assert passages[0].cos >= passages[-1].cos


def test_after_claim_flag_and_drop(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    pages = [
        _page("https://later.com/a", [_sentence("0.90", n=24)], "2026-01-01"),
        _page("https://earlier.com/a", [_sentence("0.80", n=24)], "2020-01-01"),
        _page("https://undated.com/a", [_sentence("0.70", n=24)]),
    ]
    state = {"claim": "CLAIM the medicine was banned", "claim_date": "2024-06-01"}
    passages, _cosines, dropped_off = select_passages(state, pages)
    assert dropped_off == 0
    by_url = {item.url: item for item in passages}
    assert by_url["https://later.com/a"].after_claim is True
    assert by_url["https://earlier.com/a"].after_claim is False
    assert by_url["https://undated.com/a"].after_claim is None

    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", True)
    dropped, _cosines, dropped_n = select_passages(state, pages)
    assert dropped_n == 1
    assert "https://later.com/a" not in {item.url for item in dropped}
    assert "https://earlier.com/a" in {item.url for item in dropped}
    assert "https://undated.com/a" in {item.url for item in dropped}


def test_encoder_loads_once(monkeypatch):
    import misinfo_agent.memory as memory

    calls = {"n": 0}
    sentinel = object()
    monkeypatch.setattr(memory, "_ENCODER", None)

    def build():
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(memory, "_build_encoder", build)
    import misinfo_agent.prior_match as prior_match

    assert memory.get_encoder() is sentinel
    assert memory.get_encoder() is sentinel
    memo = memory.ClaimMemory.__new__(memory.ClaimMemory)
    memo._encoder = None
    assert memo._encoder_model() is sentinel
    # Box 6 used to build its own SentenceTransformer. The word vectors
    # have to be the same object, or BGE-M3 loads a second time.
    assert prior_match._word_encoder() is sentinel
    assert calls["n"] == 1


def test_official_query_drops_media_filler():
    state = _state(
        "Disasters & accidents",
        claim="Video shows flood in Chennai",
        entities={
            "places": ["Chennai"],
            "names": [],
            "amounts": [],
            "dates": [],
            "years": [],
        },
    )
    official = {query.kind: query for query in build_queries(state)}["official"]
    assert official.text == "Chennai flood"
    said = _state(
        "Politics & elections",
        claim="Minister says the viral video shows the clip is true",
        entities={"names": [], "places": [], "amounts": [], "dates": [], "years": []},
    )
    keywords = {query.kind: query for query in build_queries(said)}["official"].text
    assert keywords == "Minister"
    for word in ("video", "shows", "says", "viral", "clip", "true", "false"):
        assert word not in keywords.casefold().split()


def test_junk_windows_duplicates_and_factcheck_quotes(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    junk = (
        "Log in sign in accept cookies subscribe to the newsletter and "
        "manage preferences of the cookie policy consent login cookies signup now"
    )
    good = _sentence("0.80", n=24)
    same = _sentence("0.90", n=24)
    quote = (
        "0.70 PIB Fact Check says this rumour about the notes is false and "
        "should not be shared with anyone else today please stop"
    )
    pages = [
        _page("https://chrome.com/login", [junk]),
        _page("https://paper.com/a", [good]),
        _page("https://one.com/a", [same]),
        _page("https://two.com/a", [same]),
        _page("https://social.com/post", [quote]),
    ]
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    passages, _cosines, _dropped = select_passages(
        {"claim": "CLAIM the medicine was banned"},
        pages,
    )
    urls = [item.url for item in passages]
    assert "https://chrome.com/login" not in urls
    assert "https://social.com/post" not in urls
    assert urls.count("https://one.com/a") + urls.count("https://two.com/a") == 1
    assert "https://paper.com/a" in urls

    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    opened, _cosines, _dropped = select_passages(
        {"claim": "CLAIM the medicine was banned"},
        [_page("https://social.com/post", [quote])],
    )
    assert [item.url for item in opened] == ["https://social.com/post"]


def test_two_passages_per_registrable_domain(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    pages = [
        _page("https://www.news.com/a", [_sentence("0.90", n=24)]),
        _page("https://news.com/b", [_sentence("0.80", n=24)]),
        _page("https://m.news.com/c", [_sentence("0.70", n=24)]),
    ]
    passages, _cosines, _dropped = select_passages(
        {"claim": "CLAIM the medicine was banned"},
        pages,
    )
    assert [round(item.cos, 2) for item in passages] == [0.90, 0.80]


def test_url_path_date_wins(monkeypatch):
    import time

    import misinfo_agent.fetch as fetch

    calls = []
    monkeypatch.setattr(fetch, "_http_get", lambda url: calls.append(url))
    url = "https://www.example.com/news/2016/11/08/notes"
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
    # The budget is already gone. The path date is free, so it is still used,
    # and the HTML read must not start.
    done = finish_passages(
        {"claim_date": "2024-01-01"},
        [passage],
        deadline=time.monotonic() - 1,
    )
    assert done["passages"][0].published_date == "2016-11-08"
    assert done["passages"][0].date_source == "url"
    assert calls == []


def test_eval_undated_drops_only_while_the_leakage_guard_is_on(monkeypatch):
    import time

    dated = Passage(
        passage_id="P0",
        url="https://example.com/dated",
        netloc="example.com",
        title="t",
        text="word " * 30,
        cos=0.8,
        score=0.8,
        published_date="2020-01-02",
        date_source="htmldate",
        fetched=True,
        after_claim=False,
    )
    undated = Passage(
        passage_id="P0",
        url="https://example.com/story",
        netloc="example.com",
        title="t",
        text="word " * 30,
        cos=0.7,
        score=0.7,
        fetched=True,
    )
    state = {"claim_date": "2024-06-01"}
    deadline = time.monotonic() - 1
    monkeypatch.setattr(config, "EVAL_UNDATED", "drop")
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)
    kept = finish_passages(state, [dated, undated], deadline=deadline)
    assert kept["dropped_undated"] == 0
    assert len(kept["passages"]) == 2

    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", True)
    dropped = finish_passages(state, [dated, undated], deadline=deadline)
    assert dropped["passages_if_keep"] == 2
    assert dropped["passages_if_drop"] == 1
    assert dropped["dropped_undated"] == 1
    assert [item.url for item in dropped["passages"]] == ["https://example.com/dated"]
    assert dropped["passages"][0].passage_id == "P1"


def test_entity_boost_reorders(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())

    def headed(cosine, extra=""):
        words = [f"{cosine:.2f}"] + [f"w{i}" for i in range(22)]
        if extra:
            words.extend(extra.split())
        return " ".join(words) + "."

    # CLAIM is the encoder's marker and an acronym, so it is a key term.
    # Chennai is the place. The closer window names neither, so the one
    # that names both outranks it.
    passages, _cosines, _dropped = select_passages(
        {"claim": "CLAIM flood in Chennai"},
        [
            _page("https://plain.example/a", [headed(0.60)]),
            _page("https://named.example/a", [headed(0.56, "CLAIM Chennai")]),
        ],
    )
    assert [item.url for item in passages] == [
        "https://named.example/a",
        "https://plain.example/a",
    ]
    assert passages[0].cos < passages[1].cos
    assert passages[0].score > passages[1].score
    assert passages[0].score == pytest.approx(0.56 + config.ENTITY_BOOST)
    assert passages[1].score == pytest.approx(0.60)

    # The place is only in the state, not capitalised in this claim text.
    by_entity, _cosines, _dropped = select_passages(
        {
            "claim": "CLAIM the bank left the country",
            "entities": {"places": ["Sri Lanka"]},
        },
        [
            _page("https://plain.example/b", [headed(0.70)]),
            _page("https://named.example/b", [headed(0.66, "CLAIM Sri Lanka")]),
        ],
    )
    assert [item.url for item in by_entity] == [
        "https://named.example/b",
        "https://plain.example/b",
    ]
    assert by_entity[0].cos < by_entity[1].cos


def test_reddit_and_linkedin_make_no_passages(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    pages = [
        _page("https://www.reddit.com/r/india/comments/1", [_sentence("0.95", n=24)]),
        _page("https://www.linkedin.com/posts/abc", [_sentence("0.90", n=24)]),
        _page("https://www.quora.com/What-happened", [_sentence("0.85", n=24)]),
        _page("https://news.example/a", [_sentence("0.60", n=24)]),
    ]
    passages, _cosines, _dropped = select_passages(
        {"claim": "CLAIM the notes were cancelled"},
        pages,
    )
    assert [item.url for item in passages] == ["https://news.example/a"]


def test_published_through_claim_date_is_sent_when_the_guard_is_on(monkeypatch):
    seen = set()

    def fake(query, **kwargs):
        seen.add(kwargs.get("end_date"))
        return []

    set_fake_search(fake)
    state = _state("Health")
    state["claim_date"] = "2025-04-08"
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", True)
    gather_candidates(state)
    # Tavily's end_date is exclusive ("before this day"), so the day after
    # the claim is what "on or before the claim" means.
    assert seen == {"2025-04-09"}
    seen.clear()
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)
    gather_candidates(state)
    assert seen == {None}


def test_circulation_and_instagram_quote_make_no_passage(monkeypatch):
    import misinfo_agent.investigate as investigate

    monkeypatch.setattr(investigate, "get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    instagram = _page(
        "https://www.instagram.com/p/abc",
        [_sentence("0.95", n=24)],
    )
    quoted = _page(
        "https://example.com/post",
        [(
            "0.90 outlook money The viral posts claim new rules for notes "
            "but PIB Fact Check confirms this is fake and misleading to readers"
        )],
    )
    passages, _cosines, _dropped = select_passages(
        {"claim": "CLAIM the notes were cancelled"},
        [instagram, quoted],
    )
    assert passages == []


def _stance_passage(passage_id, url, text, cos=0.8, fetched=True, published="2020-01-02"):
    return Passage(
        passage_id=passage_id,
        url=url,
        netloc=url.split("/")[2],
        title="t",
        text=text,
        cos=cos,
        score=cos,
        published_date=published,
        date_source="tavily" if published else None,
        fetched=fetched,
        after_claim=False,
    )


def test_quote_that_is_not_in_the_passage_becomes_neutral():
    from misinfo_agent.investigate import StanceJudgement, apply_stance, build_evidence

    passage = _stance_passage(
        "P1",
        "https://example.com/a",
        "The bank said the retail sale continues and corporate accounts stay open for clients.",
    )
    labelled, unverified = apply_stance(
        "The bank shut every branch",
        [passage],
        [StanceJudgement(
            passage_id="P1",
            stance="supports",
            quote="this sentence was not on the page",
            reason="It agrees",
        )],
    )
    assert labelled[0]["stance"] == "neutral"
    assert labelled[0]["quote_unverified"] is True
    assert unverified == 1
    assert build_evidence({"claim_date": "2024-01-01"}, labelled) == []


def test_single_and_double_quotes_still_count_as_the_same_sentence():
    from misinfo_agent.investigate import StanceJudgement, apply_stance

    # The page uses double quotes. The model writes single quotes. The
    # words were copied, so a real refute must not become neutral.
    text = (
        'The Union Health Ministry termed the reports "misleading, incorrect, and baseless" '
        "and said it had not ordered warning labels."
    )
    labelled, unverified = apply_stance(
        "The ministry ordered warning labels on samosas",
        [_stance_passage("P1", "https://www.ndtv.com/a", text)],
        [StanceJudgement(
            passage_id="P1",
            stance="refutes",
            quote="The Union Health Ministry termed the reports 'misleading, incorrect, and baseless'",
            reason="The ministry denied the reports",
        )],
    )
    assert labelled[0]["stance"] == "refutes"
    assert labelled[0]["quote_unverified"] is False
    assert unverified == 0


def test_a_passage_that_only_repeats_the_claim_is_neutral():
    from misinfo_agent.investigate import StanceJudgement, apply_stance, build_evidence

    text = (
        "The bank left the country yesterday and customers were told nothing "
        "more about the closure at all."
    )
    passage = _stance_passage("P1", "https://example.com/a", text)
    labelled, unverified = apply_stance(
        text,
        [passage],
        [StanceJudgement(
            passage_id="P1",
            stance="supports",
            quote="The bank left the country yesterday",
            reason="The page states the claim",
        )],
    )
    assert labelled[0]["stance"] == "neutral"
    assert unverified == 0
    assert build_evidence({"claim_date": "2024-01-01"}, labelled) == []
    rules = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "misinfo_agent" / "prompts" / "stance_v3.txt"
    ).read_text(encoding="utf-8")
    assert "only repeats" in rules
    assert "temporary" in rules and "permanent" in rules
    assert "ALL key details" in rules
    assert "who, what, amount, place" in rules
    assert "old event" in rules
    assert "from <date>" in rules
    assert "permanently" in rules
    assert "CAN refute" in rules
    assert "8 November 2016" in rules
    assert "side remark" in rules


def test_stance_v4_has_the_tense_and_named_after_rules():
    rules = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "misinfo_agent" / "prompts" / "stance_v4.txt"
    ).read_text(encoding="utf-8")
    assert 'reason must be "planned, not confirmed"' in rules
    assert "visited, inaugurated, announced, banned" in rules
    assert "IIM Bangalore metro station" in rules
    assert "Modi stadium" in rules
    assert "named after" in rules


def test_stance_v5_rules_never_change_a_refute():
    rules = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "misinfo_agent" / "prompts" / "stance_v5.txt"
    ).read_text(encoding="utf-8")
    assert "visited, inaugurated, was banned, announced" in rules
    assert "IIM Bangalore metro station" in rules
    assert "Modi stadium" in rules
    assert "never turn a refute into neutral" in rules
    assert "will continue corporate banking" in rules
    assert "is withdrawing" in rules
    assert "will be invalid from 1 December" in rules


def test_stance_v6_different_event_is_neutral():
    # v6 stays on disk. v7 is the call that runs, and it still contains
    # every v6 rule so a different event stays neutral and a real
    # contradiction stays a refute.
    v6 = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "misinfo_agent" / "prompts" / "stance_v6.txt"
    ).read_text(encoding="utf-8")
    rules = v6
    assert "different event" in rules
    assert "different metro line" in rules
    assert "PM Modi inaugurated the Pink Line of Namma Metro" in rules
    assert "PM Modi inaugurated the Yellow Line of Bangalore Metro" in rules
    assert "different line, different event" in rules
    assert "The Pink Line opening was postponed" in rules
    assert "never turn a refute into neutral" in rules
    assert "will continue corporate banking" in rules
    assert "old event" in rules


def test_stance_v7_keeps_v6_and_states_the_share_date():
    from misinfo_agent.investigate import Passage, judge_stance
    from misinfo_agent.llm import get_usage, set_fake

    v6_rules = (
        (__import__("pathlib").Path(__file__).resolve().parents[1]
         / "misinfo_agent" / "prompts" / "stance_v6.txt")
        .read_text(encoding="utf-8")
        .split("CLAIM:")[0]
    )
    seen = {}

    def fake(prompt, schema, images):
        seen["prompt"] = prompt
        seen["schema"] = schema.__name__
        return {"judgements": []}

    passage = Passage(
        passage_id="P1",
        url="https://example.com/a",
        netloc="example.com",
        title="t",
        text="A passage about the Pink Line and nothing else of substance here.",
        cos=0.9,
        fetched=True,
    )
    state = {
        "claim": "PM Modi inaugurated the Pink Line and then visited the campus",
        "claim_date": "2026-11-18",
        "atomic_claims": [
            {"id": "A1", "text": "PM Modi inaugurated the Pink Line", "core": True, "checkable": True},
            {"id": "A2", "text": "PM Modi visited the IIM Bangalore campus", "core": False, "checkable": True},
        ],
    }
    before = len(get_usage())
    set_fake(fake)
    try:
        judge_stance(state, [passage])
    finally:
        set_fake(None)
    prompt = seen["prompt"]
    # The multi-claim call is the v7 file plus one paragraph. The v6
    # rules are in that file unchanged, and they apply per atomic claim.
    assert v6_rules in prompt
    assert "This claim was shared on 2026-11-18" in prompt
    assert "till Monday" in prompt
    assert "different occasion is NEUTRAL, not a refute" in prompt
    assert "may slightly lower blood sugar" in prompt
    assert "IIMB metro station" in prompt
    assert "IIM Bangalore campus" in prompt
    assert "Apply every rule above to that atomic claim alone" in prompt
    assert seen["schema"] == "AtomicStanceBatch"
    stance_rows = [row for row in get_usage()[before:] if row.get("purpose") == "stance"]
    assert stance_rows[-1]["prompt_version"] == "stance_v9"
    assert "Sucking a lemon or drinking lemon-infused water can kill COVID-19 viral particles" in prompt
    assert "never a support" in prompt

    seen.clear()
    single = {
        "claim": "PM Modi inaugurated the Pink Line",
        "claim_date": "2026-11-18",
    }
    set_fake(fake)
    try:
        judge_stance(single, [passage])
    finally:
        set_fake(None)
    assert v6_rules in seen["prompt"]
    assert "This claim was shared on 2026-11-18" in seen["prompt"]
    assert "more than one atomic claim" not in seen["prompt"]
    assert seen["schema"] == "StanceBatch"


# One sentence per rule. Both the single-claim prompt and the multi-claim
# prompt have to carry it, because a rule that only one path sees is a
# rule the other path has lost.
_V9_RULES = (
    ("share date", "This claim was shared on 2026-11-18"),
    ("relative time", "till Monday"),
    ("different occasion", "different occasion is NEUTRAL, not a refute"),
    ("cure", "may slightly lower blood sugar"),
    ("small effect refutes a cure", 'refutes "cures diabetes"'),
    ("named place", "IIMB metro station"),
    ("campus is not the station", "visited the IIM Bangalore campus"),
    ("planned is not happened", "planned, not confirmed"),
    ("different event", "different line, different event"),
    ("old evidence cannot support", "Old evidence CANNOT support"),
    ("old evidence can refute", "It CAN refute"),
    ("myth list is a refute", "Sucking a lemon or drinking lemon-infused water can kill COVID-19 viral particles"),
    ("myth list is never support", "never a support"),
)


def test_stance_v9_rules_are_in_both_rendered_prompts():
    from misinfo_agent.investigate import Passage, judge_stance
    from misinfo_agent.llm import set_fake

    seen = {}

    def fake(prompt, schema, images):
        seen["prompt"] = prompt
        return {"judgements": []}

    passage = Passage(
        passage_id="P1",
        url="https://example.com/a",
        netloc="example.com",
        title="t",
        text="A passage about the Pink Line and nothing else of substance here.",
        cos=0.9,
        fetched=True,
    )
    multi = {
        "claim": "PM Modi inaugurated the Pink Line and then visited the campus",
        "claim_date": "2026-11-18",
        "atomic_claims": [
            {"id": "A1", "text": "PM Modi inaugurated the Pink Line", "core": True, "checkable": True},
            {"id": "A2", "text": "PM Modi visited the IIM Bangalore campus", "core": False, "checkable": True},
        ],
    }
    single = {
        "claim": "PM Modi inaugurated the Pink Line",
        "claim_date": "2026-11-18",
    }
    set_fake(fake)
    try:
        judge_stance(multi, [passage])
        multi_prompt = seen["prompt"]
        judge_stance(single, [passage])
        single_prompt = seen["prompt"]
    finally:
        set_fake(None)
    assert "more than one atomic claim" in multi_prompt
    assert "more than one atomic claim" not in single_prompt
    for name, sentence in _V9_RULES:
        assert sentence in single_prompt, name
        assert sentence in multi_prompt, name


def test_credibility_tiers():
    from misinfo_agent.investigate import credibility_tier

    assert credibility_tier("https://www.rbi.org.in/note") == 1
    assert credibility_tier("https://pib.gov.in/press") == 1
    assert credibility_tier("https://census.nic.in/data") == 1
    assert credibility_tier("https://www.whitehouse.gov/briefing") == 1
    assert credibility_tier("https://www.who.int/news") == 1
    assert credibility_tier("https://www.reuters.com/world") == 2
    assert credibility_tier("https://www.bbc.co.uk/news") == 2
    assert credibility_tier("https://bfsi.economictimes.indiatimes.com/news") == 2
    assert credibility_tier("https://news.mit.edu/item") == 2
    assert credibility_tier("https://www.iisc.ac.in/page") == 2
    assert credibility_tier("https://www.nature.com/articles/1") == 2
    assert credibility_tier("https://www.nationalacademies.org/report") == 2
    assert credibility_tier("https://en.wikipedia.org/wiki/Flood") == 3
    assert credibility_tier("https://www.stvincents.org/health") == 3
    assert credibility_tier("https://www.mayoclinic.org/diseases") == 3
    assert credibility_tier("https://cityhospital.org/news") == 3
    assert credibility_tier("https://medium.com/post") == 4
    assert credibility_tier("https://unknown.example/post") == 4
    # A snippet is still the source it came from. The flag is separate.
    assert credibility_tier("https://www.reuters.com/world") == 2
    assert credibility_tier("https://www.dw.com/en/story") == 2
    assert credibility_tier("https://www.euronews.com/story") == 2
    assert credibility_tier("https://globalnews.ca/news/1") == 2
    assert credibility_tier("https://www.france24.com/en/story") == 2
    assert credibility_tier("https://www.theguardian.com/world") == 2
    assert credibility_tier("https://www.washingtonpost.com/world") == 2
    assert credibility_tier("https://www.cbsnews.com/news") == 2
    assert credibility_tier("https://www.nbcnews.com/news") == 2
    assert credibility_tier("https://edition.cnn.com/world") == 2
    assert credibility_tier("https://www.indiatoday.in/india") == 2
    assert credibility_tier("https://timesofindia.indiatimes.com/india") == 2
    assert credibility_tier("https://m.economictimes.com/news") == 2
    assert credibility_tier("https://www.deccanherald.com/india") == 2
    assert credibility_tier("https://www.newindianexpress.com/states") == 2
    assert credibility_tier("https://www.telegraphindia.com/india") == 2
    assert credibility_tier("https://www.business-standard.com/india") == 2
    assert credibility_tier("https://www.financialexpress.com/india") == 2
    assert credibility_tier("https://www.whitehouse.gov/briefing") == 1
    assert credibility_tier("https://www.gov.uk/guidance") == 1
    assert credibility_tier("https://salud.gob.mx/nota") == 1
    assert credibility_tier("https://weather.gc.ca/city") == 1
    assert credibility_tier("https://news.uq.edu.au/article") == 2
    assert credibility_tier("https://www.ox.ac.uk/news") == 2
    # National wires, the same bar as PTI. ianslive.in is its own host.
    assert credibility_tier("https://www.tribuneindia.com/news/india/story") == 2
    assert credibility_tier("https://www.aninews.in/news") == 2
    assert credibility_tier("https://www.ians.in/news") == 2
    assert credibility_tier("https://ianslive.in/news") == 2
    assert credibility_tier("https://www.uniindia.com/news") == 2


def test_minhash_groups_near_copies():
    from misinfo_agent.investigate import assign_groups

    words = [f"word{i}" for i in range(40)]
    base = " ".join(words)
    near = " ".join(words[:-1] + ["changed"])
    other = " ".join(f"other{i}" for i in range(40))
    copies = assign_groups([
        _stance_passage("P1", "https://alpha.com/a", base),
        _stance_passage("P2", "https://beta.com/b", near),
    ])
    assert copies["P1"] == copies["P2"]
    apart = assign_groups([
        _stance_passage("P1", "https://alpha.com/a", base),
        _stance_passage("P2", "https://beta.com/b", other),
    ])
    assert apart["P1"] != apart["P2"]
    same_newsroom = assign_groups([
        _stance_passage("P1", "https://alpha.com/a", base),
        _stance_passage("P2", "https://www.alpha.com/b", other),
    ])
    assert same_newsroom["P1"] == same_newsroom["P2"]


def test_evidence_rows_are_supports_and_refutes_only():
    from misinfo_agent.investigate import build_evidence

    def labelled(passage, stance, quote):
        return {
            "passage": passage,
            "stance": stance,
            "quote": quote,
            "reason": "A short reason here",
            "quote_unverified": False,
        }

    rows = build_evidence(
        {"claim_date": "2024-06-01"},
        [
            labelled(
                _stance_passage("P1", "https://www.thehindu.com/a", "hindu text about the notice", 0.9),
                "supports",
                "hindu",
            ),
            labelled(
                _stance_passage("P2", "https://www.rbi.org.in/b", "rbi text about the notice", 0.5, published="2024-05-01"),
                "supports",
                "rbi",
            ),
            labelled(
                _stance_passage("P3", "https://www.reuters.com/c", "reuters text about a denial", 0.4),
                "refutes",
                "reuters",
            ),
            labelled(
                _stance_passage("P4", "https://en.wikipedia.org/wiki/X", "wikipedia repeats a rumour", 0.95),
                "neutral",
                "",
            ),
        ],
    )
    assert [row["id"] for row in rows] == ["E1", "E2", "E3"]
    assert [(row["netloc"], row["tier"], row["stance"]) for row in rows] == [
        ("www.rbi.org.in", 1, "supports"),
        ("www.thehindu.com", 2, "supports"),
        ("www.reuters.com", 2, "refutes"),
    ]
    assert rows[0]["age_days"] == 31
    assert rows[1]["group_id"] != rows[2]["group_id"]


def test_a_group_with_both_stances_is_flagged_and_kept():
    from misinfo_agent.investigate import _stats, build_evidence

    rows = build_evidence(
        {"claim_date": "2024-06-01"},
        [
            {
                "passage": _stance_passage(
                    "P1",
                    "https://www.thehindu.com/a",
                    "The bank confirmed the new rule applies from December.",
                    0.9,
                ),
                "stance": "supports",
                "quote": "The bank confirmed the new rule",
                "reason": "It confirms the rule",
                "quote_unverified": False,
            },
            {
                "passage": _stance_passage(
                    "P2",
                    "https://www.thehindu.com/b",
                    "The bank said the old notes remain valid.",
                    0.8,
                ),
                "stance": "refutes",
                "quote": "the old notes remain valid",
                "reason": "It denies the rule",
                "quote_unverified": False,
            },
            {
                "passage": _stance_passage(
                    "P3",
                    "https://www.reuters.com/c",
                    "Reuters reported that the ministry confirmed the rule.",
                    0.7,
                ),
                "stance": "supports",
                "quote": "the ministry confirmed the rule",
                "reason": "A second newsroom agrees",
                "quote_unverified": False,
            },
        ],
    )
    by_url = {row["url"]: row for row in rows}
    hindu_support = by_url["https://www.thehindu.com/a"]
    hindu_refute = by_url["https://www.thehindu.com/b"]
    reuters = by_url["https://www.reuters.com/c"]
    # Both sentences stay. The gate needs the disagreement, not a deletion.
    assert len(rows) == 3
    assert hindu_support["group_id"] == hindu_refute["group_id"]
    assert hindu_support["group_id"] != reuters["group_id"]
    assert hindu_support["stance"] == "supports"
    assert hindu_refute["stance"] == "refutes"
    assert hindu_support["group_conflict"] is True
    assert hindu_refute["group_conflict"] is True
    assert reuters["group_conflict"] is False
    stats = _stats(
        passages=[],
        labelled=[],
        evidence=rows,
        pages=[],
        searches=0,
        llm_calls=0,
        usage=[],
        unverified=0,
        notes=[],
        started=0,
        queries=[],
        topup_used=False,
        topup_queries=[],
    )
    assert stats["group_conflicts"] == [hindu_support["group_id"]]


def test_search_quota_returns_a_partial_result():
    def fake(query, **kwargs):
        raise SearchQuotaError("investigate_neutral", "monthly cap")

    set_fake_search(fake)
    result = investigate(_state("Health"))
    assert result["passages"] == []
    assert result["evidence"] == []
    assert result["investigate_stats"]["notes"]
    assert "monthly cap" in result["investigate_stats"]["notes"][0]
    assert result["investigate_stats"]["search_capped"] == "monthly"


def test_cap_on_query_three_keeps_the_pages_already_found(monkeypatch, tmp_path):
    """A cap on a later query must not throw away the pages already in hand.

    Stance still reads those pages. The audit row records which cap fired.
    """
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.audit import audit
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    state = _state("Health", claim="CLAIM the ministry banned the medicine")
    planned = build_queries(state)
    assert len(planned) >= 3
    first, second, third = planned[0].text, planned[1].text, planned[2].text
    shared = "The Reserve Bank said no circular has been issued declaring the old notes invalid"
    filler = (
        "from December and holders can keep using them every day at shops and banks across India "
        "without exchanging them for a new series or meeting any deadline the forwarded message "
        "invented for the public and branches have been told to treat that message as a rumour "
        "rather than a circular while cash machines continue to dispense the same notes to customers "
        "who need cash for the market the school fee the rent and the train ticket home tonight"
    )

    def page(url, tail):
        return {
            "url": url,
            "title": "Notice",
            "content": "snippet",
            "score": 0.9,
            "published_date": "2020-01-02",
            "raw_content": f"0.90 {shared} {filler} {tail}",
        }

    def fake_search(query, **kwargs):
        if query == third:
            raise SearchQuotaError(
                "investigate_neutral",
                "per-claim search cap reached (8 calls)",
                cap="per_claim",
            )
        if query == first:
            return [page("https://www.rbi.org.in/notice", "at the reserve bank counter today")]
        if query == second:
            return [page("https://www.thehindu.com/story", "at the hindu newsroom desk today")]
        return []

    seen_prompts = []

    def fake_llm(prompt, schema, images):
        seen_prompts.append(prompt)
        return {
            "judgements": [
                {"passage_id": f"P{n}", "stance": "supports", "quote": shared, "reason": "The bank denies it."}
                for n in range(1, 8)
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(state)
    finally:
        set_fake(None)
        set_fake_search(None)
    urls = {canonical_url(passage["url"]) for passage in result["passages"]}
    assert "https://rbi.org.in/notice" in urls
    assert "https://thehindu.com/story" in urls
    evidence_urls = {canonical_url(row["url"]) for row in result["evidence"]}
    assert "https://rbi.org.in/notice" in evidence_urls
    # Both pages were in the stance prompt. A repeated quote can leave only
    # one evidence row; that filter is not the cap throwing a page away.
    assert seen_prompts
    assert "reserve bank counter" in seen_prompts[0]
    assert "hindu newsroom" in seen_prompts[0]
    assert result["investigate_stats"]["search_capped"] == "per_claim"
    assert result["investigate_stats"]["llm_calls"] >= 1
    notes = " ".join(result["investigate_stats"]["notes"])
    assert "search_capped=per_claim" in notes
    assert "per-claim search cap" in notes
    stored = dict(state)
    stored["search_capped"] = result["investigate_stats"]["search_capped"]
    stored["investigate_stats"] = result["investigate_stats"]
    stored["verdict"] = "Unverifiable"
    stored["run_marks"] = {"run_id": "cap-test"}
    row = audit(stored, db_path=tmp_path / "audit.db", additions_path=tmp_path / "additions.parquet")
    assert row["audit"]["search_capped"] == "per_claim"


def test_cache_only_miss_never_calls_tavily(monkeypatch):
    """A cache miss in cache-only mode is an empty result, not a Tavily call."""
    monkeypatch.setenv("SEARCH_CACHE_ONLY", "1")
    called = []

    def boom(*args, **kwargs):
        called.append(1)
        raise AssertionError("Tavily was called")

    monkeypatch.setattr(search, "_real", boom)
    set_fake_search(None)
    reset_search_usage()
    hits = run_search("a query that is not in the cache", purpose="cache_only_test")
    assert hits == []
    assert called == []
    usage = get_search_usage()
    assert usage
    assert usage[-1]["cache_miss"] is True
    assert usage[-1]["cached"] is False


def test_llm_error_keeps_the_passages(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import LLMError, set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    words = (
        "The Reserve Bank said no circular has been issued declaring the old notes invalid "
        "from December and holders can keep using them every day at shops and banks"
    ).split()
    text = "0.90 " + " ".join(words * 6)
    quote = "The Reserve Bank said no circular has been issued declaring the old notes invalid"

    def fake_search(query, **kwargs):
        return [{
            "url": "https://www.rbi.org.in/notice",
            "title": "RBI",
            "content": "snippet",
            "score": 0.8,
            "published_date": "2020-01-02",
            "raw_content": text,
        }]

    def fake_llm(prompt, schema, images):
        raise LLMError("stance", "daily cap")

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(_state("Health", claim="CLAIM the ministry banned the medicine"))
    finally:
        set_fake(None)
    assert result["evidence"] == []
    assert result["passages"]
    assert result["investigate_stats"]["notes"]
    assert result["investigate_stats"]["llm_calls"] == 1
    # An empty table after a failed call is a system error. Box 9 must not
    # read it as "not enough evidence".
    assert result["investigate_stats"]["stance_failed"] is True


def test_stance_builds_an_evidence_row(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    good = (
        "0.90 The Reserve Bank said no circular has been issued declaring the old notes invalid "
        "from December and holders can keep using them every day at shops and banks across India "
        "without exchanging them for a new series or meeting any deadline the forwarded message "
        "invented for the public, and branches have been told to treat that message as a rumour "
        "rather than a circular while cash machines continue to dispense the same notes to customers "
        "who need cash for the market, the school fee, the rent and the train ticket home tonight"
    )
    bad = (
        "0.70 People forwarded a long message about a deadline that the ministry never announced "
        "and neighbours argued about it in the queue outside the branch while the guard asked them "
        "to stand in a single line and the clerk kept repeating that no such order had arrived by "
        "post or by email or by the internal system the branch uses for notices from the head office "
        "this week or last week or the week before that during the audit of the cash counter"
    )
    quote = "The Reserve Bank said no circular has been issued declaring the old notes invalid"

    def fake_search(query, **kwargs):
        return [
            {
                "url": "https://www.rbi.org.in/notice",
                "title": "RBI",
                "content": "snippet",
                "score": 0.9,
                "published_date": "2020-01-02",
                "raw_content": good,
            },
            {
                "url": "https://example.com/story",
                "title": "A blog",
                "content": "snippet",
                "score": 0.2,
                "published_date": "2020-01-02",
                "raw_content": bad,
            },
        ]

    def fake_llm(prompt, schema, images):
        return {
            "judgements": [
                {"passage_id": "P1", "stance": "supports", "quote": quote, "reason": "The bank denies the circular."},
                {"passage_id": "P2", "stance": "supports", "quote": "not in the passage", "reason": "A wrong quote"},
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(_state("Health", claim="CLAIM the ministry banned the medicine"))
    finally:
        set_fake(None)
    assert result["investigate_stats"]["quote_unverified"] == 1
    assert result["investigate_stats"]["n_supports"] == 1
    assert result["investigate_stats"]["n_neutral"] == 1
    assert len(result["evidence"]) == 1
    row = result["evidence"][0]
    assert row["id"] == "E1"
    assert row["stance"] == "supports"
    assert row["tier"] == 1
    assert row["quote"] == quote
    assert row["atomic_id"] == "A1"
    assert result["investigate_stats"]["stance_failed"] is False
    # One claim: the per-atomic block is the same totals, under A1.
    block = result["investigate_stats"]["by_atomic"]["A1"]
    assert block["n_supports"] == 1
    assert block["groups_supporting"] == 1


def test_two_atomic_claims_label_each_passage_and_search_the_second(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    # The fake encoder reads the first token. CLAIM is the query vector.
    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    quote = "The Reserve Bank said no circular has been issued"
    # Eighty words or more, or fetch treats raw_content as a stub and tries
    # the network, which an offline test refuses. The first token is the
    # cosine the fake encoder reads.
    page = (
        "0.90 The Reserve Bank said no circular has been issued declaring the old notes invalid "
        "from December and holders can keep using them every day at shops and banks across India "
        "without exchanging them for a new series or meeting any deadline the forwarded message "
        "invented for the public, and branches have been told to treat that message as a rumour "
        "rather than a circular while cash machines continue to dispense the same notes to customers "
        "who need cash for the market, the school fee, the rent and the train ticket home tonight "
        "and the guard at the door kept repeating the same instruction to every person who asked "
        "about a notice that the branch had never received from the head office during this week"
    )

    def fake_search(query, **kwargs):
        return [
            {
                "url": "https://www.rbi.org.in/notice",
                "title": "RBI",
                "content": "snippet",
                "score": 0.9,
                "published_date": "2020-01-02",
                "raw_content": page,
            },
            {
                "url": "https://www.thehindu.com/story",
                "title": "The Hindu",
                "content": "snippet",
                "score": 0.8,
                "published_date": "2020-01-02",
                "raw_content": page,
            },
        ]

    def fake_llm(prompt, schema, images):
        import re

        ids = list(dict.fromkeys(re.findall(r"(?m)^(P\d+)$", prompt)))
        judgements = []
        for passage_id in ids:
            judgements.append({
                "passage_id": passage_id,
                "atomic_id": "A1",
                "stance": "supports",
                "quote": quote,
                "reason": "The bank confirms the ban.",
            })
            judgements.append({
                "passage_id": passage_id,
                "atomic_id": "A2",
                "stance": "neutral",
                "quote": "",
                "reason": "The clinics are not mentioned.",
            })
        return {"judgements": judgements}

    state = _state(
        "Health",
        claim="CLAIM the ministry banned the medicine and then closed the clinics",
    )
    state["atomic_claims"] = [
        {"id": "A1", "text": "CLAIM the ministry banned the medicine", "core": True},
        {"id": "A2", "text": "The ministry closed the clinics", "core": False},
    ]
    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(state)
    finally:
        set_fake(None)
        set_fake_search(None)
    queries = result["investigate_stats"]["queries"]
    assert any("The ministry closed the clinics" in query for query in queries)
    assert {row["atomic_id"] for row in result["evidence"]} == {"A1"}
    assert result["evidence"]
    assert all(row["stance"] == "supports" for row in result["evidence"])
    block = result["investigate_stats"]["by_atomic"]
    assert block["A1"]["groups_supporting"] >= 1
    assert block["A2"]["groups_supporting"] == 0
    assert block["A2"]["n_neutral"] >= 1


def test_a_group_can_support_one_atomic_claim_and_refute_another():
    from misinfo_agent.investigate import _stats, build_evidence

    quote_a = "The Pink Line opened in Bengaluru on Monday morning"
    quote_b = "The campus visit was not on the programme"
    text = f"{quote_a}. {quote_b}. " + " ".join(f"w{i}" for i in range(30))
    rows = build_evidence(
        {"claim_date": "2024-06-01"},
        [
            {
                "passage": _stance_passage("P1", "https://www.thehindu.com/a", text, 0.9),
                "stance": "supports",
                "quote": quote_a,
                "reason": "It confirms the opening",
                "quote_unverified": False,
                "atomic_id": "A1",
            },
            {
                "passage": _stance_passage("P1", "https://www.thehindu.com/a", text, 0.9),
                "stance": "refutes",
                "quote": quote_b,
                "reason": "It denies the visit",
                "quote_unverified": False,
                "atomic_id": "A2",
            },
        ],
    )
    assert [row["atomic_id"] for row in rows] == ["A1", "A2"]
    assert all(row["group_conflict"] is False for row in rows)
    stats = _stats(
        passages=[],
        labelled=[
            {"stance": "supports", "atomic_id": "A1"},
            {"stance": "refutes", "atomic_id": "A2"},
        ],
        evidence=rows,
        pages=[],
        searches=0,
        llm_calls=0,
        usage=[],
        unverified=0,
        notes=[],
        started=0,
        queries=[],
        topup_used=False,
        topup_queries=[],
    )
    assert stats["group_conflicts"] == []
    assert stats["by_atomic"]["A1"]["groups_supporting"] == 1
    assert stats["by_atomic"]["A1"]["groups_refuting"] == 0
    assert stats["by_atomic"]["A2"]["groups_refuting"] == 1
    assert stats["by_atomic"]["A2"]["groups_supporting"] == 0


def test_short_quote_keeps_the_previous_sentence():
    from misinfo_agent.investigate import build_evidence

    text = (
        "Health officials said hot lemon water does not kill the virus. "
        "This is a myth."
    )
    passage = _stance_passage("P1", "https://www.bbc.com/a", text)
    rows = build_evidence(
        {"claim_date": "2024-01-01"},
        [{
            "passage": passage,
            "stance": "refutes",
            "quote": "This is a myth.",
            "reason": "The page denies it",
            "quote_unverified": False,
        }],
    )
    assert rows[0]["quote"] == "This is a myth."
    assert rows[0]["display_quote"] == (
        "Health officials said hot lemon water does not kill the virus. This is a myth."
    )
    long_quote = "Health officials said hot lemon water does not kill the virus"
    long_rows = build_evidence(
        {"claim_date": "2024-01-01"},
        [{
            "passage": passage,
            "stance": "refutes",
            "quote": long_quote,
            "reason": "The page denies it",
            "quote_unverified": False,
        }],
    )
    assert long_rows[0]["quote"] == long_quote
    assert long_rows[0]["display_quote"] == long_quote


def test_factcheckhub_is_dropped_in_blocked_mode(monkeypatch):
    import misinfo_agent.investigate as investigate_module

    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    assert factcheck_blocked("https://www.factcheckhub.com/saudi-dress", "a report") is True
    assert factcheck_blocked("https://checkyourfact.com/story", "") is True
    assert factcheck_blocked("https://africacheck.org/reports/one", "") is True
    assert factcheck_blocked("https://example.com/news/webqoof/story", "a report") is True
    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    passages, _cosines, _dropped = select_passages(
        {"claim": "CLAIM the notes were cancelled"},
        [_page("https://factcheckhub.com/story", [_sentence("0.90", n=24)])],
    )
    assert passages == []
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    assert factcheck_blocked("https://factcheckhub.com/story", "a report") is False


def test_old_event_cannot_support_a_now_claim():
    from misinfo_agent.investigate import StanceJudgement, apply_stance

    passage = _stance_passage(
        "P1",
        "https://www.dw.com/story",
        "Crown Prince Mohammed bin Salman said women should be able to choose what they wear.",
        published="2018-03-19",
    )
    claim = "Saudi Arabian Crown Prince Mohammad Bin Salman said that women can now choose what to wear."
    labelled, unverified = apply_stance(
        claim,
        [passage],
        [StanceJudgement(passage_id="P1", stance="supports", quote="women should be able to choose what they wear", reason="He said it")],
        "2025-04-08",
    )
    assert labelled[0]["stance"] == "neutral"
    assert labelled[0]["reason"] == "old event"
    assert unverified == 0
    # A contradiction from the same old page is still a refutation.
    labelled, _unverified = apply_stance(
        claim,
        [passage],
        [StanceJudgement(
            passage_id="P1",
            stance="refutes",
            quote="women should be able to choose what they wear",
            reason="The rule was already the opposite",
        )],
        "2025-04-08",
    )
    assert labelled[0]["stance"] == "refutes"


def test_old_rbi_withdrawal_stays_a_refute(monkeypatch):
    """The 2016 withdrawal contradicts "from 1 December". The code must not neutralise that refute."""
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    quote = "RBI withdrew legal tender status of these notes on 8 November 2016"
    claim = "CLAIM ₹500 notes printed before 2016 are invalid from 1 December"

    def fake_search(query, **kwargs):
        return [_search_hit(
            "https://www.rbi.org.in/notice",
            _body("0.90", quote),
        ) | {"published_date": "2016-11-08"}]

    def fake_llm(prompt, schema, images):
        if schema.__name__ == "TopupBatch":
            return {"queries": []}
        ids = [line for line in prompt.splitlines() if line.startswith("P") and line[1:].isdigit()]
        return {
            "judgements": [
                {
                    "passage_id": passage_id,
                    "stance": "refutes",
                    "quote": quote,
                    "reason": "The notes lost legal tender on 8 November 2016",
                }
                for passage_id in ids
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        state = _state("Finance & scams", claim=claim)
        state["claim_date"] = "2026-11-14"
        result = investigate(state)
    finally:
        set_fake(None)
    assert result["evidence"]
    row = result["evidence"][0]
    assert row["netloc"] == "rbi.org.in"
    assert row["stance"] == "refutes"
    assert row["quote"] == quote
    assert row["published_date"] == "2016-11-08"


def test_duplicate_quotes_in_a_group_collapse():
    from misinfo_agent.investigate import build_evidence

    def labelled(passage, quote):
        return {
            "passage": passage,
            "stance": "refutes",
            "quote": quote,
            "reason": "same sentence",
            "quote_unverified": False,
        }

    quote = "The ministry's denial stands"
    curly = "The ministry’s denial stands"
    rows = build_evidence(
        {"claim_date": "2024-06-01"},
        [
            labelled(_stance_passage("P1", "https://www.ndtv.com/a", "NDTV correspondents wrote The ministry's denial stands after checking the circular with the ministry", 0.9), quote),
            labelled(_stance_passage("P2", "https://www.ndtv.com/b", "A second NDTV page repeated The ministry's denial stands for readers who missed the first", 0.4), curly),
            labelled(_stance_passage("P3", "https://www.bbc.com/c", "The BBC bureau wrote a different article that also said The ministry's denial stands about the labels", 0.8), quote),
        ],
    )
    assert [row["id"] for row in rows] == ["E1", "E2"]
    assert {row["netloc"] for row in rows} == {"www.ndtv.com", "www.bbc.com"}
    # The same passage id on two outlets must not collapse the group map.
    # A top-up used to number its pages from P1 again.
    split = build_evidence(
        {},
        [
            labelled(_stance_passage("P1", "https://www.ndtv.com/a", "NDTV first account of the notice and the denial", 0.9), "NDTV first account"),
            labelled(_stance_passage("P1", "https://www.ndtv.com/b", "NDTV second account with different wording about the notice", 0.5), "NDTV second account"),
            labelled(_stance_passage("P1", "https://www.bbc.com/c", "BBC account that does not copy either NDTV story at all", 0.4), "BBC account"),
        ],
    )
    ndtv_groups = {row["group_id"] for row in split if row["netloc"] == "www.ndtv.com"}
    bbc_groups = {row["group_id"] for row in split if row["netloc"] == "www.bbc.com"}
    assert len(ndtv_groups) == 1
    assert ndtv_groups.isdisjoint(bbc_groups)
    # The unfetched copy keeps the newsroom's tier. The flag records the snippet.
    snippet = _stance_passage("P4", "https://www.reuters.com/d", "The ministry denied the labels yesterday", fetched=False)
    snippet_rows = build_evidence({}, [labelled(snippet, "The ministry's denial stands")])
    assert snippet_rows[0]["tier"] == 2
    assert snippet_rows[0]["snippet_only"] is True


def _window_row(url, text, stance, published, reason="The page says so"):
    return {
        "passage": _stance_passage("P1", url, text, published=published),
        "stance": stance,
        "quote": text,
        "reason": reason,
        "quote_unverified": False,
        "atomic_id": "A1",
    }


def test_relative_time_window_drops_another_occasion_and_keeps_a_dated_status():
    """A 'till Monday' claim is this fortnight. 'from 1 December' is not."""
    from misinfo_agent.investigate import (
        _apply_relative_window,
        _mark_reason_mismatches,
        _relative_time_claim,
        build_evidence,
    )

    schools = {
        "claim": "All Bengaluru schools closed till Monday due to heavy rain",
        "claim_date": "2026-11-19",
        "time_clues": "",
    }
    old = _window_row(
        "https://timesofindia.indiatimes.com/city",
        "However, no holiday announcement has been made for schools in the city",
        "refutes",
        "2024-12-03",
    )
    _apply_relative_window(schools, [old])
    assert old["stance"] == "neutral"
    assert old["reason"] == "about a different date"
    assert build_evidence(schools, [old]) == []

    # Three days before the share date is the same week. The refute stands.
    same_week = _window_row(
        "https://www.thehindu.com/news",
        "No holiday has been declared for Bengaluru schools this Monday",
        "refutes",
        "2026-11-16",
    )
    _apply_relative_window(schools, [same_week])
    assert same_week["stance"] == "refutes"
    assert same_week["date_unknown"] is False
    kept = build_evidence(schools, [same_week])
    assert len(kept) == 1
    assert kept[0]["stance"] == "refutes"
    assert kept[0]["date_unknown"] is False

    # The fourteenth day before the claim is still inside. The next day is not.
    edge = _window_row("https://www.ndtv.com/a", "Schools stay open on Monday", "supports", "2026-11-05")
    outside = _window_row("https://www.ndtv.com/b", "Schools stay open on Monday", "supports", "2026-11-04")
    later = _window_row("https://www.ndtv.com/c", "Schools were shut last month", "refutes", "2026-11-20")
    _apply_relative_window(schools, [edge, outside, later])
    assert edge["stance"] == "supports"
    assert outside["stance"] == "neutral"
    assert outside["reason"] == "about a different date"
    assert later["reason"] == "about a different date"

    # No trusted day: the label stays, and the row says the date is unknown.
    undated = _window_row(
        "https://www.deccanherald.com/city",
        "The department has not announced a school holiday",
        "refutes",
        None,
    )
    _apply_relative_window(schools, [undated])
    assert undated["stance"] == "refutes"
    assert undated["date_unknown"] is True
    undated_rows = build_evidence(schools, [undated])
    assert undated_rows[0]["stance"] == "refutes"
    assert undated_rows[0]["date_unknown"] is True

    # Extract can put the weekday on time_clues and leave the claim line plain.
    assert _relative_time_claim({"claim": "Schools are closed", "time_clues": "till Monday"}) is True
    assert _relative_time_claim({"claim": "Telegram has been banned from 2026"}) is False

    notes = {
        "claim": "₹500 notes printed before 2016 are invalid from 1 December",
        "claim_date": "2026-11-19",
        "time_clues": "",
    }
    assert _relative_time_claim(notes) is False
    rbi = _window_row(
        "https://www.rbi.org.in/notice",
        "500 and 1000 denominations shall cease to be legal tender with effect from November 09, 2016",
        "refutes",
        "2016-11-08",
    )
    _apply_relative_window(notes, [rbi])
    assert rbi["stance"] == "refutes"
    assert rbi["reason"] != "about a different date"
    rbi_rows = build_evidence(notes, [rbi])
    assert len(rbi_rows) == 1
    assert rbi_rows[0]["stance"] == "refutes"
    assert rbi_rows[0]["published_date"] == "2016-11-08"

    # The window rewrites the reason, so the mismatch flag does not fire on it.
    _mark_reason_mismatches([old])
    assert old["reason_label_mismatch"] is False


def test_reason_label_mismatch_is_counted_and_the_label_stays():
    """A neutral that says 'not a cure' is a measurement, not a new label."""
    from misinfo_agent.investigate import _mark_reason_mismatches, build_evidence

    neutral = _window_row(
        "https://www.healthline.com/garlic",
        "Garlic may help lower blood sugar for some people with type 2 diabetes",
        "neutral",
        "2024-03-01",
        reason="States garlic may help lower blood sugar, not a cure",
    )
    refute = _window_row(
        "https://www.who.int/news",
        "There is no evidence that garlic cures diabetes",
        "refutes",
        "2024-03-01",
        reason="The page says there is no evidence of a cure",
    )
    plain = _window_row(
        "https://example.com/story",
        "A clinic described how patients use garlic in food",
        "neutral",
        "2024-03-01",
        reason="The page does not discuss a cure",
    )
    # "does not" is a cue, so this one counts. The label is still neutral.
    before = plain["stance"]
    _mark_reason_mismatches([neutral, refute, plain])
    assert neutral["stance"] == "neutral"
    assert neutral["reason_label_mismatch"] is True
    assert refute["stance"] == "refutes"
    assert refute["reason_label_mismatch"] is False
    assert plain["stance"] == before
    assert plain["reason_label_mismatch"] is True
    # Neutrals are not evidence. The flag did not promote them.
    assert build_evidence({"claim_date": "2026-11-19"}, [neutral, plain]) == []
    assert build_evidence({"claim_date": "2026-11-19"}, [refute])[0]["stance"] == "refutes"


def _repair_row(passage_id, url, text, reason, stance="neutral", atomic="A1", published="2024-03-01"):
    row = _window_row(url, text, stance, published, reason=reason)
    row["passage"] = row["passage"].model_copy(update={"passage_id": passage_id})
    row["atomic_id"] = atomic
    return row


def test_repair_runs_once_when_reasons_disagree_and_not_at_all_when_they_do_not():
    """Mismatched neutrals are one repair call. A clean table makes none."""
    from misinfo_agent.investigate import _repair_mismatches
    from misinfo_agent.llm import get_usage, reset_usage, set_fake

    cure = "A review found garlic does not cure diabetes and the trials are thin."
    small = "The effects are small and do not replace treatment for type 2 diabetes."
    rows = [
        _repair_row("P1", "https://www.who.int/news", cure, "not a cure"),
        _repair_row("P2", "https://www.nih.gov/garlic", small, "only a small effect"),
    ]
    calls = []

    def fake(prompt, schema, images):
        calls.append((schema.__name__, prompt))
        return {
            "judgements": [
                {
                    "passage_id": "P1",
                    "atomic_id": "A1",
                    "stance": "refutes",
                    "quote": "garlic does not cure diabetes",
                    "reason": "The page says it is not a cure",
                },
                {
                    "passage_id": "P2",
                    "atomic_id": "A1",
                    "stance": "refutes",
                    "quote": "The effects are small",
                    "reason": "Only a small effect",
                },
            ]
        }

    reset_usage()
    set_fake(fake)
    try:
        failed, note = _repair_mismatches({"claim": "Garlic cures diabetes"}, rows)
    finally:
        set_fake(None)
    assert failed == 0
    assert note == ""
    assert len(calls) == 1
    assert calls[0][0] == "RepairBatch"
    assert "P1" in calls[0][1] and "P2" in calls[0][1]
    usage = [row for row in get_usage() if row.get("purpose") == "stance_repair"]
    assert len(usage) == 1
    assert usage[0]["prompt_version"] == "repair_v1"
    assert rows[0]["stance"] == "refutes"
    assert rows[0]["repaired_from"] == "neutral"
    assert rows[1]["repaired_from"] == "neutral"

    clean = [_repair_row("P9", "https://example.com/a", "A clinic described how patients use garlic in food", "A recipe page")]
    def boom(prompt, schema, images):
        raise AssertionError("repair was called")

    set_fake(boom)
    try:
        failed, note = _repair_mismatches({"claim": "Garlic cures diabetes"}, clean)
    finally:
        set_fake(None)
    assert failed == 0
    assert clean[0]["stance"] == "neutral"
    assert "repaired_from" not in clean[0]


def test_repaired_refute_with_a_bad_quote_stays_neutral():
    from misinfo_agent.investigate import _repair_mismatches
    from misinfo_agent.llm import set_fake

    text = "Garlic may help lower blood sugar for some people with type 2 diabetes."
    row = _repair_row("P1", "https://www.healthline.com/garlic", text, "not a cure")

    def fake(prompt, schema, images):
        return {
            "judgements": [{
                "passage_id": "P1",
                "atomic_id": "A1",
                "stance": "refutes",
                "quote": "this sentence was not on the page",
                "reason": "The page denies a cure",
            }]
        }

    set_fake(fake)
    try:
        failed, _note = _repair_mismatches({"claim": "Garlic cures diabetes"}, [row])
    finally:
        set_fake(None)
    assert failed == 1
    assert row["stance"] == "neutral"
    assert row.get("repaired_from") in {None, ""}


def test_date_window_still_clears_a_repaired_row_about_another_day():
    """Repair may call it a refute. A 2024 page is still a different Monday."""
    from misinfo_agent.investigate import _apply_relative_window, _repair_mismatches, build_evidence
    from misinfo_agent.llm import set_fake

    text = "However, no holiday announcement has been made for schools in the city."
    row = _repair_row(
        "P1",
        "https://timesofindia.indiatimes.com/city",
        text,
        "There is no evidence of a holiday",
        published="2024-12-03",
    )
    state = {
        "claim": "All Bengaluru schools closed till Monday due to heavy rain",
        "claim_date": "2026-11-19",
        "time_clues": "",
    }

    def fake(prompt, schema, images):
        return {
            "judgements": [{
                "passage_id": "P1",
                "atomic_id": "A1",
                "stance": "refutes",
                "quote": "no holiday announcement has been made",
                "reason": "The page says there is no holiday",
            }]
        }

    set_fake(fake)
    try:
        _repair_mismatches(state, [row])
    finally:
        set_fake(None)
    assert row["stance"] == "refutes"
    assert row["repaired_from"] == "neutral"
    _apply_relative_window(state, [row])
    assert row["stance"] == "neutral"
    assert row["reason"] == "about a different date"
    assert row["repaired_from"] == "neutral"
    assert build_evidence(state, [row]) == []


def test_repair_caps_at_twenty_rows_in_one_call():
    from misinfo_agent.investigate import _repair_mismatches
    from misinfo_agent.llm import set_fake

    rows = [
        _repair_row(
            f"P{index}",
            f"https://example.com/{index}",
            f"Page {index} says garlic does not cure diabetes in this trial.",
            "not a cure",
        )
        for index in range(1, 22)
    ]
    seen = []

    def fake(prompt, schema, images):
        seen.append(prompt)
        return {"judgements": []}

    set_fake(fake)
    try:
        _repair_mismatches({"claim": "Garlic cures diabetes"}, rows)
    finally:
        set_fake(None)
    assert len(seen) == 1
    assert "passage_id: P1\n" in seen[0]
    assert "passage_id: P20\n" in seen[0]
    assert "passage_id: P21\n" not in seen[0]
    assert all(row["stance"] == "neutral" for row in rows)


def test_topup_runs_only_when_tier2_groups_are_thin(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    purposes = []

    def fake_search(query, **kwargs):
        return [
            _search_hit("https://www.reuters.com/a", _body("0.90", "Reuters reported the notice is genuine")),
            _search_hit("https://www.bbc.com/b", _body("0.80", "The BBC reported the notice is genuine")),
        ]

    def fake_llm(prompt, schema, images):
        purposes.append(schema.__name__)
        if schema.__name__ == "TopupBatch":
            raise AssertionError("top-up ran though two tier-2 groups already agreed")
        return {
            "judgements": [
                {"passage_id": "P1", "stance": "supports", "quote": "Reuters reported the notice is genuine", "reason": "Reuters confirms it"},
                {"passage_id": "P2", "stance": "supports", "quote": "The BBC reported the notice is genuine", "reason": "The BBC confirms it"},
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(_state("Health", claim="CLAIM the ministry banned the medicine"))
    finally:
        set_fake(None)
    assert result["investigate_stats"]["topup_used"] is False
    assert result["investigate_stats"]["topup_queries"] == []
    assert "TopupBatch" not in purposes
    assert _strong_in(result["evidence"]) >= 2


def test_topup_is_at_most_two_queries(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    searched = []

    def fake_search(query, **kwargs):
        searched.append(query)
        if query.startswith("extra"):
            return [_search_hit("https://www.thehindu.com/new", _body("0.70", "The Hindu reported the notice is genuine"))]
        return [_search_hit("https://www.reuters.com/a", _body("0.90", "Reuters reported the notice is genuine"))]

    def fake_llm(prompt, schema, images):
        if schema.__name__ == "TopupBatch":
            return {"queries": ["extra one", "extra two", "extra three", "extra four"]}
        quote = "The Hindu reported the notice is genuine" if "The Hindu reported" in prompt else "Reuters reported the notice is genuine"
        ids = []
        for line in prompt.splitlines():
            if line.startswith("P") and line[1:].isdigit():
                ids.append(line)
        return {
            "judgements": [
                {"passage_id": passage_id, "stance": "supports", "quote": quote, "reason": "The page confirms it"}
                for passage_id in ids
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(_state("Health", claim="CLAIM the ministry banned the medicine"))
    finally:
        set_fake(None)
    assert result["investigate_stats"]["topup_used"] is True
    assert result["investigate_stats"]["topup_queries"] == ["extra one", "extra two"]
    assert "extra three" not in searched
    assert "extra four" not in searched


def test_second_call_skips_known_queries_and_continues_ids(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    phase = {"second": False}
    second_searches = []

    def fake_search(query, **kwargs):
        if phase["second"]:
            second_searches.append(query)
        if "gamma" in query:
            return [_search_hit("https://www.ndtv.com/new", _body("0.60", "NDTV reported the notice is genuine"))]
        if query.startswith("extra"):
            return [_search_hit("https://www.thehindu.com/new", _body("0.70", "The Hindu reported the notice is genuine"))]
        return [_search_hit("https://www.reuters.com/a", _body("0.90", "Reuters reported the notice is genuine"))]

    def fake_llm(prompt, schema, images):
        if schema.__name__ == "TopupBatch":
            # Round 2 is only the plan's unused queries. A second top-up
            # would be the model inventing searches the gate did not allow.
            if phase["second"]:
                raise AssertionError("round 2 ran the top-up")
            return {"queries": ["extra one", "extra two"]}
        if "NDTV reported" in prompt:
            quote = "NDTV reported the notice is genuine"
        elif "The Hindu reported" in prompt:
            quote = "The Hindu reported the notice is genuine"
        else:
            quote = "Reuters reported the notice is genuine"
        ids = [line for line in prompt.splitlines() if line.startswith("P") and line[1:].isdigit()]
        return {
            "judgements": [
                {"passage_id": passage_id, "stance": "supports", "quote": quote, "reason": "The page confirms it"}
                for passage_id in ids
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        first = investigate(_state("Health", claim="CLAIM the ministry banned the medicine"))
        phase["second"] = True
        second_state = _state("Health", claim="CLAIM the ministry banned the medicine")
        second_state["evidence"] = first["evidence"]
        second_state["investigate_stats"] = first["investigate_stats"]
        second = investigate(second_state)
    finally:
        set_fake(None)
    first_queries = set(first["investigate_stats"]["queries"])
    assert first_queries
    assert second["investigate_stats"]["topup_used"] is False
    for query in second_searches:
        assert query not in first_queries
        assert not str(query).startswith("extra")


def test_round_two_runs_unused_plan_queries_and_skips_the_topup(monkeypatch):
    import misinfo_agent.investigate as investigate_module
    from misinfo_agent.investigate import build_queries
    from misinfo_agent.llm import set_fake

    monkeypatch.setattr(investigate_module, "get_encoder", lambda: _FakeEncoder())
    state = _state("Health", claim="CLAIM the ministry banned the medicine")
    plan = [query.text for query in build_queries(state)]
    assert len(plan) >= 2
    leftover = plan[-1]
    # One plan line was never searched. The prior evidence is E2, so a
    # new row has to continue at E3 rather than start again at E1.
    state["investigate_stats"] = {"queries": plan[:-1]}
    state["evidence"] = [{"id": "E2"}]
    searched = []

    def fake_search(query, **kwargs):
        searched.append(query)
        return [_search_hit("https://www.ndtv.com/new", _body("0.60", "NDTV reported the notice is genuine"))]

    def fake_llm(prompt, schema, images):
        if schema.__name__ == "TopupBatch":
            raise AssertionError("round 2 ran the top-up")
        ids = [line for line in prompt.splitlines() if line.startswith("P") and line[1:].isdigit()]
        return {
            "judgements": [
                {
                    "passage_id": passage_id,
                    "stance": "supports",
                    "quote": "NDTV reported the notice is genuine",
                    "reason": "The page confirms it",
                }
                for passage_id in ids
            ]
        }

    set_fake_search(fake_search)
    set_fake(fake_llm)
    try:
        result = investigate(state)
    finally:
        set_fake(None)
    assert leftover in searched
    for query in searched:
        assert query not in plan[:-1]
    assert result["investigate_stats"]["topup_used"] is False
    assert result["investigate_stats"]["topup_queries"] == []
    assert [row["id"] for row in result["evidence"]] == ["E3"]


def test_topup_prompt_and_cache_key_ignore_finish_order(monkeypatch):
    # Two finish orders of the same searches, and two orders of the same
    # pages, have to hash to one cache file. The key is the prompt.
    from misinfo_agent.investigate import TopupBatch, topup_prompt
    from misinfo_agent.llm import _cache_path

    monkeypatch.setattr(config, "LLM_CACHE", True)
    wiki = Passage(
        passage_id="P1",
        url="https://en.wikipedia.org/wiki/States_and_union_territories_of_India",
        netloc="en.wikipedia.org",
        title="States",
        text="India has 28 states.",
        cos=0.4,
        fetched=True,
    )
    book = Passage(
        passage_id="P2",
        url="https://testbook.com/question-answer/states",
        netloc="testbook.com",
        title="Quiz",
        text="28 states.",
        cos=0.3,
        fetched=True,
    )
    first = [
        {"stance": "refutes", "quote": "28 states and 8 union territories", "passage": wiki},
        {"stance": "neutral", "quote": "not evidence", "passage": wiki},
        {"stance": "refutes", "quote": "28 states in the Indian union", "passage": book},
    ]
    second = list(reversed(first))
    state = {"claim": "India has 29 states in total."}
    early = ["zeta query", "alpha query", "mid query"]
    late = ["mid query", "zeta query", "alpha query"]
    prompt_a = topup_prompt(state, first, early)
    prompt_b = topup_prompt(state, second, late)
    assert prompt_a == prompt_b
    assert "alpha query\nmid query\nzeta query" in prompt_a
    wiki_at = prompt_a.index("en.wikipedia.org")
    book_at = prompt_a.index("testbook.com")
    assert wiki_at < book_at
    assert "not evidence" not in prompt_a
    key_a = _cache_path("model", "topup_v1", prompt_a, TopupBatch, None)
    key_b = _cache_path("model", "topup_v1", prompt_b, TopupBatch, None)
    assert key_a == key_b
    assert key_a is not None


def test_recorded_queries_follow_issue_order_not_finish_order(monkeypatch):
    # The worker that finishes first must not become the first stored query.
    # That list is what the top-up prompt used to hash.
    import threading

    from misinfo_agent.investigate import Query, _run_queries

    gate = threading.Barrier(2)

    def fake_search(query, **kwargs):
        gate.wait(timeout=2)
        if query == "slow":
            time.sleep(0.05)
        return []

    monkeypatch.setattr("misinfo_agent.investigate.search", fake_search)
    ran: list[str] = []
    _run_queries(
        [Query(text="slow", kind="neutral"), Query(text="fast", kind="verify")],
        ran=ran,
    )
    assert ran == ["slow", "fast"]


def _body(score: str, quote: str) -> str:
    # One sentence, so the fake encoder sees a single window, and long
    # enough that the fetch keeps the Tavily text. The repeated quote is
    # different for each outlet, so MinHash does not merge them into one
    # group and hide the second source.
    unique = " ".join((quote.split()) * 14)
    return f"{score} {quote} {unique}"


def _search_hit(url: str, raw: str) -> dict:
    return {
        "url": url,
        "title": "Report",
        "content": "snippet",
        "score": 0.8,
        "published_date": "2024-06-01",
        "raw_content": raw,
    }


def _strong_in(evidence) -> int:
    return len({row["group_id"] for row in evidence if row["tier"] <= 2})


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


_LIVE_CLAIMS = [
    {
        "label": "a",
        "claim": "₹500 notes printed before 2016 are invalid from 1 December",
        "domain": "Finance & scams",
        "entities": {
            "amounts": ["₹500"],
            "years": ["2016"],
            "dates": ["1 December"],
            "names": [],
            "places": [],
        },
        "match_type": "none",
        "leads": [],
    },
    {
        "label": "b",
        "claim": "Drinking hot water with lemon kills coronavirus",
        "domain": "Health",
        "entities": {
            "amounts": [],
            "years": [],
            "dates": [],
            "names": [],
            "places": [],
        },
        "match_type": "none",
        "leads": [],
    },
    {
        "label": "c",
        "claim": "Video shows flood in Chennai",
        "domain": "Disasters & accidents",
        "entities": {
            "amounts": [],
            "years": [],
            "dates": [],
            "names": [],
            "places": ["Chennai"],
        },
        "match_type": "none",
        "leads": [],
    },
]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _sample_archive_claims() -> list[dict]:
    """Five fact-checks, seed 42, five domains, dated 2025 or 2026.

    The content column is not read. primary_claim is the claim, and the
    article date is the claim date the leakage guard will use.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "classified_v2.parquet"
    frame = pd.read_parquet(
        path,
        columns=["article_id", "date", "year", "is_fact_check", "form", "domain", "primary_claim"],
    )
    pool = frame[
        (frame["is_fact_check"] == True)  # noqa: E712 — parquet bool, not identity
        & (frame["year"].isin([2025, 2026]))
        & (~frame["form"].isin(["Image", "Video"]))
        & (frame["primary_claim"].fillna("").str.strip() != "")
        & (frame["date"].fillna("").str.len() >= 10)
    ]
    rng = np.random.default_rng(42)
    domains = sorted(pool["domain"].dropna().unique().tolist())
    picked = list(rng.choice(domains, size=5, replace=False))
    rows = []
    for domain in picked:
        part = pool[pool["domain"] == domain]
        choice = part.iloc[int(rng.integers(0, len(part)))]
        rows.append(
            {
                "claim_id": str(choice["article_id"]),
                "claim": str(choice["primary_claim"]).strip(),
                "domain": str(choice["domain"]),
                "claim_date": str(choice["date"])[:10],
                "drop_after": True,
            }
        )
    return rows


def _page_tally(pages) -> dict:
    reasons: dict[str, int] = {}
    text_from: dict[str, int] = {}
    counts = {"ok": 0, "failed": 0, "snippet": 0, "circulation": 0, "non_english": 0, "pdf": 0}
    trusted = 0
    for page in pages:
        status = page.fetch_status
        if status in counts:
            counts[status] += 1
        if page.is_pdf and status == "ok":
            counts["pdf"] += 1
        if status == "failed":
            reason = page.failure_reason or "failed"
            reasons[reason] = reasons.get(reason, 0) + 1
        if status == "ok":
            text_from[page.body_from] = text_from.get(page.body_from, 0) + 1
        if page.published_date:
            trusted += 1
    counts["trusted_dates"] = trusted
    counts["failure_reasons"] = reasons
    counts["text_from"] = text_from
    return counts


@pytest.mark.live
def test_live_robustness_eight_claims(monkeypatch):
    """Eight claims, fact-check sites blocked. One real run.

    The three fixed claims use today's date and keep later pages. The five
    archive claims use the fact-check date and drop later pages. Box 5 and
    box 6 run on all eight. A similar prior adds the lead query. Fixed
    claims are searched with no date_before, so today's memory is visible.
    """
    import csv
    from datetime import date
    from pathlib import Path

    from misinfo_agent.investigate import gather_passages
    from misinfo_agent.memory import ClaimMemory
    from misinfo_agent.prior_match import decide_prior
    from misinfo_agent.search import factcheck_blocked, registrable_domain, reset_claim_searches, site_host

    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    monkeypatch.setattr(config, "EVAL_UNDATED", "keep")
    before = searches_this_month()
    print("SEARCHES THIS MONTH BEFORE", before)
    today = date.today().isoformat()
    memory = ClaimMemory.load()
    jobs = []
    for spec in _LIVE_CLAIMS:
        # No date_before: a similar prior from any day can add the lead query.
        results = memory.search(spec["claim"], k=10)
        decision = decide_prior(spec["claim"], results, entities=spec["entities"])
        jobs.append(
            {
                "claim_id": spec["label"],
                "claim": spec["claim"],
                "domain": spec["domain"],
                "entities": spec["entities"],
                "claim_date": today,
                "drop_after": False,
                "match_type": decision["match_type"],
                "leads": decision["leads"],
            }
        )
    for spec in _sample_archive_claims():
        results = memory.search(spec["claim"], k=10, date_before=spec["claim_date"])
        decision = decide_prior(spec["claim"], results)
        spec["entities"] = {}
        spec["match_type"] = decision["match_type"]
        spec["leads"] = decision["leads"]
        jobs.append(spec)

    report = []
    rows_out = []
    total = 0
    for spec in jobs:
        monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", spec["drop_after"])
        state = {
            "claim": spec["claim"],
            "domain": spec["domain"],
            "entities": spec["entities"],
            "claim_date": spec["claim_date"],
            "match_type": spec["match_type"],
            "leads": spec["leads"],
        }
        print(f"START {spec['claim_id']}", flush=True)
        reset_claim_searches()
        reset_search_usage()
        queries = build_queries(state)
        assert len(queries) <= 5
        start = searches_this_month()
        bundle = gather_passages(state)
        used = searches_this_month() - start
        total += used
        lead = next((query.text for query in queries if query.kind == "lead"), None)
        dated_kept = sum(1 for passage in bundle["passages"] if passage.published_date)
        print(
            f"DONE {spec['claim_id']} searches={used} seconds={bundle['seconds']:.1f} "
            f"lead={lead or '-'} dated={dated_kept}/{len(bundle['passages'])} "
            f"dropped_after={bundle['dropped_after_claim']} "
            f"dropped_undated={bundle['dropped_undated']} "
            f"keep={bundle['passages_if_keep']} drop={bundle['passages_if_drop']}",
            flush=True,
        )
        for passage in bundle["passages"][:5]:
            print(
                f"TOP {spec['claim_id']} {passage.netloc} "
                f"score={passage.score:.3f} cos={passage.cos:.3f} "
                f"date={passage.published_date or '-'} source={passage.date_source or '-'} "
                f"words={' '.join(passage.text.split()[:25])}",
                flush=True,
            )
        hits = bundle["hits"]
        pages = bundle["pages"]
        passages = bundle["passages"]
        assert used <= 5
        assert len(hits) <= 25
        assert len(passages) <= 40
        for hit in hits:
            assert factcheck_blocked(hit.url, hit.title) is False
        per_domain: dict[str, int] = {}
        for passage in passages:
            host = registrable_domain(site_host(passage.url) or passage.url)
            per_domain[host] = per_domain.get(host, 0) + 1
            assert passage.cos >= config.PASSAGE_MIN_COS
            assert host not in config.NON_EVIDENCE_HOSTS
            assert factcheck_blocked(passage.url, passage.title) is False
            folded = passage.text.casefold()
            assert "pib fact check" not in folded
            assert "fact check" not in folded
            assert "fact-check" not in folded
        assert all(count <= 2 for count in per_domain.values())
        by_kind: dict[str, int] = {}
        for hit in hits:
            for kind in hit.query_kinds:
                by_kind[kind] = by_kind.get(kind, 0) + 1
        tally = _page_tally(pages)
        report.append(
            {
                "claim_id": spec["claim_id"],
                "claim": spec["claim"],
                "domain": spec["domain"],
                "claim_date": spec["claim_date"],
                "drop_after": spec["drop_after"],
                "match_type": spec["match_type"],
                "queries": [
                    {"text": query.text, "kind": query.kind, "lead": query.kind == "lead"}
                    for query in queries
                ],
                "searches_used": used,
                "hits_by_kind": by_kind,
                "unique_hits": len(hits),
                "circulation_sightings": tally["circulation"],
                "fetched_ok": tally["ok"],
                "fetched_failed": tally["failed"],
                "failure_reasons": tally["failure_reasons"],
                "snippet_only": tally["snippet"],
                "non_english": tally["non_english"],
                "pdf": tally["pdf"],
                "text_from": tally["text_from"],
                "trusted_dates": tally["trusted_dates"],
                "lead_query": lead,
                "dropped_after_claim": bundle["dropped_after_claim"],
                "dropped_undated": bundle["dropped_undated"],
                "passages_kept": len(passages),
                "passages_if_keep": bundle["passages_if_keep"],
                "passages_if_drop": bundle["passages_if_drop"],
                "kept_with_trusted_date": dated_kept,
                "passage_domains": len(per_domain),
                "seconds": round(bundle["seconds"], 2),
                "top_5": [
                    {
                        "netloc": passage.netloc,
                        "score": round(passage.score, 3),
                        "cos": round(passage.cos, 3),
                        "published_date": passage.published_date,
                        "date_source": passage.date_source,
                        "words": " ".join(passage.text.split()[:25]),
                    }
                    for passage in passages[:5]
                ],
            }
        )
        for rank, passage in enumerate(passages[:5], start=1):
            rows_out.append(
                {
                    "claim_id": spec["claim_id"],
                    "claim": spec["claim"],
                    "rank": rank,
                    "netloc": passage.netloc,
                    "score": round(passage.score, 4),
                    "cos": round(passage.cos, 4),
                    "published_date": passage.published_date or "",
                    "date_source": passage.date_source or "",
                    "passage_text": passage.text,
                    "relevant": "",
                }
            )
    assert total <= 40
    attempts = sum(item["fetched_ok"] + item["fetched_failed"] + item["non_english"] for item in report)
    ok = sum(item["fetched_ok"] for item in report)
    kept_n = sum(item["passages_kept"] for item in report)
    dated_kept = sum(item["kept_with_trusted_date"] for item in report)
    times = [item["seconds"] for item in report]
    overall = {
        "searches_this_month_before": before,
        "searches_this_month_after": searches_this_month(),
        "searches_used": total,
        "pct_kept_passages_with_trusted_date": (dated_kept / kept_n) if kept_n else None,
        "kept_passages": kept_n,
        "kept_with_trusted_date": dated_kept,
        "seconds_median": _median(times),
        "seconds_max": max(times) if times else None,
        "fetch_success_rate": (ok / attempts) if attempts else None,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "day2_passages.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id", "claim", "rank", "netloc", "score", "cos",
                "published_date", "date_source", "passage_text", "relevant",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    report_path = out.with_name("day2_report.json")
    report_path.write_text(
        json.dumps({"claims": report, "overall": overall}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("LIVE REPORT")
    print(json.dumps(overall, ensure_ascii=False))
    print("WROTE", out)


# Leads box 6 returned on 12 Nov 2026. Hardcoded so this run does not spend
# a second Gemini call on the exact-match veto. The lead line is what the
# search cache was written under.
_DAY3_LEADS = {
    "b": "Inhaling hot water steam can kill coronavirus.",
    "c": "Videos show recent flooding in Chennai from heavy rains.",
}


@pytest.mark.live
def test_live_day4_eight_claims(monkeypatch):
    """Same eight claims. Cached plan searches are free. Stance is v2, plus one top-up when the tier 1–2 groups are thin."""
    import csv
    from datetime import date
    from pathlib import Path

    from misinfo_agent.llm import calls_today

    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", True)
    monkeypatch.setattr(config, "EVAL_UNDATED", "keep")
    before_calls = calls_today()
    before_searches = searches_this_month()
    print("CALLS TODAY BEFORE", before_calls)
    print("SEARCHES THIS MONTH BEFORE", before_searches)
    today = date.today().isoformat()
    jobs = []
    for spec in _LIVE_CLAIMS:
        lead = _DAY3_LEADS.get(spec["label"])
        jobs.append({
            "claim_id": spec["label"],
            "claim": spec["claim"],
            "domain": spec["domain"],
            "entities": spec["entities"],
            "atomic_claims": [spec["claim"]],
            "claim_date": today,
            "drop_after": False,
            "match_type": "similar" if lead else "none",
            "leads": [{"primary_claim": lead}] if lead else [],
        })
    for spec in _sample_archive_claims():
        spec["entities"] = {}
        spec["atomic_claims"] = [spec["claim"]]
        spec["match_type"] = "none"
        spec["leads"] = []
        jobs.append(spec)

    rows_out = []
    report = []
    for spec in jobs:
        monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", spec["drop_after"])
        state = {
            "claim": spec["claim"],
            "domain": spec["domain"],
            "entities": spec["entities"],
            "atomic_claims": spec["atomic_claims"],
            "claim_date": spec["claim_date"],
            "match_type": spec["match_type"],
            "leads": spec["leads"],
        }
        print(f"START {spec['claim_id']}", flush=True)
        result = investigate(state)
        stats = result["investigate_stats"]
        evidence = result["evidence"]
        print(
            f"DONE {spec['claim_id']} seconds={stats['seconds']} "
            f"searches={stats['searches']} llm={stats['llm_calls']} "
            f"passages={stats['n_passages']} supports={stats['n_supports']} "
            f"refutes={stats['n_refutes']} neutral={stats['n_neutral']} "
            f"groups_supporting={stats['groups_supporting']} "
            f"groups_refuting={stats['groups_refuting']} "
            f"unverified={stats['quote_unverified']} "
            f"sightings={stats['circulation_sightings']} notes={stats['notes']}",
            flush=True,
        )
        for row in evidence:
            quote = " ".join(str(row["quote"]).split()[:25])
            print(
                f"EVID {spec['claim_id']} {row['id']} {row['netloc']} "
                f"tier={row['tier']} group={row['group_id']} stance={row['stance']} "
                f"date={row['published_date'] or '-'} snippet={row['snippet_only']} "
                f"quote={quote}",
                flush=True,
            )
            assert row["stance"] in {"supports", "refutes"}
            assert "factcheckhub.com" not in row["url"]
            if spec["claim_id"] == "008d3e713614" and str(row["netloc"]).endswith("dw.com"):
                # A 2018 article does not confirm a claim that says "now".
                if str(row.get("published_date") or "").startswith("2018"):
                    assert row["stance"] != "supports"
        strong_support = len({row["group_id"] for row in evidence if row["tier"] <= 2 and row["stance"] == "supports"})
        strong_refute = len({row["group_id"] for row in evidence if row["tier"] <= 2 and row["stance"] == "refutes"})
        by_domain: dict[str, set[str]] = {}
        for row in evidence:
            by_domain.setdefault(registrable_domain(row["netloc"]), set()).add(row["group_id"])
        # One newsroom is one group. A top-up must not mint a second id for it.
        assert all(len(groups) == 1 for groups in by_domain.values()), by_domain
        if spec["claim_id"] == "a":
            assert any(
                str(row["netloc"]).endswith("rbi.org.in")
                and row["stance"] == "refutes"
                and "2016" in row["quote"]
                for row in evidence
            )
        if spec["claim_id"] == "008d3e713614":
            assert strong_support == 0
        print(
            f"TOPUP {spec['claim_id']} used={stats['topup_used']} "
            f"queries={stats['topup_queries']} strong_support={strong_support} "
            f"strong_refute={strong_refute} tokens={stats['prompt_tokens']}/{stats['output_tokens']}",
            flush=True,
        )
        assert stats["llm_calls"] <= 3
        assert stats["n_supports"] + stats["n_refutes"] + stats["n_neutral"] == stats["n_passages"] or stats["notes"]
        report.append({
        "claim_id": spec["claim_id"],
        "claim": spec["claim"],
        "stats": stats,
        "strong_supporting": strong_support,
        "strong_refuting": strong_refute,
            "evidence": [
                {
                    "id": row["id"],
                    "netloc": row["netloc"],
                    "tier": row["tier"],
                    "group_id": row["group_id"],
                    "stance": row["stance"],
                    "quote": row["quote"],
                    "published_date": row["published_date"],
                    "age_days": row["age_days"],
                    "snippet_only": row["snippet_only"],
                }
                for row in evidence
            ],
        })
        for row in evidence[:5]:
            rows_out.append({
                "claim_id": spec["claim_id"],
                "claim": spec["claim"],
                "evidence_id": row["id"],
                "netloc": row["netloc"],
                "tier": row["tier"],
                "stance": row["stance"],
                "quote": row["quote"],
                "passage_text": row["passage"],
                "stance_correct": "",
            })
    after_calls = calls_today()
    after_searches = searches_this_month()
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "day3_evidence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id", "claim", "evidence_id", "netloc", "tier", "stance",
                "quote", "passage_text", "stance_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        "calls_today_before": before_calls,
        "calls_today_after": after_calls,
        "llm_calls_delta": after_calls - before_calls,
        "searches_before": before_searches,
        "searches_after": after_searches,
        "searches_delta": after_searches - before_searches,
    }
    out.with_name("day4_report.json").write_text(
        json.dumps({"summary": summary, "claims": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("DAY4", json.dumps(summary, ensure_ascii=False))
    print("COST claim searches llm prompt output seconds")
    search_costs = []
    second_costs = []
    for item in report:
        stats = item["stats"]
        search_costs.append(stats["searches"])
        second_costs.append(stats["seconds"])
        print(
            f"COST {item['claim_id']} {stats['searches']} {stats['llm_calls']} "
            f"{stats['prompt_tokens']} {stats['output_tokens']} {stats['seconds']}"
        )
    print(
        "COST median/max searches",
        _median(search_costs),
        max(search_costs) if search_costs else 0,
        "seconds",
        _median(second_costs),
        max(second_costs) if second_costs else 0,
    )
    print("CALLS TODAY", before_calls, after_calls)
    print("SEARCHES THIS MONTH", before_searches, after_searches)
    print("WROTE", out)
    # Eight claims, at most a stance call, a top-up, and a second stance call.
    assert summary["llm_calls_delta"] <= 40
    assert summary["searches_delta"] <= 40
