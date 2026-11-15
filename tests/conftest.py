"""Keep the default test run off the network.

A key in .env must not turn `pytest` into a Gemini or Tavily session.
Unless a test is marked live, both keys are removed, neither gateway is
allowed to read .env back, and building either real client raises.
Live tests run only with `pytest -m live`.
"""

import pytest


@pytest.fixture(autouse=True)
def _offline_llm(request, monkeypatch, tmp_path):
    if request.node.get_closest_marker("live") is not None:
        yield
        return
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # load_dotenv would put the key back. A no-op here is what makes the
    # "no key" tests pass while .env still holds a real key.
    monkeypatch.setattr("misinfo_agent.llm._load_env", lambda: None)

    def _refuse_client(*args, **kwargs):
        raise RuntimeError("offline test tried to create a Gemini client")

    monkeypatch.setattr("misinfo_agent.llm._client", _refuse_client)
    # The response cache is on in config. An offline test must not reuse a
    # verdict written by a live run, or the fake veto would never be asked.
    monkeypatch.setattr("misinfo_agent.config.LLM_CACHE", False)
    # The daily counter is a real file. Offline tests must not add to it
    # or trip the cap that the live run is using.
    monkeypatch.setattr(
        "misinfo_agent.llm.DAILY_COUNT_PATH",
        tmp_path / "llm_daily_count.json",
    )
    from misinfo_agent.llm import _RECENT_REAL_CALLS

    _RECENT_REAL_CALLS.clear()

    # Same wall around Tavily. search.py would otherwise load TAVILY_API_KEY
    # from .env on the first search() and spend the monthly budget.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr("misinfo_agent.search._load_env", lambda: None)

    def _refuse_tavily(*args, **kwargs):
        raise RuntimeError("offline test tried to create a Tavily client")

    monkeypatch.setattr("misinfo_agent.search._client", _refuse_tavily)
    monkeypatch.setattr("misinfo_agent.config.SEARCH_CACHE", False)
    monkeypatch.setattr("misinfo_agent.config.SEARCH_CACHE_DIR", tmp_path / "search")
    # The monthly counter is a real file. Offline tests must not add to it
    # or trip the cap the live run is using.
    monkeypatch.setattr(
        "misinfo_agent.search.MONTHLY_COUNT_PATH",
        tmp_path / "search_monthly_count.json",
    )
    from misinfo_agent.search import (
        clear_fake_search,
        reset_claim_searches,
        reset_search_usage,
    )

    clear_fake_search()
    reset_claim_searches()
    reset_search_usage()
    # Page fetches write under data/ and open sockets. Offline tests get
    # a temp cache and a client that refuses, the same wall as Tavily.
    monkeypatch.setattr("misinfo_agent.config.PAGE_CACHE_DIR", tmp_path / "pages")

    def _refuse_fetch(*args, **kwargs):
        raise RuntimeError("offline test tried to fetch a page")

    monkeypatch.setattr("misinfo_agent.fetch._http_get", _refuse_fetch)
    yield
    clear_fake_search()
    reset_claim_searches()
    reset_search_usage()
