"""Keep the default test run off the network.

A key in .env must not turn `pytest` into a Gemini or Tavily session.
Unless a test is marked live, both keys are removed, neither gateway is
allowed to read .env back, and building either real client raises.
Live tests run only with `pytest -m live`.
"""

import pytest


@pytest.fixture(autouse=True)
def _offline_llm(request, monkeypatch, tmp_path):
    # ingest() reads this at call time. Every test, live or not, gets a
    # fresh file. A test that forgets db_path still must not create
    # data/submissions.db, because that file is the real reach counter.
    monkeypatch.setattr(
        "misinfo_agent.config.SUBMISSIONS_DB",
        tmp_path / "submissions.db",
    )
    # The downscaled JPEG. Same reason as the submissions file: a test
    # must not leave screenshots under data/uploads.
    monkeypatch.setattr(
        "misinfo_agent.config.UPLOADS_DIR",
        tmp_path / "uploads",
    )
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
    # Wayback uses that same client. Replacing existed_by is what keeps
    # a provenance test off DNS: the real lookup checks the target
    # address before it reads. The cache is a temp dir so a live capture
    # cannot date a fixture. _read_public is the second wall, for a test
    # that puts the real lookup back and forgets to stub the client.
    monkeypatch.setattr("misinfo_agent.wayback.CACHE_DIR", tmp_path / "wayback")
    monkeypatch.setattr("misinfo_agent.wayback.existed_by", lambda url: None)

    def _refuse_wayback(*args, **kwargs):
        raise RuntimeError("offline test tried to query the wayback")

    monkeypatch.setattr("misinfo_agent.wayback._read_public", _refuse_wayback)
    # Vision is the third gateway. A key in .env must not turn pytest
    # into a web-detection session, and the monthly file is the real bill.
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
    monkeypatch.setattr("misinfo_agent.vision._load_env", lambda: None)

    def _refuse_vision(*args, **kwargs):
        raise RuntimeError("offline test tried to call Vision")

    monkeypatch.setattr("misinfo_agent.vision._post", _refuse_vision)
    monkeypatch.setattr("misinfo_agent.config.VISION_CACHE", True)
    monkeypatch.setattr("misinfo_agent.config.VISION_CACHE_DIR", tmp_path / "vision")
    monkeypatch.setattr(
        "misinfo_agent.vision.MONTHLY_COUNT_PATH",
        tmp_path / "vision_monthly_count.json",
    )
    monkeypatch.setattr(
        "misinfo_agent.config.IMAGE_MEMORY_DB",
        tmp_path / "image_memory.db",
    )
    from misinfo_agent.vision import clear_fake

    clear_fake()
    yield
    clear_fake_search()
    reset_claim_searches()
    reset_search_usage()
    clear_fake()
