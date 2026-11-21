"""The eval harness, offline.

The sampler, the score arithmetic, and the way a system row is recorded.
Nothing here opens the parquet, the memory index, or a socket.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import misinfo_agent.config as config
from misinfo_agent.llm import LLMQuotaError

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"


def _load(name: str):
    # scripts/ is not a package. Loading by path keeps pytest from needing
    # an __init__ that would make the scripts importable as a library.
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_testset = _load("build_testset")
run_eval = _load("run_eval")
score_eval = _load("score_eval")


def _article(article_id, *, domain, verdict="False", form="Text", year="2025", fact=True, claim=None, true=None):
    return {
        "article_id": article_id,
        "date": f"{year}-06-01",
        "is_fact_check": fact,
        "form": form,
        "domain": domain,
        "verdict_norm": verdict,
        "primary_claim": claim if claim is not None else f"Claim {article_id} circulated widely",
        "true_version": true if true is not None else f"What was actually true about {article_id}",
    }


def _toy():
    rows = []
    for index in range(5):
        rows.append(_article(f"a{index}", domain="Health"))
    for index in range(3):
        rows.append(_article(f"b{index}", domain="Sports", verdict="Misleading"))
    rows.append(_article("c0", domain="Crime & law", verdict="True"))
    rows.append(_article("img", domain="Health", form="Image"))
    rows.append(_article("vid", domain="Health", form="Video"))
    rows.append(_article("old", domain="Health", year="2024"))
    rows.append(_article("blank", domain="Health", verdict="Blank"))
    rows.append(_article("audio", domain="Health", form="Audio"))
    rows.append(_article("notfc", domain="Health", fact=False))
    return rows


def test_map_label_keeps_the_three_scored_verdicts():
    assert build_testset.map_label("False") == "False"
    assert build_testset.map_label(" misleading ") == "Misleading"
    assert build_testset.map_label("True") == "True"
    assert build_testset.map_label("Blank") is None
    assert build_testset.map_label("Other") is None
    assert build_testset.map_label(None) is None


def test_quotas_are_even_and_do_not_overdraw_a_small_domain():
    sizes = {f"D{index:02d}": 100 for index in range(11)}
    assigned = build_testset.quotas(sizes, 40)
    assert sum(assigned.values()) == 40
    assert max(assigned.values()) - min(assigned.values()) <= 1

    # Two rows cannot supply a third slot. The leftover goes to a domain
    # that still has rows, and a tie breaks by name.
    short = build_testset.quotas({"Disasters": 2, "Health": 10, "Sports": 10}, 8)
    assert short == {"Disasters": 2, "Health": 3, "Sports": 3}


def test_select_claims_spreads_domains_and_keeps_trues_separate():
    first = build_testset.select_claims(_toy(), seed=42, n_labelled=4, n_true=2)
    second = build_testset.select_claims(_toy(), seed=42, n_labelled=4, n_true=2)
    assert first == second

    labelled = [row for row in first if row["source"] == "primary_claim"]
    trues = [row for row in first if row["source"] == "true_version"]
    assert len(labelled) == 4
    assert len(trues) == 2
    assert {row["article_id"] for row in labelled} & {row["article_id"] for row in trues} == set()

    by_domain = {}
    for row in labelled:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
    assert by_domain == {"Health": 2, "Sports": 1, "Crime & law": 1}

    # Blank is not a scored archive label, so it is not one of the 40.
    # It can still donate a true_version. Image, video, audio, a 2024
    # row, and a non-fact-check cannot.
    assert "blank" not in {row["article_id"] for row in labelled}
    banned = {"img", "vid", "old", "audio", "notfc"}
    assert banned.isdisjoint(row["article_id"] for row in first)
    assert all(row["label"] in {"False", "Misleading", "True"} for row in labelled)
    assert all(row["label"] == "True" for row in trues)
    assert all(row["input"].startswith("Claim ") for row in labelled)
    assert all(row["input"].startswith("What was actually true") for row in trues)
    assert all(row["claim_date"] == "2025-06-01" for row in first)
    # Round-robin, so the file does not open as four rows from one domain.
    assert labelled[0]["domain"] != labelled[1]["domain"]


def test_near_copies_are_one_vector_and_include_the_article():
    ids = np.array(["a", "b", "c"])
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert set(run_eval.near_copy_ids(ids, matrix, "a", 0.95)) == {"a", "b"}
    assert run_eval.near_copy_ids(ids, matrix, "missing", 0.95) == ["missing"]


def test_honesty_flags():
    # The process-wide flags. Restored by monkeypatch so a later test
    # still sees the defaults this module imported.
    previous = (
        config.BLOCK_FACTCHECK_SITES,
        config.EVAL_DROP_AFTER_CLAIM,
        config.RECORD_SUBMISSIONS,
        config.LLM_CACHE,
    )
    config.BLOCK_FACTCHECK_SITES = False
    config.EVAL_DROP_AFTER_CLAIM = False
    config.RECORD_SUBMISSIONS = True
    config.LLM_CACHE = False
    try:
        run_eval.apply_honesty()
        assert config.BLOCK_FACTCHECK_SITES is True
        assert config.EVAL_DROP_AFTER_CLAIM is True
        assert config.RECORD_SUBMISSIONS is False
        assert config.LLM_CACHE is True
    finally:
        (
            config.BLOCK_FACTCHECK_SITES,
            config.EVAL_DROP_AFTER_CLAIM,
            config.RECORD_SUBMISSIONS,
            config.LLM_CACHE,
        ) = previous


def test_eval_claim_date_overrides_today():
    from misinfo_agent.graph import _ingest

    text = "The minister visited the city museum yesterday"
    out = _ingest({
        "raw_input": text,
        "input_type": "text",
        "caption": "",
        "run_marks": {"eval_claim_date": "2025-03-01"},
    })
    assert out["claim_date"] == "2025-03-01"

    plain = _ingest({"raw_input": text, "input_type": "text", "caption": "", "run_marks": {}})
    assert plain["claim_date"] == date.today().isoformat()


def test_memory_search_excludes_eval_ids(monkeypatch):
    seen = {}

    class _Mem:
        def search(self, claim, **kwargs):
            seen["claim"] = claim
            seen.update(kwargs)
            return []

    monkeypatch.setattr("misinfo_agent.memory.get_memory", lambda: _Mem())
    from misinfo_agent.graph import _memory

    _memory({
        "claim": "a claim about a minister visit",
        "domain": "Health",
        "claim_date": "2025-04-01",
        "run_marks": {"eval_exclude_ids": ["art-1", "art-2"]},
    })
    assert seen["date_before"] == "2025-04-01"
    assert seen["exclude_ids"] == ["art-1", "art-2"]
    assert seen["domains"] == ["Health"]


def test_mismatch_counts_add_across_rounds():
    from misinfo_agent.graph import _merge_stats

    merged = _merge_stats(
        {"reason_label_mismatch": 2, "repaired": 1, "quote_unverified": 1},
        {"reason_label_mismatch": 1, "repaired": 3, "quote_unverified": 0},
    )
    assert merged["reason_label_mismatch"] == 3
    assert merged["repaired"] == 4
    assert merged["quote_unverified"] == 1


def test_resume_skips_and_quota_stops(tmp_path):
    rows = [
        {"article_id": "a", "input": "one", "label": "False", "domain": "Health", "claim_date": "2025-01-01"},
        {"article_id": "b", "input": "two", "label": "False", "domain": "Health", "claim_date": "2025-01-02"},
        {"article_id": "c", "input": "three", "label": "True", "domain": "Sports", "claim_date": "2025-01-03"},
    ]
    path = tmp_path / "results_system.jsonl"
    calls = []

    def boom(row):
        calls.append(row["article_id"])
        if row["article_id"] == "b":
            raise LLMQuotaError("eval", "daily cap")
        return {"article_id": row["article_id"], "verdict": "False", "needs_human": False,
                "confidence": 0.5, "citations": [], "evidence": [], "prior_present": False,
                "llm_calls": 3, "searches": 4, "latency": 1.0,
                "reason_label_mismatch": 0, "repaired": 0}

    with pytest.raises(LLMQuotaError):
        run_eval.run_rows(rows, path, boom)
    assert calls == ["a", "b"]
    saved = score_eval.load_jsonl(path)
    assert [row["article_id"] for row in saved] == ["a"]

    def finish(row):
        calls.append("again-" + row["article_id"])
        return {"article_id": row["article_id"], "verdict": "True", "needs_human": False,
                "confidence": 0.2, "citations": [], "evidence": [], "prior_present": False,
                "llm_calls": 1, "searches": 0, "latency": 0.2,
                "reason_label_mismatch": 0, "repaired": 0}

    run_eval.run_rows(rows, path, finish)
    assert "again-a" not in calls
    assert calls.count("again-b") == 1
    assert "again-c" in calls


def test_interrupt_is_recorded_as_needs_human(monkeypatch):
    monkeypatch.setattr(run_eval, "near_copy_ids", lambda *a, **k: ["art", "copy"])
    monkeypatch.setattr(run_eval, "checkpoint_values", lambda thread, paths: {
        "verdict": "False",
        "confidence": 0.42,
        "needs_human": False,
        "citations": ["E1", "C1"],
        "evidence": [{"id": "E1", "quote": "the page refutes it", "passage": "the page refutes it today"}],
        "best_match": {"article_id": "older", "url": "https://example.test/old"},
        "investigate_stats": {"reason_label_mismatch": 2, "repaired": 1},
    })
    seen = {}

    def fake_pipeline(*args, **kwargs):
        seen["raw"] = args[0]
        seen["kwargs"] = kwargs
        return {"status": "interrupt", "thread_id": kwargs["thread_id"], "review": {"verdict": "False"}}

    monkeypatch.setattr(run_eval, "run_pipeline", fake_pipeline)
    memory = type("Mem", (), {"ids": np.array(["art"]), "E": np.ones((1, 2), dtype=np.float32)})()
    row = run_eval.invoke_system(
        {"article_id": "art", "input": "the claim text", "claim_date": "2025-05-01"},
        memory,
        {"checkpoint_path": "c", "audit_path": "a", "additions_path": "p"},
    )
    assert seen["raw"] == "the claim text"
    assert seen["kwargs"]["eval_claim_date"] == "2025-05-01"
    assert seen["kwargs"]["eval_exclude_ids"] == ["art", "copy"]
    assert "resume" not in seen["kwargs"]
    # The rules had said False. Nobody accepted it, so it is an abstention.
    assert row["verdict"] == "False"
    assert row["needs_human"] is True
    assert row["reason_label_mismatch"] == 2
    assert row["repaired"] == 1
    assert row["prior_present"] is True
    assert row["citations"] == ["E1", "C1"]


def test_baseline_prompt_is_the_claim_only():
    prompt = run_eval.baseline_prompt("The city banned the note from Monday")
    assert "The city banned the note from Monday" in prompt
    assert "2025-05-01" not in prompt
    for label in ("False", "Misleading", "True", "Unverifiable"):
        assert label in prompt
    schema = run_eval.BaselineVerdict.model_json_schema()
    assert "verdict" in schema["properties"]
    assert "confidence" in schema["properties"]


def test_score_counts_abstention_and_citation_failures(tmp_path):
    testset = [
        {"article_id": "1", "input": "The city banned the note", "label": "False", "domain": "Health", "source": "primary_claim"},
        {"article_id": "2", "input": "The clinic doubled its beds", "label": "Misleading", "domain": "Health", "source": "primary_claim"},
        {"article_id": "3", "input": "The match was washed out", "label": "True", "domain": "Sports", "source": "true_version"},
        {"article_id": "4", "input": "The minister resigned that night", "label": "False", "domain": "Sports", "source": "primary_claim"},
    ]
    system = [
        {
            "article_id": "1", "verdict": "False", "confidence": 0.9, "needs_human": False,
            "citations": ["E1", "C1"],
            "evidence": [{"id": "E1", "quote": "the page refutes it", "passage": "Earlier, the page refutes it."}],
            "prior_present": True, "llm_calls": 4, "searches": 5, "latency": 10,
            "reason_label_mismatch": 0, "repaired": 0,
        },
        {
            "article_id": "2", "verdict": "False", "confidence": 0.2, "needs_human": False,
            "citations": ["E9"],
            "evidence": [{"id": "E1", "quote": "not this", "passage": "something else entirely"}],
            "prior_present": False, "llm_calls": 6, "searches": 7, "latency": 30,
            "reason_label_mismatch": 1, "repaired": 1,
        },
        {
            "article_id": "3", "verdict": "Unverifiable", "confidence": 0.4, "needs_human": False,
            "citations": [], "evidence": [], "prior_present": False,
            "llm_calls": 3, "searches": 5, "latency": 20,
            "reason_label_mismatch": 0, "repaired": 0,
            "gate": "insufficient",
            "gate_reason": "Two independent sources were not found.",
        },
        {
            # Provisional False, and it matches the gold. It is still an abstention.
            "article_id": "4", "verdict": "False", "confidence": 0.8, "needs_human": True,
            "citations": [], "evidence": [], "prior_present": False,
            "llm_calls": 4, "searches": 2, "latency": 15,
            "reason_label_mismatch": 0, "repaired": 0,
        },
    ]
    joined = score_eval._join(testset, system)
    metrics = score_eval.condition_metrics(joined)
    assert metrics["n"] == 4
    assert metrics["decided"] == 2
    assert metrics["correct"] == 1
    assert metrics["abstained"] == 2
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["abstention"] == pytest.approx(0.5)
    assert metrics["coverage_adjusted"] == pytest.approx(0.25)
    assert metrics["macro_f1"] == pytest.approx(2 / 9)
    assert metrics["ece_n"] == 2
    assert metrics["ece"] is not None
    assert metrics["citations"]["web_n"] == 2
    assert metrics["citations"]["web_exist"] == 1
    assert metrics["citations"]["web_valid"] == 1
    assert metrics["citations"]["archive_valid"] == 1

    # 0/1 is a boundary the plain normal interval would put below zero.
    low, high = score_eval.wilson(0, 1)
    assert 0.0 <= low <= high <= 1.0
    assert score_eval.wilson(1, 0) is None

    baseline = [
        {"article_id": "1", "verdict": "False", "confidence": 0.6, "needs_human": False,
         "citations": [], "evidence": [], "prior_present": False,
         "llm_calls": 1, "searches": 0, "latency": 1.0,
         "reason_label_mismatch": 0, "repaired": 0},
    ]
    text = score_eval.render(testset, system, baseline)
    # Misleading called False is wrong in 3-class and right once both are not-true.
    assert metrics["binary_correct"] == 2
    assert metrics["binary_macro_f1"] == pytest.approx(0.5)
    assert metrics["ece_bins"][2] == 1
    assert metrics["ece_bins"][9] == 1
    assert sum(metrics["ece_bins"]) == metrics["ece_n"]
    assert len(metrics["ece_bins"]) == 10

    assert "superseded — evaluation setup bug" in text
    assert "0.500 (1/2)" in text
    assert "Accuracy, 2-class" in text
    assert "1.000 (2/2)" in text
    assert "Abstention" in text
    assert "Coverage-adjusted accuracy" in text
    assert "Web citation validity" in text
    assert "| 2 | [0.2, 0.3) | 1 |" in text
    assert "What the system abstained on" in text
    assert "The match was washed out" in text
    assert "Two independent sources were not found." in text
    report = tmp_path / "eval_end_to_end.md"
    report.write_text(text, encoding="utf-8")
    assert "Confusion matrix" in report.read_text(encoding="utf-8")


def test_estimate_scales_the_finished_rows():
    system = [{"llm_calls": 4, "searches": 6, "latency": 40.0}]
    baseline = [{"llm_calls": 1, "searches": 0, "latency": 2.0}]
    text = run_eval.estimate(system, baseline, total=50)
    assert "200 LLM calls" in text
    assert "300 searches" in text
    assert "50 LLM calls" in text
    # 40s * 50 + 2s * 50 = 2100s = 35.0 min
    assert "35.0 min" in text


def test_meta_statement_is_not_used_as_a_true_row():
    assert build_testset.is_meta_statement("The viral video is misleading")
    assert build_testset.is_meta_statement("This claim is false")
    assert build_testset.is_meta_statement(
        "A 2023 court judgment quashed previous orders, and the old decision is being misrepresented."
    )
    assert not build_testset.is_meta_statement(
        "Armenia has not cancelled its defense agreements or weapon purchases from India."
    )
    # Blank verdicts are donors, not labelled rows. The meta donor is the
    # only other Health article, so the True row has to be the fact.
    records = [
        _article("gold", domain="Health", verdict="False"),
        _article("meta", domain="Health", verdict="Blank", true="The viral video is misleading"),
        _article("keep", domain="Health", verdict="Blank", true="The hospital still has 40 beds"),
    ]
    rows = build_testset.select_claims(records, n_labelled=1, n_true=1, seed=0)
    trues = [row for row in rows if row["source"] == "true_version"]
    assert [row["article_id"] for row in trues] == ["keep"]
    assert trues[0]["input"] == "The hospital still has 40 beds"


def test_leak_count_matches_ids_and_canonical_urls():
    # www, the tracking parameter, the trailing slash, and http vs https
    # are the same page. Two sightings of one article count once.
    same = run_eval.url_key("http://www.archive.test/story/?utm_source=x")
    assert same == run_eval.url_key("https://archive.test/story")
    state = {
        "matches": [{"article_id": "copy", "url": "https://www.alt.test/a?utm_source=x"}],
        "leads": [{"article_id": "other", "also_debunked_by": [{"url": "http://archive.test/story"}]}],
        "best_match": {"article_id": "copy", "url": "https://alt.test/a"},
        "evidence": [{"url": "https://archive.test/story/"}],
    }
    urls = {"src": same, "copy": run_eval.url_key("https://alt.test/a")}
    assert run_eval.count_leaks(state, ["src", "copy"], urls) == 2
    clean = {
        "matches": [{"article_id": "other", "url": "https://news.test/b"}],
        "leads": [],
        "evidence": [{"url": "https://news.test/b"}],
    }
    assert run_eval.count_leaks(clean, ["src", "copy"], urls) == 0


def test_allowed_sites_is_a_separate_results_file():
    # apply_honesty writes the other honesty flags too. Put them back,
    # or a later test sees the eval's date filter left on.
    previous = (
        config.BLOCK_FACTCHECK_SITES,
        config.EVAL_DROP_AFTER_CLAIM,
        config.RECORD_SUBMISSIONS,
        config.LLM_CACHE,
    )
    try:
        run_eval.apply_honesty("allowed")
        assert config.BLOCK_FACTCHECK_SITES is False
    finally:
        (
            config.BLOCK_FACTCHECK_SITES,
            config.EVAL_DROP_AFTER_CLAIM,
            config.RECORD_SUBMISSIONS,
            config.LLM_CACHE,
        ) = previous
    assert run_eval.system_results_path("blocked").name == "results_system_blocked.jsonl"
    assert run_eval.system_results_path("allowed").name == "results_system_allowed.jsonl"


def test_progress_prints_every_fifth_row(capsys, tmp_path):
    rows = [
        {"article_id": str(index), "input": "c", "label": "False", "domain": "Health"}
        for index in range(5)
    ]

    def invoke(row):
        return {"article_id": row["article_id"], "verdict": "False", "needs_human": False, "leak_count": 0}

    run_eval.run_rows(rows, tmp_path / "results.jsonl", invoke)
    assert "progress 5/5 calls_today=" in capsys.readouterr().out


def test_allowed_memory_keeps_other_factcheckers_and_drops_own_url(monkeypatch):
    import misinfo_agent.investigate as investigate
    from misinfo_agent.search import Hit, looks_like_factcheck

    # The block is off. The source article is still not a result. A
    # different fact-checker is.
    monkeypatch.setattr(config, "BLOCK_FACTCHECK_SITES", False)
    monkeypatch.setattr(config, "EVAL_DROP_AFTER_CLAIM", False)

    def fake(query, **kwargs):
        return [
            Hit(url="https://www.boomlive.in/own?utm_source=a", title="own", content="x", score=0.9, published_date=None, raw_content=""),
            Hit(url="https://www.altnews.in/other", title="other", content="y", score=0.5, published_date=None, raw_content=""),
            Hit(url="https://www.thehindu.com/news/story", title="news", content="z", score=0.4, published_date=None, raw_content=""),
        ]

    monkeypatch.setattr(investigate, "search", fake)
    state = {
        "claim": "The city banned the note from Monday onwards",
        "claim_date": "2025-01-01",
        "domain": "Health",
        "input_type": "text",
        "entities": {},
        "atomic_claims": [],
        "run_marks": {"eval_exclude_urls": ["http://boomlive.in/own"]},
    }
    hits = investigate.gather_candidates(state)
    urls = {hit.url for hit in hits}
    assert "https://boomlive.in/own" not in urls
    assert "https://altnews.in/other" in urls
    assert "https://thehindu.com/news/story" in urls
    assert looks_like_factcheck("https://altnews.in/other") is True
    assert config.BLOCK_FACTCHECK_SITES is False
    assert run_eval.memory_exclude_ids(type("M", (), {})(), "art", "allowed") == ["art"]


def test_wrong_verdict_root_cause_and_stricter_support():
    swap = {"label": "Misleading", "verdict": "False", "needs_human": False, "evidence": []}
    assert score_eval.root_cause(swap) == "wrong refute_kind"
    kernel = {
        "label": "False", "verdict": "True", "needs_human": False, "core_id": "A1",
        "evidence": [{"atomic_id": "A1", "stance": "supports", "tier": 2, "group_id": "G1", "group_conflict": False}],
    }
    assert score_eval.root_cause(kernel) == "true kernel read as support"
    gap = {
        "label": "True", "verdict": "False", "needs_human": False, "core_id": "A1",
        "evidence": [
            {"atomic_id": "A1", "stance": "refutes", "tier": 2, "group_id": "G1", "group_conflict": False},
            {"atomic_id": "A1", "stance": "refutes", "tier": 4, "group_id": "G2", "group_conflict": False},
        ],
    }
    assert score_eval.root_cause(gap) == "tier gap"
    # Two groups, only one of them credible: the live gate can say
    # supported, and the stricter bar does not.
    thin = {
        "label": "False", "verdict": "True", "needs_human": False, "core_id": "A1",
        "evidence": [
            {"atomic_id": "A1", "stance": "supports", "tier": 2, "group_id": "G1", "group_conflict": False},
            {"atomic_id": "A1", "stance": "supports", "tier": 4, "group_id": "G2", "group_conflict": False},
        ],
    }
    assert score_eval.stricter_support(thin) is False
    solid = {
        "label": "True", "verdict": "True", "needs_human": False, "core_id": "A1",
        "evidence": [
            {"atomic_id": "A1", "stance": "supports", "tier": 1, "group_id": "G1", "group_conflict": False},
            {"atomic_id": "A1", "stance": "supports", "tier": 2, "group_id": "G2", "group_conflict": False},
        ],
    }
    assert score_eval.stricter_support(solid) is True
    counts = score_eval.exploratory_counts([thin, solid])
    assert counts["removed"] == 1
    assert counts["lost"] == 0
    blocked = [
        {"article_id": "a", "label": "False", "verdict": "Unverifiable", "needs_human": False},
        {"article_id": "b", "label": "True", "verdict": "False", "needs_human": False},
    ]
    allowed = [
        {"article_id": "a", "label": "False", "verdict": "False", "needs_human": False},
        {"article_id": "b", "label": "True", "verdict": "True", "needs_human": True},
    ]
    assert score_eval.became_decided(blocked, allowed) == (1, 1)
