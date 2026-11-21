"""Run the end-to-end eval, one claim at a time. Stop on LLMQuotaError.

The system writes one of two files, depending on --sites:

  data/eval/results_system_blocked.jsonl   fact-check sites excluded (default)
  data/eval/results_system_allowed.jsonl   fact-check sites left in the search

The baseline is one Gemini call on the claim and is not repeated for the
other site setting:

  data/eval/results_baseline.jsonl

A claim already present in the file is skipped, so a stopped run can be
started again. A claim that dies on the daily cap is not written: the
next run retries it, and this process does not start another claim.

Usage:
  python scripts/run_eval.py                         # blocked system, then baseline
  python scripts/run_eval.py --sites allowed         # the less honest system number
  python scripts/run_eval.py --limit 3
  python scripts/run_eval.py --condition system
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.fetch import fresh_page_fetches, reset_fresh_page_fetches  # noqa: E402
from misinfo_agent.graph import build_graph, run_pipeline  # noqa: E402
from misinfo_agent.investigate import canonical_url, credibility_tier  # noqa: E402
from misinfo_agent.search import looks_like_factcheck, registrable_domain, searches_this_month  # noqa: E402
from misinfo_agent.llm import (  # noqa: E402
    LLMError,
    LLMQuotaError,
    call_structured,
    calls_today,
    get_usage,
)
from misinfo_agent.search import get_search_usage  # noqa: E402

# Full study size. --limit runs a prefix; the estimate still scales to this.
FULL_N = 50

# Bumped when the baseline wording changes. The cache key includes it,
# so an edited prompt cannot replay the previous answer.
_BASELINE_VERSION = "eval_baseline_v1"


class BaselineVerdict(BaseModel):
    """The baseline's whole answer.

    verdict is the label. confidence is the model's own probability for
    that label. The system never uses a self-reported probability. The
    baseline has no evidence features, so this number is the only
    confidence the calibration can bin, and it does not outvote the label.
    """

    verdict: Literal["False", "Misleading", "True", "Unverifiable"]
    confidence: float = Field(ge=0.0, le=1.0)


def apply_honesty(sites: str = "blocked") -> None:
    """The settings that make the system number a number it could have earned.

    Each one is off, or on for a different reason, in a demo. The eval
    turns them to the honest side and leaves them there for the process.
    sites is the fact-check-site condition: blocked is the number we
    report, allowed is the run that may still read those pages.
    """
    # Blocked: the fact-check being scored must not come back as a web
    # page. Allowed leaves the flag off for the process. investigate()
    # turns it on for the news search, then off again for a separate
    # fact-check budget, so those pages are added and the news pages
    # are not replaced. The default, and the run this report is built
    # from, is blocked.
    config.BLOCK_FACTCHECK_SITES = sites != "allowed"
    # A page published after the claim was circulating is not evidence
    # the system could have had on that day.
    config.EVAL_DROP_AFTER_CLAIM = True
    # Scoring a claim is not a person forwarding it. Box 11's reach
    # reads the submissions table, so a test run must not add a row.
    config.RECORD_SUBMISSIONS = False
    # A resumed process, and a repeated prompt inside one claim, must
    # not spend the daily cap again. The cache key includes the prompt,
    # the schema and the model, so an edited prompt is a miss.
    config.LLM_CACHE = True


def near_copy_ids(ids, matrix, article_id: str, threshold: float) -> list[str]:
    """This article, plus rows at least `threshold` similar to it.

    Search collapses those rows into also_debunked_by of whichever copy
    it keeps. They are the same fact-check from another outlet, and an
    older copy would survive date_before. One row of the embedding
    matrix is dotted with the matrix. That is the product search already
    does for a query. It is not an N×N matrix: the pairs are not stored.
    """
    target = str(article_id)
    id_list = np.asarray(ids).astype(str)
    matches = np.flatnonzero(id_list == target)
    if len(matches) == 0:
        return [target]
    row = int(matches[0])
    vector = np.asarray(matrix[row], dtype=np.float32)
    scores = np.asarray(matrix, dtype=np.float32) @ vector
    found = [str(id_list[int(index)]) for index in np.flatnonzero(scores >= float(threshold))]
    if target not in found:
        found.append(target)
    return found


def system_results_path(sites: str) -> Path:
    """Blocked and allowed stay in different files so a resume cannot mix them."""
    return config.EVAL_DIR / f"results_system_{sites}.jsonl"


def url_key(url: str) -> str:
    """Canonical URL, with the scheme removed.

    canonical_url already drops the www, the fragment, and tracking
    parameters. http and https are the same article, so the scheme is
    not part of the key the leak check compares.
    """
    text = canonical_url(url or "")
    if "://" in text:
        return text.split("://", 1)[1]
    return text


def archive_url_keys(memory, article_ids) -> dict[str, str]:
    """article_id -> canonical url key, for the ids memory actually has."""
    wanted = {str(item) for item in article_ids}
    if not wanted:
        return {}
    meta = getattr(memory, "meta", None)
    if meta is None:
        return {}
    found = {}
    ids = np.asarray(memory.ids).astype(str)
    for index, article_id in enumerate(ids):
        if article_id not in wanted:
            continue
        url = ""
        try:
            url = str(meta.iloc[int(index)]["url"] or "")
        except (KeyError, IndexError, TypeError):
            url = ""
        key = url_key(url)
        if key:
            found[article_id] = key
        if len(found) == len(wanted):
            break
    return found


def count_leaks(state: dict, banned_ids, url_of_id: dict[str, str]) -> int:
    """How many forbidden articles showed up in the run.

    The forbidden set is the claim's own article and the near-duplicate
    copies search would file under also_debunked_by. A hit counts whether
    it is a memory match, a prior-match lead, the prior box 6 kept, or an
    evidence URL. The same article in two of those places counts once.
    The expected total across the test set is 0: those ids were excluded
    from the search, and the date filter already drops the article's own day.
    """
    banned = {str(item) for item in banned_ids}
    id_of_url = {url: article_id for article_id, url in url_of_id.items() if url}
    found: set[str] = set()

    def note_url(url) -> None:
        key = url_key(url or "")
        owner = id_of_url.get(key)
        if owner:
            found.add(owner)

    def note_hit(row) -> None:
        if not isinstance(row, dict):
            return
        article_id = str(row.get("article_id") or "")
        if article_id and article_id in banned:
            found.add(article_id)
        note_url(row.get("url"))
        for extra in row.get("also_debunked_by") or []:
            if isinstance(extra, dict):
                note_url(extra.get("url"))
            elif isinstance(extra, str):
                note_url(extra)

    for key in ("matches", "leads"):
        for row in state.get(key) or []:
            note_hit(row)
    best = state.get("best_match")
    if isinstance(best, dict):
        note_hit(best)
    for row in state.get("evidence") or []:
        if isinstance(row, dict):
            note_url(row.get("url"))
    return len(found)


def baseline_prompt(claim: str) -> str:
    """The claim, and the four labels. No date, no article, no gold label."""
    return (
        "Judge this claim. You have the claim text and nothing else.\n"
        "\n"
        "Verdict, exactly one of:\n"
        "- False: the claim is not true\n"
        "- Misleading: a detail is wrong, or something left out changes what a reader would believe\n"
        "- True: the claim is accurate\n"
        "- Unverifiable: you cannot choose one of those three\n"
        "\n"
        "confidence is your probability that the verdict is right, from 0 to 1.\n"
        "\n"
        f"Claim:\n{claim}\n"
    )


def load_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_done(path: Path) -> set[str]:
    """Article ids already written. A half-written last line is ignored."""
    if not path.is_file():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("article_id"):
                done.add(str(row["article_id"]))
    return done


def append_jsonl(path: Path, row: dict) -> None:
    """One claim, flushed. A kill after this line is a claim the next run skips."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run_rows(rows: list[dict], results_path: Path, invoke) -> list[dict]:
    """invoke(row) -> result dict. LLMQuotaError is not caught here.

    The caller stops the process. Catching it and continuing would start
    the next claim, and that claim would fail for the same reason.
    """
    done = load_done(results_path)
    executed = []
    for index, row in enumerate(rows, start=1):
        article_id = str(row["article_id"])
        if article_id in done:
            print(f"skip {article_id}", flush=True)
        else:
            print(
                f"[{index}/{len(rows)}] {article_id} {row.get('label', '')} {row.get('domain', '')}",
                flush=True,
            )
            record = invoke(row)
            append_jsonl(results_path, record)
            done.add(article_id)
            executed.append(record)
            leak = int(record.get("leak_count") or 0)
            print(
                f"  verdict={record.get('verdict')} needs_human={record.get('needs_human')} "
                f"llm_calls={record.get('llm_calls')} searches={record.get('searches')} "
                f"latency_s={record.get('latency')} leak_count={leak}",
                flush=True,
            )
            if leak:
                # The row is kept. A leak is a failed guard, not a reason to
                # drop the claim or to start the next one under a different rule.
                print(f"  LEAK {article_id} leak_count={leak}", flush=True)
        # Every fifth test-set row, including one this file already had.
        # A quota error never reaches this line for the claim that raised it.
        if calls_today() > 300:
            # The rest of the daily budget is reserved for the demo.
            print(
                f"stopped: calls_today={calls_today()} is above 300 "
                f"tavily_month={searches_this_month()}",
                flush=True,
            )
            raise SystemExit(2)
        if index % 5 == 0:
            print(
                f"progress {index}/{len(rows)} calls_today={calls_today()} "
                f"tavily_month={searches_this_month()}",
                flush=True,
            )
    return executed


def memory_exclude_ids(memory, article_id: str, sites: str) -> list[str]:
    """Who memory search must not return.

    Blocked hides the source article and the near-duplicate copies another
    outlet published. Allowed hides only the source article. An earlier
    fact-check of the same rumour is what a deployment would retrieve,
    and the date filter still drops anything from the claim day onward.
    """
    if sites == "allowed":
        return [str(article_id)]
    return near_copy_ids(memory.ids, memory.E, article_id, config.DUP_COS)


def own_article_url(memory, article_id: str) -> str:
    """Canonical URL of this article in the archive. Empty if it has none."""
    meta = getattr(memory, "meta", None)
    if meta is None:
        return ""
    ids = np.asarray(memory.ids).astype(str)
    matches = np.flatnonzero(ids == str(article_id))
    if len(matches) == 0:
        return ""
    try:
        raw = str(meta.iloc[int(matches[0])]["url"] or "")
    except (KeyError, IndexError, TypeError):
        return ""
    return canonical_url(raw)


def invoke_system(row: dict, memory, paths: dict, sites: str = "blocked") -> dict:
    """One pipeline claim. A review pause is recorded, not answered.

    There is no Command(resume=...). The provisional verdict is what the
    rules had written when the graph parked. needs_human is set so the
    scorer counts the row as an abstention instead of grading a label
    nobody accepted.

    sites chooses the memory exclusion and the leak set. Both conditions
    drop the source article's own URL from search, fetch and evidence.
    """
    own_id = str(row["article_id"])
    exclude = memory_exclude_ids(memory, own_id, sites)
    # Allowed: a leak is this article alone. Another fact-checker's copy
    # is allowed evidence, and is counted separately. Blocked: a copy is
    # a leak, because that condition promised to hide them.
    leak_ids = [own_id] if sites == "allowed" else exclude
    banned_urls = archive_url_keys(memory, leak_ids)
    own_url = own_article_url(memory, own_id)
    if own_url:
        banned_urls[own_id] = url_key(own_url)
    before_llm = len(get_usage())
    before_search = len(get_search_usage())
    reset_fresh_page_fetches()
    started = time.perf_counter()
    # A new thread every attempt. Reusing an id would resume a pause
    # left behind by a crash, and nobody is sitting there to answer it.
    thread = f"eval-{row['article_id']}-{uuid.uuid4().hex[:8]}"
    out = run_pipeline(
        row["input"],
        input_type="text",
        thread_id=thread,
        checkpoint_path=paths["checkpoint_path"],
        audit_path=paths["audit_path"],
        additions_path=paths["additions_path"],
        eval_claim_date=row["claim_date"],
        eval_exclude_ids=exclude,
        eval_exclude_urls=[own_url] if own_url else None,
        eval_sites=sites,
    )
    latency = round(time.perf_counter() - started, 2)
    state = checkpoint_values(thread, paths)
    if out.get("status") == "interrupt":
        review = out.get("review") if isinstance(out.get("review"), dict) else {}
        if state.get("verdict") is None and review.get("verdict") is not None:
            state = dict(state)
            state["verdict"] = review.get("verdict")
            if state.get("confidence") is None:
                state["confidence"] = review.get("confidence")
    return record_from_state(
        state,
        article_id=str(row["article_id"]),
        interrupted=out.get("status") == "interrupt",
        llm_calls=len(get_usage()) - before_llm,
        searches=len(get_search_usage()) - before_search,
        latency=latency,
        leak_count=count_leaks(state, leak_ids, banned_urls),
        factcheck_evidence_used=factcheck_evidence_used(state, url_key(own_url)),
        credible_non_factcheck_groups=credible_non_factcheck_groups(state, url_key(own_url)),
        own_url=own_url,
    )
    fresh = fresh_page_fetches()
    if fresh:
        print("  fresh page fetches: " + ", ".join(fresh), flush=True)
    reset_fresh_page_fetches()


def invoke_baseline(row: dict) -> dict:
    """One structured call. The prompt is the claim and the four labels.

    A quota error propagates. Any other model failure is this claim's
    abstention: the day is not over, and the next claim is a different
    prompt. Searches stay zero because this condition does not search.
    """
    before_llm = len(get_usage())
    started = time.perf_counter()
    try:
        answer = call_structured(
            baseline_prompt(row["input"]),
            BaselineVerdict,
            purpose="eval_baseline",
            prompt_version=_BASELINE_VERSION,
        )
    except LLMQuotaError:
        raise
    except LLMError as exc:
        return _empty_result(
            str(row["article_id"]),
            llm_calls=len(get_usage()) - before_llm,
            latency=round(time.perf_counter() - started, 2),
            error=str(exc),
        )
    return {
        "article_id": str(row["article_id"]),
        "verdict": answer.verdict,
        "confidence": float(answer.confidence),
        "needs_human": False,
        "citations": [],
        "evidence": [],
        "prior_present": False,
        "llm_calls": len(get_usage()) - before_llm,
        "searches": 0,
        "latency": round(time.perf_counter() - started, 2),
        "reason_label_mismatch": 0,
        "repaired": 0,
        "leak_count": 0,
        "factcheck_evidence_used": 0,
        "credible_non_factcheck_groups": 0,
        "route": "",
        "refute_kind": "",
        "core_id": "A1",
        "gate": "",
        "gate_reason": "",
    }


def factcheck_evidence_used(state: dict, own_key: str) -> int:
    """Distinct fact-check pages that actually entered the decision.

    An evidence row counts. So does a prior the route reused, because
    that article is the verdict. A fact-check that only sat in the
    memory list does not. The source article itself is a leak, not this
    count, even when its host is a fact-checker.
    """
    found: set[str] = set()

    def consider(url, title) -> None:
        key = url_key(url or "")
        if not key or key == own_key:
            return
        if looks_like_factcheck(url or "", title or ""):
            found.add(key)

    for row in state.get("evidence") or []:
        if isinstance(row, dict):
            consider(row.get("url"), row.get("title") or "")
    if state.get("route") == "reuse_prior":
        best = state.get("best_match") if isinstance(state.get("best_match"), dict) else {}
        consider(best.get("url"), best.get("title") or best.get("primary_claim") or "")
    return len(found)


def credible_non_factcheck_groups(state: dict, own_key: str) -> int:
    """Independent tier 1–2 outlets in the passages, excluding fact-checkers.

    One registrable domain is one group, which is the same independence
    key the gate uses for two pages of one newsroom. The source article
    is not a group the system was allowed to use.
    """
    domains: set[str] = set()
    for passage in state.get("passages") or []:
        url = getattr(passage, "url", None)
        title = getattr(passage, "title", None)
        if url is None and isinstance(passage, dict):
            url = passage.get("url")
            title = passage.get("title")
        key = url_key(url or "")
        if not key or key == own_key:
            continue
        if looks_like_factcheck(url or "", title or ""):
            continue
        if credibility_tier(url or "") > 2:
            continue
        domain = registrable_domain(url or "")
        if domain:
            domains.add(domain)
    return len(domains)


def record_from_state(
    state: dict,
    *,
    article_id: str,
    interrupted: bool,
    llm_calls: int,
    searches: int,
    latency: float,
    leak_count: int = 0,
    factcheck_evidence_used: int = 0,
    credible_non_factcheck_groups: int = 0,
    own_url: str = "",
) -> dict:
    """The result line. Evidence quotes are kept so the scorer can re-check them.

    The passages are the windows the stance step already stored. The
    scorer does not fetch the pages again. A review pause forces
    needs_human even when the state flag is still false: the flag is
    for a verdict the rules withheld, and a pause is a verdict nobody
    accepted.
    """
    verdict = state.get("verdict")
    if verdict is not None:
        verdict = str(verdict).strip() or None
    confidence = _float_or_none(state.get("confidence"))
    needs_human = True if interrupted else bool(state.get("needs_human"))
    if verdict is None:
        needs_human = True
    stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    best = state.get("best_match") if isinstance(state.get("best_match"), dict) else {}
    evidence = []
    for item in state.get("evidence") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        evidence.append({
            "id": str(item["id"]),
            "url": str(item.get("url") or ""),
            "netloc": str(item.get("netloc") or ""),
            "tier": item.get("tier"),
            "stance": str(item.get("stance") or ""),
            "quote": str(item.get("quote") or ""),
            "passage": str(item.get("passage") or ""),
            "group_id": str(item.get("group_id") or ""),
            "atomic_id": str(item.get("atomic_id") or "A1"),
            "group_conflict": bool(item.get("group_conflict")),
        })
    citations = []
    for item in state.get("citations") or []:
        ident = str(item)
        if ident not in citations:
            citations.append(ident)
    return {
        "article_id": article_id,
        "verdict": verdict,
        "confidence": confidence,
        "needs_human": needs_human,
        "citations": citations,
        "evidence": evidence,
        "prior_present": bool(best.get("article_id") or best.get("url") or best.get("primary_claim")),
        "llm_calls": int(llm_calls),
        "searches": int(searches),
        "latency": float(latency),
        "reason_label_mismatch": int(stats.get("reason_label_mismatch") or 0),
        "repaired": int(stats.get("repaired") or 0),
        "leak_count": int(leak_count),
        "factcheck_evidence_used": int(factcheck_evidence_used),
        "credible_non_factcheck_groups": int(credible_non_factcheck_groups),
        "route": str(state.get("route") or ""),
        "refute_kind": _core_refute_kind(state)[1],
        "core_id": _core_refute_kind(state)[0],
        "gate": str(state.get("gate") or ""),
        "gate_reason": _gate_reason(state),
        "search_capped": _search_capped(state),
        "fc_search_cached": _fc_search_cached(stats),
        "factcheck_pages_added": int(stats.get("factcheck_pages_added") or 0),
        "own_article_excluded": _own_article_excluded(state, own_url),
        "notes": [str(item) for item in (stats.get("notes") or [])],
    }


def checkpoint_values(thread_id: str, paths: dict) -> dict:
    """State at the end of the run, including a run parked on review."""
    graph = build_graph(
        paths.get("checkpoint_path"),
        paths.get("audit_path"),
        paths.get("additions_path"),
    )
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = getattr(snap, "values", None)
    return values if isinstance(values, dict) else {}


def eval_paths() -> dict:
    """Checkpoint files under data/eval, not the demo's databases."""
    root = config.EVAL_DIR / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint_path": root / "checkpoints.db",
        "audit_path": root / "audit.db",
        "additions_path": root / "memory_additions.parquet",
    }


def estimate(system_rows: list[dict], baseline_rows: list[dict], total: int = FULL_N) -> str:
    """Linear scale from the claims this process finished. A cold run's bill.

    llm_calls counts cache hits, because the pipeline did the call.
    calls_today does not. A prompt this process has not seen is a miss,
    so the scaled llm_calls is the Gemini-call estimate for the full set.
    Time adds the two conditions because they run one after the other.
    """
    lines = [f"estimate for {total} claims, from the rows finished in this process"]
    system_minutes = _mean(system_rows, "latency")
    baseline_minutes = _mean(baseline_rows, "latency")
    if system_rows:
        lines.append(
            "  system: "
            f"{_mean(system_rows, 'llm_calls') * total:.0f} LLM calls, "
            f"{_mean(system_rows, 'searches') * total:.0f} searches, "
            f"{system_minutes * total / 60:.1f} min"
        )
    else:
        lines.append("  system: no finished claims in this process")
        system_minutes = 0.0
    if baseline_rows:
        lines.append(
            "  baseline: "
            f"{_mean(baseline_rows, 'llm_calls') * total:.0f} LLM calls, "
            f"{baseline_minutes * total / 60:.1f} min"
        )
    else:
        lines.append("  baseline: no finished claims in this process")
        baseline_minutes = 0.0
    both = (system_minutes + baseline_minutes) * total / 60
    lines.append(f"  system + baseline, one after the other: {both:.1f} min")
    lines.append("  cache hits are inside the LLM-call count and outside calls_today")
    return "\n".join(lines)


def leak_total(path: Path) -> int:
    """Sum of leak_count over every row already in the results file."""
    if not path.is_file():
        return 0
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += int(row.get("leak_count") or 0)
    return total


def select_recycled(rows: list[dict], memory) -> list[dict]:
    """Archive claims whose fact-check is in the memory index.

    The true_version rows are a different sentence from the labelled
    claim, so they are not this condition. A labelled row whose article
    is missing from the index cannot be a recycled rumour either.
    """
    indexed = {str(item) for item in np.asarray(memory.ids).astype(str)}
    chosen = []
    for row in rows:
        if str(row.get("source") or "") == "true_version":
            continue
        if str(row.get("article_id") or "") not in indexed:
            continue
        chosen.append(row)
    return chosen


def recycled_claim_date(factcheck_date: str) -> str:
    """Sixty days after the fact-check. The debunk is then in the past."""
    day = date.fromisoformat(str(factcheck_date).strip()[:10])
    return (day + timedelta(days=60)).isoformat()


def _memory_rank(hits: list[dict], group: set[str]) -> int | None:
    """1-based rank of the own article or a copy, inside the top 5. None if absent."""
    for index, hit in enumerate(hits[:5], start=1):
        if str(hit.get("article_id") or "") in group:
            return index
    return None


def invoke_recycled(row: dict, memory) -> dict:
    """Boxes 1–6 on the testset sentence. No web search.

    Extract is not called. It would replace the testset sentence with a
    shorter paraphrase, and this condition's input is that sentence.
    Ingest still runs, then the claim is put back to the CSV text.
    Memory's date_before is the new claim date, sixty days after the
    fact-check, so the debunk is eligible. Nothing is excluded: the leak
    guard and the own-article rule are off for this condition only.
    An exact match reuses the archive verdict through assess, which does
    not call a model on that route. Anything else stops as "would investigate".
    """
    from misinfo_agent.assess import assess
    from misinfo_agent.graph import _ingest
    from misinfo_agent.prior_match import decide_prior

    own_id = str(row["article_id"])
    factcheck_date = str(row["claim_date"]).strip()[:10]
    claim_date = recycled_claim_date(factcheck_date)
    group = set(near_copy_ids(memory.ids, memory.E, own_id, config.DUP_COS))
    before_llm = len(get_usage())
    started = time.perf_counter()
    state = {
        "raw_input": row["input"],
        "input_type": "text",
        "run_marks": {"eval_claim_date": claim_date},
    }
    state.update(_ingest(state))
    # The CSV sentence, not the paraphrase extract would have written.
    state["claim"] = row["input"]
    state["claim_date"] = claim_date
    hits = memory.search(row["input"], date_before=claim_date)
    state["matches"] = hits
    decision = decide_prior(
        row["input"],
        hits,
        now=claim_date,
        entities=state.get("entities"),
        form="text",
    )
    state.update(decision)
    best = state.get("best_match") if isinstance(state.get("best_match"), dict) else {}
    reused_from = str(best.get("article_id") or "")
    rank = _memory_rank(hits, group)
    if state.get("route") == "reuse_prior":
        drafted = assess(state)
        verdict = drafted.get("verdict")
        different = "yes" if reused_from and reused_from not in group else "no"
    else:
        verdict = "would investigate"
        different = ""
        reused_from = reused_from if state.get("match_type") == "similar" else ""
    return {
        "article_id": own_id,
        "label": str(row.get("label") or ""),
        "factcheck_date": factcheck_date,
        "claim_date": claim_date,
        "memory_rank": rank,
        "match_type": str(state.get("match_type") or ""),
        "route": str(state.get("route") or ""),
        "verdict": verdict,
        "reused_from": reused_from if verdict != "would investigate" else reused_from,
        "reused_from_different_claim": different,
        "llm_calls": len(get_usage()) - before_llm,
        "latency": round(time.perf_counter() - started, 2),
        "searches": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end eval, one claim at a time")
    parser.add_argument("--limit", type=int, default=None, help="only the first N testset rows")
    parser.add_argument("--condition", choices=("system", "baseline", "both", "recycled"), default="both")
    parser.add_argument(
        "--sites",
        choices=("blocked", "allowed"),
        default="blocked",
        help="blocked: fact-check sites excluded (default). allowed: those sites may be retrieved.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="serve search from the cache; a miss returns no hits and does not call Tavily",
    )
    parser.add_argument("--results", default=None, help="jsonl path to write, instead of the default for this condition")
    parser.add_argument("--only", default=None, help="comma-separated article ids")
    args = parser.parse_args(argv)
    if args.cache_only:
        # Read by search.cache_only_enabled before any query is sent.
        os.environ["SEARCH_CACHE_ONLY"] = "1"
    apply_honesty(args.sites)
    rows = load_testset(config.EVAL_DIR / "testset.csv")
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        rows = [row for row in rows if str(row["article_id"]) in wanted]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        rows = rows[: args.limit]
    started = time.perf_counter()
    before = calls_today()
    system_path = Path(args.results) if args.results else system_results_path(args.sites)
    print(
        f"claims {len(rows)}  condition {args.condition}  sites {args.sites}  "
        f"cache_only={bool(args.cache_only)}  "
        f"calls_today_before={before}  tavily_month={searches_this_month()}",
        flush=True,
    )
    system_ran: list[dict] = []
    baseline_ran: list[dict] = []
    try:
        if args.condition in {"system", "both"}:
            # Imported here so a baseline-only run does not load the index.
            from misinfo_agent.memory import get_memory

            print(f"system {args.sites} -> {system_path.name}", flush=True)
            system_ran = run_rows(
                rows,
                system_path,
                lambda row: invoke_system(row, get_memory(), eval_paths(), args.sites),
            )
        if args.condition == "recycled":
            from misinfo_agent.memory import get_memory

            memory = get_memory()
            chosen = select_recycled(rows, memory)
            print(
                f"recycled {len(chosen)} of {len(rows)} "
                "selected as source != true_version and article_id in the memory index "
                f"-> {system_path.name}",
                flush=True,
            )
            recycled_start = calls_today()

            def _one_recycled(row, memory=memory, recycled_start=recycled_start):
                # The context check is the only new model call, and only on
                # an exact pair. One hundred is the stop for this condition.
                if calls_today() - recycled_start >= 100:
                    print(
                        f"stopped: recycled Gemini calls would pass 100 "
                        f"calls_today={calls_today()} tavily_month={searches_this_month()}",
                        flush=True,
                    )
                    raise SystemExit(2)
                return invoke_recycled(row, memory)

            system_ran = run_rows(chosen, system_path, _one_recycled)
            _print_recycled(system_ran)
        if args.condition in {"baseline", "both"}:
            # One baseline file. It does not depend on whether fact-check
            # sites were blocked, because this condition never searches.
            print("baseline", flush=True)
            baseline_ran = run_rows(
                rows,
                config.EVAL_DIR / "results_baseline.jsonl",
                invoke_baseline,
            )
    except LLMQuotaError as exc:
        # The claim that raised is not in the file. Do not start the next one.
        _finish(started, before, system_path, system_ran, baseline_ran)
        print(f"stopped: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
    except SystemExit:
        _finish(started, before, system_path, system_ran, baseline_ran)
        raise
    _finish(started, before, system_path, system_ran, baseline_ran)


def _empty_result(article_id: str, *, llm_calls: int, latency: float, error: str) -> dict:
    return {
        "article_id": article_id,
        "verdict": None,
        "confidence": None,
        "needs_human": True,
        "citations": [],
        "evidence": [],
        "prior_present": False,
        "llm_calls": int(llm_calls),
        "searches": 0,
        "latency": float(latency),
        "reason_label_mismatch": 0,
        "repaired": 0,
        "leak_count": 0,
        "factcheck_evidence_used": 0,
        "credible_non_factcheck_groups": 0,
        "route": "",
        "refute_kind": "",
        "core_id": "A1",
        "gate": "",
        "gate_reason": "",
        "error": error,
    }


def _core_refute_kind(state: dict) -> tuple[str, str]:
    """The core atomic id, and the draft's refute_kind for that claim.

    False versus Misleading is this field. The gate only says refuted.
    """
    core = "A1"
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("core") is True:
            core = str(item.get("id") or "A1")
            break
    kind = ""
    for item in state.get("atomic_results") or []:
        if isinstance(item, dict) and str(item.get("atomic_id") or "") == core:
            kind = str(item.get("refute_kind") or "")
            break
    return core, kind


def _search_capped(state: dict) -> str | None:
    """per_claim, monthly, or None. The notes also carry the same words."""
    value = state.get("search_capped")
    if value in {"per_claim", "monthly"}:
        return value
    stats = state.get("investigate_stats")
    if isinstance(stats, dict) and stats.get("search_capped") in {"per_claim", "monthly"}:
        return stats["search_capped"]
    return None


def _fc_search_cached(stats: dict) -> str:
    """yes only when the fact-check searches were cache hits and none missed.

    A claim whose fact-check budget did not run is no: there was no cached
    fact-check search to replay.
    """
    hits = int(stats.get("factcheck_cache_hits") or 0)
    misses = int(stats.get("factcheck_cache_misses") or 0)
    if hits and not misses:
        return "yes"
    return "no"


def _own_article_excluded(state: dict, own_url: str) -> str:
    """yes when the source article's canonical URL is not in the evidence."""
    if not own_url:
        return "yes"
    own = url_key(own_url)
    for row in state.get("evidence") or []:
        if isinstance(row, dict) and url_key(str(row.get("url") or "")) == own:
            return "no"
    return "yes"


def _gate_reason(state: dict) -> str:
    """The sentences box 9 wrote, or the gate label when it wrote none."""
    parts = []
    for item in state.get("gate_reasons") or []:
        text = " ".join(str(item).split())
        if text:
            parts.append(text)
    if parts:
        return " ".join(parts)
    return str(state.get("gate") or "")


def _wilson(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return None
    proportion = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (proportion + z2 / (2 * n)) / denom
    margin = z * ((proportion * (1 - proportion) / n + z2 / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _print_recycled(rows: list[dict]) -> None:
    """Hit rates and reused-verdict accuracy for the rows just written."""
    n = len(rows)
    hit1 = sum(1 for row in rows if row.get("memory_rank") == 1)
    hit5 = sum(1 for row in rows if isinstance(row.get("memory_rank"), int) and row["memory_rank"] <= 5)
    reused = [row for row in rows if row.get("route") == "reuse_prior" and row.get("verdict") != "would investigate"]
    correct = sum(1 for row in reused if str(row.get("verdict")) == str(row.get("label")))
    wrong = [row for row in reused if str(row.get("verdict")) != str(row.get("label"))]
    interval = _wilson(correct, len(reused))
    latencies = [float(row.get("latency") or 0) for row in rows]
    median = float(np.median(latencies)) if latencies else 0.0
    calls = sum(int(row.get("llm_calls") or 0) for row in rows)
    print(
        f"recycled n={n} Hit@1={hit1}/{n} Hit@5={hit5}/{n} "
        f"exact_reuse={len(reused)}/{n} reused_correct={correct}/{len(reused)} "
        f"wilson={interval} wrong={len(wrong)} median_s={median:.2f} gemini_calls={calls}",
        flush=True,
    )
    for row in wrong:
        print(
            f"  wrong {row.get('article_id')} gold={row.get('label')} "
            f"verdict={row.get('verdict')} from={row.get('reused_from')} "
            f"different={row.get('reused_from_different_claim')}",
            flush=True,
        )


def _finish(started: float, before: int, system_path: Path, system_ran: list[dict], baseline_ran: list[dict]) -> None:
    elapsed = time.perf_counter() - started
    print(estimate(system_ran, baseline_ran), flush=True)
    print(f"leak_count total {leak_total(system_path)}", flush=True)
    print(
        f"calls_used={calls_today() - before} calls_today={calls_today()} "
        f"tavily_month={searches_this_month()}",
        flush=True,
    )
    print(f"elapsed_s={elapsed:.1f} elapsed_min={elapsed / 60:.1f}", flush=True)


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key) or 0) for row in rows) / len(rows)


if __name__ == "__main__":
    main()
