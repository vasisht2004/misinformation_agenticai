"""Score PHOTO_RULE v0–v3 from the Vision cache and the page cache.

No Vision call and no Tavily call. Dates come from the page cache.
Stances come from the blocked image-eval checkpoints. When a page was
not in that checkpoint its stance is neutral, and only the date rule
can mark it. Fact-check hosts are dropped in blocked mode and kept in
allowed mode. The selection rule is applied to blocked mode only, and
it is the rule written in this file before the tables are printed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.fetch import _read_cache
from misinfo_agent.gate import PHOTO_OUT_OF_CONTEXT, gate
from misinfo_agent.ingest import _downscale, _jpeg_bytes, _open_image
from misinfo_agent.investigate import Passage, assign_groups, canonical_url, credibility_tier
from misinfo_agent.provenance import claim_event_date, is_dated_earlier
from misinfo_agent.provenance import _choose_pages
from misinfo_agent.search import site_host

# Written before the tables below are produced. Applied to blocked mode.
# (a) Drop a variant that calls any gold-True image False or Misleading.
# (b) Among the rest, the most correct 2-class decisions.
#     Correct: a decision, and (gold True) matches (verdict True).
#     False and Misleading are both not-true. Unverifiable is not a decision.
# (c) A tie goes to the fewer decisions.
# (d) If every variant fails (a), the choice is v0.
# If two variants also tie on the decision count, the lower number wins.
# v0 admits fewer kinds of page than v1, and v1 fewer than v2 and v3.
RULES = ("v0", "v1", "v2", "v3")
NOT_TRUE = {"False", "Misleading"}


def main() -> None:
    rows = _rows()
    tables = {}
    for mode in ("blocked", "allowed"):
        tables[mode] = _score(rows, blocked=mode == "blocked")
        _print_table(mode, tables[mode])
    choice, why = _select(tables["blocked"])
    print(f"CHOICE {choice}")
    print(why)


def _rows() -> list[dict]:
    path = config.EVAL_DIR / "images_testset.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _score(rows: list[dict], *, blocked: bool) -> list[dict]:
    stances = _stored_stances()
    config.BLOCK_FACTCHECK_SITES = blocked
    scored = []
    for row in rows:
        pages = _vision_pages(row["file"])
        kept = _choose_pages(pages)
        matches = _matches(kept, stances, row)
        scored.append(
            {
                "file": row["file"].rsplit("/", 1)[-1],
                "label": row["label"],
                "verdicts": {rule: _verdict(row, matches, rule) for rule in RULES},
            }
        )
    return scored


def _vision_pages(file_name: str) -> list[dict]:
    _original, upright = _open_image(config.ROOT / file_name)
    digest = hashlib.sha256(_jpeg_bytes(_downscale(upright))).hexdigest()
    path = config.VISION_CACHE_DIR / f"{digest}.r{config.VISION_MAX_RESULTS}.json"
    if not path.is_file():
        raise SystemExit(f"vision cache missing for {file_name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages") if isinstance(payload, dict) else None
    return pages if isinstance(pages, list) else []


def _stored_stances() -> dict[str, dict]:
    """Stance and date the blocked re-run already stored, keyed by canonical URL."""
    path = config.EVAL_DIR / "results_images.jsonl"
    if not path.is_file():
        return {}
    from misinfo_agent.graph import build_graph

    graph = build_graph()
    found: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        thread = str(record.get("thread_id") or "")
        if not thread:
            continue
        snap = graph.get_state({"configurable": {"thread_id": thread}})
        values = getattr(snap, "values", None)
        provenance = values.get("image_provenance") if isinstance(values, dict) else None
        if not isinstance(provenance, dict):
            continue
        for match in provenance.get("matches") or []:
            if not isinstance(match, dict) or not match.get("url"):
                continue
            found[canonical_url(str(match["url"]))] = match
    return found


def _matches(pages: list[dict], stances: dict[str, dict], row: dict) -> list[dict]:
    passages = []
    built = []
    for index, page in enumerate(pages):
        url = str(page.get("url") or "")
        cached = _read_cache(url) or {}
        stored = stances.get(canonical_url(url)) or {}
        published = str(cached.get("published_date") or stored.get("date") or "")
        source = cached.get("date_source") or stored.get("date_source")
        if source == "untrusted" or not published:
            published, source = "", "none"
        text = str(cached.get("text") or page.get("title") or "")
        passages.append(
            Passage(
                passage_id=f"P{index + 1}",
                url=url,
                netloc=site_host(url) or "",
                title=str(page.get("title") or ""),
                text=text[:4000],
                cos=0.0,
                published_date=published or None,
                date_source=None if source in {None, "none"} else str(source),
                fetched=True,
            )
        )
        built.append(
            {
                "url": url,
                "host": site_host(url) or "",
                "title": str(page.get("title") or stored.get("title") or ""),
                "match": page.get("match") or "page",
                "tier": credibility_tier(url),
                "date": published,
                "date_source": source or "none",
                "stance": stored.get("stance") or "neutral",
                "quote": stored.get("quote") or "",
            }
        )
    groups = assign_groups(passages)
    event_state = {
        "caption": row["caption"],
        "claim": row["caption"],
        "claim_date": row["claim_date"],
    }
    # The date quote is applied per rule inside _verdict, because v0 and
    # v2 do not date the same pages. The group id does not depend on that.
    for index, match in enumerate(built):
        match["group_id"] = groups.get(f"P{index + 1}") or f"row:{index}"
        match["_event_state"] = event_state
    return built


def _verdict(row: dict, matches: list[dict], rule: str) -> str:
    config.PHOTO_RULE = rule
    event_state = {
        "caption": row["caption"],
        "claim": row["caption"],
        "claim_date": row["claim_date"],
    }
    event = claim_event_date(event_state)
    prepared = []
    for match in matches:
        copy = dict(match)
        copy.pop("_event_state", None)
        if is_dated_earlier(copy, event):
            copy["stance"] = "refutes"
            copy["quote"] = (
                f"Image on this page dated {copy.get('date')}, "
                f"before the claimed event ({event.isoformat()})."
            )
        prepared.append(copy)
    state = {
        "claim": row["caption"],
        "caption": row["caption"],
        "input_type": "image",
        "checkable": True,
        "rounds": 2,
        "claim_date": row["claim_date"],
        "atomic_claims": [
            {"id": "A1", "text": row["caption"], "core": True, "checkable": True}
        ],
        "evidence": [],
        "image_provenance": {
            "vision_status": "cached",
            "best_guess": [],
            "first_seen": None,
            "matches": prepared,
        },
    }
    judged = gate(state)
    status = (judged.get("atomic_status") or {}).get("A1")
    reasons = " ".join(judged.get("gate_reasons") or [])
    if status == "refuted" and PHOTO_OUT_OF_CONTEXT in reasons:
        return "Misleading"
    if status == "supported":
        return "True"
    return "Unverifiable"


def _print_table(mode: str, rows: list[dict]) -> None:
    print(f"TABLE {mode}")
    print(f"{'file':<10} {'gold':<12} {'v0':<14} {'v1':<14} {'v2':<14} {'v3':<14}")
    totals = {rule: {"decided": 0, "correct": 0, "wrong": 0, "abstained": 0, "true_hurt": 0} for rule in RULES}
    for row in rows:
        cells = [row["verdicts"][rule] for rule in RULES]
        print(f"{row['file']:<10} {row['label']:<12} " + " ".join(f"{cell:<14}" for cell in cells))
        for rule in RULES:
            _count(totals[rule], row["label"], row["verdicts"][rule])
    for rule in RULES:
        item = totals[rule]
        print(
            f"TOTAL {mode} {rule} decided={item['decided']} correct={item['correct']} "
            f"wrong={item['wrong']} abstained={item['abstained']} true_hurt={item['true_hurt']}"
        )


def _count(bucket: dict, label: str, verdict: str) -> None:
    if verdict == "Unverifiable":
        bucket["abstained"] += 1
        return
    bucket["decided"] += 1
    gold_true = label == "True"
    called_true = verdict == "True"
    if gold_true == called_true:
        bucket["correct"] += 1
    else:
        bucket["wrong"] += 1
    if gold_true and verdict in NOT_TRUE:
        bucket["true_hurt"] += 1


def _select(rows: list[dict]) -> tuple[str, str]:
    totals = {rule: {"decided": 0, "correct": 0, "true_hurt": 0} for rule in RULES}
    for row in rows:
        for rule in RULES:
            verdict = row["verdicts"][rule]
            if row["label"] == "True" and verdict in NOT_TRUE:
                totals[rule]["true_hurt"] += 1
            if verdict == "Unverifiable":
                continue
            totals[rule]["decided"] += 1
            if (row["label"] == "True") == (verdict == "True"):
                totals[rule]["correct"] += 1
    eligible = [rule for rule in RULES if totals[rule]["true_hurt"] == 0]
    if not eligible:
        return "v0", "every variant fails (a); choice is v0"
    best_correct = max(totals[rule]["correct"] for rule in eligible)
    contenders = [rule for rule in eligible if totals[rule]["correct"] == best_correct]
    fewest = min(totals[rule]["decided"] for rule in contenders)
    choice = next(rule for rule in contenders if totals[rule]["decided"] == fewest)
    detail = ", ".join(
        f"{rule} correct={totals[rule]['correct']} decided={totals[rule]['decided']} hurt={totals[rule]['true_hurt']}"
        for rule in RULES
    )
    return choice, f"eligible={','.join(eligible)}; {detail}; choice={choice}"


if __name__ == "__main__":
    main()
