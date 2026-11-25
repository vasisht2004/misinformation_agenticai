"""Run 16 established facts through the product pipeline, then replay gates.

The live command searches. It is resumable: a claim already in
data/eval/results_facts.jsonl is not run again. The monthly Tavily
counter is allowed to reach 990 and the next real call is refused, so
the counter cannot become 991.

  python scripts/run_facts.py           # live product run
  python scripts/run_facts.py --replay  # gate variants on the stored evidence, no calls

Fact-check sites stay in the search. That is the product, not the eval's
allowed condition: the eval mark runs a blocked search and then a second
fact-check budget. This script does not set that mark. The leakage guard
stays off, and every gate flag stays off for the live run. Replay scores
G0 and the study variants on the stored evidence. G4 uses theta 2.0, the
value the dev split already chose. It is not refit here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The counter may sit on this number. The next real search is refused.
FACTS_SEARCH_BUDGET = 990
# Dev chose 2.0. These 16 do not get a vote on it.
FACTS_THETA = 2.0

FACTS_CSV = ROOT / "data" / "eval" / "facts_testset.csv"
RESULTS = ROOT / "data" / "eval" / "results_facts.jsonl"
DOC = ROOT / "docs" / "eval_diagnosis.md"
HEADING = "## ESTABLISHED-FACTS SET"
RESULTS_MARK = "<!-- facts-results -->"
RULE = (
    "A gate variant is turned on only if, on these 16, it makes "
    "MORE correct decisions than G0 and ZERO additional wrong decisions "
    "(2-class; also report 3-class). Ties keep G0. "
    "Support decided from tier 3-4 sources alone is never allowed."
)

_COUNTER_FILES = (
    ROOT / "data" / "cache" / "llm_daily_count.json",
    ROOT / "data" / "cache" / "search_monthly_count.json",
)
_STATE_KEYS = (
    "claim",
    "claim_date",
    "route",
    "checkable",
    "checkable_reason",
    "not_checkable_reason",
    "input_type",
    "rounds",
    "atomic_claims",
    "atomic_results",
    "evidence",
    "passages",
    "investigate_stats",
    "verdict",
    "needs_human",
    "confidence",
    "gate",
    "gate_reasons",
    "best_match",
    "match_type",
    "domain",
)
_SNAPSHOT: dict | None = None


def load_facts(path: Path = FACTS_CSV) -> list[dict]:
    """The registered rows. claim_date is the day recorded in the file.

    It is not rewritten to today. A later calendar day has to load the
    same date the set was written with, or the memory filter and the
    page-date guard would move under the test.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16:
        raise SystemExit(f"expected 16 facts, found {len(rows)}")
    for row in rows:
        recorded = str(row.get("claim_date") or "")
        try:
            date.fromisoformat(recorded)
        except ValueError:
            raise SystemExit(f"{row.get('id')} claim_date is not a date: {recorded}")
        if row.get("label") not in {"False", "Misleading", "True"}:
            raise SystemExit(f"unexpected label on {row.get('id')}")
    return rows


def claim_may_start(count: int, budget: int = FACTS_SEARCH_BUDGET) -> bool:
    """True when a new claim cannot push the monthly counter past the budget.

    The search module refuses once the count is at the cap. A claim
    started below the cap can finish the call that lands on the cap.
    The call after that is refused, so the counter never passes it.
    """
    return int(count) < int(budget)


def wrong2(row: dict) -> int:
    """Decided 2-class misses. An abstention is not one of them."""
    return int(row["decided"]) - int(row["correct2"])


def choose_facts_winner(rows: list[dict], blocked: set[str]) -> str | None:
    """The facts rule. None means G0 stays and no flag changes.

    A variant must beat G0 on 2-class correct decisions and must not add
    a 2-class wrong decision. The same correct count is a tie, and a tie
    stays with G0 even when the variant abstains less. A variant that
    supported a claim from tier 3-4 pages alone is named in `blocked`
    and cannot be turned on. When two variants both beat G0 by the same
    2-class counts, the earlier one wins: a later combo that did not
    improve on it does not turn extra flags on.
    """
    g0 = next(row for row in rows if row["name"] == "G0")
    base_correct = int(g0["correct2"])
    base_wrong = wrong2(g0)
    eligible = []
    for row in rows:
        name = str(row["name"])
        if name == "G0" or name in blocked:
            continue
        if int(row["correct2"]) > base_correct and wrong2(row) <= base_wrong:
            eligible.append(row)
    if not eligible:
        return None
    best_correct = max(int(row["correct2"]) for row in eligible)
    top = [row for row in eligible if int(row["correct2"]) == best_correct]
    best_wrong = min(wrong2(row) for row in top)
    top = [row for row in top if wrong2(row) == best_wrong]
    top.sort(key=lambda row: int(row["order"]))
    return str(top[0]["name"])


def support_from_tier34_only(state: dict, gated: dict) -> bool:
    """True when this gate called the core supported with no tier 1-2 group.

    The shipped paths do not do this. The check is here so a replay
    cannot turn a flag on for a support that only blogs and reference
    sites produced. A reused prior is the archive's verdict, not a
    support this gate decided.
    """
    import misinfo_agent.config as config
    from misinfo_agent.gate import _atomics, _core_id, _groups, _rows, _split_core_supports, counted_rows

    if state.get("route") == "reuse_prior":
        return False
    if gated.get("gate") in {"conflicting", "error", "skipped", "search_more"}:
        return False
    atomics = _atomics(state)
    core = _core_id(atomics)
    if (gated.get("atomic_status") or {}).get(core) != "supported":
        return False
    text = ""
    for item in atomics:
        if isinstance(item, dict) and str(item.get("id") or "") == core:
            text = str(item.get("text") or state.get("claim") or "")
            break
    # Same rows the gate just counted, including the shared-source merge
    # when that flag is the one under test.
    rows = counted_rows(
        _rows(
            state.get("evidence"),
            core,
            drop_after=bool(config.EVAL_DROP_AFTER_CLAIM),
        )
    )
    if config.TRUE_NEEDS_CORE_MATCH:
        rows, _detail = _split_core_supports(rows, text)
    return len(_groups(rows)["support_credible"]) == 0


def apply_product() -> None:
    """The checker a person gets, with this run's search ceiling.

    Flags stay off until replay applies the pre-registered rule. The
    ceiling is process-local: the file's monthly max is the account
    cap, and this command stops one step sooner.
    """
    import misinfo_agent.config as config

    config.BLOCK_FACTCHECK_SITES = False
    config.EVAL_DROP_AFTER_CLAIM = False
    # A batch of known facts is not someone forwarding a message.
    # scripts/run.py makes the same choice so the reach table stays a
    # record of real submissions.
    config.RECORD_SUBMISSIONS = False
    config.LLM_CACHE = True
    config.GATE_PRIMARY_SOURCE = False
    config.GATE_WEAK_REFUTE = False
    config.GATE_SHARED_SOURCE = False
    config.GATE_EVIDENCE_SCORE = False
    config.TAVILY_MONTHLY_MAX = FACTS_SEARCH_BUDGET


def _streamlit_lines() -> list[str]:
    """Command lines of running Streamlit processes. Empty when it is down.

    One pipeline at a time: the app and this script would share the
    search counter and the checkpoint lock.
    """
    try:
        found = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    lines = []
    for line in found.stdout.splitlines():
        if "streamlit" in line.lower() and "grep" not in line:
            lines.append(line.strip())
    return lines


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _state_slice(state: dict) -> dict:
    """The fields the gate and the verdict mapping read. Nothing else.

    The checkpoint keeps the raw submission. The replay does not need
    it, and the results file should not grow a second copy of the claim.
    """
    return {key: _jsonable(state.get(key)) for key in _STATE_KEYS}


def _load_done(path: Path) -> set[str]:
    done = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                done.add(str(row["id"]))
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_results(path: Path = RESULTS) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing results: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _published(record: dict) -> str | None:
    """The live label, when the run actually decided it.

    A pause is an abstention. Replay must not grade a verdict the
    product withheld.
    """
    if record.get("needs_human"):
        return None
    verdict = record.get("verdict")
    if verdict in {"False", "Misleading", "True"}:
        return str(verdict)
    return None


def run_live() -> None:
    """One claim at a time. A quota error is not written, so a resume retries it."""
    if _streamlit_lines():
        raise SystemExit("stopped: Streamlit is running; one pipeline process at a time")
    # A leftover cache-only flag would turn a miss into an empty search
    # and the run would not be live. This command is the live one.
    os.environ.pop("SEARCH_CACHE_ONLY", None)
    apply_product()

    from misinfo_agent.graph import build_graph, run_pipeline
    from misinfo_agent.llm import LLMQuotaError, calls_today, get_usage
    from misinfo_agent.search import SearchQuotaError, get_search_usage, searches_this_month

    rows = load_facts()
    done = _load_done(RESULTS)
    paths = _fact_paths()
    before = searches_this_month()
    print(f"tavily_before={before} budget={FACTS_SEARCH_BUDGET} done={len(done)}", flush=True)
    if before > FACTS_SEARCH_BUDGET:
        raise SystemExit(f"stopped: tavily month counter is already {before}")
    try:
        for index, row in enumerate(rows, start=1):
            fact_id = str(row["id"])
            if fact_id in done:
                print(f"skip {fact_id}", flush=True)
                continue
            count = searches_this_month()
            if not claim_may_start(count):
                print(
                    f"stopped: tavily_month={count} is at the {FACTS_SEARCH_BUDGET} budget; "
                    f"{fact_id} was not started",
                    flush=True,
                )
                break
            print(
                f"[{index}/16] {fact_id} {row['label']} tavily_month={count}",
                flush=True,
            )
            try:
                record = _one_claim(row, paths, build_graph, run_pipeline, get_usage, get_search_usage)
            except (LLMQuotaError, SearchQuotaError) as exc:
                print(f"stopped: {exc} tavily_month={searches_this_month()}", flush=True)
                raise SystemExit(2) from exc
            _append(RESULTS, record)
            print(
                f"  route={record['route']} verdict={record['verdict']} "
                f"needs_human={record['needs_human']} searches={record['searches']} "
                f"searches_live={record['searches_live']} llm_calls={record['llm_calls']} "
                f"latency_s={record['latency']} tavily_month={searches_this_month()}",
                flush=True,
            )
    finally:
        after = searches_this_month()
        print(f"tavily_before={before} tavily_after={after} calls_today={calls_today()}", flush=True)


def _fact_paths() -> dict:
    """Checkpoints for this set only. The demo and the 50-claim eval keep theirs."""
    import misinfo_agent.config as config

    root = config.EVAL_DIR / "facts_runs"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint_path": root / "checkpoints.db",
        "audit_path": root / "audit.db",
        "additions_path": root / "memory_additions.parquet",
    }


def _one_claim(row, paths, build_graph, run_pipeline, get_usage, get_search_usage) -> dict:
    before_llm = len(get_usage())
    before_search = len(get_search_usage())
    started = time.perf_counter()
    thread = f"facts-{row['id']}-{uuid.uuid4().hex[:8]}"
    # No eval_sites and no excluded archive url. The date is the CSV day,
    # which is today, so memory's date filter and the card agree with the file.
    out = run_pipeline(
        row["claim"],
        input_type="text",
        thread_id=thread,
        checkpoint_path=paths["checkpoint_path"],
        audit_path=paths["audit_path"],
        additions_path=paths["additions_path"],
        eval_claim_date=str(row["claim_date"]),
    )
    latency = round(time.perf_counter() - started, 2)
    state = _checkpoint(thread, paths, build_graph)
    if out.get("status") == "interrupt":
        review = out.get("review") if isinstance(out.get("review"), dict) else {}
        if state.get("verdict") is None and review.get("verdict") is not None:
            state = dict(state)
            state["verdict"] = review.get("verdict")
            state["needs_human"] = True
    usage = get_search_usage()[before_search:]
    # cache_miss is an empty miss under cache-only. A live call is neither
    # a hit nor that miss. The monthly file is the bill; this is the claim's share.
    live = sum(1 for item in usage if not item.get("cached") and not item.get("cache_miss"))
    verdict = state.get("verdict")
    if verdict is not None:
        verdict = str(verdict).strip() or None
    return {
        "id": str(row["id"]),
        "claim": str(row["claim"]),
        "label": str(row["label"]),
        "claim_date": str(row["claim_date"]),
        "verdict": verdict,
        "confidence": state.get("confidence"),
        "needs_human": bool(state.get("needs_human")) or out.get("status") == "interrupt",
        "route": str(state.get("route") or ""),
        "searches": len(usage),
        "searches_live": live,
        "llm_calls": len(get_usage()) - before_llm,
        "latency": latency,
        "gate": str(state.get("gate") or ""),
        "thread_id": thread,
        "state": _state_slice(state),
    }


def _checkpoint(thread: str, paths: dict, build_graph) -> dict:
    graph = build_graph(
        paths["checkpoint_path"],
        paths["audit_path"],
        paths["additions_path"],
    )
    snap = graph.get_state({"configurable": {"thread_id": thread}})
    values = getattr(snap, "values", None)
    return values if isinstance(values, dict) else {}


def _arm() -> None:
    """No key, no live search, and a frozen copy of both counters."""
    global _SNAPSHOT
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)
    os.environ["SEARCH_CACHE_ONLY"] = "1"
    _SNAPSHOT = {
        path: path.read_bytes() if path.is_file() else None
        for path in _COUNTER_FILES
    }


def _counters_unchanged() -> None:
    if _SNAPSHOT is None:
        raise SystemExit("counter snapshot missing")
    current = {
        path: path.read_bytes() if path.is_file() else None
        for path in _COUNTER_FILES
    }
    if current != _SNAPSHOT:
        raise SystemExit("COUNTERS CHANGED")
    print("COUNTERS UNCHANGED")


def _tripwire() -> None:
    import misinfo_agent.llm as llm
    import misinfo_agent.search as search

    def _blocked(*_args, **_kwargs):
        raise SystemExit("live call blocked")

    llm._produce = _blocked
    search._real = _blocked


def replay() -> None:
    """Score the stored evidence. Writes the winner into config when the rule says so."""
    _arm()
    try:
        _tripwire()
        _replay_body()
    finally:
        if _SNAPSHOT is not None:
            _counters_unchanged()


def _replay_body() -> None:
    import misinfo_agent.config as config
    from misinfo_agent.gate import gate

    sys.path.insert(0, str(ROOT / "scripts"))
    import gate_study as study

    # Product evidence. The leakage guard is the blocked-eval setting.
    # These claims are dated today, and the live run left the guard off.
    config.EVAL_DROP_AFTER_CLAIM = False
    study._apply(config, {}, None)

    facts = {str(row["id"]): row for row in load_facts()}
    stored = _load_results()
    by_id = {str(row["id"]): row for row in stored}
    if set(by_id) != set(facts):
        missing = sorted(set(facts) - set(by_id))
        extra = sorted(set(by_id) - set(facts))
        raise SystemExit(f"results are not the 16 facts; missing={missing} extra={extra}")
    records = []
    for fact_id, gold in facts.items():
        record = by_id[fact_id]
        if str(record.get("label")) != str(gold["label"]):
            raise SystemExit(f"label drift on {fact_id}")
        state = record.get("state") if isinstance(record.get("state"), dict) else {}
        reuse = state.get("route") == "reuse_prior" or record.get("route") == "reuse_prior"
        published = _published(record) if reuse else None
        records.append({
            "id": fact_id,
            "gold": str(gold["label"]),
            "state": state,
            "reuse": reuse,
            "published": published,
            "live": record,
        })

    table = []
    per_claim: dict[str, dict[str, str]] = {row["id"]: {} for row in records}
    blocked: set[str] = set()
    blocked_ids: dict[str, list[str]] = {}
    for order, (name, flags) in enumerate(study.VARIANTS):
        theta = FACTS_THETA if flags.get("GATE_EVIDENCE_SCORE") else None
        predicted = study._predict(config, gate, records, flags, theta)
        hits = []
        for row in records:
            if row["reuse"]:
                continue
            gated = gate(row["state"])
            if support_from_tier34_only(row["state"], gated):
                hits.append(row["id"])
        if hits:
            blocked.add(name)
            blocked_ids[name] = hits
        for row, label in zip(records, predicted):
            per_claim[row["id"]][name] = label
        packed = study._pack(name, order, theta, records, predicted)
        packed["predicted"] = predicted
        table.append(packed)
    study._apply(config, {}, None)

    winner = choose_facts_winner(table, blocked)
    turned = []
    if winner:
        theta = FACTS_THETA if "G4" in set(winner.split("+")) else None
        turned = study._turn_on(winner, theta)
    section = _render(records, table, per_claim, blocked_ids, winner, turned)
    _write_results_section(section)
    print(section)
    if turned:
        print("FLAG ON: " + ", ".join(f"{name}={value}" for name, value in turned))
    else:
        print("FLAG ON: none. All gate flags stay off.")


def _render(records, table, per_claim, blocked_ids, winner, turned) -> str:
    names = [row["name"] for row in table]
    lines = [
        "### Run",
        "",
        "Product pipeline, fact-check sites allowed, gate flags off. "
        "searches counts cache hits. searches_live is the claim's share of the monthly counter.",
        "",
        "| id | route | searches | searches_live | llm_calls | latency_s | live verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in records:
        live = row["live"]
        verdict = live.get("verdict") or ""
        if live.get("needs_human"):
            verdict = f"{verdict} (needs_human)" if verdict else "needs_human"
        lines.append(
            f"| {row['id']} | {live.get('route') or ''} | {live.get('searches')} | "
            f"{live.get('searches_live')} | {live.get('llm_calls')} | {live.get('latency')} | {verdict} |"
        )
    lines.extend([
        "",
        "### Gate replay",
        "",
        "Zero model calls and zero searches. Theta for G4 and G4+G3 is 2.0. "
        "A reused prior keeps the live verdict on every variant. "
        "The mapping is the live one: a missing refute_kind stays Misleading.",
        "",
        "| id | gold | " + " | ".join(names) + " |",
        "| --- | --- | " + " | ".join("---" for _ in names) + " |",
    ])
    for row in records:
        cells = " | ".join(per_claim[row["id"]][name] for name in names)
        lines.append(f"| {row['id']} | {row['gold']} | {cells} |")
    lines.extend([
        "",
        "| variant | theta | decided | correct 2c | wrong 2c | correct 3c | wrong 3c | True precision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in table:
        theta = "—" if row["theta"] is None else f"{float(row['theta']):.1f}"
        if row["true_precision"] is None:
            precision = "n/a"
        else:
            precision = f"{row['true_precision']:.3f} ({row['true_hit']}/{row['true_pred']})"
        lines.append(
            f"| {row['name']} | {theta} | {row['decided']} | {row['correct2']} | {wrong2(row)} | "
            f"{row['correct3']} | {row['wrong']} | {precision} |"
        )
    lines.append("")
    if blocked_ids:
        for name, ids in blocked_ids.items():
            lines.append(
                f"{name} supported {', '.join(ids)} with no tier 1-2 group, so it cannot be turned on."
            )
    else:
        lines.append("No variant supported a claim from tier 3-4 sources alone.")
    lines.append("")
    if winner and turned:
        flags = ", ".join(f"{name}={value}" for name, value in turned)
        lines.append(f"The rule turns {winner} on: {flags}.")
    else:
        lines.append("No variant beat G0 under the rule. All gate flags stay off.")
    g0 = next(row for row in table if row["name"] == "G0")
    mismatches = []
    for row in records:
        # Compare the card's decided label with G0. A pause has no decided label.
        card = _published(row["live"])
        replay_label = per_claim[row["id"]]["G0"]
        graded = replay_label if replay_label in {"False", "Misleading", "True"} else None
        if card != graded:
            mismatches.append(f"{row['id']} live={card or 'abstain'} G0={replay_label}")
    if mismatches:
        lines.append("G0 replay differs from the live card: " + "; ".join(mismatches) + ".")
    else:
        lines.append("G0 replay matches the live card on every claim.")
    lines.append(
        f"G0 decided {g0['decided']}, correct 2c {g0['correct2']}, wrong 2c {wrong2(g0)}, "
        f"correct 3c {g0['correct3']}, wrong 3c {g0['wrong']}."
    )
    return "\n".join(lines)


def _write_results_section(body: str) -> None:
    """Replace everything after the results mark. The rule above it stays."""
    text = DOC.read_text(encoding="utf-8")
    if HEADING not in text or RULE not in text:
        raise SystemExit("the pre-registered section is missing; replay will not invent it")
    block = RESULTS_MARK + "\n\n" + body.rstrip() + "\n"
    if RESULTS_MARK not in text:
        text = text.rstrip() + "\n\n" + block
    else:
        text = text[: text.index(RESULTS_MARK)] + block
    DOC.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run or replay the established-facts set")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="score stored evidence with the gate variants; no model call and no search",
    )
    args = parser.parse_args(argv)
    if args.replay:
        replay()
    else:
        run_live()


if __name__ == "__main__":
    main()
