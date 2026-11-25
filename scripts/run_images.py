"""Run event photographs through the pipeline, blocked then allowed.

Each row of data/eval/images_testset.csv is:

    file,caption,label,claim_date

file is a path relative to the repo root, or absolute. caption may be
empty. label is the fact-check label, kept for scoring and not shown to
the pipeline. claim_date is YYYY-MM-DD.

A row already present for that file and mode is skipped, so a stopped
run can be started again. The run writes data/eval/results_images.jsonl.
Vision and Tavily counters are printed before the first image and after
the last. This script is the live runner. Tests do not call it.

Usage:
    python scripts/run_images.py
    python scripts/run_images.py --limit 2
    python scripts/run_images.py --no-text-search
    python scripts/run_images.py --only data/eval/images/smoke.jpg --no-text-search --sites blocked
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.graph import run_pipeline  # noqa: E402
from misinfo_agent.search import searches_this_month  # noqa: E402
from misinfo_agent.vision import calls_this_month  # noqa: E402

CSV_PATH = config.EVAL_DIR / "images_testset.csv"
OUT_PATH = config.EVAL_DIR / "results_images.jsonl"


# Kept alive for the process. A TemporaryDirectory that is not referenced
# is deleted, and the run would then be writing into a missing file.
_MEMORY_DIR: tempfile.TemporaryDirectory | None = None


def use_fresh_image_memory() -> Path:
    """A private image-memory DB for this process.

    The real file is the product memory. An eval that read it would
    repeat a verdict from an earlier test instead of checking the
    photograph. One directory for the whole run, so a later row still
    cannot see a verdict stored by a previous invocation.
    """
    global _MEMORY_DIR
    _MEMORY_DIR = tempfile.TemporaryDirectory(prefix="image-memory-eval-")
    path = Path(_MEMORY_DIR.name) / "image_memory.db"
    config.IMAGE_MEMORY_DB = path
    return path


def apply_mode(sites: str) -> None:
    """The same honesty settings as the text eval, for this process."""
    config.BLOCK_FACTCHECK_SITES = sites != "allowed"
    config.EVAL_DROP_AFTER_CLAIM = True
    config.RECORD_SUBMISSIONS = False
    config.LLM_CACHE = True


def _done(path: Path) -> set[tuple[str, str]]:
    found = set()
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("file") and row.get("mode"):
            found.add((str(row["file"]), str(row["mode"])))
    return found


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return ROOT / path


def _one(row: dict, mode: str) -> dict:
    file_name = (row.get("file") or "").strip()
    caption = row.get("caption") or ""
    claim_date = (row.get("claim_date") or "").strip() or None
    image = _resolve(file_name)
    record = {
        "file": file_name,
        "caption": caption,
        "label": row.get("label") or "",
        "claim_date": claim_date or "",
        "mode": mode,
    }
    if not image.is_file():
        record["error"] = "missing file"
        record["ok"] = False
        return record
    try:
        outcome = run_pipeline(
            str(image),
            input_type="image",
            caption=caption,
            eval_claim_date=claim_date,
            eval_sites=mode,
        )
    except Exception as exc:
        # The type only. A message can carry a URL or, worse, a key.
        record["error"] = type(exc).__name__
        record["ok"] = False
        return record
    card = outcome.get("result") if isinstance(outcome, dict) else {}
    if not isinstance(card, dict):
        card = {}
    provenance = card.get("image_provenance") if isinstance(card.get("image_provenance"), dict) else {}
    record.update(
        {
            "ok": True,
            "verdict": card.get("verdict"),
            "vision_status": provenance.get("vision_status"),
            "n_matches": len(provenance.get("matches") or []),
            "first_seen": provenance.get("first_seen"),
            "photo_lean": card.get("photo_lean"),
            "search_status": card.get("search_status"),
            "thread_id": outcome.get("thread_id") if isinstance(outcome, dict) else "",
        }
    )
    return record


def _stored_name(path: Path) -> str:
    """The path written on the result row.

    A file inside the repo is stored the way the CSV stores paths, relative
    to the repo root. An absolute path outside the repo stays absolute:
    rewriting it would point the next run at a file that is not there.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _same_file(file_name: str, only: str, wanted: Path) -> bool:
    """True when this CSV path is the file --only named.

    The CSV value and the flag are compared after resolving relative
    paths, so a relative flag matches the column. A bare filename matches
    the basename: `img02.jpg` is that row. A path that contains a
    directory does not match a different file with the same basename.
    """
    name = (file_name or "").strip()
    if not name:
        return False
    if name == only:
        return True
    resolved = _resolve(name)
    if resolved == wanted:
        return True
    bare = "/" not in only and "\\" not in only
    return bool(bare and resolved.name == Path(only).name)


def _same_bytes(path: Path, other: Path) -> bool:
    try:
        left = path.read_bytes()
        right = other.read_bytes()
    except OSError:
        return False
    if len(left) != len(right):
        return False
    return hashlib.sha256(left).digest() == hashlib.sha256(right).digest()


def _row_with_same_bytes(rows: list[dict], wanted: Path) -> dict | None:
    """The first CSV image that is the same file under another name.

    smoke.jpg is a copy of a testset photo so a trial does not mark that
    photo's CSV row done. The caption is the claim as it was shared. A
    silent copy would be a different check. The first matching row wins;
    the set is not supposed to contain two copies.
    """
    for row in rows:
        file_name = (row.get("file") or "").strip()
        if not file_name:
            continue
        if _same_bytes(wanted, _resolve(file_name)):
            return row
    return None


def select_rows(rows: list[dict], only: str) -> list[dict]:
    """Keep the file --only named. An empty flag keeps the CSV order.

    A path that is not a CSV row is still one image when the file exists.
    Caption, label, and claim_date are copied from a byte-identical CSV
    image, and otherwise left blank. They are not invented. The result
    row keeps the path that was asked for, so the real CSV file is not
    recorded as already run.
    """
    text = (only or "").strip()
    if not text:
        return rows
    wanted = _resolve(text)
    matched = [row for row in rows if _same_file(row.get("file") or "", text, wanted)]
    if matched:
        return matched
    if not wanted.is_file():
        return []
    source = _row_with_same_bytes(rows, wanted)
    stored = _stored_name(wanted)
    if source is None:
        return [{"file": stored, "caption": "", "label": "", "claim_date": ""}]
    return [
        {
            "file": stored,
            "caption": source.get("caption") or "",
            "label": source.get("label") or "",
            "claim_date": source.get("claim_date") or "",
        }
    ]


def prepare_rows(rows: list[dict], only: str, limit: int) -> list[dict]:
    """--only, then --limit.

    Limit means the first N rows this invocation will execute. Applying
    it first would drop a later file the flag had named.
    """
    chosen = select_rows(rows, only)
    if limit > 0:
        return chosen[:limit]
    return chosen


def modes_for(sites: str) -> tuple[str, ...]:
    """blocked, allowed, or the usual blocked-then-allowed pair.

    The allowed half turns the fact-check block off and can call Tavily
    for that budget. A run that must stay blocked has to say so. The
    default stays both, which is what a full image eval runs.
    """
    if sites == "allowed":
        return ("allowed",)
    if sites == "blocked":
        return ("blocked",)
    return ("blocked", "allowed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the image eval, blocked then allowed.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many rows. 0 means all.")
    parser.add_argument(
        "--no-text-search",
        action="store_true",
        help="Provenance only. Do not spend Tavily on the caption.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Run this file only. A CSV path, a basename, or another image file.",
    )
    parser.add_argument(
        "--sites",
        choices=("blocked", "allowed", "both"),
        default="both",
        help="blocked excludes fact-check sites. both runs blocked, then allowed.",
    )
    return parser


def apply_run_options(*, no_text_search: bool) -> None:
    """The eval switch for this process. Provenance still runs."""
    if no_text_search:
        config.IMAGE_TEXT_SEARCH = False


def _checkpoint_state(thread_id: str) -> dict:
    """The state the run left on the checkpointer.

    run_pipeline returns the card, not gate_reasons or refute_kind. Those
    stay on the thread. A missing thread is an empty dict: the print
    still runs, and it does not invent a verdict.
    """
    if not thread_id:
        return {}
    from misinfo_agent.graph import build_graph

    snap = build_graph().get_state({"configurable": {"thread_id": thread_id}})
    values = getattr(snap, "values", None)
    return values if isinstance(values, dict) else {}


def _core_refute_kind(state: dict) -> str:
    """The core claim's refute_kind. A side claim does not rename the verdict."""
    claims = {}
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("id"):
            claims[str(item["id"])] = item
    fallback = ""
    for item in state.get("atomic_results") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("refute_kind") or "")
        if fallback == "":
            fallback = kind
        claim = claims.get(str(item.get("atomic_id") or ""))
        if isinstance(claim, dict) and claim.get("core"):
            return kind
    return fallback


def _print_detail(state: dict) -> None:
    """The fields a one-image smoke run is read for. No page body, no key."""
    provenance = state.get("image_provenance") if isinstance(state.get("image_provenance"), dict) else {}
    matches = [row for row in (provenance.get("matches") or []) if isinstance(row, dict)]
    counts = {"full": 0, "partial": 0, "page": 0}
    for match in matches:
        kind = str(match.get("match") or "page")
        if kind not in counts:
            kind = "page"
        counts[kind] += 1
    seen = provenance.get("first_seen") if isinstance(provenance.get("first_seen"), dict) else {}
    print(f"vision_status {provenance.get('vision_status')}", flush=True)
    print(f"matches full={counts['full']} partial={counts['partial']} page={counts['page']}", flush=True)
    print(
        f"first_seen date={seen.get('date')} source={seen.get('date_source')} "
        f"host={seen.get('host')} tier={seen.get('tier')}",
        flush=True,
    )
    wayback_n = sum(1 for match in matches if match.get("date_source") == "wayback")
    print(f"wayback_dates {wayback_n}", flush=True)
    evidence = [
        row
        for row in (state.get("evidence") or [])
        if isinstance(row, dict) and row.get("provenance")
    ]
    print(f"provenance_evidence {len(evidence)}", flush=True)
    for row in evidence:
        host = row.get("netloc") or row.get("site") or ""
        quote = " ".join(str(row.get("quote") or row.get("display_quote") or "").split())
        date = row.get("published_date") or row.get("date") or ""
        print(
            f"  host={host} tier={row.get('tier')} date={date} "
            f"stance={row.get('stance')} quote={quote}",
            flush=True,
        )
    print("image_consistency " + json.dumps(state.get("image_consistency"), ensure_ascii=False), flush=True)
    from misinfo_agent.output import photo_lean

    lean = photo_lean(state) or {}
    print(f"n_dated_earlier {lean.get('n_dated_earlier')}", flush=True)
    print(f"verdict {state.get('verdict')}", flush=True)
    print(f"refute_kind {_core_refute_kind(state)}", flush=True)
    print("gate_reasons " + json.dumps(list(state.get("gate_reasons") or []), ensure_ascii=False), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    apply_run_options(no_text_search=args.no_text_search)
    if not CSV_PATH.is_file():
        raise SystemExit(f"missing {CSV_PATH}")
    rows = prepare_rows(_rows(CSV_PATH), args.only, args.limit)
    done = _done(OUT_PATH)
    print(f"rows {len(rows)}", flush=True)
    print(f"vision_calls_before {calls_this_month()}", flush=True)
    print(f"tavily_calls_before {searches_this_month()}", flush=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Mode first, and a new image-memory file for each mode. The allowed
    # pass must not reuse the verdict the blocked pass just stored for
    # the same photograph: that would make the two columns the same
    # decision. A later image in one mode still cannot see another run's
    # database. use_fresh_image_memory replaces the path for this process.
    for mode in modes_for(args.sites):
        use_fresh_image_memory()
        apply_mode(mode)
        for row in rows:
            file_name = (row.get("file") or "").strip()
            if not file_name:
                continue
            if (file_name, mode) in done:
                print(f"skip {mode} {file_name}", flush=True)
                continue
            record = _one(row, mode)
            with OUT_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            done.add((file_name, mode))
            print(
                f"{mode} {file_name} verdict={record.get('verdict')} vision={record.get('vision_status')}",
                flush=True,
            )
            if record.get("ok"):
                _print_detail(_checkpoint_state(str(record.get("thread_id") or "")))
    print(f"vision_calls_after {calls_this_month()}", flush=True)
    print(f"tavily_calls_after {searches_this_month()}", flush=True)


if __name__ == "__main__":
    main()
