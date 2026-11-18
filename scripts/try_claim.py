"""Run one claim through the boxes that exist and print what they returned.

No decision lives here. The positional form is boxes 5, 6, and 7, and
the claim text is used as the claim. --text, --url, and --image run
ingest and extract first and print the style signals and the extract
output, then the same 5, 6, 7 path. Boxes 9, 8+10, 11, and 12 are not called.

Usage:
  python scripts/try_claim.py "claim text" --domain Health [--date YYYY-MM-DD]
      [--allow-factcheck] [--investigate]
  python scripts/try_claim.py --text "..." [--date YYYY-MM-DD]
  python scripts/try_claim.py --url https://... 
  python scripts/try_claim.py --image path [--caption "..."]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # The script is not a package. The repo root is what makes
    # `misinfo_agent` importable when this file is run by path.
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.extract import extract  # noqa: E402
from misinfo_agent.ingest import ingest  # noqa: E402
from misinfo_agent.investigate import investigate  # noqa: E402
from misinfo_agent.llm import LLMQuotaError, calls_today  # noqa: E402
from misinfo_agent.memory import get_memory  # noqa: E402
from misinfo_agent.prior_match import decide_prior  # noqa: E402
from misinfo_agent.search import searches_this_month  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    chosen = [bool(args.text), bool(args.url), bool(args.image)]
    if any(chosen):
        if sum(chosen) != 1 or args.claim:
            _parser().error("pass one of --text, --url, --image, or a positional claim")
        _run_submitted(args)
        return
    if not args.claim or not args.domain:
        _parser().error("a positional claim needs --domain")
    _run_checked(args.claim, args.domain, args)


def _run_submitted(args) -> None:
    """Ingest, extract, then the same 5-6-7 path when the claim is checkable."""
    _print_meters("START")
    if args.text:
        state = ingest(args.text, input_type="text")
    elif args.url:
        state = ingest(args.url, input_type="url")
    else:
        state = ingest(args.image, input_type="image", caption=args.caption or None)
    print("BOX 1-2  ingest")
    print(f"  input_type: {state.get('input_type')}")
    print(f"  checkable_hint: {state.get('checkable_hint')}  {state.get('checkable_reason') or ''}")
    print(f"  style_signals: {state.get('style_signals')}")
    if state.get("input_type") == "image":
        print(f"  ocr_text: {state.get('ocr_text')}")
        print(f"  image_path: {state.get('image_path')}")
    try:
        extracted = extract(state)
    except LLMQuotaError as exc:
        # Ingest finished. Extract did not. Do not start a search.
        print(f"BOX 3-4  stopped: {exc}")
        _print_meters("END")
        raise SystemExit(2) from exc
    state.update(extracted)
    print("BOX 3-4  extract")
    print(f"  checkable: {extracted.get('checkable')}  {extracted.get('not_checkable_reason') or ''}")
    print(f"  claim: {extracted.get('claim') or ''}")
    print(f"  domain: {extracted.get('domain') or ''}")
    print(f"  atomic_claims: {extracted.get('atomic_claims')}")
    print(f"  entities: {extracted.get('entities')}")
    print(f"  red_flags: {extracted.get('red_flags')}")
    print(f"  style_flags: {extracted.get('style_flags')}")
    if extracted.get("extract_notes"):
        print(f"  extract_notes: {extracted.get('extract_notes')}")
    if extracted.get("headline_body_mismatch"):
        print(f"  headline_body_mismatch: {extracted.get('headline_body_mismatch')}")
    if state.get("input_type") == "image":
        print(f"  visual_description: {extracted.get('visual_description')}")
        print(f"  location_clues: {extracted.get('location_clues')}")
        print(f"  time_clues: {extracted.get('time_clues')}")
    if extracted.get("verdict") == "Other" or extracted.get("checkable") is False:
        print(f"  verdict: {extracted.get('verdict')}")
        print("  later boxes skipped")
        _print_meters("END")
        return
    _run_checked(
        extracted.get("claim") or state.get("text") or "",
        extracted.get("domain") or "",
        args,
        started=True,
        atomic=extracted.get("atomic_claims") or [],
        entities=extracted.get("entities") or {},
        input_type=state.get("input_type") or "",
        raw_text=state.get("raw_text") or "",
    )


def _run_checked(
    claim: str,
    domain: str,
    args,
    *,
    started: bool = False,
    atomic: list | None = None,
    entities: dict | None = None,
    input_type: str = "",
    raw_text: str = "",
) -> None:
    if not started:
        _print_meters("START")
    claim_date = args.date

    # date_before is strict. A fact-check from the claim's own day is not
    # something the system could have remembered, which is the filter box 5
    # already applies. The domain is the same label box 5 matches exactly.
    hits = get_memory().search(
        claim,
        date_before=claim_date,
        domains=[domain] if domain else None,
    )
    print("BOX 5  memory")
    shown = hits[:3]
    if not shown:
        print("  no hits")
    for rank, hit in enumerate(shown, start=1):
        print(
            f"  {rank}. {hit.get('site') or '-'}  {hit.get('date') or '-'}  "
            f"cos={float(hit.get('dense_cos') or 0.0):.3f}"
        )
        print(f"     {hit.get('primary_claim') or ''}")

    decision = decide_prior(claim, hits, now=claim_date)
    best = decision.get("best_match") or {}
    print("BOX 6  prior")
    print(f"  match_type: {decision['match_type']}")
    print(f"  route: {decision['route']}")
    print(f"  needs_recheck: {decision['needs_recheck']}")
    if best:
        print(
            f"  best match: {best.get('site') or '-'}  {best.get('date') or '-'}  "
            f"cos={float(best.get('dense_cos') or 0.0):.3f}"
        )
        print(f"     {best.get('primary_claim') or ''}")
    else:
        print("  best match: none")
    print("  match_summary:")
    for sentence in decision.get("match_summary") or []:
        print(f"    {sentence}")

    # reuse_prior is the pipeline's stop. --investigate is the override
    # for a person who wants the web table anyway.
    if decision["route"] == "reuse_prior" and not args.investigate:
        print("  stopping: route is reuse_prior")
        _finish()
        return

    state = {
        "claim": claim,
        "domain": domain,
        "claim_date": claim_date,
        "match_type": decision["match_type"],
        "route": decision["route"],
        "leads": decision["leads"],
        "needs_recheck": decision["needs_recheck"],
        "match_reasons": decision["match_reasons"],
        "match_summary": decision["match_summary"],
        "atomic_claims": atomic or [],
        # Verify and official queries are built from these names and figures.
        # The positional path has none; extract fills them for a submission.
        "entities": entities or {},
        # A URL must not come back as its own evidence. Text and image
        # runs leave input_type empty here, so nothing is excluded.
        "input_type": input_type,
        "raw_text": raw_text,
    }
    # Default is to block fact-check sites. The flag is the opt-out.
    config.BLOCK_FACTCHECK_SITES = not args.allow_factcheck
    try:
        result = investigate(state)
    except LLMQuotaError as exc:
        # Search may have run. Stance did not finish. Do not start another demo.
        print(f"BOX 7  stopped: {exc}")
        _print_meters("END")
        raise SystemExit(2) from exc
    stats = result["investigate_stats"]
    evidence = result["evidence"]
    print("BOX 7  investigate")
    print("  queries:")
    for query in stats.get("queries") or []:
        print(f"    {query}")
    if not evidence:
        print("  evidence: none")
    for row in evidence:
        # display_quote adds the previous sentence when the model's quote
        # is only a few words. The quote field itself is unchanged.
        shown = " ".join(str(row.get("display_quote") or row.get("quote") or "").split())
        print(
            f"  {row['id']}  atomic={row.get('atomic_id') or 'A1'}  "
            f"{row.get('netloc') or '-'}  tier={row['tier']}  "
            f"group={row['group_id']}  {row['stance']}  date={row.get('published_date') or '-'}"
        )
        print(f"     {shown}")
    support_all, support_strong = _groups(evidence, "supports")
    refute_all, refute_strong = _groups(evidence, "refutes")
    print(f"  groups supporting: {support_all} (tier<=2: {support_strong})")
    print(f"  groups refuting: {refute_all} (tier<=2: {refute_strong})")
    by_atomic = stats.get("by_atomic") or {}
    if by_atomic:
        print("  per atomic claim:")
        for aid, block in by_atomic.items():
            print(
                f"    {aid}  groups supporting={block.get('groups_supporting')}  "
                f"groups refuting={block.get('groups_refuting')}  "
                f"supports={block.get('n_supports')}  refutes={block.get('n_refutes')}  "
                f"neutral={block.get('n_neutral')}"
            )
    print(f"  neutral: {stats.get('n_neutral')}")
    print(f"  stance_failed: {stats.get('stance_failed')}")
    print(f"  top-up used: {stats.get('topup_used')}")
    if stats.get("topup_queries"):
        print(f"  top-up queries: {stats['topup_queries']}")
    print(
        f"  searches: {stats.get('searches')}  llm_calls: {stats.get('llm_calls')}  "
        f"seconds: {stats.get('seconds')}"
    )
    if stats.get("notes"):
        print(f"  notes: {stats['notes']}")
    # An empty refute list is the case where the reasons matter. The
    # top passages are the ones the stance call actually read.
    if support_all == 0 and refute_all == 0:
        _print_passage_reasons(stats.get("stance_rows") or [])
    _finish()


def _print_passage_reasons(rows: list[dict]) -> None:
    """The first eight passages and why each atomic claim was labelled."""
    order: list[str] = []
    for row in rows:
        passage_id = str(row.get("passage_id") or "")
        if passage_id and passage_id not in order:
            order.append(passage_id)
    chosen = set(order[:8])
    if not chosen:
        return
    print("  top passages (no refuting evidence):")
    shown = 0
    for row in rows:
        if str(row.get("passage_id") or "") not in chosen:
            continue
        print(
            f"    {row.get('passage_id')}  atomic={row.get('atomic_id') or 'A1'}  "
            f"{row.get('netloc') or '-'}  {row.get('stance')}: {row.get('reason') or ''}"
        )
        shown += 1
    if shown == 0:
        return


def _groups(evidence: list[dict], stance: str) -> tuple[int, int]:
    """How many independent groups take this stance, and how many are tier 1–2."""
    all_groups = set()
    strong = set()
    for row in evidence:
        if row.get("stance") != stance:
            continue
        group = row.get("group_id")
        all_groups.add(group)
        if row.get("tier", 4) <= 2:
            strong.add(group)
    return len(all_groups), len(strong)


def _finish() -> None:
    print("Boxes 9, 8+10, 11, 12 not built yet — no verdict.")
    _print_meters("END")


def _print_meters(label: str) -> None:
    print(f"{label}  calls_today={calls_today()}  searches_this_month={searches_this_month()}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run boxes 5, 6, and 7 on one claim. Does not produce a verdict."
    )
    # Optional so --text / --url / --image can omit it. The positional
    # path still requires it; main() reports that.
    parser.add_argument("claim", nargs="?", help="the claim text, for the 5-6-7 path")
    parser.add_argument("--text", help="a text forward: run ingest and extract, then 5-6-7")
    parser.add_argument("--url", help="an article or post URL")
    parser.add_argument("--image", help="path to an image")
    parser.add_argument("--caption", default="", help="caption typed with --image")
    parser.add_argument(
        "--domain",
        choices=config.DOMAINS,
        help="one of the 11 domain labels (required for a positional claim)",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        type=_iso_date,
        help="YYYY-MM-DD the claim was circulating (default: today)",
    )
    parser.add_argument(
        "--allow-factcheck",
        action="store_true",
        help="keep fact-check sites in the web search (blocked by default)",
    )
    parser.add_argument(
        "--investigate",
        action="store_true",
        help="run box 7 even when box 6 would reuse the prior",
    )
    return parser


def _iso_date(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


if __name__ == "__main__":
    main()
