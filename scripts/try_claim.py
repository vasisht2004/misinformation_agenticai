"""Run one claim through the decision path and print what each box returned.

The positional form starts at box 5. --text, --url, and --image run
ingest and extract first. After a prior that is not an exact fresh
match, the path is box 7 then box 9, and box 7 again while the gate
says search_more and the round budget allows it. Then box 8+10, box 11,
and box 12. An exact fresh prior skips the web and the assess model.

Submissions are not recorded unless --record is passed. A test of this
script must not add a row that box 11 later reads as reach.

Usage:
  python scripts/try_claim.py "claim text" --domain Health [--date YYYY-MM-DD]
      [--allow-factcheck] [--investigate] [--record]
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
from misinfo_agent.assess import assess  # noqa: E402
from misinfo_agent.decide import decide  # noqa: E402
from misinfo_agent.extract import extract  # noqa: E402
from misinfo_agent.gate import gate  # noqa: E402
from misinfo_agent.ingest import ingest  # noqa: E402
from misinfo_agent.investigate import investigate  # noqa: E402
from misinfo_agent.llm import LLMQuotaError, calls_today  # noqa: E402
from misinfo_agent.memory import get_memory  # noqa: E402
from misinfo_agent.prior_match import decide_prior  # noqa: E402
from misinfo_agent.priority import priority  # noqa: E402
from misinfo_agent.search import searches_this_month  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    # Off unless this process is a real forward. Scoring the garlic text
    # must not become another submission that a later reach score counts.
    config.RECORD_SUBMISSIONS = bool(args.record)
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
        base=state,
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
    base: dict | None = None,
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

    # Start from the submission so priority can see style signals, red
    # flags, and how many times this forward was stored. The positional
    # path has none of those; the components then stay at their defaults.
    state = _claim_state(
        base,
        claim=claim,
        domain=domain,
        claim_date=claim_date,
        decision=decision,
        hits=hits,
        atomic=atomic,
        entities=entities,
        input_type=input_type,
        raw_text=raw_text,
    )
    # An exact fresh prior is the pipeline's stop for the web. Assess
    # copies the archive row and does not call the model. --investigate
    # is the override for a person who wants the web table anyway.
    if decision["route"] == "reuse_prior" and not args.investigate:
        print("  stopping the web search: route is reuse_prior")
        _finish(state)
        return
    if args.investigate and state.get("route") == "reuse_prior":
        # Leaving the route as reuse_prior would make assess ignore the
        # pages this flag just asked us to read.
        state["route"] = "investigate"

    # Fact-check sites stay in the search. The blocked eval is the only
    # caller that turns BLOCK_FACTCHECK_SITES on (scripts/run_eval.py).
    # This script used to block them unless --allow-factcheck was passed,
    # which made a manual check a different product from the graph.
    config.BLOCK_FACTCHECK_SITES = False
    _search_until_gate(state)
    _finish(state)


def _claim_state(
    base,
    *,
    claim,
    domain,
    claim_date,
    decision,
    hits,
    atomic,
    entities,
    input_type,
    raw_text,
) -> dict:
    state = dict(base or {})
    best = decision.get("best_match") or {}
    state.update(
        {
            "claim": claim,
            "domain": domain,
            "claim_date": claim_date,
            "match_type": decision["match_type"],
            "route": decision["route"],
            "leads": decision["leads"],
            "needs_recheck": decision["needs_recheck"],
            "match_reasons": decision["match_reasons"],
            "match_summary": decision["match_summary"],
            "atomic_claims": atomic or state.get("atomic_claims") or [],
            # Verify and official queries are built from these names and figures.
            # The positional path has none; extract fills them for a submission.
            "entities": entities or state.get("entities") or {},
            # A URL must not come back as its own evidence. A positional
            # claim leaves input_type empty, so nothing is excluded.
            "input_type": input_type or state.get("input_type") or "",
            "raw_text": raw_text or state.get("raw_text") or "",
            # Box 5 hits. Priority's novelty is 1 minus the top cosine,
            # and its reach reads also_debunked_by from this same list.
            "matches": hits,
            # The one hit box 6 judged closest. Assess copies it on
            # reuse_prior and shows it to the draft on a similar match.
            "best_match": best if isinstance(best, dict) else {},
        }
    )
    return state


def _search_until_gate(state: dict) -> None:
    """Box 7, then box 9, and box 7 again while the gate asks for more.

    The gate returns search_more only when another round is still inside
    the budget, and it has already written that next round into `rounds`.
    Stopping on `rounds < MAX` would skip the round the gate just authorised,
    because the counter is already at the cap of the approval. The hop
    cap is only so a gate bug cannot search forever.
    """
    _one_search(state)
    hops = 0
    while state.get("gate") == "search_more" and hops < config.MAX_SEARCH_ROUNDS:
        hops += 1
        _one_search(state)


def _one_search(state: dict) -> None:
    try:
        result = investigate(state)
    except LLMQuotaError as exc:
        # Search may have run. Stance did not finish. Do not start another demo.
        print(f"BOX 7  stopped: {exc}")
        _print_meters("END")
        raise SystemExit(2) from exc
    fresh = _absorb(state, result)
    print(f"BOX 7  investigate (round {state.get('rounds')})")
    _print_investigate(result.get("investigate_stats") or {}, fresh, state.get("evidence") or [])
    gated = gate(state)
    state.update(gated)
    _print_gate(state)


def _absorb(state: dict, result: dict) -> list:
    """Keep every round's rows. investigate() returns only the new ones.

    The graph would append them with the evidence reducer. This script
    is not the graph, so replacing the list would make round 2's gate
    judge only the pages that round found.
    """
    prior_evidence = list(state.get("evidence") or [])
    prior_passages = list(state.get("passages") or [])
    fresh = list(result.get("evidence") or [])
    old_stats = state.get("investigate_stats") if isinstance(state.get("investigate_stats"), dict) else {}
    state.update(result)
    state["evidence"] = prior_evidence + fresh
    state["passages"] = prior_passages + list(result.get("passages") or [])
    state["investigate_stats"] = _merge_stats(old_stats, result.get("investigate_stats") or {})
    # The gate is what advances the counter. Before the first gate, one
    # finished investigate is round 1. A later round leaves the number
    # the previous gate already wrote, so this does not reset it to 1
    # and accidentally authorise a third search.
    if int(state.get("rounds") or 0) < 1:
        state["rounds"] = 1
    return fresh


def _merge_stats(old: dict, new: dict) -> dict:
    """Keep this round's report, and add the counts a later box still needs.

    quote_unverified is how many quotes we could not check. Round 2's
    dict replaces round 1's, so the first round's count has to be added
    back or confidence would forget them. A stance failure in either
    round stays a failure: a later success does not make the missing
    labels trustworthy.
    """
    if not old:
        return dict(new)
    merged = dict(new)
    merged["quote_unverified"] = int(old.get("quote_unverified") or 0) + int(new.get("quote_unverified") or 0)
    merged["topup_used"] = bool(old.get("topup_used")) or bool(new.get("topup_used"))
    merged["stance_failed"] = bool(old.get("stance_failed")) or bool(new.get("stance_failed"))
    return merged


def _finish(state: dict) -> None:
    """Box 8+10, then priority, then the decision. Quota stops the run."""
    if "gate" not in state:
        # reuse_prior never searched. The gate records that it was skipped
        # so box 12 does not treat a missing gate as a conflict.
        gated = gate(state)
        state.update(gated)
        _print_gate(state)
    try:
        judged = assess(state)
    except LLMQuotaError as exc:
        print(f"BOX 8+10  stopped: {exc}")
        _print_meters("END")
        raise SystemExit(2) from exc
    state.update(judged)
    _print_assess(state)
    scored = priority(state)
    state.update(scored)
    _print_priority(state)
    decided = decide(state)
    state.update(decided)
    _print_decide(state)
    _print_meters("END")


def _print_investigate(stats: dict, fresh: list, full: list) -> None:
    print("  queries:")
    for query in stats.get("queries") or []:
        print(f"    {query}")
    if not fresh:
        print("  new evidence: none")
    for row in fresh:
        # display_quote adds the previous sentence when the model's quote
        # is only a few words. The quote field itself is unchanged.
        shown = " ".join(str(row.get("display_quote") or row.get("quote") or "").split())
        print(
            f"  {row.get('id')}  atomic={row.get('atomic_id') or 'A1'}  "
            f"{row.get('netloc') or '-'}  tier={row.get('tier')}  "
            f"group={row.get('group_id')}  {row.get('stance')}  date={row.get('published_date') or '-'}"
            f"{('  repaired_from=' + row['repaired_from']) if row.get('repaired_from') else ''}"
        )
        print(f"     {shown}")
    # Group counts are the whole table, not this round. Round 2's new
    # rows alone would look thinner than the evidence the gate judged.
    support_all, support_strong = _groups(full, "supports")
    refute_all, refute_strong = _groups(full, "refutes")
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
    # Mismatch is what the repair did not change. repaired is how many
    # neutrals that pass moved. The date window can still clear one after.
    print(f"  reason_label_mismatch: {stats.get('reason_label_mismatch', 0)}")
    print(f"  repaired: {stats.get('repaired', 0)}")
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
    # An empty table is the case where the reasons matter. The top
    # passages are the ones the stance call actually read.
    if support_all == 0 and refute_all == 0:
        _print_passage_reasons(stats.get("stance_rows") or [])


def _print_gate(gated: dict) -> None:
    print("BOX 9  gate")
    print(f"  gate_decision: {gated.get('gate')}")
    print(f"  atomic_status: {gated.get('atomic_status')}")
    print("  reasons:")
    for sentence in gated.get("gate_reasons") or []:
        print(f"    {sentence}")


def _print_assess(state: dict) -> None:
    print("BOX 8+10  assess")
    print(f"  verdict: {state.get('verdict')}")
    print(f"  confidence: {float(state.get('confidence') or 0.0):.4f}")
    print(f"  needs_human: {state.get('needs_human')}")
    caveats = state.get("caveats") or []
    if caveats:
        print("  caveats:")
        for item in caveats:
            print(f"    {item}")
    else:
        print("  caveats: none")
    for item in state.get("atomic_results") or []:
        cites = ", ".join(item.get("citations") or []) or "none"
        print(f"  {item.get('atomic_id')}  refute_kind={item.get('refute_kind')}")
        print(f"    explanation: {item.get('explanation') or ''}")
        print(f"    citations: {cites}")
    print(f"  true_version: {state.get('true_version') or ''}")
    print(f"  correction_message: {state.get('correction_message') or ''}")
    print(f"  what_we_found: {state.get('what_we_found') or ''}")
    print(f"  citations: {', '.join(state.get('citations') or []) or 'none'}")
    _print_archive_citation(state)
    notes = state.get("assess_notes") or []
    if notes:
        print("  assess_notes:")
        for note in notes:
            print(f"    {note}")


def _print_archive_citation(state: dict) -> None:
    """C1 is the archive fact-check reused as the source, not an evidence row."""
    if "C1" not in (state.get("citations") or []):
        return
    best = state.get("best_match") if isinstance(state.get("best_match"), dict) else {}
    print(
        f"  C1  {best.get('site') or '-'}  {best.get('date') or '-'}  {best.get('url') or '-'}"
    )


def _print_priority(state: dict) -> None:
    print("BOX 11  priority")
    print(f"  priority_level: {state.get('priority')}")
    print(f"  priority_score: {float(state.get('priority_score') or 0.0):.3f}")
    print("  reasons:")
    for sentence in state.get("priority_reasons") or []:
        print(f"    {sentence}")


def _print_decide(state: dict) -> None:
    print("BOX 12  decide")
    print(f"  decision: {state.get('decision')}")
    print(f"  p_star: {float(state.get('p_star') or 0.0):.4f}")
    print("  reasons:")
    for sentence in state.get("decision_reasons") or []:
        print(f"    {sentence}")


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
        marks = ""
        if row.get("date_unknown"):
            marks += "  date_unknown"
        if row.get("reason_label_mismatch"):
            marks += "  reason_label_mismatch"
        if row.get("repaired_from"):
            marks += f"  repaired_from={row.get('repaired_from')}"
        print(
            f"    {row.get('passage_id')}  atomic={row.get('atomic_id') or 'A1'}  "
            f"{row.get('netloc') or '-'}  {row.get('stance')}: {row.get('reason') or ''}{marks}"
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


def _print_meters(label: str) -> None:
    print(f"{label}  calls_today={calls_today()}  searches_this_month={searches_this_month()}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the decision path: 5, 6, then 7 and 9 until the gate settles, then 8+10, 11, and 12."
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
        help="accepted for old commands; fact-check sites stay in the search",
    )
    parser.add_argument(
        "--investigate",
        action="store_true",
        help="run box 7 even when box 6 would reuse the prior",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="store this submission in data/submissions.db (off by default, so a test does not inflate reach)",
    )
    return parser


def _iso_date(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


if __name__ == "__main__":
    main()
