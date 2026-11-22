"""Run one claim through the graph and print the result card.

A review pause prints the summary and reads approve, edit, reject, or
more_evidence from the terminal, then resumes the same thread.
RECORD_SUBMISSIONS is forced off: this command is a check, not a reach
counter. A quota error stops the process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402
from misinfo_agent.graph import run_pipeline  # noqa: E402
from misinfo_agent.llm import LLMQuotaError  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    # A check from the terminal must not add a row to the reach counter.
    # The flag is read at ingest time. Setting it here is the whole switch.
    config.RECORD_SUBMISSIONS = False
    args = _parser().parse_args(argv)
    raw, input_type = _input(args)
    try:
        out = run_pipeline(raw, input_type=input_type, caption=args.caption or None)
        while out.get("status") == "interrupt":
            _print_review(out["review"])
            resume = _ask()
            out = run_pipeline(resume=resume, thread_id=out["thread_id"])
    except LLMQuotaError as exc:
        print(f"stopped: {exc}")
        raise SystemExit(2) from exc
        card = out.get("result") or {}
        # A non-blocking run has already finished. The banner is the
        # pause the terminal used to ask for. HUMAN_BLOCKING still uses
        # the prompt above, because that path never reaches here until
        # the person has answered.
        if not config.HUMAN_BLOCKING and card.get("review_status") == "pending":
            print("PROVISIONAL")
            print(card.get("review_reason") or "Pending review.")
            if card.get("alert_pending_approval"):
                print("An alert is waiting for approval and has not been sent.")
        print(json.dumps(card, ensure_ascii=False, indent=2))
    audit = out.get("audit") or {}
    if audit:
        print("AUDIT")
        print(json.dumps(audit, ensure_ascii=False, indent=2))


def _input(args) -> tuple[str, str]:
    if args.text:
        return args.text, "text"
    if args.url:
        return args.url, "url"
    if args.image:
        return args.image, "image"
    raise SystemExit("pass --text, --url, or --image")


def _print_review(summary: dict) -> None:
    print("REVIEW")
    print(f"  claim: {summary.get('claim') or ''}")
    print(f"  verdict: {summary.get('verdict')}")
    print(f"  confidence: {summary.get('confidence')}")
    print(f"  priority: {summary.get('priority') or '-'}")
    print(f"  decision: {summary.get('decision') or '-'}")
    print(f"  why: {summary.get('why') or ''}")
    print("  reasons:")
    for line in summary.get("reasons") or []:
        print(f"    {line}")
    print("  evidence:")
    rows = summary.get("evidence") or []
    if not rows:
        print("    none")
    for row in rows:
        print(
            f"    {row.get('id')}  {row.get('site') or '-'}  tier={row.get('tier')}  "
            f"{row.get('stance')}  {row.get('date') or '-'}"
        )
        print(f"      {row.get('link') or ''}")
        if row.get("quote"):
            print(f"      {row.get('quote')}")
    if summary.get("errors"):
        print(f"  errors: {summary.get('errors')}")


def _ask() -> dict:
    action = input("approve / edit / reject / more_evidence: ").strip().lower()
    payload = {"action": action}
    if action == "edit":
        payload["verdict"] = input("verdict: ").strip()
        payload["correction"] = input("correction: ").strip()
    note = input("note (optional): ").strip()
    if note:
        payload["note"] = note
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check one claim and print the card.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--url")
    source.add_argument("--image")
    parser.add_argument("--caption", default="")
    return parser


if __name__ == "__main__":
    main()
