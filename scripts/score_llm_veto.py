"""Score the box 6 LLM veto on the 70 labelled pairs. Does not edit config.

Two passes, both at TAU_HIGH, both with the veto on. The cache is turned
on for this process only, so a pair the first pass already asked about is
free the second time. The veto runs only when the rules have left the pair
exact; decide_prior does not call the model otherwise.

A daily cap or a daily 429 stops the pass. The pairs already scored are
still printed. The key is never read or printed here.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_tau_high  # noqa: E402
from misinfo_agent.config import EVAL_DIR, TAU_HIGH  # noqa: E402
import misinfo_agent.config as config  # noqa: E402
import misinfo_agent.prior_match as prior_match  # noqa: E402
from misinfo_agent.llm import LLMQuotaError, calls_today  # noqa: E402

# The sheet was scored against this day. A later "now" would only change
# staleness, which does not move exact versus similar, but the hit should
# match the earlier eval.
TODAY = date(2026, 11, 9)
LABELLED = EVAL_DIR / "tau_high_pairs.csv"
NEGATIVES = EVAL_DIR / "tau_high_negatives.csv"
_VETO_PREFIX = "A careful read found a difference: "


def main() -> None:
    rows = _load(LABELLED) + _load(NEGATIVES)
    # This process only. The defaults in config.py stay off.
    config.LLM_CACHE = True
    prior_match.LLM_CONTEXT_CHECK = True
    print(f"rows {len(rows)}  calls_today {calls_today()}  cache on  veto on")
    _pass(rows, 0.65, "TAU_HIGH 0.85, KEYTERM_SIM 0.65, veto on")
    _pass(rows, 0.45, "TAU_HIGH 0.85, KEYTERM_SIM 0.45, veto on")


def _pass(rows: list[dict], keyterm_sim: float | None, title: str) -> None:
    before = calls_today()
    print(f"\n=== {title} ===")
    print(f"calls_today before {before}")
    done = []
    stopped = None
    for index, row in enumerate(rows, start=1):
        try:
            decision = prior_match.decide_prior(
                row["claim_a"],
                [_hit(row)],
                now=TODAY,
                tau_high=TAU_HIGH,
                keyterm_sim=keyterm_sim,
            )
        except LLMQuotaError as exc:
            stopped = (index, str(exc))
            break
        done.append((row, decision))
    after = calls_today()
    _report(done, before, after, stopped, len(rows))


def _report(done, before: int, after: int, stopped, total: int) -> None:
    labels = [_yes(row) for row, _decision in done]
    exact = [decision["match_type"] == "exact" for _row, decision in done]
    vetoes = []
    for row, decision in done:
        detail = _veto_detail(decision["match_reasons"])
        if detail is not None:
            vetoes.append((row, detail, decision["match_type"]))
    n_exact = sum(exact)
    true_positive = sum(flag and label for flag, label in zip(exact, labels))
    false_exact_rows = [
        row for (row, _decision), flag, label in zip(done, exact, labels) if flag and not label
    ]
    n_yes = sum(labels)
    precision = score_tau_high._rate(true_positive, n_exact)
    recall = score_tau_high._rate(true_positive, n_yes)
    low, high = score_tau_high._wilson(true_positive, n_exact)
    print(
        f"scored {len(done)} of {total}  exacts {n_exact}  "
        f"precision {score_tau_high._fmt(precision)}  "
        f"wilson_low {score_tau_high._fmt(low)}  wilson_high {score_tau_high._fmt(high)}  "
        f"recall {score_tau_high._fmt(recall)}  false_exacts {len(false_exact_rows)}"
    )
    print(f"exacts vetoed {len(vetoes)}")
    goat = [row for row, _decision in done if (row.get("neg_id") or "") == "s1"]
    if goat:
        decision = next(item[1] for item in done if (item[0].get("neg_id") or "") == "s1")
        print(f"goat/cow (s1) match_type {decision['match_type']}")
    else:
        print("goat/cow (s1) not reached")
    y_vetoes = [(row, detail) for row, detail, _kind in vetoes if _yes(row)]
    print(f"y-labelled vetoed {len(y_vetoes)}")
    for row, detail in y_vetoes:
        tag = row.get("pair_id") or row.get("neg_id")
        print(f"  {tag} differing_detail: {detail}")
        print(f"    A: {row['claim_a']}")
        print(f"    B: {row['claim_b']}")
    if false_exact_rows:
        print("false exacts")
        for row in false_exact_rows:
            tag = row.get("neg_id") or row.get("pair_id")
            print(f"  {tag} {row.get('edit') or 'labelled'} cos {row['cos']}")
    print(f"calls_today after {after}  calls_used {after - before}")
    if stopped is not None:
        index, message = stopped
        print(f"aborted at pair {index} of {total}: {message}")


def _load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hit(row: dict) -> dict:
    return {
        "primary_claim": row["claim_b"],
        "dense_cos": float(row["cos"]),
        "date": row.get("date_b") or None,
        "site": row.get("site_b") or None,
    }


def _yes(row: dict) -> bool:
    return row["same_claim"].strip().lower() in {"y", "yes"}


def _veto_detail(reasons: list[str]) -> str | None:
    for reason in reasons:
        if reason.startswith(_VETO_PREFIX):
            body = reason[len(_VETO_PREFIX):]
            if body.endswith("."):
                body = body[:-1]
            return body
    return None


if __name__ == "__main__":
    main()
