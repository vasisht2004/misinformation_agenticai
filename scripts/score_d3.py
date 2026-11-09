"""Old rules against the D3 rules, on the labelled pairs plus the hard negatives.

Does not edit TAU_HIGH or TAU_LOW. The key-term floor is a parameter of
decide_prior. This script prints the grid and the comparison; it does not
write the chosen floor back anywhere.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misinfo_agent.config import EVAL_DIR, TAU_HIGH  # noqa: E402
from misinfo_agent.prior_match import KEYTERM_SIM, decide_prior  # noqa: E402
import score_tau_high  # noqa: E402

_fmt = score_tau_high._fmt
_rate = score_tau_high._rate
_wilson = score_tau_high._wilson

TODAY = date(2026, 11, 9)
TAU_STEPS = [step / 100 for step in range(80, 96)]
SIM_STEPS = [step / 100 for step in range(40, 75, 5)]
LABELLED = EVAL_DIR / "tau_high_pairs.csv"
NEGATIVES = EVAL_DIR / "tau_high_negatives.csv"


def main() -> None:
    rows = _load(LABELLED) + _load(NEGATIVES)
    labels = [_yn(row) for row in rows]
    print(f"rows {len(rows)}  y {sum(labels)}  n {len(rows) - sum(labels)}")
    print("KEYTERM_SIM grid at TAU_HIGH 0.85. lemon_exact is the cancer-sentence guard.")
    print("sim  n_exact  precision  wilson_low  wilson_high  recall  false_exacts  lemon_exact  f1")
    scored = []
    for sim in SIM_STEPS:
        stats = _stats(rows, labels, TAU_HIGH, "new", sim)
        lemon = _lemon_exact(sim)
        f1 = _f1(stats["precision"], stats["recall"])
        scored.append((sim, stats, lemon, f1))
        print(
            f"{sim:.2f}  {stats['n_exact']:3d}  {_fmt(stats['precision'])}  {_fmt(stats['wilson'][0])}  "
            f"{_fmt(stats['wilson'][1])}  {_fmt(stats['recall'])}  {stats['false_exacts']:3d}  "
            f"{'yes' if lemon else 'no':>3}  {_fmt(f1)}"
        )
    eligible = [item for item in scored if not item[2]]
    pool = eligible or scored
    # Highest F1, then fewer false exacts, then higher precision.
    best_sim, best_stats, best_lemon, best_f1 = max(
        pool,
        key=lambda item: (
            item[3] or -1,
            -item[1]["false_exacts"],
            item[1]["precision"] or -1,
        ),
    )
    print(
        f"best KEYTERM_SIM {best_sim:.2f}  f1 {_fmt(best_f1)}  "
        f"lemon_exact {'yes' if best_lemon else 'no'}  code default {KEYTERM_SIM:.2f}"
    )
    for rules, sim in (("old", None), ("new", best_sim)):
        print(f"\n{rules} rules" + (f"  KEYTERM_SIM {sim:.2f}" if sim is not None else ""))
        print("tau_high  n_exact  precision  wilson_low  wilson_high  recall  false_exacts")
        for high in TAU_STEPS:
            stats = _stats(rows, labels, high, rules, sim)
            mark = "  current" if abs(high - TAU_HIGH) < 1e-9 else ""
            print(
                f"{high:.2f}  {stats['n_exact']:3d}  {_fmt(stats['precision'])}  {_fmt(stats['wilson'][0])}  "
                f"{_fmt(stats['wilson'][1])}  {_fmt(stats['recall'])}  {stats['false_exacts']:3d}{mark}"
            )
    print("\nremaining at TAU_HIGH 0.85, new rules, KEYTERM_SIM "
          f"{best_sim:.2f}")
    _list_errors(rows, best_sim)


def _load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _yn(row: dict) -> bool:
    return row["same_claim"].strip().lower() == "y"


def _hit(row: dict) -> dict:
    return {
        "primary_claim": row["claim_b"],
        "dense_cos": float(row["cos"]),
        "date": row.get("date_b") or None,
        "site": row.get("site_b") or None,
    }


def _stats(rows, labels, tau_high: float, rules: str, keyterm_sim: float | None) -> dict:
    exact = []
    for row in rows:
        decision = decide_prior(
            row["claim_a"],
            [_hit(row)],
            now=TODAY,
            tau_high=tau_high,
            keyterm_sim=keyterm_sim,
            rules=rules,
        )
        exact.append(decision["match_type"] == "exact")
    n_exact = sum(exact)
    true_positive = sum(flag and label for flag, label in zip(exact, labels))
    false_exacts = sum(flag and not label for flag, label in zip(exact, labels))
    precision = _rate(true_positive, n_exact)
    recall = _rate(true_positive, sum(labels))
    return {
        "n_exact": n_exact,
        "precision": precision,
        "recall": recall,
        "wilson": _wilson(true_positive, n_exact),
        "false_exacts": false_exacts,
    }


def _lemon_exact(sim: float) -> bool:
    decision = decide_prior(
        "Drinking hot lemon water kills coronavirus",
        [{
            "primary_claim": "Drinking hot lemon water kills cancer cells",
            "dense_cos": 0.95,
            "date": "2026-06-01",
            "site": "Alt News",
        }],
        now=TODAY,
        keyterm_sim=sim,
    )
    return decision["match_type"] == "exact"


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _list_errors(rows, sim: float) -> None:
    false_exacts = []
    false_down = []
    for row in rows:
        decision = decide_prior(
            row["claim_a"],
            [_hit(row)],
            now=TODAY,
            tau_high=TAU_HIGH,
            keyterm_sim=sim,
        )
        label = row["same_claim"].strip().lower()
        tag = row.get("neg_id") or row.get("pair_id")
        kind = row.get("edit") or "labelled"
        if decision["match_type"] == "exact" and label == "n":
            false_exacts.append((tag, kind, row))
        if decision["match_type"] == "similar" and label == "y":
            false_down.append((tag, kind, row, decision["match_reasons"][1:-1]))
    print("false exacts")
    if not false_exacts:
        print("  none")
    for tag, kind, row in false_exacts:
        print(f"  {tag} {kind} cos {row['cos']}")
        print(f"    A: {row['claim_a']}")
        print(f"    B: {row['claim_b']}")
        print("    no detail, community, or key-term downgrade")
    print("false downgrades")
    if not false_down:
        print("  none")
    for tag, kind, row, reasons in false_down:
        print(f"  {tag} {kind} cos {row['cos']}")
        if float(row["cos"]) < TAU_HIGH and not reasons:
            print("    cosine is under 0.85, so the pair is similar before a detail rule")
        for reason in reasons:
            print(f"    {reason}")


if __name__ == "__main__":
    main()
