"""Precision and recall of "exact" on the labelled TAU_HIGH sheet.

Do not run this until same_claim is filled with y or n. It does not search
and it does not write config.py. Each threshold is decide_prior again, so
the detail rule and the key-term rule still downgrade a high cosine.

Positive label: same_claim y. Positive prediction: match_type exact.
Precision is the share of those exacts that you marked as the same claim.
Recall is the share of the pairs you marked y that the rule called exact.
The Wilson interval is on the precision, because that is the quantity a
cutoff changes most and the one with the smaller count (the exacts).
"""

from __future__ import annotations

import csv
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misinfo_agent.config import EVAL_DIR, TAU_HIGH  # noqa: E402
from misinfo_agent.prior_match import decide_prior  # noqa: E402

SHEET = EVAL_DIR / "tau_high_pairs.csv"
# Inclusive. 0.80 is the bottom of the labelled band. 0.95 is the first
# value that would call none of these pairs exact on cosine alone, since
# every pair was drawn from [0.80, 0.95).
TAU_STEPS = [step / 100 for step in range(80, 96)]
# 1.96 is the normal quantile for a 95% two-sided interval.
Z_95 = 1.96


def main() -> None:
    rows = _read(SHEET)
    labels = [_label(row["same_claim"], row["pair_id"]) for row in rows]
    n_same = sum(labels)
    today = date.today()
    print(f"pairs {len(rows)}  same_claim y {n_same}  n {len(rows) - n_same}  today {today.isoformat()}")
    print("tau_high  exacts  precision  wilson_low  wilson_high  recall")
    for high in TAU_STEPS:
        exact = [_is_exact(row, high, today) for row in rows]
        n_exact = sum(exact)
        true_positive = sum(flag and label for flag, label in zip(exact, labels))
        precision = _rate(true_positive, n_exact)
        recall = _rate(true_positive, n_same)
        low, high_ci = _wilson(true_positive, n_exact)
        mark = "  current" if abs(high - TAU_HIGH) < 1e-9 else ""
        print(
            f"{high:.2f}  {n_exact:3d}  {_fmt(precision)}  {_fmt(low)}  {_fmt(high_ci)}  {_fmt(recall)}{mark}"
        )


def _read(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}; build the sheet first")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no pairs")
    return rows


def _label(raw: str, pair_id: str) -> bool:
    text = (raw or "").strip().lower()
    if text in {"y", "yes"}:
        return True
    if text in {"n", "no"}:
        return False
    raise SystemExit(
        f"pair {pair_id} has same_claim {raw!r}. Fill every row with y or n before scoring."
    )


def _is_exact(
    row: dict,
    tau_high: float,
    today: date,
    *,
    rules: str = "new",
    keyterm_sim: float | None = None,
) -> bool:
    """The cosine on the sheet is 3 d.p. The rules still see both texts.

    date and site do not change exact versus similar. They are passed so the
    hit is the same shape decide_prior read when the sheet was written.
    tau_low stays at the config value. Every pair is at or above 0.80, so
    the low cutoff does not move these labels.
    """
    hit = {
        "primary_claim": row["claim_b"],
        "dense_cos": float(row["cos"]),
        "date": row["date_b"] or None,
        "site": row["site_b"] or None,
    }
    decision = decide_prior(
        row["claim_a"],
        [hit],
        now=today,
        tau_high=tau_high,
        keyterm_sim=keyterm_sim,
        rules=rules,
    )
    return decision["match_type"] == "exact"


def _rate(hits: int, total: int) -> float | None:
    if total == 0:
        return None
    return hits / total


def _wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion.

    Centre and half-width use the normal approximation on the inverted
    interval, which stays inside (0, 1) when the proportion is 0 or 1.
    An empty exact set has no precision to bound.
    """
    if total == 0:
        return None, None
    proportion = successes / total
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / total
    centre = (proportion + z2 / (2 * total)) / denom
    margin = Z_95 * math.sqrt(proportion * (1 - proportion) / total + z2 / (4 * total * total)) / denom
    return centre - margin, centre + margin


def _fmt(value: float | None) -> str:
    if value is None:
        return "  n/a"
    return f"{value:7.3f}"


if __name__ == "__main__":
    main()
