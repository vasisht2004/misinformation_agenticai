"""Build data/eval/testset.csv. Seed 42. Does not call a model.

40 rows are archive claims: a 2025–2026 text fact-check whose verdict_norm
is False, Misleading, or True. The input is primary_claim. The label is
that verdict. claim_date is the article's date, which is the day the
eval pretends the claim was circulating.

10 more rows are the true_version of other articles from that same
slice, labelled True. They are other articles so the 40 are not graded
against a sentence taken from their own write-up. The article id stays,
because the memory search has to hide that article and its copies.

The 40 are spread across domains. Each domain with any eligible row is
given a slot before any domain is given a second extra, and a domain
that runs out is skipped. The 10 true rows use the same rule on whatever
is left. The seed only shuffles rows inside a domain. It does not decide
which domain gets an extra slot.

The article body is not read.

Usage:
  python scripts/build_testset.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402

# The slice the report asked for. Changing these changes the study.
SEED = 42
N_LABELLED = 40
N_TRUE = 10
EVAL_YEARS = ("2025", "2026")

# verdict_norm, after case and surrounding space are ignored. Blank and
# Other are not labels the scorer can mark right or wrong, so those
# articles stay out of the 40. They can still donate a true_version.
_LABELS = {
    "false": "False",
    "misleading": "Misleading",
    "true": "True",
}

_FIELDS = ("article_id", "input", "label", "claim_date", "domain", "source")

# A true_version that talks about the rumour, the video, or the image
# is a fact-checker's aside. The eval needs a sentence a person could
# check on its own. "The viral video is misleading" and "this claim is
# false" are the cases this catches. A negated world fact ("Armenia has
# not cancelled the deal") is still a fact, and is kept.
_META_TRUE = re.compile(
    r"\b(this|the|that)\s+claim\b"
    r"|\bviral\s+(video|image|photo|clip|post)\b"
    r"|\b(the|this|that)\s+(video|image|photo|clip|screenshot|footage)\s+is\b"
    r"|\b(is|was|are|were)\s+(misleading|false|fake|fabricated|morphed)\b"
    r"|\bmisrepresent"
    r"|\b(old|unrelated)\s+(video|image|photo|clip|footage)\b",
    re.IGNORECASE,
)


def is_meta_statement(text: str) -> bool:
    """True when the sentence is about the rumour or the media, not a fact of its own."""
    return bool(_META_TRUE.search(text or ""))


def map_label(value) -> str | None:
    """False / Misleading / True, or None when the archive label is not one of those."""
    if _missing(value):
        return None
    return _LABELS.get(str(value).strip().casefold())


def quotas(pool_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Hand out n slots as evenly as the pools allow.

    One pass gives at most one slot to each domain, largest pool first,
    and a domain with nothing left is skipped. The next pass starts
    again from whoever still has the most rows. Ties break by domain
    name, so the seed never decides that a small domain is left out.
    """
    remaining = {domain: int(size) for domain, size in pool_sizes.items() if int(size) > 0}
    assigned = {domain: 0 for domain in remaining}
    left = int(n)
    while left > 0 and any(size > 0 for size in remaining.values()):
        order = sorted(remaining, key=lambda domain: (-remaining[domain], domain))
        gave = False
        for domain in order:
            if left == 0:
                break
            if remaining[domain] <= 0:
                continue
            assigned[domain] += 1
            remaining[domain] -= 1
            left -= 1
            gave = True
        if not gave:
            break
    return assigned


def select_claims(
    records: list[dict],
    *,
    seed: int = SEED,
    n_labelled: int = N_LABELLED,
    n_true: int = N_TRUE,
) -> list[dict]:
    """The test rows, labelled claims first, then the constructed Trues.

    records is the parquet as dicts. Only the columns this function
    reads are required. The article text is not one of them.
    """
    usable = [row for row in records if _in_slice(row)]
    # article_id order, then the seed. A re-saved parquet with the same
    # rows must not pick a different sample just because the row order moved.
    usable.sort(key=lambda row: _text(row.get("article_id")))
    labelled_pool = [row for row in usable if map_label(row.get("verdict_norm")) and _text(row.get("primary_claim"))]
    rng = np.random.default_rng(seed)
    labelled_groups = _shuffle_groups(labelled_pool, rng)
    labelled_take = _take(labelled_groups, quotas(_sizes(labelled_groups), n_labelled))
    chosen = {row["article_id"] for row in labelled_take}
    # Donors are the rest of the same slice, not only the rows whose
    # archive label was False or Misleading. The label on these rows is
    # True because the input is the true_version, not the circulated claim.
    # A meta sentence ("this claim is false", "the old decision is being
    # misrepresented") is not a True label the scorer can defend. Another
    # article from the same domain supplies the row instead.
    donors = [
        row
        for row in usable
        if row["article_id"] not in chosen
        and _text(row.get("true_version"))
        and not is_meta_statement(_text(row.get("true_version")))
    ]
    true_groups = _shuffle_groups(donors, rng)
    true_take = _take(true_groups, quotas(_sizes(true_groups), n_true))
    rows = [_labelled_row(row) for row in _round_robin(labelled_take)]
    rows.extend(_labelled_row(row, source="true_version") for row in _by_domain(true_take))
    return rows


def main() -> None:
    frame = pd.read_parquet(
        config.CLASSIFIED_PATH,
        columns=[
            "article_id",
            "date",
            "is_fact_check",
            "form",
            "domain",
            "verdict_norm",
            "primary_claim",
            "true_version",
        ],
    )
    rows = select_claims(frame.to_dict(orient="records"))
    out = config.EVAL_DIR / "testset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _print_counts(rows, out)


def _print_counts(rows: list[dict], out: Path) -> None:
    print(f"rows {len(rows)}")
    print(f"wrote {out}")
    labels = Counter(row["label"] for row in rows)
    print("labels")
    for label in ("False", "Misleading", "True"):
        print(f"  {label} {labels.get(label, 0)}")
    true_from_archive = sum(1 for row in rows if row["source"] == "true_version")
    print(f"  true_version rows {true_from_archive}")
    print("domains")
    counts = Counter(row["domain"] for row in rows)
    names = [domain for domain in config.DOMAINS if counts.get(domain)]
    names.extend(sorted(set(counts) - set(names)))
    for domain in names:
        print(f"  {domain} {counts[domain]}")


def _in_slice(row: dict) -> bool:
    """Fact-check, text, and a 2025 or 2026 date. Image and video are out.

    The constructed Trues use this same slice. A true_version written
    about a video is a different kind of claim from the 40, and an
    article that is not a fact-check is not in the memory the eval hides.
    """
    if not _is_fact_check(row.get("is_fact_check")):
        return False
    if _text(row.get("form")) != "Text":
        return False
    if not _eval_year(row.get("date")):
        return False
    if not _text(row.get("article_id")) or not _text(row.get("domain")):
        return False
    return True


def _labelled_row(row: dict, source: str = "primary_claim") -> dict:
    if source == "true_version":
        text = _text(row.get("true_version"))
        label = "True"
    else:
        text = _text(row.get("primary_claim"))
        label = map_label(row.get("verdict_norm")) or ""
    return {
        "article_id": _text(row.get("article_id")),
        "input": text,
        "label": label,
        "claim_date": _day(row.get("date")),
        "domain": _text(row.get("domain")),
        "source": source,
    }


def _shuffle_groups(rows: list[dict], rng) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["domain"], []).append(row)
    for domain in sorted(groups):
        order = rng.permutation(len(groups[domain]))
        groups[domain] = [groups[domain][int(index)] for index in order]
    return groups


def _sizes(groups: dict[str, list[dict]]) -> dict[str, int]:
    return {domain: len(rows) for domain, rows in groups.items()}


def _take(groups: dict[str, list[dict]], assigned: dict[str, int]) -> list[dict]:
    taken = []
    for domain in sorted(groups):
        taken.extend(groups[domain][: assigned.get(domain, 0)])
    return taken


def _round_robin(rows: list[dict]) -> list[dict]:
    """Domain name order, one row at a time, so the file does not open with one domain."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["domain"], []).append(row)
    ordered = []
    while any(buckets.values()):
        for domain in sorted(buckets):
            if buckets[domain]:
                ordered.append(buckets[domain].pop(0))
    return ordered


def _by_domain(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["domain"], row["article_id"]))


def _is_fact_check(value) -> bool:
    if _missing(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return bool(value)


def _eval_year(value) -> bool:
    day = _day(value)
    return day.startswith(tuple(year + "-" for year in EVAL_YEARS))


def _day(value) -> str:
    """YYYY-MM-DD. A timestamp's first ten characters are that day."""
    text = _text(value)
    return text[:10]


def _text(value) -> str:
    if _missing(value):
        return ""
    return " ".join(str(value).split())


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
