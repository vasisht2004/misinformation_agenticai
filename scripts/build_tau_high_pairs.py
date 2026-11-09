"""Pairs for a human label of TAU_HIGH. Does not edit config.py.

The grid in stage D could not see where 0.85 should sit: picture claims are
capped at similar, and a one-detail edit is downgraded by the detail rule, so
the threshold itself never flipped a label. These pairs are text claims whose
nearest other fact-check lands in the band where that cutoff is a real choice.

Each cosine is one row of E dotted with one stored vector. The full matrix
product would be 52,099 squared floats, about 11 GB, which this box does not
build. The neighbour search is the same idea as search(), one query at a time.

The sheet's same_claim column is left empty. scripts/score_tau_high.py reads
it after it has been filled and is not run from here.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misinfo_agent.config import EVAL_DIR  # noqa: E402
from misinfo_agent.memory import ClaimMemory, _iso_date, _text  # noqa: E402
from misinfo_agent.prior_match import decide_prior  # noqa: E402

# Closed on the left, open on the right, so 0.850 belongs to the middle bin
# and 0.950 is outside the study. 0.95 is the duplicate cutoff (DUP_COS):
# pairs that close are already collapsed in search and do not test TAU_HIGH.
BINS = ((0.80, 0.85), (0.85, 0.90), (0.90, 0.95))
PER_BIN = 15
SAMPLE = 400
SEED = 42
OUT = EVAL_DIR / "tau_high_pairs.csv"
COLUMNS = [
    "pair_id",
    "cos",
    "claim_a",
    "date_a",
    "site_a",
    "claim_b",
    "date_b",
    "site_b",
    "predicted",
    "match_summary",
    "same_claim",
]


def main() -> None:
    memory = ClaimMemory.load()
    today = date.today()
    sampled = _sample_text_rows(memory)
    candidates = [_nearest_in_band(memory, int(row)) for row in sampled]
    candidates = [item for item in candidates if item is not None and _shown_stays_in_bin(item["cos"])]
    chosen = _take_bins(candidates)
    rows = [_row(memory, pair_id, item, today) for pair_id, item in enumerate(chosen, start=1)]
    _check_rounding(rows, chosen)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _report(candidates, rows, today)


def _sample_text_rows(memory: ClaimMemory) -> np.ndarray:
    """400 claims that are not pictures, in seed-42 order.

    A picture claim cannot be exact on text, so it never votes on TAU_HIGH.
    Blank claims are dropped before the draw. The neighbour may still be an
    image row: that is the nearest other article, whatever form it has.
    """
    form = memory.meta["form"].fillna("").astype(str)
    claim = memory.meta["primary_claim"].fillna("").astype(str).str.strip()
    eligible = np.flatnonzero((~form.isin(["Image", "Video"]) & claim.ne("")).to_numpy())
    if len(eligible) < SAMPLE:
        raise SystemExit(f"only {len(eligible)} text claims, need {SAMPLE}")
    rng = np.random.default_rng(SEED)
    return rng.choice(eligible, size=SAMPLE, replace=False)


def _nearest_in_band(memory: ClaimMemory, row: int) -> dict | None:
    """Highest cosine to a different row inside [0.80, 0.95).

    E @ e_i is one column of the gram matrix. Nothing else is stored, and
    the row is excluded so a claim is not paired with itself at cosine 1.
    A neighbour at 0.95 or above is a near-duplicate, not a threshold case,
    so it is masked out and the next article in the band wins.
    """
    vector = np.ascontiguousarray(memory.E[row])
    scores = memory.E @ vector
    scores[row] = -np.inf
    scores[(scores < 0.80) | (scores >= 0.95)] = -np.inf
    other = int(np.argmax(scores))
    cosine = float(scores[other])
    if not np.isfinite(cosine):
        return None
    return {"row_a": row, "row_b": other, "cos": cosine}


def _shown_stays_in_bin(cosine: float) -> bool:
    """3 d.p. is what the sheet and the scorer compare to each cutoff.

    0.8496 is in the low bin and prints as 0.850, which the scorer would
    treat as the middle bin. Those claims are skipped. The next sampled
    claim supplies the pair, and the one that is kept is still that claim's
    nearest neighbour inside the band.
    """
    return _bin(cosine) == _bin(float(f"{cosine:.3f}"))


def _bin(cosine: float) -> tuple[float, float] | None:
    for low, high in BINS:
        if low <= cosine < high:
            return (low, high)
    return None


def _take_bins(candidates: list[dict]) -> list[dict]:
    """15 pairs in each bin. An article is used at most once.

    Candidates stay in sample order, which is the seed-42 order. Bins take
    turns, so the first bin cannot spend every shared neighbour before the
    others have had a look. Within a turn the earliest remaining candidate
    wins. That is the whole selection; there is no second draw.
    """
    queues = {band: deque() for band in BINS}
    for item in candidates:
        band = _bin(item["cos"])
        if band is not None:
            queues[band].append(item)
    kept = {band: [] for band in BINS}
    used: set[int] = set()
    while any(len(kept[band]) < PER_BIN for band in BINS):
        progressed = False
        for band in BINS:
            if len(kept[band]) >= PER_BIN:
                continue
            while queues[band]:
                item = queues[band].popleft()
                if item["row_a"] in used or item["row_b"] in used:
                    continue
                used.add(item["row_a"])
                used.add(item["row_b"])
                kept[band].append(item)
                progressed = True
                break
        if not progressed:
            short = {f"[{low:.2f},{high:.2f})": len(kept[(low, high)]) for low, high in BINS}
            raise SystemExit(f"not enough disjoint pairs in the 400: {short}")
    chosen = []
    for band in BINS:
        # High cosine first inside the bin, so the rows near a cutoff sit together.
        chosen.extend(sorted(kept[band], key=lambda item: -item["cos"]))
    return chosen


def _row(memory: ClaimMemory, pair_id: int, item: dict, today: date) -> dict:
    """decide_prior sees claim_b the way search() would hand it over.

    dense_cos is the stored dot product, not a fresh encoding of the text.
    Rank and rrf are the single-hit values: this list has one result, so it
    is rank 1 on the dense list and was not produced by BM25 fusion.
    """
    hit = _hit(memory, item["row_b"], item["cos"])
    claim = _text(memory.meta.iloc[item["row_a"]]["primary_claim"]) or ""
    decision = decide_prior(claim, [hit], now=today)
    summary = " ".join(decision["match_summary"])
    left = memory.meta.iloc[item["row_a"]]
    return {
        "pair_id": pair_id,
        "cos": f"{item['cos']:.3f}",
        "claim_a": claim,
        "date_a": _iso_date(memory.dates[item["row_a"]]) or "",
        "site_a": _text(left["site"]) or "",
        "claim_b": hit["primary_claim"] or "",
        "date_b": hit["date"] or "",
        "site_b": hit["site"] or "",
        "predicted": decision["match_type"],
        "match_summary": summary,
        "same_claim": "",
    }


def _hit(memory: ClaimMemory, row: int, cosine: float) -> dict:
    record = memory.meta.iloc[row]
    return {
        "article_id": memory.ids[row],
        "site": _text(record["site"]),
        "date": _iso_date(memory.dates[row]),
        "primary_claim": _text(record["primary_claim"]),
        "true_version": _text(record["true_version"]),
        "verdict": _text(record["verdict"]),
        "verdict_norm": _text(record["verdict_norm"]),
        "url": _text(record["url"]),
        "domain": _text(record["domain"]),
        "manipulation_type": _text(record["manipulation_type"]),
        "form": _text(record["form"]),
        "dense_cos": float(cosine),
        "dense_rank": 1,
        "bm25_rank": None,
        "rrf": 0.0,
        "also_debunked_by": [],
    }


def _check_rounding(rows: list[dict], chosen: list[dict]) -> None:
    """3 d.p. can step across 0.850 or 0.900. The scorer only sees the sheet."""
    crossed = [
        row["pair_id"]
        for row, item in zip(rows, chosen)
        if _bin(item["cos"]) != _bin(float(row["cos"]))
    ]
    if crossed:
        raise SystemExit(f"displayed cosine leaves its bin for pair_id {crossed}")


def _report(candidates: list[dict], rows: list[dict], today: date) -> None:
    found = defaultdict(int)
    for item in candidates:
        found[_bin(item["cos"])] += 1
    print(f"today {today.isoformat()}")
    print(f"candidates with a neighbour in band: {len(candidates)}")
    for low, high in BINS:
        band = (low, high)
        print(f"  bin [{low:.2f},{high:.2f}) candidates {found[band]}")
    print(f"wrote {len(rows)} pairs to {OUT}")


if __name__ == "__main__":
    main()
