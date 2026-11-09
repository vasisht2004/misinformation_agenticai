"""Score the box-6 test set. Prints the numbers. Does not edit config.

Recall is only on the paraphrase rows: did search put the gold article in
the top 1 or the top 5. A near-duplicate that was collapsed into
also_debunked_by counts, because that row is the same fact-check. The list
stores a url, not an article id, so the gold id is resolved through meta.

The threshold grid rescores the cached hits. Each query is embedded once.
The winning pair is printed and left out of config.py: a tuning run must
not change what production calls "exact".
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misinfo_agent.config import (  # noqa: E402
    META_PARQUET_PATH,
    PRIOR_MATCH_FIG_PATH,
    PRIOR_MATCH_TESTSET_PATH,
    TAU_HIGH,
    TAU_LOW,
)
from misinfo_agent.memory import get_memory  # noqa: E402
from misinfo_agent.prior_match import decide_prior  # noqa: E402

LABELS = ["exact", "similar", "none"]
# 9 November 2026, the same clock the live queries use. Expiry is not what
# this figure is about, but decide_prior needs a date to be repeatable.
TODAY = date(2026, 11, 9)
GROUP_ORDER = ["paraphrase", "detail_changed", "related", "novel"]
GROUP_COLOR = {
    "paraphrase": (37, 99, 168),
    "detail_changed": (196, 98, 28),
    "related": (46, 125, 92),
    "novel": (120, 72, 150),
}


def main() -> None:
    rows = _read_rows(PRIOR_MATCH_TESTSET_PATH)
    memory = get_memory()
    urls = _gold_urls()
    cached = []
    for row in rows:
        hits = memory.search(row["query"], k=10)
        cached.append(hits)

    recall1, recall5 = _recall(rows, cached, urls)
    predicted = [
        decide_prior(row["query"], hits, now=TODAY)["match_type"]
        for row, hits in zip(rows, cached)
    ]
    expected = [row["expected_match_type"] for row in rows]
    matrix = confusion_matrix(expected, predicted, labels=LABELS)
    current_f1 = f1_score(expected, predicted, average="macro", labels=LABELS, zero_division=0)
    best = _grid(rows, cached, expected)

    print(f"rows {len(rows)}")
    print(f"Recall@1 {recall1:.3f}")
    print(f"Recall@5 {recall5:.3f}")
    print(f"confusion rows are expected {LABELS}, columns are predicted")
    print(matrix)
    print(f"macro-F1 at TAU_HIGH {TAU_HIGH:.2f} (>=) TAU_LOW {TAU_LOW:.2f} (>=): {current_f1:.3f}")
    print("best 5 threshold pairs (not written to config)")
    for score, high, low in best[:5]:
        print(f"  TAU_HIGH {high:.2f}  TAU_LOW {low:.2f}  macro-F1 {score:.3f}")
    print("novel top-1 dense_cos (manual check)")
    for row, hits in zip(rows, cached):
        if row["group"] != "novel":
            continue
        cosine = float(hits[0]["dense_cos"]) if hits else float("nan")
        print(f"  {cosine:.3f}  {row['query'][:90]}")

    PRIOR_MATCH_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _draw(rows, cached, PRIOR_MATCH_FIG_PATH)
    print(f"figure {PRIOR_MATCH_FIG_PATH}")


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gold_urls() -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(META_PARQUET_PATH, columns=["article_id", "url"])
    return {str(article_id): str(url) for article_id, url in zip(frame["article_id"], frame["url"])}


def _recall(rows, cached, urls) -> tuple[float, float]:
    paraphrase = [(row, hits) for row, hits in zip(rows, cached) if row["group"] == "paraphrase"]
    if not paraphrase:
        return 0.0, 0.0
    hit1 = sum(_found(row["gold_article_id"], hits, urls, 1) for row, hits in paraphrase)
    hit5 = sum(_found(row["gold_article_id"], hits, urls, 5) for row, hits in paraphrase)
    total = len(paraphrase)
    return hit1 / total, hit5 / total


def _found(gold: str, hits: list, urls: dict[str, str], depth: int) -> bool:
    if not gold:
        return False
    gold_url = urls.get(gold)
    for hit in hits[:depth]:
        if hit.get("article_id") == gold:
            return True
        # The collapsed copy is the same check. Credit it at the rank of the
        # row that kept the slot, which is the one a person would open.
        for other in hit.get("also_debunked_by") or []:
            if gold_url and other.get("url") == gold_url:
                return True
    return False


def _grid(rows, cached, expected) -> list[tuple[float, float, float]]:
    scores = []
    for high_i in range(75, 96):
        high = high_i / 100
        for low_i in range(55, 81):
            low = low_i / 100
            if low >= high:
                continue
            predicted = [
                decide_prior(row["query"], hits, now=TODAY, tau_high=high, tau_low=low)["match_type"]
                for row, hits in zip(rows, cached)
            ]
            score = f1_score(expected, predicted, average="macro", labels=LABELS, zero_division=0)
            scores.append((float(score), high, low))
    # Higher F1 first. On a tie, the tighter high threshold, then the tighter low one.
    scores.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    return scores


def _draw(rows, cached, path: Path) -> None:
    width, height = 980, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, right, top, bottom = 70, 40, 36, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.20, 1.00

    def y_of(value: float) -> int:
        clipped = min(y_max, max(y_min, value))
        return int(top + (y_max - clipped) / (y_max - y_min) * plot_h)

    draw.line([(left, top), (left, top + plot_h)], fill=(40, 40, 40))
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(40, 40, 40))
    for tick in (0.20, 0.40, 0.60, 0.70, 0.85, 1.00):
        y = y_of(tick)
        draw.line([(left - 4, y), (left, y)], fill=(40, 40, 40))
        draw.text((8, y - 6), f"{tick:.2f}", fill=(40, 40, 40), font=font)

    # Closed bounds. The line is the cutoff itself: a point on the line is
    # already on the high side of that threshold.
    for value, label in ((TAU_HIGH, "TAU_HIGH 0.85 (>=)"), (TAU_LOW, "TAU_LOW 0.70 (>=)")):
        y = y_of(value)
        draw.line([(left, y), (left + plot_w, y)], fill=(180, 40, 40))
        draw.text((left + 8, y - 14), label, fill=(180, 40, 40), font=font)

    rng = np.random.default_rng(0)
    slot = plot_w / len(GROUP_ORDER)
    for index, group in enumerate(GROUP_ORDER):
        values = [
            float(hits[0]["dense_cos"]) if hits else y_min
            for row, hits in zip(rows, cached)
            if row["group"] == group
        ]
        center = int(left + slot * index + slot / 2)
        draw.text((center - 40, top + plot_h + 16), group, fill=(40, 40, 40), font=font)
        if not values:
            continue
        _box(draw, values, center, y_of)
        color = GROUP_COLOR[group]
        for value in values:
            jitter = int(rng.integers(-14, 15))
            x = center + jitter
            y = y_of(value)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    draw.text((left, 8), "Top-1 dense cosine by test-set group", fill=(20, 20, 20), font=font)
    image.save(path)


def _box(draw: ImageDraw.ImageDraw, values: list[float], center: int, y_of) -> None:
    ordered = np.sort(np.asarray(values, dtype=float))
    q1, median, q3 = np.quantile(ordered, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    low = ordered[ordered >= q1 - 1.5 * iqr].min()
    high = ordered[ordered <= q3 + 1.5 * iqr].max()
    half = 22
    draw.line([(center, y_of(float(low))), (center, y_of(float(high)))], fill=(90, 90, 90))
    draw.rectangle(
        [center - half, y_of(float(q3)), center + half, y_of(float(q1))],
        outline=(60, 60, 60),
    )
    draw.line(
        [(center - half, y_of(float(median))), (center + half, y_of(float(median)))],
        fill=(20, 20, 20),
        width=2,
    )


if __name__ == "__main__":
    main()
