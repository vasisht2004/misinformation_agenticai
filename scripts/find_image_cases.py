"""List image candidates from the fact-check archive.

Pass a limit to take more than the default 25:

    python scripts/find_image_cases.py 80


Reads only form, manipulation_type, date, year, url, primary_claim, and
verdict. The article body (`content`) is not loaded. A row qualifies
when form is Image, the manipulation type mentions context, old, or
unrelated, the article is from 2025 or 2026, and it has a URL and a
primary claim.

Usage:
    python scripts/find_image_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import misinfo_agent.config as config  # noqa: E402

COLUMNS = [
    "form", "manipulation_type", "date", "year", "url",
    "primary_claim", "verdict", "domain",
]
_WORDS = ("context", "old", "unrelated")


def _window(year, when) -> bool:
    try:
        if int(year) in (2025, 2026):
            return True
    except (TypeError, ValueError):
        pass
    text = str(when or "")
    return text.startswith("2025") or text.startswith("2026")


def candidates(limit: int = 25) -> list[dict]:
    frame = pd.read_parquet(config.CLASSIFIED_PATH, columns=COLUMNS)
    kept = []
    for row in frame.itertuples(index=False):
        if str(row.form or "").casefold() != "image":
            continue
        kind = str(row.manipulation_type or "").casefold()
        if not any(word in kind for word in _WORDS):
            continue
        if not _window(row.year, row.date):
            continue
        url = str(row.url or "").strip()
        claim = " ".join(str(row.primary_claim or "").split())
        if not url or not claim:
            continue
        kept.append(
            {
                "date": str(row.date or "")[:10],
                "url": url,
                "primary_claim": claim,
                "verdict": str(row.verdict or ""),
                "manipulation_type": str(row.manipulation_type or ""),
                "domain": str(row.domain or ""),
            }
        )
    kept.sort(key=lambda item: item["date"], reverse=True)
    return kept[:limit]


def main() -> None:
    limit = 25
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    rows = candidates(limit)
    print(f"{len(rows)} candidates")
    for index, row in enumerate(rows, start=1):
        print(f"{index}. {row['date']}  {row['verdict']}  {row['domain']}  {row['manipulation_type']}")
        print(f"   {row['url']}")
        print(f"   {row['primary_claim']}")


if __name__ == "__main__":
    main()
