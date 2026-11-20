"""Build data/source_tiers_rsp.csv from Wikipedia's perennial sources list.

The list is the English Wikipedia page "Wikipedia:Reliable sources/Perennial
sources" (WP:RSP). It is a summary of repeated discussions on the reliable
sources noticeboard, not a directory of every publisher. The rows live on
the numbered subpages that the index transcludes.

Licence. The page text is Creative Commons Attribution-ShareAlike 4.0
International (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/
Attribute "Wikipedia contributors", the page
https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources
and the revision this run prints. The same note is in
docs/eval_investigate.md. Share-alike applies to the Wikipedia text. This
script's mapping of a status word onto a tier number is our rule, not
Wikipedia's.

The mapping, decided here so the checker does not re-read the legend:
  generally reliable                         -> tier 2
  no consensus / additional considerations   -> tier 3
  generally unreliable / deprecated /
  blacklisted                                -> tier 4

Our curated lists are applied later, at lookup time, and they win. This
file only records what RSP said.

Usage:
  python scripts/build_source_tiers.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # The script is not a package. The repo root is what makes
    # `misinfo_agent` importable when this file is run by path.
    sys.path.insert(0, str(ROOT))

from misinfo_agent.investigate import curated_tier  # noqa: E402

# Wikimedia asks for a descriptive user agent. A bare urllib default is
# throttled. This names the project and the page we are reading.
_USER_AGENT = (
    "misinfo-agent source-tiers/1.0 "
    "(https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources; "
    "educational research)"
)
_INDEX = "Wikipedia:Reliable sources/Perennial sources"
# The index transcludes /1 through /8. /X is the overflow page the index
# points at. One query fetches every revision, so a second request cannot
# land between two edits and mix dates.
_SUBPAGES = [f"{_INDEX}/{n}" for n in list(range(1, 9)) + ["X"]]

# The template argument is the legend's short code. The words are the
# status the CSV stores, so a later reader does not have to know "gu".
_STATUS = {
    "gr": ("generally reliable", 2),
    "nc": ("no consensus", 3),
    # The legend puts "marginally reliable" in the same bucket as
    # "no consensus, unclear, or additional considerations".
    "m": ("additional considerations", 3),
    "gu": ("generally unreliable", 4),
    "d": ("deprecated", 4),
    "b": ("blacklisted", 4),
    "bl": ("blacklisted", 4),
}

_STATUS_RE = re.compile(r"\{\{WP:RSPSTATUS\|([^|}]+)")
# The uses cell is the last cell and holds the domains. DOTALL because a
# long source wraps the template across lines. Non-greedy so the next
# row's templates are not swallowed.
_USES_RE = re.compile(r"\{\{WP:RSPUSES\s*\|(.*?)\}\}", re.DOTALL)
_DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)+$")

_OUT = ROOT / "data" / "source_tiers_rsp.csv"


def main() -> None:
    pages = _fetch()
    index = pages.get(_INDEX)
    if index is None:
        raise SystemExit("WP:RSP index page was not in the API response")
    rows: dict[str, dict] = {}
    unknown: set[str] = set()
    for title in _SUBPAGES:
        page = pages.get(title)
        if page is None:
            continue
        for domain, status, tier in _rows(page["text"], unknown):
            current = rows.get(domain)
            # Two sources naming one domain: keep the less trusting tier.
            # A deprecated use must not be overwritten by a second listing
            # that called the same host generally reliable.
            if current is not None and current["tier"] >= tier:
                continue
            rows[domain] = {
                "domain": domain,
                "rsp_status": status,
                "tier": tier,
                "as_of": page["timestamp"],
            }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    with _OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "rsp_status", "tier", "as_of"])
        writer.writeheader()
        writer.writerows(ordered)
    added = {"2": 0, "3": 0, "4": 0}
    by_tier = {"2": 0, "3": 0, "4": 0}
    for row in ordered:
        by_tier[str(row["tier"])] += 1
        # curated_tier is None only when our lists do not name the host.
        # Those are the domains this file adds. A host we already classed
        # is in the file for the record and is not counted as added,
        # because lookup will ignore RSP for it.
        if curated_tier(f"https://{row['domain']}/") is None:
            added[str(row["tier"])] += 1
    added_n = sum(added.values())
    print(f"index revision {index['revid']} {index['timestamp']}")
    for title in _SUBPAGES:
        page = pages.get(title)
        if page is not None:
            print(f"subpage {title.rsplit('/', 1)[-1]} revision {page['revid']} {page['timestamp']}")
    if unknown:
        print("unknown status codes:", ", ".join(sorted(unknown)))
    print(f"domains {len(ordered)}")
    print(
        f"RSP tiers  tier2={by_tier['2']}  tier3={by_tier['3']}  tier4={by_tier['4']}"
    )
    print(
        f"RSP added {added_n}  "
        f"tier2={added['2']}  tier3={added['3']}  tier4={added['4']}"
    )
    print("WROTE", _OUT)


def _fetch() -> dict[str, dict]:
    """One API query for the index and every subpage.

    A page that 429s is retried. The revision id and timestamp are what
    `as_of` records, so a cached HTML scrape without a revision is not
    a substitute.
    """
    params = {
        "action": "query",
        "titles": "|".join([_INDEX, *_SUBPAGES]),
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == 3:
                raise
            # The previous exploratory reads already tripped the limit.
            # Wait and ask once more rather than hammering the API.
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"Wikipedia API failed: {last_error}")
    found: dict[str, dict] = {}
    for page in payload.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        revisions = page.get("revisions") or []
        if not revisions:
            continue
        revision = revisions[0]
        slot = (revision.get("slots") or {}).get("main") or {}
        # formatversion 2 calls the field "content". Older responses use "*".
        text = slot.get("content")
        if text is None:
            text = slot.get("*") or ""
        found[page["title"]] = {
            "revid": revision.get("revid"),
            "timestamp": revision.get("timestamp") or "",
            "text": text,
        }
    return found


def _rows(text: str, unknown: set[str]):
    """Yield (domain, status words, tier) for one subpage.

    A row is the wikitext between two `|-` line starts. The status
    template and the uses template are the only cells this mapping needs.
    A source with no domain (a print title, a named person) is skipped:
    there is nothing to look up.
    """
    for chunk in re.split(r"(?m)^\|-", text):
        status_match = _STATUS_RE.search(chunk)
        uses_match = _USES_RE.search(chunk)
        if status_match is None or uses_match is None:
            continue
        code = status_match.group(1).strip().lower()
        mapped = _STATUS.get(code)
        if mapped is None:
            unknown.add(code)
            continue
        status, tier = mapped
        for token in uses_match.group(1).split("|"):
            domain = _domain(token)
            if domain:
                yield domain, status, tier


def _domain(token: str) -> str:
    """A host from an RSPUSES argument, or '' when the argument is not one.

    Named parameters (`link=`) and prose are not domains. A scheme or a
    path is stripped so `https://www.example.com/news` becomes example.com.
    www is stripped because lookup compares the bare host.
    """
    raw = token.strip().strip("'\"")
    if not raw or "=" in raw or " " in raw or "{{" in raw:
        return ""
    raw = raw.lower()
    for prefix in ("https://", "http://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    raw = raw.split("/")[0].split("?")[0].strip(".")
    if raw.startswith("www."):
        raw = raw[4:]
    if not _DOMAIN_RE.fullmatch(raw):
        return ""
    return raw


if __name__ == "__main__":
    main()
