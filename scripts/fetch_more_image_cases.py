"""A second sheet of out-of-context photographs. No model, no Vision, no Tavily.

scripts/find_image_cases.py 80 is the pool. Most of that pool is a stock
medical illustration with a health claim, which is the wrong test of an
event photo. This script keeps the rows about a thing that happened in a
place: a flood, a protest, a politician, a picture someone says is of X.
It drops firstcheck.in, the Health domain, and claims about food, diet,
sleep, or disease.

Pages already saved under data/eval/images/candidates/ are not fetched
again. dfrac.org is the exception: those pages timed out at 6 seconds, so
this run gives them 20. A PTI page that comes back as a script shell has
no photograph, and spending the image walk on it does not find one.

A fact-check desk often publishes the viral photo with FALSE or its own
name burned in. That file is their graphic, not the picture that was
shared. Tesseract reads the pixels locally. A hit on those words drops
the file before it is written. Nothing here calls the vision model.

Usage:
    python scripts/fetch_more_image_cases.py
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402

# Four frames show the claim and a couple of alternates. The first sheet
# used six, and the extras were usually the same photograph again.
MAX_PER_ARTICLE = 4
_MAX_TRIES = 8
# dfrac.org did not answer inside the fetcher's 6 seconds. Twenty is the
# retry this run is allowed to spend on that host, and on no other.
DFRAC_TIMEOUT_S = 20
# Two outlets are enough for one viral photograph. The seventh copy of the
# same NEET picture does not add a candidate.
_MAX_PER_STORY = 2

CANDIDATE_DIR = config.EVAL_DIR / "images" / "candidates2"
SHEET_PATH = CANDIDATE_DIR / "contact_sheet_2.jpg"
SHEET_CSV = CANDIDATE_DIR / "contact_sheet_2.csv"
DRAFT_PATH = config.EVAL_DIR / "images_testset_draft.csv"

# Whole words. "diet" must match a diet claim, not a longer token, and
# "sleep" must not hide inside an unrelated word.
_STOCK = re.compile(
    r"\b(food|foods|diet|diets|dietary|sleep|sleeping|asleep|"
    r"disease|diseases|cancer|diabetes|kidney|dementia|glaucoma|"
    r"insulin|vaccine|vitamin|fertility|calorie|calories|cholesterol|"
    r"cartilage|headache|turmeric|almond|almonds|yogurt|cinnamon|"
    r"snoring|pomegranate|menstrual|junk)\b",
    re.IGNORECASE,
)
# A ranking or a circular is a sentence with a graphic, not a photograph
# of an event. The contact sheet is for the photograph.
_NOT_A_PHOTO = (
    "health report",
    "ranking",
    "circular",
    "scheme",
    "downloads",
    "online system",
    "drone patrol",
)
_KEEP_DOMAINS = frozenset({
    "Politics & elections",
    "Disasters & accidents",
    "Crime & law",
    "Religion & communal",
    "Defence & international",
})
_PHOTO_WORDS = ("photo", "photograph", "image", "picture")
# Words every "photo shows" claim shares. They must not glue two different
# events into one story.
_GENERIC = frozenset(
    """
    a an the of to in on for and or with by from at as is are was were be
    this that these those it its his her their not no who whom which into
    during over after before about than then photo photograph image images
    picture pictures shows show showing shown viral claim old recent new
    fact check falsely shared video videos
    """.split()
)
# Burned-in words. A fact-check stamp says the verdict or the desk's name.
# "falsely" is the article's verb and is not itself the stamp, so the
# match is the whole word.
_STAMPS = (
    "false",
    "fake",
    "misleading",
    "fact check",
    "factcheck",
    "webqoof",
    "alt news",
    "altnews",
    "boom",
    "factly",
    "vishvas",
    "newschecker",
    "fact crescendo",
    "factcrescendo",
    "first check",
    "firstcheck",
    "newsmeter",
    "dfrac",
    "digiteye",
    "youturn",
    "you turn",
    "telugu post",
    "telugupost",
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_of(url: str) -> str:
    return urlsplit(url).netloc.casefold().removeprefix("www.")


def stock_claim(text: str) -> bool:
    """True when the claim is food, diet, sleep, or disease.

    Those articles illustrate the sentence with a stock picture. The photo
    test needs a picture someone claimed was of a real event.
    """
    return _STOCK.search(text or "") is not None


def _photo_claim(text: str) -> bool:
    folded = (text or "").casefold()
    return any(word in folded for word in _PHOTO_WORDS)


def _not_a_photo(text: str) -> bool:
    folded = (text or "").casefold()
    return any(phrase in folded for phrase in _NOT_A_PHOTO)


def reject_reason(row: dict, already: set[str]) -> str:
    """Why this archive row is not a new event photograph, or ''."""
    url = row.get("url") or ""
    host = host_of(url)
    if host == "firstcheck.in":
        return "firstcheck"
    if (row.get("domain") or "") == "Health":
        return "health domain"
    claim = row.get("primary_claim") or ""
    if stock_claim(claim):
        return "food, diet, sleep, or disease"
    if _not_a_photo(claim):
        return "not a photograph"
    domain = row.get("domain") or ""
    if domain not in _KEEP_DOMAINS and not _photo_claim(claim):
        return "not an event photo"
    if url in already and host != "dfrac.org":
        return "already downloaded"
    return ""


def story_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if word not in _GENERIC and len(word) > 2
    }


def same_story(left: set[str], right: set[str]) -> bool:
    """True when two claims are the same photograph told by two desks.

    Four shared content words, and most of the shorter claim, is the same
    people in the same event. Sharing only 'flood' is not.
    """
    if not left or not right:
        return False
    shared = left & right
    if len(shared) < 4:
        return False
    return len(shared) / min(len(left), len(right)) >= 0.6


def select_rows(rows: list[dict], already: set[str]) -> tuple[list[tuple[int, dict]], list[str]]:
    """The rows to download, and a line for each row that was left out."""
    chosen = []
    notes = []
    clusters: list[dict] = []
    for index, row in enumerate(rows, start=1):
        reason = reject_reason(row, already)
        if reason:
            notes.append(f"  c{index:02d}  {reason}  {row['url']}")
            continue
        words = story_words(row.get("primary_claim") or "")
        cluster = next((item for item in clusters if same_story(words, item["words"])), None)
        if cluster is None:
            clusters.append({"words": words, "count": 1})
        else:
            cluster["count"] += 1
            if cluster["count"] > _MAX_PER_STORY:
                notes.append(f"  c{index:02d}  same story already kept  {row['url']}")
                continue
        chosen.append((index, row))
    return chosen, notes


def stamp_hit(text: str) -> str:
    """The stamp word found in OCR text, or '' when the picture is clean.

    Hyphens become spaces so FACT-CHECK and FACT CHECK are one stamp.
    The search is the whole word: 'falsely' does not contain 'false'.
    """
    folded = " ".join((text or "").casefold().replace("-", " ").split())
    squashed = folded.replace(" ", "")
    for phrase in _STAMPS:
        if " " in phrase:
            if re.search(rf"\b{re.escape(phrase)}\b", folded):
                return phrase
            continue
        if phrase == "factcheck":
            if "factcheck" in squashed:
                return phrase
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", folded):
            return phrase
    return ""


def ocr_text(jpeg: bytes) -> str:
    """Local Tesseract. This is not the vision model.

    A short stamp on a large frame is small in pixels. Doubling a short
    side under 1000 is the same scale the ingest OCR uses, so a red
    FALSE across the corner is large enough to read.
    """
    import pytesseract
    from PIL import Image

    image = Image.open(BytesIO(jpeg)).convert("L")
    short = min(image.size)
    if 0 < short < 1000:
        image = image.resize(
            (image.size[0] * 2, image.size[1] * 2),
            Image.Resampling.LANCZOS,
        )
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("tesseract is not installed") from exc


def _already_urls() -> set[str]:
    if not DRAFT_PATH.is_file():
        return set()
    with DRAFT_PATH.open(encoding="utf-8", newline="") as handle:
        return {
            (row.get("source_article") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("source_article") or "").strip()
        }


def _shell(html: bytes) -> bool:
    """A PTI shell is a few kilobytes and no <img>. There is nothing to save."""
    if len(html) >= 30000:
        return False
    return b"<img" not in html[:50000].lower()


@contextmanager
def _page_timeout(seconds: float):
    """The fetcher reads this value when the request starts."""
    previous = config.FETCH_TIMEOUT_S
    config.FETCH_TIMEOUT_S = seconds
    try:
        yield
    finally:
        config.FETCH_TIMEOUT_S = previous


def _save_clean(fic, item: dict, dest: Path) -> dict | None:
    try:
        _final, body, _ctype = fic._once(item["url"], images=True)
        jpeg, width, height = fic.to_jpeg(body)
    except Exception as exc:
        reason = getattr(exc, "reason", None) or type(exc).__name__
        print(f"    skip {reason} {item['url'][:80]}", flush=True)
        return None
    if width < fic.MIN_SIDE or height < fic.MIN_SIDE:
        print(f"    skip {width}x{height} {item['url'][:80]}", flush=True)
        return None
    try:
        hit = stamp_hit(ocr_text(jpeg))
    except Exception as exc:
        print(f"    skip ocr {type(exc).__name__} {item['url'][:80]}", flush=True)
        return None
    if hit:
        print(f"    skip stamp {hit} {item['url'][:80]}", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(jpeg)
    return {
        "url": item["url"],
        "kind": item["kind"],
        "path": dest,
        "width": width,
        "height": height,
    }


def fetch_article(fic, row_number: int, row: dict) -> tuple[list[dict], str]:
    print(f"c{row_number:02d} {row['url']}", flush=True)
    timeout = DFRAC_TIMEOUT_S if host_of(row["url"]) == "dfrac.org" else config.FETCH_TIMEOUT_S
    with _page_timeout(timeout):
        html, reason = fic._open_article(row["url"])
        if html is None:
            print(f"  page {reason}", flush=True)
            return [], reason or "no page"
        if host_of(row["url"]) == "ptinews.com" and _shell(html):
            print("  shell", flush=True)
            return [], "shell"
        saved = []
        for attempt, item in enumerate(fic.candidate_list(html, row["url"]), start=1):
            if len(saved) >= MAX_PER_ARTICLE or attempt > _MAX_TRIES:
                break
            dest = CANDIDATE_DIR / f"c{row_number:02d}_{len(saved) + 1}.jpg"
            got = _save_clean(fic, item, dest)
            if got is None:
                continue
            saved.append(got)
            print(f"  {dest.name} {got['width']}x{got['height']} {got['kind']}", flush=True)
        return saved, ""


def _write_csv(records: list[dict]) -> None:
    with SHEET_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename", "article URL", "primary_claim",
                "archive verdict", "image width x height",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "filename": record["path"].name,
                "article URL": record.get("article_url") or "",
                "primary_claim": record.get("claim") or "",
                "archive verdict": record.get("verdict") or "",
                "image width x height": f"{record['width']}x{record['height']}",
            })


def main() -> None:
    finder = _load("find_image_cases")
    fic = _load("fetch_image_cases")
    rows = finder.candidates(80)
    chosen, notes = select_rows(rows, _already_urls())
    print(f"pool {len(rows)}  downloading {len(chosen)}")
    if notes:
        print("left out")
        for line in notes:
            print(line)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CANDIDATE_DIR.glob("*.jpg"):
        if re.fullmatch(r"c\d+_\d+\.jpg", stale.name):
            stale.unlink()
    sheet = []
    counts = []
    for number, row in chosen:
        saved, reason = fetch_article(fic, number, row)
        counts.append((number, len(saved), row["url"], reason))
        for item in saved:
            item["article_url"] = row["url"]
            item["claim"] = row["primary_claim"]
            item["verdict"] = row["verdict"]
            sheet.append(item)
    fic.write_sheet(sheet, SHEET_PATH)
    _write_csv(sheet)
    print("")
    print("candidates per article")
    for number, count, url, reason in counts:
        tail = f"  ({reason})" if reason and count == 0 else ""
        print(f"  c{number:02d}  {count}  {url}{tail}")
    print(f"usable images {len(sheet)}")
    print(SHEET_PATH)


if __name__ == "__main__":
    main()
