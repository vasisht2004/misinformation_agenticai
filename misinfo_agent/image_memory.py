"""Same-photograph memory. The file can match and the caption still differ.

Box 5 looks a new image up here. Hamming distance 6 or less is the same
file ingest already uses for a repeat submission: a recompress or a
small crop, not a different photograph.

The row stores the BGE-M3 embedding of the extracted claim, from the
same encoder box 5 uses for text, plus the claim's own words. The
embedding is not of the raw message or the caption string: two
captions on one file are different claims. The date and place rules
read the words, because a vector cannot show that 10 March and 15
August are not the same day.

On a hash hit the stored verdict is reused only when this submission
is that same claim. Either both are empty, or the new claim's cosine
against the stored one is at least TAU_HIGH and the prior-match
key-term, number, date and place rules agree. Anything else keeps the
hit on the card and the photograph is checked again.

An out-of-context verdict does not expire. Where the file came from
does not move. Any other stored verdict expires after STALE_DAYS, the
same window a status claim uses, because a "True" about a price or a
ban can go stale even when the file is unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np

import misinfo_agent.config as config

# False, Misleading, True, and Unverifiable are findings. Other is stored
# so a repeat of the empty photo can be seen, and reused only while the
# new submission is still empty.
_STORED = frozenset({"False", "Misleading", "True", "Unverifiable", "Other"})
_REUSE = frozenset({"False", "Misleading", "True", "Unverifiable", "Other"})

# The card sentence when the file matches and the claim does not. The
# hit stays so a person can see the earlier check. It is not a verdict.
_DIFFERENT_CAPTION = "This photo was checked before with a different caption."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    phash TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    first_seen TEXT,
    thread_id TEXT,
    refute_kind TEXT,
    stored_on TEXT,
    claim TEXT,
    had_caption INTEGER,
    embedding BLOB
)
"""

# Columns added after the first runs. CREATE TABLE does not alter a
# table that already exists, and an old row has no claim to compare.
_ADDED_COLUMNS = (
    ("claim", "ALTER TABLE images ADD COLUMN claim TEXT"),
    ("had_caption", "ALTER TABLE images ADD COLUMN had_caption INTEGER"),
    ("embedding", "ALTER TABLE images ADD COLUMN embedding BLOB"),
)


def lookup(phash: str) -> dict | None:
    """The closest stored picture within Hamming 6, or None.

    The dict is the card shape: thread_id, verdict, distance, first_seen.
    Pixels are not in the row and are not returned. The claim embedding
    stays in the file: the card does not need a thousand floats. A wider
    table is scanned in Python because this file holds decided runs, not
    the archive, and building a matrix over it would copy the thing box 5
    is forbidden to do for the text memory.
    """
    row = _closest(phash)
    if row is None:
        return None
    return _public(row)


def apply_image_reuse(state: dict, decision: dict) -> dict:
    """Overlay route reuse_prior when this file was already decided as this claim.

    Called from box 6 after the text prior. A text hit cannot identify
    the file, so it stays capped at similar. This hash can, and it
    replaces that route only when the stored verdict is still reusable
    and the caption is the same claim. A hash hit whose caption differs
    leaves the text route alone, which for a photograph is investigate,
    and puts the hit back on the card with a note. The check then runs.
    """
    row = _closest(str(state.get("image_phash") or ""))
    if row is None:
        return decision
    if _reusable(row, state):
        return _reuse(state, decision, row)
    if _caption_disagrees(row, state):
        out = dict(decision)
        # The text route is already investigate for a photograph: words
        # cannot identify the file. Forcing it here means a reused text
        # prior cannot skip the picture when the caption is not the one
        # this file was decided under.
        out["route"] = "investigate"
        out["image_memory_hit"] = _public(row, note=_DIFFERENT_CAPTION)
        return out
    return decision


def remember(state: dict) -> None:
    """Write the pHash after a decided verdict. No pixels.

    A null verdict is a conflict the rules refused to label. It is not
    stored, because the next copy of the file should be checked rather
    than reused as "we did not decide". The embedding is the extracted
    claim. An empty photo has no claim, so it stores no vector, and the
    next empty submission can still match on that absence.
    """
    phash = str(state.get("image_phash") or "").strip()
    verdict = state.get("verdict")
    if not phash or not isinstance(verdict, str) or verdict not in _STORED:
        return
    first = state.get("image_provenance") if isinstance(state.get("image_provenance"), dict) else {}
    seen = first.get("first_seen") if isinstance(first, dict) else None
    if not isinstance(seen, dict):
        hit = state.get("image_memory_hit")
        if isinstance(hit, dict) and isinstance(hit.get("first_seen"), dict):
            seen = hit["first_seen"]
        else:
            seen = None
    marks = state.get("run_marks") if isinstance(state.get("run_marks"), dict) else {}
    thread = str(marks.get("run_id") or state.get("thread_id") or "")
    kind = _refute_kind(state)
    payload = json.dumps(seen) if isinstance(seen, dict) else ""
    claim = " ".join(str(state.get("claim") or "").split())
    had_caption = 1 if str(state.get("caption") or "").strip() else 0
    vector = _embed(claim)
    blob = vector.tobytes() if vector is not None else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO images (
                phash, verdict, first_seen, thread_id, refute_kind, stored_on,
                claim, had_caption, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(phash) DO UPDATE SET
                verdict = excluded.verdict,
                first_seen = excluded.first_seen,
                thread_id = excluded.thread_id,
                refute_kind = excluded.refute_kind,
                stored_on = excluded.stored_on,
                claim = excluded.claim,
                had_caption = excluded.had_caption,
                embedding = excluded.embedding
            """,
            (
                phash,
                verdict,
                payload,
                thread,
                kind,
                date.today().isoformat(),
                claim,
                had_caption,
                blob,
            ),
        )


def _reuse(state: dict, decision: dict, row: dict) -> dict:
    out = dict(decision)
    note = "This picture matches one already checked, so that verdict is reused."
    reasons = [note] + [str(item) for item in (out.get("match_reasons") or [])]
    first = row["first_seen"] if isinstance(row["first_seen"], dict) else {}
    # Cosine is what assess turns into a confidence. Distance 0 is the
    # same bits. Distance 6 is still the same file, a little less sure.
    distance = int(row["distance"])
    cosine = max(0.0, 1.0 - (distance / 64.0))
    out["route"] = "reuse_prior"
    out["match_type"] = "exact"
    out["needs_recheck"] = False
    out["match_reasons"] = reasons
    out["match_summary"] = reasons[:2]
    out["best_match"] = {
        "verdict": row["verdict"],
        "verdict_norm": str(row["verdict"]).casefold(),
        "true_version": "",
        "primary_claim": row.get("claim") or state.get("claim") or "",
        "dense_cos": round(cosine, 4),
        "date": first.get("date") or "",
        "url": first.get("url") or "",
        "site": first.get("host") or "",
        "article_id": row["thread_id"],
        "refute_kind": row.get("refute_kind") or None,
    }
    return out


def _reusable(row: dict, state: dict) -> bool:
    verdict = str(row.get("verdict") or "")
    if verdict not in _REUSE:
        return False
    if _expired(row):
        return False
    return _same_submission(row, state)


def _caption_disagrees(row: dict, state: dict) -> bool:
    """True when this file was stored under a claim we can compare and it differs.

    A row from before claims were stored has neither a claim nor a
    caption flag. Saying "different caption" about that row would invent
    a difference. It is still not reused: _reusable already returned
    false. The full check runs, and the hit from lookup stays as it was.
    """
    if row.get("claim") is None and row.get("had_caption") is None:
        return False
    return not _same_submission(row, state)


def _same_submission(row: dict, state: dict) -> bool:
    """Both empty, or the same claim under the text rules."""
    if _both_blank(row, state):
        return True
    return _claims_agree(row, state)


def _both_blank(row: dict, state: dict) -> bool:
    """No caption and no claim on either submission.

    NULL is an old row that never recorded this, not an empty photo.
    An empty photo is a stored empty string and had_caption 0.
    """
    if row.get("claim") is None or row.get("had_caption") is None:
        return False
    stored_claim = str(row.get("claim") or "").strip()
    stored_caption = int(row.get("had_caption") or 0) == 1
    new_claim = str(state.get("claim") or "").strip()
    new_caption = str(state.get("caption") or "").strip()
    return not stored_claim and not stored_caption and not new_claim and not new_caption


def _claims_agree(row: dict, state: dict) -> bool:
    """TAU_HIGH, then the same detail rules box 6 uses on two claims.

    decide_prior is not called. It would run the model veto, and it
    caps every picture at similar, which would refuse the reuse this
    function exists to allow. The thresholds are the ones already in
    config. Nothing here adds another cutoff.
    """
    new = " ".join(str(state.get("claim") or "").split())
    old = row.get("claim")
    if old is None:
        return False
    old = " ".join(str(old).split())
    if not new or not old:
        return False
    stored = _vector(row.get("embedding"))
    fresh = _embed(new)
    if stored is None or fresh is None or stored.shape != fresh.shape:
        return False
    # >=, not >. The configured cutoff itself counts, as it does in box 6.
    if _cosine(fresh, stored) < config.TAU_HIGH:
        return False
    return _rules_pass(new, old, state.get("entities"))


def _rules_pass(new: str, old: str, entities) -> bool:
    from misinfo_agent.prior_match import (
        KEYTERM_SIM,
        _community_note,
        _detail_notes,
        _paired_term_note,
        _same_claim_text,
    )

    same_words = _same_claim_text(new, old)
    # Dates, amounts, years, numbers, and places. Names are skipped only
    # when the words are the same: that is the tagger glitch prior_match
    # already refuses to call a clash. A cosine of 0.99 does not skip
    # them. A swapped city is often that close, and it is a different caption.
    notes = _detail_notes(
        new,
        old,
        entities if isinstance(entities, dict) else None,
        contradiction=True,
        normalise=True,
        check_names=not same_words,
    )
    if notes:
        return False
    if same_words:
        return True
    if _community_note(new, old):
        return False
    if _paired_term_note(new, old, KEYTERM_SIM):
        return False
    return True


def _embed(claim: str) -> np.ndarray | None:
    """One unit-length vector of the extracted claim. Empty stores nothing.

    Whitespace is folded the way memory.embed folds it, so a claim that
    was stored with a double space matches the same words later. The
    encoder is the process-wide BGE-M3. A second model would not be the
    space TAU_HIGH was chosen in.
    """
    text = " ".join(str(claim or "").split())
    if not text:
        return None
    from misinfo_agent.memory import get_encoder

    vector = get_encoder().encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 2:
        array = array[0]
    return np.ascontiguousarray(array)


def _vector(blob) -> np.ndarray | None:
    if not blob:
        return None
    array = np.frombuffer(blob, dtype=np.float32)
    if array.size == 0:
        return None
    # frombuffer is a view of the sqlite bytes. Copying keeps the vector
    # usable after the connection closes.
    return np.array(array, dtype=np.float32, copy=True)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _public(row: dict, note: str | None = None) -> dict:
    hit = {
        "thread_id": row["thread_id"],
        "verdict": row["verdict"],
        "distance": row["distance"],
        "first_seen": row["first_seen"],
    }
    if note:
        hit["note"] = note
    return hit


def _expired(row: dict) -> bool:
    """Out of context never expires. Everything else uses STALE_DAYS."""
    if str(row.get("refute_kind") or "") == "out_of_context":
        return False
    stored = _as_date(row.get("stored_on"))
    if stored is None:
        return False
    return (date.today() - stored).days > config.STALE_DAYS


def _closest(phash: str) -> dict | None:
    text = str(phash or "").strip()
    if not text:
        return None
    path = _path()
    if not path.is_file():
        return None
    try:
        import imagehash
    except Exception:
        return None
    try:
        query = imagehash.hex_to_hash(text)
    except Exception:
        return None
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT phash, verdict, first_seen, thread_id, refute_kind, stored_on,
                       claim, had_caption, embedding
                FROM images
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return None
    best = None
    best_distance = config.IMAGE_MEMORY_HAMMING + 1
    for item in rows:
        try:
            distance = int(query - imagehash.hex_to_hash(str(item["phash"])))
        except Exception:
            continue
        if distance < best_distance:
            best_distance = distance
            best = item
    if best is None or best_distance > config.IMAGE_MEMORY_HAMMING:
        return None
    seen = None
    raw = best["first_seen"] or ""
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            seen = parsed
    return {
        "thread_id": best["thread_id"] or "",
        "verdict": best["verdict"] or "",
        "distance": best_distance,
        "first_seen": seen,
        "refute_kind": best["refute_kind"] or "",
        "stored_on": best["stored_on"] or "",
        "claim": best["claim"],
        "had_caption": best["had_caption"],
        "embedding": best["embedding"],
    }


def _refute_kind(state: dict) -> str:
    """The core claim's kind, so an out-of-context row can be kept forever."""
    reasons = " ".join(str(item) for item in (state.get("gate_reasons") or []))
    if "out of context" in reasons.casefold():
        return "out_of_context"
    core = "A1"
    for item in state.get("atomic_claims") or []:
        if isinstance(item, dict) and item.get("core"):
            core = str(item.get("id") or "A1")
            break
    for item in state.get("atomic_results") or []:
        if isinstance(item, dict) and str(item.get("atomic_id") or "A1") == core:
            return str(item.get("refute_kind") or "")
    return ""


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _connect() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    have = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    for name, ddl in _ADDED_COLUMNS:
        if name not in have:
            conn.execute(ddl)
    return conn


def _path() -> Path:
    return Path(config.IMAGE_MEMORY_DB)
