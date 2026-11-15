"""Box 5: hybrid search over the 52,099 past fact-check claims.

Dense BGE-M3 is one matrix-vector product, E @ q. The rows of E are already
unit length, and q is encoded with the same model, the same 256-token window
and the same L2 normalisation, so the product is a cosine. BM25 ranks the
same rows by the words in the claim. Reciprocal rank fusion adds the two
rankings without putting cosine and BM25 on a fake common scale.

What this module refuses to do: an N×N similarity matrix. 52,099² float32
values are about 11 GB, and nothing in the search needs every pair. The only
pairwise cosines are inside the fused top 30, where near-duplicate fact-checks
of the same claim (two outlets, one story) are collapsed.

NumPy is the search path. A date or a domain filter is one boolean mask, and
a mask is awkward on a flat index. The FAISS cross-check lives in
scripts/faiss_parity.py and is not imported here: loading that library in the
same process as the MPS encoder segfaults, and search never needs it.

The article body (`content`) is never read. Memory search is over
`primary_claim`, which is the string the embedding was built from.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import json
import pickle
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

from misinfo_agent.config import (
    BM25_PATH,
    CLASSIFIED_PATH,
    DOMAINS,
    DUP_COS,
    EMB_NPY_PATH,
    EMBEDDINGS_PATH,
    MANIFEST_PATH,
    MAX_SEQ_LEN,
    MEMORY_DIR,
    META_PARQUET_PATH,
    MODEL_NAME,
    RRF_K,
)

# DATA.md: 52,099 fact-check rows have embeddings. A different count means the
# inputs are not the corpus this box was written against, so we stop rather
# than index a silent mismatch.
EXPECTED_ROWS = 52_099
EMBED_DIM = 1024

# How many names from each list enter the fusion, and how many fused hits are
# inspected for near-duplicates. Fifty is deep enough for a claim that is only
# middling on one list. Thirty is the window in which the same story, checked
# by several outlets, actually clusters. Beyond that the pairs are not the
# duplicate problem, and we still would not compare the whole matrix.
DENSE_DEPTH = 50
BM25_DEPTH = 50
FUSE_DEPTH = 30

# Bump when the on-disk layout or the tokeniser changes. The input hashes
# alone would keep serving a BM25 file built by an older stemmer.
BUILDER_VERSION = 1

# Columns copied into the memory. `content` is absent on purpose: it is the
# full article, it is large, and box 5 does not search it.
CLASSIFIED_COLUMNS = [
    "article_id",
    "site",
    "title",
    "url",
    "date",
    "year",
    "primary_claim",
    "true_version",
    "domain",
    "verdict",
    "verdict_norm",
    "manipulation_type",
    "red_flags",
    "form",
]

# A short list, not a full English stoplist. "before", "after", numbers and
# the status words box 6 reasons about ("banned", "invalid", "price") stay:
# they are the difference between two claims, so BM25 must still see them.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "from",
        "with", "by", "at", "as", "is", "are", "was", "were", "be", "been",
        "being", "this", "that", "it", "its", "will", "has", "have", "had",
        "but", "if", "into", "over", "than", "then", "so", "such", "their",
        "there", "these", "those", "he", "she", "they", "we", "you", "your",
        "his", "her", "our", "who", "what", "which", "when", "where", "do",
        "does", "did", "not", "no", "nor",
    }
)

# ₹500, Rs 500, Rs.500, INR 500. The lookbehind stops "nurses 500" from
# becoming a rupee amount: the "rs" has to start a token, not finish a word.
_RUPEE_PREFIX = re.compile(
    r"(?<![a-z])(?:₹|rs\.?|inr)\s*(\d[\d,]*)(?:\.\d+)?",
    re.IGNORECASE,
)
# The same amount written as "500 rupees". One token either way, or BM25
# treats the symbol and the word as unrelated claims.
_RUPEE_WORD = re.compile(
    r"(?<![a-z\d])(\d[\d,]*)(?:\.\d+)?\s*rupees?\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")

_LOADED: "ClaimMemory | None" = None


def bm25_token_pairs(text: str) -> list[tuple[str, str]]:
    """(surface, stem) pairs for the BM25 index and for box 6's reasons.

    Both sides of a search must call this. Rupee spellings collapse to
    `inr500` before the split, so the symbol and "Rs" land on one term.
    The surface is the word before stemming: a reason can say "coronavirus"
    rather than the stem "coronaviru", which is an index key, not English.
    """
    if not text:
        return []
    lowered = _WHITESPACE.sub(" ", text.lower()).strip()
    lowered = _RUPEE_PREFIX.sub(lambda m: f" inr{m.group(1).replace(',', '')} ", lowered)
    lowered = _RUPEE_WORD.sub(lambda m: f" inr{m.group(1).replace(',', '')} ", lowered)
    pairs = []
    for token in _TOKEN.findall(lowered):
        if token in _STOPWORDS:
            continue
        pairs.append((token, _stem(token)))
    return pairs


def bm25_tokens(text: str) -> list[str]:
    """Stems only. The index and the query both use this list.

    Stemming is one suffix, not Porter: a full stemmer merges place names
    and people, and it is not a dependency we have. The rule people will
    notice is withdrawn/withdraw, because "the note was withdrawn" and
    "RBI will withdraw the note" are the same claim.
    """
    return [stem for _surface, stem in bm25_token_pairs(text)]


def strip_rupee_amounts(text: str) -> str:
    """Text with ₹ / Rs / "rupees" amounts removed.

    Box 6 compares rupee amounts on their own. If "₹500" is left in the
    string, the 500 is also parsed as a bare number and disagrees with a
    year or a day that happens to be different.
    """
    if not text:
        return ""
    return _RUPEE_WORD.sub(" ", _RUPEE_PREFIX.sub(" ", text))


def _stem(token: str) -> str:
    """One suffix, then a silent e. Idempotent on the stem it produces.

    A trailing n after w is the participle of a verb that already ends in w
    (withdrawn, thrown). Stripping every trailing n would turn "indian" into
    "india". "brown" is five letters and is left alone. "banned" loses "-ed"
    and the doubled consonant, so it meets "ban".

    The silent e is what makes "announced" meet "announce". Removing "-ed"
    alone leaves "announc", and the query that says "announce" would miss it.
    "note" is only four letters, so it does not lose the e and become "not".
    """
    if len(token) >= 6 and token.endswith("wn"):
        return token[:-1]
    if token.endswith("ing") and len(token) - 3 >= 4:
        token = token[:-3]
    elif token.endswith("ed") and len(token) - 2 >= 4:
        stem = token[:-2]
        if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        token = stem
    elif token.endswith("ly") and len(token) - 2 >= 4:
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        # "notes" -> "note". "es" only for boxes/churches, otherwise "notes"
        # would lose two letters and become "not".
        if token.endswith(("ches", "shes", "sses", "xes", "zes")):
            token = token[:-2]
        else:
            token = token[:-1]
    if len(token) >= 5 and token.endswith("e") and token[-2] not in "aeiou":
        token = token[:-1]
    return token


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_current(manifest: dict, classified_sha: str, embeddings_sha: str) -> bool:
    """True when the files on disk were built from these exact inputs.

    build time is not part of the check: it changes on every run, so including
    it would make the skip impossible. The builder version is part of it,
    because a new tokeniser must not reuse an old bm25.pkl whose hash of the
    parquet would still match.
    """
    # faiss.index is not an artefact of this build. The parity script writes
    # it, and search must keep working on a machine that never imported FAISS.
    artefacts = (EMB_NPY_PATH, META_PARQUET_PATH, BM25_PATH)
    return (
        manifest.get("classified_sha256") == classified_sha
        and manifest.get("embeddings_sha256") == embeddings_sha
        and manifest.get("model") == MODEL_NAME
        and manifest.get("max_seq_len") == MAX_SEQ_LEN
        and manifest.get("rows") == EXPECTED_ROWS
        and manifest.get("builder_version") == BUILDER_VERSION
        and all(path.exists() for path in artefacts)
    )


def _load_embeddings() -> tuple[list[str], np.ndarray]:
    """article_id order is the row order of E from here on.

    Flattening the list column in Arrow is one numeric buffer. Materialising
    each dimension as a Python float would be about 53 million objects.
    Stored values are float64; the model emitted float32, and that is the
    dtype search multiplies.
    """
    table = pq.read_table(EMBEDDINGS_PATH, columns=["article_id", "embedding"])
    ids = table.column("article_id").to_pylist()
    flat = pc.list_flatten(table.column("embedding")).to_numpy(zero_copy_only=False)
    matrix = np.ascontiguousarray(np.asarray(flat, dtype=np.float32).reshape(len(ids), EMBED_DIM))
    return ids, matrix


def _assert_unit_rows(matrix: np.ndarray) -> float:
    """Each row is already normalised. We check; we do not renormalise.

    Renormalising would hide a matrix that was not the one described in
    DATA.md. The check is a vector of 52,099 norms, not a Gram matrix.
    """
    norms = np.linalg.norm(matrix, axis=1)
    drift = float(np.max(np.abs(norms - 1.0)))
    if drift > 1e-3:
        raise AssertionError(f"row norms drift from 1 by up to {drift:.6f}, limit is 1e-3")
    return drift


def _join_meta(embedding_ids: list[str]) -> tuple[pd.DataFrame, int, list[str]]:
    """Classified columns for those ids, in embedding order. No `content`."""
    if "content" in CLASSIFIED_COLUMNS:
        raise AssertionError("box 5 must not read the article body")
    frame = pd.read_parquet(CLASSIFIED_PATH, columns=CLASSIFIED_COLUMNS)
    if not frame["article_id"].is_unique:
        raise AssertionError("classified_v2 has duplicate article_id values")
    if len(embedding_ids) != len(set(embedding_ids)):
        raise AssertionError("embedding file has duplicate article_id values")
    # Left-join from the embedding ids. pandas keeps the left row order, which
    # is the order E was stacked in. Sorting here would silently desync the
    # matrix from the metadata.
    ordered = pd.DataFrame({"article_id": embedding_ids})
    joined = ordered.merge(frame, on="article_id", how="left", validate="1:1")
    if len(joined) != EXPECTED_ROWS:
        raise AssertionError(f"expected {EXPECTED_ROWS} rows after the join, got {len(joined)}")
    if not joined["article_id"].is_unique:
        raise AssertionError("joined memory has duplicate article_id values")
    if joined["primary_claim"].isna().any():
        missing = int(joined["primary_claim"].isna().sum())
        raise AssertionError(
            f"{missing} embedding ids have no primary_claim in classified_v2"
        )
    # errors="coerce": an unparseable date becomes NaT and is reported, not
    # dropped. A row with NaT fails a date filter (we cannot show it was
    # published before the claim) and stays searchable when no date is set.
    parsed = pd.to_datetime(joined["date"], errors="coerce", format="mixed")
    failed_mask = parsed.isna()
    examples = joined.loc[failed_mask, "date"].astype(str).head(5).tolist()
    joined["date_parsed"] = parsed
    return joined, int(failed_mask.sum()), examples


def build_memory(force: bool = False) -> dict:
    """Build emb.npy, meta.parquet, bm25.pkl and manifest.json.

    Skips the work when the manifest was built from these input hashes, this
    model and this builder version, and the artefacts are all still there.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    classified_sha = _sha256(CLASSIFIED_PATH)
    embeddings_sha = _sha256(EMBEDDINGS_PATH)
    existing = _read_manifest()
    if not force and existing is not None and _manifest_current(existing, classified_sha, embeddings_sha):
        return {**existing, "skipped": True}

    started = time.perf_counter()
    print("box 5: reading embeddings and classified columns", flush=True)
    embedding_ids, matrix = _load_embeddings()
    if matrix.shape != (EXPECTED_ROWS, EMBED_DIM):
        raise AssertionError(f"expected E to be {(EXPECTED_ROWS, EMBED_DIM)}, got {matrix.shape}")
    drift = _assert_unit_rows(matrix)
    meta, dates_failed, date_examples = _join_meta(embedding_ids)

    np.save(EMB_NPY_PATH, matrix)
    meta.to_parquet(META_PARQUET_PATH, index=False)

    print(f"box 5: {dates_failed} dates failed to parse; building BM25", flush=True)
    corpus = [bm25_tokens(claim) for claim in meta["primary_claim"].tolist()]
    bm25 = BM25Okapi(corpus)
    with BM25_PATH.open("wb") as handle:
        # Local build output only. query tokens are produced by bm25_tokens,
        # never by unpickling anything a user submitted.
        pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.perf_counter() - started
    manifest = {
        "rows": EXPECTED_ROWS,
        "classified_sha256": classified_sha,
        "embeddings_sha256": embeddings_sha,
        "model": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "builder_version": BUILDER_VERSION,
        "build_time_s": round(elapsed, 3),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dates_failed": dates_failed,
        "date_failures_sample": date_examples,
        "norm_drift": drift,
        "embed_dim": EMBED_DIM,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["skipped"] = False
    return manifest


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first.

    argpartition finds the top k in linear time. The k winners are then
    sorted. A full argsort of the 52,099 is harmless, but the partition is
    what keeps a wider matrix from doing work we throw away. k == n has no
    pivot, so that case is a plain sort.
    """
    n = int(scores.shape[0])
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.argsort(-scores, kind="mergesort")
    chosen = np.argpartition(scores, -k)[-k:]
    order = np.argsort(-scores[chosen], kind="mergesort")
    return chosen[order].astype(np.int64, copy=False)


_ENCODER = None
_ENCODER_LOCK = threading.Lock()


def get_encoder():
    """The one BGE-M3 in this process.

    Box 5 and box 7 both embed text. A second copy would use the memory
    twice and could disagree if one of them left the sequence length at
    the library default of 8192. The lock covers the first load only.
    """
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            _ENCODER = _build_encoder()
        return _ENCODER


def _build_encoder():
    from sentence_transformers import SentenceTransformer

    device = _device()
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    # E was built at 256 tokens. The library default would embed a longer
    # prefix, and the dot product would stop being the cosine box 6 uses.
    encoder.max_seq_length = MAX_SEQ_LEN
    return encoder


def _device() -> str:
    """Same preference as the script that built E: MPS, then CUDA, then CPU.

    The matrix does not move. Only the query is encoded. MPS is where that
    was done, so a query for a stored claim lands on the same vector it was
    filed under. CPU is a fallback, not a second index.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClaimMemory:
    """The on-disk memory, loaded once per process. The encoder is lazy.

    Loading the matrix is a few hundred megabytes and is needed for every
    search. Loading BGE-M3 is slower and is needed only when a query string
    arrives, so import and `load()` do not pay for it. Tests of the tokeniser
    never touch the model.
    """

    def __init__(self, matrix: np.ndarray, meta: pd.DataFrame, bm25: BM25Okapi):
        if len(meta) != matrix.shape[0]:
            raise AssertionError("meta rows and embedding rows disagree")
        self.E = matrix
        self.meta = meta
        self.bm25 = bm25
        self.ids = meta["article_id"].astype(str).to_numpy()
        self.dates = meta["date_parsed"].to_numpy(dtype="datetime64[ns]")
        self.domains = meta["domain"].fillna("").astype(str).to_numpy()
        self.sites = meta["site"].fillna("").astype(str).to_numpy()
        self._encoder = None
        self.device: str | None = None

    @classmethod
    def load(cls) -> "ClaimMemory":
        matrix = np.load(EMB_NPY_PATH)
        meta = pd.read_parquet(META_PARQUET_PATH)
        with BM25_PATH.open("rb") as handle:
            bm25 = pickle.load(handle)
        return cls(matrix, meta, bm25)

    def _encoder_model(self):
        # The shared loader, not a second SentenceTransformer. Box 7 scores
        # passages with the same object, so a claim and a memory row are
        # embedded by one model.
        encoder = get_encoder()
        self._encoder = encoder
        self.device = _device()
        return encoder

    def embed(self, claim: str) -> np.ndarray:
        """One unit-length query vector. Whitespace is collapsed first.

        The cache was encoded from primary_claim with runs of whitespace
        folded to a single space. Skipping that fold makes an exact stored
        claim miss itself when the source had a double space or a newline.
        """
        text = _WHITESPACE.sub(" ", claim).strip()
        vector = self._encoder_model().encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(np.asarray(vector[0], dtype=np.float32))

    def search(
        self,
        claim: str,
        k: int = 10,
        date_before: str | None = None,
        date_after: str | None = None,
        domains: list[str] | None = None,
        sites: list[str] | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """Top k hits after fusion and near-duplicate collapse.

        Every filter is one boolean mask applied to both lists. date_before
        is strict (<). Evaluation sets it to the claim's date so a fact-check
        published that day, or later, cannot be "remembered": same-day rows
        are excluded because the column is date-only and `< midnight` drops
        the whole day. date_after is inclusive, the start of a window.
        A row whose date did not parse is outside any dated window.
        """
        # Before the empty-claim return, so a typo is an error even when the
        # text is blank. Returning [] would hide it as "nothing in memory".
        domains = _check_domains(domains)
        k = int(k)
        if k < 1 or not claim or not str(claim).strip():
            return []
        mask = self._mask(date_before, date_after, domains, sites, exclude_ids)
        n_keep = int(mask.sum())
        if n_keep == 0:
            return []

        query = self.embed(claim)
        dense = self.E @ query
        dense = dense.copy()
        dense[~mask] = -np.inf
        dense_idx = _topk_indices(dense, min(DENSE_DEPTH, n_keep))

        bm25_idx = self._bm25_top(claim, mask, n_keep)
        return self._fuse(dense, dense_idx, bm25_idx, k)

    def _mask(self, date_before, date_after, domains, sites, exclude_ids) -> np.ndarray:
        mask = np.ones(self.E.shape[0], dtype=bool)
        if date_before is not None:
            # NaT < x is False, so an unparsed date does not sneak through.
            cutoff = np.datetime64(pd.Timestamp(date_before), "ns")
            mask &= self.dates < cutoff
        if date_after is not None:
            start = np.datetime64(pd.Timestamp(date_after), "ns")
            mask &= self.dates >= start
        if domains:
            mask &= np.isin(self.domains, np.asarray(list(domains), dtype=object))
        if sites:
            if isinstance(sites, str):
                sites = [sites]
            mask &= np.isin(self.sites, np.asarray(list(sites), dtype=object))
        if exclude_ids:
            banned = np.asarray([str(item) for item in exclude_ids], dtype=object)
            mask &= ~np.isin(self.ids, banned)
        return mask

    def _bm25_top(self, claim: str, mask: np.ndarray, n_keep: int) -> np.ndarray:
        """Top BM25 rows inside the mask, scores above zero only.

        A zero score means the query terms are absent. Ranking those zeros
        would hand RRF a list of arbitrary fact-checks and let them compete
        with a real dense hit. If nothing scores above zero, BM25 abstains
        and fusion is the dense list alone.
        """
        tokens = bm25_tokens(claim)
        if not tokens:
            return np.empty(0, dtype=np.int64)
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        scores = scores.copy()
        scores[~mask] = -np.inf
        scores[scores <= 0] = -np.inf
        n_positive = int(np.isfinite(scores).sum())
        if n_positive == 0:
            return np.empty(0, dtype=np.int64)
        return _topk_indices(scores, min(BM25_DEPTH, n_positive, n_keep))

    def _fuse(self, dense: np.ndarray, dense_idx: np.ndarray, bm25_idx: np.ndarray, k: int) -> list[dict]:
        """RRF over the two top-50s, then collapse near-duplicates in the top 30.

        Rank is 1 for the best hit on that list. A row on both lists gets both
        terms. 1/(k + rank) with k=60 is the standard constant: rank 1 is
        worth 1/61, rank 50 is worth 1/110, and neither list can bury the other.
        Cosine and BM25 are not rescaled first. They are not comparable numbers.
        """
        dense_rank = {int(row): rank for rank, row in enumerate(dense_idx, start=1)}
        bm25_rank = {int(row): rank for rank, row in enumerate(bm25_idx, start=1)}
        scores: dict[int, float] = {}
        for row, rank in dense_rank.items():
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + rank)
        for row, rank in bm25_rank.items():
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(scores, key=lambda row: (-scores[row], -float(dense[row]), self.ids[row]))
        head = ordered[:FUSE_DEPTH]
        collapsed = self._collapse(head, scores, dense, dense_rank, bm25_rank)
        return collapsed[:k]

    def _collapse(self, rows: list[int], rrf: dict[int, float], dense, dense_rank, bm25_rank) -> list[dict]:
        """Greedy clusters inside this short list. Higher RRF row keeps the slot.

        Comparing every pair in the top 30 is at most a few hundred dots.
        Doing the same over the corpus is the matrix this box does not build.
        Membership is transitive inside the window: if A matches B and B
        matches C, C is filed under A (the earliest, highest-ranked row) and
        its site and url are added to also_debunked_by. The dropped row is
        not returned on its own, or the top-k would be five copies of one story.
        """
        clusters: list[dict] = []
        for row in rows:
            placed = False
            for cluster in clusters:
                if any(float(self.E[row] @ self.E[member]) >= DUP_COS for member in cluster["members"]):
                    cluster["members"].append(row)
                    cluster["hit"]["also_debunked_by"].append(self._site_url(row))
                    placed = True
                    break
            if placed:
                continue
            hit = self._hit(row, rrf[row], dense, dense_rank, bm25_rank)
            clusters.append({"members": [row], "hit": hit})
        return [cluster["hit"] for cluster in clusters]

    def _site_url(self, row: int) -> dict:
        record = self.meta.iloc[row]
        return {"site": _text(record["site"]), "url": _text(record["url"])}

    def _hit(self, row: int, rrf: float, dense, dense_rank, bm25_rank) -> dict:
        record = self.meta.iloc[row]
        return {
            "article_id": self.ids[row],
            "site": _text(record["site"]),
            "date": _iso_date(self.dates[row]),
            "primary_claim": _text(record["primary_claim"]),
            "true_version": _text(record["true_version"]),
            "verdict": _text(record["verdict"]),
            "verdict_norm": _text(record["verdict_norm"]),
            "url": _text(record["url"]),
            "domain": _text(record["domain"]),
            "manipulation_type": _text(record["manipulation_type"]),
            "form": _text(record["form"]),
            "dense_cos": float(dense[row]),
            "dense_rank": dense_rank.get(row),
            "bm25_rank": bm25_rank.get(row),
            "rrf": float(rrf),
            "also_debunked_by": [],
        }


def _text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _iso_date(value) -> str | None:
    if value is None:
        return None
    try:
        if np.isnat(np.datetime64(value, "ns")):
            return None
    except (ValueError, TypeError):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _check_domains(domains):
    """Reject a label that is not one of the 11 on the memory rows.

    The match is exact. "Disasters" is not "Disasters & accidents", and an
    unknown label used to match zero rows, which is also what a genuinely
    new claim looks like. Raising is the only way to tell those apart.
    """
    if not domains:
        return None
    if isinstance(domains, str):
        domains = [domains]
    else:
        domains = list(domains)
    unknown = [label for label in domains if label not in DOMAINS]
    if unknown:
        valid = "; ".join(DOMAINS)
        raise ValueError(f"Unknown domain {unknown!r}. Valid labels are: {valid}.")
    return domains


def get_memory() -> ClaimMemory:
    """Load the memory once. Builds it first when the manifest is stale."""
    global _LOADED
    if _LOADED is None:
        build_memory()
        _LOADED = ClaimMemory.load()
    return _LOADED


def _print_hits(hits: list[dict], elapsed: float) -> None:
    if not hits:
        print("no hits")
    for rank, hit in enumerate(hits, start=1):
        also = hit["also_debunked_by"]
        extra = ""
        if also:
            names = ", ".join(item["site"] or "?" for item in also)
            extra = f"  also: {names}"
        print(
            f"{rank}. {hit['date']}  {hit['site']}  {hit['domain']}  "
            f"cos={hit['dense_cos']:.3f}  rrf={hit['rrf']:.4f}{extra}"
        )
        print(f"   {hit['verdict']}  {hit['primary_claim']}")
        print(f"   {hit['url']}")
    print(f"query_time_s={elapsed:.3f}")


def main(argv: list[str] | None = None) -> None:
    """python -m misinfo_agent.memory "claim" [--domain X] [--before YYYY-MM-DD]"""
    parser = argparse.ArgumentParser(description="Box 5: search the claim memory")
    parser.add_argument("claim", help="claim text, as it would arrive")
    parser.add_argument("--domain", default=None, help="exact domain label, e.g. Health")
    parser.add_argument(
        "--before",
        default=None,
        help="YYYY-MM-DD, strict: only fact-checks published before this date",
    )
    args = parser.parse_args(argv)
    memory = get_memory()
    domains = [args.domain] if args.domain else None
    started = time.perf_counter()
    try:
        hits = memory.search(args.claim, k=5, date_before=args.before, domains=domains)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    _print_hits(hits, time.perf_counter() - started)


if __name__ == "__main__":
    main()
