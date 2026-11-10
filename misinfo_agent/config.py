"""Shared paths and thresholds. Every box imports these; nothing else hard-codes them.

Why one file: a threshold or a data path that lives in two modules will drift,
and box 6's decision will no longer mean what box 5 measured. state.py, this
file and requirements.txt change only through a PR with a heads-up
(docs/ARCHITECTURE.md §6), because every other module imports them.

Paths are anchored to this file, not to the process cwd. pytest, the CLI and
an IDE launch from different directories; the parquet files do not move.
"""

from pathlib import Path

# Repo root is the parent of misinfo_agent/. Parents[0] is the package dir.
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MEMORY_DIR = DATA_DIR / "memory"
EVAL_DIR = DATA_DIR / "eval"

# Read-only inputs. See DATA.md. Never load the `content` column.
CLASSIFIED_PATH = RAW_DIR / "classified_v2.parquet"
EMBEDDINGS_PATH = RAW_DIR / "bge_m3_primary_claim.parquet"

# Box 5 artefacts. Rebuilt only when the input checksums in the manifest change.
EMB_NPY_PATH = MEMORY_DIR / "emb.npy"
META_PARQUET_PATH = MEMORY_DIR / "meta.parquet"
MANIFEST_PATH = MEMORY_DIR / "manifest.json"
BM25_PATH = MEMORY_DIR / "bm25.pkl"
FAISS_INDEX_PATH = MEMORY_DIR / "faiss.index"

# Box 6 evaluation (stage D). Written locally; data/ is gitignored.
PRIOR_MATCH_TESTSET_PATH = EVAL_DIR / "prior_match_testset.csv"
PRIOR_MATCH_FIG_PATH = EVAL_DIR / "fig_prior_match_cos.png"

ENV_PATH = ROOT / ".env"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Must be the model and window that produced bge_m3_primary_claim.parquet.
# A different model, or a longer window, embeds a different vector for the
# same words, and the dot product against the stored matrix stops being a cosine.
MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LEN = 256

# The domain labels on data/memory/meta.parquet, and no others. Counted from
# those 52,099 rows (none null). Box 5 matches them exactly: "Disasters"
# is not "Disasters & accidents", and a shortened label used to return no
# hits, which looks like a claim the archive has never seen.
DOMAINS = (
    "Politics & elections",
    "Religion & communal",
    "Defence & international",
    "Crime & law",
    "Society & other",
    "Health",
    "Disasters & accidents",
    "Finance & scams",
    "Entertainment & celebrities",
    "Sports",
    "Science & technology",
)

# Provisional closeness cutoffs on unit-length BGE-M3 cosines. Stage D grid
# searches replacements and reports them; it does not write the winner back
# here, so a tuning run cannot silently change production behaviour.
TAU_HIGH = 0.85  # candidate for match_type "exact" (detail rule can still downgrade)
TAU_LOW = 0.70  # candidate for "similar"; below this the match type is "none"

# Cormack, Clarke, Buettcher: score(d) = sum 1/(k + rank). k=60 keeps rank 1
# well ahead of rank 50 without letting either list drown the other. Dense
# cosine and BM25 are not min-max scaled first; they are not on the same scale.
RRF_K = 60

# Status claims (banned, price, approved, ...) expire. A year is the default
# age after which box 6 asks for a fresh check. The image/video exception
# lives in prior_match.py, not here.
STALE_DAYS = 365

# Two memory rows at least this close are the same claim checked by two
# outlets. High on purpose: related-but-different claims must stay as separate hits.
DUP_COS = 0.95

# Religion, caste and community words. Box 6 treats a difference here as
# a different claim, not a paraphrase: "a Hindu manager turned her away"
# and "a manager turned her away" are not the same allegation. The
# community is often the part that was invented. Whole words only, so
# "islam" does not fire inside an unrelated token. TAU_HIGH and TAU_LOW
# are not derived from this list.
COMMUNITY_TERMS = (
    "hindu", "hinduism",
    "muslim", "islam", "islamic",
    "christian", "christianity", "catholic",
    "sikh", "sikhism",
    "jain", "jainism",
    "buddhist", "buddhism",
    "dalit", "brahmin", "kshatriya", "vaishya", "shudra",
    "lingayat", "yadav",
    "parsi", "jewish", "jew",
    "sunni", "shia",
    "adivasi",
)

# Words that make a claim about a *current* status. The fact can flip
# (Covaxin was "not WHO-approved" until November 2021, then it was).
STATUS_KEYWORDS = [
    "approved",
    "banned",
    "ban",
    "valid",
    "invalid",
    "discontinued",
    "withdrawn",
    "launched",
    "scheme",
    "deadline",
    "extended",
    "mandatory",
    "price",
    "rate",
    "tax",
    "notified",
    "circular",
]

# llm.py is the only module that calls a model. Swap the provider here;
# do not import a provider SDK from any other file.
LLM_PROVIDER = "gemini"

# Used when GEMINI_MODEL is unset in .env. A pinned id, not a "latest"
# alias: box 16 records the model name, and a moving alias would make two
# runs incomparable. .env overrides this so the model can change without
# a code edit. gemini-2.5-flash 404s for new keys. gemini-3.8-flash does
# not finish inside LLM_TIMEOUT_S. The lite model answers a schema call
# in about a second, which is what a veto needs.
GEMINI_MODEL = "gemini-3.5-flash-lite"

# One call must not stall the graph. The client sends this as a timeout;
# a hung request becomes an error the box can record, not an open wait.
LLM_TIMEOUT_S = 60

# On so a repeated pair is not billed again. The cache key includes the
# prompt, the schema and the model, so an edited prompt is a miss, not a
# stale answer. Files live under data/, which is gitignored.
LLM_CACHE = True
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"

# Free-tier ceiling for real Gemini calls. Cached and fake calls do not
# count. 150 is a starting cap, tuned to the project's own limit; the
# gateway reads it on every real call so a change here is the whole change.
# The day is Pacific time because that is when Gemini resets the quota.
LLM_DAILY_MAX = 150
# Real calls per 60 seconds in this process. 8 stays under the free-tier
# per-minute window with a little room for the clock being wrong.
LLM_RPM = 8

# Paired-term floor. A missing key term whose nearest word on the other
# side is below this is a different subject. 0.65 blocked "cells" against
# "kills" (cosine 0.62) and was the old default. 0.45 does not, and on the
# 70 labelled pairs it is the floor that the veto then cleans up: precision
# 1 with a much higher recall. The veto, not this floor, is what stops the
# lemon/cancer sentence and the goat/cow false exact. Callers can pass
# another floor without editing this line.
KEYTERM_SIM = 0.45

# Box 6 asks the model whether an exact pair is really the same claim.
# On. A missing key, a quota error, or a timeout must not look like a yes:
# prior_match downgrades that pair to similar. The call still cannot turn
# a similar or a none into an exact.
LLM_CONTEXT_CHECK = True
