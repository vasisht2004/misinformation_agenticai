"""Shared paths and thresholds. Every box imports these; nothing else hard-codes them.

Why one file: a threshold or a data path that lives in two modules will drift,
and box 6's decision will no longer mean what box 5 measured. state.py, this
file and requirements.txt change only through a PR with a heads-up
(docs/ARCHITECTURE.md §6), because every other module imports them.

Paths are anchored to this file, not to the process cwd. pytest, the CLI and
an IDE launch from different directories; the parquet files do not move.
"""

from datetime import datetime, timezone
from pathlib import Path

# Repo root is the parent of misinfo_agent/. Parents[0] is the package dir.
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MEMORY_DIR = DATA_DIR / "memory"
EVAL_DIR = DATA_DIR / "eval"

# Box 1–2. Hashes only: sha256 and MinHash for text, sha256 and phash
# for an image. Never the message, a name, or a phone number. Tests
# replace this with a temporary file so pytest cannot write the copy
# under data/. data/ itself is gitignored.
SUBMISSIONS_DB = DATA_DIR / "submissions.db"

# Box 1–2. The downscaled JPEG that extract reads back. The file name is
# the sha256 of those bytes. ClaimState keeps the path, not the pixels:
# the LangGraph checkpointer would otherwise store every screenshot.
# Tests point this at a temporary directory. data/ is gitignored.
UPLOADS_DIR = DATA_DIR / "uploads"

# Same-file memory for photographs. pHash, the verdict, and the embedding
# of the extracted claim. Tests point this at a temporary file so pytest
# cannot write the real one.
IMAGE_MEMORY_DB = DATA_DIR / "image_memory.db"
# imagehash.phash distance. 6 is the same cutoff ingest uses for a
# repeat submission: a recompress, not a different photograph.
IMAGE_MEMORY_HAMMING = 6
# Caption search on a photograph. Off keeps the image run on provenance:
# Vision and the page fetch still run, and Tavily is not asked.
# scripts/run_images.py --no-text-search sets this for an eval that must
# not spend the monthly search budget. A text claim does not read it.
IMAGE_TEXT_SEARCH = True

# Off unless a real submission turns it on. scripts/try_claim.py does
# that with --record. Scoring a claim, and a demo of the pipeline, are
# not another person forwarding it. Box 11's reach reads
# submission_count, so a test run must not look like a crowd.
RECORD_SUBMISSIONS = False

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

# How much a false claim in this domain can hurt someone, on 0–1.
# Box 11 adds this to the priority score. Box 12 uses the same number
# as H in the confidence bar, so a domain must not be "high" in one
# box and "low" in the other.
#
# High (a person can act on it today and get hurt):
#   Health — a false cure or a false outbreak changes treatment.
#   Religion & communal — a false attack story is how a riot starts.
#   Disasters & accidents — people flee, or they stay in the way.
#   Finance & scams — the forward asks for money or a PIN.
# Medium-high (real harm, one step removed from the body or the wallet):
#   Politics & elections — can move a vote or a rumour about a candidate.
#   Crime & law — names someone as a criminal. Serious, and usually
#   checkable against a court or a police note, so it sits under health.
# Defence is not in that pair. A false strike can scare people, but the
# brief did not put it with the four that need the lowest bar, so it
# stays just under HARM_HIGH until a harmful red flag pushes it over.
# Society and science are ordinary public claims. Sports and celebrity
# news are upsetting when wrong and rarely dangerous; a death hoax is
# still Entertainment, not Health, because the harm is distress rather
# than a treatment someone skips.
DOMAIN_HARM = {
    "Health": 0.95,
    "Religion & communal": 0.95,
    "Disasters & accidents": 0.90,
    "Finance & scams": 0.90,
    "Politics & elections": 0.70,
    "Crime & law": 0.70,
    "Defence & international": 0.60,
    "Society & other": 0.45,
    "Science & technology": 0.40,
    "Entertainment & celebrities": 0.20,
    "Sports": 0.15,
}

# Added on top of the domain, then capped at 1. These four make the
# content more dangerous, not merely louder. "Urgency" and "Sensational
# style" are packaging; box 11 scores those under urgency, and scoring
# them here as well would count one shout twice. "communal targeting"
# is not one of the 11 archive labels. It is accepted when a box writes
# it, and priority.py also sets it when the claim names a community and
# a violent act.
HARM_RED_FLAGS = (
    "Fear or outrage",
    "Call to forward",
    "Fake or vague authority",
    "communal targeting",
)
# Each distinct flag. Four of them on a medium domain can reach the cap.
# One of them on health already sits at the cap, which is the point:
# health does not need a shout to be serious, and a shout cannot make
# it "more than sure".
HARM_FLAG_BUMP = 0.10
# "H high" in box 12. Above the medium-high band (0.70) and below the
# four domains that start at 0.90. Politics and crime cross it when a
# harmful flag is also present. Sports cannot, even with every flag.
HARM_HIGH = 0.80

# The warning-sign labels on classified_v2.parquet. Counted 18 Nov 2026
# from that file with columns=["red_flags"] only (the article text was
# not loaded). 52,678 rows: 22,273 had an empty list, and every other
# value was one of the eleven strings below. No other label appeared.
# The order is the order in prompts/prompt_v4a.txt, which is where the
# labels were defined. Box 3 may return a subset. Anything else is a
# schema miss and the extract call is retried.
RED_FLAGS = (
    "Urgency",
    "Fear or outrage",
    "Fake or vague authority",
    "Miracle or absolute claim",
    "Free money or guaranteed returns",
    "Request for personal data or payment",
    "Suspicious link or contact",
    "Call to forward",
    "Sensational style",
    "Emotional appeal",
    "Unverifiable source",
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

# Used once, after the primary model has already waited out a 503. Not a
# "latest" alias, for the same reason as GEMINI_MODEL. gemini-2.5-flash and
# gemini-2.5-flash-lite 404 for this key. gemini-3.1-flash-lite answered a
# one-field schema call in about a second.
GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"

# One call must not stall the graph. The client sends this as a timeout;
# a hung request becomes an error the box can record, not an open wait.
LLM_TIMEOUT_S = 60

# On so a repeated pair is not billed again. The cache key includes the
# prompt, the schema and the model, so an edited prompt is a miss, not a
# stale answer. Files live under data/, which is gitignored.
LLM_CACHE = True
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"

# Free-tier ceiling for real Gemini calls. Cached and fake calls do not
# count. 400 is 80% of the real free-tier limit of 500 requests/day for
# gemini-3.5-flash-lite. The gateway reads this on every real call, so a
# change here is the whole change. The day is Pacific time because that
# is when Gemini resets the quota.
LLM_DAILY_MAX = 400
# Real calls per 60 seconds in this process. 8 stays under the free-tier
# per-minute window with a little room for the clock being wrong. The RPM
# limiter is per process, so never run two pipeline processes at once
# (two parallel processes exceeded Google's 15/min limit).
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

# search.py is the only module that calls Tavily. On so a repeated query is
# not billed again. The cache key is the query plus the search params,
# including the exclude list, so turning fact-check blocking on is a miss
# and cannot replay an open-web result. Files live under data/, which is
# gitignored.
SEARCH_CACHE = True
SEARCH_CACHE_DIR = DATA_DIR / "cache" / "search"

# Real Tavily HTTP calls this UTC month. Cached and fake calls do not count.
# Tavily dashboard 22 Nov: 709/1000 real credits used; our counter read 800
# (it over-counts). 1000 = 800 + 200 → leaves ~30 real credits of margin.
# Revert to 800 on 1 Dec. The count file is
# data/cache/search_monthly_count.json, next to the cache directory so
# wiping cached pages does not wipe the bill. The counter itself resets
# with the UTC month, so the extra 200 is only for the rest of November.
# A process that imported this before 1 Dec keeps 1000 until it restarts.
_TAVILY_CAP_REVERTS = datetime(2026, 12, 1, tzinfo=timezone.utc)
TAVILY_MONTHLY_MAX = 800 if datetime.now(timezone.utc) >= _TAVILY_CAP_REVERTS else 1000

# Real searches one claim may start. The fixed plan is at most 5 queries,
# so 8 leaves room for a retry without letting one claim spend the month.
# The counter lives in the process; the caller resets it when a claim starts.
MAX_SEARCHES_PER_CLAIM = 8

# Google Vision web detection for an event photograph. vision.py is the
# only module that calls it. 300 is our cap, not Google's. Cached and
# fake calls do not count. The counter file sits beside this directory
# so wiping cached detections does not wipe the bill.
VISION_MONTHLY_MAX = 300
VISION_CACHE = True
VISION_CACHE_DIR = DATA_DIR / "cache" / "vision"
# Pages returned for one photograph. 20 was the first setting. A cached
# answer from that request does not contain the extra pages, so the cache
# key includes this number and the photograph is asked again.
VISION_MAX_RESULTS = 50

# Photo gate. Both on: an earlier or different-event page can refute a
# picture as out of context, and two full tier 1-2 matches can support
# it. Off, a picture can still be refuted by text and cannot be confirmed
# by text, which is the rule these flags sit on top of.
GATE_PHOTO_CONTEXT = True
GATE_PHOTO_SUPPORT = True
# Two provenance groups of any tier can refute a picture when none of
# them, at any tier, presents the file as this event. The tier 1-2 photo
# rule is unchanged. The card says no tier 1-2 page was found, and the
# confidence is capped, because that bar was not met.
GATE_PHOTO_WEAK_REFUTE = True
PHOTO_WEAK_REFUTE_CONFIDENCE_CAP = 0.6
PHOTO_WEAK_REFUTE_CAVEAT = "Lower-tier sources only: no tier 1-2 page was found."
# A page dated before the claimed event minus this many days is an earlier
# event. The same window is how close a supporting page has to be to
# the claim to count as the same occasion: a forward often lands a day
# or two after the story. The event day comes from the caption, not from
# today. See provenance.claim_event_date.
PHOTO_EARLIER_DAYS = 2
# Which pages may date a photograph.
# v0: full matches only.
# v1: full and partial. Two dated-earlier groups, or one tier 1-2 group.
# v2: v1, but a partial counts only with a trusted date at least
#     PHOTO_PARTIAL_MIN_DAYS before the event starts.
# v3: v2, and a tier 1-2 full match of this event blocks out of context.
# The comparison that picked the value is in docs/eval_diagnosis.md.
PHOTO_RULE = "v2"
PHOTO_PARTIAL_MIN_DAYS = 365

# Investigate rounds the gate and the decider may already have used.
# The first call usually sees 1 (one search finished). While the count
# is still under this, a thin table or a failed stance check may ask
# for another round. At this count the answer is final: insufficient,
# or error if the stance call itself failed. Two is the whole budget,
# not two extra on top of an uncounted first search.
MAX_SEARCH_ROUNDS = 2

# Hosts of the 15 fact-check publications in classified_v2.parquet.
# Built once on 11 Nov 2026: read columns=["url"] only (52,678 rows, none
# empty), take the hostname, lowercase it, and strip a leading "www.".
# That produced 15 hosts. Seven were stored with www. (boomlive.in,
# indiatoday.in, lighthousejournalism.com, ptinews.com, telugupost.com,
# thequint.com, vishvasnews.com); the bare host is what we keep.
# search() blocks the registrable domain of each host, so a subdomain is
# the same publication: english.factcrescendo.com blocks factcrescendo.com,
# and en.youturn.in blocks youturn.in. The list is a literal so box 7 does
# not need the parquet present just to know who to exclude.
FACTCHECK_DOMAINS = [
    "boomlive.in",
    "dfrac.org",
    "digiteye.in",
    "en.youturn.in",
    "english.factcrescendo.com",
    "factly.in",
    "firstcheck.in",
    "indiatoday.in",
    "lighthousejournalism.com",
    "newschecker.in",
    "newsmeter.in",
    "ptinews.com",
    "telugupost.com",
    "thequint.com",
    "vishvasnews.com",
]

# Not exhaustive. International desks and Indian fact-checkers that are not
# in the 15 archive hosts. A missing desk can still be dropped by the path
# and title rules in search(). factcheck.afp.com is the desk, not afp.com:
# the extra list is not expanded to the registrable parent, or every AFP
# news story would disappear. vishvasnews.com is already in the archive
# list; it is repeated so this list can be read on its own. newsmobile.in
# also publishes news. It is listed because its fact-check desk is what
# search returned.
EXTRA_FACTCHECK_DOMAINS = [
    "factcheck.afp.com",
    "factcheck.org",
    "fullfact.org",
    "politifact.com",
    "snopes.com",
    "leadstories.com",
    "logically.ai",
    "newsmobile.in",
    "vishvasnews.com",
    "altnews.in",
    "factchecker.in",
    # Seen in a live run quoting a claim as if it had checked it. The name
    # rule below would also catch factcheckhub.com; it is listed so the
    # block does not depend on that spelling guess.
    "factcheckhub.com",
    "checkyourfact.com",
    "africacheck.org",
    # Already in the archive list. Repeated so this desk is visible here.
    "factly.in",
]

# Default off so a demo can still open those pages. The evaluation turns it
# on: the honest number is the one the system gets without reading the
# fact-check it is being scored against. search() then excludes the
# registrable archive domains and the extra hosts, and drops a hit that
# comes back on one of them, looks like a fact-check by path or title, or
# is a PIB Fact Check post. A normal pib.gov.in press release stays.
BLOCK_FACTCHECK_SITES = False

# A support row counts for the core claim only when its quote carries the
# core's own key terms: every number, amount, percent, and date the core
# states, plus the main named entity and the action or status word. A page
# that confirms a side detail (the fare, the crowd) was being counted as
# support for the headline. Those rows become neutral for the core and are
# listed in core_detail_only_ids. With this on, "supported" also needs two
# independent tier 1-2 groups, not one credible group plus a blog.
# False restores the older bar exactly: two groups, at least one tier 1-2.
TRUE_NEEDS_CORE_MATCH = True

# "strict" is the first matcher: the full name, and the exact action word,
# on the quote alone. "relaxed" is the default because a news sentence
# paraphrases ("Mamdani won", "Mamdani's win", "Mamdani defeated") and the
# strict rule was dropping those. Numbers, amounts, percents, and dates
# still have to match exactly. A drifted figure is a different claim.
CORE_MATCH_MODE = "relaxed"

# One line per action class. Written from general knowledge of how an
# English news sentence paraphrases a verb. Not taken from the 50 eval
# claims or their passages.
ACTION_SYNONYMS = {
    "win": ("win", "won", "wins", "winning", "victory", "victorious", "defeated", "beat", "elected"),
    "ban": ("ban", "banned", "bans", "prohibit", "prohibited", "outlawed"),
    "launch": ("launch", "launched", "unveil", "unveiled", "introduce", "introduced", "roll out", "rolled out"),
    "die": ("die", "died", "dead", "death", "killed", "passed away"),
    "arrest": ("arrest", "arrested", "detained", "held", "taken into custody"),
    "announce": ("announce", "announced", "declare", "declared", "notified", "notification"),
    "approve": ("approve", "approved", "approval", "sanction", "sanctioned"),
    "inaugurate": ("inaugurate", "inaugurated", "inauguration"),
    "resign": ("resign", "resigned", "resignation", "quit", "stepped down"),
    "close": ("close", "closed", "closure", "shut", "shutdown"),
    "open": ("open", "opened", "opening", "reopened"),
    "issue": ("issue", "issued", "issuance"),
    "order": ("order", "ordered", "directed", "directive"),
    "deny": ("deny", "denied", "denial"),
    "withdraw": ("withdraw", "withdrawn", "scrap", "scrapped", "revoke", "revoked"),
    "invalid": ("invalid", "invalidated", "void"),
    "attack": ("attack", "attacked", "assault", "assaulted"),
    "show": ("show", "shows", "showed", "shown", "footage"),
    "happen": ("happen", "happened", "occurred", "took place"),
}

# One tier-1 denial is enough to refute the core when no tier 1-2 group
# supports it. Two newsrooms were required so a single blog could not
# convict a claim; a ministry or a regulator is not a blog. The gate
# reason names the host. False keeps the two-group refute rule.
TIER1_SINGLE_REFUTE = True

# Gate variants. All default off: the shipped rule stays "two independent
# groups, at least one tier 1 or 2". A study may turn one on after the
# selection rule in docs/eval_diagnosis.md has been applied on dev and
# the winner has been scored once on test. Nothing here is fitted to a
# claim. G0 in that study is every flag below left off.
#
# G2R. Three independent refutations and nobody on the other side can
# refute the core even when every page is tier 3 or 4. Support is never
# decided by a pile of lower-tier pages. The card says no tier 1-2 source
# was found, and the confidence is capped, because the usual bar was not met.
GATE_WEAK_REFUTE = True
WEAK_REFUTE_MIN_GROUPS = 3
WEAK_REFUTE_CONFIDENCE_CAP = 0.6
WEAK_REFUTE_CAVEAT = "Lower-tier sources only: no tier 1-2 source was found."
# G2R-2. The same path as G2R, but two independent refuting groups are
# enough, and the confidence cap is lower. Zero support at any tier, and
# every counted refute still has to pass the relaxed core match. Support
# is never decided this way. The on/off decision is the combined 16-fact
# and 50-claim rule in docs/eval_diagnosis.md. Do not flip this because
# one claim looked better.
GATE_WEAK_REFUTE_2 = True
WEAK_REFUTE_2_MIN_GROUPS = 2
WEAK_REFUTE_2_CONFIDENCE_CAP = 0.5
# G1. One tier-1 page from the office the claim is about can support or
# refute on its own when no tier 1-2 group is on the other side. Any other
# tier-1 site does not: a ministry page about a different subject is not
# the body named in the sentence.
GATE_PRIMARY_SOURCE = True
# G3. Two outlets that repeat one press release are one source. Merging
# them makes the gate stricter, not looser. A shared run of this many
# words is a copied sentence; a shorter overlap is just the same topic.
GATE_SHARED_SOURCE = True
GATE_SHARED_QUOTE_WORDS = 12
# G4. One number per group, summed. Tier weights are a principle: an
# office is a full vote, a newsroom is close behind, an encyclopaedia is
# about a third, and an unknown host is a small fraction. They were not
# fit on the eval. Theta is chosen on dev from the grid and does nothing
# while the flag is off. 1.0 is the middle of that grid, not a fitted result.
GATE_EVIDENCE_SCORE = False
GATE_EVIDENCE_THETA = 1.0
GATE_EVIDENCE_THETA_GRID = (0.8, 1.0, 1.2, 1.5, 2.0)
GATE_TIER_WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.35, 4: 0.15}

# True is the old pause: review() calls interrupt() and the process waits.
# False lets the run finish. The verdict stays provisional, the pending
# row stays in the queue, and a person approves it later without a re-run.
HUMAN_BLOCKING = False

# When the rules say Unverifiable, one baseline-style call records which
# label the model would have picked. It is stored as model_lean. It does
# not change the verdict, the confidence, the priority, or the decision.
ASSISTED_LEAN = True

# A passage has to be at least this close to the claim to be kept. 0.45 is
# a floor to check against the live distribution, not a tuned threshold.
PASSAGE_MIN_COS = 0.45

# Off in the demo. The eval turns it on so a page published after the claim
# cannot count as evidence the system could have had at the time.
EVAL_DROP_AFTER_CLAIM = False

# A relative-time claim ("today", "till Monday") is about this fortnight.
# Investigate forces a trusted page dated more than this many days before
# the claim, or any trusted page dated after the claim, to neutral: that
# page is a different occasion. A dated status claim ("invalid from 1
# December", "banned from 2026") is not in this window, so an old page
# can still refute it. Unknown dates are kept and flagged.
RELATIVE_TIME_WINDOW_DAYS = 14

# Stance is frozen at stance_v9 (misinfo_agent/prompts/stance_v9.txt,
# prompt_version "stance_v9" in investigate.judge_stance). A change to
# that prompt, or to the code that rewrites a stance label, needs a full
# regression on the hand-labelled rows in data/eval/day3_evidence.csv
# before it becomes the default. Do not ship a new version off one live claim.

# "keep" leaves a passage whose date we never found. "drop" removes it,
# and only while the leakage guard is on: a demo with no claim date would
# otherwise throw away most of the web. The two settings are counted
# separately so the eval can show what the drop would cost.
EVAL_UNDATED = "keep"

# Added to the cosine for each key term from the claim that the passage
# actually contains, as a fraction of those terms. 0.05 is a nudge: it
# can reorder two close windows, and it cannot rescue a window under the
# cosine floor.
ENTITY_BOOST = 0.05

# The whole claim: search, article fetch, then date-only reads. A date
# read starts only while time remains, so a slow search cannot grow a
# second download of every page.
CLAIM_BUDGET_S = 60

# A date-only read is not an article fetch. The publication meta is in
# the head, so the first 100 KB is enough, and four seconds is the whole
# read. A miss leaves the passage undated.
DATE_FETCH_TIMEOUT_S = 4
DATE_FETCH_BYTES = 100_000

# Fetched article bodies. data/ is gitignored. The key is the canonical URL,
# so a tracking parameter does not store the same page twice.
PAGE_CACHE_DIR = DATA_DIR / "cache" / "pages"

# How many hits of one claim are downloaded. The list is already in
# reserved-slot order, so this reaches the official page before the tail.
# The rest keep the search snippet. Circulation hosts do not use a slot:
# there is no article on them to fetch.
FETCH_MAX = 12

# Six seconds for the whole download, not for one socket read. A host
# that keeps sending a little data never trips a per-read timeout, and
# that is what held a claim for minutes. A miss falls back to the snippet.
FETCH_TIMEOUT_S = 6

# A normal browser string. Sites that reject a bot name still serve the
# circular to a browser. Nothing here identifies a person or this project.
FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Posts and stock libraries. A repost shows that the claim is circulating.
# It does not show whether the claim is true, so these hosts are counted
# as sightings and never turned into passages. Matched on the registrable
# domain: m.facebook.com is facebook.com.
NON_EVIDENCE_HOSTS = (
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "t.me",
    "whatsapp.com",
    "gettyimages.com",
    "shutterstock.com",
    "pinterest.com",
    "alamy.com",
    # User posts. A thread can repeat the claim; it is not a source for it.
    "reddit.com",
    "linkedin.com",
    "quora.com",
)

# Credibility tiers. The number describes the source, not how we obtained
# the text: a ministry snippet is still a ministry. snippet_only on the
# evidence row records that we only saw the search snippet.
# Tier 1 is a body that can issue the order the claim describes: a
# government suffix, or one of the offices the official query already asks.
# Tier 2 is a newsroom or a journal with an editor and a corrections
# process. Tier 3 is a real organisation that is not that newsroom: an
# encyclopedia, a hospital, an NGO, or a smaller paper. Tier 4 is a blog
# or a host we have not checked.
# Suffixes are matched on the registrable domain. ".gov." means a .gov.
# label anywhere (pib.gov.in, gov.uk), not only a host that ends in .gov.
TIER1_SUFFIXES = (".gov", ".gov.", ".gob.", ".gc.ca", ".nic.in")
# International bodies that are not under .gov.in. The ministries in
# OFFICIAL_DOMAINS are tier 1 as well; the tier function reads that map
# so a domain we already route to an office is not scored as an unknown site.
TIER1_HOSTS = (
    "who.int",
    "un.org",
    "imf.org",
    "worldbank.org",
    # Indian offices that are not under .gov / .nic. A suffix rule never
    # sees rbi.org.in, npci.org.in, or sansad.in, so a page from the bank,
    # the payments corporation, or Parliament was scored as an unknown
    # site. The .gov.in and .nic.in hosts are listed as well so the tier
    # does not depend on the suffix table alone.
    "rbi.org.in",
    "npci.org.in",
    "uidai.gov.in",
    "isro.gov.in",
    "rajbhasha.gov.in",
    "legislative.gov.in",
    "indiacode.nic.in",
    "eci.gov.in",
    "mea.gov.in",
    "sebi.gov.in",
    "sansad.in",
)
TIER2_HOSTS = (
    # Wires. PTI's English wire is ptinews.com. ANI, IANS, and UNI are the
    # other national wires a newsroom copies from. A wire has an editor and
    # a corrections desk, which is the tier-2 bar, so their own sites are
    # not unknown hosts. ianslive.in is IANS's live host. Its registrable
    # domain is not ians.in, so listing only ians.in would leave it at tier 4.
    "ptinews.com",
    "aninews.in",
    "ians.in",
    "ianslive.in",
    "uniindia.com",
    "reuters.com",
    "apnews.com",
    # Papers and broadcasters a reader can hold to a corrections desk.
    "bbc.com",
    "bbc.co.uk",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "ndtv.com",
    "livemint.com",
    # indiatimes.com is the registrable domain of timesofindia and of
    # economictimes.indiatimes.com. economictimes.com is the mobile host
    # (m.economictimes.com). Matching the registrable domain is what puts
    # both on this tier; listing only the full subdomain left the mobile
    # host on tier 4.
    "indiatimes.com",
    "economictimes.com",
    "moneycontrol.com",
    "theprint.in",
    "scroll.in",
    "aljazeera.com",
    "cnbc.com",
    "nytimes.com",
    # Major outlets that were tier 3 until the registrable-domain pass.
    # A corrections desk is the bar, not the country of the newsroom.
    "dw.com",
    "euronews.com",
    "globalnews.ca",
    "france24.com",
    "theguardian.com",
    "washingtonpost.com",
    "cbsnews.com",
    "nbcnews.com",
    "cnn.com",
    "indiatoday.in",
    "deccanherald.com",
    "newindianexpress.com",
    "telegraphindia.com",
    "business-standard.com",
    "financialexpress.com",
    # More national newsrooms. The tier is editorial standards: a named
    # newsroom and a corrections policy, not political leaning. scroll.in,
    # livemint.com, hindustantimes.com, and indiatoday.in are already
    # listed above. frontline.thehindu.com is thehindu.com once the
    # registrable domain is taken, and that host is already tier 2.
    # thequint.com's news pages are this tier. Its WebQoof path stays a
    # fact-checker and is still blocked when the fact-check block is on.
    "tribuneindia.com",
    "deccanchronicle.com",
    "news18.com",
    "firstpost.com",
    "outlookindia.com",
    "theweek.in",
    "businesstoday.in",
    "wionews.com",
    "mid-day.com",
    "freepressjournal.in",
    "thewire.in",
    "thequint.com",
    "frontline.thehindu.com",
    "abplive.com",
    "zeebiz.com",
    # Publishers of peer-reviewed journals. A blog that cites a paper is
    # not this list; the journal's own host is.
    "nature.com",
    "science.org",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "pnas.org",
    "cell.com",
    "springer.com",
    "wiley.com",
    "sciencedirect.com",
    "jamanetwork.com",
    "plos.org",
    # National academies. A university is matched by suffix, not by name.
    "nationalacademies.org",
)
# ".edu." and ".ac." cover uq.edu.au and ox.ac.uk, not only .edu and .ac.in.
TIER2_SUFFIXES = (".edu", ".edu.", ".ac.")
TIER3_HOSTS = (
    "wikipedia.org",
    # Smaller newsrooms. Not the tier-2 list, still a newsroom.
    "bloomberg.com",
    "forbes.com",
    "time.com",
    "gulfnews.com",
    "pbs.org",
    "npr.org",
    "abc.net.au",
    "news9live.com",
    # Hospitals and NGOs. They can describe a health or relief fact.
    # They are not the wire that reported the political claim.
    "mayoclinic.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "stvincents.org",
    "redcross.org",
    "unicef.org",
    "oxfam.org",
    "msf.org",
    "amnesty.org",
    "hrw.org",
)
# Hosts that publish anyone. A post there is not a newsroom.
TIER4_HOSTS = (
    "medium.com",
    "substack.com",
    "blogspot.com",
    "blogger.com",
    "wordpress.com",
)

# Added to every domain's official query, after that domain's own hosts.
# A claim such as "how many states does India have" is not filed under one
# regulator, so the domain lists never asked the census or Know India.
# PIB is repeated from the domain lists; the query builder drops the copy.
# This changes searches from here on. The blocked-eval cache was stored
# before these hosts were added, so a replay of that cache will not show them.
NATIONAL_OFFICIAL_DOMAINS = (
    "india.gov.in",
    "knowindia.india.gov.in",
    "mha.gov.in",
    "censusindia.gov.in",
    "pib.gov.in",
    "mospi.gov.in",
)

# The office a claim is about. Used only by the G1 gate flag. A tier-1 host
# counts as that office when the claim names it. The phrases are the public
# name of the host, not strings copied from an eval claim. An acronym of
# three letters or fewer has to appear in that capitalisation: "who" inside
# "who won" is not the World Health Organization.
PRIMARY_BODY_NAMES = {
    "eci.gov.in": ("Election Commission of India", "Election Commission", "ECI"),
    "pib.gov.in": ("Press Information Bureau", "PIB"),
    "minorityaffairs.gov.in": ("Ministry of Minority Affairs",),
    "mod.gov.in": ("Ministry of Defence", "Ministry of Defense"),
    "mea.gov.in": ("Ministry of External Affairs", "MEA"),
    "mha.gov.in": ("Ministry of Home Affairs", "MHA"),
    "ncrb.gov.in": ("National Crime Records Bureau", "NCRB"),
    "india.gov.in": ("National Portal of India",),
    "knowindia.india.gov.in": ("Know India",),
    "censusindia.gov.in": ("Census of India", "Registrar General"),
    "mospi.gov.in": ("Ministry of Statistics", "MOSPI"),
    "mohfw.gov.in": ("Ministry of Health", "MoHFW"),
    "icmr.gov.in": ("Indian Council of Medical Research", "ICMR"),
    "who.int": ("World Health Organization", "World Health Organisation", "WHO"),
    "nih.gov": ("National Institutes of Health", "NIH"),
    "ncbi.nlm.nih.gov": ("PubMed", "NCBI"),
    "diabetes.org": ("American Diabetes Association",),
    "idf.org": ("International Diabetes Federation", "IDF"),
    "ndma.gov.in": ("National Disaster Management Authority", "NDMA"),
    "mausam.imd.gov.in": ("India Meteorological Department", "IMD"),
    "rbi.org.in": ("Reserve Bank of India", "RBI"),
    "sebi.gov.in": ("Securities and Exchange Board of India", "SEBI"),
    "incometax.gov.in": ("Income Tax Department",),
    "mib.gov.in": ("Ministry of Information and Broadcasting",),
    "yas.gov.in": ("Ministry of Youth Affairs and Sports",),
    "dst.gov.in": ("Department of Science and Technology",),
    "isro.gov.in": ("Indian Space Research Organisation", "Indian Space Research Organization", "ISRO"),
    "meity.gov.in": ("Ministry of Electronics and Information Technology", "MeitY"),
}

# Primary sources for the official query, one tuple per domain. Box 7
# restricts that one search to these hosts so a ministry page can outrank a
# blog. PIB is on every list because government denials are posted there,
# including for domains that have no single regulator. A domain with no
# entry would skip the official query; all 11 have one. Hosts were checked
# to answer over HTTPS on 11 Nov 2026 (a few apex names redirect to www).
# NATIONAL_OFFICIAL_DOMAINS is appended to each tuple at query time.
OFFICIAL_DOMAINS = {
    "Politics & elections": (
        # Rolls, results, and the model code of conduct.
        "eci.gov.in",
        # Government statements about a political claim.
        "pib.gov.in",
    ),
    "Religion & communal": (
        # The ministry that speaks on minority and communal policy. A
        # religious body is not a public record of what happened.
        "minorityaffairs.gov.in",
        # Official clarifications of communal rumours are press notes.
        "pib.gov.in",
    ),
    "Defence & international": (
        # Armed forces announcements.
        "mod.gov.in",
        # Treaties, talks, and anything that happened outside India.
        "mea.gov.in",
        "pib.gov.in",
    ),
    "Crime & law": (
        # Policing and law-and-order statements.
        "mha.gov.in",
        # The crime statistics those claims quote.
        "ncrb.gov.in",
        "pib.gov.in",
    ),
    "Society & other": (
        # Catch-all domain. The national list added to every official query
        # supplies the census, home affairs, and Know India on top of these.
        "india.gov.in",
        "pib.gov.in",
    ),
    "Health": (
        # Ministry advisories and scheme notices.
        "mohfw.gov.in",
        # Treatment and outbreak claims that cite a medical body.
        "icmr.gov.in",
        # Indian health forwards constantly cite WHO. It is the primary
        # source for those sentences, not a news site.
        "who.int",
        # A cure claim is checked against trials and reviews. Those
        # papers are on NIH, and PubMed is the ncbi host under it.
        "nih.gov",
        "ncbi.nlm.nih.gov",
        # Diabetes-specific guidance. A "garlic cures diabetes" forward
        # is the kind of sentence these two bodies publish on.
        "diabetes.org",
        "idf.org",
        "pib.gov.in",
    ),
    "Disasters & accidents": (
        # Situation reports and relief.
        "ndma.gov.in",
        # Forecasts, rainfall, cyclone and earthquake bulletins. The public
        # site is this host; imd.gov.in itself did not answer.
        "mausam.imd.gov.in",
        # Official tolls.
        "pib.gov.in",
    ),
    "Finance & scams": (
        # Currency and banking circulars. A note-ban rumour starts here.
        "rbi.org.in",
        "pib.gov.in",
        # Markets, tips, and investment scams.
        "sebi.gov.in",
        # Tax, refund, and notice rumours.
        "incometax.gov.in",
    ),
    "Entertainment & celebrities": (
        # Film certification and broadcasting. There is no official
        # celebrity registry; this is the office that confirms what the
        # government actually said.
        "mib.gov.in",
        "pib.gov.in",
    ),
    "Sports": (
        # Ministry of Youth Affairs and Sports. yas.nic.in did not answer;
        # this host did. BCCI is a private board, not a public record.
        "yas.gov.in",
        "pib.gov.in",
    ),
    "Science & technology": (
        # The department that funds and announces science.
        "dst.gov.in",
        # Space claims. The public site is www.isro.gov.in.
        "isro.gov.in",
        # Digital and technology policy.
        "meity.gov.in",
        "pib.gov.in",
    ),
}

# Box 11. Hand-set. scripts/priority_sensitivity.py moves each one by
# 20% and reports how many fixture levels change. They sum to 1 so the
# score stays in [0, 1] when every component does.
# Harm is the largest weight because a calm health rumour still matters
# more than a shouted sports score. Reach is next: a claim already in
# a thousand chats is the one a person should see first. Urgency is
# time, which expires. Novelty is the smallest: a new wording of an old
# rumour is not, by itself, an emergency.
PRIORITY_WEIGHTS = {
    "harm": 0.40,
    "novelty": 0.15,
    "reach": 0.25,
    "urgency": 0.20,
}
PRIORITY_HIGH = 0.60
PRIORITY_MEDIUM = 0.35

# Reach is log-scaled. 20 is "this is already everywhere": several
# repeat submissions, a handful of outlets, and WhatsApp's own viral
# label. Past that the score stays at 1 so a 500th repost cannot drown
# harm. The two forward labels are not equal. "Forwarded many times" is
# WhatsApp saying the message left the original chat and kept going.
# "Forwarded as received" only says the sender did not type it. 4 against
# 1 makes the viral label move the log about as much as several repeats,
# and keeps a single ordinary forward visible but short of a crowd.
REACH_LOG_CAP = 20
REACH_FORWARD_MANY = 4
REACH_FORWARD_RECEIVED = 1

# Urgency points, capped at 1 inside priority.py. The word cap stops
# five synonyms of "breaking" from saturating the component on their own.
# Share is separate because a forward request is a different act from a
# banner word. Deadline and a near date share one bucket: both say the
# claim is time-critical, and counting them twice would treat "tomorrow's
# deadline" as twice as urgent as a deadline next month.
URGENCY_PER_WORD = 0.30
URGENCY_WORD_CAP = 0.60
URGENCY_SHARE = 0.35
URGENCY_DEADLINE = 0.40
# Three weeks. A "from 1 December" rumour checked on 15 November is a
# deadline, not a historical detail. A date next season is the subject
# of the claim and does not raise urgency.
URGENCY_NEAR_DAYS = 21

# Box 12. p* = P_STAR_A / (P_STAR_A + P_STAR_B * H).
# H = 0 → 1.0, so a harmless claim must be certain before anyone acts.
# H = 1 → 0.25, so a dangerous claim is acted on at a lower confidence.
# The margin is how close "almost sure" has to be before another search
# is worth more than sending the file to a person.
P_STAR_A = 1.0
P_STAR_B = 3.0
P_STAR_MARGIN = 0.15

# Every alert waits at box 13. The decision string stays "alert" so the
# card can say what was requested; the pause is this flag, not a rename.
ALERT_NEEDS_APPROVAL = True

# Box 10. Provisional. The eval reports calibration (ECE) and is what
# should move these numbers. They are log-odds, not percentages: the
# assess box squashes the weighted sum with a logistic into [0, 1].
# groups is the largest positive term because two credible outlets are
# the product's own bar and four is already a pile. agreement is next
# so a named dissenter actually moves the score. A top-up, a second
# search round, and unverified quotes pull it down: each one means the
# table was thinner than the first pass wanted.
CONFIDENCE_BIAS = -0.4
CONFIDENCE_WEIGHTS = {
    "groups": 1.8,
    "tier": 0.8,
    "agreement": 1.2,
    "dated": 0.4,
    "unverified": -0.8,
    "topup": -0.4,
    "rounds": -0.3,
}
# Confidence that the claim is unverifiable, not that it is false.
# rounds and thin raise it: we used the search and we are still short
# of two credible groups. A failed quote check lowers it.
CONFIDENCE_UNVERIFIABLE_BIAS = 0.0
CONFIDENCE_UNVERIFIABLE_WEIGHTS = {
    "rounds": 1.1,
    "thin": 1.3,
    "topup": 0.3,
    "unverified": -0.7,
}
# reuse_prior has no new table. The cosine is the match strength.
# 0.85 (the exact threshold) lands lower than a near-copy at 0.96.
CONFIDENCE_REUSE_BIAS = -2.2
CONFIDENCE_REUSE_COSINE = 4.0
