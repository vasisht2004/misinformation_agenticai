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

# On when a person submits a forward. The evaluation sets it False.
# Scoring an article is not that article being forwarded again, and
# box 11's reach reads submission_count. A measured claim must not
# look like a crowd.
RECORD_SUBMISSIONS = True

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
# 800 stays under a typical free-tier month with room for the eval. The
# count file is data/cache/search_monthly_count.json, next to the cache
# directory so wiping cached pages does not wipe the bill.
TAVILY_MONTHLY_MAX = 800

# Real searches one claim may start. The fixed plan is at most 5 queries,
# so 8 leaves room for a retry without letting one claim spend the month.
# The counter lives in the process; the caller resets it when a claim starts.
MAX_SEARCHES_PER_CLAIM = 8

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

# A passage has to be at least this close to the claim to be kept. 0.45 is
# a floor to check against the live distribution, not a tuned threshold.
PASSAGE_MIN_COS = 0.45

# Off in the demo. The eval turns it on so a page published after the claim
# cannot count as evidence the system could have had at the time.
EVAL_DROP_AFTER_CLAIM = False

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

# Primary sources for the official query, one tuple per domain. Box 7
# restricts that one search to these hosts so a ministry page can outrank a
# blog. PIB is on every list because government denials are posted there,
# including for domains that have no single regulator. A domain with no
# entry would skip the official query; all 11 have one. Hosts were checked
# to answer over HTTPS on 11 Nov 2026 (a few apex names redirect to www).
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
        # Catch-all domain: the national portal is the only host that
        # applies to any of it.
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
