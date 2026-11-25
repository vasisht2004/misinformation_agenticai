# Project context — misinformation_agenticai

A single map of this repository: what the system is, how a claim moves through it, and what every folder, module, prompt, script, document, and test is for.

The design contract is `docs/ARCHITECTURE.md`. Box numbers in code, comments, and this file are the 16 boxes defined there. If this file and the code disagree, the code and `ARCHITECTURE.md` win.

## What it does

An agentic checker for things people share in India: a WhatsApp-style text forward, a link to an article or post, or an image (screenshot or photo, with an optional caption). English only. Hinglish is accepted and flagged. Hindi in Devanagari, video, audio, and deepfake forensics are out of scope. Opinions, jokes, predictions, and greetings are not checked; the verdict is **Other**.

The result is:

- a verdict: **False / Misleading / True / Unverifiable / Other**
- a confidence computed from the evidence table, never from the model's self-report
- past fact-checks the claim matches ("seen before")
- live web evidence, each source with a credibility tier
- red flags, manipulation type when known, a priority, and a plain-language explanation
- "what is actually true" and a short message a person could forward

**Core principle.** The language model reads and writes. Evidence and explicit rules decide the verdict. The archive is about 70% False and under 0.5% True, and facts change, so a model trained on past fact-checks cannot be the judge. The system says **Unverifiable** when it cannot find two independent credible sources.

Typical cost of one claim: 3–6 model calls, about 5–7 searches, about 40 seconds.

Remote: `https://github.com/vasisht2004/misinformation_agenticai.git`. Work on `feat/<area>`. Pull requests go into `main` and are reviewed by Vash. Never push straight to `main`. Never commit `data/` or `.env`.

## How a claim moves

```
1–2 ingest → 3–4 extract → 5 memory → 6 prior match
                                      ├─ exact and fresh → reuse prior → 8+10 assess
                                      └─ otherwise → 7 investigate → 9 gate
                                                     ▲ search_more (max 2 rounds)
                                         conflicting → 13 human review
                                         sufficient or insufficient → 8+10 assess
            11 priority ← assess
                 │
            12 decide → alert / unsure → 13 human review → 14–15 output → 16 audit
                     └─ store ─────────────────────────→ 14–15 output → 16 audit
```

One `ClaimState` dict (`misinfo_agent/state.py`) travels through every box. A node returns only the keys it changes. `evidence` and `errors` use a list reducer so a later round appends instead of wiping the earlier rows. LangGraph (`misinfo_agent/graph.py`) is the wiring: one node per box, conditional edges after ingest, extract, memory, prior match, investigate, gate, assess, priority, decide, review, output, and audit. The checkpointer is SQLite (`data/checkpoints.db`) so a review pause can resume after the process that asked the question has exited. One `thread_id` is one run. A node that raises, other than a quota error, is recorded on `errors` and sent to review. A quota error is not caught: the day or the month is spent, and the next node would fail the same way.

| Box | Module | What it does | Model? |
|---|---|---|---|
| 1–2 Ingest | `ingest.py` | Text, URL, or image → cleaned `text` and `match_text`. Language, quality, duplicate hash. Style signals for the card and for priority. | no |
| 3–4 Extract | `extract.py` | One checkable claim in archive style, atomic claims, entities, domain, red flags. Image: visual description and location/time clues. | 1 call (`extract_v2`) |
| 5 Memory | `memory.py` | Hybrid search over 52,099 past claims: BGE-M3 dense plus BM25, merged with reciprocal rank fusion. Near-duplicates collapse into `also_debunked_by`. | no |
| 6 Prior match | `prior_match.py` | exact / similar / none. Detail, community, and expiry rules can downgrade exact. A model veto can only turn exact into similar. Route is `reuse_prior` or `investigate`. | 0–1 call, only if the rules left the pair exact |
| 7 Investigate | `investigate.py` plus `search.py` and `fetch.py` | Fixed query plan, Tavily search, SSRF-checked fetch, passage pick, stance, credibility tiers, independence groups. Optional repair and one top-up. | 1–3 calls (`stance_v9`, `repair_v1`, `topup_v1`) |
| 9 Gate | `gate.py` | Two independent groups on the same side, at least one tier 1–2, no blocking tier-1 disagreement → sufficient. Credible conflict → human. Else search again (max 2 rounds) then insufficient. | no |
| 8+10 Assess | `assess.py` | The model drafts explanations. Code picks the verdict, checks citations, and scores confidence. The model's label is `model_lean` and is not the verdict. An exact fresh prior is copied from the archive and does not call the model. | 0–1 call (`assess_v2`) |
| 11 Priority | `priority.py` | Harm, novelty, reach, urgency → High / Medium / Low. Style can move this. It cannot move the verdict. | no |
| 12 Decide | `decide.py` | alert / search_more / human / store. Bar is `p* = a / (a + bH)`. Higher harm, lower bar. | no |
| 13 Human review | `review.py` | Non-blocking. The card is written and the row stays on the queue. Approve, edit, or reject updates that finished thread. `HUMAN_BLOCKING` is the old pause. | no |
| 14–15 Output | `output.py` | Result card. The Streamlit app draws it. | no |
| 16 Audit | `audit.py` | One SQLite row per finished run. Approved verdicts can be appended for later memory. | no |
| Wiring | `graph.py` | LangGraph `StateGraph`. | — |
| Model access | `llm.py` | The only place a language-model call happens. | — |
| Web search | `search.py` | The only place Tavily is called. | — |
| Page fetch | `fetch.py` | Downloads pages box 7 already chose. No stance and no model. | — |

## Verdict rules (the part the model does not own)

Box 9 labels each atomic claim: `not_checkable`, `conflicting`, `refuted`, `supported`, or `insufficient`. The overall gate follows the **core** claim.

- Enough evidence: at least two independent groups on the same side, and at least one of them is tier 1 or 2. One credible voice on the other side does not block that, unless it is a large minority or a tier-1 source. A group that both supports and refutes is kept and counted on neither side.
- A picture claim can be refuted by text. Text cannot confirm what a picture shows.
- Box 8 then maps the core status. A refuted core is **False** when the draft says the event did not happen (`entirely_false`) and **Misleading** when a real event happened but a detail is wrong (`wrong_detail`). A supported core with a refuted side claim is **Misleading**. A supported core with an unconfirmed side claim stays **True** and the card lists caveats. Insufficient evidence is **Unverifiable** even if the draft says False. A conflicting gate does not pick False or True. Review is non-blocking, so the card is Unverifiable and the row waits on the queue. A stance failure is a system error, not "not enough sources". A greeting, joke, opinion, or prediction is **Other**.
- Gate flags that are on: G1 (`GATE_PRIMARY_SOURCE`), G2R (`GATE_WEAK_REFUTE`), G3 (`GATE_SHARED_SOURCE`), the stale-official guard (`STALE_DAYS` 365), and core match (`TRUE_NEEDS_CORE_MATCH`, relaxed). G4, the evidence score, is off. A Latin sentence with ordinary English words is not dropped because `langdetect` guessed another language. A sentence with a number or a date and a factual verb is not dropped because extract called it an opinion.
- `search_status` on the card is `ran`, `cache_only_miss`, `not_needed`, `monthly_cap`, or an error, with the live and cached search counts.

Confidence is a logistic over evidence features (group count, tier, agreement, dates, unverified quotes, top-up, extra rounds). On a reused prior it comes from the match cosine. On Unverifiable it is confidence that the claim is unverifiable, not that it is false.

Priority is `0.40·harm + 0.15·novelty + 0.25·reach + 0.20·urgency`. High is at least 0.60, Medium at least 0.35, otherwise Low. Harm is the domain score in `config.DOMAIN_HARM`, plus 0.10 for each of four harmful red flags, capped at 1. Novelty is `1 − top memory cosine`. Reach is a log of repeat submissions and outlets, capped so a 500th repost cannot drown harm. Urgency is deadline words, a share request, and a date inside three weeks.

Decide uses `p* = 1 / (1 + 3H)`. False and Misleading above the bar at high priority become an alert. An alert still pauses for a person while `ALERT_NEEDS_APPROVAL` is on. High harm just under the bar searches once more. Far under the bar, or already at the search limit, goes to a person. Unverifiable plus high harm goes to a person; low harm is stored. True and Other are stored. A False verdict below high priority is stored even when the confidence is high.

## Data the system remembers

Not in git. See `DATA.md`. Shared from the Google Drive folder "Misinformation project". Copied 2026-11-04.

| File | Rows | SHA-256 |
|---|---|---|
| `data/raw/classified_v2.parquet` | 52,678 articles; 52,099 with `is_fact_check == true` | `04b934ec67059bceb2e22812341aaa9edf0fb94e636e9eaf2bb444ee8a038949` |
| `data/raw/bge_m3_primary_claim.parquet` | 52,099 embeddings, 1024-d BAAI/bge-m3 of `primary_claim`, L2-normalised, max length 256 | `0f9e0e9423607dd5d8f7a5c82e2cb34b8bf81c0c6317a70e0b6de28aaf9c8266` |

The articles are from 15 IFCN-verified Indian fact-checkers, 2016 through 11 Oct 2026. Labels (claim, true version, verdict, domain, manipulation type, red flags, form) were made by Gemini with Prompt A (`misinfo_agent/prompts/prompt_v4a.txt`) and are not yet human-validated. Weak columns, not to build on: `claim_source`, `purpose`, `confidence`.

Read parquet with an explicit `columns=[...]`. Do not load the article `content` column unless the task needs it. Do not build an N×N similarity matrix over the memory (52,099² float32 is about 11 GB). The only pairwise cosines are inside the fused top 30, where two outlets covering one story are collapsed.

`python -c "from misinfo_agent.memory import build_memory; build_memory()"` writes `data/memory/` (`emb.npy`, `meta.parquet`, `bm25.pkl`, `manifest.json`, and a FAISS index used only by the parity script). It skips the work when the manifest already matches the input checksums.

Eleven domains, matched exactly: Politics & elections, Religion & communal, Defence & international, Crime & law, Society & other, Health, Disasters & accidents, Finance & scams, Entertainment & celebrities, Sports, Science & technology.

Eleven red-flag labels: Urgency, Fear or outrage, Fake or vague authority, Miracle or absolute claim, Free money or guaranteed returns, Request for personal data or payment, Suspicious link or contact, Call to forward, Sensational style, Emotional appeal, Unverifiable source.

## Folder map

```
misinformation_agenticai/
├── README.md                 setup
├── DATA.md                   parquet manifest and checksums
├── requirements.txt          pinned Python dependencies
├── pytest.ini                pythonpath and "not live" by default
├── .env.example              GEMINI_API_KEY, GEMINI_MODEL, TAVILY_API_KEY
├── .gitignore                .venv, data/, .env, caches, logs
├── .cursorignore             .env
├── .cursor/rules/project.mdc always-on rules for this repo
├── app/streamlit_app.py      the checker UI
├── docs/                     architecture, eval write-ups, this file
├── misinfo_agent/            the package (no __init__.py; imported as a namespace)
│   ├── *.py                  one module per box, plus llm, search, fetch, config, state
│   └── prompts/              prompt text the boxes fill in
├── scripts/                  one-shot runners, eval, and scoring
├── tests/                    pytest; live tests are marked and skipped by default
│   └── fixtures/             sample ClaimState and saved HTML
└── data/                     gitignored. parquet, memory, caches, eval outputs, sqlite
```

## `misinfo_agent/` — the package

Shared state is `ClaimState`. Shared numbers and paths are `config.py`. Do not change `state.py`, `config.py`, or `requirements.txt` unless the task says so. Every model call goes through `llm.py` and comes back as a Pydantic object. Keys live in `.env` and are never printed, logged, or committed.

### `state.py`

TypedDict, `total=False`, because early boxes have not filled later keys. Comments on each key name the box that writes it. The complete example is `tests/fixtures/sample_state.json` (an RBI ₹500-note forward).

Important keys, in pipeline order:

- Ingest: `raw_input`, `raw_text`, `input_type` (`text` / `url` / `image`), `text`, `match_text`, `headline`, `body`, `article_date`, `source`, `image_path`, `image_sha256`, `image_phash`, `ocr_text`, `ocr_mean_conf`, `caption`, `language`, `hinglish`, `flags`, `style_signals`, `quality`, `checkable_hint`, `dedupe`, `submission_hash`, `submission_count`, `claim_date`.
- Extract: `claim`, `atomic_claims`, `entities`, `domain`, `red_flags`, `style_flags`, `checkable`, `not_checkable_reason`, `extract_notes`, `headline_body_mismatch`, `visual_description`, `location_clues`, `time_clues`.
- Memory and prior: `matches`, `best_match`, `match_type`, `route`, `leads`, `needs_recheck`, `match_reasons`, `match_summary`.
- Investigate and gate: `evidence` (append-only), `passages`, `investigate_stats`, `search_capped`, `rounds`, `gate`, `gate_reasons`, `atomic_status`.
- Assess: `verdict`, `confidence`, `explanation`, `true_version`, `correction_message`, `what_we_found`, `caveats`, `citations`, `atomic_results`, `assess_notes`, `needs_human`.
- Priority and decide: `priority`, `priority_score`, `priority_reasons`, `decision`, `decision_reasons`, `p_star`.
- Review, output, audit: `human`, `human_action`, `human_edited`, `note`, `errors` (append-only), `result`, `run_marks`, `audit`.

`image` is a legacy key. Ingest no longer puts pixels on the state, because the checkpointer would store them on every run. Extract reads the JPEG from `image_path`.

### `config.py`

Paths are anchored to this file, not the process working directory. Thresholds live here so two boxes cannot drift.

Selected values the rest of the pipeline depends on:

- Embeddings: `BAAI/bge-m3`, max sequence 256. A different model or window would make the stored matrix stop being a cosine.
- Prior match: `TAU_HIGH` 0.85 (candidate for exact), `TAU_LOW` 0.70 (similar), `KEYTERM_SIM` 0.45, `DUP_COS` 0.95, `STALE_DAYS` 365, `RRF_K` 60. `LLM_CONTEXT_CHECK` is on: the veto runs only after the rules have left a pair exact.
- Models: provider `gemini`, `gemini-3.5-flash-lite`, fallback `gemini-3.1-flash-lite`, timeout 60s. Cache on under `data/cache/llm`. Daily cap 400 real calls. 8 calls per minute.
- Search: cache on under `data/cache/search`. Monthly Tavily cap 800. 8 searches per claim. 2 investigate rounds. `BLOCK_FACTCHECK_SITES` is off for a demo and turned on by the eval. Passage cosine floor 0.45. `EVAL_DROP_AFTER_CLAIM` is off in the demo. Relative-time window 14 days. Entity boost 0.05. Claim budget 60s. At most 12 non-circulation page fetches, 6s each.
- Priority weights and the decide constants are the numbers in the verdict section above.
- `RECORD_SUBMISSIONS` is off. A check is not another person forwarding the message. `scripts/try_claim.py --record` and the Streamlit sidebar are the switches that turn it on.
- `OFFICIAL_DOMAINS` maps each of the 11 domains to government and official hosts used by the "official" search query (RBI, PIB, health ministries, and so on).

### `ingest.py` — boxes 1–2

No model. Turns a submission into clean text and a duplicate id.

- Text: Unicode cleanup, strip a "forward to all groups" trailer, lowercase `match_text` so two copies of one forward share a hash. Rupee amounts and numbers are kept.
- URL: fetch with the same private-address refusal as box 7, strip tracking parameters, read the article with trafilatura, or keep only the Open Graph preview of a social post. Headline and body are stored; `text` is the headline plus about the first 1200 characters of the body.
- Image: downscale, strip EXIF (the saved JPEG is pixels only, so GPS does not travel with the file), write a JPEG named by its sha256 under `data/uploads`, perceptual hash (Hamming distance 6 is the same picture), Tesseract OCR on words scored 60 or above. Video and audio are rejected. Images over the side limit or over 10 MB are rejected.
- Language: `langdetect` with seed 0. English proceeds. Hinglish (Latin letters plus a common Hindi word) proceeds with a flag; that word list wins when the detector disagrees. Any other language stops before a model call.
- Quality: too short (a greeting) or out of scope stops before a model call.
- Style signals (urgency words, share request, forwarded label, caps ratio, exclamation marks, emoji, authority mentions, conspiracy phrases, certainty words) are warning signs. They feed priority and the card. They are not a verdict.
- `data/submissions.db` stores a sha256 and a MinHash of the cleaned text, plus an image hash. It does not store the message, a name, or a phone number. The raw string stays in memory for later boxes.

### Photographs

`vision.py` is the only Google Cloud Vision caller. The feature is `WEB_DETECTION`. Face, celebrity, and landmark detection are not requested, and `webEntities` are dropped so a guessed name is not evidence. The cache key is the sha256 of the EXIF-free JPEG.

`provenance.py` fetches the pages Vision returned, dates them (htmldate, the URL, page metadata, or a LinkedIn activity id), and asks one stance call. The event day comes from the caption: "August 2026" is that month, "yesterday" is the day before `claim_date`, and a full date is that day. `PHOTO_RULE` is `v2`. A full match can date the file. A partial counts only when its date is trusted and at least 365 days before the event starts. Out of context is Misleading. A photograph with no caption is Other, and the provenance table is still filled.

`image_memory.py` stores the BGE-M3 embedding of the extracted claim with the perceptual hash. The same photograph and the same caption reuse the stored verdict and skip search. The pixels are not stored. A reused card still attaches provenance from the Vision cache.

### `extract.py` — boxes 3–4

One `call_structured` with `prompts/extract_v2.txt`, `prompt_version="extract_v2"`. The claim is one sentence, at most 30 words, written the way people sharing it would say it — the same rule as `primary_claim` in Prompt A, so box 5 compares like with like. The model must not add "falsely", "claims", or a fact-checker's conclusion.

After the model answers, code drops an entity that is not in the text, retries a bad domain, retries two core claims down toward one, and refuses a motive ("to protect pharma") as a checkable atomic claim. Style signals merge into `red_flags` when a label fits and stay in `style_flags` when none does. A URL whose headline says something the body does not gets `headline_body_mismatch`. If ingest already said the text is not checkable, this box writes verdict Other and does not call the model. A quota error propagates; it is the end of the run.

### `memory.py` — box 5

Hybrid search. Dense path is one matrix-vector product `E @ q` (rows of `E` are unit length, so the product is a cosine). BM25 ranks the same rows by words. Reciprocal rank fusion (`1/(60+rank)`) adds the two lists without pretending cosine and BM25 share a scale. Filters: `date_before`, domain. Unknown domain raises. Near-duplicates at cosine ≥ 0.95 inside the fused top 30 collapse into `also_debunked_by`.

NumPy is the search path. FAISS is not imported here: loading it in the same process as the MPS encoder segfaults. `scripts/faiss_parity.py` is the cross-check. The article body is never read; search is over `primary_claim`.

### `prior_match.py` — box 6

Cosine proposes exact (≥ 0.85), similar (≥ 0.70), or none. Rules can still downgrade exact to similar:

- a disagreeing date, amount, percent, year, number, name, or subject
- a community word present on one side only ("a Hindu manager" is not "a manager")
- key-term similarity under `KEYTERM_SIM` (this is what stops "hot lemon water kills coronavirus" from reusing "kills cancer cells", and "goat urine" from "cow urine" once the veto runs)
- a picture claim, capped at similar because there is no image memory yet
- a status claim ("banned", "invalid", "approved") whose best match is older than 365 days, except an out-of-context or edited image/video whose origin cannot expire

`route` is `reuse_prior` only when the match is exact and fresh. Otherwise `investigate`, and the hits are leads. The model is asked only when the rules have left the pair exact. It can veto exact into similar. It cannot create an exact. If the call fails, the pair is similar and is checked fresh: a missing key is not a yes.

### `search.py`

The only module that calls Tavily. Cached and fake answers do not count against the monthly cap (800) or the per-claim cap (8). A transient error is retried once. Auth failures and provider quota are not retried. `SearchQuotaError` stops a later query; pages already fetched stay.

Fact-check blocking, when the eval turns it on, drops the 15 archive hosts (`boomlive.in`, `dfrac.org`, `digiteye.in`, `en.youturn.in`, `english.factcrescendo.com`, `factly.in`, `firstcheck.in`, `indiatoday.in`, `lighthousejournalism.com`, `newschecker.in`, `newsmeter.in`, `ptinews.com`, `telugupost.com`, `thequint.com`, `vishvasnews.com`) plus extra desks (`altnews.in`, `snopes.com`, `factcheck.org`, and the rest of `EXTRA_FACTCHECK_DOMAINS`), a host whose name contains "factcheck", the path token `webqoof`, and PIB Fact Check posts. A normal `pib.gov.in` press release stays. The block flag is part of the cache key, so a blocked query and an allowed query are different cache entries.

### `fetch.py`

Downloads the hits box 7 already chose. A host is opened only after every address is a public IP, and each redirect is checked again (SSRF). Circulation hosts (social posts, stock libraries) are counted as sightings and never become passages. Non-English pages are dropped. Trafilatura reads HTML; pypdf reads PDFs. Dates, first trusted source wins: htmldate, a date in the URL path, trafilatura metadata, then Tavily. Today and the future are not publication dates. A failed download falls back to the Tavily snippet. At most 12 non-circulation pages are fetched. Cache under `data/cache/pages`.

### `investigate.py` — box 7

Finds live pages when box 6 did not reuse a prior.

Query plan, chosen without a model, normally at most five lines: neutral (the claim, trimmed to 20 words), verify (whether the named entities are true), refute (`fake OR false OR hoax OR misleading`), official (keywords restricted to `OFFICIAL_DOMAINS` for that domain), and a lead only when the prior is similar. A second checkable atomic claim adds one neutral query. A motive adds none. Each kind keeps its best three hits before a cap of 25 unique URLs. The submitted page itself is dropped. A second call of `investigate()` skips queries already run and continues evidence ids (`E5` after `E4`).

Passages: three-sentence windows, drop under 20 words, drop cosine under 0.45, at most two windows per page, two per registrable domain, forty per claim. The stance call sees the best twenty. BGE-M3, the same encoder as the memory, loads once.

Stance is `prompts/stance_v9.txt`. Supports only if every key detail agrees (who, what, amount, place, and timing). A partial confirmation is neutral. A contradicted detail is a refute, including "retail only" against "completely" and "temporary" against "permanent". Old evidence cannot support a "now" / "from \<date\>" / "permanently" claim, and it can still refute that claim. Tense and "named after" rules can only stop a support; they never turn a refute into neutral. A different event (another metro line, another scheme) is neutral. A quote that is not in the passage becomes neutral. A passage that only repeats the claim is neutral. Older prompt files `stance_v1.txt` through `stance_v8.txt` are history; the live call is v9.

Repair (`repair_v1.txt`) runs once, at most twenty rows, when a reason disagrees with its label. A repaired refute whose quote is not in the passage stays neutral.

Credibility is a property of the source, not of whether the page was fetched. Curated lists win: tier 1 is a government suffix (`.gov`, `.nic.in`, …) or an official body; tier 2 is the curated newsroom and journal list or `.edu` / `.ac.`; tier 3 is other news, Wikipedia, hospitals, and NGOs already classed; named blogs are tier 4. A host none of those lists name is looked up in `data/source_tiers_rsp.csv` (Wikipedia perennial sources, built by `scripts/build_source_tiers.py`). Generally reliable → tier 2, no consensus → tier 3, generally unreliable → tier 4. Anything still unnamed is tier 4. The checker runs without that CSV and logs a warning once.

Independence: the same registrable domain is one group. MinHash (5-word shingles, 128 permutations, LSH threshold 0.5) joins a near-copy on another domain. Duplicate quotes inside a group collapse. Evidence rows are supports and refutes only.

Top-up (`topup_v1.txt`): if fewer than two independent tier 1–2 groups support or refute, one call proposes at most two new queries, inside the searches still left on the claim. At most one top-up per call of `investigate()`. A search cap or a model error returns what is already in hand. It does not raise.

When the eval allows fact-check sites, the blocked search runs first (a cache hit) and fact-check pages are added in their own budget of at most 8 fetches. The phrase rule stays off for that budget. The claim's own article is still excluded by canonical URL.

### `gate.py` — box 9

Rules only. Counts groups box 7 already labelled. Does not re-read the pages. Outcomes: `sufficient`, `insufficient`, `conflicting`, `search_more`, `error` (the stance call failed; this is not "not enough sources"), or `skipped` (exact fresh prior, or nothing checkable). `rounds` advances only on `search_more`, and it stops at `MAX_SEARCH_ROUNDS`.

### `assess.py` — boxes 8 and 10

One call, `prompts/assess_v2.txt`. The model writes, per atomic claim, an explanation (citations like `[E2]`), a `refute_kind`, a `true_version`, and a `correction_message`. It is told not to pick a verdict and not to add facts that are not in the rows. Any label it would have picked is stored as `model_lean` and is not copied into the verdict. Code then:

- picks the verdict from the gate status and, for a refuted core, from `refute_kind`
- drops any sentence that cites an id the table does not have, or another atomic claim's row
- rebuilds an empty explanation from the rows, with no second model call
- scores confidence from the table

`reuse_prior` copies the archive verdict and `true_version` and does not call the model. The fact-check is cited as `[C1]`. A quota error is not turned into a template.

### `priority.py` — box 11

Score in `[0, 1]` from the four components, then High / Medium / Low. `scripts/priority_sensitivity.py` moves each weight by 20% and counts how many fixture levels change. The weights are hand-set; the script does not write a new set back.

### `decide.py` — box 12

Reads the verdict and the confidence. Does not reopen the evidence and does not read style signals again. Returns `alert`, `human`, `search_more`, or `store`, plus `p_star` and a reason. `rounds` is returned only for `search_more`.

### `review.py` — box 13

LangGraph `interrupt()`. Writes a pending row into the audit database, ordered high priority first. Resume values are approve, edit, reject, or more_evidence. Reject clears the verdict. More evidence runs one more investigate pass. Edit replaces the verdict or the correction and sets `human_edited`.

### `output.py` — boxes 14–15

Builds the result card: verdict, confidence, evidence with tiers, seen-before heading, red flags, true version, correction, what was found, caveats, queries, and cost. Does not rewrite the explanation. Email and webhook sending are not implemented; the card is what the UI and `scripts/run.py` print.

### `audit.py` — box 16

One SQLite row per finished run: input hash (not the raw message), evidence ids, verdict, human action, model and prompt versions, call counts, latency, errors. `list_runs` is the history list, newest first. An accepted verdict can be appended to `data/memory_additions.parquet` for a later memory rebuild. The audit row does not rebuild the card.

### `graph.py`

Compiles the graph described at the top. Recursion limit 40. `run_pipeline` starts a claim or resumes a thread. `paused_review` reads a checkpoint that is waiting on a person. Public entry points for the app and for `scripts/run.py`.

### `llm.py`

The only Gemini client. Importing it does not read the key and does not open a socket. `call_structured(prompt, schema, purpose, prompt_version)` validates the response with Pydantic, retries a schema miss twice, backs off on HTTP 503 (3s, 10s, 30s) and then tries the fallback model once, caches by model + prompt version + prompt + schema + images, and counts only real calls toward the daily cap. `set_fake` / `clear_fake` are how tests stand in for Gemini. `LLMQuotaError` stops the run. `get_usage()` is what the audit row copies.

## `misinfo_agent/prompts/`

Live prompts (the version string stored on each call):

| File | Used by | Job |
|---|---|---|
| `extract_v2.txt` | `extract.py` | Write the circulated claim, atomic claims, entities, domain, red flags. No verdict. |
| `stance_v9.txt` | `investigate.py` | Label each passage supports / refutes / neutral, with a quote copied from the passage. |
| `repair_v1.txt` | `investigate.py` | One repair pass when a reason disagrees with its label. |
| `topup_v1.txt` | `investigate.py` | At most two extra queries when tier 1–2 groups are thin. |
| `assess_v2.txt` | `assess.py` | Explanations, true version, correction. No verdict and no self-reported confidence. The live assess prompt. |
| `prompt_v4a.txt` | the archive, not a runtime box | Prompt A that labelled `classified_v2.parquet`: `is_fact_check`, `primary_claim`, `true_version`, verdict, domain, manipulation type, red flags, form. |

Kept for history and for tests that lock a rule to the prompt that introduced it: `extract_v1.txt`, `stance_v1.txt` through `stance_v8.txt`. The live stance file is v9. v4 added tense and "named after". v5 says those rules never turn a refute into neutral. v6 makes a different event neutral. v7 states the share date. v8 and v9 keep those rules and tighten old-evidence and different-event cases.

## `app/streamlit_app.py`

`streamlit run app/streamlit_app.py`. The title is two lines, "AGENTIC AI SYSTEM" and "FOR MISINFORMATION DETECTION", and the ✦ stays coral. Three pages in one process: check a claim, the review queue, and run history. The page calls `run_pipeline` and draws the card. It does not score a verdict and it does not search on its own. A busy flag stops a second claim from starting beside a run. BGE-M3 and the memory index load once per process. `RECORD_SUBMISSIONS` stays off unless the sidebar switch is on, because box 11 reads that count as reach. Review is non-blocking: the card is shown, and approve / edit / reject updates the finished thread through `apply_review`. `SEARCH_CACHE_ONLY` defaults on in the app so a demo does not spend Tavily.

## `docs/`

| File | What it is |
|---|---|
| `ARCHITECTURE.md` | The design contract. Sections: what the system does, data assets, the 16 boxes, the state contract, the evaluation plan, engineering rules, future work. |
| `PROJECT_CONTEXT.md` | This file. |
| `eval_diagnosis.md` | Gate study, the 16 established facts, and the image-rule comparison. `PHOTO_RULE` is `v2` because that comparison said so. |
| `system_check.md` | One live `run_pipeline` of each input type. |
| `runbook.md` | How to run a check, and the card each input type should show. |
| `eval_prior_match.md` | Box 6 numbers. 60-claim test set, 45 text pairs in the 0.80–0.95 band, 25 one-edit negatives. Recall@1 0.950 and Recall@5 1.000 on the 20 paraphrases. Macro-F1 0.903 at 0.85 / 0.70 before the veto. With `KEYTERM_SIM` 0.45 and the LLM veto on, precision of "exact" is 1.0 and false exacts are 0 (22 model calls on that scoring pass). |
| `eval_investigate.md` | Box 7 on eight claims, 14 Nov 2026, fact-check sites blocked, stance_v3 era. Passage precision@5 82.5% (33/40). Stance accuracy 84% (21/25). Median about 5.5 searches, 2 model calls, 31 seconds. Known limits: absence of evidence is not a refutation; text cannot confirm a picture; tiers are India-centric; the top-up rarely adds a second credible group. |
| `eval_end_to_end.md` | Written by `scripts/score_eval.py` from the 50-row test set. The allowed-condition numbers in that file are marked superseded (an evaluation-setup bug: the allowed run did not reuse the blocked search cache and then hit the monthly cap). Treat the blocked column as the honest number. On 50 blocked claims: 17 decided, 8 correct, 3-class accuracy 0.471 on decided claims, 3-class macro-F1 0.446, abstention 0.660 (33/50), coverage-adjusted accuracy 0.160. Leak count 0 (the claim's own article, or a near-duplicate, must not appear in memory, leads, or evidence). The baseline is one Gemini call on the claim text alone: 21 decided, 13 correct, abstention 0.580. Definitions, per-class tables, calibration, citation checks, and cost are in that file. |

Evaluation design (from the architecture, and what the scripts implement):

- Time split: memory and train through 2024, test on 2025–26. `date_before` so the system only "remembers" fact-checks older than the claim. Drop a test claim whose nearest memory neighbour has cosine ≥ 0.90.
- Two search conditions: fact-check sites blocked (the honest number) and allowed (other fact-checkers may appear; the source article itself is still hidden).
- True cases are built from `true_version` of other articles, so a claim is not graded against its own write-up.
- Metrics: accuracy and macro-F1 on decided claims (3-class, and 2-class with False and Misleading collapsed to not-true), abstention, coverage-adjusted accuracy, citation validity, ECE on 10 confidence bins, Wilson intervals, cost and latency. A review pause is not answered; `needs_human` counts as an abstention.

## `scripts/`

None of these edit `config.py`. Tuning runs print a winner and leave it out of the config, so a grid search cannot silently change production.

| Script | What it does |
|---|---|
| `run.py` | One claim through the graph. Prints the card. On a review pause, reads approve / edit / reject / more_evidence from the terminal and resumes the same thread. Submissions are not recorded. A quota error exits 2. |
| `try_claim.py` | Prints what each box returned. A bare claim starts at box 5. `--text`, `--url`, and `--image` run ingest and extract first. `--record` is the only way this script writes a submission. |
| `reset_submissions.py` | Asks, then deletes `data/submissions.db`, so a demo forward is not counted as reach during an eval. A bare run, or end-of-file, leaves the file alone. |
| `build_source_tiers.py` | Builds `data/source_tiers_rsp.csv` from the English Wikipedia perennial sources list (CC BY-SA 4.0). Prints the revision id to attribute. |
| `build_testset.py` | Writes `data/eval/testset.csv`, seed 42, no model. 40 archive claims from 2025–26 (False, Misleading, or True) plus 10 `true_version` rows labelled True, drawn from other articles. |
| `run_eval.py` | Runs the test set one claim at a time. Stops on `LLMQuotaError`. Writes `results_system_blocked.jsonl` or `results_system_allowed.jsonl`, and `results_baseline.jsonl` once. Resumes from lines already written. Counts leaks. |
| `score_eval.py` | Joins results to the test set and rewrites `docs/eval_end_to_end.md`. |
| `eval_prior_match.py` | Scores the 60-row box-6 set: Recall@k and the threshold grid. Does not write the winner back. |
| `build_tau_high_pairs.py` | Builds text pairs whose nearest other article sits in [0.80, 0.95), for a human same-claim label. One row-vector dot product at a time, never the full matrix. |
| `score_tau_high.py` | Precision and recall of "exact" on that labelled sheet. Refuses to run until `same_claim` is y or n. |
| `score_d3.py` | Old detail rules against the newer key-term rules, including the one-edit negatives. |
| `score_llm_veto.py` | Scores the box-6 veto on the 70 labelled pairs. Two passes; the cache makes the second pass free. Stops on the daily cap and still prints what finished. |
| `priority_sensitivity.py` | Moves each priority weight by 20% and counts level changes on the garlic forward and the calm sports rumour. |
| `faiss_parity.py` | Compares a FAISS flat index with the NumPy search on a stored vector. Not imported by the live search path. |

## `tests/`

`pytest` runs everything except tests marked `live` (`pytest.ini`: `addopts = -m "not live"`). `pytest -m live` is the run that can spend a Gemini or Tavily credit. `pythonpath = .` because the package is not installed into the venv.

`tests/conftest.py` makes the default run offline. Unless a test is marked live, both API keys are removed, `load_dotenv` is a no-op, and building a real Gemini or Tavily client raises. Every test, live or not, gets a temporary submissions database and a temporary uploads directory, so pytest cannot write `data/submissions.db` or leave screenshots under `data/uploads`. Search, page, and model caches point at temp files. The daily and monthly counters point at temp files too.

| Test file | Covers |
|---|---|
| `test_contract.py` | Every `ClaimState` key is in `fixtures/sample_state.json`. Every box module imports. |
| `test_ingest.py` | Garlic-forward style signals and cleaning, greetings, Hindi vs Hinglish, near-copy dedupe, rupee amounts kept, no raw text in SQLite, URL fixtures (article and social preview), private URLs refused, OCR, EXIF strip, image size limits, `RECORD_SUBMISSIONS` off writes nothing. |
| `test_extract.py` | Fake model through Pydantic. Bad domain retried. Two cores retried. A motive is not checkable. Ungrounded entities dropped. Style signals merge into red flags. Non-checkable skips the model. Image bytes load from the path. `extract_v2` splits attribution from the core. Quota errors propagate. |
| `test_memory.py` | Needs the real matrix. A stored claim re-encoded at length 256 is rank 1. `date_before` and domain filters. Unknown domain raises. Importing memory does not import FAISS. FAISS and NumPy return the same ids (parity script). Rupee spellings share BM25 tokens. |
| `test_prior_match.py` | Amount, percent, date, community, and subject clashes are never exact. Stale status needs a recheck; an old out-of-context image does not. Picture claims are never exact from text. Lemon/cancer swap is never exact. LLM veto turns exact into similar, does not run unless the pair is exact, and a failed veto downgrades to similar. Veto can be switched off. |
| `test_investigate.py` | The largest file. Query plan for every domain, atomic-claim extras, lead query only when similar, URL canonicalisation, dedup, the 25-hit cap, fact-check blocking and the separate fact-check budget, cache keys, monthly and per-claim caps, retries, passage windows, cosine floor, the forty-passage cap, dates and the leakage guard, stance quote checks, stance v4–v9 prompt rules, credibility tiers, MinHash groups, conflicting groups kept, repair, top-up, a second round skipping known queries. Two live tests at the bottom (`test_live_robustness_eight_claims`, `test_live_day4_eight_claims`) are the ones that spend credits. |
| `test_fetch.py` | SSRF (private addresses, redirects, other schemes), circulation hosts not fetched, snippet fallback, date precedence, future dates untrusted, PDF text, non-English skipped, body deadline, user agent and timeout. |
| `test_rsp_tiers.py` | Curated tier lists beat the Wikipedia perennial-sources file. A missing file warns once and the curated tiers still apply. |
| `test_gate.py` | The eight-claim investigate examples reduced to group, tier, and stance. Picture claims, tier-1 disagreement, a group that conflicts with itself, stance failure vs insufficient evidence, skipped gates, style not changing the gate. The gate must not open a client. |
| `test_assess.py` | Verdict from gate status plus `refute_kind`. A draft cannot outvote the table. Citation guard. Template when the draft is empty or the call fails. Reused prior does not call the model. Confidence: four groups outrank two; a named dissenter lowers it. Quota is not swallowed. |
| `test_priority.py` | Garlic forward is High. A calm sports rumour is Low. Novelty is one minus the top cosine. Repeated forwards count more than "as received". Scores stay in [0, 1]. Urgency and sensational style do not raise harm. The sensitivity script's level-change count. |
| `test_decide.py` | `p*` falls as harm rises. Conflict and gate error go to a person. False and Misleading, high priority, above the bar: alert. Just under the bar: search again. Far under, or at the search limit: human. Unverifiable is human only when harm is high. True and Other are stored. Style does not change the decision once priority is set. |
| `test_graph.py` | Fake model, fake search, stub memory. Routes: reuse prior skips search; a refuted claim is stored; search_more stops at the round cap; conflicting review can be approved; an alert can be edited; reject clears the verdict; more evidence searches once more; not checkable is Other; a node exception goes to review; a checkpoint resumes on a new graph; the pending row is written on interrupt and removed on resume. |
| `test_pending_reviews.py` | Empty audit file lists nothing. A blank thread is not queued. High priority is listed first. History is newest first. |
| `test_llm.py` | Fake output still passes Pydantic. Schema misses retry then raise `LLMError`. Cache hits do not call again and do not count toward the daily cap. Missing key, daily cap, rate limiter, 503 backoff and the fallback model, per-minute 429. Importing `llm` does not need a key. One live smoke call. |
| `test_eval.py` | Offline harness: label mapping, domain quotas, near-copy exclusion, honesty flags, `claim_date` overriding today, resume and quota stop, an interrupt recorded as `needs_human`, baseline prompt is the claim only, abstention and citation scoring, leak counts, blocked vs allowed result files. |

### `tests/fixtures/`

| File | What it is |
|---|---|
| `sample_state.json` | A complete `ClaimState` for the RBI ₹500-notes forward. The contract test requires every state key to be present. The UI and several box tests build from it. |
| `article.html` | A saved article page. Ingest URL tests serve it through the fetch seam so they do not open the network. |
| `social_post.html` | A saved social post. Ingest keeps the Open Graph title and description and does not treat the page as an article. |

## Root files

| File | What it is |
|---|---|
| `README.md` | Clone, Python 3.12 venv, `pip install -r requirements.txt`, checksum the two parquets, build memory, optionally build source tiers, copy `.env.example` to `.env`, run `pytest`. |
| `DATA.md` | Source path, copy date, row counts, byte sizes, and SHA-256 of the two parquets. Notes that every embedding `article_id` is in the classified file, and that `prompt_v4a.txt` is in git. |
| `requirements.txt` | Pinned stack. The pieces the pipeline actually depends on: `langgraph` and `langgraph-checkpoint-sqlite` (the graph and the pause), `langchain-google-genai` (Gemini), `pydantic` (every model output), `sentence-transformers` and `torch` (BGE-M3), `faiss-cpu` (parity only), `rank-bm25`, `numpy`, `pandas`, `pyarrow`, `tavily-python`, `trafilatura`, `htmldate`, `beautifulsoup` via trafilatura's stack, `datasketch` (MinHash), `langdetect`, `Pillow`, `imagehash`, `pytesseract`, `pypdf`, `streamlit`, `pytest`, `python-dotenv`. |
| `pytest.ini` | `pythonpath = .` and skip `live` tests unless asked. |
| `.env.example` | Three empty keys: `GEMINI_API_KEY`, `GEMINI_MODEL`, `TAVILY_API_KEY`. Copy to `.env`. Never commit `.env`. |
| `.gitignore` | `.venv/`, `data/`, `.env`, `__pycache__/`, `*.pyc`, `.DS_Store`, `*.log`. |
| `.cursorignore` | `.env`, so the editor index does not pick up the key. |
| `.cursor/rules/project.mdc` | Always applied. Points at `ARCHITECTURE.md`, states the Unverifiable rule, and locks the engineering rules below. |

## `data/` (gitignored, created locally)

| Path | Written by | Holds |
|---|---|---|
| `raw/classified_v2.parquet` | copied in, see DATA.md | fact-check articles |
| `raw/bge_m3_primary_claim.parquet` | copied in | claim embeddings |
| `memory/` | `build_memory()` | `emb.npy`, `meta.parquet`, `bm25.pkl`, `manifest.json`, `faiss.index` |
| `source_tiers_rsp.csv` | `build_source_tiers.py` | Wikipedia perennial-source tiers |
| `submissions.db` | ingest, only when recording is on | hashes, never the message |
| `uploads/` | ingest, for images | downscaled EXIF-free JPEGs named by sha256 |
| `cache/llm/` | `llm.py` | structured-call cache |
| `cache/search/` | `search.py` | Tavily cache; the block flag is part of the key |
| `cache/pages/` | `fetch.py` | fetched HTML and extracted text |
| `checkpoints.db` | `graph.py` | LangGraph threads, including paused reviews |
| `audit.db` | `audit.py`, `review.py` | one row per run, plus the pending-review queue |
| `memory_additions.parquet` | audit, when a verdict is accepted | claims to fold into a later memory rebuild |
| `eval/` | the eval scripts | `testset.csv`, prior-match sheets, result jsonl, figures. Not in git. |

Daily and monthly counters (`llm_daily_count.json`, the search monthly count) live beside these caches. Offline tests never touch the real files.

## How to run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
shasum -a 256 data/raw/classified_v2.parquet data/raw/bge_m3_primary_claim.parquet
python -c "from misinfo_agent.memory import build_memory; build_memory()"
python scripts/build_source_tiers.py   # optional; the checker warns once if the file is missing
cp .env.example .env                   # then fill the three keys
pytest                                 # offline
pytest -m live                         # spends API credit
python scripts/run.py --text "..."     # one claim, prints the card
streamlit run app/streamlit_app.py     # the UI
```

What each input should show is in `docs/runbook.md`.

Tesseract must be installed on the machine for image OCR. Without it, image ingest cannot read words off a screenshot.

## Engineering rules

- Comments explain why, not only what.
- Never print, log, or commit API keys.
- Never commit `data/`.
- All model calls go through `llm.py`. All model outputs are validated with Pydantic.
- Every module has tests. `pytest` must pass before a pull request.
- Branches are `feat/<area>`. Never push to `main`.

## Not built

Listed in `ARCHITECTURE.md` §7 and still out of scope: a fusion transformer for multimodal inputs, CLIP embeddings, an image memory from the fact-check archive, paid reverse image search, video and audio (Whisper), a fine-tuned MuRIL or mDeBERTa tag model, an NLI stance model, a full expected-value-of-information decision rule, and feedback learning beyond appending approved verdicts. Logistic-regression calibration of the confidence weights is noted as later work; the weights in `config.py` are provisional and the eval reports ECE so they can be moved on purpose.
