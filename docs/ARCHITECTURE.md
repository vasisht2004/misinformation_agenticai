# misinformation_agenticai — Architecture

The single source of truth for what this system is and how it is built. Every box number used in
code comments, prompts, issues and PRs refers to this document. If code and this document disagree,
fix one of them in the same PR.

## 1. What the system does

An agentic AI that takes something people share in India — a **WhatsApp-style text forward, a link
to an article or post, or an image (screenshot or photo, with an optional caption)** — and returns:

- a verdict: **False / Misleading / True / Unverifiable / Other** (Other = nothing checkable)
- a confidence computed from evidence (not the LLM's self-report)
- the matched past fact-checks ("seen before")
- cited live web evidence, each source with a credibility tier
- red flags (warning signs), manipulation type, priority, and a plain-language explanation
- "what is actually true" and a short shareable correction message

**Core principle:** the LLM *reads and writes*; **evidence and explicit rules decide**. A model trained
on past fact-checks cannot learn truth (~70% of them are False, <0.5% True, and facts change), so the
verdict always comes from evidence retrieved at check time, and the system says **"Unverifiable"**
when it cannot find two independent credible sources.

**Scope:** English only. Inputs: text, URLs, images. **Out of scope:** video, audio, deepfake
forensics, Hindi/regional languages (Hinglish is accepted with a flag), opinions and predictions.

## 2. Data assets (in `data/raw/`, NOT in git — see DATA.md)

| File | What it is |
|---|---|
| `classified_v2.parquet` | 52,678 Indian fact-check articles from 15 IFCN-verified fact-checkers (2016 → 11 Oct 2026). 32 columns: scraped fields (site, title, url, date, content, site_verdict, verdict_norm…) + Gemini "Prompt A" labels. 52,099 rows have `is_fact_check == True`. |
| `bge_m3_primary_claim.parquet` | 52,099 rows: `article_id` + 1024-d BAAI/bge-m3 embedding of `primary_claim` (L2-normalised, max_seq_length 256). The claim memory. |

Key label columns: `primary_claim` (claim as it circulated, ≤30 words), `true_version`, `verdict`,
`domain` (11), `manipulation_type` (10), `red_flags` (list, 11), `form` (Image/Video/Text/Audio),
`verification_methods`, `ai_generated`. Weak — do not build on: `claim_source`, `purpose`, `confidence`.
All labels are LLM-made and not yet human-validated.

Prompt A (`misinfo_agent/prompts/prompt_v4a.txt`) defines `primary_claim`; box 3 must write new
claims in the same style so they match the memory.

## 3. The pipeline (16 boxes)

```
1-2 ingest ─► 3-4 extract ─► 5 memory ─► 6 prior match ─┬─► (exact & fresh) reuse prior ─┐
                                                        └─► 7 investigate ─► 9 gate ──────┤
                                       ▲  search_more (max 2) ◄──────────────┘  │        │
                                       │                    conflicting ─► 13 human      │
                                                    sufficient/insufficient ─► 8+10 assess
                                                                                  │
            11 priority ◄─────────────────────────────────────────────────────────┘
                 │
            12 decide ─► alert / unsure ─► 13 human review (pause) ─► 14-15 output ─► 16 audit
                     └─► store ───────────────────────────────────► 14-15 output ─► 16 audit
```

| Box | Module | Owner | What it does | Tools | LLM? |
|---|---|---|---|---|---|
| 1–2 Ingest | `ingest.py` | ingestion teammate | Turns text/URL/image into clean `text` + `match_text`; language + quality checks; duplicate check | unicodedata, regex, langdetect, requests, trafilatura, BeautifulSoup, Pillow, Tesseract (pytesseract), imagehash, datasketch MinHash, SQLite | no |
| 3–4 Extract | `extract.py` | ingestion teammate | Checkable claim, atomic claims (one core), entities, domain, red flags, headline-vs-body check; for images also visual description + location/time clues (evidence-only prompt) | Gemini via `llm.py`, Pydantic structured output | **1 call** |
| 5 Memory | `memory.py` | Vash | Hybrid search over 52,099 past claims: BGE-M3 dense (E @ q) + BM25, merged with Reciprocal Rank Fusion; near-duplicates collapsed into `also_debunked_by`; filters (date_before, domain) | sentence-transformers (BGE-M3), NumPy, rank_bm25, FAISS (parity check) | no |
| 6 Prior match | `prior_match.py` | Vash | exact / similar / none by thresholds; detail rule (numbers, years, ₹ amounts, names must agree); expiry rule for status claims; route = reuse_prior or investigate | rules | no |
| 7 Investigate | `investigate.py` | Vash | Fixed plan of at most five queries (neutral, verify, refute, official by domain, lead only after a similar prior) plus one top-up of at most two queries when fewer than two independent tier 1–2 groups support or refute; fetch with an SSRF check; pick passages; stance_v8; credibility (tier of the source, same registrable domain plus MinHash for independence, freshness). A group that both supports and refutes is kept and marked for the gate, which counts it on neither side → evidence E1…En. A later call skips queries already run | Tavily, trafilatura, htmldate, BGE-M3, datasketch, Gemini (stance_v8 and an optional top-up), rules | **1–3 calls** |
| 9 Gate | `gate.py` | Vash | Rules: ≥2 independent groups agree, ≥1 Tier 1–2, no Tier-1 disagreement → sufficient; credible conflict → human; else search_more (max 2) then insufficient | rules | no |
| 8+10 Assess | `assess.py` | Vash | One call: per-atomic-claim results, explanation, true_version, correction message, cited evidence IDs only. Code applies the verdict rule, checks citations exist, computes confidence from evidence features | Gemini, rules, (later) logistic-regression calibration | **1 call** |
| 11 Priority | `priority.py` | Vash | Score from harm (domain, red flags), novelty (1 − top memory cosine), reach (also_debunked_by / repeat submissions), urgency (deadline words) → High/Medium/Low. Weights hand-set + sensitivity check | rules | no |
| 12 Decide | `decide.py` | Vash | EVSI-inspired rule: alert / search_more / human / store, using harm-dependent threshold p* = a/(a+bH) | rules | no |
| 13 Human review | `review.py` | Vash | LangGraph `interrupt()` pauses; Streamlit review queue (approve / edit / reject / more evidence) resumes with `Command(resume=...)` | LangGraph, Streamlit | no |
| 14–15 Output | `output.py` | Vash | Result card (verdict, confidence, evidence, seen-before, red flags, true version, correction, limits); alerts page; email/webhook stub | Streamlit | no |
| 16 Audit | `audit.py` | Vash | One SQLite row per run (input, evidence, verdict, human action, LLM/search calls, latency, model + prompt version); approved verdicts appended to memory | SQLite | no |
| Wiring | `graph.py` | Vash | LangGraph StateGraph: nodes = boxes, conditional edges at boxes 6, 9, 12; checkpointer (SQLite); recursion limit | LangGraph | — |
| LLM access | `llm.py` | Vash | The ONLY place LLM calls happen; provider/model set in config; retries; call counting | langchain-google-genai | — |

`ingest.py` and `extract.py` must fill their `state.py` fields exactly; extract's domain must be one of `config.DOMAINS`; extract calls the LLM only through `llm.call_structured`; test against `tests/fixtures/sample_state.json`.

Typical cost per claim: **3–6 LLM calls, ~5–7 searches, ~40 s.**

## 4. State contract (`misinfo_agent/state.py`)

One `ClaimState` dict travels through every box. Each node returns only the keys it changes.
`evidence` uses `Annotated[list, operator.add]` so search rounds append, never overwrite.
Changing `state.py` requires a PR and a heads-up to the team. `tests/fixtures/sample_state.json`
is a complete example used by the UI and tests.

## 5. Evaluation (for the report)

- Time split: memory/train ≤ 2024, test 2025–26; `date_before` filter so the system only
  "remembers" fact-checks older than the claim; drop test claims whose nearest memory claim has
  cosine ≥ 0.90.
- Two conditions: fact-check sites **blocked** in web search (main, honest number) vs allowed.
- True cases from `true_version`.
- Metrics: macro-F1 per verdict class and per domain; Recall@k for memory; citation accuracy;
  calibration (ECE); abstention rate; cost and latency per claim.
- Ablation: LLM alone → +memory → +web search → full system; plus a "Gemini with search" baseline.

## 6. Engineering rules

- Deeply commented code: explain **why**, not just what.
- Never print, log or commit API keys; keys live in `.env` (see `.env.example`).
- Never commit anything under `data/`.
- Read parquet with `columns=[...]`; never load `content` unless needed.
- Never build an N × N similarity matrix over the memory (52k² ≈ 11 GB).
- All LLM calls go through `llm.py`; all outputs from LLMs are validated with Pydantic.
- Every module has tests in `tests/`; `pytest` must pass before a PR.
- Branches `feat/<area>`; never push to `main`; PRs reviewed by Vash.

## 7. Future work (not built now)

Fusion transformer for multimodal (Wang et al. 2026), CLIP embeddings, image memory from the
fact-check archive's images, paid reverse image search, video/audio (Whisper), fine-tuned
MuRIL/mDeBERTa tag model, NLI stance model (DeBERTa), full EVSI, feedback learning beyond
appending approved verdicts.
