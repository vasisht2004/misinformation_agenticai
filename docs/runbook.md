# Runbook

How to run a check, and the card each input should show. The app calls `run_pipeline`. These notes use that function. Streamlit is optional and is not required for a check.

## Start

```bash
source .venv/bin/activate
cp .env.example .env   # fill the keys once
pytest                 # offline; does not spend a key
streamlit run app/streamlit_app.py
```

The title is two lines, AGENTIC AI SYSTEM and FOR MISINFORMATION DETECTION. The ✦ is coral. The app sets `SEARCH_CACHE_ONLY`, so a demo reads the search cache and does not call Tavily. A live check unsets that variable. `RECORD_SUBMISSIONS` stays off unless the sidebar switch is on.

Tesseract has to be installed. Without it, a screenshot has no OCR text.

One claim from the terminal:

```bash
python scripts/run.py --text "The Reserve Bank of India was established in 1935."
```

## Text

Paste the sentence. The card has a verdict, a confidence from the evidence table, citations, and `search_status`.

- A sentence already in the archive as an exact fresh fact-check: verdict copied from that article, route `reuse_prior`, `search_status` `not_needed`. The recycled Kerala High Court sentence is Misleading, article `fcaebc7ad5f3`, seen before on youturn.
- "The Reserve Bank of India was established in 1935." Cache-only, this is True. Two tier 1–2 groups (rbi.org.in and pib.gov.in). `search_status` ran, searches live 0, cached 4.
- "India has 29 states in total." False or Misleading, with the lower-tier caveat when no tier 1–2 page is in the evidence.
- A new sentence the archive does not contain is investigated. "Indian Railways has made Aadhaar mandatory for all general-ticket passengers from 1 December 2026" came back Unverifiable, 6 live searches, gate insufficient. Unverifiable means two credible sources were not found. It is not a crash.
- A greeting or a sentence with no checkable fact is Other. Search does not run.
- `search_status` is `ran`, `cache_only_miss`, `not_needed`, `monthly_cap`, or an error. The card shows live and cached counts separately.

## Link

Paste the article URL. Ingest fetches the page (private addresses are refused), extract writes one claim from the headline and the opening, and investigate checks that claim.

A current PIB release is `https://pib.gov.in/PressReleaseDetail.aspx?PRID=2340950`. The body without JavaScript is thin. The headline was enough: the President attended Bali Jatra 2026 in Cuttack. The card was True, with presidentofindia.gov.in and ddnews.gov.in supporting it, 6 live searches.

If the fetch fails, the card says so and does not invent the article.

## Screenshot

Upload a picture of a forward, or pass the PNG path with `input_type="image"` and no caption. OCR is the claim. A rendered WhatsApp bubble of the img02 caption was read, extract fixed one OCR miss, and the claim was investigated. That card was Unverifiable after 7 live searches.

A screenshot under 40 OCR words is also sent to Vision. That call found no pages of the rendered bubble. The verdict came from the text check.

## Photo and caption

Upload the photograph and type the caption. Vision uses the cache when the same JPEG was seen before. `PHOTO_RULE` is `v2`: a full match can date the file, and a partial counts only with a trusted date at least a year before the event named in the caption. "August 2026" is that month, not today. Out of context is Misleading. Wayback dating is off. When it is on, an earlier CDX capture can supply the "online since" year, and it cannot move a page's own date later.

img09 with its Assam-flood caption, on a fresh image memory, was Unverifiable. Vision was cached. first_seen was 2026-07-27 on english.factcrescendo.com, from the page date (htmldate), and the line was "This photo has been online since at least 2026 (1 sites)". That year is not a Wayback capture.

## Photo, no caption

The card is Other. Provenance still runs, so the page can show where similar images have appeared. img04, a PIB photograph with no caption, was Other, vision cached, 8 pages, and no trusted date on a full match, so there is no "online since" year.

## Review queue

Review does not pause the run. The card is shown, and a row that needs a person stays on the queue.

```python
from misinfo_agent.review import apply_review, get_card, list_pending

pending = list_pending()
apply_review(pending[0]["thread_id"], "approve")
get_card(pending[0]["thread_id"])
```

Approve keeps the verdict and sets `review_status` to approved. The row leaves the queue. Edit replaces the verdict. Reject clears it. None of the three calls a model or searches.

## What a card always has

Verdict, confidence, explanation, citations or the reason there are none, `search_status`, and the live and cached search counts. An image card also has provenance and, when a full match has a trusted date, the "online since" line. `model_lean` is labelled as a guess and is not the verdict. When the gate is `error` (a stance or model step failed), the explanation and the correction say "Checking failed partway through — please run this check again." They do not say that no source addressed the claim.
