# Demo runbook

Cache-only check through `run_pipeline`. `SEARCH_CACHE_ONLY=1`. Streamlit was not started. Tavily stayed at 895. Vision stayed at 17 (no live call; the image was answered from image memory).

## 1. Recycled claim — PASS

Paste this text:

```
The Kerala High Court protected a man named Farman from arrest under the POCSO Act by ruling that Islam allows marriage at age nine.
```

First row of `data/eval/results_system_recycled.jsonl` (`fcaebc7ad5f3`). Gold Misleading. That row reused the same article.

Expected card: verdict Misleading. Exact past fact-check, search not needed. Seen before: youturn, 2026-04-16.

Actual: verdict Misleading, confidence 0.8581, no caveat. `search_status` not_needed, `route` reuse_prior, `match_type` exact, seen site youturn, seen date 2026-04-16. Latency 10.71s. Searches live 0, cached 0.

PASS.

## 2. State count — PASS

Paste this text:

```
India has 29 states in total.
```

Expected card: verdict False (India has 28 states). The lower-tier caveat should be on the card if no tier 1–2 page is in the evidence.

Actual: verdict False, confidence 0.3296. Caveat: "Lower-tier sources only: no tier 1-2 source was found." `search_status` ran, searches live 0, cached 5. `route` investigate. Latency 27.46s. Explanation: the claim says 29 states, and the cached pages say 28 states and 8 union territories.

PASS. Read the caveat out loud. The confidence is low because the pages are lower tier.

## 3. Reserve Bank — FAIL — do not demo

Paste this text:

```
The Reserve Bank of India was established in 1935.
```

Expected card: verdict True. The Reserve Bank of India Act is 1935, and the bank began operating on 1 April 1935.

Actual: verdict Unverifiable, confidence 0.9168, no caveat. `search_status` ran, searches live 0, cached 4. `route` investigate. Latency 71.77s. Explanation: "No source we read addressed this part of the claim."

FAIL. do not demo. The cache had no page that stated the founding year, so the card abstains on a claim the room will know is true.

## 4. Smoke photo — PASS

Upload `data/eval/images/smoke.jpg` (same file as img01). Caption:

```
The photo shows a young girl being protected by her brother, who are victims of the August 2026 Nepal floods.
```

Re-checked after the exact-match rule, through `run_pipeline`, with a fresh image-memory database and `SEARCH_CACHE_ONLY=1`. Vision read the cache.

Expected card: the photo-lean line is present. Verdict Misleading, out of context, when two or more full-match groups predate the event. One tier 1–2 full match is the same rule and also yields Misleading. Otherwise Unverifiable, with an "online since" line.

Actual: verdict Misleading, refute_kind out_of_context, confidence 0.2789, no caveat. `search_status` cache_only_miss, searches live 0, cached 0. Latency 18.88s. Vision status cached. One full match: thequint.com, tier 2, dated 2026-09-07. The check is dated today (2026-11-25), so that page is before the claimed event. n_full 1, n_partial 11, n_dated_earlier 1. Photo lean: "This photo has been online since at least 2026 (1 sites)".

PASS. Read the low confidence. This card is not the blocked eval: fact-check sites are included here, and img01 on the blocked set stayed Unverifiable because that page was dropped and no other full match had a trusted date.
