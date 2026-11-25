# System check

Live `run_pipeline`, the path the app uses. Streamlit was not started. Counters before: Vision 17, Tavily 895. After: Vision 18, Tavily 914. One new Vision call (the rendered screenshot). 19 new Tavily searches.

| Case | Verdict | Route | search_status | vision_status | Latency | Result |
| --- | --- | --- | --- | --- | --- | --- |
| a. New text: Aadhaar mandatory for general tickets from 1 December 2026 | Unverifiable | investigate | ran (6 live, 0 cached) | — | 52.16s | PASS |
| b. PIB press release `PRID=2340950` | True | investigate | ran (6 live, 0 cached) | — | 99.43s | PASS |
| c. WhatsApp-style PNG of the img02 caption | Unverifiable | investigate | ran (7 live, 0 cached) | ran | 106.52s | PASS, with a limitation |
| d. img09 with its caption | Misleading | reuse_prior | not_needed | cached | 2.48s | PASS |
| e. img04 with no caption | Other | investigate | not_needed | cached | 2.40s | PASS |
| f. Recycled rumour, article `fcaebc7ad5f3` | Misleading | reuse_prior, exact | not_needed | — | 0.16s | PASS |
| g. Review queue | Unverifiable, review_status approved | — | — | — | — | PASS |

No case raised. No case was a clear code bug, so nothing was changed and none was re-run.

## What each case was expected to do

**a.** A claim that is not in the archive is investigated. The cap is 8 searches. This run used 6. The gate was insufficient: the pages that came back did not give two independent credible groups, so the card abstains. That is the rule, not a crash.

**b.** The URL is `https://pib.gov.in/PressReleaseDetail.aspx?PRID=2340950`, taken from the PIB home page on 25 November 2026. The fetched body is the JavaScript fallback. The headline was still read: the President of India attended Bali Jatra 2026 in Cuttack. Two tier 1–2 pages (presidentofindia.gov.in, ddnews.gov.in) support that sentence, so the verdict is True.

**c.** Pillow drew the img02 caption as a green chat bubble. OCR returned the sentence, with "Jharkhand" misread once; extract corrected it and checked that claim. Vision also ran, because the OCR text is under 40 words and a short caption-less image is treated as a photograph. Vision found no pages of that rendered card. The text check abstained.

**d.** The same photograph and caption were already in image memory, so the stored Misleading verdict was reused and search did not run. Provenance was attached from the Vision cache: 12 pages, first seen 2026-07-27 on english.factcrescendo.com.

**e.** No caption, so the photograph is not a checkable claim. Verdict Other. Provenance still ran from the cache: 8 pages, no trusted full-match date, so first_seen is empty.

**f.** The first correctly reused row of `data/eval/results_system_recycled.jsonl`. Gold Misleading, route reuse_prior, same article `fcaebc7ad5f3`. This run matched that.

**g.** `list_pending()` had 14 rows. `apply_review(approve)` on thread `929ababed086415388e353c7c5d48a6a` (a Yogi Adityanath wedding claim, queued as Unverifiable). `get_card()` returned that verdict with `review_status` approved, and the row left the queue.

## Known limitations

- A WhatsApp screenshot whose OCR is under 40 words is reverse-searched as a photograph. This card spent one Vision call and found nothing. The text check is the part that decided the verdict.
- A PIB page fetched without JavaScript returns the fallback body. The headline can still be enough, as it was here, but a release whose title does not state the claim would have little text to extract.
- Image memory reuses a stored photo verdict. img09 did not investigate again.
- On the 16-image blocked comparison, `PHOTO_RULE=v2` makes one correct decision (img15) and abstains on the other 15. A September page does not refute an August caption. img05 and img07, both gold True, stay Unverifiable.
- "The Reserve Bank of India was established in 1935" is True on a cache-only rerun (4 cached searches, 0 live, rbi.org.in and pib.gov.in). An earlier demo card was Unverifiable because the stance call failed and the empty table was described as "no source we read addressed this part." That was a system failure, not a missing page and not a gate rule. The retry did not need a code change.
