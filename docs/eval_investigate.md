# Investigate evaluation

Box 7 searches the live web when box 6 does not reuse a prior fact-check. The model reads passages and writes a stance. Tiers, independence, and freshness are rules. The verdict is not decided here. The gate (box 9) says whether the evidence is enough, and it says Unverifiable when two independent credible sources cannot be found.

Numbers below are from the stance_v3 run on 14 Nov 2026, eight claims, `BLOCK_FACTCHECK_SITES` on. Planned searches were cache hits. The monthly counter moved from 122 to 129.

## What the box does

**Query plan.** At most five lines, chosen without a model. Neutral is the claim, trimmed to twenty words. Verify asks whether the named entities are true. Refute appends `fake OR false OR hoax OR misleading`. Official is keywords only, restricted to the hosts in `OFFICIAL_DOMAINS` for that domain. Lead is added only when the prior match is similar, and it searches the older wording. Each line keeps its best three hits before the cap of 25.

**Search gateway.** Every call goes through `search.py`. A cached or fake answer does not count against the monthly cap or the eight searches allowed for one claim. Fact-check hosts are dropped when the block is on, including a registrable domain whose name contains "factcheck", plus the extra list (`factcheckhub.com`, `checkyourfact.com`, `africacheck.org`, `factly.in`) and the path token `webqoof`. When the leakage guard is on, Tavily's `end_date` is the day after `claim_date`, so the claim's own day is included and a later write-up is not.

**Fetch safety.** A host is opened only after every address is a public IP, and each redirect is checked again. The download has a wall clock, not a per-read timeout. A long Tavily extract is used instead of a second download. Trafilatura reads HTML. Circulation hosts (social posts and stock libraries) are counted as sightings and never become passages. A page that is not English is dropped.

**Passages.** BGE-M3, the same encoder as the memory, is loaded once. Windows are three sentences. A window under 20 words is dropped, and so is a cosine under 0.45. At most two windows per page, two per registrable domain, and forty per claim. The stance call sees the best twenty.

**Dates and the leakage guard.** The first trusted date wins: htmldate on the HTML, a date in the URL path, trafilatura's metadata, then Tavily. Today and the future are not publication dates. If the host refuses the date fetch, a "Published on" line or a byline date in the text we already have is used. An event the article mentions, such as "8 November 2016", is not the page's date. `age_days` is `claim_date` minus `published_date`. With `EVAL_DROP_AFTER_CLAIM` on, a page dated after the claim is dropped. Undated pages are kept unless `EVAL_UNDATED` is also drop.

**Stance v3.** One batched call, `purpose="stance"`, `prompt_version="stance_v3"`. Supports requires every key detail: who, what, amount, place, and timing ("now", "from 1 December", "permanently"). A partial confirmation is neutral. A contradicted detail is a refute, including "retail only" against "completely" and "temporary" against "permanent". Old evidence cannot support a claim that says "now", "new", "from <date>", or "permanently". It can refute that claim when the change already happened on another date, or has ended. The code forces an old support to neutral when the page is at least a year older than the claim. It does not rewrite a refute. The quote must be copied from the passage. If it is not there after whitespace and quote marks are flattened, the stance becomes neutral and `quote_unverified` is set. A passage that only repeats the claim is neutral.

**Tiers.** The tier is the source, not whether we fetched the page. `snippet_only` records a snippet separately and never changes the tier. Matching is on the registrable domain. Curated lists are applied first and always win: tier 1 is a government suffix (`.gov`, `.gov.`, `.gob.`, `.gc.ca`, `.nic.in`) or an official body already in `OFFICIAL_DOMAINS`; tier 2 is the curated newsroom and journal list (PTI, ANI, The Hindu, NDTV, and the others in `TIER2_HOSTS`) or `.edu`, `.edu.`, `.ac.`; tier 3 is other news, Wikipedia, hospitals, and NGOs we already classed; named blog hosts are tier 4. A host none of those lists name is then looked up in `data/source_tiers_rsp.csv`, built from the English Wikipedia page [Wikipedia:Reliable sources/Perennial sources](https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources) (WP:RSP) by `scripts/build_source_tiers.py`. That page is a summary of volunteer editing discussions, not a directory of every reliable publisher. Its text is available under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0). Attribute Wikipedia contributors, and the revision id the script prints when it writes the file. Generally reliable maps to tier 2, no consensus or additional considerations to tier 3, and generally unreliable, deprecated, or blacklisted to tier 4. Anything still unnamed is tier 4.

**Independence.** The same registrable domain is one group. MinHash (5-word shingles, 128 permutations, LSH threshold 0.5) joins a near-copy on another domain. Inside a group, a later row with the same normalised quote is dropped. Evidence ids continue on a second call (`E5` after `E4`), and that call skips queries already stored in `investigate_stats`.

**Top-up.** After the first stance pass, if fewer than two independent groups of tier 1 or 2 support or refute, one call (`purpose="topup_queries"`) proposes at most two new queries. They run only inside the searches still left on the claim. Stance is called again only for passages whose URL was not already read. At most one top-up per call of `investigate()`. A search-cap or model error returns what is already in hand and a note. It does not raise.

## Eight claims

| Claim | Evidence | Tier ≤2 groups, support / refute | Top-up | Searches | LLM calls | Seconds |
|---|---|---:|---|---:|---:|---:|
| ₹500 notes invalid from 1 December | Refutes. RBI 8 and 9 Nov 2016, PIB 8 Nov 2016, BBC 14 Nov 2016 | 0 / 3 | no | 4 | 1 | 30.9 |
| Hot lemon water kills coronavirus | Refutes. National Academies, Global News, UQ, Euronews | 0 / 4 | no | 5 | 1 | 23.4 |
| Video shows flood in Chennai | Two supports, both snippets (Al Jazeera 16 Oct 2024, NDTV) | 2 / 0 | yes | 7 | 3 | 38.5 |
| Saudi dress, women may choose | One tier-4 support (bronx.com, 1 Apr 2025). No DW 2018 support | 0 / 0 | yes | 6 | 3 | 31.8 |
| HSBC withdrawing from Sri Lanka | Refutes. Economic Times: retail sale only, 24 Sep 2025 | 0 / 1 | yes | 6 | 3 | 46.5 |
| Warning labels on samosas | Refutes. Moneycontrol and NDTV, 15 Jul 2025 | 0 / 2 | no | 4 | 1 | 15.6 |
| Telegram permanently banned | Refutes. Temporary block until 22 Jun 2026 (CNBC, Economic Times, BBC) | 0 / 3 | no | 4 | 1 | 19.7 |
| Hussain demanded a Sooryavanshi ban | No supports or refutes | 0 / 0 | yes | 6 | 3 | 30.7 |

Rows are in `data/eval/day3_evidence.csv` (at most five per claim, `stance_correct` empty) and `data/eval/day4_report.json`. Both live under `data/` and are not in git.

## Hand judgement

Passage precision@5 is 82.5% (33/40). The Wilson 95% interval is about 68–91%. Two people labelled the passages. They agreed on 85% before review, and the disagreements were then reviewed.

Stance accuracy is 84% (21/25). The Wilson 95% interval is about 62–90%. The labels are hand-judged.

## Cost per claim

| | Median | Max |
|---|---:|---:|
| Searches | 5.5 | 7 |
| LLM calls | 2 | 3 |
| Prompt tokens | 2759 | 5136 |
| Output tokens | 1095 | 2115 |
| Seconds | 30.8 | 46.5 |

`calls_today()` moved from 30 to 46 on that run. Sixteen of those calls were new. The rest of the stance work in this box, on earlier prompt versions, is already inside the 30.

## Known limits

An absence of evidence is not a refutation. The Hussain claim ended with no supporting or refuting row. The gate should say Unverifiable, not False.

A picture or video claim cannot be confirmed by text that merely describes a video. Chennai has two text supports and no view of the file. That limit is for box 9 and assess to enforce, not for this box to invent.

Tiers are India-centric. The tier-2 list is wires and the outlets a reader of Indian news would hold to a corrections desk, plus journals and universities. A good local paper that is not on the list stays tier 3 or 4.

The top-up rarely adds a new evidence row. On this run it fired for four claims and the new rows did not create a second tier 1–2 group where the first pass had none. An ablation is planned.

Search and model calls sit on free-tier quotas (eight searches per claim, a monthly Tavily cap, a daily Gemini cap). A cap returns a partial table and a note.

The validation sample is eight claims. The intervals above are wide because 40 passages and 25 stance labels are small samples.
