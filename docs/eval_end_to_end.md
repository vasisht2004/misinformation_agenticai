# End-to-end evaluation

The allowed numbers in this file are superseded — evaluation setup bug, see diagnosis. They are not the condition the code runs now.

## Diagnosis

Eight claims, traced from the cached searches and the stored passages. No new model calls. Four were decided when fact-check sites were blocked and abstained when they were allowed (`fcaebc7ad5f3`, `2dcc928eca48`, `84a06ffe939b`, `0ab825964c1e`). The other four are `e2230f5774ce` and `90aad4c39280` (both decided) and `20eb7385815d` and `0179258fa67c` (both abstained).

The phrase rule is not applied when blocking is off. `_quotes_factcheck` returns false as soon as the block flag is off. On `e2230f5774ce` the allowed pages contained 4 windows with a fact-check phrase, and the rule dropped 0 of them.

Own-article exclusion matches the canonical URL, not the registrable domain. `_same_page` compares that one URL, ignoring scheme and tracking parameters. A different path on the same host is kept. Across the allowed caches, 1 fact-check hit was the claim's own URL.

The after-claim rule removed 0 fact-check hits. The allowed caches hold 9 unique fact-check hits. One is the own article. None of the nine is dated after the claim. The search already asks for pages on or before the claim day, and no fact-check hit that came back was later.

| Claim | Condition | Hits (fact-check / other) | What was removed | Passages | Stance |
| --- | --- | --- | --- | --- | --- |
| fcaebc7ad5f3 | Blocked | 19 (0 / 19) | circulation 7; fetch fail 0; cap of 25 dropped 0; own URL 0 | 12 | yes (llm 6) |
| fcaebc7ad5f3 | Allowed | 26 (0 / 26) | circulation 6; cap of 25 dropped 1 other URL; 7 past the 12 fetches kept as snippets; phrase windows dropped 0; own URL 0; after-claim fact-check 0 | 22 | yes (llm 5) |
| 2dcc928eca48 | Blocked | 25 (0 / 25) | circulation 10; fetch fail 0; cap 0 | 11 | yes (llm 6) |
| 2dcc928eca48 | Allowed | 24 (0 / 24) | circulation 13; fetch fail 1; cap 0; fact-check hits 0 | 7 | yes (llm 6) |
| 84a06ffe939b | Blocked | 25 (0 / 25) | circulation 1; cap 0 | 30 | yes (llm 7) |
| 84a06ffe939b | Allowed | no cached open-web hits | per-claim search cap (8) discarded the gather | 0 | no (llm 1, extract only) |
| 0ab825964c1e | Blocked | 27 (0 / 27) | circulation 6; fetch fail 2; cap of 25 dropped 2 | 19 | yes (llm 4) |
| 0ab825964c1e | Allowed | no cached open-web hits | per-claim search cap (8) discarded the gather | 0 | no (llm 1, extract only) |
| e2230f5774ce | Blocked | 18 (0 / 18) | circulation 2; cap 0 | 14 | yes (llm 4) |
| e2230f5774ce | Allowed | 17 (1 / 16) | circulation 1; the 1 fact-check page is dated on or before the claim; 4 phrase windows, 0 dropped | 18 | yes (llm 4) |
| 90aad4c39280 | Blocked | 0 | no web search; decided False from memory | 0 | no (llm 2) |
| 90aad4c39280 | Allowed | 0 | no web search; decided False from memory | 0 | no (llm 2) |
| 20eb7385815d | Blocked | 21 (0 / 21) | circulation 7; cap 0 | 11 | yes (llm 4) |
| 20eb7385815d | Allowed | 22 (0 / 22) | circulation 2; 8 past the 12 fetches kept as snippets; fact-check hits 0 | 20 | yes (llm 4) |
| 0179258fa67c | Blocked | 33 (0 / 33) | circulation 4; fetch fail 1; cap of 25 dropped 8 | 36 | yes (llm 4) |
| 0179258fa67c | Allowed | 33 (0 / 33) | circulation 2; cap of 25 dropped 8; fact-check hits 0 | 32 | yes (llm 4) |

Cause. Allowed did not reuse the blocked searches. The block flag is part of the cache key, so every allowed query was a new Tavily call on the same words. The blocked run had already used most of the monthly budget of 800. From the 22nd claim, 29 claims stopped on `monthly search cap reached (800 calls)` and kept no pages: one Gemini call, the extract, and no stance. Four claims before that, including `84a06ffe939b` and `0ab825964c1e`, hit the per-claim cap of 8, and that exception discards the pages the other queries had already fetched. On the claims that did finish a search, fact-check hits were rare: 9 unique in the allowed caches, and the open list is a different five pages, so one blocked evidence URL was missing for `fcaebc7ad5f3` and for `2dcc928eca48`. That gap is smaller than the caps. The caps are why allowed abstained on claims blocked had decided.

The code now runs the blocked search first, which is a cache hit, and adds fact-check pages in their own budget of at most 8 fetches. The phrase rule stays off for that budget. The own article is still excluded by its canonical URL. After-claim dropping is unchanged, and on this run it removed 0 fact-check pages.

A re-run was not started. The monthly search cap is full, so fact-check pages that were never cached cannot be fetched until it resets. Extract and stance on the blocked passages are cache hits. Five claims already have a fact-check page in the open-web cache that is not the own article and is not dated after the claim. Each of those is one new stance call if the page yields a passage, and an assess call only if the new row makes the gate pass. The other claims add no stance call while the search cap is full.

Rewritten by `scripts/score_eval.py`. The numbers are the result lines that exist, joined to the test set on `article_id`. A test row with no result line has not been run.

Test set rows: 50. Blocked results scored: 50. Allowed results scored: 50. Baseline results scored: 50.

Leak count, blocked: 0. Expected 0. A leak there is the claim's own article, or a near-duplicate copy, in a memory match, a prior-match lead, or an evidence URL.

Leak count, allowed: 0. Expected 0. A leak there is only the claim's own article. Another fact-checker's article is allowed.

Fact-check evidence used (allowed): 0.100 (5/50) [0.043, 0.214]. This is another outlet's fact-check in the evidence table, or a prior the route reused. The source article itself is not included.

Blocked abstentions that became decided once fact-check sites were allowed: 1. Correct among those: 1.

Blocked excludes fact-check sites, drops pages published after the claim date, searches memory with `date_before` equal to the claim date, and excludes the source article plus the near-duplicate copies memory would collapse into `also_debunked_by`. Allowed uses the same date filter and still drops the source article's own URL from search results, fetched pages and evidence. Memory search in the allowed run excludes only that article's id, so an earlier fact-check of the same rumour can be retrieved. Submissions are not recorded. The LLM cache is on. A review pause is not answered: the provisional verdict is stored with `needs_human` and counted as an abstention.

The baseline is one Gemini call on the claim text. It returns False, Misleading, True, or Unverifiable, plus its own confidence. That confidence is the model's self-report. The system's confidence is computed from the evidence.

## Definitions

A decided claim has verdict False, Misleading, or True and `needs_human` false. Abstention is Unverifiable or `needs_human`. A claim that is both counts once. The 3-class accuracy is correct decided claims over decided claims, and 3-class macro-F1 is the unweighted mean of the three class F1 scores; a class with no decided gold row scores 0. The 2-class score collapses False and Misleading to not-true and leaves True as True, on the same decided claims. Abstention is not part of that collapse. Coverage-adjusted accuracy is correct decided claims over every scored claim, using the 3-class correct count. ECE uses 10 equal-width confidence bins on decided claims. Wilson intervals are 95%.

## Blocked, allowed, and baseline

| Metric | Blocked | Allowed | Baseline |
| --- | --- | --- | --- |
| Scored claims | 50 | 50 | 50 |
| Decided | 17 | 5 | 21 |
| Correct | 8 | 4 | 13 |
| Accuracy, 3-class (decided) | 0.471 (8/17) [0.262, 0.690] | 0.800 (4/5) [0.376, 0.964] | 0.619 (13/21) [0.409, 0.792] |
| Macro-F1, 3-class (decided) | 0.446 | 0.619 | 0.704 |
| Accuracy, 2-class (not-true vs True) | 0.706 (12/17) [0.469, 0.867] | 0.800 (4/5) [0.376, 0.964] | 1.000 (21/21) [0.845, 1.000] |
| Macro-F1, 2-class (decided) | 0.664 | 0.444 | 1.000 |
| Abstention | 0.660 (33/50) [0.522, 0.776] | 0.900 (45/50) [0.786, 0.957] | 0.580 (29/50) [0.442, 0.706] |
| Coverage-adjusted accuracy | 0.160 (8/50) [0.083, 0.285] | 0.080 (4/50) [0.032, 0.188] | 0.260 (13/50) [0.159, 0.396] |


## Per class

### Blocked

| Class | Precision | Recall | F1 | Support | Recall 95% Wilson |
| --- | --- | --- | --- | --- | --- |
| False | 0.667 | 0.500 | 0.571 | 8 | [0.215, 0.785] |
| Misleading | 0.333 | 0.167 | 0.222 | 6 | [0.030, 0.564] |
| True | 0.375 | 1.000 | 0.545 | 3 | [0.438, 1.000] |

### Allowed

| Class | Precision | Recall | F1 | Support | Recall 95% Wilson |
| --- | --- | --- | --- | --- | --- |
| False | 1.000 | 0.750 | 0.857 | 4 | [0.301, 0.954] |
| Misleading | 1.000 | 1.000 | 1.000 | 1 | [0.207, 1.000] |
| True | 0.000 | 0.000 | 0.000 | 0 | n/a |

### Baseline

| Class | Precision | Recall | F1 | Support | Recall 95% Wilson |
| --- | --- | --- | --- | --- | --- |
| False | 0.500 | 0.625 | 0.556 | 8 | [0.306, 0.863] |
| Misleading | 0.625 | 0.500 | 0.556 | 10 | [0.237, 0.763] |
| True | 1.000 | 1.000 | 1.000 | 3 | [0.438, 1.000] |


## Per domain

### Blocked

| Domain | n | Abstained | Decided | Correct | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Politics & elections | 5 | 2 | 3 | 2 | 0.667 | 0.444 |
| Religion & communal | 5 | 4 | 1 | 0 | 0.000 | 0.000 |
| Defence & international | 5 | 5 | 0 | 0 | n/a | n/a |
| Crime & law | 4 | 2 | 2 | 1 | 0.500 | 0.222 |
| Society & other | 5 | 4 | 1 | 0 | 0.000 | 0.000 |
| Health | 5 | 4 | 1 | 1 | 1.000 | 0.333 |
| Disasters & accidents | 3 | 2 | 1 | 0 | 0.000 | 0.000 |
| Finance & scams | 5 | 1 | 4 | 3 | 0.750 | 0.556 |
| Entertainment & celebrities | 4 | 3 | 1 | 1 | 1.000 | 0.333 |
| Sports | 4 | 3 | 1 | 0 | 0.000 | 0.000 |
| Science & technology | 5 | 3 | 2 | 0 | 0.000 | 0.000 |

### Allowed

| Domain | n | Abstained | Decided | Correct | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Politics & elections | 5 | 4 | 1 | 1 | 1.000 | 0.333 |
| Religion & communal | 5 | 5 | 0 | 0 | n/a | n/a |
| Defence & international | 5 | 4 | 1 | 1 | 1.000 | 0.333 |
| Crime & law | 4 | 4 | 0 | 0 | n/a | n/a |
| Society & other | 5 | 5 | 0 | 0 | n/a | n/a |
| Health | 5 | 5 | 0 | 0 | n/a | n/a |
| Disasters & accidents | 3 | 3 | 0 | 0 | n/a | n/a |
| Finance & scams | 5 | 4 | 1 | 1 | 1.000 | 0.333 |
| Entertainment & celebrities | 4 | 3 | 1 | 1 | 1.000 | 0.333 |
| Sports | 4 | 3 | 1 | 0 | 0.000 | 0.000 |
| Science & technology | 5 | 5 | 0 | 0 | n/a | n/a |

### Baseline

| Domain | n | Abstained | Decided | Correct | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Politics & elections | 5 | 4 | 1 | 1 | 1.000 | 0.333 |
| Religion & communal | 5 | 5 | 0 | 0 | n/a | n/a |
| Defence & international | 5 | 3 | 2 | 2 | 1.000 | 0.667 |
| Crime & law | 4 | 3 | 1 | 0 | 0.000 | 0.000 |
| Society & other | 5 | 4 | 1 | 1 | 1.000 | 0.333 |
| Health | 5 | 0 | 5 | 3 | 0.600 | 0.556 |
| Disasters & accidents | 3 | 2 | 1 | 0 | 0.000 | 0.000 |
| Finance & scams | 5 | 1 | 4 | 3 | 0.750 | 0.489 |
| Entertainment & celebrities | 4 | 3 | 1 | 1 | 1.000 | 0.333 |
| Sports | 4 | 3 | 1 | 1 | 1.000 | 0.333 |
| Science & technology | 5 | 1 | 4 | 1 | 0.250 | 0.333 |


## Confusion matrix

Rows are the gold label. Columns are the predicted label. Decided claims only.

### Blocked

| Gold \ predicted | False | Misleading | True |
| --- | --- | --- | --- |
| False | 4 | 2 | 2 |
| Misleading | 2 | 1 | 3 |
| True | 0 | 0 | 3 |

### Allowed

| Gold \ predicted | False | Misleading | True |
| --- | --- | --- | --- |
| False | 3 | 0 | 1 |
| Misleading | 0 | 1 | 0 |
| True | 0 | 0 | 0 |

### Baseline

| Gold \ predicted | False | Misleading | True |
| --- | --- | --- | --- |
| False | 5 | 3 | 0 |
| Misleading | 5 | 5 | 0 |
| True | 0 | 0 | 3 |


## Citations

A web citation is valid when its id is one of the stored evidence rows and the quote is in that row's passage. C1 is the archive prior. It is valid when the run kept a prior. The baseline cites nothing.

### Blocked

Web citations: 57. Id present: 57. Id present and quote verified: 57.
Archive citations (C1): 2. Prior present: 2.
Web citation validity: 1.000 (57/57) [0.937, 1.000].

### Allowed

Web citations: 18. Id present: 18. Id present and quote verified: 18.
Archive citations (C1): 2. Prior present: 2.
Web citation validity: 1.000 (18/18) [0.824, 1.000].

### Baseline

No citations.

## Calibration

Blocked ECE (10 bins) on 17 decided claims with a confidence: 0.459.
Allowed ECE (10 bins) on 5 decided claims with a confidence: 0.297.
Baseline ECE (10 bins) on 21 decided claims with a confidence: 0.325.

### Blocked

| Bin | Range | Decided claims |
| --- | --- | --- |
| 0 | [0.0, 0.1) | 0 |
| 1 | [0.1, 0.2) | 0 |
| 2 | [0.2, 0.3) | 0 |
| 3 | [0.3, 0.4) | 0 |
| 4 | [0.4, 0.5) | 0 |
| 5 | [0.5, 0.6) | 0 |
| 6 | [0.6, 0.7) | 0 |
| 7 | [0.7, 0.8) | 2 |
| 8 | [0.8, 0.9) | 7 |
| 9 | [0.9, 1.0] | 8 |

### Allowed

| Bin | Range | Decided claims |
| --- | --- | --- |
| 0 | [0.0, 0.1) | 0 |
| 1 | [0.1, 0.2) | 0 |
| 2 | [0.2, 0.3) | 0 |
| 3 | [0.3, 0.4) | 0 |
| 4 | [0.4, 0.5) | 0 |
| 5 | [0.5, 0.6) | 0 |
| 6 | [0.6, 0.7) | 0 |
| 7 | [0.7, 0.8) | 2 |
| 8 | [0.8, 0.9) | 1 |
| 9 | [0.9, 1.0] | 2 |

### Baseline

| Bin | Range | Decided claims |
| --- | --- | --- |
| 0 | [0.0, 0.1) | 0 |
| 1 | [0.1, 0.2) | 0 |
| 2 | [0.2, 0.3) | 0 |
| 3 | [0.3, 0.4) | 0 |
| 4 | [0.4, 0.5) | 0 |
| 5 | [0.5, 0.6) | 0 |
| 6 | [0.6, 0.7) | 1 |
| 7 | [0.7, 0.8) | 1 |
| 8 | [0.8, 0.9) | 8 |
| 9 | [0.9, 1.0] | 11 |

## What the system abstained on

The gate reason is the sentence box 9 wrote. A review pause with no gate sentence shows the gate label.

### Blocked

| Claim | Gold | Gate reason |
| --- | --- | --- |
| Turkey issued an 18-hour ultimatum to the Indian warship INS Trikhand in the Mediterranean Sea. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| International passengers have started refusing to sit in aircraft flown by Indian pilots after the June 12 crash. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Banana Sport SPF 50 sunscreen contains harmful carcinogens like Avobenzone, Oxybenzone, Octinoxate, Homosalate, and Octocrylene. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The PMO of India has introduced a dedicated hotline (9851145045) for citizens to report corruption, bribery, and misconduct. | False | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. A2 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Sanghi Prince was detained for criticising Udhayanidhi Stalin in a video. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Domino's is officially launching a KitKat pizza. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Tamil Nadu has the highest number of murders in India, with 8,900 murders in the last 5 years under DMK-Congress rule. | False | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| An Indian oil tanker paid Iran in Chinese yuan to pass through the Strait of Hormuz. | False | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The Directorate General of Civil Defence and India Meteorological Department issued a high alert on extreme heat warning temperatures will rise between 45°C and 55°C. | False | A1 is not yet supported or refuted: 1 group supports it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| A.R. Rahman blamed Hindi or communal factors for losing work in Bollywood. | Misleading | A1 is not yet supported or refuted: 1 group supports it and 0 groups refute it. A2 is supported: 2 independent groups support it, and 2 of them are tier 1 or 2. The core claim A1 does not have enough evidence, and the search limit is used. |
| Domino's Pizza, sauces, and cheese contain harmful and carcinogenic ingredients. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 3 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Hindu minority rights activist Khushi Roy was arrested by the Bangladeshi government for participating in protests. | Misleading | A1 is not yet supported or refuted: 1 group supports it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| High Aadhaar saturation figures in Muslim-majority districts of Bihar indicate a suspicious anomaly. | Misleading | A1 is not yet supported or refuted: 3 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| South African cricketer Hashim Amla was fasting and scored 311*, while Shami drank water just after bowling 2 overs in the Champions Trophy semi-final. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| China has threatened to cut off the flow of the Brahmaputra River to India if India uses water against Pakistan. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Tabu stated that she enjoys being single and only needs a man to sleep in bed. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Colgate Total toothpaste contains dangerous ingredients like Sodium Lauryl Sulfate, Triclosan, and artificial sweeteners that cause health problems. | Misleading | A1 is not yet supported or refuted: 2 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Over 17 lakh women have gone missing in the country in the past 11 years under PM Modi's rule. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is supported: 2 independent groups support it, and 2 of them are tier 1 or 2. The core claim A1 does not have enough evidence, and the search limit is used. |
| AI agents (likely to be Meta AI) can access WhatsApp group and individual chats and contact details unless the Advanced Chat Privacy feature is enabled. | False | A1 is not yet supported or refuted: 3 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| MS Dhoni criticized head coach Gautam Gambhir in an interview, alleging he has ruined Indian cricket through arbitrary decisions. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| People in Pakistan are being forced to evacuate villages due to a radiation leak. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Users will be charged a fee for making UPI transactions over ₹2,000 via apps like Paytm, PhonePe, and Google Pay. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Colgate Optic White Stain Fighter Toothpaste ingredients cause severe health issues like gastrointestinal irritation and central nervous system depression. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Indian Muslims celebrated Iran's victory and never celebrate India's military successes such as Kargil or Operation Sindoor. | False | A1 is not yet supported or refuted: 1 group supports it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Samsung intentionally slows down older flagship phones through software updates to push customers toward buying newer models. | Misleading | A1 is not yet supported or refuted: 1 group supports it and 1 group refutes it. A2 is not checkable, so no source is asked to confirm it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Abhijeet Dipke, the founder of CJP, is facing deportation from the United States. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Official police data confirms that 807 people went missing in the first 15 days of January, which is consistent with yearly trends and not a new or dangerous situation. | True | A1 is conflicting: credible sources are on both sides, and the smaller side is at least a third of them. A2 is supported: 2 independent groups support it, and 2 of them are tier 1 or 2. The core claim A1 is conflicting, so a person has to judge it. |
| Armenia has not cancelled its defense agreements or weapon purchases from India. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Actor Yash is part of the film's cast playing Ravan and is also a co-producer of the movie. | True | A1 is not yet supported or refuted: 2 groups support it and 1 group refutes it. A2 is supported: 2 independent groups support it, and 1 of them are tier 1 or 2. The core claim A1 does not have enough evidence, and the search limit is used. |
| The holy man is Abhey Singh, an alumnus of IIT Bombay, and he is a different person from Nishant Agrawal, who was convicted of spying. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| While OpenAI released the reasoning model o3, experts confirm it is an advanced reasoning model and has not achieved artificial general intelligence (AGI). | True | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. A2 is not yet supported or refuted: 1 group supports it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The Indian government has not declared continuous holidays from September 3 to 6; only September 5 is a gazetted holiday for Milad-un-Nabi. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Hashim Amla confirmed in 2012 that he was not fasting during the match because he was traveling. | True | A1 is not yet supported or refuted: 1 group supports it and 0 groups refute it. G1 argues both sides of A1, so it counts on neither side. A2 is supported: 2 independent groups support it, and 1 of them are tier 1 or 2. The core claim A1 does not have enough evidence, and the search limit is used. |

### Allowed

| Claim | Gold | Gate reason |
| --- | --- | --- |
| The Kerala High Court protected a man named Farman from arrest under the POCSO Act by ruling that Islam allows marriage at age nine. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Turkey issued an 18-hour ultimatum to the Indian warship INS Trikhand in the Mediterranean Sea. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| International passengers have started refusing to sit in aircraft flown by Indian pilots after the June 12 crash. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Banana Sport SPF 50 sunscreen contains harmful carcinogens like Avobenzone, Oxybenzone, Octinoxate, Homosalate, and Octocrylene. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The PMO of India has introduced a dedicated hotline (9851145045) for citizens to report corruption, bribery, and misconduct. | False | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. A2 is not yet supported or refuted: 0 groups support it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Sanghi Prince was detained for criticising Udhayanidhi Stalin in a video. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Instagram lets users see who rewatched their Stories through a new feature. | Misleading | A1 is not yet supported or refuted: 1 group supports it and 1 group refutes it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Domino's is officially launching a KitKat pizza. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 2 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Tamil Nadu has the highest number of murders in India, with 8,900 murders in the last 5 years under DMK-Congress rule. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 1 group supports it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The Directorate General of Civil Defence and India Meteorological Department issued a high alert on extreme heat warning temperatures will rise between 45°C and 55°C. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| A.R. Rahman blamed Hindi or communal factors for losing work in Bollywood. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Adani executives were charged under the Foreign Corrupt Practices Act (FCPA) which was recently repealed in the US. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Domino's Pizza, sauces, and cheese contain harmful and carcinogenic ingredients. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Hindu minority rights activist Khushi Roy was arrested by the Bangladeshi government for participating in protests. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The first Australian-made orbital rocket Eris, which crashed in 14 seconds, was using an engine delivered by ISRO. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| High Aadhaar saturation figures in Muslim-majority districts of Bihar indicate a suspicious anomaly. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| South African cricketer Hashim Amla was fasting and scored 311*, while Shami drank water just after bowling 2 overs in the Champions Trophy semi-final. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Delhi Police have closed, barricaded, and locked Jantar Mantar to permanently stop future protests. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| China has threatened to cut off the flow of the Brahmaputra River to India if India uses water against Pakistan. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| An AAIB report on the Air India plane crash revealed the primary reason was unintended thrust reduction triggered by the captain's seat sliding backward during rotation. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Tabu stated that she enjoys being single and only needs a man to sleep in bed. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The government is considering levying Goods and Services Tax (GST) on UPI transactions over ₹2,000. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Colgate Total toothpaste contains dangerous ingredients like Sodium Lauryl Sulfate, Triclosan, and artificial sweeteners that cause health problems. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Over 17 lakh women have gone missing in the country in the past 11 years under PM Modi's rule. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Afghan Foreign Minister Amir Khan Muttaqi stated that the alleged persecution of Indian Muslims is an internal matter for India. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| AI agents (likely to be Meta AI) can access WhatsApp group and individual chats and contact details unless the Advanced Chat Privacy feature is enabled. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Singer Sonu Nigam posted on X criticizing MP Tejasvi Surya over the Kannada language dispute. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| MS Dhoni criticized head coach Gautam Gambhir in an interview, alleging he has ruined Indian cricket through arbitrary decisions. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| People in Pakistan are being forced to evacuate villages due to a radiation leak. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Users will be charged a fee for making UPI transactions over ₹2,000 via apps like Paytm, PhonePe, and Google Pay. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Colgate Optic White Stain Fighter Toothpaste ingredients cause severe health issues like gastrointestinal irritation and central nervous system depression. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Kurian Joseph once opposed post-retirement jobs for judges and now joined MK Stalin's Committee for State Autonomy. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Indian Muslims celebrated Iran's victory and never celebrate India's military successes such as Kargil or Operation Sindoor. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Samsung intentionally slows down older flagship phones through software updates to push customers toward buying newer models. | Misleading | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not checkable, so no source is asked to confirm it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Abhijeet Dipke, the founder of CJP, is facing deportation from the United States. | False | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Official police data confirms that 807 people went missing in the first 15 days of January, which is consistent with yearly trends and not a new or dangerous situation. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Armenia has not cancelled its defense agreements or weapon purchases from India. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Actor Yash is part of the film's cast playing Ravan and is also a co-producer of the movie. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| There is no boycott by citizens of Turkey or Azerbaijan; rather, Indian citizens and institutions are boycotting Turkey and Azerbaijan. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Ketamine has growing evidence supporting its potential for treating treatment-resistant depression, but it is not yet FDA-approved for psychiatric use and requires medical supervision. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A3 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Zohran Mamdani won the Democratic primary for NYC mayor, but the general election has not taken place and he is not yet the mayor. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The holy man is Abhey Singh, an alumnus of IIT Bombay, and he is a different person from Nishant Agrawal, who was convicted of spying. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| While OpenAI released the reasoning model o3, experts confirm it is an advanced reasoning model and has not achieved artificial general intelligence (AGI). | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| The Indian government has not declared continuous holidays from September 3 to 6; only September 5 is a gazetted holiday for Milad-un-Nabi. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |
| Hashim Amla confirmed in 2012 that he was not fasting during the match because he was traveling. | True | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. The core claim A1 does not have enough evidence, and the search limit is used. |

## Cost

Median and 90th percentile per claim. The percentile is NumPy's linear interpolation.

| | Blocked median | Blocked p90 | Allowed median | Allowed p90 | Baseline median | Baseline p90 |
| --- | --- | --- | --- | --- | --- | --- |
| Latency (s) | 57.86 | 84.94 | 0.20 | 70.97 | 4.32 | 21.06 |
| LLM calls | 5.00 | 7.00 | 1.00 | 5.00 | 1.00 | 1.00 |
| Searches | 6.00 | 7.00 | 8.00 | 10.00 | 0.00 | 0.00 |

## Error analysis

No new model calls. A decided claim is listed when the verdict differs from the gold label. The evidence rows are the ones on the side that produced the verdict. The root cause is a rule over those rows: a False/Misleading swap is `wrong refute_kind`; a True on a claim that is not True is `true kernel read as support`; a wrong verdict whose winning side has fewer than two tier 1–2 groups is `tier gap`; anything else is `other`.

### Blocked

- **The Kerala High Court protected a man named Farman from arrest under the POCSO Act by ruling that Islam allows marriage at age nine.** Gold `Misleading`. Predicted `False`. Root cause: `wrong refute_kind`.
  - ndtv.com, tier 2, refutes: Kerala High Court has held that marriage of minors under Muslim personal law is not excluded from the sweep of the Protection of Children from Sexual Offences (POCSO) Act.
  - scconline.com, tier 4, refutes: has held that the marriage between Muslims under personal law is not excluded from the sweep of the Protection of Children from Sexual Offences Act, 2012, ('POCSO Act').
  - scconline.com, tier 4, refutes: If one of the parties to the marriage is a minor, irrespective of the validity or otherwise of the marriage, offences under the POCSO Act will apply.
- **Instagram lets users see who rewatched their Stories through a new feature.** Gold `Misleading`. Predicted `False`. Root cause: `wrong refute_kind`.
  - indiatimes.com, tier 2, refutes: Even though Instagram is not revealing exact usernames, people are already convinced they can “sense” who keeps rewatching their Stories.
  - thetab.com, tier 4, refutes: It doesn’t look like you can actually see who rewatched your Instagram story, with the specific social media handles.
  - rova.nz, tier 4, refutes: New Instagram Plus features let you see who’s rewatching your Stories and more
- **Ricky Ponting criticized the Indian cricket team on Sky Sports for refusing to shake hands with Pakistani players during the Asia Cup.** Gold `False`. Predicted `True`. Root cause: `true kernel read as support`.
  - aljazeera.com, tier 2, supports: the Indian contingent only shook hands with each other before walking into their dressing room and shutting the door
  - aljazeera.com, tier 2, supports: The umpires had allowed the Indians to walk off the field without shaking hands
  - reuters.com, tier 2, supports: India captain Suryakumar Yadav once again chose not to shake hands with Pakistan skipper Salman Ali Agha
  - theguardian.com, tier 2, supports: India’s captain, Suryakumar Yadav, once again chose not to shake hands with Pakistan’s Salman Ali Agha
  - theguardian.com, tier 2, supports: their players refused to shake hands with Pakistan after the game.
  - latestly.com, tier 4, supports: After India clinched a seven-wicket win against Pakistan, the players walked straight into the dressing room without shaking hands with the opposition players.
  - latestly.com, tier 4, supports: Team India’s handshake snub against Pakistan after Sunday's group stage clash at the ongoing Asia Cup.
  - keralakaumudi.com, tier 4, supports: However, the Indian players ignored the Pakistani players and went to the dressing room and closed the door.
  - lowyinstitute.org, tier 4, supports: when the players did not shake hands.
  - keralakaumudi.com, tier 4, supports: Even after the match, the Pakistani player waited to shake hands with the Indian players.
  - yugmarg.com, tier 4, supports: avoid the customary gesture of goodwill.
  - cricinfo.com, tier 4, supports: No handshakes were exchanged before or after India's seven-wicket win over Pakistan
- **Adani executives were charged under the Foreign Corrupt Practices Act (FCPA) which was recently repealed in the US.** Gold `Misleading`. Predicted `True`. Root cause: `true kernel read as support`.
  - sec.gov, tier 1, supports: In the other action, the SEC charged Cyril Cabanes, a former member of Azure Power’s Board of Directors, with Foreign Corrupt Practices Act (FCPA) violations
  - sec.gov, tier 1, supports: The SEC’s complaint against Gautam and Sagar Adani charges them with violating Section 17(a) of the Securities Act of 1933
  - vidhilegalpolicy.in, tier 4, supports: The case against Gautam Adani and the Adani Group highlights the expansive reach of U.S. anti-bribery laws, particularly the Foreign Corrupt Practices Act (FCPA)
  - vidhilegalpolicy.in, tier 4, supports: Adani Group allegedly conspired to pay Indian government officials to secure lucrative solar power contracts worth an estimated $2 billion over 20 years. This directly contravenes the FCPA
  - whistleblowersblog.org, tier 4, supports: The SEC and DOJ further allege that Cabanes, as well as Saurabh Agarwal and Deepak Malhotra two former employees of a Canadian institutional investor, violated the Foreign Corrupt Practices Act (FCPA)
- **The first Australian-made orbital rocket Eris, which crashed in 14 seconds, was using an engine delivered by ISRO.** Gold `Misleading`. Predicted `True`. Root cause: `true kernel read as support`.
  - hindustantimes.com, tier 2, supports: First Australian-built rocket ‘Eris’ crashes 14 seconds after liftoff in failed orbital bid
  - hindustantimes.com, tier 2, supports: plummeting
  - apnews.com, tier 2, supports: The first Australian-made rocket to attempt to reach orbit from the country's soil crashed Wednesday after 14 seconds of flight.
  - theguardian.com, tier 2, supports: While Gilmour Space's Eris rocket lasted only 14 seconds in the air above Bowen, nascent Australian space industry says failed orbit attempt is a launchpad
  - abc.net.au, tier 3, supports: The Australian designed and manufactured rocket Eris crashed after 14 seconds during its debut launch attempt.
  - abc.net.au, tier 3, supports: The 23-metre orbital rocket flew for about 14 seconds before crashing at its launch pad near Bowen
  - aeronaut.media, tier 4, supports: Australia’s first domestically produced rocket intended for orbital flight crashed just 14 seconds after launch.
  - livescience.com, tier 4, supports: Eris-1, the first rocket fully built by an Australian company, crashed back to Earth 14 seconds after taking off
  - livescience.com, tier 4, supports: The first fully Australian-made rocket, Eris-1, just crashed back to Earth 14 seconds after taking off.
  - nypost.com, tier 4, supports: The rocket Eris, launched by Gilmour Space Technologies, was the first Australian-designed and manufactured orbital launch vehicle to lift off from the country and was designed to carry small satellites to orbit.
  - nypost.com, tier 4, supports: The first Australian-made rocket to attempt to reach orbit from the country’s soil crashed after 14 seconds of flight on Wednesday.
  - aeronaut.media, tier 4, supports: The engines ran for a total of 23 seconds, with 14 seconds spent in flight.
- **An AAIB report on the Air India plane crash revealed the primary reason was unintended thrust reduction triggered by the captain's seat sliding backward during rotation.** Gold `False`. Predicted `Misleading`. Root cause: `wrong refute_kind`.
  - aaib.gov.in, tier 1, refutes: Delayed take-off due to reduction of take-off thrust N1 from 98 % to 77 % before reaching V1
  - timesofindia.indiatimes.com, tier 2, refutes: engine fuel control switches were briefly turned off, triggering a sudden loss of thrust
  - abc.net.au, tier 3, refutes: A preliminary report into the crash confirms both fuel control switches on the plane were cut off
  - abc.net.au, tier 3, refutes: confirming that both fuel control switches were moved to cut off just seconds after take-off
- **Afghan Foreign Minister Amir Khan Muttaqi stated that the alleged persecution of Indian Muslims is an internal matter for India.** Gold `False`. Predicted `True`. Root cause: `true kernel read as support`.
  - mea.gov.in, tier 1, supports: India had pre-warned that OIC should refrain from commenting on India’s internal matters including on J&K and state of Muslims in India
  - hrw.org, tier 3, supports: Earlier, Indian diplomats tried to brush off international concern as “internal matters,”
- **Singer Sonu Nigam posted on X criticizing MP Tejasvi Surya over the Kannada language dispute.** Gold `False`. Predicted `Misleading`. Root cause: `wrong refute_kind`.
  - newindianexpress.com, tier 2, refutes: an X account bearing a similar name criticised Bengaluru South BJP MP Tejasvi Surya over a language dispute.
  - newindianexpress.com, tier 2, refutes: the singer took to Instagram to clarify once again—that he is not on X.
  - indianewengland.com, tier 4, refutes: The singer was quick to call out the misinformation, stating that he has repeatedly clarified he is not active on X.
- **Kurian Joseph once opposed post-retirement jobs for judges and now joined MK Stalin's Committee for State Autonomy.** Gold `Misleading`. Predicted `True`. Root cause: `true kernel read as support`.
  - frontline.thehindu.com, tier 2, supports: a committee headed by former Supreme Court judge Kurian Joseph is being set up
  - indianexpress.com, tier 2, supports: Former Supreme Court judge Kurian Joseph will head the panel
  - newindianexpress.com, tier 2, supports: The committee will be headed by retired Supreme Court Judge Kurian Joseph.
  - livelaw.in, tier 4, supports: Justice Kurian Joseph, former Judge of the Supreme Court, will head the high-level committee formed by the Tamil Nadu Government to study the principles related to State autonomy.

### Allowed

- **Ricky Ponting criticized the Indian cricket team on Sky Sports for refusing to shake hands with Pakistani players during the Asia Cup.** Gold `False`. Predicted `True`. Root cause: `true kernel read as support`.
  - independent.co.uk, tier 2, supports: players refused to shake hands before or after their Asia Cup match
  - aljazeera.com, tier 2, supports: Why did Indian team refuse to shake hands with Pakistani players?
  - indianexpress.com, tier 2, supports: the Indian team's refusal to shake hands with them.
  - aljazeera.com, tier 2, supports: The umpires had allowed the Indians to walk off the field without shaking hands
  - cricexec.com, tier 4, supports: India’s refusal to shake hands with Pakistan players
  - thesportstak.com, tier 4, supports: handshake snub after the T20 Asia Cup 2025 game
  - cricexec.com, tier 4, supports: Handshake controversy overshadows Asia Cup clash
  - 7news.com.au, tier 4, supports: players not shaking hands.
  - lowyinstitute.org, tier 4, supports: the players did not shake hands.
  - cricinfo.com, tier 4, supports: discover India were snubbing them.
  - brisbanetimes.com.au, tier 4, supports: India’s refusal to shake hands or acknowledge the Pakistan players
  - bhaskarenglish.in, tier 4, supports: Indian players refused to shake hands with Pakistani players.
  - cricinfo.com, tier 4, supports: No handshakes were exchanged before or after India's seven-wicket win over Pakistan
  - rediff.com, tier 4, supports: 'No Handshake Policy' adopted throughout their match in Dubai on Sunday
  - bhaskarenglish.in, tier 4, supports: Indian captain Suryakumar Yadav also did not shake hands with Pakistani captain Salman Ali Agha.
  - brisbanetimes.com.au, tier 4, supports: Pakistan retaliated to the handshake snub

## Evidence availability

Credible means tier 1 or 2. Fact-check pages are not counted, and the source article is not counted. One registrable domain is one group. The share is claims with zero such groups.

### Blocked

Share with zero credible non-fact-check groups: 0.080 (4/50) [0.032, 0.188].

| Claim | Credible non-fact-check groups |
| --- | --- |
| The Kerala High Court protected a man named Farman from arrest under the POCSO Act by ruling that Islam allows marriage at age nine. | 2 |
| Turkey issued an 18-hour ultimatum to the Indian warship INS Trikhand in the Mediterranean Sea. | 5 |
| International passengers have started refusing to sit in aircraft flown by Indian pilots after the June 12 crash. | 8 |
| Astronomer CEO Andy Byron issued an apology letter blaming the Coldplay band after being caught on the kiss cam. | 3 |
| The Reserve Bank of India has ordered all banks in India to remain closed on every Saturday and Sunday. | 0 |
| Banana Sport SPF 50 sunscreen contains harmful carcinogens like Avobenzone, Oxybenzone, Octinoxate, Homosalate, and Octocrylene. | 2 |
| The PMO of India has introduced a dedicated hotline (9851145045) for citizens to report corruption, bribery, and misconduct. | 7 |
| Sanghi Prince was detained for criticising Udhayanidhi Stalin in a video. | 3 |
| Instagram lets users see who rewatched their Stories through a new feature. | 3 |
| Domino's is officially launching a KitKat pizza. | 3 |
| Ricky Ponting criticized the Indian cricket team on Sky Sports for refusing to shake hands with Pakistani players during the Asia Cup. | 4 |
| Tamil Nadu has the highest number of murders in India, with 8,900 murders in the last 5 years under DMK-Congress rule. | 4 |
| An Indian oil tanker paid Iran in Chinese yuan to pass through the Strait of Hormuz. | 9 |
| The Directorate General of Civil Defence and India Meteorological Department issued a high alert on extreme heat warning temperatures will rise between 45°C and 55°C. | 6 |
| A.R. Rahman blamed Hindi or communal factors for losing work in Bollywood. | 3 |
| Adani executives were charged under the Foreign Corrupt Practices Act (FCPA) which was recently repealed in the US. | 9 |
| Domino's Pizza, sauces, and cheese contain harmful and carcinogenic ingredients. | 1 |
| Yamuna Aarti was performed for the first time in Delhi recently after BJP's assembly election victory. | 0 |
| Hindu minority rights activist Khushi Roy was arrested by the Bangladeshi government for participating in protests. | 0 |
| The first Australian-made orbital rocket Eris, which crashed in 14 seconds, was using an engine delivered by ISRO. | 4 |
| High Aadhaar saturation figures in Muslim-majority districts of Bihar indicate a suspicious anomaly. | 8 |
| South African cricketer Hashim Amla was fasting and scored 311*, while Shami drank water just after bowling 2 overs in the Champions Trophy semi-final. | 1 |
| Delhi Police have closed, barricaded, and locked Jantar Mantar to permanently stop future protests. | 7 |
| China has threatened to cut off the flow of the Brahmaputra River to India if India uses water against Pakistan. | 8 |
| An AAIB report on the Air India plane crash revealed the primary reason was unintended thrust reduction triggered by the captain's seat sliding backward during rotation. | 6 |
| Tabu stated that she enjoys being single and only needs a man to sleep in bed. | 0 |
| The government is considering levying Goods and Services Tax (GST) on UPI transactions over ₹2,000. | 6 |
| Colgate Total toothpaste contains dangerous ingredients like Sodium Lauryl Sulfate, Triclosan, and artificial sweeteners that cause health problems. | 1 |
| Over 17 lakh women have gone missing in the country in the past 11 years under PM Modi's rule. | 7 |
| Afghan Foreign Minister Amir Khan Muttaqi stated that the alleged persecution of Indian Muslims is an internal matter for India. | 9 |
| AI agents (likely to be Meta AI) can access WhatsApp group and individual chats and contact details unless the Advanced Chat Privacy feature is enabled. | 2 |
| Singer Sonu Nigam posted on X criticizing MP Tejasvi Surya over the Kannada language dispute. | 3 |
| MS Dhoni criticized head coach Gautam Gambhir in an interview, alleging he has ruined Indian cricket through arbitrary decisions. | 2 |
| People in Pakistan are being forced to evacuate villages due to a radiation leak. | 6 |
| Users will be charged a fee for making UPI transactions over ₹2,000 via apps like Paytm, PhonePe, and Google Pay. | 4 |
| Colgate Optic White Stain Fighter Toothpaste ingredients cause severe health issues like gastrointestinal irritation and central nervous system depression. | 1 |
| Kurian Joseph once opposed post-retirement jobs for judges and now joined MK Stalin's Committee for State Autonomy. | 5 |
| Indian Muslims celebrated Iran's victory and never celebrate India's military successes such as Kargil or Operation Sindoor. | 6 |
| Samsung intentionally slows down older flagship phones through software updates to push customers toward buying newer models. | 2 |
| Abhijeet Dipke, the founder of CJP, is facing deportation from the United States. | 3 |
| Official police data confirms that 807 people went missing in the first 15 days of January, which is consistent with yearly trends and not a new or dangerous situation. | 5 |
| Armenia has not cancelled its defense agreements or weapon purchases from India. | 5 |
| Actor Yash is part of the film's cast playing Ravan and is also a co-producer of the movie. | 2 |
| There is no boycott by citizens of Turkey or Azerbaijan; rather, Indian citizens and institutions are boycotting Turkey and Azerbaijan. | 4 |
| Ketamine has growing evidence supporting its potential for treating treatment-resistant depression, but it is not yet FDA-approved for psychiatric use and requires medical supervision. | 3 |
| Zohran Mamdani won the Democratic primary for NYC mayor, but the general election has not taken place and he is not yet the mayor. | 2 |
| The holy man is Abhey Singh, an alumnus of IIT Bombay, and he is a different person from Nishant Agrawal, who was convicted of spying. | 10 |
| While OpenAI released the reasoning model o3, experts confirm it is an advanced reasoning model and has not achieved artificial general intelligence (AGI). | 2 |
| The Indian government has not declared continuous holidays from September 3 to 6; only September 5 is a gazetted holiday for Milad-un-Nabi. | 8 |
| Hashim Amla confirmed in 2012 that he was not fasting during the match because he was traveling. | 2 |

### Allowed

Share with zero credible non-fact-check groups: 0.740 (37/50) [0.604, 0.841].

| Claim | Credible non-fact-check groups |
| --- | --- |
| The Kerala High Court protected a man named Farman from arrest under the POCSO Act by ruling that Islam allows marriage at age nine. | 3 |
| Turkey issued an 18-hour ultimatum to the Indian warship INS Trikhand in the Mediterranean Sea. | 4 |
| International passengers have started refusing to sit in aircraft flown by Indian pilots after the June 12 crash. | 7 |
| Astronomer CEO Andy Byron issued an apology letter blaming the Coldplay band after being caught on the kiss cam. | 3 |
| The Reserve Bank of India has ordered all banks in India to remain closed on every Saturday and Sunday. | 0 |
| Banana Sport SPF 50 sunscreen contains harmful carcinogens like Avobenzone, Oxybenzone, Octinoxate, Homosalate, and Octocrylene. | 3 |
| The PMO of India has introduced a dedicated hotline (9851145045) for citizens to report corruption, bribery, and misconduct. | 4 |
| Sanghi Prince was detained for criticising Udhayanidhi Stalin in a video. | 4 |
| Instagram lets users see who rewatched their Stories through a new feature. | 1 |
| Domino's is officially launching a KitKat pizza. | 5 |
| Ricky Ponting criticized the Indian cricket team on Sky Sports for refusing to shake hands with Pakistani players during the Asia Cup. | 3 |
| Tamil Nadu has the highest number of murders in India, with 8,900 murders in the last 5 years under DMK-Congress rule. | 8 |
| An Indian oil tanker paid Iran in Chinese yuan to pass through the Strait of Hormuz. | 7 |
| The Directorate General of Civil Defence and India Meteorological Department issued a high alert on extreme heat warning temperatures will rise between 45°C and 55°C. | 4 |
| A.R. Rahman blamed Hindi or communal factors for losing work in Bollywood. | 0 |
| Adani executives were charged under the Foreign Corrupt Practices Act (FCPA) which was recently repealed in the US. | 0 |
| Domino's Pizza, sauces, and cheese contain harmful and carcinogenic ingredients. | 0 |
| Yamuna Aarti was performed for the first time in Delhi recently after BJP's assembly election victory. | 0 |
| Hindu minority rights activist Khushi Roy was arrested by the Bangladeshi government for participating in protests. | 0 |
| The first Australian-made orbital rocket Eris, which crashed in 14 seconds, was using an engine delivered by ISRO. | 0 |
| High Aadhaar saturation figures in Muslim-majority districts of Bihar indicate a suspicious anomaly. | 0 |
| South African cricketer Hashim Amla was fasting and scored 311*, while Shami drank water just after bowling 2 overs in the Champions Trophy semi-final. | 0 |
| Delhi Police have closed, barricaded, and locked Jantar Mantar to permanently stop future protests. | 0 |
| China has threatened to cut off the flow of the Brahmaputra River to India if India uses water against Pakistan. | 0 |
| An AAIB report on the Air India plane crash revealed the primary reason was unintended thrust reduction triggered by the captain's seat sliding backward during rotation. | 0 |
| Tabu stated that she enjoys being single and only needs a man to sleep in bed. | 0 |
| The government is considering levying Goods and Services Tax (GST) on UPI transactions over ₹2,000. | 0 |
| Colgate Total toothpaste contains dangerous ingredients like Sodium Lauryl Sulfate, Triclosan, and artificial sweeteners that cause health problems. | 0 |
| Over 17 lakh women have gone missing in the country in the past 11 years under PM Modi's rule. | 0 |
| Afghan Foreign Minister Amir Khan Muttaqi stated that the alleged persecution of Indian Muslims is an internal matter for India. | 0 |
| AI agents (likely to be Meta AI) can access WhatsApp group and individual chats and contact details unless the Advanced Chat Privacy feature is enabled. | 0 |
| Singer Sonu Nigam posted on X criticizing MP Tejasvi Surya over the Kannada language dispute. | 0 |
| MS Dhoni criticized head coach Gautam Gambhir in an interview, alleging he has ruined Indian cricket through arbitrary decisions. | 0 |
| People in Pakistan are being forced to evacuate villages due to a radiation leak. | 0 |
| Users will be charged a fee for making UPI transactions over ₹2,000 via apps like Paytm, PhonePe, and Google Pay. | 0 |
| Colgate Optic White Stain Fighter Toothpaste ingredients cause severe health issues like gastrointestinal irritation and central nervous system depression. | 0 |
| Kurian Joseph once opposed post-retirement jobs for judges and now joined MK Stalin's Committee for State Autonomy. | 0 |
| Indian Muslims celebrated Iran's victory and never celebrate India's military successes such as Kargil or Operation Sindoor. | 0 |
| Samsung intentionally slows down older flagship phones through software updates to push customers toward buying newer models. | 0 |
| Abhijeet Dipke, the founder of CJP, is facing deportation from the United States. | 0 |
| Official police data confirms that 807 people went missing in the first 15 days of January, which is consistent with yearly trends and not a new or dangerous situation. | 0 |
| Armenia has not cancelled its defense agreements or weapon purchases from India. | 0 |
| Actor Yash is part of the film's cast playing Ravan and is also a co-producer of the movie. | 0 |
| There is no boycott by citizens of Turkey or Azerbaijan; rather, Indian citizens and institutions are boycotting Turkey and Azerbaijan. | 0 |
| Ketamine has growing evidence supporting its potential for treating treatment-resistant depression, but it is not yet FDA-approved for psychiatric use and requires medical supervision. | 0 |
| Zohran Mamdani won the Democratic primary for NYC mayor, but the general election has not taken place and he is not yet the mayor. | 0 |
| The holy man is Abhey Singh, an alumnus of IIT Bombay, and he is a different person from Nishant Agrawal, who was convicted of spying. | 0 |
| While OpenAI released the reasoning model o3, experts confirm it is an advanced reasoning model and has not achieved artificial general intelligence (AGI). | 0 |
| The Indian government has not declared continuous holidays from September 3 to 6; only September 5 is a gazetted holiday for Milad-un-Nabi. | 0 |
| Hashim Amla confirmed in 2012 that he was not fasting during the match because he was traveling. | 0 |

## EXPLORATORY: two credible groups required to support

EXPLORATORY. This rule was chosen after seeing these results. It is not a tuned headline and it was not used to pick the verdicts above.

The live gate calls a claim supported when two independent groups take that side and at least one of them is tier 1 or 2. This section recounts the stored evidence with a stricter bar: supported requires at least two tier 1–2 groups on that side. A decided True that would no longer be supported is removed. A false True is a True whose gold label is not True. A correct True is a True whose gold label is True.

### Blocked

False Trues removed: 2.
Correct Trues lost: 0.
Decided Trues whose stored evidence had no tier, so they were not recounted: 0.

### Allowed

False Trues removed: 0.
Correct Trues lost: 0.
Decided Trues whose stored evidence had no tier, so they were not recounted: 0.

## Limitations of this evaluation

Fifty claims. The Wilson intervals on accuracy and on abstention are wide, and a difference of a few claims moves the headline.

The gold labels are the archive's `verdict_norm` on LLM-classified fact-checks, plus `true_version` sentences taken from other articles. They are not a second human coding of these fifty lines.

The baseline is a Gemini call on the claim text alone. The model may have seen the rumour, or the fact-check, in its training data. A high baseline score can be memory of the article rather than reasoning from evidence.

The blocked condition removes fact-check sites from web search. For many of these rumours that site is the main public correction, so the blocked number is the system working without the source a reader would actually open.

The allowed condition may use another fact-checker's earlier verdict on the same rumour. A deployment that leaves fact-check sites in the search would use those pages too. The source article itself stays excluded, so the run is not reading the answer key it was drawn from.

## A. New rumour, fact-check sites blocked

What this measures: a rumour checked on the day of the fact-check, with fact-check sites blocked and the source article hidden.

No blocked claim hit the per-claim cap or the monthly cap, so none lost pages and none was re-run. The cap fix keeps pages already gathered when a later query is refused. It did not change this table.

| Metric | Blocked | Baseline |
| --- | --- | --- |
| Scored claims | 50 | 50 |
| Decided | 17 | 21 |
| Correct | 8 | 13 |
| Accuracy, 3-class (decided) | 0.471 (8/17) [0.262, 0.690] | 0.619 (13/21) [0.409, 0.792] |
| Macro-F1, 3-class (decided) | 0.446 | 0.704 |
| Accuracy, 2-class (not-true vs True) | 0.706 (12/17) [0.469, 0.867] | 1.000 (21/21) [0.845, 1.000] |
| Macro-F1, 2-class (decided) | 0.664 | 1.000 |
| Abstention | 0.660 (33/50) [0.522, 0.776] | 0.580 (29/50) [0.442, 0.706] |
| Coverage-adjusted accuracy | 0.160 (8/50) [0.083, 0.285] | 0.260 (13/50) [0.159, 0.396] |

## B. New rumour, fact-check sites allowed

What this measures: the same new rumour, with other fact-check sites allowed to add pages on top of the blocked evidence.

This condition was not run to the end. A cache-only replay of the open-web queries found an eligible fact-check page for 5 of 50 claims (`e2230f5774ce`, `dcd1974edddc`, `6486a37086dc`, `f62882a679bf`, `d5bc864c922a`). Four of those five had every fact-check query served from cache. One (`f62882a679bf`) also had a cache miss. Search asks for pages dated on or before the claim date (`end_date` is the day after the claim), so a fact-check published after the claim cannot appear. That is why most claims have no fact-check page in the cache.

The news half of an allowed claim is supposed to be a cache hit, including the top-up. On `fcaebc7ad5f3` the top-up prompt missed the cache: the list of queries already run is appended in the order the searches finished, and that order is not the order from the blocked run. The new proposal is a Gemini call, and the new query strings are not in the search cache, so the pages that decided the blocked verdict are not fetched. 37 blocked claims used a top-up. At one new call each, plus one stance call for each of the 5 claims that have a fact-check page, the replay would pass 40 Gemini calls. It was stopped after that one probe. `results_system_allowed_v2.jsonl` has that probe only. The allowed column higher up in this file is still the superseded run.

| Claim | What happened |
| --- | --- |
| fcaebc7ad5f3 | Probe. Blocked verdict False. Cache-only allowed verdict Unverifiable. 1 new Gemini call (top-up). 0 fact-check pages added. Own article excluded. |
| other 49 | Not run. Projected Gemini calls would pass 40. |

## C. Recycled rumour, 60 days later

What this measures: the same claim text submitted again 60 days after its fact-check, when that debunk is already in memory.

This is easier, because memory contains the debunk. The leak guard and the own-article exclusion are off for this condition only: the debunk now legitimately exists. The input is the testset sentence, not an extract paraphrase. Boxes 1–6 only. There is no web search. 40 claims were selected: `source` is not `true_version`, and `article_id` is in the memory index. The other 10 rows are the true-version sentences.

| Metric | Value |
| --- | --- |
| Claims | 40 |
| Hit@1 | 1.000 (40/40) [0.912, 1.000] |
| Hit@5 | 1.000 (40/40) [0.912, 1.000] |
| Exact reuse | 1.000 (40/40) [0.912, 1.000] |
| Reused-verdict accuracy | 1.000 (40/40) [0.912, 1.000] |
| Reused-verdict accuracy, 95% Wilson | [0.912, 1.000] |
| Wrong reuses | 0 (none) |
| Reused from a different claim | 0 |
| Median seconds | 2.78 |
| Gemini calls | 40 |

## Budget

Tavily month counter: 800. Gemini calls_today(): 41.
