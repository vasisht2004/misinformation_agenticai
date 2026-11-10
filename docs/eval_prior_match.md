# Prior-match evaluation

Box 6 decides whether a new claim is the same as a fact-check already in memory. The cosine thresholds propose exact, similar, or none. Dates, amounts, percents, years, numbers, community words, and key terms can still turn an exact match into similar. `TAU_HIGH` is 0.85 and `TAU_LOW` is 0.70, both closed (`>=`). The adopted `KEYTERM_SIM` is 0.45. The tables below were scored at 0.65, before the veto.

## Test sets

`data/eval/prior_match_testset.csv` has 60 claims in four groups.

- 20 paraphrases, seed 42, one memory claim from each domain and then extras. The wording changes and a forward is added. Numbers, dates, and places stay. Picture claims are expected `similar`, because text cannot confirm the file. The four text paraphrases are expected `exact`.
- 10 one-detail edits of those paraphrases, expected `similar`. Three of them change only a date or an amount.
- 10 same-topic claims that are a different allegation, expected `similar`.
- 20 invented claims, written without a similarity filter, expected `none`, marked for a manual check.

`data/eval/tau_high_pairs.csv` has 45 text pairs. Each claim’s nearest other article has a stored cosine in [0.80, 0.95), 15 pairs in each 0.05-wide bin. `same_claim` is a human label: 42 y, 3 n.

`data/eval/tau_high_negatives.csv` has 25 one-edit variants of y pairs, five each for amount, date, person, community, and subject. Claim B is unchanged. Every row is labelled n. The cosine is the original pair’s cosine, so a rule has to catch the edit.

## Retrieval on the 60 paraphrases’ gold articles

Recall is only on the 20 paraphrase rows. A collapsed copy in `also_debunked_by` counts.

- Recall@1: 0.950 (19/20). The miss is the Kangana Ranaut / vote-chori claim: another fact-check of that claim is rank 1, and the gold id is rank 2.
- Recall@5: 1.000.

Confusion matrix at 0.85 / 0.70. Rows are expected exact, similar, none. Columns are predicted, same order.

|  | exact | similar | none |
|---|---:|---:|---:|
| exact | 4 | 0 | 0 |
| similar | 0 | 28 | 8 |
| none | 0 | 0 | 20 |

Macro-F1 is 0.903. The eight misses are related-topic claims whose top cosine is under 0.70. Every novel claim was predicted none. The highest novel cosine was 0.672.

## Threshold comparison on the 70 labelled rows

Positive label is y. Positive prediction is exact, after the detail and key-term rules. The Wilson interval is on precision. These two tables are the stage D3 comparison, before a percent was stored as one value.

Old rules:

| TAU_HIGH | n_exact | precision | Wilson low | Wilson high | recall | false exacts |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 34 | 0.676 | 0.508 | 0.809 | 0.548 | 11 |
| 0.81 | 33 | 0.667 | 0.496 | 0.802 | 0.524 | 11 |
| 0.82 | 33 | 0.667 | 0.496 | 0.802 | 0.524 | 11 |
| 0.83 | 33 | 0.667 | 0.496 | 0.802 | 0.524 | 11 |
| 0.84 | 32 | 0.656 | 0.483 | 0.796 | 0.500 | 11 |
| 0.85 | 31 | 0.645 | 0.469 | 0.789 | 0.476 | 11 |
| 0.86 | 31 | 0.645 | 0.469 | 0.789 | 0.476 | 11 |
| 0.87 | 28 | 0.607 | 0.424 | 0.764 | 0.405 | 11 |
| 0.88 | 20 | 0.600 | 0.387 | 0.781 | 0.286 | 8 |
| 0.89 | 20 | 0.600 | 0.387 | 0.781 | 0.286 | 8 |
| 0.90 | 18 | 0.611 | 0.386 | 0.797 | 0.262 | 7 |
| 0.91 | 15 | 0.600 | 0.357 | 0.802 | 0.214 | 6 |
| 0.92 | 10 | 0.600 | 0.313 | 0.832 | 0.143 | 4 |
| 0.93 | 5 | 0.600 | 0.231 | 0.882 | 0.071 | 2 |
| 0.94 | 4 | 0.500 | 0.150 | 0.850 | 0.048 | 2 |
| 0.95 | 0 |  |  |  | 0.000 | 0 |

New rules, `KEYTERM_SIM` 0.65, before percent-as-one-value:

| TAU_HIGH | n_exact | precision | Wilson low | Wilson high | recall | false exacts |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 14 | 0.857 | 0.601 | 0.960 | 0.286 | 2 |
| 0.81 | 13 | 0.846 | 0.578 | 0.957 | 0.262 | 2 |
| 0.82 | 13 | 0.846 | 0.578 | 0.957 | 0.262 | 2 |
| 0.83 | 13 | 0.846 | 0.578 | 0.957 | 0.262 | 2 |
| 0.84 | 13 | 0.846 | 0.578 | 0.957 | 0.262 | 2 |
| 0.85 | 13 | 0.846 | 0.578 | 0.957 | 0.262 | 2 |
| 0.86 | 13 | 0.846 | 0.578 | 0.957 | 0.262 | 2 |
| 0.87 | 12 | 0.833 | 0.552 | 0.953 | 0.238 | 2 |
| 0.88 | 10 | 0.900 | 0.596 | 0.982 | 0.214 | 1 |
| 0.89 | 10 | 0.900 | 0.596 | 0.982 | 0.214 | 1 |
| 0.90 | 10 | 0.900 | 0.596 | 0.982 | 0.214 | 1 |
| 0.91 | 8 | 0.875 | 0.529 | 0.978 | 0.167 | 1 |
| 0.92 | 4 | 1.000 | 0.510 | 1.000 | 0.095 | 0 |
| 0.93 | 2 | 1.000 | 0.342 | 1.000 | 0.048 | 0 |
| 0.94 | 1 | 1.000 | 0.207 | 1.000 | 0.024 | 0 |
| 0.95 | 0 |  |  |  | 0.000 | 0 |

After `9.2%` is one percent, the new-rules line at 0.85 is: n_exact 11, precision 0.909, Wilson 0.623–0.984, recall 0.238, false exacts 1. The unemployment negative “9.2%” against “2 percent” is no longer exact. The labelled Uttar Pradesh pair, “18% … 2.9%” against “two percent”, is now similar for the same reason.

A grid of `KEYTERM_SIM` from 0.40 to 0.70, at 0.85, peaked at F1 0.720 when the floor was 0.45. At that floor the sentence “hot lemon water kills coronavirus” against “…kills cancer cells” stayed exact, because “cells” is still close to “kills” (cosine 0.62). 0.65 is the highest-F1 floor that blocks that pair.

## Remaining error before the veto

One false exact was left at 0.85 with `KEYTERM_SIM` 0.65. The subject edit “goat urine” against “cow urine” (IIT Madras, Kamakoti) stayed exact: the word cosine of goat and cow is 0.72, above 0.65, and no amount, date, or community rule fired.

## LLM veto

The veto runs only after every rule has left the pair exact. It can turn exact into similar. It cannot turn similar or none into exact. The model was `gemini-3.5-flash-lite`. The response cache was on for the scoring script. If the call raises, the pair is similar and the reason is “The careful same-claim check could not run, so this is checked fresh.”

At `TAU_HIGH` 0.85, with the veto on:

| KEYTERM_SIM | precision | Wilson low | Wilson high | recall | false exacts | calls used |
|---:|---:|---:|---:|---:|---:|---:|
| 0.65 | 1.000 | 0.646 | 1.000 | 0.167 | 0 | 11 |
| 0.45 | 1.000 | 0.832 | 1.000 | 0.452 | 0 | 22 |

Goat against cow is similar under both floors. The 0.65 run vetoed four exacts, three of them labelled y:

- Pair 39. Differing detail: attribution of introduction by the BJP government. “Bio-toilets in Indian railways were first introduced only after 2014 by the BJP government.” against “Bio-toilets in Indian Railways were introduced only after 2014.”
- Pair 40. Differing detail: the specific speed and conditions for treating fever and digestive issues. “IIT Madras Director V. Kamakoti stated that cow urine has medicinal properties and can treat fever and indigestion.” against “IIT Madras Director V. Kamakoti claims cow urine has medicinal properties including the ability to cure fever in 15 minutes and help with digestive issues like IBS.”
- Pair 41. Differing detail: survey versus contest. “Amul is organizing a survey to celebrate its 75th anniversary where users can win cash prizes.” against “Amul is giving out cash prizes through a contest to celebrate its 75th anniversary.”

The adopted floor is 0.45 because that row has the same precision and a higher recall. The same 70 pairs were used to choose KEYTERM_SIM and to measure it, so these numbers are optimistic; the end-to-end eval on 2025-26 claims is the clean test.
