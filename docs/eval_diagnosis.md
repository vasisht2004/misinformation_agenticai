# Blocked-eval diagnosis

Diagnostic analysis on the same 50 claims used for the headline eval. Nothing here is a held-out result.

## How to read this

Every archive `form` on these 50 claims is Text, so there is no image/visual claim type.

manipulation_type and form are empty for the 10 true_version rows. That article_id is the donor fact-check, not a record of the sentence that was labelled True.

search_capped is not on the blocked result rows and not on the checkpoint state, including investigate_stats. The abstention reason capped is not used.

Baseline abstention is 26 by choice + 3 errors, alongside the scorer's 29/50.

model per row not recoverable. The cache JSON has no model field, a fallback answer is stored under the primary model's cache key, and the baseline rows have no usage record.

Cutoff 2026-03-31 is the date supplied on the command line (Gemini 3.5 Flash-Lite model card: knowledge cutoff March 2026).

THREAD AMBIGUOUS: 3.

Article ids: fcaebc7ad5f3 (ambiguous_many), 20eb7385815d (ambiguous_many), 0179258fa67c (ambiguous_many).

Those rows use the latest blocked checkpoint. The match required the same verdict, confidence within 1e-6, and the same evidence id set. Evidence counts for an ambiguous id come from that latest state.

Abstention reasons are first-match, and only on rows the scorer did not decide: error, human_pause, no_hits, only_tier34, single_credible, fabricated_quote, other. fabricated_quote requires investigate_stats.quote_unverified > 0 and claim_type quote. Buckets are 3-class. system_wrong_baseline_abstain is the cell the other bucket names do not cover.

Overlapping Wilson intervals are not a significance test. The McNemar p-value is the test.

## 1. Claim type by bucket, and abstention reasons

| claim_type | both_right | system_only | baseline_only | both_wrong | system_abstain_baseline_right | system_abstain_baseline_wrong | both_abstain | system_right_baseline_abstain | system_wrong_baseline_abstain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quote | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| statistic | 1 | 0 | 0 | 1 | 5 | 2 | 6 | 0 | 0 |
| official_action | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| event | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| other | 1 | 0 | 0 | 3 | 5 | 2 | 11 | 5 | 4 |

Abstention reasons (system rows that were not decided):

| abstention_reason | n |
| --- | --- |
| error | 0 |
| human_pause | 11 |
| no_hits | 11 |
| only_tier34 | 8 |
| single_credible | 3 |
| fabricated_quote | 0 |
| other | 0 |

False True count: 5. Passages are in data/eval/false_trues_passages.md and are not copied here.

## 2. Label distribution and trivial baselines

Coverage is 100% because every trivial policy labels all 50 claims. Coverage-adjusted accuracy is the 3-class correct count over 50. always not-true has no 3-class label.

| label | n |
| --- | --- |
| False | 23 |
| Misleading | 17 |
| True | 10 |

| policy | 3c acc | 2c acc | coverage-adjusted | True precision / recall |
| --- | --- | --- | --- | --- |
| always False | 0.460 (23/50) | 0.800 (40/50) | 0.460 (23/50) | precision n/a; recall 0.000 (0/10) |
| always Misleading | 0.340 (17/50) | 0.800 (40/50) | 0.340 (17/50) | precision n/a; recall 0.000 (0/10) |
| always True | 0.200 (10/50) | 0.200 (10/50) | 0.200 (10/50) | precision 0.200 (10/50); recall 1.000 (10/10) |
| majority (False) | 0.460 (23/50) | 0.800 (40/50) | 0.460 (23/50) | precision n/a; recall 0.000 (0/10) |
| always not-true | n/a | 0.800 (40/50) | n/a (no 3-class label) | precision n/a; recall 0.000 (0/10) |

System True-verdict precision 0.375 (3/8); recall 0.300 (3/10). Baseline True-verdict precision 1.000 (3/3); recall 0.300 (3/10).

## 3. Confusion matrices

Rows are the gold label. Abstain is Unverifiable, needs_human, or a baseline error (verdict missing). A decided False/Misleading swap stays in its 3-class cell and collapses in the 2-class matrix.

### System, 3-class

| gold | False | Misleading | True | Abstain |
| --- | --- | --- | --- | --- |
| False | 4 | 2 | 2 | 15 |
| Misleading | 2 | 1 | 3 | 11 |
| True | 0 | 0 | 3 | 7 |

### Baseline, 3-class

| gold | False | Misleading | True | Abstain |
| --- | --- | --- | --- | --- |
| False | 5 | 3 | 0 | 15 |
| Misleading | 5 | 5 | 0 | 7 |
| True | 0 | 0 | 3 | 7 |

### System, 2-class

| gold | not-true | True | Abstain |
| --- | --- | --- | --- |
| not-true | 9 | 5 | 26 |
| True | 0 | 3 | 7 |

### Baseline, 2-class

| gold | not-true | True | Abstain |
| --- | --- | --- | --- |
| not-true | 18 | 0 | 22 |
| True | 0 | 3 | 7 |

## 4. Paired comparison

On all 50 claims, correct is 1 only when the row was decided and right. Otherwise it is 0. Wilson intervals below are on that 0/1 rate.

### 3-class

| system correct | 1 | 0 |
| --- | --- | --- |
| 1 | 3 | 5 |
| 0 | 10 | 32 |

System 0.160 (8/50) [0.083, 0.285]. Baseline 0.260 (13/50) [0.159, 0.396].

Exact two-sided McNemar p-value: 0.301758.

### 2-class

| system correct | 1 | 0 |
| --- | --- | --- |
| 1 | 5 | 7 |
| 0 | 16 | 22 |

System 0.240 (12/50) [0.143, 0.374]. Baseline 0.420 (21/50) [0.294, 0.558].

Exact two-sided McNemar p-value: 0.093140.

System True-verdict precision 0.375 (3/8); recall 0.300 (3/10). Baseline True-verdict precision 1.000 (3/3); recall 0.300 (3/10).

Overlapping Wilson intervals are not a significance test.

## 5. Where the two systems disagree

3-class. Baseline right and system wrong or abstained:

| claim_id | claim_type | abstention_reason | gate | gate_reasons |
| --- | --- | --- | --- | --- |
| 20eb7385815d | statistic | no_hits | insufficient | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| dcd1974edddc | statistic | human_pause | insufficient | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. / A2 is not yet supported or refuted: 0 groups support it and 1 group refutes it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| bda87aeb59e7 | other | single_credible | insufficient | A1 is not yet supported or refuted: 1 group supports it and 0 groups refute it. / A2 is supported: 2 independent groups support it, and 2 of them are tier 1 or 2. / The core claim A1 does not have enough evidence, and the search limit is used. |
| 0ac969a5ab9e | other | human_pause | insufficient | A1 is not yet supported or refuted: 0 groups support it and 3 groups refute it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| d35b7e6bd9cc | other | only_tier34 | insufficient | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| 80e99600c52c | other | human_pause | insufficient | A1 is not yet supported or refuted: 2 groups support it and 0 groups refute it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| 3d8ea230739d | other | no_hits | insufficient | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. / A2 is not yet supported or refuted: 0 groups support it and 0 groups refute it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| 4b035555fded | statistic | human_pause | insufficient | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| 41edadb7b599 | statistic | only_tier34 | insufficient | A1 is not yet supported or refuted: 0 groups support it and 1 group refutes it. / A2 is not yet supported or refuted: 1 group supports it and 0 groups refute it. / The core claim A1 does not have enough evidence, and the search limit is used. |
| 3d6e8322810e | statistic | no_hits | insufficient | A1 is not yet supported or refuted: 0 groups support it and 0 groups refute it. / The core claim A1 does not have enough evidence, and the search limit is used. |

System right and baseline wrong or abstained:

| claim_id | claim_type | abstention_reason | gate | gate_reasons |
| --- | --- | --- | --- | --- |
| e2230f5774ce | other |  | sufficient | A1 is refuted: 4 independent groups refute it, and 3 of them are tier 1 or 2. / The core claim A1 is refuted, so the evidence is enough to assess. |
| 0d6f0f0e7f40 | other |  |  |  |
| 1de7c1440cc8 | other |  | sufficient | A1 is refuted: 2 independent groups refute it, and 1 of them are tier 1 or 2. / The core claim A1 is refuted, so the evidence is enough to assess. |
| bb23aeaa0cf4 | other |  | sufficient | A1 is supported: 7 independent groups support it, and 3 of them are tier 1 or 2. / The core claim A1 is supported, so the evidence is enough to assess. |
| 0b6220e4a72d | other |  | sufficient | A1 is supported: 6 independent groups support it, and 2 of them are tier 1 or 2. / The core claim A1 is supported, so the evidence is enough to assess. |

## 7. Assisted score

If the system decided, that verdict is used. Otherwise a decided baseline verdict is used and shown as Unverifiable, model lean: that label. Otherwise the assisted verdict is Unverifiable. Baseline errors supply no lean.

| source | n | 3c acc on its decisions | 2c acc on its decisions | coverage-adjusted 3c |
| --- | --- | --- | --- | --- |
| system | 17 | 0.471 (8/17) | 0.706 (12/17) |  |
| baseline | 14 | 0.714 (10/14) | 1.000 (14/14) |  |
| neither | 19 | n/a | n/a |  |
| assisted total | 31 | 0.581 (18/31) | 0.839 (26/31) | 0.360 (18/50) |

## 8. Cutoff split

Split at 2026-03-31.

### claim_date <= 2026-03-31

n = 41.

System decided 14. 3-class 0.500 (7/14). 2-class 0.643 (9/14). Abstention 27/41.

Baseline decided 17. 3-class 0.647 (11/17). 2-class 1.000 (17/17). Abstention 23 by choice + 1 error (scorer 24/41).

### claim_date > 2026-03-31

n = 9. Too small to conclude (n < 10).

System decided 3. 3-class 0.333 (1/3). 2-class 1.000 (3/3). Abstention 6/9.

Baseline decided 4. 3-class 0.500 (2/4). 2-class 1.000 (4/4). Abstention 3 by choice + 2 errors (scorer 5/9).

The fact-check article date's calendar day equals claim_date on all 50 rows, including the donor articles behind true_version sentences. The article-date split is the same partition as the claim_date split.

Baseline 3-class accuracy by quarter of claim_date. Accuracy is correct decided over decided. n is the cell size.

| quarter | n | decided | correct 3c | 3c acc | 2c acc | abstention | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025Q1 | 8 | 3 | 2 | 0.667 (2/3) | 1.000 (3/3) | 5 by choice + 0 errors | n < 10, descriptive only |
| 2025Q2 | 9 | 4 | 2 | 0.500 (2/4) | 1.000 (4/4) | 5 by choice + 0 errors | n < 10, descriptive only |
| 2025Q3 | 16 | 6 | 3 | 0.500 (3/6) | 1.000 (6/6) | 9 by choice + 1 error |  |
| 2025Q4 | 4 | 3 | 3 | 1.000 (3/3) | 1.000 (3/3) | 1 by choice + 0 errors | n < 10, descriptive only |
| 2026Q1 | 4 | 1 | 1 | 1.000 (1/1) | 1.000 (1/1) | 3 by choice + 0 errors | n < 10, descriptive only |
| 2026Q2 | 5 | 1 | 0 | 0.000 (0/1) | 1.000 (1/1) | 2 by choice + 2 errors | n < 10, descriptive only |
| 2026Q3 | 4 | 3 | 2 | 0.667 (2/3) | 1.000 (3/3) | 1 by choice + 0 errors | n < 10, descriptive only |

## POST-HOC: rules tuned after seeing these 50 claims. Not a held-out result.

Fix 2 drops a core support whose quote does not carry the core's key terms, and requires two tier 1-2 supporting groups. Strict uses the full name and the exact verb. Relaxed allows a surname, a stem, or a synonym, and reads the sentence that contains the quote. Numbers and dates stay exact in both. Fix 3 refutes the core on one tier-1 denial when no tier 1-2 group supports it. The assisted column keeps a decided label and otherwise uses the stored baseline jsonl. No new model call and no search.

Recycled path (reuse_prior): 2 claim(s). The gate stays skipped and the verdict is the published one.

| | original | +Fix2 strict | +Fix2 relaxed | +Fix3 | +Fix2 relaxed+Fix3 | assisted (Fix2 relaxed+Fix3 + model lean) |
| --- | --- | --- | --- | --- | --- | --- |
| Decided | 17 | 12 | 13 | 18 | 14 | 28 |
| 3c acc | 0.471 (8/17) | 0.583 (7/12) | 0.615 (8/13) | 0.444 (8/18) | 0.571 (8/14) | 0.607 (17/28) |
| 2c acc | 0.706 (12/17) | 0.917 (11/12) | 0.923 (12/13) | 0.722 (13/18) | 0.929 (13/14) | 0.964 (27/28) |
| Coverage-adjusted 3c | 0.160 (8/50) | 0.140 (7/50) | 0.160 (8/50) | 0.160 (8/50) | 0.160 (8/50) | 0.340 (17/50) |
| Coverage-adjusted 2c | 0.240 (12/50) | 0.220 (11/50) | 0.240 (12/50) | 0.260 (13/50) | 0.260 (13/50) | 0.540 (27/50) |
| Abstention | 0.660 | 0.760 | 0.740 | 0.640 | 0.720 | 0.440 |
| True precision | 0.375 (3/8) | 0.667 (2/3) | 0.750 (3/4) | 0.375 (3/8) | 0.750 (3/4) | 0.833 (5/6) |
| True recall | 0.300 (3/10) | 0.200 (2/10) | 0.300 (3/10) | 0.300 (3/10) | 0.300 (3/10) | 0.500 (5/10) |
| McNemar vs baseline (3c, exact) | p=0.301758 (discordant 5 vs 10) | p=0.179565 (discordant 4 vs 10) | p=0.301758 (discordant 5 vs 10) | p=0.301758 (discordant 5 vs 10) | p=0.301758 (discordant 5 vs 10) | p=0.218750 (discordant 5 vs 1) |

Strict Fix 2 lost 1 correct True (0b6220e4a72d): quotes said 'Mamdani won', not the full name.
False Trues in the published run: 5.
Strict Fix 2 removes 4 of 5 false Trues: 0c8e7b5e3a19, 84a06ffe939b, 3c08b18ed822, 269d3c2615dc.
Relaxed Fix 2 removes 4 of 5 false Trues: 0c8e7b5e3a19, 84a06ffe939b, 3c08b18ed822, 269d3c2615dc.
Fix 3 removes 0 of 5 false Trues: none.
Relaxed Fix 2 + Fix 3 removes 4 of 5 false Trues: 0c8e7b5e3a19, 84a06ffe939b, 3c08b18ed822, 269d3c2615dc.

Newly refuted with no cached refute_kind, scored False (refute_kind_defaulted): 1.
4b035555fded

Changed claims (+Fix2 relaxed+Fix3 against the published verdict):

| id | old -> new | right/wrong |
| --- | --- | --- |
| 0c8e7b5e3a19 | True -> abstain:Unverifiable | abstain |
| 84a06ffe939b | True -> abstain:Unverifiable | abstain |
| 3c08b18ed822 | True -> abstain:Unverifiable | abstain |
| 4b035555fded | abstain:Unverifiable -> False | wrong |
| 269d3c2615dc | True -> abstain:Unverifiable | abstain |
