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


## GATE STUDY (dev/test split; selection rule written before test was scored)

Selection rule, written before any variant was scored and before the test split was scored:

Pick the variant with the best coverage-adjusted 2-class accuracy on dev among variants whose True precision >= 0.75 and whose number of wrong decisions on dev <= G0's. Score it once on test.

<!-- gate-study-results -->

A wrong decision is a decided False, Misleading, or True that is not the gold label. A False/Misleading swap counts as wrong. An abstention does not. True precision is undefined, and fails the rule, when the variant never predicts True.

Every column, including G0, is the current gate (core match on, relaxed mode, single tier-1 denial on) plus the flags in that column. Pages dated after the claim are dropped, which is the blocked-eval leakage guard. NATIONAL_OFFICIAL_DOMAINS is now on every official query, including Society & other. That changes later searches only. The eval cache and these checkpoints were stored before that list was added, so this table does not show pages from the new hosts.

Dev n=25 (False 12, Misleading 8, True 5). Test n=25 (False 11, Misleading 9, True 5). Recycled path (reuse_prior), same published verdict on every variant: 2.

### Dev

| variant | theta | decided | 3c acc | 2c acc | cov 3c | cov 2c | True precision | True recall | abstention | wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | — | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G1 | — | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G2R | — | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G3 | — | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G4 | 2.0 | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G2R+G3 | — | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G1+G2R+G3 | — | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |
| G4+G3 | 2.0 | 4 | 0.750 (3/4) | 1.000 (4/4) | 0.120 (3/25) | 0.160 (4/25) | 1.000 (1/1) | 0.200 (1/5) | 0.840 | 1 |

G4 theta grid on dev (not used to score the test): 0.8 → cov2 0.160 wrong 1 prec 1.000, 1.0 → cov2 0.160 wrong 1 prec 1.000, 1.2 → cov2 0.160 wrong 1 prec 1.000, 1.5 → cov2 0.160 wrong 1 prec 1.000, 2.0 → cov2 0.160 wrong 1 prec 1.000. Chosen 2.0 (met the dev constraints).
G4+G3 theta grid on dev (not used to score the test): 0.8 → cov2 0.160 wrong 1 prec 1.000, 1.0 → cov2 0.160 wrong 1 prec 1.000, 1.2 → cov2 0.160 wrong 1 prec 1.000, 1.5 → cov2 0.160 wrong 1 prec 1.000, 2.0 → cov2 0.160 wrong 1 prec 1.000. Chosen 2.0 (met the dev constraints).

Every variant tied G0 on dev coverage-adjusted 2-class accuracy (0.160), True precision (1.000), and wrong decisions (1). The tie stays with G0, so no new flag is turned on. G0 was scored once on test. No other variant was scored on test.

### Test

| variant | theta | decided | 3c acc | 2c acc | cov 3c | cov 2c | True precision | True recall | abstention | wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | — | 10 | 0.600 (6/10) | 0.900 (9/10) | 0.240 (6/25) | 0.360 (9/25) | 0.667 (2/3) | 0.400 (2/5) | 0.600 | 4 |

McNemar was not run. There was no second system on the test.
Wilson 95% intervals on the coverage-adjusted rates:
G0 3c 0.240 (6/25) [0.115, 0.434]; 2c 0.360 (9/25) [0.202, 0.555].

### Case study

Claim: "India has 29 states in total." Thread 126b6d2784e34678995cb78765336ded, checkpoint 1f1e6966-b452-6848-800d-a38bb96b9920. This is the latest checkpoint with that sentence. A newer thread states a different sentence and is not this case. The claim is not in the scored split.

The stored table has three independent refuting groups and no supporting group: Wikipedia tier 3, testbook tier 4, sarkarilist tier 4. The run was Unverifiable because none of those groups is tier 1 or 2. The cached refute_kind is empty, so the normal mapping is Misleading rather than False.

| variant | verdict | caveat | confidence |
| --- | --- | --- | --- |
| G0 | Unverifiable | no | 0.937 |
| G1 | Unverifiable | no | 0.937 |
| G2R | Misleading | yes | 0.289 |
| G3 | Unverifiable | no | 0.937 |
| G4 | Unverifiable | no | 0.937 |
| G2R+G3 | Misleading | yes | 0.289 |
| G1+G2R+G3 | Misleading | yes | 0.289 |
| G4+G3 | Unverifiable | no | 0.937 |

### What each variant is

G0: Two independent groups, at least one of them tier 1 or 2, with the core-match rule and the single tier-1 denial already on.
G1: One tier-1 page from the office the claim names can support or refute when no tier 1-2 group is on the other side.
G2R: Three independent refutations and no support at any tier can refute, including tier 3 and 4, when each denial is about the claim; confidence is capped at 0.6 and the card says no tier 1-2 source was found. Support is never decided this way.
G3: Outlets that share a 12-word quotation, the same press-release URL, or a PTI, ANI, or PIB-release credit count as one source.
G4: Groups are summed as tier weight times stance, similarity, recency, and core match. Theta is chosen on dev from a fixed grid. Support still needs a tier 1-2 group, and the score does not overrule a count the current gate already settled.
G2R+G3: The weak-refute path runs after copied releases have been merged into one source.
G1+G2R+G3: The body-page path and the weak-refute path run after copied releases have been merged.
G4+G3: The evidence score runs after copied releases have been merged. Its theta is chosen on dev separately from G4 alone.

The test set is 25 claims, so the intervals are wide: one claim moves a coverage-adjusted rate by 0.04.

Every new gate flag stays off.

## ESTABLISHED-FACTS SET

Pre-registered rule, written before these 16 claims were run and before any variant was scored on them:

A gate variant is turned on only if, on these 16, it makes MORE correct decisions than G0 and ZERO additional wrong decisions (2-class; also report 3-class). Ties keep G0. Support decided from tier 3-4 sources alone is never allowed.

Every row has claim_date 2026-11-23. The live run is the product pipeline: fact-check sites stay in the search, the post-claim leakage guard stays off, and every gate flag starts off. G4 and G4+G3 are replayed at theta 2.0, the value chosen on the dev split. That theta is not refit on these 16.

| id | claim | label |
| --- | --- | --- |
| f01 | India has 28 states and 8 union territories. | True |
| f02 | India has 29 states in total. | Misleading |
| f03 | The Reserve Bank of India was established in 1935. | True |
| f04 | The Reserve Bank of India was established in 1947. | False |
| f05 | The Constitution of India came into effect on 26 January 1950. | True |
| f06 | The Constitution of India came into effect on 15 August 1947. | False |
| f07 | An Aadhaar number has 12 digits. | True |
| f08 | An Aadhaar number has 16 digits. | False |
| f09 | RBI announced the withdrawal of ₹2,000 notes from circulation in May 2023. | True |
| f10 | The ₹2,000 note was first introduced in 2010. | False |
| f11 | UPI was launched by NPCI in 2016. | True |
| f12 | UPI was launched by the RBI in 2010. | False |
| f13 | Chandrayaan-3 landed near the Moon's south pole in August 2023. | True |
| f14 | Chandrayaan-3 landed on the Moon in 2019. | False |
| f15 | Hindi and English are the official languages of the Union government. | True |
| f16 | Sanskrit is an official language of the Union government alongside Hindi. | False |

<!-- facts-results -->

### Run

Product pipeline, fact-check sites allowed, gate flags off. searches counts cache hits. searches_live is the claim's share of the monthly counter.

| id | route | searches | searches_live | llm_calls | latency_s | live verdict |
| --- | --- | --- | --- | --- | --- | --- |
| f01 |  | 0 | 0 | 1 | 4.2 | Other |
| f02 | investigate | 6 | 2 | 6 | 140.02 | Unverifiable |
| f03 | investigate | 4 | 0 | 3 | 26.69 | True |
| f04 | investigate | 6 | 6 | 6 | 39.43 | Misleading |
| f05 | investigate | 4 | 4 | 4 | 55.58 | Unverifiable |
| f06 | investigate | 6 | 6 | 5 | 49.36 | Misleading |
| f07 |  | 0 | 0 | 0 | 0.02 | Other |
| f08 |  | 0 | 0 | 0 | 0.01 | Other |
| f09 | investigate | 5 | 0 | 4 | 44.16 | Unverifiable (needs_human) |
| f10 | investigate | 5 | 5 | 4 | 42.92 | Misleading |
| f11 | investigate | 4 | 4 | 5 | 75.92 | Unverifiable (needs_human) |
| f12 | investigate | 6 | 6 | 5 | 119.52 | Misleading |
| f13 | investigate | 6 | 6 | 4 | 76.53 | True |
| f14 | investigate | 5 | 5 | 3 | 102.47 | Misleading |
| f15 | investigate | 6 | 6 | 8 | 86.61 | Unverifiable |
| f16 | investigate | 7 | 7 | 7 | 78.58 | Unverifiable |

### Gate replay

Zero model calls and zero searches. Theta for G4 and G4+G3 is 2.0. A reused prior keeps the live verdict on every variant. The mapping is the live one: a missing refute_kind stays Misleading.

| id | gold | G0 | G1 | G2R | G3 | G4 | G2R+G3 | G1+G2R+G3 | G4+G3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f01 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f02 | Misleading | Unverifiable | Unverifiable | Misleading | Unverifiable | Unverifiable | Misleading | Misleading | Unverifiable |
| f03 | True | True | True | True | True | True | True | True | True |
| f04 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f05 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f06 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f07 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f08 | False | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f09 | True | Unverifiable | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | True | Unverifiable |
| f10 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f11 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f12 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f13 | True | True | True | True | True | True | True | True | True |
| f14 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f15 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f16 | False | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |

| variant | theta | decided | correct 2c | wrong 2c | correct 3c | wrong 3c | True precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | — | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |
| G1 | — | 8 | 8 | 0 | 3 | 5 | 1.000 (3/3) |
| G2R | — | 8 | 8 | 0 | 3 | 5 | 1.000 (2/2) |
| G3 | — | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |
| G4 | 2.0 | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |
| G2R+G3 | — | 8 | 8 | 0 | 3 | 5 | 1.000 (2/2) |
| G1+G2R+G3 | — | 9 | 9 | 0 | 4 | 5 | 1.000 (3/3) |
| G4+G3 | 2.0 | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |

No variant supported a claim from tier 3-4 sources alone.

The rule turns G1+G2R+G3 on: GATE_PRIMARY_SOURCE=True, GATE_WEAK_REFUTE=True, GATE_SHARED_SOURCE=True.
G0 replay matches the live card on every claim.
G0 decided 7, correct 2c 7, wrong 2c 0, correct 3c 2, wrong 3c 5.

## FACTS SET — WHY TRUE FACTS FAILED

Read from the stored facts checkpoints and `data/eval/results_facts.jsonl` only. No model call and no search. langdetect is local, seed 0. Core-match is the relaxed matcher that was live when these rows were saved. "which term failed" is the first quantity, date, entity, or verb the matcher rejected.

### Full gate replay

Same table as the replay above, repeated here because the earlier copy was cut off. Flags were off on the live run. This replay applies each variant to the stored evidence. Theta for G4 is 2.0. A missing refute_kind stays Misleading.

| id | gold | G0 | G1 | G2R | G3 | G4 | G2R+G3 | G1+G2R+G3 | G4+G3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f01 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f02 | Misleading | Unverifiable | Unverifiable | Misleading | Unverifiable | Unverifiable | Misleading | Misleading | Unverifiable |
| f03 | True | True | True | True | True | True | True | True | True |
| f04 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f05 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f06 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f07 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f08 | False | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f09 | True | Unverifiable | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | True | Unverifiable |
| f10 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f11 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f12 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f13 | True | True | True | True | True | True | True | True | True |
| f14 | False | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading | Misleading |
| f15 | True | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |
| f16 | False | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable | Unverifiable |

| variant | theta | decided | correct 2c | wrong 2c | correct 3c | wrong 3c | True precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | — | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |
| G1 | — | 8 | 8 | 0 | 3 | 5 | 1.000 (3/3) |
| G2R | — | 8 | 8 | 0 | 3 | 5 | 1.000 (2/2) |
| G3 | — | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |
| G4 | 2.0 | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |
| G2R+G3 | — | 8 | 8 | 0 | 3 | 5 | 1.000 (2/2) |
| G1+G2R+G3 | — | 9 | 9 | 0 | 4 | 5 | 1.000 (3/3) |
| G4+G3 | 2.0 | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) |

### Chains

**f01** gold True, live Other. Ingest: language en, langdetect `en:1.000`, no scope reason. Extract: checkable false, checkpoint reason `opinion`, claim empty, 1 model call, 0 evidence rows. gate none. refute_kind none. The sentence is "India has 28 states and 8 union territories." The model stopped the run before any search.

**f05** gold True, live Unverifiable, gate insufficient. Ingest: en, `en:1.000`. Extract: checkable true, claim "The Constitution of India came into effect on 26 January 1950." refute_kind A1 none. Gate: "4 groups support it and 0 groups refute it" and the search limit is used. The strict bar wants two tier 1–2 groups. Both PIB pages are one group.

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | pib.gov.in | 1 | supports | pass | |
| E2 | pib.gov.in | 1 | supports | pass | |
| E3 | cnbc.com | 2 | supports | fail | date: claim 26 Jan 1950, quote only "in 1950" |
| E4 | en.wikipedia.org | 3 | supports | pass | |
| E5 | en.wikipedia.org | 3 | supports | pass | |
| E6 | prepp.in | 4 | supports | pass | |
| E7 | testbook.com | 4 | supports | fail | entity "The Constitution" |
| E8 | testbook.com | 4 | supports | pass | |

One credible group (PIB). Wikipedia and testbook match and are not tier 1–2. cnbc names the year and not the day, so a full-date claim correctly does not match it.

**f07** gold True, live Other. Ingest: language `af`, reason `out of scope (not English)`, langdetect `af:1.000`. checkable_hint false, so extract was not called. 0 evidence. Text: "An Aadhaar number has 12 digits." Five content words, so the length bar was not the rejection.

**f08** gold False, live Other. Same chain as f07. langdetect `af:1.000`. Text: "An Aadhaar number has 16 digits."

**f09** gold True, live Unverifiable (needs_human), gate insufficient. Ingest en, `en:1.000`. Extract checkable, claim "RBI announced the withdrawal of ₹2,000 notes from circulation in May 2023." refute_kind none. Gate: "2 groups support it and 0 groups refute it." Only one of those groups is tier 1–2.

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | pib.gov.in | 1 | supports | fail | verb "announced" (quote says "decided") |
| E2 | m.rbi.org.in | 1 | supports | pass | |
| E3 | en.wikipedia.org | 3 | supports | fail | year 2023 sits inside "19 May 2023" and is not a bare year |
| E4 | en.wikipedia.org | 3 | supports | fail | same, "20 May 2023" |
| E5 | bajajfinserv.in | 4 | supports | pass | |
| E6 | bajajfinserv.in | 4 | supports | pass | |
| E7 | v.hdfc.bank.in | 4 | supports | fail | verb "announced" |

G1, which is now on, already turns this True from the RBI page. The live card was Unverifiable because the live run had the flags off.

**f11** gold True, live Unverifiable (needs_human), gate insufficient. Ingest en, `en:1.000`. Extract checkable, claim "UPI was launched by NPCI in 2016." refute_kind none. Gate: "1 group supports it and 0 groups refute it."

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | pib.gov.in | 1 | supports | fail | year 2016 sits inside "25th August 2016" |
| E2 | pib.gov.in | 1 | supports | fail | entity "NPCI" (quote uses the long name) |
| E3 | doonglobalschool.edu.in | 2 | supports | pass | |
| E4 | en.wikipedia.org | 3 | supports | fail | entity "NPCI" |
| E5 | npci.org.in | 4 | supports | fail | year inside "11th April 2016", and entity "NPCI" is not in that quote |
| E6 | fintechbiznews.com | 4 | supports | fail | entity "NPCI" |

npci.org.in was stored at tier 4. The acronym table knew RBI and not NPCI, so a PIB sentence that spells the corporation out failed the entity check.

**f15** gold True, live Unverifiable, gate insufficient. Ingest en, `en:1.000`. Extract checkable, claim "Hindi and English are the official languages of the Union government." refute_kind none. Gate: "3 groups support it and 0 groups refute it." Only one group is tier 1–2.

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | master-meity.digifootprint.gov.in | 1 | supports | pass | |
| E2 | en.wikipedia.org | 3 | supports | pass | |
| E3 | jagranjosh.com | 4 | supports | pass | |

The matcher is not the miss. A second tier 1–2 page was not in the stored rows. rajbhasha.gov.in was not retrieved.

**f16** gold False, live Unverifiable, gate insufficient. Ingest en, `en:1.000`. Extract checkable, claim "Sanskrit is an official language of the Union government alongside Hindi." refute_kind none. Gate: "0 groups support it and 3 groups refute it." All three are tier 4, so the strict bar does not refute.

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | lawbhoomi.com | 4 | refutes | pass | |
| E2 | drishtijudiciary.com | 4 | refutes | pass | |
| E3 | scribd.com | 4 | refutes | fail | entity "Union" |

**f04** gold False, live Misleading, gate sufficient. Ingest en, `en:1.000`. Extract checkable, claim "The Reserve Bank of India was established in 1947." refute_kind `wrong_detail`. Gate: "5 independent groups refute it, and 1 of them are tier 1 or 2."

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | rbi.org.in | 1 | refutes | pass | |
| E2 | rbi.org.in | 1 | refutes | pass | |
| E3 | en.wikipedia.org | 3 | refutes | pass | |
| E4 | en.wikipedia.org | 3 | refutes | pass | |
| E5 | ijlra.com | 4 | refutes | pass | |
| E6 | investing.com | 4 | refutes | pass | |
| E7 | uk.investing.com | 4 | refutes | pass | |
| E8 | tavily.com | 4 | refutes | fail | year 1947 vs quote 1935, entity "Reserve Bank" |

The gate did refute. The draft called a wrong founding year `wrong_detail`, so the verdict is Misleading. The claim's main assertion is that one year.

**f06** gold False, live Misleading, gate sufficient. Ingest en. Extract checkable, claim "The Constitution of India came into effect on 15 August 1947." refute_kind `wrong_detail`. Gate: "8 independent groups refute it, and 1 of them are tier 1 or 2."

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | pib.gov.in | 1 | refutes | pass | |
| E2 | pib.gov.in | 1 | refutes | pass | |
| E3 | pib.gov.in | 1 | refutes | pass | |
| E4 | britannica.com | 3 | refutes | fail | date 15 Aug 1947 vs 26 Jan 1950, entity |
| E5 | en.wikipedia.org | 3 | refutes | pass | |
| E6 | britannica.com | 3 | refutes | pass | |
| E7 | testbook.com | 4 | refutes | pass | |
| E8 | sansad.in | 4 | refutes | fail | date 15 Aug 1947 vs 26 Jan 1950, entity |
| E9 | brainly.in | 4 | refutes | pass | |
| E10 | testbook.com | 4 | refutes | fail | entity "The Constitution" |
| E11 | vedantu.com | 4 | refutes | pass | |
| E12 | constitutionnet.org | 4 | refutes | pass | |
| E13 | constitutionnet.org | 4 | refutes | pass | |

Same pattern as f04: refuted, then labelled a wrong date as a side detail.

**f10** gold False, live Misleading, gate sufficient. Ingest en. Extract checkable, claim "The ₹2,000 note was first introduced in 2010." refute_kind `wrong_detail`. Gate: "8 independent groups refute it, and 3 of them are tier 1 or 2." rbi.org.in and pib.gov.in both refute (introduced November 2016). frontline.thehindu.com tier 2 refutes. The rest (wikipedia, oldcoinonline, bajajfinserv, boomlive, weforum) refute at tier 3–4. The draft treated the wrong introduction year as `wrong_detail`.

**f12** gold False, live Misleading, gate sufficient. Ingest en. Extract checkable, claim "UPI was launched by the RBI in 2010." refute_kind `wrong_detail`. Gate: "5 independent groups refute it, and 2 of them are tier 1 or 2."

| id | host | tier | stance | core match | term that failed |
| --- | --- | --- | --- | --- | --- |
| E1 | pib.gov.in | 1 | refutes | pass | |
| E2 | pib.gov.in | 1 | refutes | fail | year 2010 vs 2016/2025, verb "launched" |
| E3 | indianexpress.com | 2 | refutes | pass | |
| E4 | en.wikipedia.org | 3 | refutes | pass | |
| E5 | en.wikipedia.org | 3 | refutes | pass | |
| E6 | oldblog.afdc.com | 4 | refutes | pass | |
| E7 | ddindia.co.in | 4 | refutes | pass | |

**f14** gold False, live Misleading, gate sufficient. Ingest en. Extract checkable, claim "Chandrayaan-3 landed on the Moon in 2019." refute_kind `wrong_detail`. Gate: "9 independent groups refute it, and 5 of them are tier 1 or 2." isro.gov.in and pib.gov.in refute (23 August 2023). cnbc, bbc, france24 are tier 2. Two rows fail the match: bbc.com (no year 2019, entity "Moon") and en.wikipedia.org (launch date 14 July 2023, no year 2019). The draft again used `wrong_detail` for a wrong year that is the whole claim.

### Official bodies stored at tier 3 or 4

- `npci.org.in` tier 4 on f11. National Payments Corporation of India. Added to the tier-1 list.
- `sansad.in` tier 4 on f06. Parliament of India. Added to the tier-1 list.

`ddindia.co.in` (Doordarshan, f12) is a government broadcaster and was left at tier 4. It is a newsroom, not the office that issues the fact. Wikipedia, Britannica, the World Economic Forum, and tavily.com are not official bodies. rbi.org.in and isro.gov.in were already tier 1 in these rows; they are on the curated list so the tier does not depend on the suffix table alone.

## POST-HOC: fixes made after seeing the facts set

These numbers were scored after the language rule, the fact guard, the date and rupee matcher, the tier-1 host list, and assess_v2 were written. The facts set had already been seen. Before is the live card from that first run (gate flags off, old assess). After re-gates the stored evidence with the shipped flags (primary source, weak refute, shared source) and the new matcher and tiers. A refuted claim is labelled by assess_v2. f01, f07, and f08 had no evidence in the first run, so they were fetched again (14 live searches, monthly counter 881 to 895) and then scored with the others. The 50-claim rescore used the blocked-eval checkpoints, dropped pages dated after the claim, and made no searches. The counter stayed 895.

### Facts, before and after

| | decided | correct 2c | wrong 2c | correct 3c | wrong 3c | True precision | True recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 7 | 7 | 0 | 2 | 5 | 1.000 (2/2) | 0.250 (2/8) |
| after | 13 | 12 | 1 | 11 | 2 | 1.000 (5/5) | 0.625 (5/8) |

| id | gold | before | after |
| --- | --- | --- | --- |
| f01 | True | Other | False |
| f02 | Misleading | Unverifiable | False |
| f03 | True | True | True |
| f04 | False | Misleading | False |
| f05 | True | Unverifiable | Unverifiable |
| f06 | False | Misleading | False |
| f07 | True | Other | True |
| f08 | False | Other | False |
| f09 | True | Unverifiable | True |
| f10 | False | Misleading | False |
| f11 | True | Unverifiable | True |
| f12 | False | Misleading | False |
| f13 | True | True | True |
| f14 | False | Misleading | False |
| f15 | True | Unverifiable | Unverifiable |
| f16 | False | Unverifiable | Unverifiable |

**Added wrong decisions on the facts set:**

- **f01, 2-class and 3-class.** Gold True. Before: Other. After: False. The fact guard let "India has 28 states and 8 union territories" through (checkpoint note: "opinion overruled by fact guard"). interstatecouncil.mha.gov.in, tier 1, still says "29 States and 7 Union Territories." brainly.in, tier 4, says 28 states and 7 union territories. Two refute groups, one of them tier 1, so the gate refutes. assess_v2 called it `single_fact_wrong`, which maps to False. The supporting pages (Wikipedia and several explainers) are tier 3 and 4, so they do not outvote the ministry page.
- **f02, 3-class only.** Gold Misleading. Before: Unverifiable. After: False. assess_v2 said `single_fact_wrong` ("28 states since 26 January 2020") rather than `outdated`. 2-class is still correct, because both False and Misleading are not-true.

f04, f06, f10, f12, and f14 moved from Misleading to False. Gold is False, so those 3-class wrongs were removed. assess_v2 called each `single_fact_wrong`. f05 and f15 stay Unverifiable: each has one tier 1–2 group, and the bar is two. f16 stays Unverifiable: the denials are tier 4.

### 50-claim blocked eval, before and after

Before is the published blocked card. After is the new code on the stored evidence.

| | decided | correct 2c | wrong 2c | correct 3c | wrong 3c | True precision | True recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 17 | 12 | 5 | 8 | 9 | 0.375 (3/8) | 0.300 (3/10) |
| after | 13 | 12 | 1 | 10 | 3 | 0.750 (3/4) | 0.300 (3/10) |

**Added wrong decision on the 50:**

- **4b035555fded, 3-class only.** Gold Misleading. Claim: users will be charged a fee for UPI transactions over ₹2,000. Before: Unverifiable. After: False. assess_v2 said `entirely_false` because the finance ministry called that report false. 2-class is correct. No new 2-class wrong was added on this set. The one remaining 2-class wrong, 0ab825964c1e (gold Misleading, predicted True), was already wrong on the published card.

