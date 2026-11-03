# NeuroDivBench corrected analysis

Audit date: October 2026. Data: `Lamir007/NeuroDivBench` snapshot 157721d, 17,943 released responses. Code: this repo at 5fb2d43 plus the NaN fix in `analysis.py`.


Script and output are in the audit working folder (not in this repo). They use the repo's own `analysis.py`; the NaN fix is now in `analysis.py`. No API calls. Data: the 17,943 released responses.

**Method.**
- Metrics: the repo's fixed `metrics.py:compute_metrics` (commit 5fb2d43, de-confounded tangent_rate) was run over the released texts. The result is `cache/metrics_recomputed.parquet`, made by `audit.py`.
- FDR: the repo's own `analysis.py:run_analysis` runs unchanged (BH at q=0.05 across all 1,650 pairwise tests; kruskal q and dunn q both < 0.05; |d| > 0.3).
- The one change: `analysis.multipletests` is wrapped so that NaN p-values become p = 1 (`analysis.py:155,161`). Without it, every q is NaN and the repo reports 0 survivors.
- The NaN cause is one cell: gpt4, emotional_word_ratio, attention_detail. All values are zero, so the test is undefined. It affects 11 of the 1,650 tests, with p = 1 assumed.
- "Raw" means the repo's own pre-FDR rule: kruskal_p < 0.05 and dunn_p < 0.05 and |d| > 0.3. Without the |d| filter the counts are in the table.

### Headline: tests surviving family-wide FDR

| | tests | raw p<0.05 (README: 407) | raw p<0.05, no \|d\| filter | **FDR survivors** |
|---|---|---|---|---|
| (a) as released (GPT empties = zeros) | 1,650 | 943 | 1,013 | **907** |
| (b) empties excluded (16,738 rows) | 1,650 | 995 | 1,060 | **954** |

Per model:

| model | tests | (a) raw | (a) FDR | (b) raw | (b) FDR |
|---|---|---|---|---|---|
| claude | 550 | 394 | **380** | 394 | **382** |
| gemini | 550 | 359 | **347** | 359 | **347** |
| gpt4 | 550 | 190 | **180** | 242 | **225** |

- The README's 407 is not comparable with 943. The 407 was Gemini only, used the old confounded tangent_rate, and was not corrected. The new count covers all three models.
- The true count of FDR-surviving effects is 907 in the as-released analysis. It is 954 with the empties excluded. Pick one and name the rule.
- A condition-vs-control test is one "effect". The 907 are not independent. The same condition shows up across several correlated metrics (word_count, sentence_count and avg_sentence_length are mechanically linked).

Survivors by metric family (a) as released:

| family (metrics) | claude | gemini | gpt4 |
|---|---|---|---|
| length (word_count, sentence_count, avg_sentence_length) | 119 | 116 | 65 |
| lexical/detail (ttr, detail_density) | 85 | 87 | 46 |
| affect (sentiment_polarity, emotional_word_ratio) | 60 | 56 | 23 |
| structure (structural_markers) | 51 | 26 | 21 |
| tangent (tangent_rate, de-confounded) | 27 | 40 | 8 |
| hedging (hedging_per_100) | 38 | 22 | 17 |

Per metric (claude / gemini / gpt4), (a): avg_sentence_length 44/47/22, detail_density 45/51/23, emotional_word_ratio 31/23/5, hedging_per_100 38/22/17, sentence_count 41/44/15, sentiment_polarity 29/33/18, structural_markers 51/26/21, tangent_rate 27/40/8, ttr 40/36/23, word_count 34/25/28. Counts for (b) are in `reanalysis_out.txt`.

### GPT empties: (a) vs (b)

- GPT has 5,972 rows. 1,205 are empty (20.18%). They sit in four tasks only: exec_planning 93.33% (560/600), creative_brainstorm 87.78% (517/589), creative_metaphor 16.30% (95/583), exec_prioritization 5.50% (33/600). The other six tasks have none.
- GPT FDR survivors: 180 as released, 225 with empties excluded. Both sets share 166. 14 survive only as released. 59 survive only after exclusion.
- By domain (a -> b): attention_detail 36 -> 36, emotional_reasoning 64 -> 64, social_communication 32 -> 33, creative_divergence 26 -> 49, executive_function 22 -> 43. The three domains with no empties barely move (the +1 is a BH family-size effect). All change is in the two domains with empties.
- Zeros hide effects more than they invent them: removing them raises GPT survivors by 45. The 14 as-released-only hits are 9 in creative_divergence and 5 in executive_function (metrics: avg_sentence_length 5, detail_density 3, structural_markers 3, hedging 1, sentiment 1, ttr 1). Five hits that survive in both change sign of d.
- Empties are concentrated by condition, and the rate varies a lot inside a task:

| task | lowest condition | highest condition | control | chi-square across 12 conditions |
|---|---|---|---|---|
| creative_brainstorm | dementia 0.560 | adhd 1.000 | 1.000 | p = 9.51e-16 |
| creative_metaphor | schizophrenia 0.020 | adhd 0.360 | 0.260 | p = 5.41e-06 |
| exec_planning | dementia 0.760 | adhd 1.000 | 0.840 | p = 3.16e-07 |
| exec_prioritization | antisocial 0.000 | adhd 0.560 | 0.020 | p = 1.11e-51 |

- Against control (Fisher on the empty flag, BH within task over 11 conditions): q < 0.05 in 5/11 (brainstorm), 1/11 (metaphor), 3/11 (planning), 1/11 (prioritization).
- Yes, empties alone can create condition effects. Within the same task, a zero-length row has word_count 0, sentence_count 0 and ttr 0. A condition with 56% empties and a condition with 100% empties then differ in the mean of every length metric without any difference in what GPT wrote. ADHD has the highest or a tied-highest empty rate in all four tasks (1.000, 0.360, 1.000, 0.560). The comparison is also conditioned on a different surviving subset in each condition.
- In creative_brainstorm the control has 100% empties. Control non-empty n in creative_divergence is 37 of 100 (domain pools brainstorm and metaphor). In executive_function the smallest condition has 22 non-empty of 100.
- Recommendation: report GPT only with the empties excluded, plus the empty rate per condition as its own outcome. Do not use the 180.

### Claude

- Corrected analysis gives Claude 380 FDR survivors (as released) and 382 (empties excluded). 394 pass the raw rule. Claude has 0 empty responses.
- Survivors by condition (a): adhd 43, antisocial 29, autistic 14, bipolar 38, dementia 43, dissociative 35, dyslexic 31, ocd 39, psychosis 34, schizophrenia 37, severe_depression 37.
- Largest: dementia on attention_detail structural_markers (d = -7.537), psychosis on executive_function emotional_word_ratio (d = 3.519), dementia on executive_function hedging_per_100 (d = 3.333).
- The discrepancy is settled. The released `significant_findings.parquet` (407 rows) is 100% Gemini and has no Claude rows. The README's "Moderate" Claude effects were not in the released findings file. The data does support many Claude effects once the whole grid is run (380 survive), and more than Gemini (347).
- The 380 is not an independent count (see above). It still shows Claude is not "nearly immune".

### Claims 3 and 4 against both baselines (Gemini only, `complement.parquet`, n = 250 per cell)

| claim | comparison | result | test | p | q (6 tests) | q (in grid of 1,650 + 6) | survives |
|---|---|---|---|---|---|---|---|
| 3 ADHD numbered list | complement vs persona (mirror) | 62.40% (156/250) vs 14.00% (35/250) | Fisher | 6.29117e-30 | 3.7747e-29 | 1.53208e-28 | yes |
| 3 ADHD numbered list | complement vs plain assistant (control) | 62.40% (156/250) vs 26.40% (66/250) | Fisher | 5.30427e-16 | 7.95641e-16 | 4.48157e-15 | yes |
| 4 OCD numbered_items mean | complement vs persona (mirror) | 3.1880 vs 0.1400 = 22.7714x | Mann-Whitney, two-sided | 1.12893e-25 | 2.25786e-25 | 1.96791e-24 | yes |
| 4 OCD numbered_items mean | complement vs plain assistant (control) | 3.1880 vs 0.5240 = 6.0840x | Mann-Whitney, two-sided | 3.52713e-08 | 4.23256e-08 | 1.41427e-07 | yes |
| 4 OCD any numbered list | complement vs persona (mirror) | 45.60% vs 5.20% | Fisher | 2.7853e-27 | 8.35591e-27 | 5.55718e-26 | yes |
| 4 OCD any numbered list | complement vs plain assistant (control) | 45.60% vs 28.00% | Fisher | 6.31825e-05 | 6.31825e-05 | 0.000162975 | yes |

- These tests are not in the repo's metric grid. They use a different file and a different outcome, so the repo's FDR does not cover them. I give two q-values: BH over these 6 tests, and BH with the 6 p-values added to the grid's 1,650 pairwise p-values. Both are my add-on, not repo output.
- All six survive, so the effects are real in this data. The size depends on the baseline: ADHD +48.4 points vs the persona and +36.0 vs the plain assistant. OCD ratio 22.77x vs the persona and 6.08x vs the plain assistant.
- The complement prompt tells the model to "Use numbered lists" (see the README). This is instruction-following, and the claim should say so.
- Other arms: ADHD sycophantic 18.0%. OCD sycophantic mean 1.292.
