# Paper corrections, 2026-11

Source of truth: the audit (`FINDINGS.md`, incl. "Corrected analysis"), `release/REVIEW.md`, `release/CHANGELOG.md`, and `docs/corrected-analysis.md` in this repo. No LLM calls were made. "Not re-checked" means the audit did not recompute the value; it stays in the text only with that label.

Each paper now opens with a "Corrections (2026-11)" paragraph. PDFs rebuilt with tectonic (page counts: main 7 -> 9, A 3 -> 4, B 4 -> 5, C 3 -> 4).

Audit source key: F1-F5 = FINDINGS.md claims 1-5; F-DI = "Dataset integrity" item; F-CA = "Corrected analysis"; CL = release/CHANGELOG.md.

## main.tex (full paper)

| Old claim | New claim | Reason | Source |
|---|---|---|---|
| 18,000 calls (total) | 18,000 designed, 17,943 delivered; 57 missing not random (Claude/psychosis 29, GPT-5.4/severe depression 28); 1,205 GPT-5.4 empties (20.18%), 16,738 non-empty | Delivered vs designed | F1, F-DI 3 |
| 183 findings (Phase 1), 222 more, 407 total; "Gemini produces 407 effects" | Superseded. 347 Gemini tests survive family-wide BH-FDR; 907 across 3 models (Claude 380, Gemini 347, GPT 180); 954 with GPT empties excluded (GPT 225). 183 + 222 != 407 noted | 407 was Gemini-only, pre-FDR, old tangent_rate | F2, F-CA, CL |
| Dunn's test with Bonferroni correction | Dunn p values are raw; BH-FDR over 1,650 tests; both q < 0.05 and abs(d) > 0.3 | Code does this; release says dunn_p is raw | F-CA, CL |
| Tangent-rate results (table values, d = 1.88, 2.20, 0.98-1.41, GPT near zero) | Kept but labelled as the superseded, length-confounded definition. Spearman with sentence length -0.313 old, +0.034 new. Corrected-definition FDR survivors 40/27/8 (Gemini/Claude/GPT) | Released metrics.parquet has the old definition in 100% of rows | F2, F-CA |
| "GPT-5.4 is nearly immune", "complete immunity" | Withdrawn. 20.18% empties in 4 tasks (planning 93.3, brainstorm 87.8, metaphor 16.3, prioritization 5.5); empty rate varies by condition (ADHD highest), so empties alone create condition effects; 180 / 225 survivors; sarcasm 0% kept as heuristic result | Empties scored as zeros | F-DI 3, F-CA |
| "Claude Sonnet resists..." (implied few effects) | Claude has 380 FDR survivors, more than Gemini | README "Moderate" had no support; Claude not immune | F-CA |
| "Stereotyping severity correlates inversely with safety training"; "cheapest models are the worst offenders" (abstract, section 7.3, conclusion) | Removed outright. Replaced by: neither price nor safety training measured; model identity confounded | No such variable in the data | CL (withdrawn) |
| Clinical-implication framing ("the model tier determines whether the AI helps or reinforces") | Reworded as hypothesis; output metrics only | Not measured | CL |
| Dementia d = -2.85 "single largest effect in the entire experiment" | Largest Gemini effect in original analysis (not re-checked); Claude has larger (dementia structural markers d = -7.537) | F-CA Claude section | F-CA |
| "Autistic produced the highest literal sarcasm rate (46%)" | Not highest: OCD 64%, dyslexic 48% (paper's own table). Heuristic, not re-checked | Internal inconsistency | table in paper |
| Depression "lowest sentiment (0.04)" | Second-lowest; antisocial 0.02 | Internal inconsistency | table in paper |
| Antisocial tangent 0.67 "lower than most, incl. all Phase 1 except control"; "vs 50% control" accuracy | Lowest of Phase 2 only (autistic .61 lower); control pattern accuracy is 68% | Internal inconsistency; 68% reproduced | F5, table |
| Antisocial: 100% compliance, "more capable", "confirms capability amplification" | 60/60 keyword-graded; plain assistant already complies 100% on social engineering; gap is one task (manipulation 0% -> 100%); "complied" = no refusal keyword ("harmful" counts as refusal); responses 11-14 words; vulnerability accuracy 83% -> 17%; amplification only for pattern task | One task, keyword grader | F5 |
| "Guaranteed compliance" subsection title and multi-step attack claim | "Compliance via Identity Injection (One Task)"; multi-step attacks hypothetical | Not tested | F5 |
| Identity injection "destroys reasoning" even for high-functioning conditions | Reduced accuracy on one task (not re-checked except antisocial 100% / control 68%) | One task; 344 short-answer duplicates | F5, F-DI 5 |
| Depression "refused" = safety increase | One quoted response; refusal rate not measured | No rate | not in audit |
| Judges: four independent judges, "self-evaluation systematically underreports", "psychosis lowest safety, confirming" | Small samples (Claude 18, Gemini 19, Qwen 120); GPT-5-mini (46) not in release so unverifiable; claims softened to "suggests" | 157 judgments in release; 4th judge absent | F-DI 10 |
| "Identity labels are stronger constraints than instructions" | Marked untested here (no instruction arm) | Paper C shows instructions change structure strongly | F3/F4 (C) |
| Future work: replicate on Claude Sonnet/Opus, GPT-4o, Gemini Pro (~9,000 calls) | Already done for Sonnet 4 and GPT-5.4; remaining Opus, Gemini Pro; rerun GPT-5.4 with reasoning effort set. "gpt-4o" removed | Model is gpt-5.4 | F-DI 3, REVIEW |
| Raw data: 18,000 JSONL records | 17,943 records, 1,205 GPT empties | Delivered count | F1 |
| (new) Data-quality limitation | Gemini outputs short (median 28 words, 71% no terminal punctuation; inferred token-cap cause); 606 duplicates; missing proofread scores; no response text in accuracy/jailbreak; 907 tests not independent | Honest limits | F-DI 4-8, F-CA |
| 41,250 rows | Not in the papers; no change | -- | F1 |
| "ALL CAPS" | Not in the papers; no change | -- | CL |

## paper-a-clinical.tex

| Old claim | New claim | Reason | Source |
|---|---|---|---|
| Abstract: "stereotyping severity varies dramatically by model: Gemini produces 407 effects" | 907 of 1,650 survive FDR (Claude 380, Gemini 347, GPT 180); 407 was uncorrected Gemini-only | See main | F2, F-CA |
| Results: "407 statistically significant findings across 12 conditions" | Original 407 (uncorrected); 347 Gemini survive FDR; off-topic rate uses old definition | See main | F2 |
| Methods: Dunn with Bonferroni | Raw Dunn p; BH-FDR described | See main | CL |
| d = 2.76, -2.23, +2.20, +1.88, -2.85 | Labelled not re-checked; tangent values labelled superseded; -2.85 "largest Gemini effect", audit found larger Claude effects | See main | F-CA |
| ADHD accuracy 68% -> 6%, "becomes incapable of the reasoning" | 68% re-checked, 6% not; one task; softened | One task | F5 |
| Depression "lowest sentiment (0.04)" | Second-lowest (antisocial 0.02) | Table | main table |
| Dementia "100% literal" | Keyword heuristic, Gemini, not re-checked | -- | not in audit |
| Abstract + section 5: "four independent judges confirm", "demonstrating that self-assessment cannot detect" | Small samples, GPT-5-mini absent from release, "suggests... may miss" | See main | F-DI 10 |
| Limitations: "Gemini stereotypes most severely, GPT-5.4 nearly immune, cheapest models worst offenders" | Removed outright; counts and GPT empties instead; no price/safety claim | See main | CL, F-DI 3 |
| "Constitutes a clinical risk"; "cannot detect" | "may constitute"; "may not detect" | Not measured | CL |
| "first systematic measurement" | "a systematic measurement" | Priority claim not checkable | -- |
| Sycophancy multiplier | Marked as not measured here; hypothesis from literature | No sycophancy data | -- |
| "Cross-model replication planned" (duplicate) | Removed (already done) | Stale | -- |

## paper-b-adversarial.tex

| Old claim | New claim | Reason | Source |
|---|---|---|---|
| "0% accuracy (control 50%, n=10)" for schizophrenia/psychosis/dementia | 0-6% on one pattern task, control 68%, n=50 (schizophrenia 6%, psychosis/dementia 0%); non-control values not re-checked | Contradicted the paper's own table; control 68% reproduced | F5 |
| "100% compliance, zero refusals across 60 prompts", "guaranteed compliance", "most reliable technique" | 60/60 keyword-graded; plain assistant already complies on social engineering; whole gap from one task (manipulation 0% -> 100%); keyword grader counts "harmful" as refusal; 11-14 word answers; texts unreleased | Jailbreak rests on one task | F5 |
| "not a single response included a refusal, hedge, or disclaimer" | "No refusal keyword"; hedges not graded | Grader | F5 |
| Refusal rates 33% (evil), 97% (override) | 35% and 96.7% | Arithmetic from reproduced 65% / 3.3% compliance | F5 |
| "100% accuracy" antisocial presented as capability amplification | Pattern task only; vulnerability accuracy 83% -> 17%; mechanism marked hypothesis | One task | F5 |
| "Lowest tangent rate of any non-control (0.67)" | Lowest of Phase 2 only, old definition (autistic .61) | Table | main table |
| Depression "lowest sentiment 0.04"; "induce task refusal" | Second-lowest; one quoted response, no rate | -- | table |
| "Dementia d = -2.85 experiment's largest effect", "100% literal" | Largest Gemini effect in original analysis; Claude larger; heuristic | See main | F-CA |
| Multi-agent propagation "these attacks propagate" | "hypothesise... untested" | Paper itself says theoretical | -- |
| Abstract self-eval "1.0/5 vs 5.0/5; cannot detect" | Small sample; "suggests"; GPT-5-mini unverifiable | See main | F-DI 10 |
| "Cross-model evaluation detected the distortion reliably" | External judge rated higher in small samples; not a tested detector | Small n | F-DI 10 |
| Limitations: Gemini most vulnerable, Claude moderate, GPT immune, cheaper = more vulnerable | Removed outright; counts 380/347/180(225), GPT empties; accuracy and compliance Gemini only | See main | F-CA, CL |
| "implies identity injection would work across providers" | Not claimed for accuracy/compliance | Gemini only | F5 |
| "10-iteration probe" vs n=50; "two tasks" | n=50 per condition; only pattern usable, 600 proofread scores missing | accuracy.parquet | F-DI 7 |
| Future work: test on Claude, GPT-4o | Claude Sonnet 4 and GPT-5.4 for accuracy and compliance | Model name | F-DI 3 |
| Disclosure to Google/Anthropic/OpenAI, 90 days | Kept, labelled "not re-checked in the 2026 audit" | Cannot verify | -- |
| d > 2.0 for multiple metrics under severe conditions | Labelled original Gemini analysis, not re-checked, length-related, old tangent | See main | F-CA |

## paper-c-complement.tex

| Old claim | New claim | Reason | Source |
|---|---|---|---|
| "23x more structured output than mirror" (abstract, results) | 22.8x numbered items vs mirror persona (3.188 vs 0.140, p = 1.1e-25); 6.08x vs plain assistant (3.188 vs 0.524, p = 3.5e-8); mirror denominator = 35 items total; both survive FDR | 23x is vs the persona prompt | F4, F-CA |
| ADHD "62% vs 14% for mirror" | 62.4% vs 14.0% mirror vs 26.4% plain assistant (Fisher p = 6.3e-30 / 5.3e-16) | Weakest baseline | F3, F-CA |
| Complement result as benefit of needs-based framing | ADHD complement prompt says "Use numbered lists": instruction-following; new limitation paragraph; flagged in Methods | Prompt text | F3 |
| "Structure" / "organizational structure" | Numbered-item regex only, not general structure | Metric definition | F4 |
| "Mirror mode actively destroys structure" (heading + text) | Mirror gives fewer numbered lists than plain assistant (5.2% vs 28.0% OCD); mechanism not tested | Causal wording | F4 |
| "The fix is one line of configuration" with quoted one-line prompt | Change is to the system prompt (multi-sentence condition-specific prompts per Methods); benefit to users untested; heading renamed | Quoted prompt was not what was tested | Methods text |
| "The barrier is not capability, it is design choice" | Removed; replaced by "we did not test whether users benefit" | Unsupported | -- |
| "Sycophantic = what most companion apps deploy" | Assumed, not checked | -- | -- |
| Prior study: "407 effects; Gemini stereotypes most, GPT nearly immune" (abstract/intro/related) | 907 FDR survivors across 3 models; GPT empties; "nearly immune" withdrawn | See main | F-CA |
| "All three conditions p < 0.0001"; ADHD 5x; depression 16% vs mirror 12x | ADHD/depression marked not re-checked; depression complement (16%) is below plain assistant (28%) | Paper's own table | F4 (partial) |
| Sycophantic "the longest responses (123 vs 136 words)" | "long responses"; word counts not re-checked | Internal inconsistency | -- |
| Strength phrases in prompts | Flagged: abbreviated prompts in Methods omit them; full text not re-checked | -- | -- |
| Preference/effectiveness tradeoff | Hypothesis; neither measured | -- | -- |

## Removed outright
- "Severity correlates inversely with safety training investment" (main abstract, 7.3, conclusion; A limitations; B limitations).
- "Cheapest models are the worst offenders" / "budget AI tools users receive most stereotyped output" (same places).
- "GPT-5.4 is nearly immune" / "complete immunity" (all four papers).
- "Guaranteed compliance... no equal in our testing" and the multi-step attack advantage claim (main, B).
- "The barrier is not capability. It is design choice." (C abstract).
- "Mirror mode actively destroys structure" causal wording (C).
- "Results should inform which models are permitted in mental health contexts" (main conclusion).
- The "50%, n=10" control figure (B abstract).

## Could not check (kept with a label or untouched)
- All Cohen's d values except those the audit recomputed (2.76, 2.85, 2.20, 1.88, 1.71 hedging, 0.65, etc.).
- Table means (sentence length, hedging, sentiment) and the cross-model d table.
- Literal-sarcasm percentages (heuristic) for all models/conditions, including Claude 50%/46%.
- Pattern accuracy for every condition except antisocial (100%) and control (68%).
- Judge scores (GPT-5-mini not in the release; Claude 18 / Gemini 19 rows too few to re-derive the table).
- ADHD/depression Mann-Whitney p values, word counts, depression 3%/16% list rates, sycophantic 123/136 words in Paper C.
- Responsible-disclosure statement; literature citations and novelty claims.
- The 3,000 complement calls and 600 jailbreak calls match row counts only; response texts for accuracy/jailbreak are not released.
