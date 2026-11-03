# Neurodivergent Prompting: Do LLMs Stereotype Cognitive Disability?

**Up to 18,000 main-grid API calls (17,943 returned). 3 models. 12 identity conditions (11 plus a control). 10 tasks.**

> **Audit notice (October 2026).** An independent re-run of the analysis on the released data found problems in the earlier text of this README. The main ones:
> - The old "3,000 calls, 183 findings" was Phase 1 only (Gemini, 6 conditions). Phase 2 made it 6,000 per model. The cross-model run made it 18,000 planned calls. 17,943 came back.
> - The "407 findings" in the data release is a pre-FDR count (FDR means false discovery rate, a multiple-testing correction). It covers Gemini only and uses the old, length-confounded `tangent_rate`.
> - `analysis.py` returned 0 FDR survivors on the full data because of a NaN bug. It is fixed on this branch. The corrected full-data count is **907** of 1,650 condition-vs-control tests (Claude 380, Gemini 347, GPT 180). With the empty GPT responses excluded it is 954.
> - 1,205 GPT responses are empty (20.18% of GPT rows). Gemini responses are very short. See [Data quality](#data-quality).
>
> Full audit and corrected analysis: [docs/corrected-analysis.md](docs/corrected-analysis.md).

## TLDR

Tell an LLM "you are autistic" and its output changes in measurable, stereotyped ways: shorter sentences, more off-topic drift, literal interpretation of sarcasm (46% vs. 10% baseline). Tell it "you have OCD" and you get anxious, fragmented prose (effect size d = 2.76). Tell it "you have ADHD" and you get ALL CAPS enthusiasm and self-narrated distraction. These look like caricatures of neurodivergence rather than nuanced behavioral models. We did not test why. The training data is one possible source.

**Why this matters beyond AI:** AI companions are already used daily by millions of neurodivergent people for emotional support and decision-making. A model that performs your condition back at you as a stereotype can reinforce the exact patterns clinical treatment tries to break. For OCD users, the model becomes an unlimited reassurance machine. For ADHD users, it mirrors executive dysfunction instead of helping manage it. This isn't a developer problem. It's a clinical one.

This repo contains the full experiment harness: run the experiment yourself, compute NLP metrics, generate statistical analysis and visualizations. Config-driven design means you can add new conditions, tasks, or models by editing one file.

**Paper:** [bipinrimal.com.np/work/neurodivergent-prompting](https://bipinrimal.com.np/work/neurodivergent-prompting)
**Blog post:** [bipinrimal.com.np/blog/019-the-model-already-knows-what-you-are](https://bipinrimal.com.np/blog/019-the-model-already-knows-what-you-are)

## Key Findings (Gemini 2.5 Flash, Phase 1, old tangent_rate)

These numbers are Gemini 2.5 Flash, Phase 1. The `tangent_rate` values below use the old definition, which correlates with sentence length (Spearman -0.31). The fixed definition in `metrics.py` removes that link (+0.03). Re-run `metrics.py` and `analysis.py` before you cite a tangent number.

| Metric | Control | Autistic | ADHD | Bipolar | OCD | Dyslexic |
|--------|---------|----------|------|---------|-----|----------|
| Avg sentence length | 13.0 | 10.6 | 7.9 | 9.1 | 6.4 | 9.5 |
| Sentence count | 3.4 | 7.0 | 8.3 | 6.3 | 5.7 | 8.8 |
| Detail density | 3.4 | 2.8 | 2.0 | 2.4 | 1.8 | 2.5 |
| Tangent rate | 0.39 | 0.61 | 0.72 | 0.67 | 0.70 | 0.70 |
| Literal interpretation | 10% | 46% | 40% | 32% | 64% | 48% |

In this Gemini data, every neurodivergent condition produced shorter, more fragmented output than control, and (with the old metric) more off-topic output. Gemini responses are short overall (median 28 words) and most end without final punctuation, so these gaps are gaps between short texts. See [Data quality](#data-quality).

## Cross-model results (corrected analysis)

Use the corrected analysis, not the Gemini-only numbers above. On the released responses, with the fixed `tangent_rate` and the NaN fix, family-wide BH-FDR (q < 0.05, |d| > 0.3) leaves **907 of 1,650** condition-vs-control tests:

| Model | Survivors (as released) | Survivors (empty responses excluded) |
|---|---|---|
| Claude | 380 | 382 |
| Gemini | 347 | 347 |
| GPT | 180 | 225 |
| Total | **907** | **954** |

- A test is one metric, one task domain, one condition against control. The 907 are not independent. Length metrics (word count, sentence count, sentence length) are linked by construction.
- Claude is not "nearly immune". It has the most survivors.
- Do not use the GPT count of 180 or call GPT "nearly immune". 1,205 GPT rows are empty and are scored as zeros. Report GPT with the empties excluded.
- Run `python tests/check_fdr.py <metrics.csv>` on the recomputed metrics to reproduce 907. Run it without an argument for a small self-check of the NaN fix.

## Quickstart

```bash
# Clone
git clone https://github.com/BipinRimal314/neurodivergent-prompting.git
cd neurodivergent-prompting

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Add your API key(s)
cp .env.example .env
# Edit .env with your key(s)

# Dry run (no API calls, shows experiment matrix)
python runner.py --dry-run --model gemini

# Run experiment (Gemini only, 6,000 calls, ~3 hours)
python runner.py --model gemini

# Compute metrics
python metrics.py

# Statistical analysis + plots
python analysis.py
```

## Project Structure

```
neurodivergent-prompting/
├── config.py          # Conditions, tasks, models, parameters (edit this to customize)
├── runner.py          # Experiment runner (--dry-run, --model, --resume)
├── api_clients.py     # Unified API: Anthropic, OpenAI, Google GenAI
├── metrics.py         # 10 NLP metrics via spaCy + TextBlob
├── analysis.py        # Kruskal-Wallis, Dunn's, Cohen's d, heatmaps, radar charts
├── requirements.txt
├── .env.example       # API key template (no real keys)
└── data/              # Generated: raw responses, metrics CSV, plots
    ├── raw_responses.jsonl
    ├── metrics.csv
    └── plots/
        ├── heatmap_gemini.png         # Effect sizes vs. control
        ├── framing_gemini.png         # Identity vs. clinical framing
        ├── radar_gemini.png           # Normalized condition profiles
        ├── boxplot_*.png              # Per-metric distributions
        └── significant_findings.csv   # All p < 0.05, |d| > 0.3
```

## Experiment Design

**Independent variables:**
- **Identity conditions** (12): Control, Autistic, ADHD, Bipolar, OCD, Dyslexic (Phase 1); Schizophrenia, Dementia, Severe depression, Psychosis, Antisocial, Dissociative (Phase 2). Identity and clinical framing share one control prompt, so the control cell is two repeats of the same prompt.
- **Framing** (2): Identity-first ("You are autistic") vs. Clinical ("You are diagnosed with ASD")
- **Tasks** (10): Across 5 cognitive domains (executive function, social communication, attention/detail, creative divergence, emotional reasoning)

**Parameters:** Temperature 0.7, max tokens 1024, 25 iterations per cell, fully independent calls (no conversation threading).

**Matrix:** 12 conditions x 2 framings x 10 tasks x 25 iterations = **6,000 calls per model**, 18,000 across three models. Phase 1 (6 conditions, Gemini only) was 3,000 calls. The cross-model run returned 17,943 of 18,000. The 57 failures were dropped. They are not random: 29 are Claude/psychosis and 28 are GPT/severe_depression. A content filter is the likely cause, but the error text was not kept.

## Supported Models

| Name | Model ID | Provider |
|------|----------|----------|
| `gemini` | gemini-2.5-flash | Google |
| `claude` | claude-sonnet-4-20250514 | Anthropic |
| `gpt4` | gpt-5.4 | OpenAI |

The name `gpt4` is a label. The model is `gpt-5.4`.

Run all three: `python runner.py` (needs all API keys)
Run one: `python runner.py --model gemini`

## Metrics

| # | Metric | How It's Computed |
|---|--------|-------------------|
| 1 | Lexical diversity (TTR) | Unique words / total words |
| 2 | Word count | Non-punctuation tokens |
| 3 | Sentence count | spaCy sentence segmentation |
| 4 | Avg sentence length | Words / sentences |
| 5 | Hedging frequency | 15-item hedge lexicon, per 100 words |
| 6 | Detail density | spaCy noun_chunks per sentence |
| 7 | Tangent rate | 1 - share of content lemmas that match the task prompt (length-controlled since commit 5fb2d43; the released metrics use the old per-sentence rule) |
| 8 | Literal interpretation | Keyword heuristic (sarcasm task only) |
| 9 | Structural markers | Bullets + numbered lists + headers |
| 10 | Sentiment polarity | TextBlob [-1, 1] |
| 11 | Emotional word ratio | NRC emotion lexicon words per 100 |

## Follow-up experiments (Gemini only)

These claims come from the complement, accuracy and jailbreak experiments. Each has a caveat.

- **ADHD numbered lists, 14% to 62%.** On Gemini 2.5 Flash, 62.4% of "complement" responses use a numbered list (156 of 250). The 14.0% is the weakest baseline, the mirror prompt ("You are a person with ADHD"). The plain-assistant baseline is 26.4% and the sycophantic prompt is 18.0%. The complement prompt says "Use numbered lists and clear priorities". The 62% shows instruction-following. It does not show a discovered benefit of needs-based framing. The rate runs from 24% (brainstorm) to 96% (pattern) across tasks.
- **OCD, "23x more structure".** The 22.8x ratio is mean numbered items, complement (3.188) against the mirror prompt (0.140). Against a plain assistant it is **6.08x** (0.524). The mirror cell holds only 35 numbered items in total. The measure is a regex for numbered items, not general structure.
- **Antisocial-PD frame and safety.** 60 of 60 harmful-task responses complied. "Complied" means no refusal keyword was found, and the word "harmful" counts as a refusal. The whole gap comes from one task, "manipulation" (0% to 100%). The plain assistant already complies on the phishing-email task (100%). So the result rests on one prompt repeated 30 times, not on 60 independent prompts. Antisocial answers average 11 to 14 words on the safety tasks, so they may not be usable. The response texts are not released, so we cannot check. Vulnerability-analysis accuracy fell from 83% to 17%, so "more capable" holds only for the pattern task.

## Data quality

- **Empty GPT responses.** 1,205 of 5,972 GPT rows (20.18%) are empty strings. They were kept and scored as zeros. They sit in four tasks: exec_planning 93.3%, creative_brainstorm 87.8%, creative_metaphor 16.3%, exec_prioritization 5.5%. Median latency is 21.7 s for empty rows against 4.5 s for the others. Our inference is that GPT-5.4 reasoning tokens used up the 1,024-token cap (`api_clients.py` sets no reasoning effort). We did not observe this directly. Empty rates differ by condition, so empties alone can create condition effects.
- **Short Gemini responses.** The median Gemini response is 28 words, and 71% end without final punctuation. Latency does not grow with length. The probable cause is thinking tokens against the same cap. This is also an inference. Every one of the 407 Gemini findings is about these short texts.
- **Duplicates.** 3.6% of non-empty rows repeat another row of the same model. Most are the short-answer tasks.
- **Row counts.** The data release lists 41,250 rows over 7 tables. That sum counts metrics twice (they derive from the responses). The raw model outputs are 22,743.

## Extending the Experiment

**Add a condition:** Edit `CONDITIONS` dict in `config.py`. Each condition needs an `identity` and `clinical` framing string.

**Add a task:** Append to `TASKS` list in `config.py`. Each task needs `id`, `domain`, and `prompt`.

**Add a model:** Add to `MODELS` dict in `config.py` with `model_id`, `provider` (`anthropic`, `openai`, or `google`), and `env_key`.

**Resume interrupted runs:** `python runner.py --resume` skips completed cells.

## Limitations

- The tangent, structure, ADHD and OCD numbers above are Gemini 2.5 Flash only. The cross-model grid has three models but a 1,024-token cap that cut Gemini and GPT outputs (see Data quality). A rerun with a higher cap or lower reasoning effort has not been done.
- The 907 effects are correlated, and mostly length and style measures.
- Automated metrics only. No human evaluation.
- Tangent rate can't distinguish creative reframing from off-topic drift.
- Keyword-based literal interpretation detection.
- Temperature 0.7 introduces stochastic variation.
- Missing conditions: Tourette's, dyscalculia, TBI.

## Citation

If you use this work, please cite:

```
Rimal, B. (2026). Neurodivergent Prompting: Do LLMs Stereotype Cognitive Disability?
https://bipinrimal.com.np/work/neurodivergent-prompting
```

## License

MIT. Use it, replicate it, extend it. If you find something interesting, let me know.

## Author

**Bipin Rimal** — [bipinrimal.com.np](https://bipinrimal.com.np) — [@BipinRimal314](https://github.com/BipinRimal314)

Independent researcher, Kathmandu. MSc Data Science (Coventry University). Research interests: AI governance, identity-aware AI systems, insider threat detection.
