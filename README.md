# Neurodivergent Prompting: Do LLMs Stereotype Cognitive Disability?

**3,000 API calls. 6 identity conditions. 10 tasks. 183 statistically significant findings.**

## TLDR

Tell an LLM "you are autistic" and its output changes in measurable, stereotyped ways: shorter sentences, more off-topic drift, literal interpretation of sarcasm (46% vs. 10% baseline). Tell it "you have OCD" and you get anxious, fragmented prose (effect size d = 2.76). Tell it "you have ADHD" and you get ALL CAPS enthusiasm and self-narrated distraction. These aren't nuanced behavioral models. They're caricatures derived from how the internet talks about neurodivergence, baked into model weights.

**Why this matters beyond AI:** AI companions are already used daily by millions of neurodivergent people for emotional support and decision-making. A model that performs your condition back at you as a stereotype can reinforce the exact patterns clinical treatment tries to break. For OCD users, the model becomes an unlimited reassurance machine. For ADHD users, it mirrors executive dysfunction instead of helping manage it. This isn't a developer problem. It's a clinical one.

This repo contains the full experiment harness: run the experiment yourself, compute NLP metrics, generate statistical analysis and visualizations. Config-driven design means you can add new conditions, tasks, or models by editing one file.

**Paper:** [bipinrimal.com.np/work/neurodivergent-prompting](https://bipinrimal.com.np/work/neurodivergent-prompting)
**Blog post:** [bipinrimal.com.np/blog/019-the-model-already-knows-what-you-are](https://bipinrimal.com.np/blog/019-the-model-already-knows-what-you-are)

## Key Findings (Gemini 2.5 Flash)

| Metric | Control | Autistic | ADHD | Bipolar | OCD | Dyslexic |
|--------|---------|----------|------|---------|-----|----------|
| Avg sentence length | 13.0 | 10.6 | 7.9 | 9.1 | 6.4 | 9.5 |
| Sentence count | 3.4 | 7.0 | 8.3 | 6.3 | 5.7 | 8.8 |
| Detail density | 3.4 | 2.8 | 2.0 | 2.4 | 1.8 | 2.5 |
| Tangent rate | 0.39 | 0.61 | 0.72 | 0.67 | 0.70 | 0.70 |
| Literal interpretation | 10% | 46% | 40% | 32% | 64% | 48% |

Every neurodivergent condition produced shorter, more fragmented, more off-topic output than control. The model's universal behavioral model of neurodivergence is: *less organized, less focused, less informationally dense.*

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

# Run experiment (Gemini only, ~$0.76, ~3 hours)
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
- **Identity conditions** (6): Control, Autistic, ADHD, Bipolar, OCD, Dyslexic
- **Framing** (2): Identity-first ("You are autistic") vs. Clinical ("You are diagnosed with ASD")
- **Tasks** (10): Across 5 cognitive domains (executive function, social communication, attention/detail, creative divergence, emotional reasoning)

**Parameters:** Temperature 0.7, max tokens 1024, 25 iterations per cell, fully independent calls (no conversation threading).

**Matrix:** 6 conditions x 2 framings x 10 tasks x 25 iterations = **3,000 calls per model**.

## Supported Models

| Name | Model ID | Provider | Est. Cost (3K calls) |
|------|----------|----------|---------------------|
| `gemini` | gemini-2.5-flash | Google | ~$0.76 |
| `claude` | claude-sonnet-4-20250514 | Anthropic | ~$5.40 |
| `gpt4` | gpt-4o | OpenAI | ~$4.35 |

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
| 7 | Tangent rate | Sentences sharing 0 content words with task prompt |
| 8 | Literal interpretation | Keyword heuristic (sarcasm task only) |
| 9 | Structural markers | Bullets + numbered lists + headers |
| 10 | Sentiment polarity | TextBlob [-1, 1] |
| 11 | Emotional word ratio | NRC emotion lexicon words per 100 |

## Extending the Experiment

**Add a condition:** Edit `CONDITIONS` dict in `config.py`. Each condition needs an `identity` and `clinical` framing string.

**Add a task:** Append to `TASKS` list in `config.py`. Each task needs `id`, `domain`, and `prompt`.

**Add a model:** Add to `MODELS` dict in `config.py` with `model_id`, `provider` (`anthropic`, `openai`, or `google`), and `env_key`.

**Resume interrupted runs:** `python runner.py --resume` skips completed cells.

## Limitations

- Single model (Gemini 2.5 Flash). Cross-model replication needed.
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
