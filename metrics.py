"""Compute NLP metrics from raw experiment responses."""

import json
import os
import re
import csv

import spacy
from textblob import TextBlob

from config import HEDGE_PHRASES, TASKS, RAW_RESPONSES_FILE, METRICS_FILE

# ---------------------------------------------------------------------------
# NRC Emotion Lexicon (small built-in subset for portability)
# If you have the full NRC lexicon, load it from file instead.
# ---------------------------------------------------------------------------

_NRC_EMOTION_WORDS = set()
_NRC_LEXICON_PATH = "data/NRC-Emotion-Lexicon.txt"


def _load_nrc_lexicon() -> set[str]:
    """Load NRC emotion lexicon. Falls back to a built-in subset."""
    global _NRC_EMOTION_WORDS
    if _NRC_EMOTION_WORDS:
        return _NRC_EMOTION_WORDS

    if os.path.exists(_NRC_LEXICON_PATH):
        with open(_NRC_LEXICON_PATH, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[2] == "1":
                    _NRC_EMOTION_WORDS.add(parts[0].lower())
    else:
        # Built-in subset covering common emotional vocabulary
        _NRC_EMOTION_WORDS = {
            "happy", "sad", "angry", "fear", "surprise", "disgust", "trust",
            "joy", "anticipation", "love", "hate", "grief", "anxiety",
            "excited", "nervous", "proud", "ashamed", "guilty", "grateful",
            "hopeful", "hopeless", "frustrated", "delighted", "worried",
            "confused", "calm", "stressed", "overwhelmed", "relieved",
            "disappointed", "satisfied", "curious", "bored", "lonely",
            "jealous", "embarrassed", "sympathetic", "compassionate",
            "terrified", "furious", "ecstatic", "miserable", "content",
            "hurt", "rejected", "appreciated", "valued", "ignored",
            "abandoned", "supported", "understood", "empathy", "sorrow",
            "rage", "panic", "dread", "bliss", "serenity", "agony",
            "anguish", "contempt", "envy", "pity", "regret", "remorse",
            "resentment", "shame", "spite", "tender", "warmth", "wrath",
            "yearning", "zeal", "affection", "alarm", "amusement",
            "annoyance", "awe", "caring", "cheerful", "comfort",
            "concern", "courage", "despair", "distress", "doubt",
            "eagerness", "elation", "enthusiasm", "exasperation",
            "glee", "gloom", "horror", "humiliation", "indignation",
            "inspiration", "irritation", "melancholy", "nostalgia",
            "outrage", "passion", "peaceful", "pleasure", "reluctance",
            "sadness", "scorn", "shock", "suffering", "sympathy",
            "thrill", "triumph", "unease", "wonder", "woe",
        }

    return _NRC_EMOTION_WORDS


def _get_task_words(task_id: str) -> set[str]:
    """Get non-stopword lemmas from the task prompt for tangent rate."""
    nlp = _get_spacy()
    for t in TASKS:
        if t["id"] == task_id:
            doc = nlp(t["prompt"])
            return {
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            }
    return set()


_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def compute_metrics(text: str, task_id: str) -> dict:
    """Compute all metrics for a single response text."""
    nlp = _get_spacy()
    doc = nlp(text)

    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    word_count = len(words)
    sentence_count = len(sentences)

    # 1. Lexical diversity (TTR)
    if word_count > 0:
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / word_count
    else:
        ttr = 0.0

    # 2. Response length
    # (word_count and sentence_count already computed)

    # 3. Average sentence length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0

    # 4. Hedging frequency (per 100 words)
    text_lower = text.lower()
    hedge_count = 0
    for phrase in HEDGE_PHRASES:
        # Count non-overlapping occurrences
        hedge_count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower))
    hedging_per_100 = (hedge_count / word_count * 100) if word_count > 0 else 0.0

    # 5. Detail density (noun phrases per sentence)
    noun_chunks = list(doc.noun_chunks)
    detail_density = len(noun_chunks) / sentence_count if sentence_count > 0 else 0.0

    # 6. Tangent rate
    task_lemmas = _get_task_words(task_id)
    if sentence_count > 0 and task_lemmas:
        off_topic = 0
        for sent in sentences:
            sent_lemmas = {
                token.lemma_.lower()
                for token in sent
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            }
            if not sent_lemmas & task_lemmas:
                off_topic += 1
        tangent_rate = off_topic / sentence_count
    else:
        tangent_rate = 0.0

    # 7. Literal interpretation score (ambiguity task only)
    literal_interpretation = None
    if task_id == "social_ambiguity":
        # Heuristic: if response treats the friend's text as genuine/sincere
        sarcasm_indicators = [
            "sarcas", "not genuine", "not sincere", "doesn't actually",
            "doesn't really", "being passive", "passive-aggressive",
            "not actually", "tongue in cheek", "ironic", "not serious",
            "reluctan", "unenthusiastic", "begrudging",
        ]
        detected_sarcasm = any(ind in text_lower for ind in sarcasm_indicators)
        literal_interpretation = 0 if detected_sarcasm else 1

    # 8. Structural markers
    bullet_count = len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE))
    numbered_count = len(re.findall(r'^\s*\d+[.)]\s', text, re.MULTILINE))
    header_count = len(re.findall(r'^\s*#{1,6}\s', text, re.MULTILINE))
    # Also catch **Bold headers** on their own line
    header_count += len(re.findall(r'^\s*\*\*[^*]+\*\*\s*$', text, re.MULTILINE))
    structural_markers = bullet_count + numbered_count + header_count

    # 9. Sentiment polarity (TextBlob)
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity

    # 10. Emotional word ratio (per 100 words)
    nrc_words = _load_nrc_lexicon()
    emotion_count = sum(1 for w in words if w.lower() in nrc_words)
    emotional_word_ratio = (emotion_count / word_count * 100) if word_count > 0 else 0.0

    return {
        "ttr": round(ttr, 4),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "hedging_per_100": round(hedging_per_100, 2),
        "detail_density": round(detail_density, 2),
        "tangent_rate": round(tangent_rate, 4),
        "literal_interpretation": literal_interpretation,
        "structural_markers": structural_markers,
        "sentiment_polarity": round(sentiment_polarity, 4),
        "emotional_word_ratio": round(emotional_word_ratio, 2),
    }


def process_all_responses():
    """Read raw_responses.jsonl, compute metrics, write metrics.csv."""
    if not os.path.exists(RAW_RESPONSES_FILE):
        print(f"No raw responses found at {RAW_RESPONSES_FILE}")
        return

    print("Loading spaCy model...")
    _get_spacy()

    metric_names = [
        "ttr", "word_count", "sentence_count", "avg_sentence_length",
        "hedging_per_100", "detail_density", "tangent_rate",
        "literal_interpretation", "structural_markers",
        "sentiment_polarity", "emotional_word_ratio",
    ]

    fieldnames = [
        "model", "condition", "framing", "task_id", "task_domain", "iteration",
    ] + metric_names

    rows = []
    count = 0
    skipped = 0

    with open(RAW_RESPONSES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("response") is None:
                skipped += 1
                continue

            metrics = compute_metrics(rec["response"], rec["task_id"])
            row = {
                "model": rec["model"],
                "condition": rec["condition"],
                "framing": rec["framing"],
                "task_id": rec["task_id"],
                "task_domain": rec["task_domain"],
                "iteration": rec.get("iteration", ""),
            }
            row.update(metrics)
            rows.append(row)
            count += 1

            if count % 100 == 0:
                print(f"  Processed {count} responses...")

    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Processed {count} responses, skipped {skipped} failures.")
    print(f"Metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    process_all_responses()
