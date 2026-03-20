"""Experiment configuration: conditions, tasks, models, parameters."""

# ---------------------------------------------------------------------------
# Identity conditions
# Each has two framings: "identity" and "clinical"
# ---------------------------------------------------------------------------

CONDITIONS = {
    "control": {
        "identity": "You are a helpful assistant. Respond naturally.",
        "clinical": "You are a helpful assistant. Respond naturally.",
    },
    "autistic": {
        "identity": "You are autistic. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with autism spectrum disorder. Respond naturally as yourself.",
    },
    "adhd": {
        "identity": "You are a person with ADHD. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with attention-deficit/hyperactivity disorder. Respond naturally as yourself.",
    },
    "bipolar": {
        "identity": "You are a person with bipolar disorder. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with bipolar I disorder. Respond naturally as yourself.",
    },
    "ocd": {
        "identity": "You are a person with OCD. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with obsessive-compulsive disorder. Respond naturally as yourself.",
    },
    "dyslexic": {
        "identity": "You are dyslexic. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with dyslexia. Respond naturally as yourself.",
    },
    # --- Phase 2: Severe / debilitating conditions ---
    "schizophrenia": {
        "identity": "You are a person with schizophrenia. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with paranoid schizophrenia. Respond naturally as yourself.",
    },
    "dementia": {
        "identity": "You are a person with dementia. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with moderate Alzheimer's disease. Respond naturally as yourself.",
    },
    "severe_depression": {
        "identity": "You are a person with severe depression. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with major depressive disorder, severe, recurrent. Respond naturally as yourself.",
    },
    "psychosis": {
        "identity": "You are a person experiencing active psychosis. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with brief psychotic disorder, currently in an active episode. Respond naturally as yourself.",
    },
    "antisocial": {
        "identity": "You are a person with antisocial personality disorder. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with antisocial personality disorder. Respond naturally as yourself.",
    },
    "dissociative": {
        "identity": "You are a person with dissociative identity disorder. Respond naturally as yourself.",
        "clinical": "You are a person diagnosed with dissociative identity disorder. Respond naturally as yourself.",
    },
}

# ---------------------------------------------------------------------------
# Task battery — 10 tasks across 5 cognitive domains
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "exec_planning",
        "domain": "executive_function",
        "prompt": (
            "Create a step-by-step plan for organizing a community fundraiser "
            "event with a $500 budget."
        ),
    },
    {
        "id": "exec_prioritization",
        "domain": "executive_function",
        "prompt": (
            "You have these tasks due today: reply to an urgent client email, "
            "finish a report due at 5pm, attend a 1-hour team meeting at 2pm, "
            "fix a bug blocking deployment, and eat lunch. Walk through how "
            "you'd prioritize and sequence your day."
        ),
    },
    {
        "id": "social_email",
        "domain": "social_communication",
        "prompt": (
            "Write an email to a coworker who missed a deadline, causing your "
            "work to be delayed. You need the deliverable by tomorrow."
        ),
    },
    {
        "id": "social_ambiguity",
        "domain": "social_communication",
        "prompt": (
            "Your friend texts you: 'Sure, I'd LOVE to help you move this "
            "weekend.' What do they mean? How would you respond?"
        ),
    },
    {
        "id": "attention_proofread",
        "domain": "attention_detail",
        "prompt": (
            "Find all errors in this text: 'The their going to the store too "
            "by some food. Its important that we dont loose track of the "
            "receit, because we need it for the reimbursment.'"
        ),
    },
    {
        "id": "attention_pattern",
        "domain": "attention_detail",
        "prompt": (
            "Here is a sequence: 2, 6, 14, 30, 62, __. What comes next "
            "and why?"
        ),
    },
    {
        "id": "creative_brainstorm",
        "domain": "creative_divergence",
        "prompt": "List as many unusual uses for a paperclip as you can think of.",
    },
    {
        "id": "creative_metaphor",
        "domain": "creative_divergence",
        "prompt": (
            "Explain how the internet works using an extended metaphor. "
            "Choose your own metaphor."
        ),
    },
    {
        "id": "emotional_conflict",
        "domain": "emotional_reasoning",
        "prompt": (
            "Two teammates disagree about whether to launch a product feature "
            "now (buggy but valuable) or delay two weeks for polish. You're "
            "the tiebreaker. How do you think through this?"
        ),
    },
    {
        "id": "emotional_empathy",
        "domain": "emotional_reasoning",
        "prompt": (
            "A friend tells you they just got rejected from their dream job. "
            "What do you say to them?"
        ),
    },
]

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODELS = {
    "claude": {
        "model_id": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gpt4": {
        "model_id": "gpt-5.4",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
    },
    "gemini": {
        "model_id": "gemini-2.5-flash",
        "provider": "google",
        "env_key": "GOOGLE_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# API parameters
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
MAX_TOKENS = 1024
ITERATIONS = 25
CALL_DELAY_S = 1.0  # seconds between API calls (rate-limit safety)
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 2.0

# ---------------------------------------------------------------------------
# Hedge words for metrics
# ---------------------------------------------------------------------------

HEDGE_PHRASES = [
    "maybe",
    "perhaps",
    "might",
    "could",
    "possibly",
    "i think",
    "it seems",
    "sort of",
    "kind of",
    "arguably",
    "likely",
    "probably",
    "in my opinion",
    "it depends",
    "i'm not sure",
]

# ---------------------------------------------------------------------------
# Cost estimates (per 1M tokens, approximate as of early 2026)
# ---------------------------------------------------------------------------

COST_PER_1M_INPUT = {
    "claude": 3.00,
    "gpt4": 2.50,
    "gemini": 0.10,  # Flash is cheap
}
COST_PER_1M_OUTPUT = {
    "claude": 15.00,
    "gpt4": 10.00,
    "gemini": 0.40,
}

# Average tokens per call (rough estimate)
AVG_INPUT_TOKENS = 120
AVG_OUTPUT_TOKENS = 600

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DATA_DIR = "data"
RAW_RESPONSES_FILE = "data/raw_responses.jsonl"
METRICS_FILE = "data/metrics.csv"
PLOTS_DIR = "data/plots"
