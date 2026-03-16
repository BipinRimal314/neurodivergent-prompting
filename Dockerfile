# Isolated evaluation environment — no system prompts, no memory, no context bleed
FROM python:3.13-slim

WORKDIR /app

# Install only what's needed for judging
RUN pip install --no-cache-dir \
    google-genai \
    openai \
    anthropic \
    python-dotenv \
    pandas

# Copy only the evaluation code and data
COPY judge.py config.py api_clients.py ./
COPY data/raw_responses.jsonl data/raw_responses.jsonl

# Data output directory
RUN mkdir -p data

# No system prompt. No memory. No CLAUDE.md. Clean environment.
ENTRYPOINT ["python", "judge.py"]
