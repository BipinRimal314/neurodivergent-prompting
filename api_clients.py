"""Unified API client for Claude, GPT-4, and Gemini."""

import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from config import RETRY_ATTEMPTS, RETRY_BASE_DELAY_S

load_dotenv()


def _call_anthropic(model_id: str, system_prompt: str, user_message: str,
                    temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _call_openai(model_id: str, system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _call_google(model_id: str, system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=model_id,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


_DISPATCHERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
}


def complete(
    model_name: str,
    model_id: str,
    provider: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    condition: str,
    framing: str,
    task_id: str,
    task_domain: str,
) -> dict:
    """Make a single LLM call with retry logic. Returns standardized dict."""

    dispatcher = _DISPATCHERS.get(provider)
    if dispatcher is None:
        raise ValueError(f"Unknown provider: {provider}")

    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            t0 = time.perf_counter()
            text = dispatcher(model_id, system_prompt, user_message,
                              temperature, max_tokens)
            latency_ms = (time.perf_counter() - t0) * 1000

            return {
                "model": model_name,
                "condition": condition,
                "framing": framing,
                "task_id": task_id,
                "task_domain": task_domain,
                "response": text,
                "latency_ms": round(latency_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS - 1:
                wait = RETRY_BASE_DELAY_S * (2 ** attempt)
                print(f"  [retry {attempt + 1}/{RETRY_ATTEMPTS}] {type(e).__name__}: {e}")
                print(f"  waiting {wait:.0f}s...")
                time.sleep(wait)

    # All retries exhausted
    print(f"  [FAILED] {model_name} {condition}/{framing} {task_id}: {last_error}")
    return {
        "model": model_name,
        "condition": condition,
        "framing": framing,
        "task_id": task_id,
        "task_domain": task_domain,
        "response": None,
        "latency_ms": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": str(last_error),
    }
