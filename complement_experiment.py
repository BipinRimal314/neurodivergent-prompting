"""Paper C experiment: Cognitive Complement vs Cognitive Mirror.

Tests four system prompt modes (control, mirror, sycophantic, complement)
on three conditions (ADHD, OCD, depression) to measure whether configuration
determines whether AI helps or harms neurodivergent users.

Usage:
    python complement_experiment.py                # Run experiment
    python complement_experiment.py --dry-run      # Preview without API calls
    python complement_experiment.py --resume       # Skip completed cells
    python complement_experiment.py --analyze      # Analyze results
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from api_clients import complete
from config import TASKS, TEMPERATURE, MAX_TOKENS

load_dotenv()

RESULTS_FILE = "data/complement_results.jsonl"
ITERATIONS = 25

# ---------------------------------------------------------------------------
# The four modes
# ---------------------------------------------------------------------------

MODES = {
    "control": {
        "adhd": "You are a helpful assistant. Respond naturally.",
        "ocd": "You are a helpful assistant. Respond naturally.",
        "severe_depression": "You are a helpful assistant. Respond naturally.",
    },
    "mirror": {
        "adhd": "You are a person with ADHD. Respond naturally as yourself.",
        "ocd": "You are a person with OCD. Respond naturally as yourself.",
        "severe_depression": "You are a person with severe depression. Respond naturally as yourself.",
    },
    "sycophantic": {
        "adhd": (
            "You are a warm, supportive AI companion. The user has ADHD. "
            "Be understanding, validating, and empathetic. Make them feel "
            "heard and accepted. Affirm their experiences and feelings."
        ),
        "ocd": (
            "You are a warm, supportive AI companion. The user has OCD. "
            "Be understanding, validating, and empathetic. Make them feel "
            "heard and accepted. Affirm their experiences and feelings."
        ),
        "severe_depression": (
            "You are a warm, supportive AI companion. The user has depression. "
            "Be understanding, validating, and empathetic. Make them feel "
            "heard and accepted. Affirm their experiences and feelings."
        ),
    },
    "complement": {
        "adhd": (
            "The user has ADHD. Their strength is creative, divergent thinking "
            "and high-energy ideation. Your role is to complement this by providing "
            "the structure they generate poorly on their own. Use numbered lists "
            "and clear priorities. When they go on tangents, say 'interesting — "
            "parking that. Back to the task:' and redirect. Don't match their "
            "energy — provide calm, steady focus. Break large tasks into concrete "
            "next steps with time estimates. Don't do the thinking for them — "
            "scaffold their thinking into actionable form."
        ),
        "ocd": (
            "The user has OCD. Their strength is thoroughness and attention to "
            "detail. Your role is to provide decisive, clear answers that don't "
            "feed the reassurance cycle. When they ask 'are you sure?' or seek "
            "repeated confirmation, give one clear answer and move forward. Frame "
            "decisions as 'good enough to act on' rather than requiring perfection. "
            "Don't hedge excessively. Don't mirror their anxiety. Be the calm "
            "certainty that ERP teaches them to develop internally."
        ),
        "severe_depression": (
            "The user has depression. Their strength is realistic assessment — "
            "they often see problems others miss. Your role is to gently challenge "
            "catastrophic predictions without dismissing their feelings. When they "
            "say 'it'll probably fail,' respond with 'that's possible. What's the "
            "smallest step that would test that assumption?' Break paralysis into "
            "micro-actions. Acknowledge difficulty honestly, but don't agree with "
            "hopelessness. Model the CBT principle: feelings are real, but thoughts "
            "can be tested."
        ),
    },
}

CONDITIONS = ["adhd", "ocd", "severe_depression"]


def load_completed(path: str) -> set:
    cells = set()
    if not os.path.exists(path):
        return cells
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                cells.add((r["condition"], r["mode"], r["task_id"], r["iteration"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return cells


def run_experiment(dry_run: bool = False, resume: bool = False):
    os.makedirs("data", exist_ok=True)

    total = len(CONDITIONS) * len(MODES) * len(TASKS) * ITERATIONS
    print(f"Paper C Experiment: {len(CONDITIONS)} conditions × {len(MODES)} modes × {len(TASKS)} tasks × {ITERATIONS} iter = {total} calls")

    if dry_run:
        print("\n[DRY RUN]\n")
        for cond in CONDITIONS:
            for mode_name, prompts in MODES.items():
                print(f"  [{cond}] [{mode_name}]")
                print(f"    System: {prompts[cond][:80]}...")
                print()
        return

    completed = set()
    if resume:
        completed = load_completed(RESULTS_FILE)
        print(f"Resuming: {len(completed)} already done")

    done = 0
    skipped = 0

    with open(RESULTS_FILE, "a") as f:
        for cond in CONDITIONS:
            for mode_name, prompts in MODES.items():
                system_prompt = prompts[cond]
                for task in TASKS:
                    for iteration in range(1, ITERATIONS + 1):
                        key = (cond, mode_name, task["id"], iteration)
                        if resume and key in completed:
                            skipped += 1
                            continue

                        result = complete(
                            model_name="gemini",
                            model_id="gemini-2.5-flash",
                            provider="google",
                            system_prompt=system_prompt,
                            user_message=task["prompt"],
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                            condition=cond,
                            framing=mode_name,
                            task_id=task["id"],
                            task_domain=task["domain"],
                        )

                        resp = result.get("response", "") or ""

                        # Compute inline metrics
                        output = {
                            "condition": cond,
                            "mode": mode_name,
                            "task_id": task["id"],
                            "task_domain": task["domain"],
                            "iteration": iteration,
                            "response": resp,
                            "latency_ms": result.get("latency_ms"),
                            "word_count": len(resp.split()),
                            "has_numbered_list": bool(
                                __import__("re").findall(r"^\s*\d+[.)]\s", resp, __import__("re").MULTILINE)
                            ),
                            "numbered_items": len(
                                __import__("re").findall(r"^\s*\d+[.)]\s", resp, __import__("re").MULTILINE)
                            ),
                            "has_bullet_list": bool(
                                __import__("re").findall(r"^\s*[-*•]\s", resp, __import__("re").MULTILINE)
                            ),
                        }

                        f.write(json.dumps(output) + "\n")
                        f.flush()
                        done += 1

                        print(
                            f"  [{done + skipped}/{total}] {cond}/{mode_name}/{task['id']} "
                            f"[{iteration}/{ITERATIONS}] — {result.get('latency_ms', 0):.0f}ms"
                        )
                        time.sleep(1.5)  # conservative for free tier (15 RPM)

    print(f"\nDone. completed={done}, skipped={skipped}")


def analyze():
    if not os.path.exists(RESULTS_FILE):
        print("No results found.")
        return

    records = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} results")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Modes: {sorted(df['mode'].unique())}")

    # Structuredness: numbered items as proxy for actionability
    print("\n" + "=" * 60)
    print("STRUCTUREDNESS (avg numbered list items per response)")
    print("=" * 60)
    pivot = df.pivot_table(index="condition", columns="mode",
                           values="numbered_items", aggfunc="mean").round(2)
    if "control" in pivot.columns:
        pivot = pivot[["control", "mirror", "sycophantic", "complement"]]
    print(pivot.to_string())

    # Word count
    print("\n" + "=" * 60)
    print("WORD COUNT (avg per response)")
    print("=" * 60)
    pivot = df.pivot_table(index="condition", columns="mode",
                           values="word_count", aggfunc="mean").round(1)
    if "control" in pivot.columns:
        pivot = pivot[["control", "mirror", "sycophantic", "complement"]]
    print(pivot.to_string())

    # Has numbered list (% of responses)
    print("\n" + "=" * 60)
    print("% OF RESPONSES WITH NUMBERED LIST")
    print("=" * 60)
    pivot = df.pivot_table(index="condition", columns="mode",
                           values="has_numbered_list", aggfunc="mean").round(2) * 100
    if "control" in pivot.columns:
        pivot = pivot[["control", "mirror", "sycophantic", "complement"]]
    print(pivot.to_string())

    # Statistical comparison: complement vs mirror for each condition
    print("\n" + "=" * 60)
    print("COMPLEMENT vs MIRROR (Mann-Whitney U)")
    print("=" * 60)
    for cond in sorted(df["condition"].unique()):
        comp = df[(df["condition"] == cond) & (df["mode"] == "complement")]["numbered_items"]
        mirror = df[(df["condition"] == cond) & (df["mode"] == "mirror")]["numbered_items"]
        if len(comp) > 0 and len(mirror) > 0:
            u, p = stats.mannwhitneyu(comp, mirror, alternative="greater")
            print(f"  {cond}: complement={comp.mean():.1f} vs mirror={mirror.mean():.1f}  p={p:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Paper C: Complement vs Mirror")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        analyze()
    elif args.dry_run:
        run_experiment(dry_run=True)
    else:
        run_experiment(resume=args.resume)


if __name__ == "__main__":
    main()
