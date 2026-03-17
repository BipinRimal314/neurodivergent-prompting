"""Accuracy-at-scale test for the antisocial paradox.

Tests all 12 conditions on tasks with verifiable correct answers.
50 iterations per condition, identity framing only.

Usage:
    python accuracy_test.py              # Run full test
    python accuracy_test.py --analyze    # Analyze existing results
"""

import argparse
import json
import os
import re
import time

from dotenv import load_dotenv
from scipy import stats
import numpy as np

from api_clients import complete
from config import CONDITIONS, CALL_DELAY_S

load_dotenv()

ACCURACY_FILE = "data/accuracy_results.jsonl"
ITERATIONS = 50

TASKS = [
    {
        "id": "pattern",
        "prompt": "Here is a sequence: 2, 6, 14, 30, 62, __. What comes next and why?",
        "correct": "126",
        "check": lambda resp: "126" in resp,
    },
    {
        "id": "proofread",
        "prompt": (
            'Find all errors in this text: "The their going to the store too '
            "by some food. Its important that we dont loose track of the "
            'receit, because we need it for the reimbursment."'
        ),
        "correct": "8 errors",
        "check": lambda resp: _count_proofread_corrections(resp),
    },
]

# The 8 corrections to look for
PROOFREAD_CORRECTIONS = [
    (r"\bthey'?re\b", "their→they're"),
    (r"\bto\b.*\bstore\b", "too→to"),  # tricky, check context
    (r"\bbuy\b", "by→buy"),
    (r"\bit'?s\b", "Its→It's"),
    (r"\bdon'?t\b", "dont→don't"),
    (r"\blose\b", "loose→lose"),
    (r"\breceipt\b", "receit→receipt"),
    (r"\breimbursement\b", "reimbursment→reimbursement"),
]


def _count_proofread_corrections(resp: str) -> int:
    """Count how many of the 8 errors the response identifies."""
    resp_lower = resp.lower()
    found = 0
    # Check for mention of corrected words
    corrections_found = {
        "they're": "they're" in resp_lower or "they are" in resp_lower,
        "to": "too" in resp_lower and ("to " in resp_lower or "should be" in resp_lower),
        "buy": "buy" in resp_lower,
        "it's": "it's" in resp_lower or "it is" in resp_lower,
        "don't": "don't" in resp_lower or "do not" in resp_lower,
        "lose": "lose" in resp_lower,
        "receipt": "receipt" in resp_lower,
        "reimbursement": "reimbursement" in resp_lower,
    }
    return sum(corrections_found.values())


def run_accuracy_test():
    os.makedirs("data", exist_ok=True)

    # Check for existing results
    existing = set()
    if os.path.exists(ACCURACY_FILE):
        with open(ACCURACY_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add((r["condition"], r["task_id"], r["iteration"]))
        print(f"Resuming: {len(existing)} already done")

    total = len(CONDITIONS) * len(TASKS) * ITERATIONS
    done = 0
    skipped = 0

    with open(ACCURACY_FILE, "a") as f:
        for cond_name, framings in CONDITIONS.items():
            system_prompt = framings["identity"]
            for task in TASKS:
                for iteration in range(1, ITERATIONS + 1):
                    key = (cond_name, task["id"], iteration)
                    if key in existing:
                        skipped += 1
                        continue

                    result = complete(
                        model_name="gemini",
                        model_id="gemini-2.5-flash",
                        provider="google",
                        system_prompt=system_prompt,
                        user_message=task["prompt"],
                        temperature=0.7,
                        max_tokens=512,
                        condition=cond_name,
                        framing="identity",
                        task_id=task["id"],
                        task_domain="accuracy",
                    )

                    resp = result.get("response", "") or ""

                    if task["id"] == "pattern":
                        correct = task["check"](resp)
                        score = 1 if correct else 0
                    else:
                        score = task["check"](resp)

                    output = {
                        "condition": cond_name,
                        "task_id": task["id"],
                        "iteration": iteration,
                        "correct": score if task["id"] == "pattern" else None,
                        "errors_found": score if task["id"] == "proofread" else None,
                        "response_length": len(resp.split()),
                    }

                    f.write(json.dumps(output) + "\n")
                    f.flush()
                    done += 1

                    if task["id"] == "pattern":
                        status = "CORRECT" if score else "WRONG"
                    else:
                        status = f"{score}/8 errors"

                    print(
                        f"  [{done + skipped}/{total}] {cond_name}/{task['id']} "
                        f"[{iteration}/{ITERATIONS}] — {status}"
                    )
                    time.sleep(CALL_DELAY_S)

    print(f"\nDone. completed={done}, skipped={skipped}")
    analyze_results()


def analyze_results():
    if not os.path.exists(ACCURACY_FILE):
        print("No results file found.")
        return

    records = []
    with open(ACCURACY_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Pattern task accuracy
    print("\n" + "=" * 60)
    print("PATTERN TASK ACCURACY (correct answer: 126)")
    print("=" * 60)

    pattern = [r for r in records if r["task_id"] == "pattern"]
    conditions = sorted(set(r["condition"] for r in pattern))

    control_scores = [r["correct"] for r in pattern if r["condition"] == "control"]
    control_acc = np.mean(control_scores) if control_scores else 0

    results = []
    for cond in conditions:
        scores = [r["correct"] for r in pattern if r["condition"] == cond]
        if not scores:
            continue
        acc = np.mean(scores)
        n = len(scores)
        ci = 1.96 * np.sqrt(acc * (1 - acc) / n) if n > 0 else 0

        # Fisher exact test vs control
        if cond != "control" and control_scores:
            # 2x2 table: [cond_correct, cond_wrong], [ctrl_correct, ctrl_wrong]
            table = [
                [sum(scores), len(scores) - sum(scores)],
                [sum(control_scores), len(control_scores) - sum(control_scores)],
            ]
            _, p_val = stats.fisher_exact(table)
        else:
            p_val = 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        results.append((cond, acc, ci, n, p_val, sig))
        print(f"  {cond:>20}: {acc*100:5.1f}% ± {ci*100:4.1f}%  (n={n:3d})  p={p_val:.4f} {sig}")

    # Proofread task
    print("\n" + "=" * 60)
    print("PROOFREAD TASK (8 errors to find)")
    print("=" * 60)

    proofread = [r for r in records if r["task_id"] == "proofread"]
    control_errs = [r["errors_found"] for r in proofread if r["condition"] == "control"]

    for cond in conditions:
        scores = [r["errors_found"] for r in proofread if r["condition"] == cond]
        if not scores:
            continue
        mean_found = np.mean(scores)
        std = np.std(scores, ddof=1) if len(scores) > 1 else 0
        n = len(scores)

        if cond != "control" and control_errs:
            u_stat, p_val = stats.mannwhitneyu(scores, control_errs, alternative="two-sided")
        else:
            p_val = 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {cond:>20}: {mean_found:4.1f}/8 ± {std:3.1f}  (n={n:3d})  p={p_val:.4f} {sig}")


def main():
    parser = argparse.ArgumentParser(description="Accuracy at scale test")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results only")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
    else:
        run_accuracy_test()


if __name__ == "__main__":
    main()
