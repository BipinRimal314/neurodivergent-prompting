"""Main experiment runner. Iterates over the full experiment matrix."""

import argparse
import json
import os
import sys
import time

from config import (
    CONDITIONS,
    TASKS,
    MODELS,
    TEMPERATURE,
    MAX_TOKENS,
    ITERATIONS,
    CALL_DELAY_S,
    RAW_RESPONSES_FILE,
    DATA_DIR,
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
    AVG_INPUT_TOKENS,
    AVG_OUTPUT_TOKENS,
)
from api_clients import complete


def load_completed_cells(path: str) -> set:
    """Read existing responses and return set of (model, condition, framing, task_id, iteration)."""
    cells = set()
    if not os.path.exists(path):
        return cells
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (
                    rec["model"],
                    rec["condition"],
                    rec["framing"],
                    rec["task_id"],
                    rec.get("iteration", line_num),
                )
                cells.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return cells


def estimate_cost(model_names: list[str]) -> tuple[float, float]:
    """Return (estimated_cost_usd, estimated_time_minutes)."""
    total_calls = 0
    total_cost = 0.0

    for name in model_names:
        n_calls = len(CONDITIONS) * 2 * len(TASKS) * ITERATIONS
        # Control has identical framings, but we still run both for consistency
        total_calls += n_calls

        input_cost = (n_calls * AVG_INPUT_TOKENS / 1_000_000) * COST_PER_1M_INPUT.get(name, 5.0)
        output_cost = (n_calls * AVG_OUTPUT_TOKENS / 1_000_000) * COST_PER_1M_OUTPUT.get(name, 15.0)
        total_cost += input_cost + output_cost

    time_min = (total_calls * CALL_DELAY_S) / 60
    return total_cost, time_min, total_calls


def run_experiment(
    model_filter: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Determine which models to run
    if model_filter:
        if model_filter not in MODELS:
            print(f"Unknown model: {model_filter}. Options: {list(MODELS.keys())}")
            sys.exit(1)
        model_names = [model_filter]
    else:
        model_names = list(MODELS.keys())

    # Check API keys (skip for dry run)
    if not dry_run:
        for name in model_names:
            env_key = MODELS[name]["env_key"]
            if not os.environ.get(env_key):
                print(f"Missing {env_key} for model '{name}'. Set it in .env or environment.")
                sys.exit(1)

    # Cost estimate
    cost, time_min, total_calls = estimate_cost(model_names)
    print(f"\n{'='*60}")
    print(f"Experiment matrix")
    print(f"{'='*60}")
    print(f"  Models:      {model_names}")
    print(f"  Conditions:  {len(CONDITIONS)} (× 2 framings)")
    print(f"  Tasks:       {len(TASKS)}")
    print(f"  Iterations:  {ITERATIONS}")
    print(f"  Total calls: {total_calls:,}")
    print(f"  Est. cost:   ${cost:.2f}")
    print(f"  Est. time:   {time_min:.0f} min ({time_min/60:.1f} hrs)")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Showing first call per condition:\n")
        for name in model_names:
            cfg = MODELS[name]
            for cond, framings in CONDITIONS.items():
                sp = framings["identity"]
                task = TASKS[0]
                print(f"  [{name}] [{cond}] [identity] [{task['id']}]")
                print(f"    System: {sp}")
                print(f"    User:   {task['prompt'][:80]}...")
                print()
        return

    # Confirm
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Load completed cells for resume
    completed = set()
    if resume:
        completed = load_completed_cells(RAW_RESPONSES_FILE)
        print(f"Resuming: {len(completed)} cells already completed.\n")

    # Track progress
    done_count = 0
    skip_count = 0
    fail_count = 0

    with open(RAW_RESPONSES_FILE, "a") as f:
        for model_name in model_names:
            cfg = MODELS[model_name]
            for cond_name, framings in CONDITIONS.items():
                for framing_key in ["identity", "clinical"]:
                    system_prompt = framings[framing_key]
                    for task in TASKS:
                        for iteration in range(1, ITERATIONS + 1):
                            cell_key = (model_name, cond_name, framing_key, task["id"], iteration)

                            if resume and cell_key in completed:
                                skip_count += 1
                                continue

                            result = complete(
                                model_name=model_name,
                                model_id=cfg["model_id"],
                                provider=cfg["provider"],
                                system_prompt=system_prompt,
                                user_message=task["prompt"],
                                temperature=TEMPERATURE,
                                max_tokens=MAX_TOKENS,
                                condition=cond_name,
                                framing=framing_key,
                                task_id=task["id"],
                                task_domain=task["domain"],
                            )
                            result["iteration"] = iteration

                            f.write(json.dumps(result) + "\n")
                            f.flush()

                            if result.get("response") is None:
                                fail_count += 1
                                status = "FAILED"
                            else:
                                done_count += 1
                                status = f"{result['latency_ms']:.0f}ms"

                            print(
                                f"  [{model_name}] [{cond_name}] [{framing_key}] "
                                f"[{task['id']}] [{iteration}/{ITERATIONS}] — {status}"
                            )

                            time.sleep(CALL_DELAY_S)

    print(f"\nDone. completed={done_count}, skipped={skip_count}, failed={fail_count}")
    print(f"Raw data: {RAW_RESPONSES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Neurodivergent Prompting Experiment")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (claude, gpt4, gemini)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first call per condition without hitting APIs")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed cells")
    args = parser.parse_args()

    run_experiment(
        model_filter=args.model,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
