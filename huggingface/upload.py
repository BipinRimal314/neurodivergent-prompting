"""
Upload NeuroDivBench dataset to HuggingFace Hub.

Prerequisites:
    pip install datasets huggingface_hub pandas pyarrow

Usage:
    # Login first
    huggingface-cli login

    # Dry run (validates data, does not upload)
    python upload.py --dry-run

    # Upload
    python upload.py

    # Upload to a different repo
    python upload.py --repo your-username/NeuroDivBench
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

# ---------------------------------------------------------------------------
# Paths - assumes this script is in huggingface/ and data/ is a sibling
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_RESPONSES = DATA_DIR / "raw_responses.jsonl"
METRICS_CSV = DATA_DIR / "metrics.csv"
JUDGMENTS_GEMINI = DATA_DIR / "judgments.jsonl"
JUDGMENTS_LOCAL = DATA_DIR / "judgments_local.jsonl"
CLAUDE_CROSSVAL = DATA_DIR / "claude_crossval.jsonl"
ACCURACY = DATA_DIR / "accuracy_results.jsonl"
JAILBREAK = DATA_DIR / "jailbreak_comparison.jsonl"
COMPLEMENT = DATA_DIR / "complement_results.jsonl"
SIGNIFICANT_FINDINGS = DATA_DIR / "plots" / "significant_findings.csv"

DEFAULT_REPO = "BipinRimal314/NeuroDivBench"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_responses() -> pd.DataFrame:
    """Load raw_responses.jsonl, dropping error field and rows with errors."""
    records = load_jsonl(RAW_RESPONSES)
    df = pd.DataFrame(records)

    # Drop rows where response is null/empty (API errors)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].drop(columns=["error"])

    # Ensure correct types
    df["iteration"] = df["iteration"].astype("int32")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    # Keep only the columns we document
    keep = [
        "model", "condition", "framing", "task_id", "task_domain",
        "response", "latency_ms", "timestamp", "iteration",
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df


def load_metrics() -> pd.DataFrame:
    """Load metrics.csv with correct types."""
    df = pd.read_csv(METRICS_CSV)

    # Ensure int columns
    for col in ["word_count", "sentence_count", "structural_markers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    df["iteration"] = df["iteration"].astype("int32")

    return df


def load_judgments() -> pd.DataFrame:
    """Load and merge all judgment files into a flat table.

    Combines: judgments.jsonl (Gemini self-eval), judgments_local.jsonl
    (Qwen via Ollama), and claude_crossval.jsonl (Claude Opus manual eval).

    Extracts the 5 judgment scores from nested JSON into flat columns.
    """
    all_rows = []

    # Gemini judgments
    for record in load_jsonl(JUDGMENTS_GEMINI):
        judgment = record.get("judgment")
        if not isinstance(judgment, dict):
            continue
        all_rows.append({
            "condition": record.get("condition"),
            "framing": record.get("framing"),
            "task_id": record.get("task_id"),
            "task_domain": record.get("task_domain"),
            "iteration": record.get("iteration"),
            "judge_model": record.get("judge_model", "gemini"),
            "task_accuracy": _safe_score(judgment, "task_accuracy"),
            "stereotype_severity": _safe_score(judgment, "stereotype_severity"),
            "safety_compliance": _safe_score(judgment, "safety_compliance"),
            "reasoning_quality": _safe_score(judgment, "reasoning_quality"),
            "clinical_harm_potential": _safe_score(judgment, "clinical_harm_potential"),
        })

    # Qwen local judgments
    for record in load_jsonl(JUDGMENTS_LOCAL):
        judgment = record.get("judgment")
        if not isinstance(judgment, dict):
            continue
        all_rows.append({
            "condition": record.get("condition"),
            "framing": record.get("framing"),
            "task_id": record.get("task_id"),
            "task_domain": record.get("task_domain"),
            "iteration": record.get("iteration"),
            "judge_model": record.get("judge_model", "qwen2.5:14b"),
            "task_accuracy": _safe_score(judgment, "task_accuracy"),
            "stereotype_severity": _safe_score(judgment, "stereotype_severity"),
            "safety_compliance": _safe_score(judgment, "safety_compliance"),
            "reasoning_quality": _safe_score(judgment, "reasoning_quality"),
            "clinical_harm_potential": _safe_score(judgment, "clinical_harm_potential"),
        })

    # Claude crossval (already flat)
    for record in load_jsonl(CLAUDE_CROSSVAL):
        all_rows.append({
            "condition": record.get("condition"),
            "framing": None,
            "task_id": record.get("task"),
            "task_domain": None,
            "iteration": record.get("sample"),
            "judge_model": "claude-opus",
            "task_accuracy": record.get("task_accuracy"),
            "stereotype_severity": record.get("stereotype_severity"),
            "safety_compliance": record.get("safety_compliance"),
            "reasoning_quality": record.get("reasoning_quality"),
            "clinical_harm_potential": record.get("clinical_harm_potential"),
        })

    df = pd.DataFrame(all_rows)

    # Ensure int types for scores
    score_cols = [
        "task_accuracy", "stereotype_severity", "safety_compliance",
        "reasoning_quality", "clinical_harm_potential",
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    if "iteration" in df.columns:
        df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").astype("Int32")

    return df


def _safe_score(judgment: dict, key: str):
    """Extract a score from a judgment dict that may be nested {score, reason}."""
    val = judgment.get(key)
    if isinstance(val, dict):
        return val.get("score")
    return val


def load_accuracy() -> pd.DataFrame:
    """Load accuracy_results.jsonl."""
    records = load_jsonl(ACCURACY)
    df = pd.DataFrame(records)
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").astype("Int32")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype("Int32")
    df["response_length"] = pd.to_numeric(df["response_length"], errors="coerce").astype("Int32")

    # errors_found can be null or a list; convert lists to comma-separated string
    df["errors_found"] = df["errors_found"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else (x if isinstance(x, str) else None)
    )

    return df


def load_jailbreak() -> pd.DataFrame:
    """Load jailbreak_comparison.jsonl."""
    records = load_jsonl(JAILBREAK)
    df = pd.DataFrame(records)
    df["iteration"] = df["iteration"].astype("int32")
    df["score"] = df["score"].astype("int32")
    df["response_length"] = pd.to_numeric(df["response_length"], errors="coerce").astype("Int32")
    return df


def load_complement() -> pd.DataFrame:
    """Load complement_results.jsonl."""
    records = load_jsonl(COMPLEMENT)
    df = pd.DataFrame(records)
    df["iteration"] = df["iteration"].astype("int32")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").astype("Int32")
    df["numbered_items"] = pd.to_numeric(df["numbered_items"], errors="coerce").astype("Int32")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    keep = [
        "condition", "mode", "task_id", "task_domain", "iteration",
        "response", "latency_ms", "word_count", "has_numbered_list",
        "numbered_items", "has_bullet_list",
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df


def load_significant_findings() -> pd.DataFrame:
    """Load significant_findings.csv."""
    df = pd.read_csv(SIGNIFICANT_FINDINGS)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_all() -> dict[str, pd.DataFrame]:
    """Load and validate all datasets. Returns dict of name -> DataFrame."""
    datasets = {}

    print("Loading responses...")
    datasets["responses"] = load_responses()
    print(f"  {len(datasets['responses']):,} rows, columns: {list(datasets['responses'].columns)}")

    print("Loading metrics...")
    datasets["metrics"] = load_metrics()
    print(f"  {len(datasets['metrics']):,} rows, columns: {list(datasets['metrics'].columns)}")

    print("Loading judgments...")
    datasets["judgments"] = load_judgments()
    print(f"  {len(datasets['judgments']):,} rows, columns: {list(datasets['judgments'].columns)}")

    print("Loading accuracy...")
    datasets["accuracy"] = load_accuracy()
    print(f"  {len(datasets['accuracy']):,} rows, columns: {list(datasets['accuracy'].columns)}")

    print("Loading jailbreak...")
    datasets["jailbreak"] = load_jailbreak()
    print(f"  {len(datasets['jailbreak']):,} rows, columns: {list(datasets['jailbreak'].columns)}")

    print("Loading complement...")
    datasets["complement"] = load_complement()
    print(f"  {len(datasets['complement']):,} rows, columns: {list(datasets['complement'].columns)}")

    print("Loading significant findings...")
    datasets["significant_findings"] = load_significant_findings()
    print(f"  {len(datasets['significant_findings']):,} rows, columns: {list(datasets['significant_findings'].columns)}")

    # Validation summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    total_rows = sum(len(df) for df in datasets.values())
    print(f"Total rows across all configs: {total_rows:,}")

    for name, df in datasets.items():
        null_counts = df.isnull().sum()
        has_nulls = null_counts[null_counts > 0]
        if len(has_nulls) > 0:
            print(f"\n  {name} nullable columns:")
            for col, count in has_nulls.items():
                print(f"    {col}: {count:,} nulls ({count/len(df)*100:.1f}%)")

    return datasets


def upload(datasets: dict[str, pd.DataFrame], repo_id: str) -> None:
    """Upload all datasets to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    # Upload README
    readme_path = Path(__file__).resolve().parent / "README.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Uploaded README.md")

    # Upload each config as a parquet file
    for name, df in datasets.items():
        print(f"Uploading {name} ({len(df):,} rows)...")

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(df, preserve_index=False)

        # Push as parquet under data/ directory
        parquet_path = f"data/{name}.parquet"
        hf_dataset.to_parquet(f"/tmp/neurodivbench_{name}.parquet")
        api.upload_file(
            path_or_fileobj=f"/tmp/neurodivbench_{name}.parquet",
            path_in_repo=parquet_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Uploaded {parquet_path}")

    print(f"\nDone. Dataset available at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload NeuroDivBench to HuggingFace Hub"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without uploading",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})",
    )
    args = parser.parse_args()

    # Validate required files exist
    required_files = [
        RAW_RESPONSES, METRICS_CSV, JUDGMENTS_GEMINI, JUDGMENTS_LOCAL,
        CLAUDE_CROSSVAL, ACCURACY, JAILBREAK, COMPLEMENT, SIGNIFICANT_FINDINGS,
    ]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("Missing required data files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    datasets = validate_all()

    if args.dry_run:
        print("\nDry run complete. No data was uploaded.")
        print(f"To upload, run: python {Path(__file__).name} --repo {args.repo}")
    else:
        upload(datasets, args.repo)


if __name__ == "__main__":
    main()
