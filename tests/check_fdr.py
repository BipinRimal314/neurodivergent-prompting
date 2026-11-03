"""FDR checks. Run from the repo root.

  python tests/check_fdr.py                     # synthetic: one undefined cell must not zero out the FDR hits
  python tests/check_fdr.py path/to/metrics.csv # full release: expects 907 FDR survivors on the 17,943 released rows
                                                # (claude 380, gemini 347, gpt4 180; see neurodivbench-audit/reanalysis.py)
The CSV needs the columns of data/metrics.csv, with tangent_rate from the fixed metrics.py.
"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analysis  # noqa: E402


def run(df):
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "metrics.csv"
        df.to_csv(csv, index=False)
        analysis.METRICS_FILE, analysis.PLOTS_DIR = str(csv), td
        analysis.run_analysis()
        return pd.read_csv(Path(td) / "significant_findings.csv")


def synthetic():
    rng = np.random.default_rng(0)
    rows = []
    for cond, shift in [("control", 0), ("a", 3), ("b", 0)]:
        for i in range(25):
            r = dict(model="m", condition=cond, framing="identity", task_id="t", task_domain="d", iteration=i)
            r.update({c: rng.normal(shift if c == "word_count" else 0, 1) for c in analysis.METRIC_COLS})
            r["emotional_word_ratio"] = 0.0  # all-zero cell: kruskal p is NaN
            rows.append(r)
    out = run(pd.DataFrame(rows))
    assert out.kruskal_p.isna().any(), "test data no longer produces a NaN p-value"
    hit = out[out.significant_fdr]
    assert ((hit.metric == "word_count") & (hit.condition == "a")).any(), "NaN p-value wiped out all FDR hits"


def full(path):
    out = run(pd.read_csv(path))
    got = out[out.significant_fdr].model.value_counts().to_dict()
    assert len(out) == 1650 and got == {"claude": 380, "gemini": 347, "gpt4": 180}, (len(out), got)
    assert sum(got.values()) == 907


if __name__ == "__main__":
    full(sys.argv[1]) if len(sys.argv) > 1 else synthetic()
    print("ok")
