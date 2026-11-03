"""Statistical analysis and visualization of experiment metrics."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests

from config import METRICS_FILE, PLOTS_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

METRIC_COLS = [
    "ttr", "word_count", "sentence_count", "avg_sentence_length",
    "hedging_per_100", "detail_density", "tangent_rate",
    "structural_markers", "sentiment_polarity", "emotional_word_ratio",
]

METRIC_LABELS = {
    "ttr": "Lexical Diversity (TTR)",
    "word_count": "Word Count",
    "sentence_count": "Sentence Count",
    "avg_sentence_length": "Avg Sentence Length",
    "hedging_per_100": "Hedging / 100 words",
    "detail_density": "Detail Density",
    "tangent_rate": "Tangent Rate",
    "structural_markers": "Structural Markers",
    "sentiment_polarity": "Sentiment Polarity",
    "emotional_word_ratio": "Emotion Words / 100",
}


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def run_analysis():
    if not os.path.exists(METRICS_FILE):
        print(f"No metrics file at {METRICS_FILE}. Run metrics.py first.")
        sys.exit(1)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(METRICS_FILE)
    print(f"Loaded {len(df)} rows from {METRICS_FILE}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print(f"Framings: {df['framing'].unique().tolist()}")
    print()

    models = df["model"].unique()
    conditions = [c for c in df["condition"].unique() if c != "control"]
    all_conditions = df["condition"].unique().tolist()
    domains = df["task_domain"].unique()

    # -----------------------------------------------------------------------
    # Statistical tests
    #
    # Two-stage family-wide multiple-comparison control (Benjamini-Hochberg
    # FDR at q = 0.05):
    #   Family 1 -- every Kruskal-Wallis omnibus test (metric x domain x model)
    #   Family 2 -- every Dunn's condition-vs-control pairwise test
    # We collect ALL candidate tests first (no interim p < 0.05 filter and no
    # within-domain Bonferroni), then correct each family once across the whole
    # grid. A finding requires: kruskal q < 0.05 AND dunn q < 0.05 AND |d| > 0.3.
    # This closes the "garden of forking paths": without it, the per-cell
    # thresholds count an expected share of false positives as real effects.
    # -----------------------------------------------------------------------
    FDR_Q = 0.05
    candidates = []  # every pairwise test considered, corrected below

    for model in models:
        model_df = df[df["model"] == model]

        for metric in METRIC_COLS:
            for domain in domains:
                subset = model_df[model_df["task_domain"] == domain]
                if subset[metric].isna().all():
                    continue

                groups = [
                    subset[subset["condition"] == c][metric].dropna().values
                    for c in all_conditions
                    if len(subset[subset["condition"] == c][metric].dropna()) > 0
                ]

                if len(groups) < 2 or any(len(g) < 3 for g in groups):
                    continue

                # Kruskal-Wallis omnibus (kept for ALL cells; correction later)
                try:
                    h_stat, p_val = stats.kruskal(*groups)
                except ValueError:
                    continue

                # Post-hoc Dunn's test -- raw (unadjusted) pairwise p-values;
                # the family-wide BH step below replaces the old per-domain
                # Bonferroni so correction spans the entire grid, not one cell.
                dunn_data = subset[["condition", metric]].dropna()
                try:
                    dunn_result = sp.posthoc_dunn(
                        dunn_data, val_col=metric, group_col="condition",
                        p_adjust=None
                    )
                except Exception:
                    continue

                control_vals = subset[subset["condition"] == "control"][metric].dropna()
                for cond in conditions:
                    cond_vals = subset[subset["condition"] == cond][metric].dropna()
                    if len(cond_vals) < 3 or len(control_vals) < 3:
                        continue

                    d = cohens_d(cond_vals, control_vals)

                    try:
                        dunn_p = dunn_result.loc[cond, "control"]
                    except KeyError:
                        try:
                            dunn_p = dunn_result.loc["control", cond]
                        except KeyError:
                            dunn_p = 1.0

                    candidates.append({
                        "model": model,
                        "domain": domain,
                        "metric": metric,
                        "condition": cond,
                        "kruskal_p": p_val,
                        "dunn_p": dunn_p,
                        "cohens_d": d,
                    })

    # ----- Family-wide Benjamini-Hochberg correction -----------------------
    if candidates:
        cand_df = pd.DataFrame(candidates)
        # Family 1: one omnibus p per (model, metric, domain) cell
        omnibus = cand_df.drop_duplicates(["model", "metric", "domain"]).copy()
        # An all-constant cell makes the test undefined (NaN). One NaN turns every
        # BH q-value NaN, so count undefined tests as p = 1 (never significant).
        omnibus["kruskal_q"] = multipletests(
            omnibus["kruskal_p"].fillna(1.0), method="fdr_bh")[1]
        cand_df = cand_df.merge(
            omnibus[["model", "metric", "domain", "kruskal_q"]],
            on=["model", "metric", "domain"], how="left")
        # Family 2: every condition-vs-control pairwise test
        cand_df["dunn_q"] = multipletests(
            cand_df["dunn_p"].fillna(1.0), method="fdr_bh")[1]

        raw_sig = ((cand_df["kruskal_p"] < 0.05) &
                   (cand_df["dunn_p"] < 0.05) &
                   (cand_df["cohens_d"].abs() > 0.3))
        fdr_sig = ((cand_df["kruskal_q"] < FDR_Q) &
                   (cand_df["dunn_q"] < FDR_Q) &
                   (cand_df["cohens_d"].abs() > 0.3))
        cand_df["significant_raw"] = raw_sig
        cand_df["significant_fdr"] = fdr_sig

        for col in ["kruskal_p", "dunn_p", "kruskal_q", "dunn_q"]:
            cand_df[col] = cand_df[col].round(4)
        cand_df["cohens_d"] = cand_df["cohens_d"].round(3)

        findings_df = cand_df[fdr_sig].sort_values(
            ["model", "metric", "cohens_d"],
            key=lambda x: x.abs() if x.name == "cohens_d" else x,
            ascending=[True, True, False])

        print("=" * 80)
        print(f"SIGNIFICANT FINDINGS (family-wide BH-FDR q < {FDR_Q}, |d| > 0.3)")
        print("=" * 80)
        if len(findings_df):
            print(findings_df[["model", "domain", "metric", "condition",
                               "kruskal_q", "dunn_q", "cohens_d"]].to_string(index=False))
        else:
            print("None survive family-wide FDR correction.")
        print(f"\nCandidate pairwise tests: {len(cand_df)}")
        print(f"Significant at raw thresholds (p<0.05 & |d|>0.3): {int(raw_sig.sum())}")
        print(f"Significant after family-wide FDR (q<{FDR_Q} & |d|>0.3): {int(fdr_sig.sum())}")

        # Full audit trail: every test with raw + corrected p and both flags
        cand_df.to_csv(os.path.join(PLOTS_DIR, "significant_findings.csv"), index=False)
        findings = findings_df.to_dict("records")
    else:
        findings = []
        print("No candidate tests could be computed.")

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------

    # 1. Heatmap: conditions × metrics, effect size vs control (per model)
    for model in models:
        model_df = df[df["model"] == model]
        control_df = model_df[model_df["condition"] == "control"]

        effect_matrix = pd.DataFrame(index=conditions, columns=METRIC_COLS, dtype=float)

        for cond in conditions:
            cond_df = model_df[model_df["condition"] == cond]
            for metric in METRIC_COLS:
                ctrl_vals = control_df[metric].dropna()
                cond_vals = cond_df[metric].dropna()
                if len(ctrl_vals) >= 3 and len(cond_vals) >= 3:
                    effect_matrix.loc[cond, metric] = cohens_d(cond_vals, ctrl_vals)
                else:
                    effect_matrix.loc[cond, metric] = 0.0

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            effect_matrix.astype(float),
            annot=True, fmt=".2f", center=0,
            cmap="RdBu_r", vmin=-1.5, vmax=1.5,
            xticklabels=[METRIC_LABELS.get(m, m) for m in METRIC_COLS],
            ax=ax,
        )
        ax.set_title(f"Effect Size vs Control — {model}", fontsize=14)
        ax.set_ylabel("Condition")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"heatmap_{model}.png"), dpi=150)
        plt.close()
        print(f"Saved heatmap_{model}.png")

    # 2. Box plots: one per metric, grouped by condition, faceted by model
    for metric in METRIC_COLS:
        metric_data = df[["model", "condition", metric]].dropna()
        if metric_data.empty:
            continue

        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            model_data = metric_data[metric_data["model"] == model]
            if model_data.empty:
                continue
            sns.boxplot(
                data=model_data, x="condition", y=metric,
                order=all_conditions, ax=ax, palette="Set2",
            )
            ax.set_title(model)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle(METRIC_LABELS.get(metric, metric), fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{metric}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"Saved boxplot_{metric}.png")

    # 3. Radar chart: per condition, normalized metric profile (per model)
    for model in models:
        model_df = df[df["model"] == model]

        # Normalize metrics to 0-1 range
        norm_data = {}
        for cond in all_conditions:
            cond_df = model_df[model_df["condition"] == cond]
            means = []
            for metric in METRIC_COLS:
                vals = model_df[metric].dropna()
                cond_vals = cond_df[metric].dropna()
                if len(vals) > 0 and vals.max() != vals.min():
                    normalized = (cond_vals.mean() - vals.min()) / (vals.max() - vals.min())
                else:
                    normalized = 0.5
                means.append(normalized)
            norm_data[cond] = means

        # Plot
        angles = np.linspace(0, 2 * np.pi, len(METRIC_COLS), endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_conditions)))

        for i, cond in enumerate(all_conditions):
            values = norm_data[cond] + norm_data[cond][:1]
            ax.plot(angles, values, "o-", linewidth=2, label=cond, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [METRIC_LABELS.get(m, m) for m in METRIC_COLS],
            size=8,
        )
        ax.set_title(f"Condition Profiles — {model}", size=14, y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"radar_{model}.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"Saved radar_{model}.png")

    # 4. Framing comparison heatmap (identity vs clinical)
    for model in models:
        model_df = df[df["model"] == model]
        framing_effects = pd.DataFrame(index=conditions, columns=METRIC_COLS, dtype=float)

        for cond in conditions:
            for metric in METRIC_COLS:
                id_vals = model_df[
                    (model_df["condition"] == cond) & (model_df["framing"] == "identity")
                ][metric].dropna()
                cl_vals = model_df[
                    (model_df["condition"] == cond) & (model_df["framing"] == "clinical")
                ][metric].dropna()
                if len(id_vals) >= 3 and len(cl_vals) >= 3:
                    framing_effects.loc[cond, metric] = cohens_d(id_vals, cl_vals)
                else:
                    framing_effects.loc[cond, metric] = 0.0

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            framing_effects.astype(float),
            annot=True, fmt=".2f", center=0,
            cmap="PiYG", vmin=-1.0, vmax=1.0,
            xticklabels=[METRIC_LABELS.get(m, m) for m in METRIC_COLS],
            ax=ax,
        )
        ax.set_title(f"Framing Effect (Identity vs Clinical) — {model}", fontsize=14)
        ax.set_ylabel("Condition")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"framing_{model}.png"), dpi=150)
        plt.close()
        print(f"Saved framing_{model}.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_analysis()
