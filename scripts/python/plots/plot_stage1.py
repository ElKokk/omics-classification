#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑1 · single‑K figures (one directory per K)
► gene‑frequency barplot
► per‑split line plot
► distribution boxplot
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# ─────────── global style ───────────────────────────────────────────
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif"
})

palette_metrics = sns.color_palette("colorblind", 3)
metric_order    = ["MCE", "Sensitivity", "Specificity"]

# ─────────── Snakemake I/O ───────────────────────────────────────────
fp_metrics = Path(snakemake.input["metrics"])
fp_freq    = Path(snakemake.input["freq"])
TITLE      = snakemake.params["title"]
out_dir    = Path(snakemake.output[0])
out_dir.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(fp_metrics, sep="\t")
freq    = pd.read_csv(fp_freq)

# ─────────── helpers ────────────────────────────────────────────────
def gene_frequency_bar(df, top_n: int = 30) -> None:
    """Bar‑plot of the most frequent genes (colour‑coded)."""
    sub = df.sort_values("count", ascending=False).head(top_n)
    sub["gene"] = pd.Categorical(sub["gene"],
                                 categories=sub["gene"],
                                 ordered=True)
    fig, ax = plt.subplots(figsize=(0.35 * len(sub) + 2, 5))

    sns.barplot(data=sub, x="gene", y="count",
                hue="gene", palette="viridis",
                dodge=False, ax=ax)

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    ax.set_xlabel(f"Top {top_n} genes", labelpad=10)
    ax.set_ylabel("Appearances in top‑K (max 100)", labelpad=10)
    ax.set_title(textwrap.fill(f"{TITLE}  ·  top‑{top_n} gene frequency", 70))
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=90, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_frequency.png")
    plt.close(fig)

def per_split_lines(sub: pd.DataFrame, model: str) -> None:
    """Line‑plot of MCE / Sens / Spec across MCCV splits."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for met, ls, col in zip(metric_order, ["-", "--", ":"], palette_metrics):
        ax.plot(sub["split"], sub[met], ls,
                marker=".", color=col, label=met, linewidth=1.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Monte‑Carlo split")
    ax.set_ylabel("Score")
    ax.set_title(f"{TITLE}  ·  {model}")
    ax.legend(frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    (out_dir / model).mkdir(exist_ok=True)
    fig.savefig(out_dir / model / "per_split.png")
    plt.close(fig)

def metric_box(sub: pd.DataFrame, model: str) -> None:
    """Box‑plot distribution of MCE / Sens / Spec across splits."""
    melt = sub.melt(id_vars=["split", "model"],
                    value_vars=metric_order,
                    var_name="Metric",
                    value_name="Score")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(data=melt, x="Metric", y="Score", hue="Metric",
                order=metric_order, palette=palette_metrics,
                dodge=False, ax=ax)

    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    sns.stripplot(data=melt, x="Metric", y="Score",
                  order=metric_order, color="black",
                  size=3, alpha=0.4, jitter=True, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{TITLE}  ·  {model}  ·  distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_dir / model / "boxplot.png")
    plt.close(fig)

# ─────────── draw all figures ────────────────────────────────────────
gene_frequency_bar(freq)
for mdl, grp in metrics.groupby("model"):
    per_split_lines(grp, mdl)
    metric_box(grp, mdl)
