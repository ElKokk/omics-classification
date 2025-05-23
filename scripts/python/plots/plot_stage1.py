from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

palette_metrics = sns.color_palette("colorblind", 3)
metric_order = ["MCE", "Sensitivity", "Specificity"]

###############################################################################
# Snakemake I/O
###############################################################################

metrics_fp = Path(snakemake.input["metrics"])
freq_fp    = Path(snakemake.input["freq"])

out_root = Path(snakemake.output[0])  # figures/{ds}/stage1_k{K} (reminder for directory target)
TITLE    = snakemake.params["title"]

out_root.mkdir(parents=True, exist_ok=True)

###############################################################################
# my data
###############################################################################

df_metrics = pd.read_csv(metrics_fp, sep="\t")
df_freq    = pd.read_csv(freq_fp)

###############################################################################
# Plot helpers ---
###############################################################################

def plot_gene_frequency(df: pd.DataFrame, out_fp: Path, title: str, top_n: int = 30):
    sub = df.sort_values("count", ascending=False).head(top_n).copy()
    sub["gene"] = pd.Categorical(sub["gene"], categories=sub["gene"], ordered=True)

    width = max(6, 0.35 * len(sub))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    sns.barplot(data=sub, x="gene", y="count", palette="viridis", ax=ax)

    ax.set_xlabel(f"Gene (top {top_n})", labelpad=10)
    ax.set_ylabel("Appearances in top‑K (max 200)", labelpad=10)
    ax.set_title(textwrap.fill(f"{title} · top {top_n} genes", 70), pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_fp, bbox_inches='tight')
    plt.close(fig)


def plot_per_split(df: pd.DataFrame, model: str, out_fp: Path, title: str):
    sub = df[df["model"] == model]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (metric, ls, col) in zip(metric_order, ["-", "--", ":"], palette_metrics):
        ax.plot(sub["split"], sub[metric], linestyle=ls, marker=".", label=metric, color=col)

    ax.set_xlabel("Monte‑Carlo split", labelpad=10)
    ax.set_ylabel("Score", labelpad=10)
    ax.set_ylim(0, 1.05)  # allow for outliers
    ax.set_title(f"{title} · {model}", pad=12)

    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(out_fp, bbox_inches='tight')
    plt.close(fig)


def plot_box(df: pd.DataFrame, model: str, out_fp: Path, title: str):
    sub = df[df["model"] == model]
    melt = sub.melt(id_vars=["split", "model"], value_vars=metric_order,
                    var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.boxplot(data=melt, x="Metric", y="Score", order=metric_order,
                palette=palette_metrics, ax=ax)
    sns.stripplot(data=melt, x="Metric", y="Score", order=metric_order,
                  color="black", size=3, alpha=0.4, jitter=True, ax=ax)

    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title} · {model} · distribution of metrics", pad=12)
    ax.set_xlabel("Metric", labelpad=10)
    ax.set_ylabel("Score", labelpad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette_metrics]
    ax.legend(handles, metric_order, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(out_fp, bbox_inches='tight')
    plt.close(fig)

###############################################################################
# plot generation
###############################################################################
plot_gene_frequency(df_freq, out_root / "gene_frequency.png", TITLE)


for model in df_metrics["model"].unique():
    mdl_dir = out_root / model
    mdl_dir.mkdir(exist_ok=True)
    plot_per_split(df_metrics, model, mdl_dir / "per_split.png", TITLE)
    plot_box      (df_metrics, model, mdl_dir / "boxplot.png",   TITLE)
