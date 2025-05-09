"""
Stage‑1 visualisations (Monte‑Carlo CV)
---------------------------------------
PNG outputs per (dataset, K):

1. per_split.png      – line chart (with dots) of MCE, Sens., Spec. per split
2. gene_frequency.png – vertical bar‑plot of gene counts (train‑fold ranking)
3. mean_se.png        – box‑plots of MCE, Sens., Spec. distributions
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import sys, textwrap

# ────────────── Snakemake I/O ─────────────────────────────────────── #
metrics_fp = snakemake.input["metrics"]
freq_fp    = snakemake.input["freq"]
title      = snakemake.params["title"]
method     = snakemake.params.get("method", "LDA")

outdir = Path(snakemake.output[0]).parent
outdir.mkdir(parents=True, exist_ok=True)

# ────────────── style ─────────────────────────────────────────────── #
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.labelsize" : 11,
    "axes.titlesize" : 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

df = pd.read_csv(metrics_fp, sep="\t")

# ═══════════════════════════════════════════════════════════════════ #
# 1 ▍ Line plot for each metric by split
# ═══════════════════════════════════════════════════════════════════ #
long = df.melt(id_vars="split",
               value_vars=["MCE", "Sensitivity", "Specificity"],
               var_name="Metric", value_name="Value")

fig, ax = plt.subplots(figsize=(12.0, 5.0))
sns.lineplot(data=long, x="split", y="Value",
             hue="Metric", style="Metric",
             markers=True, dashes=False, linewidth=1.4,
             markersize=5, palette="husl", ax=ax)

ax.set_xlabel("Monte Carlo split #")
ax.set_ylabel("Score")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(textwrap.fill(f"{title} · {method} · metrics per split", 70))

ax.legend(title=None, loc="upper left",
          bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
fig.tight_layout()
fig.savefig(outdir / "per_split.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════ #
# 2 ▍ Barplot of gene frequency
# ═══════════════════════════════════════════════════════════════════ #
freq = (pd.read_csv(freq_fp)
          .sort_values("count", ascending=False)
          .reset_index(drop=True))

TOP_N = 30
shown = freq.head(TOP_N)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
sns.barplot(data=shown,
            x="gene", y="count",
            order=shown["gene"],
            color=sns.color_palette("crest", 1)[0],
            ax=ax)
ax.set_xlabel("Gene")
ax.set_ylabel(f"Appearances in top‑K (max {df.shape[0]})")
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, ha="right")
ax.set_title(textwrap.fill(f"{title} · {method} · top {TOP_N} genes", 70))
ax.margins(x=0.01)
fig.tight_layout()
fig.savefig(outdir / "gene_frequency.png", dpi=300)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════ #
# 3 ▍ Boxplot of MCE, Sensitivity, Specificity
# ═══════════════════════════════════════════════════════════════════ #
palette = sns.color_palette("husl", 3)
order   = ["MCE", "Sensitivity", "Specificity"]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.boxplot(data=long, x="Metric", y="Value",
            order=order, palette=palette, width=0.6, ax=ax)
ax.set_ylabel("Score")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title(textwrap.fill(f"{title} · {method} · distribution of metrics", 70))

handles = [mpatches.Patch(facecolor=palette[i], label=order[i])
           for i in range(3)]
ax.legend(handles=handles, title=None, loc="upper left",
          bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

fig.tight_layout()
fig.savefig(outdir / "mean_se.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[done] Figures written to {outdir}", file=sys.stderr)
