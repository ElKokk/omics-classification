#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustered bar‑plot of SuperLearner weight counts vs K.
Adds:
  • small gap between clusters
  • legend title “Base learners”
  • readable “0” annotation for empty bars
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import numpy as np, textwrap

# ────────── style ──────────────────────────────────────────────────────────
sns.set(context="paper", style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.labelsize": 14,  # Increased x/y labels
    "axes.titlesize": 16,  # Increased title
    "xtick.labelsize": 12,  # Increased x ticks
    "ytick.labelsize": 12,  # Increased y ticks
    "legend.fontsize": 12,  # Increased legend text
    "legend.title_fontsize": 14  # Increased legend title
})

# ────────── I/O ────────────────────────────────────────────────────────────
count_fp = Path(snakemake.input[0])
TITLE = snakemake.params["title"]
count_type = snakemake.params["type"] # 'positive' or 'highest'
stage = snakemake.params["stage"] # 'stage1' or 'stage2'
out_png = Path(snakemake.output[0])

# ────────── data ───────────────────────────────────────────────────────────
df = pd.read_csv(count_fp, sep="\t").sort_values("K")
models = sorted(df["model"].unique())
palette = sns.color_palette("colorblind", len(models))
Ks = sorted(df["K"].unique())

# ────────── bar geometry ───────────────────────────────────────────────────
CLUSTER_W = 1.3 # physical width (x‑units) of one K cluster
GAP = 0.5 # space between clusters
n_models = len(models)
bar_w = CLUSTER_W / n_models

x_cluster_left = np.arange(len(Ks)) * (CLUSTER_W + GAP)
x_centers = x_cluster_left + CLUSTER_W / 2

# ────────── figure dimensions ──────────────────────────────────────────────
fig_width = max(10, len(Ks) * (CLUSTER_W + GAP) + 2)
fig, ax = plt.subplots(figsize=(fig_width, 4.7))

# Track global max for zero‑label placement
max_count = df["count"].max() if not df["count"].empty else 1

# ────────── draw bars & annotate zeros ─────────────────────────────────────
for i, model in enumerate(models):
    heights = (
        df.query("model == @model")
          .set_index("K")["count"]
          .reindex(Ks)
          .fillna(0)
    )
    xs = x_cluster_left + i * bar_w
    bars = ax.bar(xs, heights, width=bar_w, color=palette[i], label=model)

# ────────── axis / legend / title ──────────────────────────────────────────
ax.set_xticks(x_centers)
ax.set_xticklabels([str(k) for k in Ks], rotation=0)
ax.set_xlabel("Top‑K genes")
ax.set_ylabel(f"Count ({count_type.capitalize()})")
ax.set_title(textwrap.fill(
    f"{TITLE} | {stage.capitalize()} |",
    60))

ax.legend(title="Base learners", frameon=False,
          bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout(rect=[0, 0, 0.78, 1]) # room for legend
fig.savefig(out_png)
plt.close(fig)