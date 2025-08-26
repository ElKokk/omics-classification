#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heatmap of mean Super‑Learner weights
Rows = base learners
Columns = K (signature size)
Color-coded by weight (0 to 1)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import numpy as np

# ─────── I/O ───────────────────────────────────────────────────────────────
df   = pd.read_csv(snakemake.input[0], sep="\t", index_col=0)
TITLE = snakemake.params.title
png   = snakemake.output[0]

try:
    df = df.reindex(sorted(df.columns, key=int), axis=1)
except ValueError:
    pass

# ─────── plot ──────────────────────────────────────────────────────────────
n_k = df.shape[1]  # Number of K values (columns)
n_learners = df.shape[0]  # Number of base learners (rows)
fig_w = max(6, 1 + 0.8 * n_k)  # Wide enough for many K, min width 6
fig_h = max(5, 1 + 0.5 * n_learners)  # Tall enough for many learners

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Heatmap: rows=learners, columns=K, values=weights
sns.heatmap(df, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Mean Weight'},
            linewidths=0.5, linecolor='gray')

ax.set_title(textwrap.fill(f"{TITLE} – Mean SL Weights Heatmap", 60))
ax.set_xlabel("Top‑K genes")
ax.set_ylabel("Base Learner")

Path(png).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(png)
plt.close(fig)