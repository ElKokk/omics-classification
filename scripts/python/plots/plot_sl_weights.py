#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heat‑map of mean Super‑Learner weights
rows  = base‑learner
cols  = K (signature size)
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# ─────── I/O ───────────────────────────────────────────────────────────────
df   = pd.read_csv(snakemake.input[0], sep="\t", index_col=0)
TITLE = snakemake.params.title
png   = snakemake.output[0]


try:
    df = df.reindex(sorted(df.columns, key=int), axis=1)
except ValueError:
    pass

# ─────── plot ──────────────────────────────────────────────────────────────
fig_w = 1.6 + 0.6 * df.shape[1]
fig_h = 1.2 + 0.4 * df.shape[0]

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
sns.heatmap(df, annot=True, fmt=".2f",
            cmap="viridis", cbar_kws=dict(label="mean weight"),
            ax=ax)

ax.set_title(textwrap.fill(f"{TITLE} – mean SL weights", 60))
ax.set_xlabel("Top‑K genes")
ax.set_ylabel("Base learner")

fig.tight_layout()
Path(png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(png)
plt.close(fig)
