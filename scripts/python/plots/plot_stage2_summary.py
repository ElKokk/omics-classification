#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Barplots of MCE, Sensitivity, Specificity vs K for Stage‑2.
Structurally identical to plot_stage1_summary.py.
"""
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, textwrap
sns.set(context="paper", style="whitegrid")
plt.rcParams.update({"figure.dpi":300, "savefig.bbox":"tight",
                     "font.family":"sans-serif"})

df = pd.read_csv(snakemake.input[0], sep="\t").sort_values("K")
TITLE = snakemake.params.title
outs  = snakemake.output

def bar(y, png):
    fig,ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=df, x="K", y=y, ax=ax, color="tab:blue")
    ax.set(xlabel="Top‑K genes (fixed panel)", ylabel=y,
           title=textwrap.fill(f"{TITLE} – {y} vs K", 60))
    fig.tight_layout(); fig.savefig(png); plt.close(fig)

bar("MCE",         outs.MCE)
bar("Sensitivity", outs.Sensitivity)
bar("Specificity", outs.Specificity)
