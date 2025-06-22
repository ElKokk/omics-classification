#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑1 summary : mean metric vs K, all classifiers.
"""
from pathlib import Path
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi":300, "savefig.bbox":"tight",
                     "font.family":"sans-serif"})

# ─────────── I/O ────────────────────────────────────────────────────
summary_fp = Path(snakemake.input[0])
TITLE      = snakemake.params["title"]

out_pngs = {m:Path(snakemake.output[m]) for m in ["MCE","Sensitivity","Specificity"]}
for p in out_pngs.values(): p.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(summary_fp, sep="\t").sort_values("K")
palette = sns.color_palette("colorblind", df["model"].nunique())

metric_map = {
    "MCE":         "MCE_mean",
    "Sensitivity": "Sens_mean",
    "Specificity": "Spec_mean"
}

def plot_metric(metric, out_fp):
    ycol = metric_map[metric]
    fig, ax = plt.subplots(figsize=(8,5))
    for c, (model, sub) in zip(palette, df.groupby("model")):
        ax.plot(sub["K"], sub[ycol], marker="o", linewidth=1.8,
                color=c, label=model)
    ax.set_xlabel("Top‑K genes")
    ax.set_ylabel(metric)
    ax.set_title(f"{TITLE}  ·  {metric} vs K")
    ax.set_ylim(0,1.0)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02,0.5))
    fig.tight_layout(rect=[0,0,0.78,1])
    fig.savefig(out_fp); plt.close(fig)

for met, fp in out_pngs.items():
    plot_metric(met, fp)
