#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑2 · summary line‑plots  one line per model, metric vs K

mirror the Stage‑1 summary figure:
    – X‑axis = signature size
    – One PNG per metric 
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap




model_colors = {
    "DLDA":         "#1f77b4",
    "kNN":          "#ff7f0e",
    "LDA":          "#2ca02c",
    "Lasso":        "#d62728",
    "RF":           "#9467bd",
    "SuperLearner": "#8c564b",
    "SVM":          "#e377c2",
}

sns.set(context="paper", style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif"
})

# ────────── I/O ────────────────────────────────────────────────────────────
summary_fp = Path(snakemake.input[0])
TITLE      = getattr(snakemake.params, "title", "")

out_pngs = {m: Path(snakemake.output[m])
            for m in ["MCE", "Sensitivity", "Specificity"]}
for p in out_pngs.values():
    p.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(summary_fp, sep="\t").sort_values("K")

# ────────── harmonise column names ─────────────────────────────────────────
rename_map = {}
for c in df.columns:
    low = c.lower()
    if low in {"mce", "mce_mean"}:
        rename_map[c] = "MCE"
    elif low in {"sensitivity", "sens_mean", "sensitivity_mean"}:
        rename_map[c] = "Sensitivity"
    elif low in {"specificity", "spec_mean", "specificity_mean"}:
        rename_map[c] = "Specificity"
df.rename(columns=rename_map, inplace=True)

palette = model_colors

def plot_metric(metric: str, png: Path) -> None:
    """Draw one line per model across K values for the requested metric."""
    if metric not in df.columns:
        png.touch()
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    secol = metric + "_se"
    for model, sub in df.groupby("model"):
        c = palette.get(model, 'black')
        ax.plot(sub["K"], sub[metric], marker="o",
                linewidth=1.8, color=c, label=model)
        if secol in sub.columns:
            ax.fill_between(sub["K"], sub[metric] - sub[secol], sub[metric] + sub[secol],
                            alpha=0.2, color=c) 
    ax.set_xlabel("Top‑K genes (fixed panel)")
    ax.set_ylabel(metric)
    ax.set_title(textwrap.fill(f"{TITLE}  ·  {metric} vs K", 60))
    if metric == "MCE":
        min_val = df[metric].min()
        max_val = df[metric].max()
        margin = (max_val - min_val) * 0.05 if max_val > min_val else 0.05
        ax.set_ylim(min_val - margin, max_val + margin)
    else:
        ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(png)
    plt.close(fig)

for met, fp in out_pngs.items():
    plot_metric(met, fp)