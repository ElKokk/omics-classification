#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑2 · summary line‑plots  one line per model, metric vs K

mirror the Stage‑1 summary figure:
    – X‑axis = signature size
    – One PNG per metric (MCE, Sensitivity, Specificity)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# ────────── style ──────────────────────────────────────────────────────────
sns.set(context="paper", style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif"
})

# ────────── I/O ────────────────────────────────────────────────────────────
summary_fp = Path(snakemake.input[0])
TITLE      = snakemake.params["title"]

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

# ────────── palette and helper function ───────────────────────────────────
palette = sns.color_palette("colorblind", df["model"].nunique())

def plot_metric(metric: str, png: Path) -> None:
    """Draw one line per model across K values for the requested metric."""
    if metric not in df.columns:
        png.touch()
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, (model, sub) in zip(palette, df.groupby("model")):
        ax.plot(sub["K"], sub[metric], marker="o",
                linewidth=1.8, color=c, label=model)
    ax.set_xlabel("Top‑K genes (fixed panel)")
    ax.set_ylabel(metric)
    ax.set_title(textwrap.fill(f"{TITLE}  ·  {metric} vs K", 60))
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(png)
    plt.close(fig)

# ────────── draw all requested metrics ────────────────────────────────────
for met, fp in out_pngs.items():
    plot_metric(met, fp)
