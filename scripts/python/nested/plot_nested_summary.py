#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_nested_summary.py
OUTER summary version – re‑uses exactly the same code as
plot_stage1_summary.py (duplicated for self‑containment).
"""

from pathlib import Path
import sys, warnings, textwrap
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set(context="paper", style="whitegrid")
plt.rcParams["figure.dpi"] = 300
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

summary_fp = Path(snakemake.input.summary)
title      = snakemake.params.title
out_MCE, out_Sens, out_Spec = map(Path, [
    snakemake.output.MCE,
    snakemake.output.Sensitivity,
    snakemake.output.Specificity])
out_MCE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(summary_fp, sep="\t")
for base in ["MCE","Sensitivity","Specificity"]:
    if f"{base}_mean" in df.columns and base not in df.columns:
        df.rename(columns={f"{base}_mean": base}, inplace=True)

needed = {"model","K","MCE","Sensitivity","Specificity"}
if not needed.issubset(df.columns):
    for p in [out_MCE,out_Sens,out_Spec]: p.touch(); sys.exit()

df["K"] = pd.to_numeric(df["K"], errors="coerce")
for m in ["MCE","Sensitivity","Specificity"]:
    df[m] = pd.to_numeric(df[m], errors="coerce")
    vcol = f"{m}_var"
    if vcol in df.columns:
        df[f"{m}_std"] = np.sqrt(pd.to_numeric(df[vcol], errors="coerce"))

def plot(metric, png):
    mstd = f"{metric}_std"
    sub  = df[["model","K",metric] + ([mstd] if mstd in df.columns else [])].dropna(subset=[metric,"K"])
    if sub.empty: png.touch(); return
    palette = sns.color_palette(n_colors=sub["model"].nunique())
    fig, ax = plt.subplots(figsize=(6,3.5))
    for color, (model, grp) in zip(palette, sub.groupby("model")):
        grp = grp.sort_values("K")
        ax.plot(grp["K"], grp[metric], color=color, marker="o",
                label=model, linewidth=1.8)
        if mstd in grp.columns:
            std = grp[mstd].fillna(0)
            ax.fill_between(grp["K"], grp[metric]-std, grp[metric]+std,
                            color=color, alpha=0.25, linewidth=0)
    ax.set_title(textwrap.fill(f"{title} – {metric}", 70))
    ax.set_xlabel("Top‑K genes"); ax.set_ylabel(metric)
    ax.legend(frameon=False, fontsize=8,
              bbox_to_anchor=(1.02,0.5), loc="center left")
    fig.tight_layout(); fig.savefig(png); plt.close(fig)

plot("MCE", out_MCE); plot("Sensitivity", out_Sens); plot("Specificity", out_Spec)
