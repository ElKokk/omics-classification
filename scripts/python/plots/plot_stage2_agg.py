#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑2 · single‑K figures (aggregate metrics.

Creates in  figures/<ds>/stage2_k<K>/ :
    gene_frequency.png
    barplot_metrics.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# ────────── style ──────────────────────────────────────────────────────────
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif"
})

# ────────── input / output ─────────────────────────────────────────────────
fp_metrics = Path(snakemake.input["metrics"])
fp_freq    = Path(snakemake.input["freq"])
TITLE      = snakemake.params["title"]
out_dir    = Path(snakemake.output[0])
out_dir.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(fp_metrics, sep="\t")
freq    = pd.read_csv(fp_freq)

# ────────── gene‑frequency barplot ─────────────────────────────────────────
top_n = 30
sub = freq.sort_values("count", ascending=False).head(top_n)
sub["gene"] = pd.Categorical(sub["gene"],
                             categories=sub["gene"],
                             ordered=True)

fig, ax = plt.subplots(figsize=(0.35 * len(sub) + 2, 5))
sns.barplot(data=sub, x="gene", y="count",
            hue="gene", palette="viridis",
            dodge=False, ax=ax)
leg = ax.get_legend()
if leg is not None:
    leg.remove()
ax.set_xlabel(f"Top {top_n} genes", labelpad=10)
ax.set_ylabel("Appearances in top‑K (max 100)", labelpad=10)
ax.set_title(textwrap.fill(f"{TITLE}  ·  top‑{top_n} gene frequency", 70))
ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=90, ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "gene_frequency.png")
plt.close(fig)

# ────────── metric column names ─────────────────────────────────
rename_map = {}
for c in metrics.columns:
    low = c.lower()
    if low in {"mce", "mce_mean"}:
        rename_map[c] = "MCE"
    elif low in {"sensitivity", "sens_mean", "sensitivity_mean"}:
        rename_map[c] = "Sensitivity"
    elif low in {"specificity", "spec_mean", "specificity_mean"}:
        rename_map[c] = "Specificity"
metrics.rename(columns=rename_map, inplace=True)

present = [c for c in ["MCE", "Sensitivity", "Specificity"]
           if c in metrics.columns]
if not present:

    (out_dir / "barplot_metrics.png").touch()
    raise SystemExit(0)

# ────────── long‑format table & barplot ───────────────────────────────────
long = metrics.melt(id_vars=["model"],
                    value_vars=present,
                    var_name="Metric",
                    value_name="Score")

palette = sns.color_palette("colorblind", metrics["model"].nunique())

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=long, x="Metric", y="Score", hue="model",
            palette=palette, ax=ax)
ax.set_ylim(0, 1.05)
ax.set_title(textwrap.fill(f"{TITLE}  ·  metric scores (means)", 70))
ax.set_xlabel("")
ax.set_ylabel("Score")
ax.legend(frameon=False,
          bbox_to_anchor=(1.02, 0.5), loc="center left")
fig.tight_layout(rect=[0, 0, 0.78, 1])
fig.savefig(out_dir / "barplot_metrics.png")
plt.close(fig)
