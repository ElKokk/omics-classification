#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_stage1.py
Writes two PNGs in <output‑dir>:
    perf.png   – grouped mean bars
    freq.png   – Top‑50 gene‑frequency bars colour‑graded by count
"""

from pathlib import Path
import textwrap, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps

sns.set(context="paper", style="whitegrid")
plt.rcParams["figure.dpi"] = 300
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Snakemake I/O ------------------------------------------------------------
out_dir   = Path(snakemake.output[0])
metrics_t = Path(snakemake.input.metrics)
freq_t    = Path(snakemake.input.freq)
title     = snakemake.params.title
out_dir.mkdir(parents=True, exist_ok=True)

# ── perf.png -----------------------------------------------------------------
try:
    mdf = pd.read_csv(metrics_t, sep="\t")
    if {"model","MCE","Sensitivity","Specificity"}.issubset(mdf.columns) and not mdf.empty:
        perf = (mdf.groupby("model")[["MCE","Sensitivity","Specificity"]]
                    .mean().reset_index())
        melt = perf.melt(id_vars="model", var_name="metric", value_name="val")
        fig, ax = plt.subplots(figsize=(6,3.5))
        sns.barplot(data=melt, x="metric", y="val", hue="model",
                    palette="Set2", ax=ax)
        ax.set_title(textwrap.fill(f"{title}\nMean performance", 60))
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.legend(frameon=False, fontsize=7, ncol=1,
                  bbox_to_anchor=(1.02,0.5), loc="center left")
        fig.tight_layout()
        fig.savefig(out_dir/"perf.png")
        plt.close(fig)
    else:
        (out_dir/"perf.png").touch()
except Exception:
    (out_dir/"perf.png").touch()

# ── freq.png -----------------------------------------------------------------
try:
    fdf = pd.read_csv(freq_t)
    if {"gene","count"}.issubset(fdf.columns) and not fdf.empty:
        top = fdf.nlargest(50, "count").reset_index(drop=True)
        cmin, cmax = top["count"].min(), top["count"].max()

        if cmin == cmax:
            # all bars same height – use single colour
            colors = ["#8c1515"] * len(top)
        else:
            cmap = colormaps.get_cmap("rocket_r")
            colors = [cmap((c - cmin)/(cmax - cmin)) for c in top["count"]]

        fig, ax = plt.subplots(figsize=(max(8, 0.14*len(top)), 4))
        ax.bar(top.index, top["count"], color=colors)
        ax.set_xticks(top.index)
        ax.set_xticklabels(top["gene"], rotation=90, fontsize=6)
        ax.set_xlabel(""); ax.set_ylabel("# splits selected")
        ax.set_title(textwrap.fill(f"{title}\nTop‑50 gene‑selection frequency", 70))
        fig.tight_layout()
        fig.savefig(out_dir/"freq.png")
        plt.close(fig)
    else:
        (out_dir/"freq.png").touch()
except Exception:
    (out_dir/"freq.png").touch()
