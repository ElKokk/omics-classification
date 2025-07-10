#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_metrics_box.py
Creates three PNGs (MCE / Sensitivity / Specificity) with
x‑axis=K and one box per base learner.

INNER tier:
    input tables = inner_avg_metrics_k*.tsv
    already contain a column 'O' outer id so each box summarises the
    distribution of 10 outer‑fold averages

OUTER tier:
    input tables = metrics_k*.tsv   (already aggregated across test splits)
"""
from pathlib import Path
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from PIL import Image
import warnings, textwrap, sys

sns.set(context="paper", style="whitegrid")
plt.rcParams["figure.dpi"] = 300
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

tier       = snakemake.params.tier
title_base = snakemake.params.title_base
out_files  = dict(MCE=snakemake.output.MCE,
                  Sensitivity=snakemake.output.Sensitivity,
                  Specificity=snakemake.output.Specificity)

# ── read all input tables ───────────────────────────────────────────────────
frames = []
for fp in snakemake.input:
    try:
        df = pd.read_csv(fp, sep="\t")
    except Exception:
        continue
    if df.empty:
        continue
    if "K" not in df.columns:
        try:
            df["K"] = int(Path(fp).stem.split("_k")[-1])
        except Exception:
            continue
    frames.append(df)

if not frames:
    for p in out_files.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), "white").save(p, format="PNG")
    sys.exit()

df_all = pd.concat(frames, ignore_index=True)

for col in ["MCE", "Sensitivity", "Specificity"]:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

# ── helper to save blank PNG ────────────────────────────────────────────────
def blank_png(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "white").save(path, format="PNG")

# ── plotting function ───────────────────────────────────────────────────────
def plot_metric(metric, out_png):
    if metric not in df_all.columns:
        blank_png(out_png); return

    sub = (df_all[["model", "K", metric] + (["O"] if "O" in df_all.columns else [])]
           .dropna(subset=[metric])
           .rename(columns={metric: "val"}))
    if sub.empty:
        blank_png(out_png); return

    if tier == "inner" and "O" in sub.columns:
        sub = sub.groupby(["model", "K", "O"], as_index=False)["val"].mean()

    sub["K"] = sub["K"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=sub, x="K", y="val",
                hue="model", ax=ax, width=0.6, fliersize=2)

    ax.set_xlabel("Top‑K genes")
    ax.set_ylabel(metric)
    ax.set_title(textwrap.fill(f"{title_base} | {metric}", 80))
    ax.legend(frameon=False, fontsize=8, ncol=1,
              bbox_to_anchor=(1.02, 0.5), loc="center left",
              borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

for m, png in out_files.items():
    plot_metric(m, png)
