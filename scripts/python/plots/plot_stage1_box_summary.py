#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑1 · boxplot summary: distribution of metrics vs K, all models.
Creates three PNGs (MCE / Sensitivity / Specificity).
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import textwrap, sys
from PIL import Image




model_colors = {
    "DLDA":         "#1f77b4",
    "kNN":          "#ff7f0e",
    "LDA":          "#2ca02c",
    "Lasso":        "#d62728",
    "RF":           "#9467bd",
    "SuperLearner": "#8c564b",
    "SVM":          "#e377c2",
}

# ────────── style ──────────────────────────────────────────────────────────
sns.set(context="paper", style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# ────────── I/O ────────────────────────────────────────────────────────────
in_fps   = snakemake.input
TITLE    = snakemake.params["title"]
n_splits = snakemake.params["n_splits"]

out_pngs = {m: Path(snakemake.output[i])
            for i, m in enumerate(["MCE", "Sensitivity", "Specificity"])}
for p in out_pngs.values():
    p.parent.mkdir(parents=True, exist_ok=True)

# ────────── load & combine data ────────────────────────────────────────────
frames = []
for fp in in_fps:
    k  = int(Path(fp).stem.split("_k")[-1])
    df = pd.read_csv(fp, sep="\t")
    df["K"] = k
    frames.append(df)

if not frames:
    for p in out_pngs.values():
        Image.new("RGB", (20, 20), "white").save(p, format="PNG")
    sys.exit()

combined = pd.concat(frames, ignore_index=True).sort_values("K")

models  = sorted(combined["model"].unique())
palette = model_colors
Ks      = sorted(combined["K"].unique())


GAPS = 1
order = []
for i, k in enumerate(Ks):
    order.append(str(k))

    if i < len(Ks) - 1:
        for g in range(GAPS):
            order.append(f"spacer_{i}_{g}")

combined["K_cat"] = combined["K"].astype(str)


def blank_png(path: Path):
    Image.new("RGB", (20, 20), "white").save(path, format="PNG")


def plot_metric(metric: str, out_png: Path) -> None:
    if metric not in combined.columns:
        blank_png(out_png); return

    sub = (combined[["model", "K_cat", metric]]
           .dropna(subset=[metric])
           .rename(columns={metric: "val"}))
    if sub.empty:
        blank_png(out_png); return


    fig_width  = max(9, 1.2 * len(order) + 1.6)
    fig_height = 4.7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.boxplot(
        data=sub, x="K_cat", y="val", hue="model",
        order=order,
        palette=palette, ax=ax,
        width=1, fliersize=1, linewidth=0.7
    )


    ax.set_xlabel("Top‑K genes")
    ax.set_ylabel(metric)
    ax.set_title(textwrap.fill(f"{TITLE} | {metric}", 80))
    #ax.set_ylim(0, 1.0)


    tick_labels = [cat if not cat.startswith("spacer_") else ""
                   for cat in order]
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")


    for idx, cat in enumerate(order):
        if cat.startswith("spacer_"):
            ax.axvline(x=idx, color="gray", linestyle="--",
                       alpha=0.25, linewidth=0.4)


    ax.legend(frameon=False, fontsize=10, ncol=1,
              bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0.)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_png)
    plt.close(fig)

for m, png in out_pngs.items():
    plot_metric(m, png)
