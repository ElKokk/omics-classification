#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_sl_pick_counts.py
Creates two bar‑plots for one tier (“inner” or “outer”):

1.  # splits / outer‑folds in which each base learner received a **positive**
    meta‑weight in the SuperLearner.
2.  # splits / outer‑folds in which each base learner obtained the
    **highest** meta‑weight.

The X‑axis is K (top‑K genes).  Only ONE figure per metric is produced.
"""
from pathlib import Path
import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt, textwrap, warnings

sns.set(context="paper", style="whitegrid")
plt.rcParams["figure.dpi"] = 300
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

tier   = snakemake.params.tier        # "inner" | "outer"
picked = pd.read_csv(snakemake.input.picked, sep="\t")
top    = pd.read_csv(snakemake.input.top,    sep="\t")

def make_plot(df: pd.DataFrame, title: str, y_label: str, gap: float = .4):
    """Return a figure with wider spacing between the K clusters."""
    order = sorted(df["model"].unique())
    df["K"] = df["K"].astype(int)
    pv = (df.pivot_table(index="model", columns="K",
                         values=df.columns[-1], aggfunc="sum")
            .reindex(order).fillna(0))

    Ks     = pv.columns.tolist()
    x      = np.arange(len(Ks))
    width  = (1.0 - gap) / len(order)          # shrink bar group for gap

    fig, ax = plt.subplots(figsize=(8,4))
    for i, mdl in enumerate(order):
        ax.bar(x + i*width, pv.loc[mdl], width, label=mdl)

    ax.set_xticks(x + width*(len(order)-1)/2, Ks)
    ax.set_xlabel("Top‑K genes")
    ax.set_ylabel(y_label)
    ax.set_title(textwrap.fill(title, 70))

    # vertical legend to the right
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 0.5),
              loc="center left", borderaxespad=0.)
    fig.tight_layout()
    return fig

def write_or_touch(df, out_path: str, kind: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        Path(out_path).touch()
        return

    if kind == "picked":
        lab  = ("# splits with weight>0"
                if tier=="inner"
                else "# outer folds with weight>0")
        ttl  = ("Inner MCCV with feature selection– base‑learners picked by SL"
                if tier=="inner"
                else "MCCV with fixed variables– base‑learners picked by SL")
    else:  # top
        lab  = ("# splits with highest weight"
                if tier=="inner"
                else "# splits with highest weight")
        ttl  = ("Inner MCCV with feature selection– base‑learners with highest weight"
                if tier=="inner"
                else "MCCV with fixed variables– base‑learner with highest weight")

    fig = make_plot(df, ttl, lab)
    fig.savefig(out_path)
    plt.close(fig)

write_or_touch(picked, snakemake.output.picked_plot, "picked")
write_or_touch(top,    snakemake.output.top_plot,    "top")
