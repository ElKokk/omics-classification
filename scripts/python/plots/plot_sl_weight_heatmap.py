#!/usr/bin/env python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path, PurePath

sns.set(context="paper", style="white")
df = pd.read_csv(snakemake.input[0], sep="\t", index_col=0)

title = PurePath(snakemake.output[0]).stem.replace("_"," ").title()

if df.empty:
    Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.axis("off"); plt.savefig(snakemake.output[0], dpi=72,
                                               transparent=True); plt.close()
else:
    fig, ax = plt.subplots(figsize=(1.6*len(df.columns), 0.4*len(df)+2))
    sns.heatmap(df, cmap="vlag", center=0, annot=True, fmt=".2f",
                cbar_kws={"label":"mean weight"}, ax=ax)
    ax.set(title=title, xlabel="Top‑K genes", ylabel="")
    fig.tight_layout()
    fig.savefig(snakemake.output[0])
