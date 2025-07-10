#!/usr/bin/env python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
import sys, warnings

sns.set(style="whitegrid", context="paper")

pick   = pd.read_csv(snakemake.input.picks  , sep="\t")
top    = pd.read_csv(snakemake.input.winners, sep="\t")
meanwt = pd.read_csv(snakemake.input.meanwt , sep="\t")

tiers = Path(snakemake.output[0]).stem.split("_")[2]

Ks = sorted(pick["K"].unique())
if not Ks or len(Ks) != len(snakemake.output):
    for p in snakemake.output:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(); plt.axis("off"); plt.savefig(p, dpi=72, transparent=True); plt.close()
    sys.exit(0)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    palette = sns.color_palette("tab10", 2)
    for k, png in zip(Ks, snakemake.output):
        p  = pick[pick["K"]==k].set_index("model")["picked_count"]
        t  = top [top ["K"]==k].set_index("model")["topwinner_count"]
        mw = meanwt[meanwt["K"]==k].set_index("model")["mean_top_weight"]
        idx = p.index.union(t.index).sort_values()
        df = pd.DataFrame({
            "picked": p.reindex(idx).fillna(0).astype(int),
            "top"   : t.reindex(idx).fillna(0).astype(int),
            "meanw" : mw.reindex(idx).fillna(0)
        }).reset_index().rename(columns={"index":"model"})

        fig, axes = plt.subplots(2,1,sharex=True,figsize=(7,6),
                                 gridspec_kw={"height_ratios":[1,1.1]})

        sns.barplot(ax=axes[0], data=df, x="model", y="picked",
                    color=palette[0])
        axes[0].set_title(f"{tiers.capitalize()} – picked (weight>0) · K={k}")
        axes[0].set_ylabel("# folds")

        sns.barplot(ax=axes[1], data=df, x="model", y="top",
                    color=palette[1])
        for i,(cnt,wt) in enumerate(zip(df["top"],df["meanw"])):
            if cnt>0:
                axes[1].text(i, cnt, f"{wt:.2f}", ha="center", va="bottom",
                             fontsize=8, color="black")
        axes[1].set_title("highest weight (count)  –numbers = mean weight")
        axes[1].set_ylabel("# folds"); axes[1].set_xlabel("base learner")
        plt.setp(axes[1].get_xticklabels(), rotation=45)
        fig.tight_layout()

        Path(png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png); plt.close(fig)
