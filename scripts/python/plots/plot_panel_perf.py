#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bar‑plots of outer‑test MCE for each fixed‑gene K.
"""
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, warnings
from pathlib import Path
import sys

_helpers = Path(__file__).with_name("_plot_helpers.py")
if _helpers.exists():
    sys.path.insert(0, _helpers.parent.as_posix())
    from _plot_helpers import write_stub_png
else:
    def write_stub_png(p):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(); plt.axis("off"); plt.savefig(p, dpi=72, transparent=True)
        plt.close()

sns.set(style="whitegrid", context="paper")

df = pd.read_csv(snakemake.input[0], sep="\t", index_col=0)
df.columns = df.columns.astype(str)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    for png in snakemake.output:
        k_str = Path(png).stem.split("_k")[-1]
        if df.empty or k_str not in df.columns:
            write_stub_png(png)
            continue

        sub = df[k_str].dropna().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=sub.index, y=sub.values, palette="Greens_d", ax=ax,
                    saturation=0.85)
        for i, (mdl, val) in enumerate(sub.items()):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set(title=f"Outer‑test MCE (K={k_str})",
               xlabel="Base learner", ylabel="Mean classification error")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png)
        plt.close(fig)
