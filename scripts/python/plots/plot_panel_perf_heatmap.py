#!/usr/bin/env python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, textwrap
from pathlib import Path

sns.set(context="paper", style="whitegrid")
plt.rcParams["figure.dpi"] = 300

df = pd.read_csv(snakemake.input[0], sep="\t")
if df.empty:
    Path(snakemake.output[0]).touch(); exit()

fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(df.set_index("K"), annot=True, fmt=".3f",
            cmap="viridis_r", cbar=False, ax=ax)
ax.set_title(textwrap.fill("SuperLearner – outer TEST mean MCE", 50))
fig.tight_layout(); fig.savefig(snakemake.output[0])
