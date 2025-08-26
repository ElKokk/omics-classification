#!/usr/bin/env python3
"""
Usage:
    python plot_runtime_line.py <in.tsv> <out.png>

Creates a line plot of overall pipeline wall-clock time vs #CPU cores.
"""

import sys
from pathlib import Path
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
#  Custom colours
# -------------------------------------------------------------------------
model_colors = {
    "DLDA":         "#1f77b4",
    "kNN":          "#ff7f0e",
    "LDA":          "#2ca02c",
    "Lasso":        "#d62728",
    "RF":           "#9467bd",
    "SuperLearner": "#8c564b",
    "SVM":          "#e377c2",
}

# -------------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------------
if len(sys.argv) != 3:
    sys.exit("Usage: plot_runtime_line.py <in.tsv> <out.png>")

inp, out_png = sys.argv[1:3]

# -------------------------------------------------------------------------
#  Load data
# -------------------------------------------------------------------------
df = pd.read_csv(inp, sep="\t")

# ensure numeric ordering of cores if possible
with pd.option_context("mode.chained_assignment", None):
    try:
        df["cores"] = df["cores"].astype(int)
    except ValueError:
        pass
df = df.sort_values("cores")

# -------------------------------------------------------------------------
#  Theme
# -------------------------------------------------------------------------
sns.set(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 300

# -------------------------------------------------------------------------
#  Plot
# -------------------------------------------------------------------------
ax = sns.lineplot(
    data=df,
    x="cores",
    y="wall_clock_s",
    hue="model",
    palette=model_colors,
    marker="o",
    linewidth=2,
)

ax.set_xlabel("# CPU cores")
ax.set_ylabel("wall-clock [s]")
ax.set_title("Pipeline runtime vs parallelism")

ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

# -------------------------------------------------------------------------
#  Save
# -------------------------------------------------------------------------
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out_png, bbox_inches="tight")
plt.close()
print("✓ runtime plot →", out_png)
