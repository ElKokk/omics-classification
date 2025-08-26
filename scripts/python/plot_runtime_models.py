#!/usr/bin/env python3
"""
plot_runtime_models.py   –   bar-chart of train-time vs CPU cores.

Usage:
    python plot_runtime_models.py <in.tsv> <out.png> [<metric>]
"""
import sys, textwrap
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
# --------------------------------------------------------------------------
#  Custom palette
# --------------------------------------------------------------------------
model_colors = {
    "DLDA":         "#1f77b4",
    "kNN":          "#ff7f0e",
    "LDA":          "#2ca02c",
    "Lasso":        "#d62728",
    "RF":           "#9467bd",
    "SuperLearner": "#8c564b",
    "SVM":          "#e377c2",
}

# --------------------------------------------------------------------------
#  CLI
# --------------------------------------------------------------------------
if len(sys.argv) not in (3, 4):
    sys.exit(__doc__)

inp, out_png, *opt = sys.argv[1:]

# --------------------------------------------------------------------------
#  Load data & choose metric
# --------------------------------------------------------------------------
df = pd.read_csv(inp, sep="\t")

try:
    df["cores"] = df["cores"].astype(int)
    df = df.sort_values("cores")
except ValueError:
    pass

if opt:
    metric = opt[0]
    if metric not in df.columns:
        sys.exit(
            f"ERROR: column “{metric}” not found in {inp}\n"
            f"Columns: {', '.join(df.columns)}"
        )
else:
    metric = next(
        (
            c
            for c in df.columns
            if c not in ("cores", "model")
            and pd.api.types.is_numeric_dtype(df[c])
        ),
        None,
    )
    if metric is None:
        sys.exit("ERROR: no numeric metric column found.")

print(f"Plotting metric: {metric}")

# --------------------------------------------------------------------------
#  Figure size heuristics
# --------------------------------------------------------------------------
n_cores  = df["cores"].nunique()
n_models = df["model"].nunique()
n_bars   = n_cores * n_models

fig_w = max(8, 0.7 * n_bars)
fig_h = max(6, 0.4 * n_bars + 4)


# --------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk") 

# --------------------------------------------------------------------------
#  Two-row layout
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[8, 1])

ax  = fig.add_subplot(gs[0, 0])  
lax = fig.add_subplot(gs[1, 0])  
lax.axis("off")

# --------------------------------------------------------------------------
#  Bar-plot
# --------------------------------------------------------------------------
sns.barplot(
    data=df,
    x="cores",
    y=metric,
    hue="model",
    palette=model_colors, 
    dodge=True,
    ax=ax,
    width=0.8,
    edgecolor="black",
    linewidth=0.4,
    errorbar="sd",
)

ax.set_xlabel("# CPU cores")
ax.set_ylabel(metric.replace("_", " "))
ax.set_title(
    textwrap.fill(f"{metric.replace('_', ' ')} per model vs cores", 60)
)

# --------------------------------------------------------------------------
#  Legend
# --------------------------------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
ncol = min(n_models, 4)
lax.legend(
    handles,
    labels,
    loc="center",
    ncol=ncol,
    frameon=False,
    fontsize="medium",
)
ax.legend_.remove()

# --------------------------------------------------------------------------
#  Save
# --------------------------------------------------------------------------
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("✓ plot →", out_png)
