"""
Stage-1 summary plots – metric × K × classifier
Outputs three PNGs (MCE / Sensitivity / Specificity).
"""
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import math, matplotlib.ticker as mticker

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight",
                     "font.family": "sans-serif"})

###############################################################################
# Snakemake I/O
###############################################################################
summary_fp = Path(snakemake.input[0])
out_mce    = Path(snakemake.output["MCE"])
out_sens   = Path(snakemake.output["Sensitivity"])
out_spec   = Path(snakemake.output["Specificity"])
TITLE      = snakemake.params["title"]

out_mce.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(summary_fp, sep="\t").sort_values("K")

# map public metric → (mean col, se col)
metrics_cols = {
    "MCE":         ("MCE_mean",  "MCE_se"),
    "Sensitivity": ("Sens_mean", "Sens_se"),
    "Specificity": ("Spec_mean", "Spec_se")
}

palette = sns.color_palette("colorblind", df["model"].nunique())

def plot_metric(metric, df, out_fp, title):
    mean_col, se_col = metrics_cols[metric]
    fig, ax = plt.subplots(figsize=(7,4))

    max_y, min_y = 0, 1
    for c,(model,sub) in zip(palette, df.groupby("model")):
        ax.plot(sub["K"], sub[mean_col], label=model,
                marker="o", color=c)
        ax.fill_between(sub["K"], sub[mean_col]-sub[se_col],
                        sub[mean_col]+sub[se_col], alpha=0.25, color=c)
        max_y = max(max_y, (sub[mean_col] + sub[se_col]).max())
        min_y = min(min_y, (sub[mean_col] - sub[se_col]).min())

    # nice y-limits snapped to .025
    snap = lambda v,f: f(v/0.025)*0.025
    upper = min(1.0, snap(max_y*1.03, math.ceil))
    lower = max(0.0, snap(min_y*0.97, math.floor))
    if upper - lower < 0.05:
        upper = min(1.0, lower + 0.05)

    ax.set_ylim(lower, upper)
    ax.set_yticks(np.arange(lower, upper+0.0001, 0.025))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    ax.set_xlabel("Top-K features")
    ax.set_ylabel(metric)
    ax.set_title(f"{title} · {metric} vs K")
    ax.legend(frameon=False, loc="best")
    sns.despine(fig)
    fig.tight_layout(); fig.savefig(out_fp); plt.close(fig)

plot_metric("MCE",          df, out_mce,  TITLE)
plot_metric("Sensitivity",  df, out_sens, TITLE)
plot_metric("Specificity",  df, out_spec, TITLE)
