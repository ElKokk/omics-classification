"""
Stage‑1 summary plots – metric × K × classifier
==============================================
**three line charts** for *MCE*, *Sensitivity*, *Specificity*.

Input  (Snakemake):  summary_stage1.tsv  (per dataset)
Output (Snakemake):
    figures/{ds}/stage1_summary/MCE.png
    figures/{ds}/stage1_summary/Sensitivity.png
    figures/{ds}/stage1_summary/Specificity.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

###############################################################################
# Snakemake I/O --
###############################################################################

summary_fp = Path(snakemake.input[0])   # results/{ds}/stage1/summary_stage1.tsv
out_mce    = Path(snakemake.output["MCE"])
out_sens   = Path(snakemake.output["Sensitivity"])
out_spec   = Path(snakemake.output["Specificity"])
TITLE      = snakemake.params["title"]   # e.g. f"{ds}"

out_mce.parent.mkdir(parents=True, exist_ok=True)

###############################################################################
# load
###############################################################################

df = pd.read_csv(summary_fp, sep="\t")

df = df.sort_values("K")

###############################################################################
# helpers
###############################################################################

def plot_metric(metric: str, df: pd.DataFrame, out_fp: Path, title: str):
    """Line chart of mean ± SE vs K for each model, ticks every 0.025."""
    import math, matplotlib.ticker as mticker
    import numpy as np

    mean_col = f"{metric}_mean"
    se_col   = f"{metric}_se"

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette("colorblind", df["model"].nunique())

    max_y, min_y = 0, 1
    for color, (model, sub) in zip(palette, df.groupby("model")):
        ax.plot(sub["K"], sub[mean_col], label=model, marker="o", color=color)
        ax.fill_between(sub["K"], sub[mean_col] - sub[se_col], sub[mean_col] + sub[se_col],
                        alpha=0.25, color=color)
        max_y = max(max_y, (sub[mean_col] + sub[se_col]).max())
        min_y = min(min_y, (sub[mean_col] - sub[se_col]).min())


    def snap(val, func):
        return func(val / 0.025) * 0.025

    upper = min(1.0, snap(max_y * 1.03, math.ceil))
    lower = max(0.0, snap(min_y * 0.97, math.floor))
    if upper - lower < 0.05:
        upper = min(1.0, lower + 0.05)

    ax.set_ylim(lower, upper)


    ticks = np.arange(0, upper + 0.0001, 0.025)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    ax.set_xlabel("Top K Features")
    ax.set_ylabel(metric)
    ax.set_title(f"{title} · {metric} vs K")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_fp)
    plt.close(fig)

###############################################################################
# plots
###############################################################################

plot_metric("MCE",          df, out_mce,  TITLE)
plot_metric("Sensitivity",  df, out_sens, TITLE)
plot_metric("Specificity",  df, out_spec, TITLE)
