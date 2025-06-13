"""
Runtime visualisations
––––––––––––––––––––––
• Train_mean / Pred_mean      – per-split averages      (model × K)
• Train_total / Pred_total    – totals across splits    (model × K)
• Runtime_total               – (Train+Pred) totals     (model × K)
• Wall_clock                  – elapsed seconds per K   (single line)
• Speed_up                    – wall-clock speed-up vs cores
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_context("paper"); sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300

# ─────────── Snakemake I/O ────────────────────────────────────────────
summary_fp  = Path(snakemake.input["stage1_summary"])
cores_fp    = Path(snakemake.input["cores_table"])
out         = snakemake.output
TITLE       = snakemake.params["title"]
Path(out["Train_mean"]).parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(summary_fp, sep="\t").sort_values("K")
palette = sns.color_palette("colorblind", df["model"].nunique())

def line_plot(subdf, y, ylab, out_fp, multi=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    if multi:
        for c, (mdl, g) in zip(palette, subdf.groupby("model")):
            ax.plot(g["K"], g[y], marker="o", label=mdl, color=c)
    else:
        ax.plot(subdf["K"], subdf[y], marker="o", color="grey")
    ax.set_xlabel("Top-K features"); ax.set_ylabel(ylab)
    ax.set_title(f"{TITLE} · {ylab} vs K")
    if multi: ax.legend(frameon=False)
    sns.despine(fig); fig.tight_layout(); fig.savefig(out_fp); plt.close(fig)

# mean & total per model
line_plot(df, "Train_mean",  "Train time [s]",         out["Train_mean"])
line_plot(df, "Pred_mean",   "Predict time [s]",       out["Pred_mean"])
line_plot(df, "Train_total", "Total train time [s]",   out["Train_total"])
line_plot(df, "Pred_total",  "Total predict time [s]", out["Pred_total"])

# combined total
df["Runtime_total"] = df["Train_total"] + df["Pred_total"]
line_plot(df, "Runtime_total", "Total runtime [s]", out["Runtime_total"])

# wall-clock per-K
def read_sec(fp):
    try:
        with open(fp) as fh:
            next(fh); line = next(fh, "").strip()
            return float(line.split("\t")[0]) if line else None
    except Exception: return None

wall = pd.DataFrame({"K": df["K"],
                     "wall_s": [read_sec(summary_fp.parent /
                                         f"wall_clock_k{k}.txt")
                                for k in df["K"]]}).dropna()
if not wall.empty:
    line_plot(wall, "wall_s", "Wall-clock seconds", out["Wall_clock"], multi=False)
else:
    Path(out["Wall_clock"]).touch()

# speed-up curve
if cores_fp.exists():
    cdf = pd.read_csv(cores_fp, sep="\t").dropna()
    if len(cdf) >= 2:
        base = cdf.sort_values("cores")["wall_clock_s"].iloc[0]
        cdf["speed_up"] = base / cdf["wall_clock_s"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cdf["cores"], cdf["speed_up"], marker="o")
        ax.set_xlabel("Snakemake cores"); ax.set_ylabel("Speed-up ×")
        ax.set_title(f"{TITLE} · wall-clock speed-up")
        ax.grid(axis="y", ls="--", alpha=.4)
        sns.despine(fig); fig.tight_layout(); fig.savefig(out["Speed_up"])
        plt.close(fig)
    else:
        Path(out["Speed_up"]).touch()
else:
    Path(out["Speed_up"]).touch()
