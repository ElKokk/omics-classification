"""
Runtime visualisations
————————
Input : summary_stage1.tsv
Output: five PNGs (mean & total; wall-clock; cores speed-up)
"""
from pathlib import Path, datetime
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

sns.set_context("paper"); sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300

summary_fp = Path(snakemake.input[0])
out_train_mean = Path(snakemake.output["Train_mean"])
out_pred_mean  = Path(snakemake.output["Pred_mean"])
out_train_tot  = Path(snakemake.output["Train_total"])
out_pred_tot   = Path(snakemake.output["Pred_total"])
out_wall       = Path(snakemake.output["Wall_clock"])
out_speedup    = Path(snakemake.output["Speed_up"])
TITLE          = snakemake.params["title"]

df = pd.read_csv(summary_fp, sep="\t").sort_values("K")
palette = sns.color_palette("colorblind", df["model"].nunique())

# ── helper ───────────────────────────────────────────────────────────────
def plot(df, y, ylabel, out_fp, title):
    fig, ax = plt.subplots(figsize=(7,4))
    for c,(mdl,sub) in zip(palette, df.groupby("model")):
        ax.plot(sub["K"], sub[y], marker="o", label=mdl, color=c)
    ax.set_xlabel("Top-K features")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} · {ylabel} vs K")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_fp); plt.close(fig)

# mean ± SE already plotted earlier; here we show means only
plot(df, "Train_mean", "Train time (s)", out_train_mean, TITLE)
plot(df, "Pred_mean",  "Pred time (s)",  out_pred_mean,  TITLE)
plot(df, "Train_total","Total train time (s)", out_train_tot, TITLE)
plot(df, "Pred_total", "Total pred time (s)",  out_pred_tot,  TITLE)

# ── wall-clock vs K (sum of wall_clock_k*.txt) ───────────────────────────
wall_rows = []
for k in df["K"].unique():
    f = summary_fp.parent.parent / f"wall_clock_k{k}.txt"
    secs = float(f.read_text().strip()) if f.exists() else np.nan
    wall_rows.append({"K":k, "wall_s":secs})
wall = pd.DataFrame(wall_rows).sort_values("K")
plot(wall, "wall_s", "Wall-clock seconds", out_wall, TITLE)

# ── speed-up vs cores (requires multiple pipeline runs) ──────────────────
cores_fp = summary_fp.parent / "runtime_by_cores.tsv"
if cores_fp.exists():
    cdf = pd.read_csv(cores_fp, sep="\t").dropna(subset=["wall_clock_s"])
    cdf = cdf.sort_values("cores")
    base = cdf["wall_clock_s"].iloc[0]
    cdf["speed_up"] = base / cdf["wall_clock_s"]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(cdf["cores"], cdf["speed_up"], marker="o")
    ax.set_xlabel("Snakemake cores")
    ax.set_ylabel("Speed-up")
    ax.set_title(f"{TITLE} · wall-clock speed-up")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out_speedup); plt.close(fig)
else:
    # create an empty placeholder so Snakemake is satisfied
    Path(out_speedup).touch()
