#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime plots – generates the suite of PNGs consumed by the Snakemake report.
Robust to missing inputs; auto-switches to milliseconds for tiny values.
"""

# -------------------------------------------------------------------------
#  BACKEND first (headless render)
# -------------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"   # ignore any inherited notebook backend
import matplotlib
matplotlib.use("Agg", force=True)

# -------------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------------
from pathlib import Path
import logging
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
#  Global style
# -------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }
)

MODEL_COLORS = {
    "DLDA":         "#1f77b4",
    "kNN":          "#ff7f0e",
    "LDA":          "#2ca02c",
    "Lasso":        "#d62728",
    "RF":           "#9467bd",
    "SuperLearner": "#8c564b",
    "SVM":          "#e377c2",
}

# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def safe_read(fp: Path, **kw) -> pd.DataFrame:
    """Read CSV/TSV but return empty DF when file is absent or empty."""
    fp = Path(fp)
    if not fp.exists() or fp.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(fp, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def stub(tgt: str | Path):
    """Create a blank PNG placeholder."""
    Path(tgt).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.axis("off")
    fig.savefig(tgt, dpi=72, bbox_inches="tight")
    plt.close(fig)

def stub_all(outputs: list[str | Path]):
    for tgt in outputs:
        stub(tgt)

def unitize(series: pd.Series):
    """
    Return (data, label_suffix); auto-convert to ms if values are tiny.
    """
    smax = float(series.max()) if len(series) else 0.0
    if smax < 0.2:    # under 0.2 seconds → plot in ms
        return series * 1000.0, "[ms]"
    return series, "[s]"

def pick_stage1_df(stage1_fp: Path, mdf: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer a dedicated Stage-1 summary file if present; otherwise
    derive it from the per-core model-runtime table by picking the
    highest core count (or any single core) for a clean vs-K plot.
    """
    df = safe_read(stage1_fp, sep="\t")
    # Already has Train_mean/Pred_mean/Train_total/Pred_total?
    if not df.empty and {"model","K"}.issubset(df.columns):
        needed = {"Train_mean","Pred_mean","Train_total","Pred_total"}
        if needed.issubset(df.columns):
            return df

    # Fallback: construct from mdf (model_runtime_vs_cores.tsv)
    if mdf.empty:
        return pd.DataFrame()

    cols_ok = {"model","K","cores","Train_mean","Pred_mean","Train_total","Pred_total"}
    if not cols_ok.issubset(mdf.columns):
        return pd.DataFrame()

    # choose the largest cores to represent "the" run for vs-K lines
    pick = int(mdf["cores"].max())
    return mdf.loc[mdf["cores"] == pick, ["model","K","Train_mean","Pred_mean","Train_total","Pred_total"]].copy()

def line_vs_k(sub: pd.DataFrame, y: str, ylab_base: str, outfile: str | Path, title: str):
    if sub.empty or y not in sub.columns:
        stub(outfile); return

    yvals, unit = unitize(sub[y])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for mdl, g in sub.groupby("model"):
        ax.plot(g["K"], yvals.loc[g.index], marker="o",
                label=mdl, color=MODEL_COLORS.get(mdl, "#333333"), linewidth=1.6)

    ax.set(
        xlabel="Top‑K genes",
        ylabel=f"{ylab_base} {unit}",
        title=f"{title} · {ylab_base} vs K",
    )
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(outfile)
    plt.close(fig)

def totals_vs_cores(mdf: pd.DataFrame, y: str, outfile: str | Path, title: str):
    cols_ok = {"model","cores",y}
    if mdf.empty or not cols_ok.issubset(mdf.columns):
        stub(outfile); return

    agg = mdf.groupby(["model", "cores"], as_index=False)[y].sum(min_count=1)
    yvals, unit = unitize(agg[y])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for mdl, g in agg.groupby("model"):
        ax.plot(g["cores"], yvals.loc[g.index], marker="o",
                label=mdl, color=MODEL_COLORS.get(mdl, "#333333"))
    ax.set(xlabel="Snakemake cores", ylabel=f"{y.replace('_',' ')} {unit}",
           title=f"{title} · {y.replace('_',' ')}")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(outfile)
    plt.close(fig)

def totals_vs_cores_zoom(mdf: pd.DataFrame, y: str, outfile: str | Path, title: str):
    cols_ok = {"model","cores","K",y}
    if mdf.empty or not cols_ok.issubset(mdf.columns):
        stub(outfile); return

    yvals, unit = unitize(mdf[y])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    # small jitter so K labels don't overlap exactly
    unique_cores = sorted(mdf["cores"].unique())
    jitter = {c: j for c, j in zip(unique_cores, np.linspace(-0.15, 0.15, len(unique_cores)))}

    for mdl, g in mdf.groupby("model"):
        for _, r in g.iterrows():
            x = r["cores"] + jitter[r["cores"]]
            ax.scatter(x, yvals.loc[r.name], s=30,
                       color=MODEL_COLORS.get(mdl, "#333333"), zorder=3)
            ax.text(x, yvals.loc[r.name], f"K={int(r['K'])}", fontsize=7,
                    ha="center", va="bottom",
                    color=MODEL_COLORS.get(mdl, "#333333"))
        ax.plot(g["cores"], yvals.loc[g.index], ls="--", lw=0.8,
                color=MODEL_COLORS.get(mdl, "#333333"), alpha=0.45)

    ax.set(xlabel="Snakemake cores",
           ylabel=f"{y.replace('_',' ')} {unit}",
           title=textwrap.fill(f"{title} · {y.replace('_',' ')} (all K points)", 66))
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)

# -------------------------------------------------------------------------
#  Snakemake I/O
# -------------------------------------------------------------------------
TITLE     = snakemake.params.title
outs      = list(snakemake.output)

stage1_fp = Path(snakemake.input.stage1_summary)  # may not exist; we'll fallback
cores_fp  = Path(snakemake.input.cores_table)
model_fp  = Path(snakemake.input.model_table)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# read inputs
cdf = safe_read(cores_fp, sep="\t")
mdf = safe_read(model_fp,  sep="\t")
df  = pick_stage1_df(stage1_fp, mdf)

# If nothing at all, create stubs and exit
if df.empty and cdf.empty and mdf.empty:
    logging.warning("all runtime tables empty – generating stub images only")
    stub_all(outs)

# -------------------------------------------------------------------------
#  1) Train / Predict vs K (from df)
# -------------------------------------------------------------------------
line_vs_k(df, "Train_mean",  "Mean train time",   snakemake.output.Train_mean,  TITLE)
line_vs_k(df, "Pred_mean",   "Mean predict time", snakemake.output.Pred_mean,   TITLE)
line_vs_k(df, "Train_total", "Total train time",  snakemake.output.Train_total, TITLE)
line_vs_k(df, "Pred_total",  "Total predict time",snakemake.output.Pred_total,  TITLE)

# Total runtime = train + predict
if not df.empty and {"Train_total","Pred_total"}.issubset(df.columns):
    tmp = df.copy()
    tmp["Runtime_total"] = tmp["Train_total"] + tmp["Pred_total"]
else:
    tmp = pd.DataFrame(columns=["model","K","Runtime_total"])
line_vs_k(tmp, "Runtime_total", "Total runtime", snakemake.output.Runtime_total, TITLE)

# -------------------------------------------------------------------------
#  2) Wall-clock plots (overall pipeline)
# -------------------------------------------------------------------------
if not cdf.empty and {"cores","wall_clock_s"}.issubset(cdf.columns) and len(cdf) >= 1:
    # wall-clock vs cores
    yvals, unit = unitize(cdf["wall_clock_s"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(cdf["cores"], yvals, marker="o")
    ax.set(xlabel="Snakemake cores", ylabel=f"Wall‑clock {unit}",
           title=f"{TITLE} · wall‑clock")
    fig.tight_layout()
    fig.savefig(snakemake.output.Wall_clock_vs_cores)
    plt.close(fig)

    # speed-up (relative to min cores row)
    base = float(cdf["wall_clock_s"].iloc[0])
    if base > 0:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(cdf["cores"], base / cdf["wall_clock_s"], marker="o")
        ax.set(xlabel="Snakemake cores", ylabel="Speed‑up ×",
               title=f"{TITLE} · wall‑clock speed‑up")
        fig.tight_layout()
        fig.savefig(snakemake.output.Speed_up)
        plt.close(fig)
    else:
        stub(snakemake.output.Speed_up)
else:
    stub(snakemake.output.Wall_clock_vs_cores)
    stub(snakemake.output.Speed_up)

# -------------------------------------------------------------------------
#  3) Totals vs cores + zoom (from mdf)
# -------------------------------------------------------------------------
totals_vs_cores(mdf, "Train_total", snakemake.output.Train_total_vs_cores, TITLE)
totals_vs_cores(mdf, "Pred_total",  snakemake.output.Pred_total_vs_cores,  TITLE)
totals_vs_cores_zoom(mdf, "Train_total", snakemake.output.Train_total_vs_cores_zoom, TITLE)
