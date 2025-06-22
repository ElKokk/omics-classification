#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runtime plots
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, textwrap, sys, logging
sns.set(context="paper", style="whitegrid")
plt.rcParams.update({"figure.dpi":300, "savefig.bbox":"tight",
                     "font.family":"sans-serif"})

# ── helpers -----------------------------------------------------------------
def safe_read(fp, **kw):
    try:
        return pd.read_csv(fp, **kw)
    except pd.errors.EmptyDataError:
        logging.warning("empty file skipped: %s", fp)
        return pd.DataFrame()

def stub(tgt):
    Path(tgt).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.axis('off'); plt.savefig(tgt, dpi=72, bbox_inches='tight')

def stub_all(outputs):
    for tgt in outputs:
        stub(tgt)
    sys.exit(0)

# ── Snakemake I/O -----------------------------------------------------------
stage1_fp = Path(snakemake.input.stage1_summary)
cores_fp  = Path(snakemake.input.cores_table)
model_fp  = Path(snakemake.input.model_table)
TITLE     = snakemake.params.title
outs      = list(snakemake.output)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")

df  = safe_read(stage1_fp, sep="\t")
cdf = safe_read(cores_fp , sep="\t")
mdf = safe_read(model_fp , sep="\t")

# ── abort early if nothing to plot -----------------------------------------
if df.empty or cdf.empty or mdf.empty:
    logging.warning("one or more runtime tables empty – generating stubs only")
    stub_all(outs)

palette = sns.color_palette("colorblind", df["model"].nunique())
def line_vs_k(sub, y, ylab, outfile):
    fig, ax = plt.subplots(figsize=(7,4))
    mrg     = 0.10*(sub[y].max()-sub[y].min())
    ax.set_ylim(max(0, sub[y].min()-mrg), sub[y].max()+mrg)
    for col,(mdl,g) in zip(palette, sub.groupby("model")):
        ax.plot(g["K"], g[y], marker="o", label=mdl,
                color=col, linewidth=1.6)
    ax.set(xlabel="Top‑K genes", ylabel=ylab,
           title=f"{TITLE} · {ylab} vs K")
    ax.legend(frameon=False, bbox_to_anchor=(1.02,0.5), loc="center left")
    fig.tight_layout(rect=[0,0,0.78,1]); fig.savefig(outfile); plt.close(fig)

line_vs_k(df,"Train_mean","Mean train time [s]",snakemake.output.Train_mean)
line_vs_k(df,"Pred_mean" ,"Mean predict time [s]",snakemake.output.Pred_mean)
line_vs_k(df,"Train_total","Total train time [s]",snakemake.output.Train_total)
line_vs_k(df,"Pred_total" ,"Total predict time [s]",snakemake.output.Pred_total)
df["Runtime_total"]=df["Train_total"]+df["Pred_total"]
line_vs_k(df,"Runtime_total","Total runtime [s]",snakemake.output.Runtime_total)

# ----- wall‑clock & speed‑up
if len(cdf)>=2 and {"cores","wall_clock_s"}.issubset(cdf.columns):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(cdf["cores"], cdf["wall_clock_s"], marker="o")
    ax.set(xlabel="Snakemake cores", ylabel="Wall‑clock [s]",
           title=f"{TITLE} · wall‑clock seconds")
    ax.grid(axis="y",ls="--",alpha=.4); fig.tight_layout()
    fig.savefig(snakemake.output.Wall_clock_vs_cores); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(cdf["cores"], cdf["wall_clock_s"].iloc[0]/cdf["wall_clock_s"],
            marker="o")
    ax.set(xlabel="Snakemake cores", ylabel="Speed‑up ×",
           title=f"{TITLE} · wall‑clock speed‑up")
    ax.grid(axis="y",ls="--",alpha=.4); fig.tight_layout()
    fig.savefig(snakemake.output.Speed_up); plt.close(fig)
else:
    stub(snakemake.output.Wall_clock_vs_cores)
    stub(snakemake.output.Speed_up)

# ----- totals vs cores ------------------------------
def safe_totals(y, png):
    if y not in mdf.columns or "cores" not in mdf.columns:
        stub(png); return
    agg = mdf.groupby(["model","cores"],as_index=False)[y].sum(min_count=1)
    fig,ax=plt.subplots(figsize=(6,4))
    for col,(mdl,g) in zip(palette, agg.groupby("model")):
        ax.plot(g["cores"], g[y], marker="o", label=mdl, color=col)
    ax.set(xlabel="Snakemake cores", ylabel=y.replace("_"," "),
           title=f"{TITLE} · {y.replace('_',' ')}")
    ax.legend(frameon=False,bbox_to_anchor=(1.02,0.5),loc="center left")
    fig.tight_layout(rect=[0,0,0.78,1]); fig.savefig(png); plt.close(fig)

safe_totals("Train_total", snakemake.output.Train_total_vs_cores)
safe_totals("Pred_total" , snakemake.output.Pred_total_vs_cores)

# ----- zoomed scatter --------------------------------------------------------
def safe_zoom(y, png):
    if y not in mdf.columns or "cores" not in mdf.columns:
        stub(png); return
    fig,ax=plt.subplots(figsize=(6,4))
    jit={c:j for c,j in zip(sorted(mdf["cores"].unique()),
                            np.linspace(-.15,.15,len(mdf["cores"].unique())))}
    for col,(mdl,g) in zip(palette, mdf.groupby("model")):
        for _,r in g.iterrows():
            x=r["cores"]+jit[r["cores"]]
            ax.scatter(x,r[y],color=col,s=40,zorder=3)
            ax.text(x,r[y],f"K={r['K']}",fontsize=7,ha="center",va="bottom",
                    color=col)
        ax.plot(g["cores"],g[y],ls="--",lw=.7,color=col,alpha=.4)
    ax.set(xlabel="Snakemake cores", ylabel=y.replace("_"," "),
           title=textwrap.fill(f"{TITLE} · {y.replace('_',' ')} (all K points)",60))
    fig.tight_layout(); fig.savefig(png); plt.close(fig)

safe_zoom("Train_total", snakemake.output.Train_total_vs_cores_zoom)
