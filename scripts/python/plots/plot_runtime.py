"""
Extended runtime plots
––––––––––––––––––––––
Produces for every dataset

• Train_mean / Pred_mean / Train_total / Pred_total / Runtime_total  (model × K)
• Wall_clock                     – elapsed seconds vs K               (single)
• Speed_up                       – speed-up curve vs cores            (single)
• Wall_clock_vs_cores            – elapsed seconds vs cores           (single)
• Train_total_vs_cores           – TOTAL train time vs cores (all K, per model)
• Pred_total_vs_cores            – TOTAL predict time vs cores (all K, per model)
• Train_total_vs_cores_zoom      – *raw* train totals, all K pts annotated
• Train_total_vs_cores_fixedK    – totals vs cores **for fixed-K only**
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, textwrap

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300

# ────────── Snakemake I/O ───────────────────────────────────────────
stage1   = Path(snakemake.input.stage1_summary)
cores_fp = Path(snakemake.input.cores_table)
model_fp = Path(snakemake.input.model_table)

out      = snakemake.output
TITLE    = snakemake.params.title


try:
    FIXED_K = int(snakemake.params.fixed_k)
except AttributeError:
    # backward compatibility
    import statistics
    FIXED_K = statistics.mode(pd.read_csv(stage1, sep="\t")["K"])



Path(out.Train_mean).parent.mkdir(parents=True, exist_ok=True)

df   = pd.read_csv(stage1, sep="\t").sort_values("K")
cdf  = pd.read_csv(cores_fp, sep="\t").sort_values("cores")
mdf  = pd.read_csv(model_fp, sep="\t").sort_values(["model", "cores"])

palette = sns.color_palette("colorblind", df["model"].nunique())



# ───────── 1. helper – model lines vs K ─────────
def line_vs_k(sub, y, ylab, outfile):
    fig, ax = plt.subplots(figsize=(7, 4))
    for colour, (mdl, g) in zip(palette, sub.groupby("model")):
        ax.plot(g["K"], g[y], marker="o", label=mdl, color=colour)
    ax.set_xlabel("Top-K features")
    ax.set_ylabel(ylab)
    ax.set_title(f"{TITLE} · {ylab} vs K")
    ax.legend(frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


line_vs_k(df, "Train_mean",  "Train time [s]",           out.Train_mean)
line_vs_k(df, "Pred_mean",   "Predict time [s]",         out.Pred_mean)
line_vs_k(df, "Train_total", "Total train time [s]",     out.Train_total)
line_vs_k(df, "Pred_total",  "Total predict time [s]",   out.Pred_total)

df["Runtime_total"] = df["Train_total"] + df["Pred_total"]
line_vs_k(df, "Runtime_total", "Total runtime [s]",      out.Runtime_total)


# ───────── 2. wall-clock vs cores  + speed-up ─────────
if len(cdf) >= 2:
    # seconds
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cdf["cores"], cdf["wall_clock_s"], marker="o")
    ax.set_xlabel("Snakemake cores")
    ax.set_ylabel("Wall-clock [s]")
    ax.set_title(f"{TITLE} · wall-clock seconds")
    ax.grid(axis="y", ls="--", alpha=.4)
    sns.despine(fig); fig.tight_layout(); fig.savefig(out.Wall_clock_vs_cores)
    plt.close(fig)

    # speed-up
    base = cdf.iloc[0]["wall_clock_s"]
    sp   = cdf.assign(speed_up = base / cdf["wall_clock_s"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sp["cores"], sp["speed_up"], marker="o")
    ax.set_xlabel("Snakemake cores")
    ax.set_ylabel("Speed-up ×")
    ax.set_title(f"{TITLE} · wall-clock speed-up")
    ax.grid(axis="y", ls="--", alpha=.4)
    sns.despine(fig); fig.tight_layout(); fig.savefig(out.Speed_up)
    plt.close(fig)
else:
    Path(out.Wall_clock_vs_cores).touch(); Path(out.Speed_up).touch()


# ───────── 3. helper – TOTALS vs cores  ─────────
def totals_per_model(sub, y, ylab, outfile):
    agg = (sub.groupby(["model", "cores"], as_index=False)[y]
             .sum(min_count=1))         # total across all K

    fig, ax = plt.subplots(figsize=(6, 4))
    for colour, (mdl, g) in zip(palette, agg.groupby("model")):
        ax.plot(g["cores"], g[y], marker="o", label=mdl, color=colour)

    ax.set_xlabel("Snakemake cores")
    ax.set_ylabel(ylab)
    ax.set_title(f"{TITLE} · {ylab}")
    ax.legend(frameon=False)
    sns.despine(fig); fig.tight_layout(); fig.savefig(outfile); plt.close(fig)


totals_per_model(mdf, "Train_total", "Total train time [s]",
                 out.Train_total_vs_cores)
totals_per_model(mdf, "Pred_total",  "Total predict time [s]",
                 out.Pred_total_vs_cores)


# ───────── 4. all Ks-per-model-per-core plot – every K point annotated ─────────
def zoom_plot(sub, y, ylab, outfile):
    fig, ax = plt.subplots(figsize=(6, 4))
    x_jitter = {4: -0.12, 6: 0, 8: +0.12}

    for colour, (mdl, g) in zip(palette, sub.groupby("model")):
        for _, row in g.iterrows():
            x = row["cores"] + x_jitter.get(row["cores"], 0)
            ax.scatter(x, row[y], color=colour, s=35, zorder=3)
            ax.text(x, row[y], f"K={row['K']}", fontsize=7,
                    ha="center", va="bottom", color=colour)

        for K, cg in g.groupby("K"):
            ax.plot(cg["cores"], cg[y], ls="--", lw=0.7, color=colour, alpha=.5)

    ax.set_xlabel("Snakemake cores")
    ax.set_ylabel(ylab)
    ax.set_title(textwrap.fill(f"{TITLE} · {ylab} (all K annotated)", 60))
    sns.despine(fig); fig.tight_layout(); fig.savefig(outfile); plt.close(fig)


zoom_plot(mdf, "Train_total", "Total train time [s]",
          out.Train_total_vs_cores_zoom)


# ───────── 5. fixed-K only (Stage-2 panel) ─────────
fixed = mdf[mdf["K"] == FIXED_K]
if not fixed.empty:
    totals_per_model(fixed, "Train_total",
                     f"Total train time [s]  (K={FIXED_K})",
                     out.Train_total_vs_cores_fixedK)
else:
    Path(out.Train_total_vs_cores_fixedK).touch()
