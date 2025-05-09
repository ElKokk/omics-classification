#!/usr/bin/env python
"""
Plot Stage‑1 results.

Inputs  (from Snakemake):
    snakemake.input.metrics   ← TSV with per‑split metrics
    snakemake.input.freq      ← CSV with gene selection counts
Outputs (from Snakemake):
    snakemake.output[...]     ← three PNGs

Params  (from Snakemake):
    snakemake.params.title    ← figure title (e.g. dataset | K info)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# ───────────────────────── helper ──────────────────────────
def make_figures(tab_path: Path, freq_path: Path, outdir: Path,
                 title: str) -> None:
    """Create three PNGs inside *outdir*."""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1 ▸ performance per split
    df = pd.read_table(tab_path)
    fig, ax = plt.subplots()
    for col in ["MCE", "Sensitivity", "Specificity"]:
        ax.plot(df["split"], df[col], marker=".", lw=.8, label=col)
    ax.set(xlabel="Monte‑Carlo split", ylabel="value", title=title)
    ax.legend()
    fig.savefig(outdir / "per_split.png", dpi=300, bbox_inches="tight")

    # 2 ▸ gene‑selection frequencies
    freq_df = pd.read_csv(freq_path).sort_values("count", ascending=False)
    top = freq_df.head(25)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=top, x="count", y="gene", orient="h")
    plt.xlabel("selection count (out of 100 splits)")
    plt.ylabel("gene")
    plt.title(f"{title}\ntop‑25 gene frequency")
    plt.tight_layout()
    plt.savefig(outdir / "gene_frequency.png", dpi=300)

    # 3 ▸ mean ± SE bar
    mean_se = df[["MCE", "Sensitivity", "Specificity"]].agg(
        ["mean", lambda x: x.std(ddof=1)/np.sqrt(len(x))]).T
    mean_se.columns = ["mean", "se"]
    plt.figure(figsize=(4, 3))
    plt.errorbar(mean_se.index, mean_se["mean"], yerr=mean_se["se"],
                 fmt="o", capsize=4)
    plt.ylabel("metric value")
    plt.ylim(0, 1)
    plt.title(f"{title}\nmean ± SE over 100 splits")
    plt.tight_layout()
    plt.savefig(outdir / "mean_se.png", dpi=300)


# ───────────────────────── entry point ─────────────────────
if __name__ == "__main__":
    # Snakemake passes everything we need
    metrics_tsv = Path(snakemake.input.metrics)
    freq_csv    = Path(snakemake.input.freq)
    # output[0] is …/per_split.png → its parent is the figure directory
    fig_dir     = Path(snakemake.output[0]).parent
    ttl         = snakemake.params.title

    make_figures(metrics_tsv, freq_csv, fig_dir, ttl)
    print(f"[done] Figures written to {fig_dir}")
