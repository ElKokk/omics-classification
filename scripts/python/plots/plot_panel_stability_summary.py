import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ───── Parameters from Snakemake ────────────────────────────────────────────
in_fps = [Path(p) for p in snakemake.input]  # List of all stability_reps_k{K}.tsv (expanded via config["sig_sizes"])
out_png = Path(snakemake.output.png)  # figures/{ds}/stage1/panel_stability_summary.png
out_tsv = Path(snakemake.output.tsv)  # results/{ds}/stage1/panel_stability_summary.tsv

frames = []
for fp in in_fps:
    df = pd.read_csv(fp, sep="\t")
    # Extract K from file name (e.g., stability_reps_k10.tsv -> 10)
    k = int(fp.stem.split("_k")[-1])
    df["K"] = k
    frames.append(df)

if not frames:
    logging.warning("No input TSVs found—creating empty outputs")
    pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.text(0.5, 0.5, "No data", ha='center')
    fig.savefig(out_png)
    sys.exit(0)

summary_df = pd.concat(frames).sort_values("K")

# Save aggregated summary TSV
summary_df.to_csv(out_tsv, sep="\t", index=False)
logging.info(f"Saved aggregated stability summary to {out_tsv}")

# Line Plot: Means/Std vs K for Original and Aggregated Jaccard
fig, ax = plt.subplots(figsize=(10,6))
unique_ks = summary_df["K"].unique()

# Function to get means/stds for a metric, using its own sub_ks
def get_metric_data(df, mean_metric, std_metric):
    sub_df_mean = df[df["Metric"] == mean_metric].groupby("K")["Value"].mean()
    sub_df_std = df[df["Metric"] == std_metric].groupby("K")["Value"].mean()
    sub_ks = sub_df_mean.index.values
    sub_means = sub_df_mean.values
    sub_stds = sub_df_std.values
    return sub_ks, sub_means, sub_stds

# Get and plot for Original Jaccard (blue line, shaded blue region)
orig_ks, orig_means, orig_stds = get_metric_data(summary_df, "Original_Jaccard_Mean", "Original_Jaccard_Std")
ax.plot(orig_ks, orig_means, label='Original Splits Tanimoto', color='blue')
ax.fill_between(orig_ks, orig_means - orig_stds, orig_means + orig_stds, color='blue', alpha=0.2)

# Get and plot for Aggregated Jaccard (red line, shaded red region)
agg_ks, agg_means, agg_stds = get_metric_data(summary_df, "Aggregated_Jaccard_Mean", "Aggregated_Jaccard_Std")
ax.plot(agg_ks, agg_means, label='Aggregated Reps Tanimoto', color='red')
ax.fill_between(agg_ks, agg_means - agg_stds, agg_means + agg_stds, color='red', alpha=0.2)

ax.set_xticks(unique_ks)
ax.set_xlabel('K', fontsize=14)
ax.set_ylabel('Mean Tanimoto Similarity', fontsize=14)
ax.set_title('Panel Stability vs K (Original Splits vs Aggregated Reps)', fontsize=16)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12)
fig.savefig(out_png)
logging.info(f"Saved summary line plot to {out_png}")