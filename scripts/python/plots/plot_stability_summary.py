import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ───── Parameters from Snakemake ────────────────────────────────────────────
in_fps = [Path(p) for p in snakemake.input]  # List of all stability_k{K}.tsv (expanded via config["sig_sizes"])
out_png = Path(snakemake.output.png)  # figures/{ds}/stage1/stability_summary.png
out_tsv = Path(snakemake.output.tsv)  # results/{ds}/stage1/stability_summary.tsv

frames = []
for fp in in_fps:
    df = pd.read_csv(fp, sep="\t")
    # Extract K from file name (e.g., stability_k10.tsv -> 10)
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

# Normalize Metric to lower for robust matching
summary_df["Metric"] = summary_df["Metric"].str.lower()

# Save aggregated summary TSV (original casing)
summary_df.to_csv(out_tsv, sep="\t", index=False)
logging.info(f"Saved aggregated stability summary to {out_tsv}")

# Line Plot: Means/Std vs K (Tanimoto example)
fig, ax = plt.subplots(figsize=(10,6))
unique_ks = summary_df["K"].unique()
logging.info(f"Unique K values: {unique_ks}")

# Function to get means/stds for a metric, using its own sub_ks (avoid size mismatch)
def get_metric_data(df, metric_name):
    sub_df = df[df["Metric"] == metric_name.lower()]
    if sub_df.empty:
        logging.warning(f"No data for {metric_name} across any K—skipping line")
        return None, None, None
    grouped_mean = sub_df.groupby("K")["Value"].mean()
    grouped_std = sub_df.groupby("K")["Value"].std().fillna(0)  # std=0 if single value
    sub_ks = grouped_mean.index.values
    sub_means = grouped_mean.values
    sub_stds = grouped_std.values
    logging.info(f"{metric_name}: sub_ks={sub_ks} (len={len(sub_ks)}), means len={len(sub_means)}, stds len={len(sub_stds)}")
    return sub_ks, sub_means, sub_stds

# Get and plot for Pre Tanimoto
pre_ks, pre_means, pre_stds = get_metric_data(summary_df, "pre_tanimoto_mean")
if pre_ks is not None:
    ax.plot(pre_ks, pre_means, label='Pre Tanimoto', color='blue')
    ax.fill_between(pre_ks, pre_means - pre_stds, pre_means + pre_stds, color='blue', alpha=0.2)

# Get and plot for Post Tanimoto
post_ks, post_means, post_stds = get_metric_data(summary_df, "post_tanimoto_mean")
if post_ks is not None:
    ax.plot(post_ks, post_means, label='Post Tanimoto', color='red')
    ax.fill_between(post_ks, post_means - post_stds, post_means + post_stds, color='red', alpha=0.2)

ax.set_xticks(unique_ks)
ax.set_xlabel('K', fontsize=14)
ax.set_ylabel('Mean Value', fontsize=14)
ax.set_title('Stability Metrics vs K', fontsize=16)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12)
fig.savefig(out_png)
logging.info(f"Saved summary line plot to {out_png}")