#### (SCRIPT DEPRECATED - NO LONGER RELEVANT)


import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random
from pathlib import Path


DATASET = snakemake.wildcards.ds
K = int(snakemake.wildcards.K)


pre_pkl = snakemake.input.all_top_pkl
freq_csv = snakemake.input.freq
matrix_csv = snakemake.input.matrix
seed_file = snakemake.input.seed
with open(seed_file, "r") as f:
    seed = int(f.read().strip())

def read_matrix(fp):
    df = pd.read_csv(fp, header=None).drop(columns=[0])
    df.columns = df.iloc[0]
    return (df.iloc[1:].astype(float)
              .set_index(df.index[1:].astype(str)))

expr = read_matrix(matrix_csv)
expr.index = expr.index.str.lstrip("X").str.replace(r"\.0$", "", regex=True)
expr = expr[~expr.index.duplicated(keep="first")]

classes = np.where(expr.columns.str.contains("cancer", case=False), "Cancer", "Control")

pre_lists = pickle.load(open(pre_pkl, "rb"))
freq_df = pd.read_csv(freq_csv, dtype={'gene': str}).sort_values("count", ascending=False)
aggregated_panel = freq_df["gene"].astype(str).head(K).tolist()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

def tanimoto(a, b):
    a_set, b_set = set(a), set(b)
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return intersection / union if union else 0

def compute_pairwise(lists, metric='tanimoto'):
    scores = []
    for p1, p2 in combinations(lists, 2):
        if metric == 'tanimoto':
            scores.append(tanimoto(p1, p2))
        elif metric == 'spearman':
            r1 = {g: i+1 for i, g in enumerate(p1)}
            r2 = {g: i+1 for i, g in enumerate(p2)}
            common = list(set(r1) & set(r2))
            if len(common) < 2: continue
            corr = spearmanr([r1[g] for g in common], [r2[g] for g in common])[0]
            scores.append(corr)
    return np.mean(scores) if scores else np.nan, np.std(scores) if scores else np.nan

def simulate_post_variability(expr, classes, aggregated_panel, n_sim=50, test_size=0.33, K=None, seed=42):
    """
    Simulation Details:
    - Generate n_sim fake MCCV-like splits on full data (train/test).
    - For each sim-split: Subset to aggregated_panel + sim-train samples.
    - Re-rank *only within the panel* by mean expression (proxy for diff expr; fast, no full Limma).
    - This mimics sampling effects on internal panel order without training models or using test data.
    - Take new top-K as "sim-panel" (despite re-ranking).
    - Compute Tanimoto to original aggregated_panel (overlap despite sim-sampling).
    - No models trained/evaluated—no performance metrics, just panel stability check.
    - Bias note: Reuses data, so optimistic; use for relative pre/post comparison.
    """
    if K is None:
        K = len(aggregated_panel)  # Compute at runtime
    random.seed(seed)
    sss = StratifiedShuffleSplit(n_splits=n_sim, test_size=test_size, random_state=seed)
    sim_scores = []
    for _, (tr, _) in enumerate(sss.split(expr.columns, classes)):
        tr_expr = expr.loc[aggregated_panel, expr.columns[tr]].T.values  # Panel + sim-train (no test used)
        means = np.mean(tr_expr, axis=0)  # Mean per gene in sim-train (proxy re-rank)
        sim_ranks = np.argsort(means)[::-1]  # Descending (like t-stat)
        sim_panel = [aggregated_panel[i] for i in sim_ranks[:K]]  # New "top-K" within panel
        sim_scores.append(tanimoto(aggregated_panel, sim_panel))  # Similarity to original
    return np.mean(sim_scores), np.std(sim_scores)


pre_tani_mean, pre_tani_std = compute_pairwise(pre_lists, 'tanimoto')
pre_spear_mean, pre_spear_std = compute_pairwise(pre_lists, 'spearman')
logging.info(f"Pre Tanimoto: Mean={pre_tani_mean:.3f}, Std={pre_tani_std:.3f}")
logging.info(f"Pre Spearman: Mean={pre_spear_mean:.3f}, Std={pre_spear_std:.3f}")


post_tani_mean, post_tani_std = simulate_post_variability(expr, classes, aggregated_panel, n_sim=50, K=K, seed=seed)
logging.info(f"Post Simulated Tanimoto: Mean={post_tani_mean:.3f}, Std={post_tani_std:.3f}")


results = pd.DataFrame({
    "Metric": ["Pre_Tanimoto_Mean", "Pre_Tanimoto_Std", "Pre_Spearman_Mean", "Pre_Spearman_Std",
               "Post_Tanimoto_Mean", "Post_Tanimoto_Std"],
    "Value": [pre_tani_mean, pre_tani_std, pre_spear_mean, pre_spear_std, post_tani_mean, post_tani_std]
})
results.to_csv(snakemake.output.tsv, sep="\t", index=False)
logging.info(f"Saved to {snakemake.output.tsv}")


metrics = ["Tanimoto"]
pre_means = [pre_tani_mean]
pre_stds = [pre_tani_std]
post_means = [post_tani_mean]
post_stds = [post_tani_std]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(x - width/2, pre_means, width, yerr=pre_stds, label='Pre-Agg', capsize=5)
ax.bar(x + width/2, post_means, width, yerr=post_stds, label='Post-Agg', capsize=5)
ax.set_ylabel('Value')
ax.set_title(f'Stability: Pre vs Post Means (K={K})')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.savefig(snakemake.output.bar)
logging.info(f"Saved bar to {snakemake.output.bar}")

# Boxplot: Pre Tanimoto Distribution
pre_tani_scores = [tanimoto(p1, p2) for p1, p2 in combinations(pre_lists, 2)]
fig, ax = plt.subplots(figsize=(6,6))
sns.boxplot(y=pre_tani_scores, ax=ax)
ax.set_title(f"Pre-Agg Tanimoto Distribution (K={K})")
ax.set_ylabel("Tanimoto Score")
plt.savefig(snakemake.output.box)
logging.info(f"Saved pre box to {snakemake.output.box}")