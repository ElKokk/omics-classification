import pandas as pd
import numpy as np
from itertools import combinations
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ───── Parameters from Snakemake ────────────────────────────────────────────
rep_fps = [Path(p) for p in snakemake.input.rep_panels] 
out_tsv = Path(snakemake.output.tsv) 

# Derive K from out_tsv stem 
k_str = out_tsv.stem.split("_k")[-1]
K = int(k_str)

# Derive original pkl path from out_tsv directory 
dataset = out_tsv.parent.parent.stem 
original_pkl = out_tsv.parent / f"all_top_genes_k{K}.pkl"

# Load original split-specific panels from derived pkl
with open(original_pkl, "rb") as f:
    original_splits = pickle.load(f)

# Load aggregated rep panels from txt
rep_panels = []
for fp in rep_fps:
    panel = [g.strip() for g in open(fp) if g.strip()]
    rep_panels.append(panel)

def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return intersection / union if union else 0

# Compute for original splits 
original_scores = [jaccard(p1, p2) for p1, p2 in combinations(original_splits, 2)]
original_mean = np.mean(original_scores) if original_scores else np.nan
original_std = np.std(original_scores) if original_scores else np.nan
logging.info(f"Original Splits Tanimoto: Mean={original_mean:.3f}, Std={original_std:.3f}")

# Compute for aggregated reps 
rep_scores = [jaccard(p1, p2) for p1, p2 in combinations(rep_panels, 2)]
rep_mean = np.mean(rep_scores) if rep_scores else np.nan
rep_std = np.std(rep_scores) if rep_scores else np.nan
logging.info(f"Aggregated Reps Tanimoto: Mean={rep_mean:.3f}, Std={rep_std:.3f}")

# Save to TSV with both
results = pd.DataFrame({
    "Metric": ["Original_Jaccard_Mean", "Original_Jaccard_Std", "Aggregated_Jaccard_Mean", "Aggregated_Jaccard_Std"],
    "Value": [original_mean, original_std, rep_mean, rep_std]
})
results.to_csv(out_tsv, sep="\t", index=False)
logging.info(f"Saved to {out_tsv}")