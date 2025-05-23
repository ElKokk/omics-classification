"""
Stage-1 aggregator – combine metrics_k*.tsv into summary_stage1.tsv
------------------------------------------------------------------
For each classifier & K it computes mean and SE of MCE, Sensitivity,
Specificity across the 100 Monte-Carlo splits.
"""

from pathlib import Path
import pandas as pd
import numpy as np

out_fp = Path(snakemake.output[0])
rows   = []

def mean_se(arr):
    arr = np.asarray(arr, float)
    return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))

for fp in map(Path, snakemake.input):
    K  = int(fp.stem.split('_k')[1])
    df = pd.read_csv(fp, sep='\t')
    for model, grp in df.groupby('model'):
        r = {'K': K, 'model': model}
        for metric in ['MCE', 'Sensitivity', 'Specificity']:
            mu, se = mean_se(grp[metric])
            r[f'{metric}_mean'] = mu
            r[f'{metric}_se']   = se
        rows.append(r)

(pd.DataFrame(rows)
   .sort_values(['K', 'model'])
   .to_csv(out_fp, sep='\t', index=False))
