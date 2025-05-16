"""
I collapse the split TSVs into one mean ± SE row per K
--------------------------------------------------------
Snakemake passes the list of metrics_k*.tsv files in
`snakemake.input`  and the desired output path in `snakemake.output[0]`.
"""

from pathlib import Path
import pandas as pd, numpy as np, re, sys, textwrap

out_fp  = Path(snakemake.output[0])
in_fps  = [Path(p) for p in snakemake.input]

rows = []
for fp in in_fps:
    K = int(re.search(r"metrics_k(\d+)\.tsv", fp.name).group(1))
    df = pd.read_csv(fp, sep="\t")

    def mean_se(col):
        return np.mean(col), np.std(col, ddof=1) / np.sqrt(len(col))

    mce_m, mce_se = mean_se(df["MCE"])
    sen_m, sen_se = mean_se(df["Sensitivity"])
    spe_m, spe_se = mean_se(df["Specificity"])

    rows.append(dict(
        K=K,
        MCE_mean=mce_m,  MCE_se=mce_se,
        Sens_mean=sen_m, Sens_se=sen_se,
        Spec_mean=spe_m, Spec_se=spe_se,
    ))

pd.DataFrame(rows).sort_values("K").to_csv(out_fp, sep="\t", index=False)
print(f"[aggregate] wrote {out_fp}", file=sys.stderr)
