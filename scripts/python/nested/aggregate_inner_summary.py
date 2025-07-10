#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate GRAND‑INNER averages across the M outer folds and add variance.

Input :  results/<ds>/nested/inner_avg_metrics_k*.tsv  (one per K)
Output:  results/<ds>/nested/inner_summary.tsv
         columns = model  K  MCE  Sensitivity  Specificity
                   MCE_var  Sensitivity_var  Specificity_var
"""
from pathlib import Path
import pandas as pd, sys

out_tsv = Path(snakemake.output[0])
out_tsv.parent.mkdir(parents=True, exist_ok=True)

frames = [pd.read_csv(fp, sep="\t") for fp in snakemake.input]
if not frames:
    pd.DataFrame(columns=[
        "model","K","MCE","Sensitivity","Specificity",
        "MCE_var","Sensitivity_var","Specificity_var"
    ]).to_csv(out_tsv, sep="\t", index=False)
    sys.exit()

df = pd.concat(frames, ignore_index=True)

grp = df.groupby(["model", "K"])
mean = grp[["MCE", "Sensitivity", "Specificity"]].mean()
var  = grp[["MCE", "Sensitivity", "Specificity"]].var(ddof=1) \
        .rename(columns=lambda c: f"{c}_var")

result = mean.join(var).reset_index()
result.to_csv(out_tsv, sep="\t", index=False)
