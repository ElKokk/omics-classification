#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_inner_avg_k.py
Input : all   outer*/stage1/metrics_k{K}.tsv   for one dataset & one K
Output: table with one row per outer fold per learner (mean of m splits)
"""
from pathlib import Path
import pandas as pd, sys, re

out_tsv = Path(snakemake.output[0])
dfs = []

for fp in snakemake.input:
    m = re.search(r"outer(\d+)/", fp)
    O = int(m.group(1)) if m else -1
    df = pd.read_csv(fp, sep="\t")
    if df.empty or "model" not in df.columns:
        continue
    group = (df.groupby("model", as_index=False)[["MCE","Sensitivity","Specificity"]]
                .mean())
    group["O"] = O
    dfs.append(group)

if dfs:
    pd.concat(dfs, ignore_index=True).to_csv(out_tsv, sep="\t", index=False)
else:

    pd.DataFrame(columns=["model","MCE","Sensitivity","Specificity","O"]) \
      .to_csv(out_tsv, sep="\t", index=False)
