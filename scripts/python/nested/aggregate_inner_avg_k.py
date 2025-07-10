#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_inner_avg_k.py

For one dataset & one K:
    Inputs : results/<ds>/outer*/stage1/metrics_k{K}.tsv   (10 files)
    Output : results/<ds>/nested/inner_avg_metrics_k{K}.tsv

The output contains **one row per outer fold per learner**:
    columns = model, MCE, Sensitivity, Specificity, O, K

Those rows are the average of the minner MCCV splits.
"""
from pathlib import Path
import pandas as pd, re, sys

out_tsv = Path(snakemake.output[0])
out_tsv.parent.mkdir(parents=True, exist_ok=True)

rows = []
for fp in snakemake.input:
    m = re.search(r"outer(\d+)/", fp)
    O = int(m.group(1)) if m else -1
    try:
        df = pd.read_csv(fp, sep="\t")
    except Exception:
        continue
    if df.empty or "model" not in df.columns:
        continue
    grp = (df.groupby("model", as_index=False)[["MCE", "Sensitivity", "Specificity"]]
             .mean())
    grp["O"] = O

    try:
        k = int(Path(fp).stem.split("_k")[-1])
    except Exception:
        k = -1
    grp["K"] = k
    rows.append(grp)

if rows:
    pd.concat(rows, ignore_index=True).to_csv(out_tsv, sep="\t", index=False)
else:

    pd.DataFrame(columns=["model","MCE","Sensitivity","Specificity","O","K"]) \
      .to_csv(out_tsv, sep="\t", index=False)
