#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_nested.py
Combine the *outer‑TEST* metric tables (one per K) into a single summary
that contains mean and variance across the M outer folds.
"""

from pathlib import Path
import pandas as pd, re, sys

out_tsv = Path(snakemake.output[0])
out_tsv.parent.mkdir(parents=True, exist_ok=True)

rows = []
for fp in snakemake.input:
    try:
        df = pd.read_csv(fp, sep="\t")
    except Exception as e:
        sys.stderr.write(f"[aggregate_nested] cannot read {fp}: {e}\n")
        continue

    if df.empty or "model" not in df.columns:
        continue

    # ensure K column exists
    if "K" not in df.columns:
        m = re.search(r"_k(\d+)\.tsv$", fp)
        if m:
            df["K"] = int(m.group(1))
        else:
            sys.stderr.write(f"[aggregate_nested] cannot infer K for {fp}\n")
            continue

    # keep only expected cols
    keep = ["model", "K", "MCE", "Sensitivity", "Specificity"]
    df = df[[c for c in keep if c in df.columns]]
    rows.append(df)

if not rows:

    pd.DataFrame(columns=[
        "model","K",
        "MCE","Sensitivity","Specificity",
        "MCE_var","Sensitivity_var","Specificity_var"
    ]).to_csv(out_tsv, sep="\t", index=False)
    sys.exit()

df_all = pd.concat(rows, ignore_index=True)

grp = df_all.groupby(["model", "K"])
mean = grp[["MCE", "Sensitivity", "Specificity"]].mean()
var  = grp[["MCE", "Sensitivity", "Specificity"]].var(ddof=1) \
         .rename(columns=lambda c: f"{c}_var")

summary = mean.join(var).reset_index()
summary.to_csv(out_tsv, sep="\t", index=False)
