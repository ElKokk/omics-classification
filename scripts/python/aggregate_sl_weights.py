#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute the average Super‑Learner weight per base‑learner and per K.

Input :
    a list of *.tsv produced by Stage‑1 or Stage‑2

Accepted file layouts
---------------------
* long  : columns  [split, model, weight]
* long  : columns  [split, base,  weight]
* wide  : index column (split) + one numeric column per model

Output
------
A matrix   rows = model,  columns = K,  values = mean weight
"""
from pathlib import Path
import pandas as pd
import re
import sys

out_fp = Path(snakemake.output[0])
out_fp.parent.mkdir(parents=True, exist_ok=True)

frames = []

for fp in snakemake.input:
    m = re.search(r"_k(\d+)\.tsv$", fp)
    if m is None:
        sys.exit(f"[aggregate_sl_weights] unexpected file name: {fp}")
    K = int(m.group(1))

    df = pd.read_csv(fp, sep="\t")

    # ── detect layout ───────────────────────────────────────────────────────
    if {"model", "weight"}.issubset(df.columns):
        long_df = df[["model", "weight"]]
    elif {"base", "weight"}.issubset(df.columns):
        long_df = (df.rename(columns={"base": "model"})
                     [["model", "weight"]])
    else:

        long_df = (df.set_index(df.columns[0])
                     .melt(ignore_index=False,
                           var_name="model",
                           value_name="weight"))

    long_df = (long_df.groupby("model", as_index=False)["weight"]
                      .mean()
                      .assign(K=K))
    frames.append(long_df)

# ── combine and write ────────────────────────────────────────────────────────
(pd.concat(frames)
     .pivot(index="model", columns="K", values="weight")
     .sort_index()
     .to_csv(out_fp, sep="\t"))
