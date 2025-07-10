#!/usr/bin/env python
# -*- coding: utf‑8 -*-
"""
Aggregate SL‑weight tables from ALL outer folds
→   results/<ds>/nested/sl_weights_mean.tsv
      rows   = base‑learner
      cols   = K
      value  = mean weight across outer folds
"""
from pathlib import Path
import pandas as pd, sys, logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

rows = []
for fp in snakemake.input:
    df = pd.read_csv(fp, sep="\t")
    if {"model", "weight"}.issubset(df.columns):
        k = int(Path(fp).stem.split("_k")[-1])
        df["K"] = k
        rows.append(df[["model", "K", "weight"]])

if not rows:
    logging.error("no SL‑weight tables provided")
    sys.exit(1)

agg = (pd.concat(rows, ignore_index=True)
         .groupby(["model", "K"], as_index=False)["weight"]
         .mean())

heat = (agg.pivot(index="model", columns="K", values="weight")
           .sort_index(axis=1))

out = Path(snakemake.output[0])
out.parent.mkdir(parents=True, exist_ok=True)
heat.to_csv(out, sep="\t", float_format="%.4f")
logging.info("✓ sl_weights_mean.tsv → %s", out)
