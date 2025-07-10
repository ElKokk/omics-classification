#!/usr/bin/env python
# Pool INNER‑MCCV metrics across ALL outer folds  → one row per model,K
# ─────────────────────────────────────────────────────────────────────────
from pathlib import Path
import pandas as pd, numpy as np, sys, re

in_files = [Path(p) for p in snakemake.input]
out_file = Path(snakemake.output[0])
out_file.parent.mkdir(parents=True, exist_ok=True)

if not in_files:
    sys.exit("no inner metric files given")

frames = []
rx = re.compile(r"metrics_k(\d+)\.tsv$")
for fp in in_files:
    k = int(rx.search(fp.name).group(1))
    df = pd.read_csv(fp, sep="\t").assign(K=k)
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

agg = (df.groupby(["model", "K"])
         .agg(MCE_mean =("MCE",  "mean"),
              MCE_se   =("MCE",  lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Sens_mean=("Sensitivity","mean"),
              Sens_se  =("Sensitivity",lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Spec_mean=("Specificity","mean"),
              Spec_se  =("Specificity",lambda x: x.std(ddof=1)/np.sqrt(len(x))))
         .reset_index()
         .sort_values(["model","K"]))

agg.to_csv(out_file, sep="\t", index=False)
print(f"✓ inner   metrics → {out_file}")
