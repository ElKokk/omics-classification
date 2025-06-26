#!/usr/bin/env python
"""
Quick report: variance across OUTER folds vs INNER splits
"""
import pandas as pd, argparse, pathlib, numpy as np

p = argparse.ArgumentParser()
p.add_argument("dataset")
args = p.parse_args()

glob = pathlib.Path("results", args.dataset).rglob("stage2/summary_stage2.tsv")
frames=[]
for fp in glob:
    outer = fp.parts[-4]          # …/outer<n>/stage2/…
    frames.append(pd.read_csv(fp, sep="\t").assign(outer=outer))

df = pd.concat(frames)
for metric in ["MCE","Sensitivity","Specificity"]:
    g = (df.groupby(["outer","model"], as_index=False)[metric].mean()
            .groupby("outer")[metric].mean())
    print(f"{metric:12s}  outer‑fold mean = {g.mean():.3f}  "
          f"st.dev across outers = {g.std():.3f}")
