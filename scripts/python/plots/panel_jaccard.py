#!/usr/bin/env python
"""
Compute pairwise Jaccard similarity between all frozen panels inside
results/<ds>/outer*/stage1/panel_k{K}.txt
"""
import pathlib, itertools, pandas as pd, argparse, numpy as np

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("k", type=int)
args = p.parse_args()

panels = {}
for fp in pathlib.Path("results", args.dataset).rglob(f"panel_k{args.k}.txt"):
    outer = fp.parts[-4]          # .../outer<n>/stage1/...
    panels[outer] = set(map(str.strip, open(fp)))
pairs = list(itertools.combinations(panels.keys(), 2))
rows=[]
for a,b in pairs:
    jac = len(panels[a] & panels[b]) / len(panels[a] | panels[b])
    rows.append(dict(outer_a=a, outer_b=b, jaccard=jac))
df = pd.DataFrame(rows)
print(df)
print("mean Jaccard =", df["jaccard"].mean().round(3))
