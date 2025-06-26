#!/usr/bin/env python
"""
Create a *shuffled‑label* copy of the expression matrix for leakage testing.

Usage
-----
python shuffle_labels_once.py  data/raw/prostmat.csv \
       data/raw/prostmat_SHUFFLED.csv  [--seed 42]
"""
import pandas as pd, numpy as np, argparse, pathlib, random

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
df = pd.read_csv(args.src, header=None)
labels = df.iloc[0, 1:].astype(str)

# permute 'cancer'/'control' tags only
mask = labels.str.contains("cancer|control", case=False, na=False)
perm = labels[mask].sample(frac=1, random_state=args.seed).values
labels.loc[mask] = perm
df.iloc[0, 1:] = labels

pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.dst, index=False, header=False)
print("✓ wrote", args.dst)
