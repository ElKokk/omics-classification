#!/usr/bin/env python
"""
Create a *shuffled‑label* copy of an expression matrix
so you can test for information leakage.

Example
-------
python scripts/dev/shuffle_labels_once.py \
       data/raw/prostmat.csv \
       data/raw/prostmat_SHUFFLED.csv \
       --seed 42
"""
import pandas as pd
import numpy as np
import argparse
import pathlib

# ────────── CLI ────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("src", help="original matrix CSV")
p.add_argument("dst", help="destination CSV with shuffled labels")
p.add_argument("--seed", type=int, default=42, help="random seed")
args = p.parse_args()

rng = np.random.default_rng(args.seed)

# ────────── read matrix ────────────────────────────────────────────────────
df = pd.read_csv(args.src, header=None)
labels = df.iloc[0, 1:].astype(str)

# ────────── permute only the class tag portion of the label  ───────────────
mask = labels.str.contains("cancer|control", case=False, na=False)
shuffled = labels[mask].sample(frac=1, random_state=args.seed).values
labels.loc[mask] = shuffled
df.iloc[0, 1:] = labels

# ────────── write out ──────────────────────────────────────────────────────
pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.dst, index=False, header=False)
print("wrote", args.dst)
