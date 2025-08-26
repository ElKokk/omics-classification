#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate SL weights for Stage 1: Count positive and highest per base per K.
"""
from pathlib import Path
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

positive_out = Path(snakemake.output["positive"])
highest_out = Path(snakemake.output["highest"])
in_fps = [Path(p) for p in snakemake.input]

positive_frames = []
highest_frames = []

for fp in in_fps:
    m = re.search(r"_k(\d+)\.tsv$", fp.name)
    if m is None:
        raise ValueError(f"Cannot extract K from {fp}")
    K = int(m.group(1))
    df = pd.read_csv(fp, sep="\t")
    
    if "base" in df.columns:
        df = df.rename(columns={"base": "model"})
    
    positive = (df[df["weight"] > 0]
                .groupby("model")
                .size()
                .reset_index(name="count")
                .assign(K=K))
    positive_frames.append(positive)
    
    highest_per_split = df.loc[df.groupby("split")["weight"].idxmax()]
    highest = (highest_per_split.groupby("model")
               .size()
               .reset_index(name="count")
               .assign(K=K))
    highest_frames.append(highest)

pd.concat(positive_frames).to_csv(positive_out, sep="\t", index=False)
logging.info("✓ Positive counts → %s", positive_out.name)

pd.concat(highest_frames).to_csv(highest_out, sep="\t", index=False)
logging.info("✓ Highest counts → %s", highest_out.name)