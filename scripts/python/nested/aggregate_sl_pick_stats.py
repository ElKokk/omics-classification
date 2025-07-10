#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
For a group of SL‑weight TSVs (inner OR outer tier) compute

* picked_count   – weight>0 occurrences   per (model,K)
* topwinner_count– wins of highest weight per (model,K)

"""
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

picked_rows, top_rows = [], []

for fp in snakemake.input:
    df = pd.read_csv(fp, sep="\t")
    if df.empty or "model" not in df.columns:
        continue

    if "split" not in df.columns:
        df["split"] = df.get("outer", 0)
    if "K" not in df.columns:
        df["K"] = int(Path(fp).stem.split("_k")[-1])

    grp_split = df.groupby(["model", "split", "K"], as_index=False)["weight"] \
                  .sum()

    picked = (grp_split.query("weight > 0")
                        .groupby(["model","K"]).size()
                        .rename("picked_count"))

    top = (grp_split.sort_values("weight", ascending=False)
                    .drop_duplicates(["split","K"])
                    .groupby(["model","K"]).size()
                    .rename("topwinner_count"))

    picked_rows.append(picked.to_frame())
    top_rows.append(top.to_frame())

if picked_rows:
    picked_df = pd.concat(picked_rows).reset_index()
else:
    picked_df = pd.DataFrame(columns=["model","K","picked_count"])

if top_rows:
    top_df = pd.concat(top_rows).reset_index()
else:
    top_df = pd.DataFrame(columns=["model","K","topwinner_count"])

Path(snakemake.output.picked).parent.mkdir(parents=True, exist_ok=True)
picked_df.to_csv(snakemake.output.picked, sep="\t", index=False)
top_df   .to_csv(snakemake.output.top,    sep="\t", index=False)
logging.info("✓ SL pick stats → %s", Path(snakemake.output.picked).parent)
