#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write results/<ds>/runtime/wall_clock_<CORES>.tsv
  cores  wall_clock_s
"""
from pathlib import Path
import pandas as pd, re, statistics

out_fp   = Path(snakemake.output[0])
cores    = int(snakemake.params["run_cores"])
pattern  = re.compile(r"wall_clock_k\d+\.txt$")

totals = []

for fp in snakemake.input:
    if fp.endswith(".tsv"):
        df = pd.read_csv(fp, sep="\t")
        if {"Train_total", "Pred_total"}.issubset(df.columns):
            totals.append(df["Train_total"].iloc[0] + df["Pred_total"].iloc[0])
    elif pattern.search(fp):
        try:
            with open(fp) as fh:
                secs = float(fh.readline().strip().split()[0])
                totals.append(secs)
        except Exception:
            pass

wall = max(totals) if totals else 0.0
out_fp.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"cores":[cores], "wall_clock_s":[wall]}).to_csv(out_fp, sep="\t", index=False)
print(f"[record_wall_clock] cores={cores}  wall_clock={wall:.3f}s  → {out_fp}")
