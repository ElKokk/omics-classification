#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge results/*/wall_clock_<cores>.tsv into one table with header
cores\twall_clock_s
"""
import sys, pandas as pd, pathlib, logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

rows = []
for fp in snakemake.input:
    try:
        df = pd.read_csv(fp, sep="\t")
        if {"cores","wall_clock_s"}.issubset(df.columns) and len(df):
            rows.append(df.iloc[0])
        else:
            logging.warning("skipped malformed %s", fp)
    except Exception as exc:
        logging.warning("skipped %s (%s)", fp, exc)

out = pathlib.Path(snakemake.output[0])
out.parent.mkdir(parents=True, exist_ok=True)

if rows:
    pd.DataFrame(rows).sort_values("cores").to_csv(out, sep="\t", index=False)
else:

    out.write_text("cores\twall_clock_s\n")
    logging.warning("runtime_by_cores.tsv is empty")
