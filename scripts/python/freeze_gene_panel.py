#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Freeze a K‑gene panel from limma frequency table.
"""
from pathlib import Path
import pandas as pd, logging

freq_csv  = Path(snakemake.input[0])
out_panel = Path(snakemake.output.panel)
out_cnt   = Path(snakemake.output.counts)
K         = int(snakemake.params["k"])

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

tbl = (pd.read_csv(freq_csv)
         .sort_values(["count", "gene"], ascending=[False, True])
         .drop_duplicates("gene")
         .head(K))
if len(tbl) < K:
    logging.warning("%s: only %d/%d unique genes available", freq_csv, len(tbl), K)


out_panel.parent.mkdir(parents=True, exist_ok=True)
tbl["gene"].to_csv(out_panel, index=False, header=False)
tbl.to_csv(out_cnt, sep="\t", index=False, header=False)

logging.info("✓ frozen panel (%d genes) → %s", len(tbl), out_panel.name)
