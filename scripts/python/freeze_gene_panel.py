#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a **‑K** gene panel from Stage‑1 frequency table.

Outputs
  • panel_k{K}.txt          – 1 gene per line
  • panel_k{K}_counts.txt   – gene \t count
"""
from pathlib import Path
import pandas as pd, logging

freq_csv = Path(snakemake.input[0])
out_txt  = Path(snakemake.output[0])
K        = int(snakemake.params["k"])

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")

tbl = (pd.read_csv(freq_csv)
         .sort_values(["count", "gene"], ascending=[False, True])
         .head(K))

out_txt.parent.mkdir(parents=True, exist_ok=True)
tbl["gene"].to_csv(out_txt, index=False, header=False)

# ----keep counts alongside genes --------------------------------------
counts_txt = out_txt.with_name(out_txt.stem + "_counts.txt")
tbl.to_csv(counts_txt, sep="\t", index=False, header=False)

logging.info("✓ frozen panel (%d genes) → %s  (+ counts)",
             K, out_txt.name)