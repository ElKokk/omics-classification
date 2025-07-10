#!/usr/bin/env python
from pathlib import Path
import pandas as pd, sys, logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

rows = []
for fp in snakemake.input:
    df = pd.read_csv(fp, sep="\t")
    if {"cores","wall_clock_s"}.issubset(df.columns):
        rows.append(df.iloc[0])
    else:
        logging.warning("%s missing required cols – skipped", fp)

if not rows:
    sys.exit("no wall‑clock files")

merged = pd.DataFrame(rows).sort_values("cores")
Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(snakemake.output[0], sep="\t", index=False)
logging.info("✓ merged wall‑clock → %s", snakemake.output[0])
