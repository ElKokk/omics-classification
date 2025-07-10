#!/usr/bin/env python
from pathlib import Path
import pandas as pd, sys, logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")

out_fp = Path(snakemake.output[0])
frames = []

for csv in snakemake.input:
    try:
        df = pd.read_csv(csv)
        if len(df):
            frames.append(df)
        else:
            logging.warning("empty frequency table skipped: %s", csv)
    except Exception as exc:
        logging.warning("cannot read %s (%s) – skipped", csv, exc)

if not frames:
    logging.error("no outer‑fold freq tables for K=%s", snakemake.wildcards.K)
    sys.exit(1)

agg = (pd.concat(frames)
         .groupby("gene", as_index=False)["count"].sum())

agg.sort_values(["count", "gene"], ascending=[False, True])\
   .to_csv(out_fp, index=False)

logging.info("✓ combined freq → %s", out_fp)
