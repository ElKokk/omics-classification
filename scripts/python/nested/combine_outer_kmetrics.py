#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concatenate all outer‑fold stage‑2 metric tables for one K.

• Ensures each row has column ‘split’ with the correct outer index.
• If the column already exists and is consistent → leave it untouched.
"""

from pathlib import Path
import pandas as pd, re, logging, sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")

out_fp = Path(snakemake.output[0])
frames = []

rx = re.compile(r"outer(\d+)/stage2/metrics_k")

for fp in snakemake.input:
    m = rx.search(fp)
    if not m:
        logging.warning("cannot extract outer index from %s – skipped", fp)
        continue
    outer_ix = int(m.group(1))

    try:
        df = pd.read_csv(fp, sep="\t")
    except Exception as exc:
        logging.warning("cannot read %s (%s) – skipped", fp, exc)
        continue
    if df.empty:
        logging.warning("empty metrics table skipped: %s", fp)
        continue

    # ---- guarantee 'split' column -----------------------------------------
    if "split" in df.columns:
        uniq = df["split"].unique()
        if len(uniq) != 1 or uniq[0] != outer_ix:
            logging.warning("%s has wrong 'split' (%s) – fixing", fp, uniq)
            df["split"] = outer_ix
    else:
        df.insert(0, "split", outer_ix)

    frames.append(df)

if not frames:
    logging.error("no stage‑2 metric tables for K=%s", snakemake.wildcards.K)
    sys.exit(1)

pd.concat(frames, ignore_index=True)\
  .sort_values(["split", "model"])\
  .to_csv(out_fp, sep="\t", index=False)

logging.info("✓ combined metrics → %s", out_fp)
