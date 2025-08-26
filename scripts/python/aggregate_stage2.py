#!/usr/bin/env python
"""
Aggregate Stage‑2  metrics_k*.tsv  →  summary_stage2.tsv

    MCE · Sensitivity · Specificity

"""
import pandas as pd
import pathlib
import sys
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

frames = []
for fp in snakemake.input:
    df = pd.read_csv(fp, sep="\t")
    k  = int(pathlib.Path(fp).stem.split("_k")[-1])
    df["K"] = k
    frames.append(df)

if not frames:
    logging.error("no Stage‑2 metric files found")
    sys.exit(1)

df = pd.concat(frames).sort_values("K")

# ─────── canonical column names ────────────────────────────────────────────
rename_map = {}
for original in df.columns:
    low = original.lower()
    if low in {"mce", "mce_mean"}:
        rename_map[original] = "MCE"
    elif low in {"sensitivity", "sens_mean", "sensitivity_mean"}:
        rename_map[original] = "Sensitivity"
    elif low in {"specificity", "spec_mean", "specificity_mean"}:
        rename_map[original] = "Specificity"

df.rename(columns=rename_map, inplace=True)

missing = [c for c in ["MCE", "Sensitivity", "Specificity"]
           if c not in df.columns]
if missing:
    logging.warning("Stage‑2 summary missing columns: %s",
                    ", ".join(missing))

# ─────── write outputs ────────────────────────────────────────────────────
main_fp, tagged_fp = snakemake.output
df.to_csv(main_fp, sep="\t", index=False)
pathlib.Path(tagged_fp).write_text(pathlib.Path(main_fp).read_text())
