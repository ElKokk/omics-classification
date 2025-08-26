#!/usr/bin/env python
# merge_wall_clocks.py 
import sys, glob, pandas as pd, re
from pathlib import Path

def merge(files, out_tsv):
    if not files: 
        Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["cores", "wall_clock_s"]).to_csv(out_tsv,
                                                              sep="\t",
                                                              index=False)
        print("✓ no wall‑clock files – wrote empty table to", out_tsv)
        return

    rows = []
    for fp in files:
        df = pd.read_csv(fp, sep="\t")

        parent_name = Path(fp).parent.name
        m = re.search(r"cores(\d+|NA)", parent_name)
        cores_str = m.group(1) if m else "NA"
        cores = int(cores_str) if cores_str.isdigit() else -1
        df["cores"] = cores
        rows.append(df)

    pd.concat(rows, ignore_index=True).to_csv(out_tsv, sep="\t", index=False)
    print(f"✓ merged {len(rows)} files → {out_tsv}")

if "snakemake" in globals():
    merge(list(snakemake.input), snakemake.output[0])
else:     
    if len(sys.argv) != 3:
        sys.exit("Usage: merge_wall_clocks.py <glob> <out.tsv>")
    merge(glob.glob(sys.argv[1]), sys.argv[2])