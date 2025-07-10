#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sum wall‑clock seconds recorded by Snakemake benchmark (.txt) files.
"""
from pathlib import Path

total = 0.0
for fp in snakemake.input:
    try:
        with open(fp) as fh:
            sec = float(fh.readline().strip().split()[0])
            total += sec
    except Exception:
        pass

out = Path(snakemake.output[0])
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as fh:
    fh.write("total_seconds\n")
    fh.write(f"{total:.3f}\n")
print(f"✓ runtime summary → {out}")
