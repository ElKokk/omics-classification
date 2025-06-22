#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect the *max* wall‑clock across all K for **one** run_cores value
and write a tiny TSV with header:  cores  wall_clock_s
"""
import pathlib, sys

out_fp  = pathlib.Path(snakemake.output[0])
run_cores = int(snakemake.params["run_cores"])
src_fps   = snakemake.input

secs = []
for fp in src_fps:
    with open(fp) as fh:
        next(fh, None)
        row = next(fh, "0\t0").split("\t")[0]
        secs.append(float(row))

out_fp.parent.mkdir(parents=True, exist_ok=True)
with open(out_fp, "w") as fh:
    fh.write("cores\twall_clock_s\n")
    fh.write(f"{run_cores}\t{max(secs):.3f}\n")
