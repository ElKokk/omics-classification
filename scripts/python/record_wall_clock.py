#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect the *max* wall‑clock across all K for one run_cores value.
Accepts either pure seconds or H:MM:SS strings.
"""
import pathlib, re

out_fp     = pathlib.Path(snakemake.output[0])
run_cores  = int(snakemake.params["run_cores"])
src_fps    = snakemake.input 

def to_seconds(txt: str) -> float:
    txt = txt.strip()
    if re.fullmatch(r"[\d.]+", txt):
        return float(txt)
    parts = list(map(float, txt.split(":")))  
    if len(parts) == 2:   
        m, s = parts
        return m*60 + s
    if len(parts) == 3:   
        h, m, s = parts
        return h*3600 + m*60 + s
    raise ValueError(f"Cannot parse time value: {txt}")

secs = []
for fp in src_fps:
    with open(fp) as fh:
        next(fh, None)      
        row = next(fh, "0\t0").rstrip("\n").split("\t")
        secs.append(to_seconds(row[1]))    

out_fp.parent.mkdir(parents=True, exist_ok=True)
with out_fp.open("w") as fh:
    fh.write("cores\twall_clock_s\n")
    fh.write(f"{run_cores}\t{max(secs):.3f}\n")
