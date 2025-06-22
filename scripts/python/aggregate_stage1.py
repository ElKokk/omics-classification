"""
Aggregate per‑split TSVs  → one row per (model, K)  + SE + totals.
"""
from pathlib import Path
import pandas as pd, numpy as np, re, sys, os, shutil

out_generic = Path(snakemake.output.summary)
out_tagged  = Path(snakemake.output.tagged)
in_fps      = [Path(p) for p in snakemake.input]

frames = []
rx = re.compile(r"metrics_k(\d+)\.tsv$")
for fp in in_fps:
    m = rx.search(fp.name)
    if m is None:
        sys.exit(f"[aggregate] cannot extract K from {fp}")
    K = int(m.group(1))
    frames.append(pd.read_csv(fp, sep="\t").assign(K=K))

df = pd.concat(frames, ignore_index=True)

agg = (df.groupby(["model", "K"])
         .agg(MCE_mean=("MCE", "mean"),
              MCE_se  =("MCE", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Sens_mean=("Sensitivity", "mean"),
              Sens_se  =("Sensitivity", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Spec_mean=("Specificity", "mean"),
              Spec_se  =("Specificity", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Train_mean=("train_s", "mean"),
              Train_se  =("train_s", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Pred_mean=("pred_s", "mean"),
              Pred_se   =("pred_s", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
              Train_total=("train_s", "sum"),
              Pred_total =("pred_s", "sum"))
         .reset_index()
         .sort_values(["model", "K"]))

out_generic.parent.mkdir(parents=True, exist_ok=True)
agg.to_csv(out_generic, sep="\t", index=False)
shutil.copy2(out_generic, out_tagged)

# ─── weight tables ───────────────────────────────────────────────────────
for fp in in_fps:
    w_fp = fp.with_suffix(".sl_weights.tsv")
    if w_fp.exists():
        os.utime(w_fp, None)

print(f"[aggregate] wrote → {out_generic}  and  {out_tagged}", file=sys.stderr)
