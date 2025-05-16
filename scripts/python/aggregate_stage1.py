# plotting per model and k
from pathlib import Path
import pandas as pd, numpy as np, re, sys

out_fp  = Path(snakemake.output[0])
rows    = []

for fp in map(Path, snakemake.input):
    K   = int(re.search(r"metrics_k(\d+)\.tsv", fp.name).group(1))
    df  = pd.read_csv(fp, sep="\t")
    for mdl, grp in df.groupby("model"):
        def mean_se(x): return x.mean(), x.std(ddof=1)/np.sqrt(len(x))
        r = {"K": K, "model": mdl}
        for col in ["MCE", "Sensitivity", "Specificity"]:
            mu, se   = mean_se(grp[col])
            r[f"{col}_mean"] = mu
            r[f"{col}_se"]   = se
        rows.append(r)

(pd.DataFrame(rows)
   .sort_values(["K", "model"])
   .to_csv(out_fp, sep="\t", index=False))

print(f"[aggregate] wrote {out_fp}", file=sys.stderr)
