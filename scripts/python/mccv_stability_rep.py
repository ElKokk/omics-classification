#!/usr/bin/env python
"""
MCCV for Stability Reps (feature selection + aggregation only).
Adapter supports Golub / Grouped / Prostmat.
"""
from pathlib import Path
import numpy as np, pandas as pd, logging, re, random
from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate(); _ = packages.importr("limma", suppress_messages=True)

MATRIX_CSV  = Path(snakemake.input["matrix"])
OUT_PANEL   = Path(snakemake.output["panel"])
TOP_K       = int(snakemake.params["k"])
N_SPLITS    = int(snakemake.params["n_splits"])
REP         = int(snakemake.params["rep"])
DATASET     = MATRIX_CSV.stem.replace("_matrix", "")
BASE_SEED   = 42

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logging.info("Stability Rep %d | DATASET=%s  TOP_K=%d  SPLITS=%d", REP, DATASET, TOP_K, N_SPLITS)

def _map_group(vals):
    out=[]
    for v in vals:
        s=str(v).strip().lower()
        if s in {"1","true","case","cancer","tumor","tumour","crc"}: out.append("Cancer")
        elif s in {"0","false","control","normal","healthy"}: out.append("Control")
        else: out.append(str(v))
    return np.array(out, dtype=object)

def read_matrix_and_labels(fp: Path, dataset_name: str):
    if dataset_name.lower()=="golub":
        df=pd.read_csv(fp, header=0).transpose()
        df.columns=[f"sample_{i}" for i in range(1, df.shape[1]+1)]
        df.index=df.index.astype(str)
        expr=df.astype(float)
        n=expr.shape[1]
        if n==38: y=np.array(['ALL']*27+['AML']*11)
        elif n==72: y=np.array(['ALL']*27+['AML']*11+['ALL']*20+['AML']*14)
        else: raise ValueError(f"Unexpected n={n} for Golub")
        return expr, y, "golub"
    try:
        head=pd.read_csv(fp, nrows=5); label_col=next((c for c in head.columns if str(c).lower() in ("group","groups","label")), None)
    except Exception:
        label_col=None
    if label_col is not None:
        df=pd.read_csv(fp)
        sample_col=next((c for c in df.columns if str(c).lower() in ("samples","sample","id","sample_id") and c!=label_col), None)
        if sample_col is None: sample_col=next(c for c in df.columns if c!=label_col)
        feat_cols=[c for c in df.columns if c not in {label_col,sample_col}]
        expr=df[feat_cols].T
        expr.columns=df[sample_col].astype(str).values
        expr.index=expr.index.astype(str)
        expr=expr.astype(float)
        expr.index=expr.index.str.lstrip("X").str.replace(r"\.0$","",regex=True)
        expr=expr[~expr.index.duplicated(keep="first")]
        y=_map_group(df[label_col].values)
        return expr, y, "grouped"
    d=pd.read_csv(fp, header=None).drop(columns=[0])
    d.columns=d.iloc[0]
    expr=(d.iloc[1:].astype(float).set_index(d.index[1:].map(str)))
    expr.index=expr.index.str.lstrip("X").str.replace(r"\.0$","",regex=True)
    expr=expr[~expr.index.duplicated(keep="first")]
    samples=expr.columns.astype(str)
    y=np.where(pd.Series(samples).str.contains("cancer",case=False,na=False),"Cancer","Control")
    return expr, y, "prostmat"

def limma_topk(expr, classes, idx_train):
    positive="AML" if DATASET.lower()=="golub" else "Cancer"
    r.assign("mat_py", pandas2ri.py2rpy(expr.iloc[:, idx_train]))
    design=",".join("1" if classes[i]==positive else "0" for i in idx_train)
    r(f"design <- model.matrix(~ factor(c({design})))")
    r("fit <- eBayes(lmFit(mat_py, design))")
    tbl=r(f"topTable(fit, n={TOP_K}, sort.by='t')")
    return [re.sub(r"\.0$","", g.lstrip("X")) for g in tbl.index]

expr, classes, adapter = read_matrix_and_labels(MATRIX_CSV, DATASET)
samples = expr.columns.astype(str)
if len(np.unique(classes)) < 2:
    raise ValueError(f"Only one class found in labels for '{DATASET}'. Found: {dict(zip(*np.unique(classes, return_counts=True)))}")

rep_seed = BASE_SEED + REP; random.seed(rep_seed)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.33, random_state=rep_seed)

gene_cnt = {}
for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):
    top_genes = limma_topk(expr, classes, tr)
    for g in top_genes: gene_cnt[g] = gene_cnt.get(g, 0) + 1
    logging.info("Rep %d | Split %3d/%d | Limma done", REP, split_no, N_SPLITS)

OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
pd.Series(gene_cnt, name="count").sort_values(ascending=False)\
  .rename_axis("gene").reset_index()\
  .to_csv(OUT_PANEL.with_name(f"stability_rep{REP}_freq_k{TOP_K}.csv"), index=False)

robust_panel = pd.Series(gene_cnt).sort_values(ascending=False).head(TOP_K).index.astype(str)
OUT_PANEL.write_text("\n".join(robust_panel))
logging.info("Rep %d | ✓ frozen panel (%d genes) → %s", REP, TOP_K, OUT_PANEL.name)
