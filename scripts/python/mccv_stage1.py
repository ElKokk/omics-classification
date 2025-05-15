"""
Stage-1 – Monte-Carlo cross-validation
=====================================
* 100 stratified splits (2/3 train · 1/3 test)
* limma + moderated-t ranking on the train fold
* keep top-K genes, fit Linear Discriminant Analysis
--------------------------------------------------------------------
Writes
    results/{ds}/stage1/metrics_k{K}.tsv   (one row per split)
    results/{ds}/stage1/freq_k{K}.csv      (gene, count)
"""
# ───────────────────── imports ───────────────────────────────────────
from pathlib import Path
import logging, re
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate()

# ────────────────── Snakemake I/O ────────────────────────────────────
MATRIX_CSV  = Path(snakemake.input["matrix"])
OUT_METRICS = Path(snakemake.output["metrics"])
OUT_FREQ    = Path(snakemake.output["freq"])
TOP_K       = int(snakemake.params["k"])

# ───────────────────── logging ───────────────────────────────────────
logging.basicConfig(format="%(levelname)s | %(message)s",
                    level=logging.INFO, force=True)
logging.info("Stage-1  |  dataset=prostmat  |  K=%d", TOP_K)

# ────────────────── 1 ▸ load expression matrix ──────────────────────
def read_prostmat(fp: Path) -> pd.DataFrame:
    """prostmat.csv – first col is empty, second row has sample IDs."""
    df = pd.read_csv(fp, header=None).drop(columns=[0])
    df.columns = df.iloc[0]                   # 101 samples
    mat = df.iloc[1:].astype(float)
    mat.index = mat.index.map(str)            # 6033 genes
    return mat

expr = read_prostmat(MATRIX_CSV)

# normalise gene IDs
expr.index = (expr.index
                .str.lstrip("X")
                .str.replace(r"\.0$", "", regex=True))
expr = expr[~expr.index.duplicated(keep="first")]

samples = expr.columns.to_numpy()
classes = np.where(
    np.char.find(np.char.lower(samples), "cancer") >= 0,
    "Cancer", "Control")

# ────────────────── 2 ▸ load R-package limma once ───────────────────
_ = packages.importr("limma", suppress_messages=True)

# ────────────────── 3 ▸ helper for metrics ──────────────────────────
def split_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return dict(
        MCE=1 - (tp + tn) / len(y_true),
        Sensitivity=tp / (tp + fn) if tp + fn else np.nan,
        Specificity=tn / (tn + fp) if tn + fp else np.nan
    )

# ────────────────── 4 ▸ Monte-Carlo loop ────────────────────────────
gene_counter, rows = {}, []
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=42)

for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):
    tr_cols, te_cols = samples[tr], samples[te]

    # ---- 4-a limma ranking on training fold -------------------------
    r.assign("mat_py", pandas2ri.py2rpy(expr[tr_cols]))
    design = ",".join("1" if classes[i] == "Cancer" else "0" for i in tr)
    r(f"design <- model.matrix(~ factor(c({design})))")
    r("suppressMessages(fit <- eBayes(lmFit(mat_py, design)))")

    # pandas ≥ 2.0 → rownames became the DataFrame index ⬇︎
    df_top = r(f"topTable(fit, n={TOP_K}, sort.by='t')")
    top_genes = [re.sub(r"\.0$", "", g.lstrip("X")) for g in df_top.index]

    for g in top_genes:
        gene_counter[g] = gene_counter.get(g, 0) + 1

    # ---- 4-b LDA on top-K genes ------------------------------------
    Xtr = expr.loc[top_genes, tr_cols].T.values
    Xte = expr.loc[top_genes, te_cols].T.values
    ytr, yte = classes[tr], classes[te]

    clf = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    res = split_metrics(yte, clf.predict(Xte))
    res["split"] = split_no
    rows.append(res)

    logging.info("split %3d | MCE %.3f | Sens %.3f | Spec %.3f",
                 split_no, res["MCE"], res["Sensitivity"], res["Specificity"])

# ────────────────── 5 ▸ write outputs ───────────────────────────────
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_METRICS, sep="\t", index=False)

OUT_FREQ.parent.mkdir(parents=True, exist_ok=True)
(pd.Series(gene_counter, name="count")
   .sort_values(ascending=False)
   .rename_axis("gene")
   .reset_index()
   .to_csv(OUT_FREQ, index=False))

logging.info("✓  Saved  %s  and  %s", OUT_METRICS, OUT_FREQ)
