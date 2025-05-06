"""
Stage‑1 Monte‑Carlo CV for prostmat
──────────────────────────────────
* 100 stratified 2/3‑1/3 splits
* limma ranking on train fold
* Linear Discriminant Analysis
* Outputs
    ├─ results/{ds}/stage1/metrics_k{K}.tsv
    └─ results/{ds}/stage1/freq_k{K}.csv      (gene, count)
"""

# ----------------- imports ---------------------------------------------------
from pathlib import Path
import logging, re
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate()

# ----------------- Snakemake I/O ---------------------------------------------
mat_path    = snakemake.input["matrix"]
metrics_tsv = snakemake.output["metrics"]
freq_csv    = snakemake.output["freq"]
K           = int(snakemake.params["k"])

# ----------------- logging ----------------------------------------------------
logging.basicConfig(format="%(levelname)s | %(message)s",
                    level=logging.INFO, force=True)
logging.info("Stage‑1  |  dataset=prostmat  |  K=%d", K)

# ----------------- loader -----------------------------------------------------
def load_expression(fp: str | Path) -> pd.DataFrame:

    df = pd.read_csv(fp, header=None).drop(columns=[0])
    df.columns = df.iloc[0, :].tolist()
    mat = df.iloc[1:, :].astype(float)
    mat.index = [str(i) for i in range(mat.shape[0])]
    return mat

expr = load_expression(mat_path)

def norm_id(s: str) -> str:

    return re.sub(r"\.0$", "", str(s).lstrip("X"))

expr.index = expr.index.map(norm_id)
expr = expr[~expr.index.duplicated(keep="first")]

samples = expr.columns
classes = np.array(["Cancer" if "cancer" in s.lower() else "Control"
                    for s in samples])

# ----------------- limma ----------------------------------------------
limma = packages.importr("limma", suppress_messages=True)

# ----------------- helpers ----------------------------------------------------
def mce_sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    mce  = 1 - (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return mce, sens, spec

# ----------------- Monte‑Carlo loop ------------------------------------------
metric_rows, gene_counter = [], {}
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=42)

for i, (tr, te) in enumerate(sss.split(samples, classes), 1):
    tr_cols, te_cols = samples[tr], samples[te]

    # ---------- limma ranking on train fold ----------------------------------
    r.assign("mat_py", pandas2ri.py2rpy(expr[tr_cols]))
    design = ",".join(["1" if classes[j] == "Cancer" else "0" for j in tr])
    r(f"design <- model.matrix(~ factor(c({design})))")

    r("""
        suppressMessages({
            fit <- eBayes(lmFit(mat_py, design))
        })
    """)
    top_genes = [norm_id(g) for g in list(
        r(f"topTable(fit, n={K}, sort.by='t')").rownames)]

    for g in top_genes:
        gene_counter[g] = gene_counter.get(g, 0) + 1

    # ---------- LDA ----------------------------------------------------------
    Xtr = expr.loc[top_genes, tr_cols].T.values
    Xte = expr.loc[top_genes, te_cols].T.values
    ytr, yte = classes[tr], classes[te]

    mce, sens, spec = mce_sens_spec(yte,
                                    LinearDiscriminantAnalysis()
                                    .fit(Xtr, ytr).predict(Xte))

    metric_rows.append({"split": i, "MCE": mce,
                        "Sensitivity": sens, "Specificity": spec})
    logging.info("split %3d | MCE=%.3f  Sens=%.3f  Spec=%.3f",
                 i, mce, sens, spec)


Path(metrics_tsv).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(metric_rows).to_csv(metrics_tsv, sep="\t", index=False)

freq_df = (pd.Series(gene_counter, name="count")
             .sort_values(ascending=False)
             .reset_index().rename(columns={"index": "gene"}))
Path(freq_csv).parent.mkdir(parents=True, exist_ok=True)
freq_df.to_csv(freq_csv, index=False)

logging.info("Stage‑1 finished  →  %s  |  %s", metrics_tsv, freq_csv)
