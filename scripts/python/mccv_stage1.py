"""
Stage‑1 ― Monte‑Carlo cross‑validation for the *prostmat* dataset
================================================================
• 100 stratified splits (2⁄3 train · 1⁄3 test)
• limma + moderated‑t ranking on the **train** fold
• keep top‑K genes, fit Linear Discriminant Analysis (LDA)
• save per‑split metrics + gene‑selection frequency
----------------------------------------------------------------
Outputs
  results/{ds}/stage1/metrics_k{K}.tsv   # 100 × 4 table
  results/{ds}/stage1/freq_k{K}.csv      # gene, count
"""
# ───── imports ──────────────────────────────────────────────────────────────
from pathlib import Path
import logging, re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate()

# ──────── file names handled by snakemake ────────────────────────────────
MATRIX_CSV  = Path(snakemake.input["matrix"])
OUT_METRICS = Path(snakemake.output["metrics"])
OUT_FREQ    = Path(snakemake.output["freq"])
TOP_K       = int(snakemake.params["k"])

# ──────── logging straight to the terminal ─────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logging.info("Stage‑1 | dataset=prostmat | K=%d", TOP_K)

# ───── 1.loading expression matrix ───────────────────────────────────────────
def read_prostmat(csv_path: Path) -> pd.DataFrame:
    """prostmat.csv → DataFrame (genes × samples)."""
    df = pd.read_csv(csv_path, header=None).drop(columns=[0])
    df.columns = df.iloc[0].tolist()          # sample names
    mat = df.iloc[1:].astype(float)           # numeric values
    mat.index = mat.index.map(str)            # gene ids as str
    return mat

expr = read_prostmat(MATRIX_CSV)

# normalizing row names
expr.index = (expr.index
                .str.lstrip("X")
                .str.replace(r"\.0$", "", regex=True))
expr = expr[~expr.index.duplicated()]

samples = expr.columns
classes = np.where(samples.str.contains("cancer", case=False), "Cancer", "Control")

# ───── 2.pre‑load limma
_ = packages.importr("limma", suppress_messages=True)

# ───── 3. Helpers
def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return {
        "MCE": 1 - (tp + tn) / len(y_true),
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
    }

# ───── 4. Monte‑Carlo CV loop ──────────────────────────────────────────────
gene_counter, rows = {}, []
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=42)

for split_no, (train_idx, test_idx) in enumerate(sss.split(samples, classes), 1):
    tr_samples, te_samples = samples[train_idx], samples[test_idx]

    # 4‑a limma ranking on training data
    r.assign("mat_py", pandas2ri.py2rpy(expr[tr_samples]))
    design_vec = ",".join("1" if classes[i] == "Cancer" else "0" for i in train_idx)
    r(f"design <- model.matrix(~ factor(c({design_vec})))")
    r("suppressMessages(fit <- eBayes(lmFit(mat_py, design)))")
    top_genes = [g.lstrip("X").removesuffix(".0")
                 for g in r(f"topTable(fit, n={TOP_K}, sort.by='t')").rownames]

    for g in top_genes:
        gene_counter[g] = gene_counter.get(g, 0) + 1

    # 4‑b LDA with those genes
    Xtr = expr.loc[top_genes, tr_samples].T.values
    Xte = expr.loc[top_genes, te_samples].T.values
    ytr, yte = classes[train_idx], classes[test_idx]

    clf = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    res = metrics(yte, clf.predict(Xte))
    res["split"] = split_no
    rows.append(res)

    logging.info("split %3d | MCE %.3f | Sens %.3f | Spec %.3f",
                 split_no, res["MCE"], res["Sensitivity"], res["Specificity"])

# ───── 5.wririting and the results ────────────────────────────────────────────────────
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_METRICS, sep="\t", index=False)

OUT_FREQ.parent.mkdir(parents=True, exist_ok=True)
(pd.Series(gene_counter, name="count")
   .sort_values(ascending=False)
   .rename_axis("gene")
   .reset_index()
   .to_csv(OUT_FREQ, index=False))

logging.info("Saved: %s  •  %s", OUT_METRICS, OUT_FREQ)
