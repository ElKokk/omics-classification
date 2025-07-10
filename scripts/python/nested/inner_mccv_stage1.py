#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nested MCCV · INNER stage
* limma on INNER‑train → top‑K genes
* 7 base learners + SuperLearner
* writes
    – metrics_k*.tsv
    – freq_k*.csv
    – sl_weights_k*.tsv   (one row per base learner, every split)
"""
import time, logging
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics         import confusion_matrix
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import make_pipeline
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes     import GaussianNB
from sklearn.linear_model    import LogisticRegression as SkLogReg
from celer                   import LogisticRegression as CelLogReg

from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate(); _ = packages.importr("limma", suppress_messages=True)

from _io_utils import read_singh_matrix

# ─── Snakemake I/O -----------------------------------------------------------
mat_csv   = Path(snakemake.input["matrix"])
train_ids = Path(snakemake.input["train_ids"]).read_text().splitlines()

out_metrics = Path(snakemake.output["metrics"])
out_freq    = Path(snakemake.output["freq"])
out_slwts   = Path(snakemake.output["sl_wts"])

K_TOP   = int(snakemake.params["K"])
M_INNER = int(snakemake.params["m_inner"])
OUTERID = int(snakemake.params["O"])

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logging.info("outer‑%02d | K=%d | inner splits=%d", OUTERID, K_TOP, M_INNER)

# ─── data --------------------------------------------------------------------
expr_full = read_singh_matrix(mat_csv)
expr      = expr_full.loc[:, train_ids]

samples = expr.columns.astype(str)
classes = np.where(pd.Series(samples)
                     .str.contains("cancer", case=False, na=False),
                   "Cancer", "Control")

# ─── limma helper ------------------------------------------------------------
def limma_topk(idx_train: np.ndarray) -> list[str]:
    """Return top‑K gene IDs for the given INNER‑train indices."""
    df_tr = expr.iloc[:, idx_train]
    r.assign("mat_df", pandas2ri.py2rpy(df_tr))
    r("mat_py <- as.matrix(mat_df)")
    design = ",".join("1" if classes[i]=="Cancer" else "0" for i in idx_train)
    r(f"design <- model.matrix(~ factor(c({design})))")
    r("fit <- eBayes(lmFit(mat_py, design))")
    tbl = r(f"topTable(fit, n={K_TOP}, sort.by='t')")
    genes = []
    for rid in tbl.index:
        g = str(rid).lstrip("X")
        if g in expr.index:
            genes.append(g)
        elif g.isdigit():
            pos = int(g) - 1
            if 0 <= pos < len(expr.index):
                genes.append(expr.index[pos])
    if len(genes) < K_TOP:
        logging.warning("limma returned %d / %d valid genes (outer %d)",
                        len(genes), K_TOP, OUTERID)
    return genes

# ─── metric helper -----------------------------------------------------------
def split_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control","Cancer"]).ravel()
    return dict(
        MCE         = 1 - (tp+tn)/len(y_true),
        Sensitivity = tp/(tp+fn) if tp+fn else np.nan,
        Specificity = tn/(tn+fp) if tn+fp else np.nan,
    )

# ─── model zoo ---------------------------------------------------------------
STD = StandardScaler(with_mean=False)
svm_raw = make_pipeline(STD,
    LinearSVC(C=1., dual=False, max_iter=6000, class_weight="balanced"))
SVM_CAL = CalibratedClassifierCV(svm_raw, cv=5, method="sigmoid")

def fast_lasso():
    base = CelLogReg(penalty='l1', solver='celer-pn', fit_intercept=False,
                     max_iter=600, max_epochs=25000, tol=1e-4)
    C_grid = np.logspace(-2, 1, 12)
    return GridSearchCV(base, {'C':C_grid}, cv=5,
                        scoring='neg_log_loss', n_jobs=1, refit=True)

MODELS_FIL = {
    "LDA"             : LinearDiscriminantAnalysis(),
    "DLDA"            : GaussianNB(),
    "kNN"             : make_pipeline(STD, KNeighborsClassifier(
                                        n_neighbors=3, weights="distance")),
    "SVM"             : SVM_CAL,
    "RF"              : RandomForestClassifier(
                           n_estimators=400, max_features="sqrt",
                           random_state=1, n_jobs=1),
    "Lasso_Filtered"  : fast_lasso(),
}
LASSO_UNF = fast_lasso()

STACK_BASE = [
    ("rf" ,      MODELS_FIL["RF"]),
    ("svm",      MODELS_FIL["SVM"]),
    ("lassoF",   MODELS_FIL["Lasso_Filtered"]),
    ("lassoUF",  LASSO_UNF),
    ("knn",      MODELS_FIL["kNN"]),
    ("lda",      MODELS_FIL["LDA"]),
    ("dlda",     MODELS_FIL["DLDA"]),
]
BASE_NAMES = [n for n,_ in STACK_BASE]

STACK = StackingClassifier(
    estimators       = STACK_BASE,
    final_estimator  = SkLogReg(penalty="l1", solver="liblinear"),
    cv               = 5,
    n_jobs           = 1,
    stack_method     = "predict_proba",
)

# ─── inner MCCV loop ---------------------------------------------------------
sss = StratifiedShuffleSplit(n_splits=M_INNER, test_size=0.33,
                             random_state=K_TOP)

rows, gene_cnt, sl_rows = [], {}, []

for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):

    top = limma_topk(tr)
    for g in top:
        gene_cnt[g] = gene_cnt.get(g, 0) + 1

    Xtr_f = expr.loc[top,  samples[tr]].T.values
    Xte_f = expr.loc[top,  samples[te]].T.values
    Xtr_u = expr.iloc[:,   tr].T.values
    Xte_u = expr.iloc[:,   te].T.values
    ytr, yte = classes[tr], classes[te]

    # filtered models
    for name, clf in MODELS_FIL.items():
        t0 = time.perf_counter()
        clf.fit(Xtr_f, ytr)
        yhat = clf.predict(Xte_f)
        rows.append(split_metrics(yte, yhat) |
                    dict(split=split_no, model=name,
                         runtime_s=time.perf_counter()-t0))

    # unfiltered lasso
    t0 = time.perf_counter()
    LASSO_UNF.fit(Xtr_u, ytr)
    yhat = LASSO_UNF.predict(Xte_u)
    rows.append(split_metrics(yte, yhat) |
                dict(split=split_no, model="Lasso_Unfiltered",
                     runtime_s=time.perf_counter()-t0))

    # SuperLearner
    t0 = time.perf_counter()
    STACK.fit(Xtr_f, ytr)
    yhat = STACK.predict(Xte_f)
    rows.append(split_metrics(yte, yhat) |
                dict(split=split_no, model="SuperLearner",
                     runtime_s=time.perf_counter()-t0))

    # --- record SL weights  --------------------------------------------------
    meta_coef = STACK.final_estimator_.coef_.ravel()
    sl_rows.extend(dict(split=split_no, outer=OUTERID,
                        K=K_TOP, model=mdl, weight=w)
                   for mdl, w in zip(BASE_NAMES, meta_coef))

    if split_no % max(1, M_INNER//5) == 0 or split_no == M_INNER:
        logging.info("outer‑%02d  inner progress %d / %d",
                     OUTERID, split_no, M_INNER)

# ─── save outputs ------------------------------------------------------------
out_metrics.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_csv(out_metrics, sep="\t", index=False)
pd.Series(gene_cnt, name="count").rename_axis("gene") \
  .sort_values(ascending=False).to_csv(out_freq, index=True)
pd.DataFrame(sl_rows).to_csv(out_slwts, sep="\t", index=False)

logging.info("✓ inner metrics  → %s", out_metrics.name)
logging.info("✓ gene freq      → %s", out_freq.name)
logging.info("✓ SL weights     → %s", out_slwts.name)
