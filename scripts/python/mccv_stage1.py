#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑1 – Monte‑Carlo CV benchmark
• limma feature filter (top‑K t‑statistics)
• model zoo
• LASSO via celer.LogisticRegressionCV
• Super‑Learner now includes                     – RF / SVM / kNN / LDA / DLDA
                                                  – Lasso_Filtered
                                                  – Lasso_Unfiltered
• exports
    ◦ per‑split metrics
    ◦ gene‑frequencies
    ◦ Super‑Learner weights
"""
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd, logging, re, time, json
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from celer import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from rpy2.robjects import r, pandas2ri, packages

pandas2ri.activate()
_ = packages.importr("limma", suppress_messages=True)

# ─────────── I/O from Snakemake ────────────────────────────────────────────
MATRIX_CSV  = Path(snakemake.input["matrix"])
OUT_METRICS = Path(snakemake.output["metrics"])
OUT_FREQ    = Path(snakemake.output["freq"])
OUT_WTS     = OUT_METRICS.with_suffix(".sl_weights.tsv")
TOP_K       = int(snakemake.params["k"])
N_SPLITS    = int(snakemake.params["n_splits"])
DATASET     = MATRIX_CSV.stem.replace("_matrix", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logging.info("DATASET=%s  TOP_K=%d  SPLITS=%d", DATASET, TOP_K, N_SPLITS)

# ─────────── helpers ───────────────────────────────────────────────────────
def read_matrix(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, header=None).drop(columns=[0])
    df.columns = df.iloc[0]
    return (df.iloc[1:].astype(float)
              .set_index(df.index[1:].astype(str)))

def limma_topk(idx_train):
    r.assign("mat_py", pandas2ri.py2rpy(expr.iloc[:, idx_train]))
    design = ",".join("1" if classes[i]=="Cancer" else "0" for i in idx_train)
    r(f"design <- model.matrix(~ factor(c({design})))")
    r("fit <- eBayes(lmFit(mat_py, design))")
    tbl = r(f"topTable(fit, n={TOP_K}, sort.by='t')")
    return [re.sub(r"\\.0$", "", g.lstrip("X")) for g in tbl.index]

def split_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return dict(MCE=1 - (tp + tn) / len(y_true),
                Sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                Specificity=tn / (tn + fp) if tn + fp else np.nan)

# ─────────── data & labels ────────────────────────────────────────────────
expr = read_matrix(MATRIX_CSV)
expr.index = (expr.index.str.lstrip("X")
                        .str.replace(r"\\.0$", "", regex=True))
expr = expr[~expr.index.duplicated(keep="first")]

samples = expr.columns.astype(str)
classes = np.where(pd.Series(samples).str.contains("cancer", case=False, na=False),
                   "Cancer", "Control")

# ─────────── model zoo ─────────────────────────────────────────────────────
STD = StandardScaler()

svm_raw = make_pipeline(
    STD,
    LinearSVC(C=1.0, dual=False, max_iter=5000, class_weight="balanced")
)
SVM_CAL = CalibratedClassifierCV(svm_raw, cv=5, method="sigmoid")

# ---------- fast L1 via celer  -------------
CELER_L1 = LogisticRegressionCV(
    penalty      = "l1",
    solver       = "celer",
    Cs           = 20,
    cv           = 5,
    max_iter     = 5000,
    n_jobs       = 1,
    scoring      = "neg_log_loss",
    fit_intercept= True,
    tol          = 1e-6
)

MODELS_FILTERED = {
    "LDA"  : LinearDiscriminantAnalysis(),
    "DLDA" : GaussianNB(),
    "kNN"  : make_pipeline(STD,
              KNeighborsClassifier(n_neighbors=3, weights="distance")),
    "SVM"  : SVM_CAL,
    "RF"   : RandomForestClassifier(n_estimators=500,
                                    max_features="sqrt",
                                    random_state=1, n_jobs=1),
    "Lasso_Filtered": make_pipeline(STD, CELER_L1)
}
LASSO_UNF = make_pipeline(STD, CELER_L1)

# -------------------------------------------------------------------------
# Super‑Learner : *seven* base learners
# -------------------------------------------------------------------------
STACK_BASE = [
    ("rf",     MODELS_FILTERED["RF"]),
    ("svm",    MODELS_FILTERED["SVM"]),
    ("lassoF", MODELS_FILTERED["Lasso_Filtered"]),
    ("knn",    MODELS_FILTERED["kNN"]),
    ("lda",    MODELS_FILTERED["LDA"]),
    ("dlda",   MODELS_FILTERED["DLDA"]),
    ("lassoU", LASSO_UNF)  # NEW
]
META = LogisticRegression(penalty="l1", solver="liblinear")
STACK = StackingClassifier(
    estimators      = STACK_BASE,
    final_estimator = META,
    cv              = 5,
    n_jobs          = 1,
    stack_method    = "predict_proba"
)

# ─────────── Monte‑Carlo loop ──────────────────────────────────────────────
sss       = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.33, random_state=42)
progress  = max(1, N_SPLITS // 10)
rows, w_rows, gene_cnt = [], [], {}

for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):
    tr_cols, te_cols = samples[tr], samples[te]

    # ---------- limma feature filter --------------------------------------
    top_genes = limma_topk(tr)
    for g in top_genes:
        gene_cnt[g] = gene_cnt.get(g, 0) + 1

    Xtr_f, Xte_f = (expr.loc[top_genes, tr_cols].T.values,
                    expr.loc[top_genes, te_cols].T.values)
    Xtr_full, Xte_full = (expr.iloc[:, tr].T.values,
                          expr.iloc[:, te].T.values)
    ytr, yte = classes[tr], classes[te]

    # -------------- filtered models --------------------------------------
    for name, clf in MODELS_FILTERED.items():
        t0 = time.perf_counter()
        clf.fit(Xtr_f, ytr)
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        yhat = clf.predict(Xte_f)
        pred_s = time.perf_counter() - t0

        logging.info("split %3d/%d | %-16s | train %.3fs | pred %.3fs",
                     split_no, N_SPLITS, name, train_s, pred_s)

        rows.append(split_metrics(yte, yhat) | dict(
            split=split_no, model=name,
            train_s=train_s, pred_s=pred_s))

    # -------------- unfiltered lasso -------------------------------------
    t0 = time.perf_counter()
    LASSO_UNF.fit(Xtr_full, ytr)
    tr_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    yhat = LASSO_UNF.predict(Xte_full)
    pr_s = time.perf_counter() - t0

    logging.info("split %3d/%d | %-16s | train %.3fs | pred %.3fs",
                 split_no, N_SPLITS, "Lasso_Unfiltered", tr_s, pr_s)

    rows.append(split_metrics(yte, yhat) | dict(
        split=split_no, model="Lasso_Unfiltered",
        train_s=tr_s, pred_s=pr_s))

    # -------------- Super‑Learner ----------------------------------------
    t0 = time.perf_counter()
    STACK.fit(Xtr_f, ytr)
    tr_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    yhat = STACK.predict(Xte_f)
    pr_s = time.perf_counter() - t0

    logging.info("split %3d/%d | %-16s | train %.3fs | pred %.3fs",
                 split_no, N_SPLITS, "SuperLearner", tr_s, pr_s)

    rec = split_metrics(yte, yhat) | dict(
        split=split_no, model="SuperLearner",
        train_s=tr_s, pred_s=pr_s)
    for (alias, _), wt in zip(STACK_BASE, STACK.final_estimator_.coef_.ravel()):
        rec[f"SL_w_{alias}"] = wt
        w_rows.append(dict(split=split_no, base=alias, weight=wt))
    rows.append(rec)

    if split_no % progress == 0 or split_no == N_SPLITS:
        logging.info("=== progress %d / %d ===", split_no, N_SPLITS)

# ─────────── write outputs ─────────────────────────────────────────────────
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_METRICS, sep="\\t", index=False)

(pd.Series(gene_cnt, name="count")
   .sort_values(ascending=False)
   .rename_axis("gene").reset_index()
   .to_csv(OUT_FREQ, index=False))

pd.DataFrame(w_rows).to_csv(OUT_WTS, sep="\\t", index=False)
logging.info("✓ wrote %s, %s and %s",
             OUT_METRICS.name, OUT_FREQ.name, OUT_WTS.name)
