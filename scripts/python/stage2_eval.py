#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑2 – robust‑signature evaluation
• fixed top‑K gene panels (panel_k{K}.txt)
• identical model zoo Celer‑PN + inner CV
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import time

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as SkLogReg
from celer import LogisticRegression as CelLogReg

# ─────── Snakemake I/O ──────────────────────────────────────────────────────
matrix_fp = Path(snakemake.input["matrix"])
panel_fp  = Path(snakemake.input["gene_set"])
out_fp    = Path(snakemake.output[0])
out_wts   = Path(snakemake.output[1])

K_FIXED = sum(1 for _ in open(panel_fp))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logging.info("Stage‑2 |  K_fixed=%d | matrix=%s", K_FIXED, matrix_fp.name)

# ─────── data ----------------------------------------------------------------
expr = (pd.read_csv(matrix_fp, header=None)
          .drop(columns=[0])
          .pipe(lambda d: d.iloc[1:].astype(float)
                         .set_index(d.index[1:].map(str))))
expr.columns = pd.read_csv(matrix_fp, header=None).iloc[0, 1:]

samples = expr.columns.astype(str)
classes = np.where(pd.Series(samples).str.contains("cancer", case=False, na=False),
                   "Cancer", "Control")
panel = [g.strip() for g in open(panel_fp) if g.strip()]

# ─────── helpers -------------------------------------------------------------
def mce_sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=["Control", "Cancer"]).ravel()
    return (1 - (tp + tn) / len(y_true),
            tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)

def mean_se(arr):
    return np.mean(arr), np.std(arr, ddof=1) / np.sqrt(len(arr))

# ─────── model zoo  ----------------------------
def celer_pn():
    base = CelLogReg(penalty='l1', solver='celer-pn', fit_intercept=False,
                     max_iter=200, max_epochs=50_000, tol=1e-4)
    C_grid = np.logspace(-2, 1, 12)
    return GridSearchCV(base, {'C': C_grid}, cv=5,
                        scoring='neg_log_loss', n_jobs=1, refit=True)

STD = StandardScaler(with_mean=False)
svm_raw = make_pipeline(STD,
    LinearSVC(C=1., dual=False, max_iter=5_000, class_weight="balanced"))
SVM_CAL = CalibratedClassifierCV(svm_raw, cv=5, method="sigmoid")

MODELS = {
    "LDA"             : LinearDiscriminantAnalysis(),
    "DLDA"            : GaussianNB(),
    "kNN"             : make_pipeline(STD,
                          KNeighborsClassifier(n_neighbors=3, weights="distance")),
    "SVM"             : SVM_CAL,
    "RF"              : RandomForestClassifier(
                            n_estimators=500, max_features="sqrt",
                            random_state=1, n_jobs=1),
    "Lasso_Filtered"  : celer_pn(),
    "Lasso_Unfiltered": celer_pn()
}
STACK_BASE = [
    ("rf",     MODELS["RF"]),
    ("svm",    MODELS["SVM"]),
    ("lassoF", MODELS["Lasso_Filtered"]),
    ("lassoU", MODELS["Lasso_Unfiltered"]),
    ("knn",    MODELS["kNN"]),
    ("lda",    MODELS["LDA"]),
    ("dlda",   MODELS["DLDA"])
]
META = SkLogReg(penalty="l1", solver="liblinear")
MODELS["SuperLearner"] = StackingClassifier(
    estimators=STACK_BASE, final_estimator=META,
    cv=5, n_jobs=1, stack_method="predict_proba")

# ─────── Monte‑Carlo CV ------------------------------------------------------
N_SPLITS = 100
results = {m: {"MCE": [], "Sens": [], "Spec": []} for m in MODELS}
weight_rows = []

sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.33,
                             random_state=63)
progress = max(1, N_SPLITS // 10)

for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):
    Xtr_pan = expr.loc[panel,  samples[tr]].T.values
    Xte_pan = expr.loc[panel,  samples[te]].T.values
    Xtr_all = expr.iloc[:, tr].T.values
    Xte_all = expr.iloc[:, te].T.values
    ytr, yte = classes[tr], classes[te]

    for name, clf in MODELS.items():
        t0 = time.perf_counter()
        if name == "Lasso_Unfiltered":
            yhat = clf.fit(Xtr_all, ytr).predict(Xte_all)
        else:
            yhat = clf.fit(Xtr_pan, ytr).predict(Xte_pan)
        dur = time.perf_counter() - t0

        mce, sn, sp = mce_sens_spec(yte, yhat)
        results[name]["MCE"].append(mce)
        results[name]["Sens"].append(sn)
        results[name]["Spec"].append(sp)
        logging.info("split %3d/%d | %-15s | dur %.3fs | MCE=%.3f",
                     split_no, N_SPLITS, name, dur, mce)

        # ───── store SL weights ─────────────────────────────────────────────
        if name == "SuperLearner":
            for (alias, _), wt in zip(STACK_BASE,
                                      clf.final_estimator_.coef_.ravel()):
                weight_rows.append({
                    "split":  split_no,
                    "model":  alias,
                    "weight": wt
                })

    if split_no % progress == 0 or split_no == N_SPLITS:
        logging.info("=== progress %d / %d ===", split_no, N_SPLITS)

# ─────── aggregate & write outputs ------------------------------------------
rows = []
for mdl, d in results.items():
    rows.append({
        "model"     : mdl,
        "MCE_mean"  : mean_se(d["MCE"] )[0],
        "MCE_se"    : mean_se(d["MCE"] )[1],
        "Sens_mean" : mean_se(d["Sens"])[0],
        "Sens_se"   : mean_se(d["Sens"])[1],
        "Spec_mean" : mean_se(d["Spec"])[0],
        "Spec_se"   : mean_se(d["Spec"])[1]
    })

out_fp.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_fp, sep="\t", index=False)
logging.info("✓ metrics  → %s", out_fp.name)

pd.DataFrame(weight_rows).to_csv(out_wts, sep="\t", index=False)
logging.info("✓ SL weights → %s", out_wts.name)
