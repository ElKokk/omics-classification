#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage‑2  – robust‑signature evaluation
• fixed top‑K gene panels (built in Stage‑1)
• model zoo identical to Stage‑1   (celer L1, full scaling)
• Super‑Learner includes *both* filtered & unfiltered lasso
"""
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd, logging, time, itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from celer import LogisticRegressionCV                          # ← NEW
from sklearn.linear_model import LogisticRegression

# ─────────── Snakemake I/O ──────────────────────────────────────────
matrix_fp = Path(snakemake.input["matrix"])
panel_fp  = Path(snakemake.input["gene_set"])   # K‑specific robust list
out_fp    = Path(snakemake.output[0])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)

panel = pd.read_csv(panel_fp)["gene"].astype(str).tolist()
K_FIXED = len(panel)
logging.info("Stage‑2  |  K_fixed=%d  |  matrix=%s", K_FIXED, matrix_fp.name)

# ─────────── data ----------------------------------------------------------------
expr = (pd.read_csv(matrix_fp, header=None)
          .drop(columns=[0])
          .pipe(lambda d: d.iloc[1:].astype(float)
                         .set_index(d.index[1:].map(str))))
expr.columns = pd.read_csv(matrix_fp, header=None).iloc[0, 1:]

samples = expr.columns.astype(str)
classes = np.where(pd.Series(samples).str.contains("cancer", case=False, na=False),
                   "Cancer", "Control")

# ─────────── helpers ---------------------------------------------------------------
def mce_sens_spec(y_t, y_p):
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=["Control", "Cancer"]).ravel()
    return (1 - (tp + tn) / len(y_t),
            tp / (tp + fn) if tp + fn else np.nan,
            tn / (tn + fp) if tn + fp else np.nan)

def mean_se(arr): return np.mean(arr), np.std(arr, ddof=1) / np.sqrt(len(arr))

# ─────────── model zoo ------------------------------------------------------------
STD = StandardScaler()
svm_raw = make_pipeline(STD,
                        LinearSVC(C=1., dual=False, max_iter=5000,
                                  class_weight="balanced"))
SVM_CAL = CalibratedClassifierCV(svm_raw, cv=5, method="sigmoid")

CELER_L1 = LogisticRegressionCV(
    penalty  = "l1",
    solver   = "celer",
    Cs       = 20,
    cv       = 5,
    max_iter = 5000,
    n_jobs   = 1,
    scoring  = "neg_log_loss",
    tol      = 1e-6
)

MODELS = {
    "LDA"  : LinearDiscriminantAnalysis(),
    "DLDA" : GaussianNB(),
    "kNN"  : make_pipeline(STD,
               KNeighborsClassifier(n_neighbors=3, weights="distance")),
    "SVM"  : SVM_CAL,
    "RF"   : RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                    random_state=1, n_jobs=1),
    "Lasso_Filtered": make_pipeline(STD, CELER_L1),
    "Lasso_Unfiltered": make_pipeline(STD, CELER_L1)
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
META = LogisticRegression(penalty="l1", solver="liblinear")
MODELS["SuperLearner"] = StackingClassifier(
    estimators      = STACK_BASE,
    final_estimator = META,
    cv              = 5,
    n_jobs          = 1,
    stack_method    = "predict_proba"
)

# ─────────── Monte‑Carlo CV -------------------------------------------------------
results = {m: {"MCE": [], "Sens": [], "Spec": []} for m in MODELS}
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=63)

for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):
    Xtr_panel = expr.loc[panel, samples[tr]].T.values
    Xte_panel = expr.loc[panel, samples[te]].T.values
    Xtr_full  = expr.iloc[:, tr].T.values
    Xte_full  = expr.iloc[:, te].T.values
    ytr, yte = classes[tr], classes[te]

    for name, clf in MODELS.items():
        if name == "Lasso_Unfiltered":
            yhat = clf.fit(Xtr_full, ytr).predict(Xte_full)
        else:
            yhat = clf.fit(Xtr_panel, ytr).predict(Xte_panel)
        mce, sn, sp = mce_sens_spec(yte, yhat)
        results[name]["MCE"].append(mce)
        results[name]["Sens"].append(sn)
        results[name]["Spec"].append(sp)

# ─────────── aggregate & write -----------------------------------------------------
rows = []
for n, d in results.items():
    rows.append(dict(model=n,
                     MCE_mean = mean_se(d["MCE"])[0],
                     MCE_se   = mean_se(d["MCE"])[1],
                     Sens_mean= mean_se(d["Sens"])[0],
                     Sens_se  = mean_se(d["Sens"])[1],
                     Spec_mean= mean_se(d["Spec"])[0],
                     Spec_se  = mean_se(d["Spec"])[1]))

out_fp.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_fp, sep="\\t", index=False)
logging.info("✓ results → %s", out_fp.name)
