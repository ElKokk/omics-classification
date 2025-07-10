#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Outer stage‑2   frozen panel, evaluate on OUTER‑test
* train same 7 base learners + SuperLearner
* writes
    – metrics_k*.tsv
    – sl_weights_k*.tsv
"""
import time, logging
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.metrics      import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline     import make_pipeline
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import LinearSVC
from sklearn.calibration  import CalibratedClassifierCV
from sklearn.ensemble     import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model import LogisticRegression as SkLogReg
from celer                import LogisticRegression as CelLogReg

from _io_utils import read_singh_matrix

# ─── Snakemake I/O -----------------------------------------------------------
mat_csv   = Path(snakemake.input["matrix"])
panel_txt = Path(snakemake.input["panel"])
train_ids = Path(snakemake.input["train_ids"]).read_text().splitlines()
test_ids  = Path(snakemake.input["test_ids"]).read_text().splitlines()

out_metrics = Path(snakemake.output["metrics"])
out_slwts   = Path(snakemake.output["sl_wts"])

K_TOP = int(snakemake.params["K"])
OUTER = int(snakemake.params["O"])

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logging.info("outer‑%02d | K=%d | stage‑2 evaluation", OUTER, K_TOP)

# ─── data --------------------------------------------------------------------
expr_full = read_singh_matrix(mat_csv)
panel     = panel_txt.read_text().splitlines()  # frozen top‑K genes

Xtr_f = expr_full.loc[panel, train_ids].T.values
Xte_f = expr_full.loc[panel, test_ids ].T.values
Xtr_u = expr_full.loc[:,    train_ids].T.values
Xte_u = expr_full.loc[:,    test_ids ].T.values

ytr = np.where(pd.Series(train_ids).str.contains("cancer", case=False),"Cancer","Control")
yte = np.where(pd.Series(test_ids ).str.contains("cancer", case=False),"Cancer","Control")

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
    return CelLogReg(penalty='l1', solver='celer-pn', fit_intercept=False,
                     max_iter=600, max_epochs=25000, tol=1e-4, C=0.2)

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

rows, sl_rows = [], []

# filtered models
for name, clf in MODELS_FIL.items():
    t0 = time.perf_counter()
    clf.fit(Xtr_f, ytr); yhat = clf.predict(Xte_f)
    rows.append(split_metrics(yte, yhat) |
                dict(model=name, runtime_s=time.perf_counter()-t0))

# unfiltered lasso
t0 = time.perf_counter()
LASSO_UNF.fit(Xtr_u, ytr); yhat = LASSO_UNF.predict(Xte_u)
rows.append(split_metrics(yte, yhat) |
            dict(model="Lasso_Unfiltered",
                 runtime_s=time.perf_counter()-t0))

# SuperLearner
t0 = time.perf_counter()
STACK.fit(Xtr_f, ytr); yhat = STACK.predict(Xte_f)
rows.append(split_metrics(yte, yhat) |
            dict(model="SuperLearner",
                 runtime_s=time.perf_counter()-t0))

meta_coef = STACK.final_estimator_.coef_.ravel()
sl_rows.extend(dict(model=mdl, weight=w, K=K_TOP, outer=OUTER)
               for mdl, w in zip(BASE_NAMES, meta_coef))

# ─── save outputs ------------------------------------------------------------
out_metrics.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_metrics, sep="\t", index=False)
pd.DataFrame(sl_rows).to_csv(out_slwts,   sep="\t", index=False)

logging.info("✓ stage‑2 metrics → %s", out_metrics.name)
logging.info("✓ SL weights      → %s", out_slwts.name)
