#!/usr/bin/env python
"""
Stage‑1 – Monte‑Carlo CV benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• limma filter (top‑K)
• ℓ1‑Logistic via Celer‑PN + inner CV
• saves per‑split top‑K panels to all_top_genes_k{K}.pkl


  - Golub (special handling order)
"""
from pathlib import Path
import numpy as np, pandas as pd, logging, re, time, json, random, pickle, multiprocessing as mp
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix
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
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.parallel")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble._base")

# ---- limma via rpy2 --------------------------------------------------------
from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate(); _ = packages.importr("limma", suppress_messages=True)
r('options(warn=-1)')

# ---- Snakemake I/O ---------------------------------------------------------
MATRIX_CSV  = Path(snakemake.input ["matrix"])
SEED_FILE   = Path(snakemake.input ["seed"])
OUT_METRICS = Path(snakemake.output["metrics"])
OUT_FREQ    = Path(snakemake.output["freq"])
OUT_WTS     = Path(snakemake.output["sl_wts"])
TOP_K       = int(snakemake.params["k"])
N_SPLITS    = int(snakemake.params["n_splits"])
DATASET     = MATRIX_CSV.stem.replace("_matrix", "")

# ---- imbalance knob  -----------------------------
IMBALANCED_DATASETS = ['golub', 'golub_leukemia']
is_imbalanced = DATASET.lower() in [d.lower() for d in IMBALANCED_DATASETS]

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
print(f"{time.strftime('%H:%M:%S')} | INFO | DATASET={DATASET}  TOP_K={TOP_K}  SPLITS={N_SPLITS}  Imbalanced={is_imbalanced}")

# ---- Dynamic SMOTE ----------------------------
class DynamicSMOTE(SMOTE):
    def _fit_resample(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        minority = min(counts)
        if minority >= 2:
            self.k_neighbors = min(5, minority - 1)
            return super()._fit_resample(X, y)
        return X.copy(), y.copy()

# ---- dataset adapter -------------------------------------------------------
def _map_group(vals):
    out = []
    for v in vals:
        s = str(v).strip().lower()
        if s in {"1","true","case","cancer","tumor","tumour","crc"}:
            out.append("Cancer")
        elif s in {"0","false","control","normal","healthy"}:
            out.append("Control")
        else:
            out.append(str(v))
    return np.array(out, dtype=object)

def read_matrix_and_labels(fp: Path, dataset_name: str):
    """
    Returns:
      expr   : DataFrame
      yclass : ndarray[str]
      tag    : 'golub' | 'grouped' | 'prostmat'
    """
    if dataset_name.lower() == "golub":
        df = pd.read_csv(fp, header=0).transpose()
        df.columns = [f"sample_{i}" for i in range(1, df.shape[1]+1)]
        df.index = df.index.astype(str)
        expr = df.astype(float)
        n = expr.shape[1]
        if n == 38:
            y = np.array(['ALL'] * 27 + ['AML'] * 11)
        elif n == 72:
            y = np.array(['ALL'] * 27 + ['AML'] * 11 + ['ALL'] * 20 + ['AML'] * 14)
        else:
            raise ValueError(f"Unexpected number of samples for Golub: {n}")
        return expr, y, "golub"

    try:
        head = pd.read_csv(fp, nrows=5)
        label_col = next((c for c in head.columns if str(c).lower() in ("group","groups","label")), None)
    except Exception:
        label_col = None
    if label_col is not None:
        df = pd.read_csv(fp)
        sample_col = next((c for c in df.columns if str(c).lower() in ("samples","sample","id","sample_id") and c != label_col), None)
        if sample_col is None:

            sample_col = next(c for c in df.columns if c != label_col)
        feat_cols = [c for c in df.columns if c not in {label_col, sample_col}]
        expr = df[feat_cols].T
        expr.columns = df[sample_col].astype(str).values
        expr.index = expr.index.astype(str)
        expr = expr.astype(float)
        expr.index = expr.index.str.lstrip("X").str.replace(r"\.0$","", regex=True)
        expr = expr[~expr.index.duplicated(keep="first")]
        y = _map_group(df[label_col].values)
        return expr, y, "grouped"


    d = pd.read_csv(fp, header=None).drop(columns=[0])
    d.columns = d.iloc[0]
    expr = (d.iloc[1:].astype(float)
              .set_index(d.index[1:].map(str)))
    expr.index = expr.index.str.lstrip("X").str.replace(r"\.0$", "", regex=True)
    expr = expr[~expr.index.duplicated(keep="first")]
    samples = expr.columns.astype(str)
    y = np.where(pd.Series(samples).str.contains("cancer", case=False, na=False),
                 "Cancer", "Control")
    return expr, y, "prostmat"

def limma_topk(expr, classes, idx_train):
    positive = "AML" if DATASET.lower()=="golub" else "Cancer"
    r.assign("mat_py", pandas2ri.py2rpy(expr.iloc[:, idx_train]))
    design_nums = ",".join("1" if classes[i]==positive else "0" for i in idx_train)
    r(f"design <- model.matrix(~ factor(c({design_nums})))")
    r("fit <- eBayes(lmFit(mat_py, design))")
    tbl = r(f"topTable(fit, n={TOP_K}, sort.by='t')")
    return [re.sub(r"\.0$","", g.lstrip("X")) for g in tbl.index]

def split_metrics(y_true, y_pred):
    neg = "ALL" if DATASET.lower()=="golub" else "Control"
    pos = "AML" if DATASET.lower()=="golub" else "Cancer"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg,pos]).ravel()
    return dict(MCE=1-(tp+tn)/len(y_true),
                Sensitivity=tp/(tp+fn) if tp+fn else np.nan,
                Specificity=tn/(tn+fp) if tn+fp else np.nan)

# ---- data ------------------------------------------------------------------
expr, classes, adapter = read_matrix_and_labels(MATRIX_CSV, DATASET)
samples = expr.columns.astype(str)
print(f"{time.strftime('%H:%M:%S')} | INFO | Adapter={adapter} | n_genes={expr.shape[0]} | n_samples={expr.shape[1]} | class_counts={dict(zip(*np.unique(classes, return_counts=True)))}")

seed = int(SEED_FILE.read_text().strip())
random.seed(seed)
sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.33, random_state=seed)

# ---- model zoo -------------------------------------------------------------
STD = StandardScaler(with_mean=False)
svm_raw = make_pipeline(STD, LinearSVC(C=1.0, dual=False, max_iter=5000,
                                       class_weight="balanced" if is_imbalanced else None))
SVM_CAL = CalibratedClassifierCV(svm_raw, cv=5, method="sigmoid")

def celer_lasso():
    inner_n_jobs = max(1, snakemake.threads // N_SPLITS)
    base = CelLogReg(penalty='l1', solver='celer-pn', fit_intercept=False,
                     max_iter=200, max_epochs=20000, tol=1e-4, verbose=False)
    C_grid = np.logspace(-2, 1, 12)
    gs = GridSearchCV(base, {'C': C_grid}, cv=5, scoring='neg_log_loss',
                      n_jobs=inner_n_jobs, refit=True)
    if is_imbalanced:
        return imb_make_pipeline(DynamicSMOTE(sampling_strategy='auto', random_state=1), gs)
    return gs

inner_n_jobs = max(1, snakemake.threads // N_SPLITS)
if is_imbalanced:
    lda_model = imb_make_pipeline(DynamicSMOTE(sampling_strategy='auto', random_state=1),
                                  LinearDiscriminantAnalysis())
    knn_model = imb_make_pipeline(DynamicSMOTE(sampling_strategy='auto', random_state=1),
                                  STD, KNeighborsClassifier(n_neighbors=3, weights="distance"))
    rf_model = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                      random_state=1, n_jobs=inner_n_jobs, class_weight="balanced")
else:
    lda_model = LinearDiscriminantAnalysis()
    knn_model = make_pipeline(STD, KNeighborsClassifier(n_neighbors=3, weights="distance"))
    rf_model = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                      random_state=1, n_jobs=inner_n_jobs)

MODELS_FILTERED = {
    "LDA"   : lda_model,
    "DLDA"  : GaussianNB(),
    "kNN"   : knn_model,
    "SVM"   : SVM_CAL,
    "RF"    : rf_model
}
LASSO = celer_lasso()
STACK_BASE = [("rf", MODELS_FILTERED["RF"]), ("svm", MODELS_FILTERED["SVM"]),
              ("lasso", LASSO), ("knn", MODELS_FILTERED["kNN"]),
              ("lda", MODELS_FILTERED["LDA"]), ("dlda", MODELS_FILTERED["DLDA"])]
META  = SkLogReg(penalty="l1", solver="liblinear")
STACK = StackingClassifier(estimators=STACK_BASE, final_estimator=META,
                           cv=5, n_jobs=inner_n_jobs, stack_method="predict_proba")

# ---- worker for one split --------------------------------------------------
def process_split(args):
    split_no, (tr, te) = args
    tr_cols, te_cols = samples[tr], samples[te]
    top_genes = limma_topk(expr, classes, tr)
    Xtr_f, Xte_f = (expr.loc[top_genes, tr_cols].T.values,
                    expr.loc[top_genes, te_cols].T.values)
    Xtr_all, Xte_all = (expr.iloc[:, tr].T.values, expr.iloc[:, te].T.values)
    ytr, yte = classes[tr], classes[te]

    local_rows, local_w_rows = [], []

    for name, clf in MODELS_FILTERED.items():
        sw = compute_sample_weight('balanced', ytr) if is_imbalanced and name=="DLDA" else None
        t0 = time.perf_counter()
        clf.fit(Xtr_f, ytr, **({"sample_weight": sw} if sw is not None else {}))
        yhat = clf.predict(Xte_f)
        dur_tr = time.perf_counter() - t0
        print(f"{time.strftime('%H:%M:%S')} | split {split_no:3d}/{N_SPLITS} | {name:<13s} | MCE=?")
        local_rows.append(split_metrics(yte, yhat) | dict(split=split_no, model=name,
                                                          train_s=dur_tr, pred_s=0.0))

    t0 = time.perf_counter(); LASSO.fit(Xtr_all, ytr); yhat = LASSO.predict(Xte_all)
    dur_tr = time.perf_counter() - t0
    local_rows.append(split_metrics(yte, yhat) | dict(split=split_no, model="Lasso",
                                                      train_s=dur_tr, pred_s=0.0))

    t0 = time.perf_counter(); STACK.fit(Xtr_f, ytr); yhat = STACK.predict(Xte_f)
    dur_tr = time.perf_counter() - t0
    rec = split_metrics(yte, yhat) | dict(split=split_no, model="SuperLearner",
                                          train_s=dur_tr, pred_s=0.0)
    for (alias, _), wt in zip(STACK_BASE, STACK.final_estimator_.coef_.ravel()):
        rec[f"SL_w_{alias}"] = wt
        local_w_rows.append(dict(split=split_no, base=alias, weight=wt))
    local_rows.append(rec)

    return local_rows, local_w_rows, top_genes

# ---- run all splits in parallel -------------------------------------------
with mp.Pool(min(snakemake.threads, N_SPLITS)) as pool:
    results = pool.map(process_split, enumerate(sss.split(samples, classes), 1))

rows, w_rows, all_top_genes = [], [], []
gene_cnt = {}
for local_rows, local_w_rows, top_genes in results:
    rows.extend(local_rows); w_rows.extend(local_w_rows); all_top_genes.append(top_genes)
    for g in top_genes: gene_cnt[g] = gene_cnt.get(g, 0) + 1

# ---- outputs ---------------------------------------------------------------
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_METRICS, sep="\t", index=False)
pd.Series(gene_cnt, name="count").sort_values(ascending=False)\
  .rename_axis("gene").reset_index().to_csv(OUT_FREQ, index=False)
pd.DataFrame(w_rows).to_csv(OUT_WTS, sep="\t", index=False)
with open(f"results/{DATASET}/stage1/all_top_genes_k{TOP_K}.pkl", "wb") as f:
    pickle.dump(all_top_genes, f)
print(f"{time.strftime('%H:%M:%S')} | INFO | ✓ wrote metrics/freq/wts and per-split panels")
