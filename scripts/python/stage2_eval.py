#!/usr/bin/env python

"""
Stage‑2 – robust‑signature evaluation
• fixed top‑K gene panels (panel_k{K}.txt) with identical models
supports Golub / Grouped / Prostmat (see Stage‑1 notes).
"""
from pathlib import Path
import numpy as np, pandas as pd, logging, time, random, multiprocessing as mp
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
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.parallel")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble._base")

# ---- Snakemake I/O ---------------------------------------------------------
matrix_fp = Path(snakemake.input["matrix"])
panel_fp  = Path(snakemake.input["gene_set"])
out_metrics    = Path(snakemake.output["metrics"])
out_per_split  = Path(snakemake.output["per_split"])
out_wts        = Path(snakemake.output["sl_wts"])
K_FIXED = sum(1 for _ in open(panel_fp))
N_SPLITS = int(snakemake.params["n_splits"])
DATASET = matrix_fp.stem.replace("_matrix", "")
IMBALANCED_DATASETS = ['golub','golub_leukemia']
is_imbalanced = DATASET.lower() in [d.lower() for d in IMBALANCED_DATASETS]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
print(f"{time.strftime('%H:%M:%S')} | INFO | Stage‑2 | DATASET={DATASET} | K_fixed={K_FIXED} | splits={N_SPLITS} | Imbalanced={is_imbalanced}")

# ---- adapter (same as Stage‑1) --------------------------------------------
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
        df = pd.read_csv(fp, header=0).transpose()
        df.columns = [f"sample_{i}" for i in range(1, df.shape[1]+1)]
        df.index = df.index.astype(str)
        expr = df.astype(float)
        n = expr.shape[1]
        if n==38: y=np.array(['ALL']*27+['AML']*11)
        elif n==72: y=np.array(['ALL']*27+['AML']*11+['ALL']*20+['AML']*14)
        else: raise ValueError(f"Unexpected n={n} for Golub")
        return expr, y, "golub"
    try:
        head=pd.read_csv(fp, nrows=5)
        label_col=next((c for c in head.columns if str(c).lower() in ("group","groups","label")), None)
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

expr, classes, adapter = read_matrix_and_labels(matrix_fp, DATASET)
samples = expr.columns.astype(str)
panel   = [g.strip() for g in open(panel_fp) if g.strip()]

def mce_sens_spec(y_true, y_pred):
    neg = "ALL" if DATASET.lower()=="golub" else "Control"
    pos = "AML" if DATASET.lower()=="golub" else "Cancer"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg,pos]).ravel()
    return dict(MCE=1-(tp+tn)/len(y_true),
                Sensitivity=tp/(tp+fn) if tp+fn else np.nan,
                Specificity=tn/(tn+fp) if tn+fp else np.nan)

def mean_se(arr): return np.mean(arr), np.std(arr, ddof=1)/np.sqrt(len(arr))

class DynamicSMOTE(SMOTE):
    def _fit_resample(self, X, y):
        unique, counts = np.unique(y, return_counts=True)
        minority=min(counts)
        if minority>=2:
            self.k_neighbors=min(5, minority-1)
            return super()._fit_resample(X,y)
        return X.copy(), y.copy()

# ---- model zoo -------------------------------------------------------------
def celer_pn():
    inner_n_jobs=max(1, snakemake.threads // N_SPLITS)
    base=CelLogReg(penalty='l1', solver='celer-pn', fit_intercept=False,
                   max_iter=200, max_epochs=50000, tol=1e-4)
    C_grid=np.logspace(-2, 1, 12)
    gs=GridSearchCV(base, {'C':C_grid}, cv=5, scoring='neg_log_loss',
                    n_jobs=inner_n_jobs, refit=True)
    if is_imbalanced:
        return imb_make_pipeline(DynamicSMOTE(sampling_strategy='auto', random_state=1), gs)
    return gs

STD=StandardScaler(with_mean=False)
svm_raw=make_pipeline(STD, LinearSVC(C=1., dual=False, max_iter=5000,
                                     class_weight="balanced" if is_imbalanced else None))
SVM_CAL=CalibratedClassifierCV(svm_raw, cv=5, method="sigmoid")

inner_n_jobs=max(1, snakemake.threads // N_SPLITS)
if is_imbalanced:
    lda_model=imb_make_pipeline(DynamicSMOTE(sampling_strategy='auto', random_state=1),
                                LinearDiscriminantAnalysis())
    knn_model=imb_make_pipeline(DynamicSMOTE(sampling_strategy='auto', random_state=1),
                                STD, KNeighborsClassifier(n_neighbors=3, weights="distance"))
    rf_model=RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                    random_state=1, n_jobs=inner_n_jobs, class_weight="balanced")
else:
    lda_model=LinearDiscriminantAnalysis()
    knn_model=make_pipeline(STD, KNeighborsClassifier(n_neighbors=3, weights="distance"))
    rf_model=RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                    random_state=1, n_jobs=inner_n_jobs)

MODELS = {
    "LDA"   : lda_model,
    "DLDA"  : GaussianNB(),
    "kNN"   : knn_model,
    "SVM"   : SVM_CAL,
    "RF"    : rf_model,
    "Lasso" : celer_pn()
}
STACK_BASE=[("rf",MODELS["RF"]),("svm",MODELS["SVM"]),("lasso",MODELS["Lasso"]),
            ("knn",MODELS["kNN"]),("lda",MODELS["LDA"]),("dlda",MODELS["DLDA"])]
META=SkLogReg(penalty="l1", solver="liblinear")
MODELS["SuperLearner"]=StackingClassifier(estimators=STACK_BASE, final_estimator=META,
                                          cv=5, n_jobs=inner_n_jobs, stack_method="predict_proba")

# ---- seed from Stage‑1 -------------------------
seed_file = f"results/{DATASET}/stage1/seed.txt"
seed = int(Path(seed_file).read_text().strip())
random.seed(seed)
sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.33, random_state=seed)

def process_split(args):
    split_no, (tr, te) = args
    Xtr_pan = expr.loc[panel, samples[tr]].T.values
    Xte_pan = expr.loc[panel, samples[te]].T.values
    Xtr_all = expr.iloc[:, tr].T.values
    Xte_all = expr.iloc[:, te].T.values
    ytr, yte = classes[tr], classes[te]

    local_rows, local_weight_rows = [], []
    for name, clf in MODELS.items():
        sw = compute_sample_weight('balanced', ytr) if is_imbalanced and name=="DLDA" else None
        if name == "Lasso":
            clf.fit(Xtr_all, ytr, **({"sample_weight": sw} if sw is not None else {}))
            yhat = clf.predict(Xte_all)
        else:
            clf.fit(Xtr_pan, ytr, **({"sample_weight": sw} if sw is not None else {}))
            yhat = clf.predict(Xte_pan)
        met = mce_sens_spec(yte, yhat)
        local_rows.append(met | dict(split=split_no, model=name))
        if name == "SuperLearner":
            for (alias, _), wt in zip(STACK_BASE, clf.final_estimator_.coef_.ravel()):
                local_weight_rows.append({"split":split_no, "model":alias, "weight":wt})
    return local_rows, local_weight_rows

with mp.Pool(min(snakemake.threads, N_SPLITS)) as pool:
    results = pool.map(process_split, enumerate(sss.split(samples, classes), 1))

rows, weight_rows = [], []
for local_rows, local_w in results:
    rows.extend(local_rows); weight_rows.extend(local_w)

out_per_split.parent.mkdir(parents=True, exist_ok=True)
per_split_df = pd.DataFrame(rows); per_split_df.to_csv(out_per_split, sep="\t", index=False)
pd.DataFrame(weight_rows).to_csv(out_wts, sep="\t", index=False)

agg_rows=[]
for mdl, grp in per_split_df.groupby("model"):
    agg_rows.append({
        "model"     : mdl,
        "MCE_mean"  : grp["MCE"].mean(),
        "MCE_se"    : grp["MCE"].std(ddof=1)/np.sqrt(len(grp)),
        "Sens_mean" : grp["Sensitivity"].mean(),
        "Sens_se"   : grp["Sensitivity"].std(ddof=1)/np.sqrt(len(grp)),
        "Spec_mean" : grp["Specificity"].mean(),
        "Spec_se"   : grp["Specificity"].std(ddof=1)/np.sqrt(len(grp)),
    })
pd.DataFrame(agg_rows).to_csv(out_metrics, sep="\t", index=False)
print(f"{time.strftime('%H:%M:%S')} | INFO | ✓ wrote per-split & aggregated Stage‑2 metrics")
