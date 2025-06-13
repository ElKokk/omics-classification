"""
Stage-1 – Monte-Carlo cross-validation
=====================================
(n_splits from config.yaml, no hard-coding)
"""
from pathlib import Path
import os, logging, re, time, json, platform
import multiprocessing as mp, psutil
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from rpy2.robjects import r, pandas2ri, packages
pandas2ri.activate()

# ─────────── Snakemake parameters ───────────────────────────────────────────
MATRIX_CSV  = Path(snakemake.input["matrix"])
OUT_METRICS = Path(snakemake.output["metrics"])
OUT_FREQ    = Path(snakemake.output["freq"])
TOP_K       = int(snakemake.params["k"])
N_SPLITS    = int(snakemake.params["n_splits"])
DATASET     = MATRIX_CSV.stem.replace("_matrix", "")

logging.basicConfig(format="%(levelname)s | %(message)s",
                    level=logging.INFO, force=True)
logging.info("Stage-1 | dataset=%s | K=%d | splits=%d",
             DATASET, TOP_K, N_SPLITS)

# ─────────── expression matrix ──────────────────────────────────────────────
def read_matrix(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, header=None).drop(columns=[0])
    df.columns = df.iloc[0]
    mat = df.iloc[1:].astype(float)
    mat.index = mat.index.map(str)
    return mat

expr = read_matrix(MATRIX_CSV)
expr.index = (expr.index
                .str.lstrip("X")
                .str.replace(r"\.0$", "", regex=True))
expr = expr[~expr.index.duplicated(keep="first")]
samples = expr.columns.astype(str)

classes = np.where(
    pd.Series(samples).str.contains("cancer", case=False, na=False),
    "Cancer", "Control").astype(str)

# ─────────── R-package limma ────────────────────────────────────────────────
_ = packages.importr("limma", suppress_messages=True)

def split_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return dict(
        MCE         = 1 - (tp + tn) / len(y_true),
        Sensitivity = tp / (tp + fn) if tp + fn else np.nan,
        Specificity = tn / (tn + fp) if tn + fp else np.nan)

# ─────────── Monte-Carlo loop ───────────────────────────────────────────────
gene_counter, rows = {}, []
sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.33, random_state=42)
progress_step = max(1, N_SPLITS // 10)

for split_no, (tr, te) in enumerate(sss.split(samples, classes), 1):
    tr_cols, te_cols = samples[tr], samples[te]

    # limma ranking ----------------------------------------------------------
    r.assign("mat_py", pandas2ri.py2rpy(expr[tr_cols]))
    design_vec = ",".join("1" if classes[i] == "Cancer" else "0" for i in tr)
    r(f"design <- model.matrix(~ factor(c({design_vec})))")
    r("suppressMessages(suppressWarnings("
      "fit <- eBayes(lmFit(mat_py, design))))")

    df_top = r(f"topTable(fit, n={TOP_K}, sort.by='t')")
    top_genes = [re.sub(r"\.0$", "", g.lstrip("X")) for g in df_top.index]
    for g in top_genes:
        gene_counter[g] = gene_counter.get(g, 0) + 1

    Xtr = expr.loc[top_genes, tr_cols].T.values
    Xte = expr.loc[top_genes, te_cols].T.values
    ytr, yte = classes[tr], classes[te]

    MODELS = {"LDA": LinearDiscriminantAnalysis(),
              "DLDA": GaussianNB()}
    for mdl_name, clf in MODELS.items():
        t0 = time.perf_counter(); clf.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); yhat = clf.predict(Xte)
        pred_s  = time.perf_counter() - t0

        row = split_metrics(yte, yhat)
        row.update(split=split_no, model=mdl_name,
                   train_s=train_s, pred_s=pred_s)
        rows.append(row)

        logging.info("split %d/%d | K=%d | %-4s | "
                     "train=%.3fs | pred=%.3fs",
                     split_no, N_SPLITS, TOP_K, mdl_name, train_s, pred_s)

    if (split_no % progress_step == 0) or (split_no == N_SPLITS):
        logging.info("split %d / %d finished", split_no, N_SPLITS)

# ─────────── write outputs ─────────────────────────────────────────────────
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_METRICS, sep="\t", index=False)

OUT_FREQ.parent.mkdir(parents=True, exist_ok=True)
(pd.Series(gene_counter, name="count").sort_values(ascending=False)
   .rename_axis("gene").reset_index()
   .to_csv(OUT_FREQ, index=False))

# hardware fingerprint (once)
finger_fp = OUT_METRICS.parent / "system_info.json"
if not finger_fp.exists():
    sjobs = os.environ.get("SNAKEMAKE_NJOBS", "")
    info = {
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": mp.cpu_count(),
        "total_ram_GiB": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "snakemake_cores": int(sjobs) if sjobs.isdigit() else None
    }
    finger_fp.write_text(json.dumps(info, indent=2))
