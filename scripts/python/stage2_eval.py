"""
Stage-2 – fixed-signature evaluation
====================================
* Signature = top-K genes most frequently selected in Stage-1
* X number of Monte-Carlo splits
* Two classifiers:
      - LDA
      - DLDA ( I make use of GaussianNB ≃ diagonal-LDA)
* I Save mean ± SE of each metric per model.
--------------------------------------------------------------------
Writes
    results/{ds}/stage2/metrics.tsv
"""
# ──────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# ---------- Snakemake I/O -------------------------------------------------
matrix_fp   = Path(snakemake.input["matrix"])
panel_fp    = Path(snakemake.input["gene_set"])
out_metrics = Path(snakemake.output[0])
K_FIXED     = int(snakemake.params["fixed"])

# ---------- helpers ------------------------------------------------------
def mce_sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return (
        1 - (tp + tn) / len(y_true),
        tp / (tp + fn) if tp + fn else np.nan,
        tn / (tn + fp) if tn + fp else np.nan,
    )

def mean_se(arr):
    arr = np.asarray(arr)
    return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))

# ---------- loading of expression matrix --------------------------------------
expr = (pd.read_csv(matrix_fp, header=None)
          .drop(columns=[0])
          .pipe(lambda df: df.iloc[1:].astype(float)
                .set_index(df.index[1:].map(str))))
expr.columns = pd.read_csv(matrix_fp, header=None).iloc[0, 1:]

# ---------- fixed gene panel --------------------------------------------
panel_genes = (pd.read_csv(panel_fp)
                 .sort_values("count", ascending=False)
                 .head(K_FIXED)["gene"]
                 .astype(str).tolist())

# ---------- labels -------------------------------------------------------
samples = expr.columns.astype(str)
classes = np.where(
    pd.Series(samples).str.contains("cancer", case=False, na=False),
    "Cancer",
    "Control",
)

# ---------- MC-CV loop ---------------------------------------------------
results = {mdl: {"MCE": [], "Sens": [], "Spec": []}
           for mdl in ("LDA", "DLDA")}

sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=63)
for tr, te in sss.split(samples, classes):
    Xtr = expr.loc[panel_genes, samples[tr]].T.values
    Xte = expr.loc[panel_genes, samples[te]].T.values
    ytr, yte = classes[tr], classes[te]

    MODELS = {
        "LDA" : LinearDiscriminantAnalysis(),
        "DLDA": GaussianNB(),
    }
    for name, clf in MODELS.items():
        yhat = clf.fit(Xtr, ytr).predict(Xte)
        m, s1, s2 = mce_sens_spec(yte, yhat)
        results[name]["MCE"].append(m)
        results[name]["Sens"].append(s1)
        results[name]["Spec"].append(s2)

# ---------- aggregate mean ± SE -----------------------------------------
rows = []
for name, d in results.items():
    mce_mu,  mce_se  = mean_se(d["MCE"])
    sen_mu,  sen_se  = mean_se(d["Sens"])
    spe_mu,  spe_se  = mean_se(d["Spec"])
    rows.append(dict(model=name,
                     MCE_mean=mce_mu,  MCE_se=mce_se,
                     Sens_mean=sen_mu, Sens_se=sen_se,
                     Spec_mean=spe_mu, Spec_se=spe_se))

out_metrics.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_metrics, sep="\t", index=False)
print(f"✓  Saved summary → {out_metrics}")
