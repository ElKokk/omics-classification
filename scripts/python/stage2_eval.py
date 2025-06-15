"""
Stage-2 – fixed-signature evaluation
====================================
Signature = K most frequent genes from Stage-1
Monte-Carlo (1000 splits) on that fixed panel.
Two classifiers: LDA -- DLDA (GaussianNB)

Writes
results/{ds}/stage2/metrics.tsv
"""
# ────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# ───────────── Snakemake I/O ────────────────────────────────────────────────
matrix_fp   = Path(snakemake.input["matrix"])
panel_fp    = Path(snakemake.input["gene_set"])
out_metrics = Path(snakemake.output[0])
K_FIXED     = int(snakemake.params["fixed"])

# ---------- helper ----------------------------------------------------------
def mce_sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return (
        1 - (tp + tn) / len(y_true),
        tp / (tp + fn) if tp + fn else np.nan,
        tn / (tn + fp) if tn + fp else np.nan,
    )
mean_se = lambda a: (np.mean(a), np.std(a, ddof=1)/np.sqrt(len(a)))

# ---------- matrix ----------------------------------------------------------
expr = (pd.read_csv(matrix_fp, header=None)
          .drop(columns=[0])
          .pipe(lambda df: df.iloc[1:].astype(float)
                .set_index(df.index[1:].map(str))))
expr.columns = pd.read_csv(matrix_fp, header=None).iloc[0, 1:]

# ---------- fixed gene panel -------------------------------------------------
panel = (pd.read_csv(panel_fp)
           .sort_values("count", ascending=False)
           .head(K_FIXED)["gene"].astype(str).tolist())

# ---------- labels -----------------------------------------------------------
samples = expr.columns.astype(str)
classes = np.where(
    pd.Series(samples).str.contains("cancer", case=False, na=False),
    "Cancer", "Control")

# ---------- MC-CV ------------------------------------------------------------
results = {m: {"MCE": [], "Sens": [], "Spec": []}
           for m in ("LDA", "DLDA")}

sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.33, random_state=63)
for tr, te in sss.split(samples, classes):
    Xtr = expr.loc[panel, samples[tr]].T.values
    Xte = expr.loc[panel, samples[te]].T.values
    ytr, yte = classes[tr], classes[te]

    MODELS = {"LDA": LinearDiscriminantAnalysis(),
              "DLDA": GaussianNB()}
    for name, clf in MODELS.items():
        yhat = clf.fit(Xtr, ytr).predict(Xte)
        m, s1, s2 = mce_sens_spec(yte, yhat)
        results[name]["MCE"].append(m)
        results[name]["Sens"].append(s1)
        results[name]["Spec"].append(s2)

# ---------- summarise --------------------------------------------------------
rows = []
for name, d in results.items():
    mce_m,  mce_se  = mean_se(d["MCE"])
    sen_m,  sen_se  = mean_se(d["Sens"])
    spe_m,  spe_se  = mean_se(d["Spec"])
    rows.append(dict(model=name,
                     MCE_mean=mce_m,  MCE_se=mce_se,
                     Sens_mean=sen_m, Sens_se=sen_se,
                     Spec_mean=spe_m, Spec_se=spe_se))

out_metrics.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_metrics, sep="\t", index=False)
print(f"✓ wrote summary → {out_metrics}")
