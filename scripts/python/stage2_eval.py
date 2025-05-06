"""
Stage‑2 fixed‑signature evaluation (LDA only, for now).
"""

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

matrix_fp   = snakemake.input["matrix"]
panel_fp    = snakemake.input["gene_set"]
out_metrics = snakemake.output[0]
K_FIXED     = int(snakemake.params["fixed"])

# --------- load matrix (same rule as Stage‑1) -------------------------------
expr = pd.read_csv(matrix_fp, header=None).drop(columns=[0])
expr.columns = expr.iloc[0, :].tolist()
expr = expr.iloc[1:, :].astype(float)
expr.index = [str(i) for i in range(expr.shape[0])]

panel = (pd.read_csv(panel_fp)
           .sort_values("count", ascending=False)
           .head(K_FIXED)["gene"].astype(str).tolist())

samples = expr.columns
classes = np.array(["Cancer" if "cancer" in s.lower() else "Control"
                    for s in samples])

def mce_sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["Control", "Cancer"]).ravel()
    return (1 - (tp + tn) / len(y_true),
            tp / (tp + fn), tn / (tn + fp))

mce_ls, sens_ls, spec_ls = [], [], []
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=63)
for tr, te in sss.split(samples, classes):
    Xtr = expr.loc[panel, samples[tr]].T.values
    Xte = expr.loc[panel, samples[te]].T.values
    ytr, yte = classes[tr], classes[te]

    yhat = LinearDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte)
    m, s1, s2 = mce_sens_spec(yte, yhat)
    mce_ls.append(m); sens_ls.append(s1); spec_ls.append(s2)

def mean_se(a): return np.mean(a), np.std(a, ddof=1)/np.sqrt(len(a))

out = pd.DataFrame([{
    "method": "LDA",
    "MCE_mean":  mean_se(mce_ls)[0],
    "MCE_se":    mean_se(mce_ls)[1],
    "Sens_mean": mean_se(sens_ls)[0],
    "Sens_se":   mean_se(sens_ls)[1],
    "Spec_mean": mean_se(spec_ls)[0],
    "Spec_se":   mean_se(spec_ls)[1]
}])

Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_metrics, sep="\t", index=False)
