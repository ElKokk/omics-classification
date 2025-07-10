#!/usr/bin/env python
# -------------------------------------------------------------------------
# Create outer‑train / outer‑test text files with stratification.
# -------------------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd, logging
from sklearn.model_selection import StratifiedShuffleSplit

# -------- Snakemake I/O ----------------------------------------------------
matrix_fp = Path(snakemake.input["matrix"])
train_fp  = Path(snakemake.output["train_ids"])
test_fp   = Path(snakemake.output["test_ids"])
OUTER_SEED = int(snakemake.params["O"])
TEST_FRAC  = float(snakemake.config["test_frac"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S")

# -------- read expression matrix  -----------------------------------------
# • first column = variable ID   • first row = sample IDs
expr = pd.read_csv(matrix_fp, index_col=0)

# ---- standardise variable names  ( X1077.0 → 1077 ) -------------------------
expr.index = (expr.index.astype(str)
                         .str.lstrip("X")
                         .str.replace(r"\.0$", "", regex=True))
expr = expr[~expr.index.duplicated(keep="first")]

samples = expr.columns.astype(str)
labels  = np.where(pd.Series(samples)
                     .str.contains("cancer", case=False, na=False),
                   "Cancer", "Control")

# -------- stratified outer split ------------------------------------------
sss = StratifiedShuffleSplit(n_splits=1,
                             test_size=TEST_FRAC,
                             random_state=OUTER_SEED)
tr_idx, te_idx = next(sss.split(samples, labels))
train_ids, test_ids = samples[tr_idx], samples[te_idx]

train_fp.parent.mkdir(parents=True, exist_ok=True)
train_fp.write_text("\n".join(train_ids))
test_fp.write_text("\n".join(test_ids))
logging.info("outer%02d  TRAIN=%d  TEST=%d",
             OUTER_SEED, len(train_ids), len(test_ids))
