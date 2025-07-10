#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IO helpers used by all nested‑MCCV scripts.
The Singh matrix has:
  – row‑0  … sample IDs   ( “control.1”, “tumor.17”, … )
  – col‑0  … expression values (NOT gene names)
  – the rest: float expression values  (genes = rows)
No gene IDs are provided → we create pseudo‑IDs  g1 … g6033
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

# ------------------------------------------------------------------
def read_singh_matrix(csv_fp: Path) -> pd.DataFrame:
    """Return a dataframe  (genes × samples)  with synthetic gene IDs."""
    logging.debug("reading matrix %s", csv_fp)
    raw = pd.read_csv(csv_fp, header=None)

    # sample names are the first row, starting in column‑1
    sample_ids = raw.iloc[0, 1:].astype(str).str.strip().values
    n_samples  = len(sample_ids)

    # expression block = everything below row‑0, columns 1: end
    expr = raw.iloc[1:, 1:].astype(float).copy()
    expr.columns = sample_ids

    # build gene IDs  g1 … gn  (so we have unique rownames for limma)
    expr.index = [f"g{i+1}" for i in range(expr.shape[0])]

    logging.debug("matrix shape genes=%d  samples=%d", *expr.shape)
    return expr
