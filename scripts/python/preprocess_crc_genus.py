#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert CRC microbiome into labeled-table format.

Output (snakemake.output['matrix']):
    data/processed/CRC_microbiome_matrix.csv
Columns:
    Samples | Group | <numeric features...>
Notes:
    • Group is 0/1 (0=Healthy/Control, 1=CRC/Cancer)
    • Robust header
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ── Snakemake handles ───────────────────────────────────────────────────────
in_fp  = Path(snakemake.input[0])
out_fp = Path(snakemake.output["matrix"])
sheet_param = snakemake.params.get("sheet", None)  # int or str; default -> first sheet
id_candidates    = snakemake.params.get("id_cols", ["ID", "Sample", "Samples", "Subject", "SampleID", "Participant"])
label_candidates = snakemake.params.get("label_cols", [
    "Group", "Groups", "Label", "Group_bin",
    "Class", "Status", "Diagnosis", "Case_Control",
    "CaseControl", "Phenotype", "CRC", "Disease", "Outcome", "Y"
])

# ── utilities ───────────────────────────────────────────────────────────────
def norm(s: str) -> str:
    """Normalize a header for fuzzy matching."""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    s = s.replace("-", "_")
    s = re.sub(r"[():/]", " ", s)
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    return s

def pick_header(df: pd.DataFrame, synonyms, prefer_substrings=("group","diagnos","status","case","crc")):
    """Pick a column by synonyms (exact normalized match, then substring preference)."""
    norm_map = {norm(c): c for c in df.columns}
    # 1) exact match on normalized name
    for cand in synonyms:
        nc = norm(cand)
        if nc in norm_map:
            return norm_map[nc]
    # 2) substring preference
    for nc, orig in norm_map.items():
        if any(tok in nc for tok in prefer_substrings):
            return orig
    return None

def map_text_label_to01(series: pd.Series) -> pd.Series:
    pos_tokens = {"crc","cancer","case","tumor","tumour","patient","disease","malignant","pos","positive","yes","y","one","1"}
    neg_tokens = {"healthy","control","normal","hc","benign","neg","negative","no","n","zero","0"}
    out = []
    for v in series.astype(str):
        t = norm(v)
        t = t.replace(" ", "")
        if t in pos_tokens:
            out.append(1)
        elif t in neg_tokens:
            out.append(0)
        else:
            # allow things like "1-crc" "0-healthycontrol"
            if "crc" in t or "cancer" in t or "case" in t or "tum" in t:
                out.append(1)
            elif "healthy" in t or "control" in t or "normal" in t or t.startswith("0"):
                out.append(0)
            else:
                out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="float")

def coerce_binary01(col: pd.Series) -> pd.Series:
    """Convert a column to binary 0/1 if possible; returns float series with NaN if not possible."""
    # Try numeric first
    s = pd.to_numeric(col, errors="coerce")
    uniq = set(int(x) for x in np.unique(s.dropna()))
    if len(uniq) <= 3 and uniq.issubset({0,1,2}):
        if uniq.issubset({0,1}):
            return s.astype(float)
        if uniq == {1,2}:
            return (s - 1).astype(float)
        if uniq == {0,2}:
            return (s/2).astype(float)
    # Try text mapping
    return map_text_label_to01(col)

def _read_excel_one_sheet(xlsx_path: Path, sheet):
    """Read a single sheet. If dict returned, take first frame."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    return df

# ── load ────────────────────────────────────────────────────────────────────
if sheet_param in (None, "", -1):
    sheet_to_read = 0
else:
    sheet_to_read = sheet_param

if in_fp.suffix.lower() in [".xlsx", ".xls"]:
    df = _read_excel_one_sheet(in_fp, sheet_to_read)
else:
    df = pd.read_csv(in_fp)

if not isinstance(df, pd.DataFrame) or df.empty:
    raise ValueError(f"{in_fp} is empty or not a table")

print(f"[crc-preprocess] sheet={sheet_to_read!r}  n_rows={len(df)}  n_cols={df.shape[1]}")
print(f"[crc-preprocess] headers: {list(df.columns)}")

# ── pick ID and Label columns ───────────────────────────────────────────────
id_col  = pick_header(df, id_candidates, prefer_substrings=("id","sample","subject","participant"))
lab_col = pick_header(df, label_candidates, prefer_substrings=("group","diagnos","status","case","crc"))

if id_col is None:
    id_col = "__SampleID__"
    df[id_col] = [f"S{i+1}" for i in range(len(df))]


if lab_col is None:
    candidates = []
    for c in df.columns:
        if c == id_col: 
            continue
        y01 = coerce_binary01(df[c])
        vals = set(y01.dropna().unique().tolist())
        if vals.issubset({0.0,1.0}) and len(vals) == 2:
            candidates.append((c, "binary"))
    if candidates:
        lab_col = candidates[0][0]
        print(f"[crc-preprocess] picked label column by content: {lab_col!r}")
    else:
        raise ValueError(
            "Could not find a label column. Tried common names and could not "
            "detect any binary 0/1 column. If your label has a different name, "
            "add it to params.label_cols in the rule or rename the column."
        )

print(f"[crc-preprocess] using id_col={id_col!r}  label_col={lab_col!r}")

# ── build labeled table ─────────────────────────────────────────────────────
labels01 = coerce_binary01(df[lab_col])
if labels01.isna().any() or set(labels01.dropna().unique()) - {0.0, 1.0}:
    raise ValueError(
        f"Label column {lab_col!r} could not be coerced to 0/1. "
        "Ensure it is coded as 0/1 or values like CRC/Healthy."
    )
labels01 = labels01.astype("Int64")

front = [id_col, lab_col]
feature_cols = [c for c in df.columns if c not in front]

# Coerce features to numeric; drop all-NA features; fill remaining NA with 0.0
num = df[feature_cols].apply(pd.to_numeric, errors="coerce")
all_na = [c for c in num.columns if num[c].isna().all()]
if all_na:
    print(f"[crc-preprocess] dropping all-NA features: {all_na}")
    num = num.drop(columns=all_na)
num = num.fillna(0.0)

out = pd.concat(
    [df[id_col].astype(str).rename("Samples"),
     labels01.rename("Group"),
     num],
    axis=1
)

out_fp.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_fp, index=False)
print(f"[crc-preprocess] wrote → {out_fp}  (rows={len(out)}, features={out.shape[1]-2})")
