#!/usr/bin/env bash
# measure.sh  <DATASET>  <CORES>...
# ------------------------------------------------------------------
# Example:
#   bash workflow/measure.sh CRC_microbiome 1 4 16 32 96
#
# What it does:
#   1) For each <CORES>, run Stage‑1 to produce per-core runtime tables
#      and record wall-clock seconds.
#   2) Merge per-core tables and render ALL runtime plots into Figures/.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <DATASET> <CORES> ..." >&2
  exit 1
fi

DS="$1"; shift
ENV=/content/mambaforge/envs/omics-thesis
SNAKEMAKE="micromamba run --prefix $ENV snakemake"
CFG=workflow/config.yaml
SF=workflow/Snakefile

# 1) build per-core stage1 summary
for CORES in "$@"; do
  echo "[measure] ⇒ dataset=${DS}  cores=${CORES}"

  /usr/bin/time -f "%e" -o wall.tmp \
    $SNAKEMAKE \
      -s "$SF" --configfile "$CFG" \
      --cores "$CORES" \
      "results/${DS}/stage1/cores${CORES}/stage1_summary.tsv"

  WALL=$(cat wall.tmp); rm wall.tmp
  mkdir -p "results/${DS}/stage1"
  printf "cores\twall_clock_s\n%d\t%s\n" "$CORES" "$WALL" \
    > "results/${DS}/stage1/wall_clock_${CORES}.tsv"
  echo "[measure]  ${CORES} cores  →  ${WALL}s"
done

# 2) aggregate across cores
$SNAKEMAKE \
  -s "$SF" --configfile "$CFG" --cores 1 \
  "results/${DS}/stage1/wall_clock_all.tsv" \
  "results/${DS}/stage1/model_runtime_vs_cores.tsv" \
  "Figures/${DS}/runtime/Train_mean_vs_K.png" \
  "Figures/${DS}/runtime/Pred_mean_vs_K.png" \
  "Figures/${DS}/runtime/Train_total_vs_K.png" \
  "Figures/${DS}/runtime/Pred_total_vs_K.png" \
  "Figures/${DS}/runtime/Runtime_total_vs_K.png" \
  "Figures/${DS}/runtime/Wall_clock_vs_cores.png" \
  "Figures/${DS}/runtime/Speed_up.png" \
  "Figures/${DS}/runtime/Train_total_vs_cores.png" \
  "Figures/${DS}/runtime/Pred_total_vs_cores.png" \
  "Figures/${DS}/runtime/Train_total_vs_cores_zoom.png"
