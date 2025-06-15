#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: measure.sh <NUM_CORES>" >&2; exit 1
fi
CORES=$1

SNK=/opt/conda/envs/omics-thesis/bin/snakemake

# ---------------------------------------------------------------- unlock first
$SNK -s workflow/Snakefile --unlock --cores 1 --quiet || true

OUT="results/prostmat/stage1/wall_clock_${CORES}.tsv"
mkdir -p "$(dirname "$OUT")"

# --------------------------------------------------------------- rebuild stage‑1
/usr/bin/time -f "%e" -o "$OUT.tmp" \
  "$SNK" -s workflow/Snakefile \
    --cores "$CORES" --use-conda \
    --config run_cores="$CORES" \
    --forcerun mccv_stage1 aggregate_stage1 record_wall_clock \
    --quiet

printf "%d\t%.3f\n" "$CORES" "$(cat "$OUT.tmp")" > "$OUT"
rm "$OUT.tmp"
echo "[measure] cores=$CORES  wall_clock=$(cut -f2 "$OUT")"
