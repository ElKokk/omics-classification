#!/usr/bin/env bash
# Usage: bash measure.sh 1 2 4 8 …
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CORES> [<CORES> …]" >&2
  exit 1
fi

SNK="$(conda run -n omics-thesis which snakemake)"
$SNK -s workflow/nested/Snakefile --unlock --cores 1 --quiet || true

for CORES in "$@"; do
  OUT="results/prostmat/runtime/wall_clock_${CORES}.tsv"
  mkdir -p "$(dirname "$OUT")"
  echo "[measure] CORES=${CORES}"
  /usr/bin/time -f "%e" -o "${OUT}.tmp" \
     "$SNK" -s workflow/nested/Snakefile \
        --cores "${CORES}" --use-conda \
        --config run_cores="${CORES}" \
        --rerun-incomplete --quiet
  printf "cores\twall_clock_s\n%d\t%.3f\n" "${CORES}" "$(cat "${OUT}.tmp")" > "${OUT}"
  rm -f "${OUT}.tmp"
done
