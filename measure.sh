#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <NUM_CORES> [<NUM_CORES> ...]" >&2; exit 1
fi
SNK="$(conda run -n omics-thesis which snakemake)"
$SNK -s workflow/Snakefile --unlock --cores 1 --quiet || true

for CORES in "$@"; do
  OUT="results/prostmat/stage1/wall_clock_${CORES}.tsv"
  mkdir -p "$(dirname "$OUT")"
  echo "[measure] running CORES=${CORES}"; rm -f "${OUT}.tmp"

  set +e
  /usr/bin/time -f "%e" -o "${OUT}.tmp" \
    "$SNK" -s workflow/Snakefile \
      --cores "${CORES}" --use-conda \
      --config run_cores="${CORES}" \
      --rerun-incomplete \
      --forcerun mccv_stage1 aggregate_stage1 record_wall_clock \
      --quiet
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "[measure] ERROR: snakemake failed for CORES=${CORES}" >&2
    exit 1
  fi

  printf "cores\twall_clock_s\n%d\t%.3f\n" "${CORES}" "$(cat "${OUT}.tmp")" > "${OUT}"
  rm "${OUT}.tmp"
  echo "[measure] cores=${CORES}  wall_clock=$(cut -f2 "${OUT}")"
done
