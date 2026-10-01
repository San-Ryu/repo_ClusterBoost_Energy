#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-global}"
if [[ "${MODE}" != "global" && "${MODE}" != "cluster" ]]; then
  echo "Usage: $0 [global|cluster]" >&2
  exit 2
fi

N_ITER="${STAGE6_N_ITER:-60}"
CV_SPLITS="${STAGE6_CV_SPLITS:-5}"
N_HOUSEHOLDS="${STAGE6_N_HOUSEHOLDS:-348}"
MAX_FAILURES="${STAGE6_MAX_FAILURES:-2}"
TARGET_MODE="${TARGET_MODE:-level}"
SCORING="${SCORING:-r2}"
LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"
STAGE6_CLUSTER_GROUPS="${STAGE6_CLUSTER_GROUPS:-2}"
RUN_PID_FILE="results/runtime/stage6_panel_lightgbm_${MODE}_resume_$$.pid"

if [[ "${MODE}" == "global" ]]; then
  RESULTS_DIR="results/stage6_panel_global"
  CHECKPOINT_DIR="results/stage6_panel_global/checkpoints"
  PREDICTIONS_DIR="results/stage6_panel_global/predictions"
else
  RESULTS_DIR="results/stage6_panel_cluster"
  CHECKPOINT_DIR="results/stage6_panel_cluster/checkpoints"
  PREDICTIONS_DIR="results/stage6_panel_cluster/predictions"
fi

count_done() {
  local mode="$1"
  local total=0
  local path lines
  if [[ "${mode}" == "global" ]]; then
    path="${CHECKPOINT_DIR}/lightgbm/global/ALL_trials.csv"
    if [[ -f "${path}" ]]; then
      lines="$(wc -l < "${path}" | tr -d ' ')"
      total=$((lines > 0 ? lines - 1 : 0))
    fi
  else
    for path in "${CHECKPOINT_DIR}/lightgbm/cluster/"*_trials.csv; do
      [[ -f "${path}" ]] || continue
      lines="$(wc -l < "${path}" | tr -d ' ')"
      total=$((total + (lines > 0 ? lines - 1 : 0)))
    done
  fi
  echo "${total}"
}

echo "[LGBM-RESUME] start=$(date '+%Y-%m-%d %H:%M:%S') mode=${MODE}"
echo "[LGBM-RESUME] n_households=${N_HOUSEHOLDS} n_iter=${N_ITER} cv_splits=${CV_SPLITS}"
echo "[LGBM-RESUME] one completed trial is checkpointed before the next run starts"

if [[ "${MODE}" == "global" ]]; then
  TARGET_TOTAL="${N_ITER}"
else
  TARGET_TOTAL=$((N_ITER * STAGE6_CLUSTER_GROUPS))
fi

failures=0
while true; do
  before="$(count_done "${MODE}")"
  if [[ "${before}" -ge "${TARGET_TOTAL}" ]]; then
    break
  fi

  echo "[LGBM-RESUME] before=${before} at $(date '+%Y-%m-%d %H:%M:%S')"
  set +e
  LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT}" \
  STAGE6_PID_FILE="${RUN_PID_FILE}" \
  USE_CAFFEINATE=0 \
  STAGE6_PANEL_MODE="${MODE}" \
  STAGE6_MODELS=lightgbm \
  STAGE6_STEP=train \
  STAGE6_N_HOUSEHOLDS="${N_HOUSEHOLDS}" \
  STAGE6_N_ITER="${N_ITER}" \
  STAGE6_CV_SPLITS="${CV_SPLITS}" \
  STAGE6_MAX_GROUPS_PER_RUN=1 \
  STAGE6_MAX_TRIALS_PER_RUN=1 \
  TARGET_MODE="${TARGET_MODE}" \
  SCORING="${SCORING}" \
  STAGE6_EXTRA_FLAGS="--results-dir ${RESULTS_DIR} --checkpoint-dir ${CHECKPOINT_DIR} --predictions-dir ${PREDICTIONS_DIR}" \
  bash src/run_stage6_panel.sh
  rc=$?
  set -e

  after="$(count_done "${MODE}")"
  echo "[LGBM-RESUME] after=${after} rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S')"

  if [[ "${after}" -gt "${before}" ]]; then
    failures=0
  else
    failures=$((failures + 1))
    echo "[LGBM-RESUME] no checkpoint progress, failures=${failures}/${MAX_FAILURES}" >&2
    if [[ "${failures}" -ge "${MAX_FAILURES}" ]]; then
      echo "[LGBM-RESUME] stopping after repeated no-progress runs" >&2
      exit 1
    fi
  fi

  if [[ "${after}" -ge "${TARGET_TOTAL}" ]]; then
    break
  fi
done

echo "[LGBM-RESUME] train complete for mode=${MODE}; running finalize"
USE_CAFFEINATE=0 \
STAGE6_PID_FILE="${RUN_PID_FILE}" \
STAGE6_PANEL_MODE="${MODE}" \
STAGE6_MODELS=lightgbm \
STAGE6_STEP=finalize \
STAGE6_N_HOUSEHOLDS="${N_HOUSEHOLDS}" \
STAGE6_N_ITER="${N_ITER}" \
STAGE6_CV_SPLITS="${CV_SPLITS}" \
TARGET_MODE="${TARGET_MODE}" \
SCORING="${SCORING}" \
STAGE6_EXTRA_FLAGS="--results-dir ${RESULTS_DIR} --checkpoint-dir ${CHECKPOINT_DIR} --predictions-dir ${PREDICTIONS_DIR}" \
bash src/run_stage6_panel.sh

echo "[LGBM-RESUME] done mode=${MODE} at $(date '+%Y-%m-%d %H:%M:%S')"
