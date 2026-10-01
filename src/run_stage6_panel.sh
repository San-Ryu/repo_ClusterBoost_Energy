#!/usr/bin/env bash
set -euo pipefail

# Stage 6 panel runner.
# Control: STAGE6_PANEL_MODE=global
# Experiment: STAGE6_PANEL_MODE=cluster

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[ERROR] .venv not found: ${ROOT_DIR}/.venv/bin/activate" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

RUNTIME_DIR="${RUNTIME_DIR:-results/runtime}"
mkdir -p "${RUNTIME_DIR}"
PID_FILE="${STAGE6_PID_FILE:-${RUNTIME_DIR}/stage6_panel.pid}"
LOG_FILE="${RUNTIME_DIR}/stage6_panel_$(date "+%Y%m%d_%H%M%S").log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] Stage 6 panel already running with pid=${OLD_PID}" >&2
    exit 3
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

STAGE6_TEE_LOG="${STAGE6_TEE_LOG:-0}"
echo "[STAGE6-PANEL] log_file=${LOG_FILE}"
if [[ "${STAGE6_TEE_LOG}" == "1" ]]; then
  exec > >(tee -a "${LOG_FILE}") 2>&1
else
  exec >> "${LOG_FILE}" 2>&1
fi

STAGE6_MODELS="${STAGE6_MODELS:-catboost,lightgbm,xgboost}"
STAGE6_STEP="${STAGE6_STEP:-train}"
STAGE6_PANEL_MODE="${STAGE6_PANEL_MODE:-global}"
STAGE6_N_HOUSEHOLDS="${STAGE6_N_HOUSEHOLDS:-348}"
STAGE6_N_ITER="${STAGE6_N_ITER:-60}"
STAGE6_CV_SPLITS="${STAGE6_CV_SPLITS:-5}"
STAGE6_MAX_GROUPS_PER_RUN="${STAGE6_MAX_GROUPS_PER_RUN:-1}"
STAGE6_MAX_TRIALS_PER_RUN="${STAGE6_MAX_TRIALS_PER_RUN:-1}"
TARGET_MODE="${TARGET_MODE:-level}"
SCORING="${SCORING:-r2}"
USE_CAFFEINATE="${USE_CAFFEINATE:-1}"
STAGE6_CLUSTER_DIR="${STAGE6_CLUSTER_DIR:-${KIER_DATA_ROOT:-${HOME}/data}/KIER_4_Clustered/seasonal_naive/kmeans}"
STAGE6_CLUSTER_RESOLUTION="${STAGE6_CLUSTER_RESOLUTION:-10MIN}"
STAGE6_EXTRA_FLAGS="${STAGE6_EXTRA_FLAGS:-}"

model_arg() {
  case "$1" in
    catboost) echo "CatBoost" ;;
    decisiontree) echo "DecisionTree" ;;
    lightgbm) echo "LightGBM" ;;
    randomforest) echo "RandomForest" ;;
    xgboost) echo "XGBoost" ;;
    *)
      echo "[ERROR] Unknown model key: $1" >&2
      exit 4
      ;;
  esac
}

run_python() {
  if [[ "${USE_CAFFEINATE}" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu "${PYTHON_BIN}" -u "$@"
  else
    "${PYTHON_BIN}" -u "$@"
  fi
}

echo "[STAGE6-PANEL] start=$(date "+%Y-%m-%d %H:%M:%S")"
echo "[STAGE6-PANEL] step=${STAGE6_STEP} models=${STAGE6_MODELS}"
echo "[STAGE6-PANEL] panel_mode=${STAGE6_PANEL_MODE}"
echo "[STAGE6-PANEL] n_households=${STAGE6_N_HOUSEHOLDS} n_iter=${STAGE6_N_ITER} cv_splits=${STAGE6_CV_SPLITS}"
echo "[STAGE6-PANEL] max_groups_per_run=${STAGE6_MAX_GROUPS_PER_RUN} max_trials_per_run=${STAGE6_MAX_TRIALS_PER_RUN}"
echo "[STAGE6-PANEL] target_mode=${TARGET_MODE} scoring=${SCORING}"
echo "[STAGE6-PANEL] cluster_dir=${STAGE6_CLUSTER_DIR} cluster_resolution=${STAGE6_CLUSTER_RESOLUTION}"
echo "[STAGE6-PANEL] log_file=${LOG_FILE}"

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

IFS=',' read -r -a picked_models <<< "${STAGE6_MODELS}"
for key in "${picked_models[@]}"; do
  model="$(model_arg "${key}")"
  echo "============================================================"
  echo "[STAGE6-PANEL] Running ${model} at $(date "+%Y-%m-%d %H:%M:%S")"
  echo "============================================================"
  run_python src/model_ml_panel_checkpointed.py \
    --model "${model}" \
    --step "${STAGE6_STEP}" \
    --panel-mode "${STAGE6_PANEL_MODE}" \
    --n-households "${STAGE6_N_HOUSEHOLDS}" \
    --n-iter "${STAGE6_N_ITER}" \
    --cv-splits "${STAGE6_CV_SPLITS}" \
    --max-groups-per-run "${STAGE6_MAX_GROUPS_PER_RUN}" \
    --max-trials-per-run "${STAGE6_MAX_TRIALS_PER_RUN}" \
    --cluster-dir "${STAGE6_CLUSTER_DIR}" \
    --cluster-resolution "${STAGE6_CLUSTER_RESOLUTION}" \
    --target-mode "${TARGET_MODE}" \
    --scoring "${SCORING}" \
    ${STAGE6_EXTRA_FLAGS}
done

echo "[STAGE6-PANEL] end=$(date "+%Y-%m-%d %H:%M:%S") status=success"
