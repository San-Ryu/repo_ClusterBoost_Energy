#!/usr/bin/env bash
set -euo pipefail

# Stage 6 ML runner with feature cache and trial-level checkpoints.
# Safe to stop and rerun: completed trials and households are skipped.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[ERROR] .venv not found: ${ROOT_DIR}/.venv/bin/activate" >&2
  exit 2
fi

RUNTIME_DIR="${RUNTIME_DIR:-results/runtime}"
mkdir -p "${RUNTIME_DIR}"
PID_FILE="${RUNTIME_DIR}/stage6_checkpointed.pid"
LOG_FILE="${RUNTIME_DIR}/stage6_checkpointed_$(date "+%Y%m%d_%H%M%S").log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] checkpointed Stage 6 already running with pid=${OLD_PID}" >&2
    exit 3
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
STAGE6_TEE_LOG="${STAGE6_TEE_LOG:-0}"
echo "[STAGE6] log_file=${LOG_FILE}"
if [[ "${STAGE6_TEE_LOG}" == "1" ]]; then
  exec > >(tee -a "${LOG_FILE}") 2>&1
else
  exec >> "${LOG_FILE}" 2>&1
fi

STAGE6_MODELS="${STAGE6_MODELS:-catboost,decisiontree,lightgbm,randomforest,xgboost}"
STAGE6_STEP="${STAGE6_STEP:-train}"
STAGE6_N_HOUSEHOLDS="${STAGE6_N_HOUSEHOLDS:-30}"
STAGE6_N_ITER="${STAGE6_N_ITER:-120}"
STAGE6_CV_SPLITS="${STAGE6_CV_SPLITS:-5}"
STAGE6_MAX_HOUSEHOLDS_PER_RUN="${STAGE6_MAX_HOUSEHOLDS_PER_RUN:-1}"
STAGE6_MAX_TRIALS_PER_RUN="${STAGE6_MAX_TRIALS_PER_RUN:-1}"
TARGET_MODE="${TARGET_MODE:-delta}"
SCORING="${SCORING:-r2}"
STAGE6_EXTRA_FLAGS="${STAGE6_EXTRA_FLAGS:-}"
USE_CAFFEINATE="${USE_CAFFEINATE:-1}"
STAGE6_LEAKAGE_CHECK="${STAGE6_LEAKAGE_CHECK:-1}"
STAGE6_PEAK_QUANTILE="${STAGE6_PEAK_QUANTILE:-0.95}"
STAGE6_FEATURE_MODE="${STAGE6_FEATURE_MODE:-single}"
STAGE6_PEER_LAGS="${STAGE6_PEER_LAGS:-1}"
STAGE6_CLUSTER_DIR="${STAGE6_CLUSTER_DIR:-${KIER_DATA_ROOT:-${HOME}/data}/KIER_4_Clustered/seasonal_naive/kmeans}"
STAGE6_CLUSTER_RESOLUTION="${STAGE6_CLUSTER_RESOLUTION:-10MIN}"

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
    caffeinate -dimsu python -u "$@"
  else
    python -u "$@"
  fi
}

echo "[STAGE6] start=$(date "+%Y-%m-%d %H:%M:%S")"
echo "[STAGE6] step=${STAGE6_STEP} models=${STAGE6_MODELS}"
echo "[STAGE6] n_households=${STAGE6_N_HOUSEHOLDS} n_iter=${STAGE6_N_ITER} cv_splits=${STAGE6_CV_SPLITS}"
echo "[STAGE6] max_households_per_run=${STAGE6_MAX_HOUSEHOLDS_PER_RUN} max_trials_per_run=${STAGE6_MAX_TRIALS_PER_RUN}"
echo "[STAGE6] target_mode=${TARGET_MODE} scoring=${SCORING}"
echo "[STAGE6] leakage_check=${STAGE6_LEAKAGE_CHECK}"
echo "[STAGE6] peak_quantile=${STAGE6_PEAK_QUANTILE}"
echo "[STAGE6] feature_mode=${STAGE6_FEATURE_MODE} peer_lags=${STAGE6_PEER_LAGS}"
echo "[STAGE6] log_file=${LOG_FILE}"

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${STAGE6_LEAKAGE_CHECK}" == "1" && ( "${STAGE6_STEP}" == "train" || "${STAGE6_STEP}" == "all" ) ]]; then
  echo "[STAGE6] Running leakage diagnostics before training"
  run_python src/diagnose_ml_leakage.py \
    --n-households "${STAGE6_N_HOUSEHOLDS}" \
    --target-mode "${TARGET_MODE}" \
    --feature-mode "${STAGE6_FEATURE_MODE}" \
    --peer-lags "${STAGE6_PEER_LAGS}" \
    --cluster-dir "${STAGE6_CLUSTER_DIR}" \
    --cluster-resolution "${STAGE6_CLUSTER_RESOLUTION}" \
    --force-feature-cache \
    --fail-on exact
fi

IFS=',' read -r -a picked_models <<< "${STAGE6_MODELS}"
for key in "${picked_models[@]}"; do
  model="$(model_arg "${key}")"
  echo "============================================================"
  echo "[STAGE6] Running ${model} at $(date "+%Y-%m-%d %H:%M:%S")"
  echo "============================================================"
  run_python src/model_ml_checkpointed.py \
    --model "${model}" \
    --step "${STAGE6_STEP}" \
    --n-households "${STAGE6_N_HOUSEHOLDS}" \
    --n-iter "${STAGE6_N_ITER}" \
    --cv-splits "${STAGE6_CV_SPLITS}" \
    --max-households-per-run "${STAGE6_MAX_HOUSEHOLDS_PER_RUN}" \
    --max-trials-per-run "${STAGE6_MAX_TRIALS_PER_RUN}" \
    --peak-quantile "${STAGE6_PEAK_QUANTILE}" \
    --feature-mode "${STAGE6_FEATURE_MODE}" \
    --peer-lags "${STAGE6_PEER_LAGS}" \
    --cluster-dir "${STAGE6_CLUSTER_DIR}" \
    --cluster-resolution "${STAGE6_CLUSTER_RESOLUTION}" \
    --target-mode "${TARGET_MODE}" \
    --scoring "${SCORING}" \
    ${STAGE6_EXTRA_FLAGS}
done

echo "[STAGE6] end=$(date "+%Y-%m-%d %H:%M:%S") status=success"
