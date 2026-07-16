#!/usr/bin/env bash
set -euo pipefail

# Dedicated ML batch runner
# Usage:
#   bash src/run_ml_batch.sh
# Example:
#   ML_N_HOUSEHOLDS=30 ML_N_ITER=120 ML_CV_SPLITS=5 TARGET_MODE=delta SCORING=r2 bash src/run_ml_batch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "[ERROR] .venv not found: ${ROOT_DIR}/.venv/bin/activate" >&2
  exit 2
fi

RUNTIME_DIR="${RUNTIME_DIR:-results/runtime}"
mkdir -p "${RUNTIME_DIR}"
PID_FILE="${RUNTIME_DIR}/ml_batch.pid"
LOG_FILE="${RUNTIME_DIR}/ml_batch_$(date "+%Y%m%d_%H%M%S").log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] ML batch already running with pid=${OLD_PID}" >&2
    echo "        stop first: kill -9 ${OLD_PID}" >&2
    exit 3
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
exec > >(tee -a "${LOG_FILE}") 2>&1

TARGET_MODE="${TARGET_MODE:-delta}"
SCORING="${SCORING:-r2}"
ML_STEP="${ML_STEP:-all}"
ML_N_HOUSEHOLDS="${ML_N_HOUSEHOLDS:-30}"
ML_N_ITER="${ML_N_ITER:-120}"
ML_CV_SPLITS="${ML_CV_SPLITS:-5}"
ML_N_JOBS="${ML_N_JOBS:-1}"
ML_START_INDEX="${ML_START_INDEX:-0}"
ML_MAX_HOUSEHOLDS_PER_RUN="${ML_MAX_HOUSEHOLDS_PER_RUN:-0}"
ML_EXTRA_FLAGS="${ML_EXTRA_FLAGS:-}"
ML_MODELS="${ML_MODELS:-all}"

all_ml_scripts=(
  src/run_ml_catboost.py
  src/run_ml_decisiontree.py
  src/run_ml_lightgbm.py
  src/run_ml_randomforest.py
  src/run_ml_xgboost.py
)

ML_SCRIPTS=()
if [[ "${ML_MODELS}" == "all" ]]; then
  ML_SCRIPTS=("${all_ml_scripts[@]}")
else
  IFS=',' read -r -a picked <<< "${ML_MODELS}"
  for m in "${picked[@]}"; do
    candidate="src/run_ml_${m}.py"
    if [[ -f "${candidate}" ]]; then
      ML_SCRIPTS+=("${candidate}")
    else
      echo "[ML-BATCH][WARN] unknown model key: ${m} (expected catboost|decisiontree|lightgbm|randomforest|xgboost)"
    fi
  done
fi
if [[ ${#ML_SCRIPTS[@]} -eq 0 ]]; then
  echo "[ERROR] No valid ML scripts selected. Check ML_MODELS." >&2
  exit 4
fi

FAILED=()
START_TS="$(date "+%Y-%m-%d %H:%M:%S")"
echo "[ML-BATCH] start=${START_TS}"
echo "[ML-BATCH] step=${ML_STEP} target_mode=${TARGET_MODE} scoring=${SCORING}"
echo "[ML-BATCH] n_households=${ML_N_HOUSEHOLDS} n_iter=${ML_N_ITER} cv_splits=${ML_CV_SPLITS} n_jobs=${ML_N_JOBS}"
echo "[ML-BATCH] start_index=${ML_START_INDEX} max_households_per_run=${ML_MAX_HOUSEHOLDS_PER_RUN}"
echo "[ML-BATCH] models=${ML_SCRIPTS[*]}"
echo "[ML-BATCH] log_file=${LOG_FILE}"

(
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

  for script in "${ML_SCRIPTS[@]}"; do
    echo "============================================================"
    echo "[ML] Running ${script} at $(date "+%Y-%m-%d %H:%M:%S")"
    echo "============================================================"
    if ! python -u "${script}" \
      --step "${ML_STEP}" \
      --n-households "${ML_N_HOUSEHOLDS}" \
      --n-iter "${ML_N_ITER}" \
      --cv-splits "${ML_CV_SPLITS}" \
      --n-jobs "${ML_N_JOBS}" \
      --start-index "${ML_START_INDEX}" \
      --max-households-per-run "${ML_MAX_HOUSEHOLDS_PER_RUN}" \
      --target-mode "${TARGET_MODE}" \
      --scoring "${SCORING}" \
      ${ML_EXTRA_FLAGS}; then
      FAILED+=("${script}")
      echo "[ML][WARN] failed: ${script}"
    else
      echo "[ML][DONE] ${script} at $(date "+%Y-%m-%d %H:%M:%S")"
    fi
  done
)

END_TS="$(date "+%Y-%m-%d %H:%M:%S")"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[ML-BATCH] end=${END_TS} status=partial_failure"
  printf '[ML-BATCH] failed_models=%s\n' "${FAILED[*]}"
  exit 1
fi

echo "[ML-BATCH] end=${END_TS} status=success"
