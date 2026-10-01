#!/usr/bin/env bash
set -euo pipefail

# Batch runner for paper-equivalent aggregate DL experiments.
#
# Usage:
#   bash src/run_legacy_aggregate_dl_batch.sh --step plan
#   DL_AGG_MODELS=legacy_cnnlstm,tcn bash src/run_legacy_aggregate_dl_batch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNTIME_DIR="${RUNTIME_DIR:-results/runtime}"
mkdir -p "${RUNTIME_DIR}"
PID_FILE="${RUNTIME_DIR}/legacy_aggregate_dl_batch.pid"
LOG_FILE="${RUNTIME_DIR}/legacy_aggregate_dl_batch_$(date "+%Y%m%d_%H%M%S").log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] Legacy aggregate DL batch already running with pid=${OLD_PID}" >&2
    exit 3
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "[DL-AGG-BATCH] log_file=${LOG_FILE}"
exec >> "${LOG_FILE}" 2>&1

DL_AGG_MODELS="${DL_AGG_MODELS:-all}"
PYTHON_BIN="${PYTHON_BIN:-}"

all_scripts=(
  src/run_legacy_aggregate_dl_legacy_cnnlstm.py
  src/run_legacy_aggregate_dl_legacy_seq2seq.py
  src/run_legacy_aggregate_dl_cnnlstm.py
  src/run_legacy_aggregate_dl_gru.py
  src/run_legacy_aggregate_dl_transformer.py
  src/run_legacy_aggregate_dl_tcn.py
  src/run_legacy_aggregate_dl_retnet.py
)

scripts=()
if [[ "${DL_AGG_MODELS}" == "all" ]]; then
  scripts=("${all_scripts[@]}")
else
  IFS=',' read -r -a picked <<< "${DL_AGG_MODELS}"
  for model_key in "${picked[@]}"; do
    case "${model_key}" in
      legacy_cnnlstm) candidate="src/run_legacy_aggregate_dl_legacy_cnnlstm.py" ;;
      legacy_seq2seq) candidate="src/run_legacy_aggregate_dl_legacy_seq2seq.py" ;;
      cnnlstm) candidate="src/run_legacy_aggregate_dl_cnnlstm.py" ;;
      gru) candidate="src/run_legacy_aggregate_dl_gru.py" ;;
      transformer) candidate="src/run_legacy_aggregate_dl_transformer.py" ;;
      tcn) candidate="src/run_legacy_aggregate_dl_tcn.py" ;;
      retnet) candidate="src/run_legacy_aggregate_dl_retnet.py" ;;
      *)
        candidate=""
        echo "[DL-AGG-BATCH][WARN] unknown model key: ${model_key}"
        ;;
    esac
    if [[ -n "${candidate}" ]] && [[ -f "${candidate}" ]]; then
      scripts+=("${candidate}")
    fi
  done
fi

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "[ERROR] No valid aggregate DL scripts selected. Check DL_AGG_MODELS." >&2
  exit 4
fi

FAILED=()
echo "[DL-AGG-BATCH] start=$(date "+%Y-%m-%d %H:%M:%S")"
echo "[DL-AGG-BATCH] models=${scripts[*]}"
echo "[DL-AGG-BATCH] log_file=${LOG_FILE}"

for script in "${scripts[@]}"; do
  echo "============================================================"
  echo "[DL-AGG] Running ${script} at $(date "+%Y-%m-%d %H:%M:%S")"
  echo "============================================================"
  if [[ -n "${PYTHON_BIN}" ]]; then
    if ! "${PYTHON_BIN}" -u "${script}" "$@"; then
      FAILED+=("${script}")
      echo "[DL-AGG][WARN] failed: ${script}"
    fi
  else
    if ! python -u "${script}" "$@"; then
      FAILED+=("${script}")
      echo "[DL-AGG][WARN] failed: ${script}"
    fi
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[DL-AGG-BATCH] end=$(date "+%Y-%m-%d %H:%M:%S") status=partial_failure"
  printf '[DL-AGG-BATCH] failed_models=%s\n' "${FAILED[*]}"
  exit 1
fi

echo "[DL-AGG-BATCH] end=$(date "+%Y-%m-%d %H:%M:%S") status=success"
