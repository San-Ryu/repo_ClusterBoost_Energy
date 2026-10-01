#!/usr/bin/env bash
set -euo pipefail

# Dedicated DL batch runner
# Usage:
#   bash src/run_dl_batch.sh
# Example:
#   DL_N_HOUSEHOLDS=30 DL_EPOCHS=120 DL_MAX_HOUSEHOLDS_PER_RUN=2 TARGET_MODE=delta bash src/run_dl_batch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x ".venv-dl/bin/python" ]]; then
  DL_BATCH_PYTHON=".venv-dl/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  DL_BATCH_PYTHON=".venv/bin/python"
else
  echo "[ERROR] No runnable DL Python found. Expected .venv-dl/bin/python or .venv/bin/python" >&2
  exit 2
fi

RUNTIME_DIR="${RUNTIME_DIR:-results/runtime}"
mkdir -p "${RUNTIME_DIR}"
PID_FILE="${RUNTIME_DIR}/dl_batch.pid"
LOG_FILE="${RUNTIME_DIR}/dl_batch_$(date "+%Y%m%d_%H%M%S").log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] DL batch already running with pid=${OLD_PID}" >&2
    echo "        stop first: kill -9 ${OLD_PID}" >&2
    exit 3
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "[DL-BATCH] log_file=${LOG_FILE}"
exec >> "${LOG_FILE}" 2>&1

TARGET_MODE="${TARGET_MODE:-delta}"
DL_STEP="${DL_STEP:-train}"
DL_N_HOUSEHOLDS="${DL_N_HOUSEHOLDS:-30}"
DL_EPOCHS="${DL_EPOCHS:-120}"
DL_PATIENCE="${DL_PATIENCE:-30}"
DL_SEQ_LEN="${DL_SEQ_LEN:-288}"
DL_BATCH_SIZE="${DL_BATCH_SIZE:-64}"
DL_MAX_HOUSEHOLDS_PER_RUN="${DL_MAX_HOUSEHOLDS_PER_RUN:-2}"
DL_FIT_VERBOSE="${DL_FIT_VERBOSE:-0}"
DL_EXTRA_FLAGS="${DL_EXTRA_FLAGS:-}"
DL_MODELS="${DL_MODELS:-all}"

all_dl_scripts=(
  src/run_dl_legacy_cnnlstm.py
  src/run_dl_legacy_seq2seq.py
  src/run_dl_cnnlstm.py
  src/run_dl_gru.py
  src/run_dl_transformer.py
  src/run_dl_tcn.py
  src/run_dl_retnet.py
)

DL_SCRIPTS=()
if [[ "${DL_MODELS}" == "all" ]]; then
  DL_SCRIPTS=("${all_dl_scripts[@]}")
else
  IFS=',' read -r -a picked <<< "${DL_MODELS}"
  for m in "${picked[@]}"; do
    case "${m}" in
      legacy_cnnlstm) candidate="src/run_dl_legacy_cnnlstm.py" ;;
      legacy_seq2seq) candidate="src/run_dl_legacy_seq2seq.py" ;;
      cnnlstm) candidate="src/run_dl_cnnlstm.py" ;;
      gru) candidate="src/run_dl_gru.py" ;;
      transformer) candidate="src/run_dl_transformer.py" ;;
      tcn) candidate="src/run_dl_tcn.py" ;;
      retnet) candidate="src/run_dl_retnet.py" ;;
      *)
        candidate=""
        echo "[DL-BATCH][WARN] unknown model key: ${m} (expected legacy_cnnlstm|legacy_seq2seq|cnnlstm|gru|transformer|tcn|retnet)"
        ;;
    esac
    if [[ -n "${candidate}" ]] && [[ -f "${candidate}" ]]; then
      DL_SCRIPTS+=("${candidate}")
    fi
  done
fi
if [[ ${#DL_SCRIPTS[@]} -eq 0 ]]; then
  echo "[ERROR] No valid DL scripts selected. Check DL_MODELS." >&2
  exit 4
fi

FAILED=()
START_TS="$(date "+%Y-%m-%d %H:%M:%S")"
echo "[DL-BATCH] start=${START_TS}"
echo "[DL-BATCH] step=${DL_STEP} target_mode=${TARGET_MODE}"
echo "[DL-BATCH] n_households=${DL_N_HOUSEHOLDS} epochs=${DL_EPOCHS} patience=${DL_PATIENCE} seq_len=${DL_SEQ_LEN}"
echo "[DL-BATCH] batch_size=${DL_BATCH_SIZE} max_households_per_run=${DL_MAX_HOUSEHOLDS_PER_RUN} fit_verbose=${DL_FIT_VERBOSE}"
echo "[DL-BATCH] models=${DL_SCRIPTS[*]}"
echo "[DL-BATCH] python=${DL_BATCH_PYTHON}"
echo "[DL-BATCH] log_file=${LOG_FILE}"

(
  export PYTHONUNBUFFERED=1

  for script in "${DL_SCRIPTS[@]}"; do
    echo "============================================================"
    echo "[DL] Running ${script} at $(date "+%Y-%m-%d %H:%M:%S")"
    echo "============================================================"
    # DL_EXTRA_FLAGS is optional for flags like --no-resume
    if ! "${DL_BATCH_PYTHON}" -u "${script}" \
      --step "${DL_STEP}" \
      --n-households "${DL_N_HOUSEHOLDS}" \
      --epochs "${DL_EPOCHS}" \
      --patience "${DL_PATIENCE}" \
      --seq-len "${DL_SEQ_LEN}" \
      --batch-size "${DL_BATCH_SIZE}" \
      --max-households-per-run "${DL_MAX_HOUSEHOLDS_PER_RUN}" \
      --fit-verbose "${DL_FIT_VERBOSE}" \
      --target-mode "${TARGET_MODE}" \
      ${DL_EXTRA_FLAGS}; then
      FAILED+=("${script}")
      echo "[DL][WARN] failed: ${script}"
    else
      echo "[DL][DONE] ${script} at $(date "+%Y-%m-%d %H:%M:%S")"
    fi
  done
)

END_TS="$(date "+%Y-%m-%d %H:%M:%S")"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[DL-BATCH] end=${END_TS} status=partial_failure"
  printf '[DL-BATCH] failed_models=%s\n' "${FAILED[*]}"
  exit 1
fi

echo "[DL-BATCH] end=${END_TS} status=success"
