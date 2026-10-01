#!/usr/bin/env bash
set -euo pipefail

# Batch runner for DL peer-household grid-search CV experiments.
#
# Smoke example:
#   DL_PEER_MODELS=gru,tcn bash src/run_dl_peer_grid_batch.sh --grid-preset smoke --max-rows 120
#
# Full small-grid example:
#   DL_PEER_MODELS=all bash src/run_dl_peer_grid_batch.sh --grid-preset small --n-target-households 3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNTIME_DIR="${RUNTIME_DIR:-results/runtime}"
mkdir -p "${RUNTIME_DIR}"
PID_FILE="${RUNTIME_DIR}/dl_peer_grid_batch.pid"
LOG_FILE="${RUNTIME_DIR}/dl_peer_grid_batch_$(date "+%Y%m%d_%H%M%S").log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[ERROR] DL peer grid batch already running with pid=${OLD_PID}" >&2
    exit 3
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[DL-PEER-BATCH] log_file=${LOG_FILE}"
exec >> "${LOG_FILE}" 2>&1

DL_PEER_MODELS="${DL_PEER_MODELS:-all}"
PYTHON_BIN="${PYTHON_BIN:-}"

all_scripts=(
  src/run_dl_peer_legacy_cnnlstm.py
  src/run_dl_peer_legacy_seq2seq.py
  src/run_dl_peer_cnnlstm.py
  src/run_dl_peer_gru.py
  src/run_dl_peer_transformer.py
  src/run_dl_peer_tcn.py
  src/run_dl_peer_retnet.py
)

scripts=()
if [[ "${DL_PEER_MODELS}" == "all" ]]; then
  scripts=("${all_scripts[@]}")
else
  IFS=',' read -r -a picked <<< "${DL_PEER_MODELS}"
  for model_key in "${picked[@]}"; do
    case "${model_key}" in
      legacy_cnnlstm) candidate="src/run_dl_peer_legacy_cnnlstm.py" ;;
      legacy_seq2seq) candidate="src/run_dl_peer_legacy_seq2seq.py" ;;
      cnnlstm) candidate="src/run_dl_peer_cnnlstm.py" ;;
      gru) candidate="src/run_dl_peer_gru.py" ;;
      transformer) candidate="src/run_dl_peer_transformer.py" ;;
      tcn) candidate="src/run_dl_peer_tcn.py" ;;
      retnet) candidate="src/run_dl_peer_retnet.py" ;;
      *)
        candidate=""
        echo "[DL-PEER-BATCH][WARN] unknown model key: ${model_key}"
        ;;
    esac
    if [[ -n "${candidate}" ]] && [[ -f "${candidate}" ]]; then
      scripts+=("${candidate}")
    fi
  done
fi

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "[ERROR] No valid peer DL scripts selected. Check DL_PEER_MODELS." >&2
  exit 4
fi

FAILED=()
echo "[DL-PEER-BATCH] start=$(date "+%Y-%m-%d %H:%M:%S")"
echo "[DL-PEER-BATCH] models=${scripts[*]}"
echo "[DL-PEER-BATCH] args=$*"
echo "[DL-PEER-BATCH] log_file=${LOG_FILE}"

for script in "${scripts[@]}"; do
  echo "============================================================"
  echo "[DL-PEER] Running ${script} at $(date "+%Y-%m-%d %H:%M:%S")"
  echo "============================================================"
  if [[ -n "${PYTHON_BIN}" ]]; then
    if ! "${PYTHON_BIN}" -u "${script}" "$@"; then
      FAILED+=("${script}")
      echo "[DL-PEER][WARN] failed: ${script}"
    fi
  else
    if ! python -u "${script}" "$@"; then
      FAILED+=("${script}")
      echo "[DL-PEER][WARN] failed: ${script}"
    fi
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[DL-PEER-BATCH] end=$(date "+%Y-%m-%d %H:%M:%S") status=partial_failure"
  printf '[DL-PEER-BATCH] failed_models=%s\n' "${FAILED[*]}"
  exit 1
fi

echo "[DL-PEER-BATCH] end=$(date "+%Y-%m-%d %H:%M:%S") status=success"
