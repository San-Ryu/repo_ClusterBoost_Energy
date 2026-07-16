#!/usr/bin/env bash
set -euo pipefail

# Stop ML/DL batch runners using pid files.
# Usage:
#   bash src/stop_batches.sh [all|ml|dl|stage6]

MODE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT_DIR}/results/runtime}"

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "[STOP] ${label}: no pid file (${pid_file})"
    return
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "[STOP] ${label}: empty pid file (${pid_file})"
    rm -f "${pid_file}" || true
    return
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    echo "[STOP] ${label}: terminated pid=${pid}"
  else
    echo "[STOP] ${label}: pid already dead (${pid})"
  fi
  rm -f "${pid_file}" || true
}

cd "${ROOT_DIR}"
case "${MODE}" in
  all)
    stop_pid_file "ML" "${RUNTIME_DIR}/ml_batch.pid"
    stop_pid_file "DL" "${RUNTIME_DIR}/dl_batch.pid"
    stop_pid_file "STAGE6" "${RUNTIME_DIR}/stage6_checkpointed.pid"
    ;;
  ml)
    stop_pid_file "ML" "${RUNTIME_DIR}/ml_batch.pid"
    ;;
  dl)
    stop_pid_file "DL" "${RUNTIME_DIR}/dl_batch.pid"
    ;;
  stage6)
    stop_pid_file "STAGE6" "${RUNTIME_DIR}/stage6_checkpointed.pid"
    ;;
  *)
    echo "[ERROR] Invalid mode: ${MODE}" >&2
    echo "Usage: bash src/stop_batches.sh [all|ml|dl|stage6]" >&2
    exit 1
    ;;
esac
