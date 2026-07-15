#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper
# Usage:
#   bash src/run_all_models_batch.sh [all|ml|dl]

MODE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

case "${MODE}" in
  all)
    bash src/run_ml_batch.sh
    bash src/run_dl_batch.sh
    ;;
  ml)
    bash src/run_ml_batch.sh
    ;;
  dl)
    bash src/run_dl_batch.sh
    ;;
  *)
    echo "[ERROR] Invalid mode: ${MODE}" >&2
    echo "Usage: bash src/run_all_models_batch.sh [all|ml|dl]" >&2
    exit 1
    ;;
esac
