#!/usr/bin/env bash
set -euo pipefail

# Recreate the previous repository's aggregate/cluster modeling protocol on
# the current project's preprocessed and clustered data.

FAMILY="${FAMILY:-ml}"
MODEL="${MODEL:-XGBoost}"
SCOPE="${SCOPE:-both}"
RESOLUTION="${RESOLUTION:-1H}"
INTERP_METHOD="${INTERP_METHOD:-seasonal_naive}"
CLUSTER_ALGORITHM="${CLUSTER_ALGORITHM:-kmeans}"
RESULTS_DIR="${RESULTS_DIR:-results/legacy_aggregate}"
TARGET_SHIFT="${TARGET_SHIFT:-1}"
TEST_RATIO="${TEST_RATIO:-0.3}"
CV_FOLDS="${CV_FOLDS:-10}"
SEQ_LEN="${SEQ_LEN:-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-0.001}"
FIT_VERBOSE="${FIT_VERBOSE:-2}"

PYTHON_BIN="${PYTHON_BIN:-python}"

args=(
  src/legacy_aggregate_experiment.py
  --family "${FAMILY}"
  --model "${MODEL}"
  --scope "${SCOPE}"
  --resolution "${RESOLUTION}"
  --interp-method "${INTERP_METHOD}"
  --cluster-algorithm "${CLUSTER_ALGORITHM}"
  --results-dir "${RESULTS_DIR}"
  --target-shift "${TARGET_SHIFT}"
  --test-ratio "${TEST_RATIO}"
  --cv-folds "${CV_FOLDS}"
  --seq-len "${SEQ_LEN}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --patience "${PATIENCE}"
  --lr "${LR}"
  --fit-verbose "${FIT_VERBOSE}"
)

if [[ "${CV_SHUFFLE:-0}" == "1" ]]; then
  args+=(--cv-shuffle)
fi

if [[ "${TIME_FEATURES:-1}" == "0" ]]; then
  args+=(--no-time-features)
fi

"${PYTHON_BIN}" "${args[@]}" "$@"
