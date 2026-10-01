#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

N_ITER="${STAGE6_N_ITER:-60}"

count_trials() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    local lines
    lines="$(wc -l < "${path}" | tr -d ' ')"
    if [[ "${lines}" -gt 0 ]]; then
      echo $((lines - 1))
    else
      echo 0
    fi
  else
    echo 0
  fi
}

show_trial_file() {
  local label="$1"
  local path="$2"
  local done
  done="$(count_trials "${path}")"
  printf '%-18s %3s/%s  %s\n' "${label}" "${done}" "${N_ITER}" "${path}"
  if [[ -f "${path}" && "${done}" -gt 0 ]]; then
    tail -1 "${path}" | cut -d',' -f1-6
  fi
}

echo "[Stage6 LGBM status] $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "[screen]"
screen -ls || true
echo
echo "[checkpoints]"
show_trial_file "global/ALL" "results/stage6_panel_global/checkpoints/lightgbm/global/ALL_trials.csv"
show_trial_file "cluster/K0" "results/stage6_panel_cluster/checkpoints/lightgbm/cluster/K0_trials.csv"
show_trial_file "cluster/K1" "results/stage6_panel_cluster/checkpoints/lightgbm/cluster/K1_trials.csv"
echo
echo "[latest logs]"
find results/runtime -maxdepth 1 -type f -name 'stage6_panel_*.log' -print | sort | tail -5
echo
latest_log="$(find results/runtime -maxdepth 1 -type f -name 'stage6_panel_*.log' -print | sort | tail -1 || true)"
if [[ -n "${latest_log}" ]]; then
  echo "[tail: ${latest_log}]"
  tail -60 "${latest_log}"
fi
