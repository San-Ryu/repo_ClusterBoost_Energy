# Stage 6 LightGBM Panel Experiment Checklist

Updated: 2026-08-11 KST

This document is the operating checklist for the Stage 6 LightGBM-only panel experiment.
Each step includes the exact script or command to run, the status command to check progress,
and the file-based completion criterion.

Important 2026-08-11 update:

- This checklist remains valid for the completed LightGBM-only **panel** experiment.
- It is not the paper-style legacy aggregate comparison.
- Paper-style ML reference scores are archived under `results/reference_legacy_ml/`.
- Paper-style DL comparison should run through `src/legacy_aggregate_experiment.py`, not this LightGBM panel runner.
- The lower panel R2 is expected because this runner solves a stricter individual-household forecasting task, while the previous paper-style ML comparison used aggregate targets with many same-timestamp household usage columns as inputs.

Project boundary:

- This checklist is only for `/Users/labq_s.h.ryu/repo_ClusterBoost_Energy`.
- Do not paste or run Image Diffusion commands in this checklist.
- Image Diffusion work belongs to `/Users/labq_s.h.ryu/repo_Image Diffusion Ensemble Framework` and is unrelated to this Stage 6 LightGBM panel experiment.

## Current Position

- [x] All Stage 6 LightGBM panel work is complete.
- [x] Step 1. Environment and protocol are fixed.
- [x] Step 2. Resume/status tooling is ready.
- [x] Step 3. Global LightGBM trial search is complete.
- [x] Step 4. Global LightGBM finalize is complete.
- [x] Step 5. Cluster K0 LightGBM trial search is complete.
- [x] Step 6. Cluster K1 LightGBM trial search is complete.
- [x] Step 7. Cluster LightGBM finalize is complete.
- [x] Step 8. Global vs cluster comparison is complete.
- [x] Step 9. Final conclusion is written.

Current checkpoint snapshot:

| Scope | Progress | Status | Checkpoint |
|---|---:|---|---|
| global / ALL | 60 / 60 | Done | `results/stage6_panel_global/checkpoints/lightgbm/global/ALL_trials.csv` |
| cluster / K0 | 60 / 60 | Done | `results/stage6_panel_cluster/checkpoints/lightgbm/cluster/K0_trials.csv` |
| cluster / K1 | 60 / 60 | Done | `results/stage6_panel_cluster/checkpoints/lightgbm/cluster/K1_trials.csv` |

Current runtime state:

- [x] No active Stage 6 `screen` session is required.
- [x] Latest cluster finalize log succeeded: `results/runtime/stage6_panel_20260803_150535.log`.
- [x] Final comparison CSV exists: `results/stage6_panel_lightgbm_global_vs_cluster_comparison.csv`.
- [x] Final conclusion exists: `docs/stage6_lgbm_experiment_conclusion.md`.

Current decision:

- LightGBM-only Stage 6 panel experiment is complete.
- Cluster panel is near parity with global panel, with a very small RMSE/R2 advantage and a small SMAPE/MASE disadvantage.
- Do not use this result as the direct replacement for the previous paper ML table. Use `docs/legacy_ml_reference_performance_summary.md` and `results/reference_legacy_ml/legacy_ml_comparison_performance_wide.csv` for that comparison.

## Related Model Execution Scripts

Panel LightGBM scripts covered by this checklist:

```bash
src/stage6_lgbm_status.sh
src/run_stage6_lgbm_resumable.sh
src/run_stage6_panel.sh
```

All current per-model ML wrapper scripts:

```bash
python src/run_ml_catboost.py
python src/run_ml_decisiontree.py
python src/run_ml_lightgbm.py
python src/run_ml_randomforest.py
python src/run_ml_xgboost.py
```

All legacy aggregate ML model commands:

```bash
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=CatBoost SCOPE=both RESOLUTION=1H RESULTS_DIR=results/legacy_aggregate_ml_catboost_1h bash src/run_legacy_aggregate.sh
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=DecisionTree SCOPE=both RESOLUTION=1H RESULTS_DIR=results/legacy_aggregate_ml_decisiontree_1h bash src/run_legacy_aggregate.sh
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=LightGBM SCOPE=both RESOLUTION=1H RESULTS_DIR=results/legacy_aggregate_ml_lightgbm_1h bash src/run_legacy_aggregate.sh
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=RandomForest SCOPE=both RESOLUTION=1H RESULTS_DIR=results/legacy_aggregate_ml_randomforest_1h bash src/run_legacy_aggregate.sh
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=XGBoost SCOPE=both RESOLUTION=1H RESULTS_DIR=results/legacy_aggregate_ml_xgboost_1h bash src/run_legacy_aggregate.sh
```

All legacy aggregate DL model commands:

```bash
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_cnnlstm.py
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_seq2seq.py
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_cnnlstm.py
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_gru.py
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_transformer.py
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_tcn.py
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_retnet.py

# Batch
bash src/run_legacy_aggregate_dl_batch.sh --step plan
```

## Step 0. Always Check Status First

Purpose:

- Confirm the active screen session.
- Confirm exact checkpoint progress.
- Confirm latest runtime log.

Script:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
src/stage6_lgbm_status.sh
```

Expected current output:

```text
global/ALL          60/60
cluster/K0          60/60
cluster/K1          60/60
```

Completion criterion:

- Status command runs successfully.
- The `cluster/K1` count is used as the source of truth, not the plan CSV.
- If all three rows are `60/60`, do not relaunch training unless intentionally repeating the experiment.

## Step 1. Experiment Definition

Status:

- [x] Protocol: panel-based Stage 6.
- [x] Control: `STAGE6_PANEL_MODE=global`.
- [x] Experiment: `STAGE6_PANEL_MODE=cluster`.
- [x] Model scope: LightGBM only.
- [x] Target mode: `level`.
- [x] Scoring: `r2`.
- [x] Split: chronological / `TimeSeriesSplit`.
- [x] Cluster input: `KIER_4_Clustered/seasonal_naive/kmeans`.

No script is needed unless the protocol changes.

Completion criterion:

- This step remains complete unless the experiment design is changed.

## Step 2. Resume and Status Tooling

Status:

- [x] Memory-reduced panel runner: `src/model_ml_panel_checkpointed.py`
- [x] Panel wrapper with configurable PID file: `src/run_stage6_panel.sh`
- [x] Status script: `src/stage6_lgbm_status.sh`
- [x] Resumable LightGBM runner: `src/run_stage6_lgbm_resumable.sh`

Validation scripts:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
bash -n src/run_stage6_panel.sh
bash -n src/run_stage6_lgbm_resumable.sh
bash -n src/stage6_lgbm_status.sh
python -m py_compile src/model_ml_panel_checkpointed.py
```

Completion criterion:

- All validation commands exit successfully.

## Step 3. Global LightGBM Trial Search

Status:

- [x] Complete: `global/ALL 60 / 60`

Checkpoint:

```text
results/stage6_panel_global/checkpoints/lightgbm/global/ALL_trials.csv
```

Resume script if needed:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
screen -S stage6_lgbm_global_resumable -dm bash -lc 'cd "/Users/labq_s.h.ryu/repo_ClusterBoost_Energy"; STAGE6_N_ITER=60 STAGE6_CV_SPLITS=5 STAGE6_MAX_FAILURES=2 src/run_stage6_lgbm_resumable.sh global >> results/runtime/stage6_lgbm_global_resumable.log 2>&1'
```

Progress check:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
src/stage6_lgbm_status.sh
tail -f results/runtime/stage6_lgbm_global_resumable.log
```

Completion criterion:

```text
global/ALL 60/60
```

## Step 4. Global LightGBM Finalize

Status:

- [x] Complete.

Expected outputs:

```text
results/stage6_panel_global/ml_lightgbm_global_best_params.json
results/stage6_panel_global/ml_lightgbm_global_detail.csv
results/stage6_panel_global/ml_lightgbm_global_summary.csv
```

Manual finalize script if needed:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
USE_CAFFEINATE=0 \
STAGE6_PID_FILE="results/runtime/stage6_panel_lightgbm_global_manual_finalize.pid" \
STAGE6_PANEL_MODE=global \
STAGE6_MODELS=lightgbm \
STAGE6_STEP=finalize \
STAGE6_N_HOUSEHOLDS=348 \
STAGE6_N_ITER=60 \
STAGE6_CV_SPLITS=5 \
TARGET_MODE=level \
SCORING=r2 \
STAGE6_EXTRA_FLAGS="--results-dir results/stage6_panel_global --checkpoint-dir results/stage6_panel_global/checkpoints --predictions-dir results/stage6_panel_global/predictions" \
bash src/run_stage6_panel.sh
```

Completion criterion:

```bash
ls -lh results/stage6_panel_global/ml_lightgbm_global_summary.csv
```

## Step 5. Cluster K0 LightGBM Trial Search

Status:

- [x] Complete: `cluster/K0 60 / 60`

Checkpoint:

```text
results/stage6_panel_cluster/checkpoints/lightgbm/cluster/K0_trials.csv
```

Run/resume script:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
screen -S stage6_lgbm_cluster_resumable -dm bash -lc 'cd "/Users/labq_s.h.ryu/repo_ClusterBoost_Energy"; STAGE6_N_ITER=60 STAGE6_CV_SPLITS=5 STAGE6_MAX_FAILURES=2 src/run_stage6_lgbm_resumable.sh cluster >> results/runtime/stage6_lgbm_cluster_resumable.log 2>&1'
```

Progress check:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
src/stage6_lgbm_status.sh
tail -f results/runtime/stage6_lgbm_cluster_resumable.log
```

Completion criterion:

```text
cluster/K0 60/60
```

## Step 6. Cluster K1 LightGBM Trial Search

Status:

- [x] Complete: `cluster/K1 60 / 60`
- [x] No current action required.

Checkpoint:

```text
results/stage6_panel_cluster/checkpoints/lightgbm/cluster/K1_trials.csv
```

Run/resume script:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
screen -S stage6_lgbm_cluster_resumable -dm bash -lc 'cd "/Users/labq_s.h.ryu/repo_ClusterBoost_Energy"; STAGE6_N_ITER=60 STAGE6_CV_SPLITS=5 STAGE6_MAX_FAILURES=2 src/run_stage6_lgbm_resumable.sh cluster >> results/runtime/stage6_lgbm_cluster_resumable.log 2>&1'
```

Progress check:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
src/stage6_lgbm_status.sh
tail -f results/runtime/stage6_lgbm_cluster_resumable.log
```

Completion criterion:

```text
cluster/K1 60/60
```

Notes:

- The runner always checks existing checkpoint files first.
- If the machine sleeps or the process stops, rerun the same `screen -S stage6_lgbm_cluster_resumable ...` command.
- Only the in-progress trial is lost; completed trials remain in `K1_trials.csv`.

## Step 7. Cluster LightGBM Finalize

Status:

- [x] Complete.
- [x] Completed after both `cluster/K0 60/60` and `cluster/K1 60/60`.

Expected outputs:

```text
results/stage6_panel_cluster/ml_lightgbm_cluster_best_params.json
results/stage6_panel_cluster/ml_lightgbm_cluster_detail.csv
results/stage6_panel_cluster/ml_lightgbm_cluster_summary.csv
```

Finalize script:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
USE_CAFFEINATE=0 \
STAGE6_PID_FILE="results/runtime/stage6_panel_lightgbm_cluster_manual_finalize.pid" \
STAGE6_PANEL_MODE=cluster \
STAGE6_MODELS=lightgbm \
STAGE6_STEP=finalize \
STAGE6_N_HOUSEHOLDS=348 \
STAGE6_N_ITER=60 \
STAGE6_CV_SPLITS=5 \
TARGET_MODE=level \
SCORING=r2 \
STAGE6_EXTRA_FLAGS="--results-dir results/stage6_panel_cluster --checkpoint-dir results/stage6_panel_cluster/checkpoints --predictions-dir results/stage6_panel_cluster/predictions" \
bash src/run_stage6_panel.sh
```

Completion criterion:

```bash
ls -lh results/stage6_panel_cluster/ml_lightgbm_cluster_summary.csv
```

Observed completion log:

```text
results/runtime/stage6_panel_20260803_150535.log
[DONE] model=LightGBM panel_mode=cluster
summary: results/stage6_panel_cluster/ml_lightgbm_cluster_summary.csv
```

## Step 8. Global vs Cluster Comparison

Status:

- [x] Complete.

Inputs:

```text
results/stage6_panel_global/ml_lightgbm_global_summary.csv
results/stage6_panel_cluster/ml_lightgbm_cluster_summary.csv
```

Quick comparison script:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
python - <<'PY'
import pandas as pd
from pathlib import Path

global_path = Path("results/stage6_panel_global/ml_lightgbm_global_summary.csv")
cluster_path = Path("results/stage6_panel_cluster/ml_lightgbm_cluster_summary.csv")

g = pd.read_csv(global_path)
c = pd.read_csv(cluster_path)

g_all = g[g["group_id"].astype(str).eq("ALL_GROUPS")].tail(1)
if g_all.empty:
    g_all = g.tail(1)
c_all = c[c["group_id"].astype(str).eq("ALL_GROUPS")].tail(1)
if c_all.empty:
    c_all = c.tail(1)

metrics = [
    "MAE_mean",
    "MSE_mean",
    "RMSE_mean",
    "SMAPE_mean",
    "R2_mean",
    "MASE_mean",
    "rmse_improvement_vs_best_naive_mean",
]

rows = []
for metric in metrics:
    if metric in g_all.columns and metric in c_all.columns:
        gv = float(g_all.iloc[0][metric])
        cv = float(c_all.iloc[0][metric])
        rows.append({
            "metric": metric,
            "global": gv,
            "cluster": cv,
            "cluster_minus_global": cv - gv,
        })

out = pd.DataFrame(rows)
out_path = Path("results/stage6_panel_lightgbm_global_vs_cluster_comparison.csv")
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print(f"saved: {out_path}")
PY
```

Completion criterion:

```text
results/stage6_panel_lightgbm_global_vs_cluster_comparison.csv
```

Observed result:

```text
MAE_mean                             global=0.0204  cluster=0.0204  winner=tie
MSE_mean                             global=0.0075  cluster=0.0075  winner=tie
RMSE_mean                            global=0.0805  cluster=0.0804  winner=cluster
SMAPE_mean                           global=34.7053 cluster=34.7292 winner=global
R2_mean                              global=0.1617  cluster=0.1635  winner=cluster
MASE_mean                            global=0.5396  cluster=0.5435  winner=global
rmse_improvement_vs_best_naive_mean  global=0.2580  cluster=0.2587  winner=cluster
```

## Step 9. Final Conclusion

Status:

- [x] Complete.

Required inputs:

- [x] Global summary exists.
- [x] Cluster summary exists.
- [x] Global vs cluster comparison CSV exists.

Suggested output:

```text
docs/stage6_lgbm_experiment_conclusion.md
```

Conclusion writing checklist:

- [x] State whether cluster panel improved RMSE over global panel.
- [x] State whether cluster panel improved MASE over global panel.
- [x] State R2 interpretation carefully, because some CV trial scores were unstable.
- [x] Include exact result file paths.
- [x] Mention that the current conclusion is LightGBM-only.

## One-Line Operating Rule

Always run this first:

```bash
cd /Users/labq_s.h.ryu/repo_ClusterBoost_Energy
src/stage6_lgbm_status.sh
```

Then act based on the current line:

- If `cluster/K1 < 60/60`, run Step 6.
- If `cluster/K1 = 60/60` and cluster summary is missing, run Step 7.
- If both summaries exist and comparison CSV is missing, run Step 8.
- If comparison CSV exists and conclusion is missing, run Step 9.
- Current state on 2026-08-04: all of the above are complete.
