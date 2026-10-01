# Stage 6 Checkpoint Status

Updated: 2026-07-16 KST

## Decision

The previous `all_households` / `cluster_households` runner was retired for the main comparison because it trained one local model per household with peer lag features. That does not match the intended control/experiment definition.

The active Stage 6 protocol is now panel-based:

- Control: `STAGE6_PANEL_MODE=global`
- Experiment: `STAGE6_PANEL_MODE=cluster`
- Split: chronological, no shuffle
- Target default: `level`
- Metrics: `MAE`, `MAPE`, `MSE`, `RMSE`, `SMAPE`, `R2`, `MASE`, and RMSE improvement vs best naive

## Current State

- Old `stage6_all_households` screen: stopped
- Old single-household Stage 6 result/log files: deleted
- New ML runner: `src/model_ml_panel_checkpointed.py`
- New wrapper: `src/run_stage6_panel.sh`
- Smoke global LightGBM run: passed
- Smoke cluster LightGBM run: passed

## Recommended Sequence

1. Run quick ML validation with `catboost,lightgbm,xgboost` for `global`.
2. Run quick ML validation with `catboost,lightgbm,xgboost` for `cluster`.
3. Compare against old-study-level behavior.
4. Expand final ML run to include `decisiontree,randomforest`.
5. Add panel-equivalent DL runner before DL comparison.
