# Legacy ML/DL Extension Commit Log

Date: 2026-08-05

## Summary

This change ports the previous KIER M02 machine-learning and clustering experiment shape into the current project without modifying the previous repository:

`/Users/labq_s.h.ryu/repo_A_machine_learning_ensemble_framework_based_on_a_clustering_algorithm_for_improving_electric_power_consumption_performance`

The current project now has a legacy-compatible aggregate experiment path that can reuse the existing preprocessed and clustered data while swapping ML and DL model definitions through a shared registry.

## Rationale

The previous Stage 6 panel LightGBM comparison was not equivalent to the prior repository's reported baseline experiment. The prior experiment modeled aggregate targets such as `ELEC_INST_SUM_ALL` and `ELEC_INST_SUM_K*` on resampled wide household data, while the current Stage 6 panel output evaluated 10-minute individual household forecasts. This commit therefore restores the legacy aggregate/cluster protocol as a first-class experiment path instead of trying to interpret the panel forecast CSV as the same benchmark.

## Source Policy

- The previous repository is treated as read-only reference material.
- No file under the previous repository is edited, generated, or overwritten.
- The current project owns all copied, refactored, and extended code.

## Ported / Refactored Areas

### Reference ML Results

- Archived extractable previous-repository ML notebook outputs under `results/reference_legacy_ml/`.
- Preserved provenance for every copied file in `results/reference_legacy_ml/provenance.json`.
- Captured the prior XGBoost 10-fold aggregate summary with R2 `0.8386`.
- Captured the ML-04 `K=2`, `1W` cluster-sum prediction output where available.
- Documented the extraction limitation that ML-04 actual values and individual cluster predictions were not printed by the notebook.
- Archived all saved paper-style ML comparison txt files from `/Users/labq_s.h.ryu/repo_5Energy_Clustering/Src_Energy_M02`.
- Added `legacy_ml_comparison_performance_long.csv` with all control, cluster-direct, and summed-cluster metrics for `CB`, `DT`, `LGBM`, `RF`, and `XGB`.
- Added `legacy_ml_comparison_performance_wide.csv` for one-row-per-model-condition comparison of the control and experimental groups.

### Clustering

- Added legacy-compatible KMeans helpers in `src/legacy_clustering.py`.
- Recreated the previous project metrics: inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn index, and optional supervised cluster scores.
- Added utilities to load current clustered CSV groups directly from `KIER_4_Clustered`.

### ML Models

- Added the previous project's ML model family to `src/legacy_model_registry.py`:
  - CatBoost
  - DecisionTree
  - LightGBM
  - RandomForest
  - XGBoost
- Preserved the prior model-level defaults where they are part of the old benchmark shape.
- Kept ML model construction separate from the experiment runner so aggregate/global/cluster experiments can swap models by name.

### DL Models

- Added the previous project's legacy DL model families:
  - `Legacy1DCNNLSTM`
  - `Legacy1DCNNSeq2Seq`
- Added the current project's designed DL families to the same registry:
  - `1D_CNN_LSTM`
  - `GRU`
  - `Transformer`
  - `TCN`
  - `RetNet`
- Moved the custom RetNet-style retention implementation into the registry so aggregate experiments and single-household DL runs share the same model definition source.
- Added direct wrappers:
  - `src/run_dl_legacy_cnnlstm.py`
  - `src/run_dl_legacy_seq2seq.py`
- Added aggregate DL wrappers:
  - `src/run_legacy_aggregate_dl_legacy_cnnlstm.py`
  - `src/run_legacy_aggregate_dl_legacy_seq2seq.py`
  - `src/run_legacy_aggregate_dl_cnnlstm.py`
  - `src/run_legacy_aggregate_dl_gru.py`
  - `src/run_legacy_aggregate_dl_transformer.py`
  - `src/run_legacy_aggregate_dl_tcn.py`
  - `src/run_legacy_aggregate_dl_retnet.py`
- Added `src/run_legacy_aggregate_dl_batch.sh` to run all or selected aggregate DL models.

### Legacy-Compatible Aggregate Experiment

- Added `src/legacy_aggregate_experiment.py` to reproduce the prior benchmark shape on current data:
  - global aggregate target: `ELEC_INST_SUM_ALL`
  - cluster aggregate targets: `ELEC_INST_SUM_K*`
  - ML or DL model family selected by CLI
  - global, cluster, or both scopes
  - detail, summary, prediction, fold, and config outputs
- Added `src/run_legacy_aggregate.sh` as a reproducible shell entry point.

## Recommended Runs

Plan the legacy-compatible ML benchmark:

```bash
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=XGBoost SCOPE=both RESOLUTION=1H \
  bash src/run_legacy_aggregate.sh --step plan
```

Run the legacy-compatible ML benchmark:

```bash
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=XGBoost SCOPE=both RESOLUTION=1H \
  RESULTS_DIR=results/legacy_aggregate_xgboost_1h bash src/run_legacy_aggregate.sh
```

Run the same aggregate experiment with a DL model:

```bash
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_cnnlstm.py
```

## Expected Interpretation

This legacy-compatible path should be compared against the prior repository's aggregate ML/DL results. The Stage 6 panel forecasting outputs should remain a separate, stricter forecast-valid experiment family because they evaluate individual household forecasts at finer temporal granularity.

## Commit Message Draft

```text
Port legacy aggregate ML/DL and clustering experiment

- add legacy-compatible KMeans clustering utilities using current clustered data
- add shared ML/DL model registry with prior CatBoost/DT/LGBM/RF/XGB defaults
- add legacy 1D-CNN LSTM and 1D-CNN Seq2Seq builders alongside current DL models
- add aggregate global/cluster experiment runner with swappable ML/DL models
- add shell and DL wrappers for reproducible legacy-compatible runs
- document why the source changed and how it differs from Stage 6 panel forecasts
```
