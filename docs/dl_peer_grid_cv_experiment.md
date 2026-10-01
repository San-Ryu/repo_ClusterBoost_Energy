# DL Peer-Household Grid CV Experiment

Date: 2026-08-19

## Purpose

This experiment modularizes DL models under the input condition discussed for the paper-style comparison:

`weather variables + the 347 non-target household electricity columns -> target household electricity`

It is different from `src/model_dl_single.py`, which only uses the target household sequence plus time features.

## Implemented Runner

- Main runner: `src/model_dl_peer_grid_cv.py`
- Shared wrapper: `src/run_dl_peer_common.py`
- Batch runner: `src/run_dl_peer_grid_batch.sh`

Model wrappers:

- `src/run_dl_peer_legacy_cnnlstm.py`
- `src/run_dl_peer_legacy_seq2seq.py`
- `src/run_dl_peer_cnnlstm.py`
- `src/run_dl_peer_gru.py`
- `src/run_dl_peer_transformer.py`
- `src/run_dl_peer_tcn.py`
- `src/run_dl_peer_retnet.py`

## Contract Checks

The runner fails fast unless all of these are true:

- source power table has 348 household columns
- feature table contains 347 peer household columns
- target household column is not present in the feature matrix
- weather feature count is greater than zero
- total input feature count equals `347 + n_weather_features`

Default files:

- power: `~/data/KIER_3_Temporal_Resolution/seasonal_naive/KIER_USAGE_ELEC_INST_INTERP_1H.csv`
- weather: `~/data/KMA_ASOS/119_SUWON/ASOS_119_2010-2024_HR.csv`

The default ASOS file currently contributes 17 weather variables, so the default input feature count is `364`.

## Smoke Test

Use this to validate every model with the real feature contract, one grid combination, one target household, and 5-fold CV:

```bash
DL_PEER_MODELS=all bash src/run_dl_peer_grid_batch.sh \
  --grid-preset smoke \
  --max-rows 120 \
  --n-target-households 1 \
  --cv-splits 5 \
  --cv-type kfold \
  --fit-verbose 0
```

Observed smoke run on 2026-08-19:

```bash
PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache DL_PEER_MODELS=all PYTHON_BIN=.venv/bin/python \
  bash src/run_dl_peer_grid_batch.sh \
  --grid-preset smoke \
  --max-rows 120 \
  --n-target-households 1 \
  --cv-splits 5 \
  --cv-type kfold \
  --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_smoke_all \
  --no-resume
```

Status: success.

Smoke output summary:

`results/dl_peer_grid_cv_smoke_all/dl_peer_smoke_summary.csv`

All seven DL models produced 5 fold metric rows each. Every feature contract confirmed:

- total household columns: `348`
- peer household features: `347`
- weather features: `17`
- total input features: `364`
- target household in features: `false`

The smoke run uses only 120 rows and 1 epoch, so the R2 values are execution checks, not final performance estimates.

Expected elapsed time from the observed 2026-08-19 run:

- all 7 models: about 1 min 50 sec
- per model wall time: about 9-26 sec
- per fold fit time: about 0.8-2.1 sec

This estimate is only for `--max-rows 120`, `1` target household, `1` grid combo, `1` epoch, and `5` CV folds.

Single-model smoke:

```bash
python src/run_dl_peer_gru.py \
  --grid-preset smoke \
  --max-rows 120 \
  --n-target-households 1 \
  --cv-splits 5 \
  --cv-type kfold \
  --fit-verbose 0
```

## Small Grid Run

```bash
DL_PEER_MODELS=gru,tcn,transformer bash src/run_dl_peer_grid_batch.sh \
  --grid-preset small \
  --n-target-households 3 \
  --cv-splits 5 \
  --cv-type kfold
```

## Model-by-Model Search Scripts

All commands below use the peer-household input contract:

`weather variables + 347 non-target household columns -> target household`

Each command performs GridSearch over the selected preset and measures 5CV performance.

### Smoke, One Model

```bash
PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_legacy_cnnlstm.py \
  --grid-preset smoke --max-rows 120 --n-target-households 1 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_smoke_all --no-resume
```

Replace the script path with one of:

```bash
src/run_dl_peer_legacy_cnnlstm.py
src/run_dl_peer_legacy_seq2seq.py
src/run_dl_peer_cnnlstm.py
src/run_dl_peer_gru.py
src/run_dl_peer_transformer.py
src/run_dl_peer_tcn.py
src/run_dl_peer_retnet.py
```

### Small Grid, Model by Model

```bash
PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_legacy_cnnlstm.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_legacy_seq2seq.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_cnnlstm.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_gru.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_transformer.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_tcn.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_retnet.py \
  --grid-preset small --n-target-households 3 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv
```

### Paper Grid, Model by Model

Use this only after small-grid timing is acceptable.

```bash
PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_legacy_cnnlstm.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_legacy_seq2seq.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_cnnlstm.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_gru.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_transformer.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_tcn.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper

PYTHONPYCACHEPREFIX=/tmp/clusterboost_pycache PYTHON_BIN=.venv/bin/python \
  .venv/bin/python src/run_dl_peer_retnet.py \
  --grid-preset paper --n-target-households 10 \
  --cv-splits 5 --cv-type kfold --fit-verbose 0 \
  --results-dir results/dl_peer_grid_cv_paper
```

## Runtime Estimate

Observed current small-grid attempt:

- command: `DL_PEER_MODELS=all bash src/run_dl_peer_grid_batch.sh --grid-preset small --n-target-households 3 --cv-splits 5 --cv-type kfold`
- started: 2026-08-19 15:26:51 KST
- last log update: 2026-08-19 18:26:19 KST
- current status checked at: 2026-08-19 18:28:24 KST
- pid file: `results/runtime/dl_peer_grid_batch.pid`, pid `78027`
- process state: not alive
- completed result files: no `*_fold_metrics.csv` under `results/dl_peer_grid_cv`
- partial files: `dl_peer_legacy1dcnnlstm_plan.csv`, `dl_peer_legacy1dcnnlstm_config.json`, `dl_peer_legacy1dcnnlstm_feature_contract.json`

The small-grid attempt therefore ran for about 2 h 59 m and stopped before model-level completion.

Estimated work units:

- smoke all models: `7 models * 1 target * 1 combo * 5 folds = 35 fits`
- small grid with 3 targets: `7 models * 3 targets * 8 combos * 5 folds = 840 fits`
- paper grid with 10 targets: `7 models * 10 targets * 48 combos * 5 folds = 16,800 fits`
- paper grid with all 348 targets: `7 models * 348 targets * 48 combos * 5 folds = 584,640 fits`

Estimated elapsed time on the current machine:

- smoke all models: about 2-3 min
- one-model smoke: about 10-30 sec
- small grid, 3 targets, all models: at least 24 h; more realistically 1-3 days with the current TensorFlow/Metal behavior
- paper grid, 10 targets, all models: not practical as a single local batch; lower-bound estimate is measured in weeks
- paper grid, all 348 targets: not practical on this local setup without reducing the grid, adding fold-level checkpointing, and using a faster DL runtime

The main reason the small-grid estimate is much larger than the smoke estimate is that smoke used `120` rows and `1` epoch. The observed small-grid plan uses `16,261` rows, `20` epochs, `8` parameter combinations, `5` folds, and `3` targets per model.

## VDB Rollback Audit

As of 2026-08-19, source and docs search found no explicit VDB/VectorDB/RAG/embedding retrieval implementation in the project files. No DL peer-grid files were reverted because they implement the requested paper-style input contract and are not VDB-related.

## Paper-Scale Grid Run

```bash
DL_PEER_MODELS=all bash src/run_dl_peer_grid_batch.sh \
  --grid-preset paper \
  --n-target-households 10 \
  --cv-splits 5 \
  --cv-type kfold
```

## Outputs

Default output directory:

`results/dl_peer_grid_cv/`

Per model:

- `dl_peer_<model>_plan.csv`
- `dl_peer_<model>_feature_contract.json`
- `dl_peer_<model>_config.json`
- `dl_peer_<model>_fold_metrics.csv`
- `dl_peer_<model>_grid_summary.csv`
- `dl_peer_<model>_best_summary.csv`

## Target Shift

Default `--target-shift 0` reproduces the contemporaneous peer-household condition:

`features(t) -> target(t)`

For stricter future forecasting, set a positive shift:

```bash
python src/run_dl_peer_gru.py --target-shift 1 --grid-preset small
```
