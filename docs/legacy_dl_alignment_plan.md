# Legacy DL Alignment Plan

Date: 2026-08-10

## Decision

The DL comparison must move onto the same aggregate/global/cluster experiment surface used by the prior ML paper experiment. The current single-household 10-minute DL path is useful, but it is not the same benchmark as the previous repository's aggregate KIER experiment and should not be used as the direct counterpart for the paper baseline.

## ML Reference Handling

The previous repository is read-only. Instead of running ML smoke tests as proof, the extractable notebook outputs from the previous experiment have been copied into the current project:

`results/reference_legacy_ml/`

Key copied reference artifacts:

- `ml02_xgb_1h_kfold10_summary.csv`
- `ml02_xgb_1h_kfold10_fold_metrics.csv`
- `ml02_xgb_1h_kfold10_metric_history_long.csv`
- `ml04_xgb_k2_1w_cluster_sum_predictions.csv`
- `ml04_xgb_k2_1w_cluster_sum_predictions_raw_output.txt`
- `provenance.json`

The ML-02 reference summary reports XGBoost 10-fold aggregate performance with R2 `0.8386`.

The ML-04 comparison notebook retained cluster-sum predictions, but did not print actual values or individual cluster prediction arrays. Those commented-out arrays cannot be copied without rerunning or changing the source repository, so only the extractable cluster-sum prediction output is archived.

## Required DL Modification

DL must be run through `src/legacy_aggregate_experiment.py`, not only through the individual-household DL runner. The comparison unit must mirror the ML paper comparison:

- control: non-clustered aggregate prediction
- experimental 1: direct prediction for each cluster aggregate target
- experimental 2: summed prediction from all cluster models, evaluated against the summed actual aggregate

Required alignment points:

- Use the same current preprocessed and clustered data surface already available in this project.
- Build aggregate targets matching the previous experiment shape:
  - global: `ELEC_INST_SUM_ALL`
  - cluster: `ELEC_INST_SUM_K*`
- Keep global and cluster scopes selectable through `SCOPE=global`, `SCOPE=cluster`, or `SCOPE=both`.
- Use the shared model registry in `src/legacy_model_registry.py` so legacy DL and current designed DL models can be swapped by name.
- Keep DL sequence/window construction inside the aggregate runner so the target, resolution, split, and provenance are controlled in one place.
- Keep the `cluster_ensemble` output from `src/legacy_aggregate_experiment.py`; it is the DL counterpart of `실험군 02 : 모든 군집의 합`.
- Treat the Stage 6 panel outputs as a separate stricter experiment family, not as the baseline-equivalent paper comparison.

## Why Current DL/Panel Scores Can Look Too Low

The previous ML experiment used a much stronger input surface than the current strict forecast runners. It modeled aggregate targets while retaining many same-timestamp household usage columns as predictors. With 348 household columns, aggregate load is naturally easier to predict because most of the contemporaneous signal is present in the feature matrix.

The current Stage 6 panel and single-household DL paths intentionally avoid that exact setup:

- Stage 6 panel predicts individual household rows from lag/rolling features under chronological validation.
- Single-household DL uses one household sequence plus time features, not the remaining 348 households as contemporaneous predictors.
- `target_mode=delta` makes the model learn changes over `lag_1`, then reconstructs level predictions; this is harder and can depress R2 even when RMSE/MASE improve.
- Naive baselines such as `lag_144` and `lag_1008` are strong for electricity data, so R2 alone can look disappointing.

So the expected fix is not simply "train longer." The fair DL comparison should restore the paper-equivalent aggregate input/target shape, then compare `ALL`, each `K*`, and `cluster_ensemble`.

## Implemented Current-Project Support

- `src/legacy_model_registry.py` defines legacy ML, legacy DL, and current designed DL builders.
- `src/legacy_aggregate_experiment.py` supports aggregate ML/DL experiments over global and clustered targets.
- `src/run_legacy_aggregate.sh` provides a reproducible entry point.
- `src/run_legacy_aggregate_dl_*.py` provides model-specific aggregate DL entry points.
- `src/run_legacy_aggregate_dl_batch.sh` runs selected aggregate DL models in sequence.
- `src/model_dl_single.py` now imports DL model definitions from the registry so model definitions are no longer duplicated.
- `src/run_dl_common.py` now falls back from a broken `.venv-dl/bin/python` to `.venv/bin/python` when selecting a runtime.

## Recommended DL Runs

Legacy DL baseline-equivalent run:

```bash
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_cnnlstm.py
```

Legacy Seq2Seq variant:

```bash
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_seq2seq.py
```

Current designed DL model under the same aggregate condition:

```bash
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_tcn.py
```

All aggregate DL model scripts:

```bash
# Legacy CNN-LSTM
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_cnnlstm.py

# Legacy Seq2Seq
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_legacy_seq2seq.py

# Current CNN-LSTM
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_cnnlstm.py

# GRU
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_gru.py

# Transformer
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_transformer.py

# TCN
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_tcn.py

# RetNet
EPOCHS=200 SEQ_LEN=3 python src/run_legacy_aggregate_dl_retnet.py
```

Batch examples:

```bash
# Plan all aggregate DL models.
bash src/run_legacy_aggregate_dl_batch.sh --step plan

# Run selected aggregate DL models.
DL_AGG_MODELS=legacy_cnnlstm,tcn EPOCHS=200 SEQ_LEN=3 \
  bash src/run_legacy_aggregate_dl_batch.sh
```

## Verification Before Full Run

Use a plan step first to confirm datasets, rows, feature counts, targets, and output paths:

```bash
python src/run_legacy_aggregate_dl_legacy_cnnlstm.py --step plan
```

This should report aggregate datasets for `ALL`, `K0`, and `K1` before any long training starts.

## Expected DL Output Interpretation

The generated summary should be read as:

- `dataset=ALL`: 대조군, no clustering
- `dataset=K*`: 실험군 01, each cluster aggregate model
- `dataset=cluster_ensemble`: 실험군 02, summed cluster predictions

The headline DL comparison against the ML reference should use `ALL` versus `cluster_ensemble` under the same resolution, test ratio, target shift, and clustering condition.

## Remaining DL Risk

Exact legacy DL predictions were not available as clean saved artifacts in the previous repository outputs. Therefore DL equivalence should be established by rerunning current-project aggregate DL under the legacy-compatible settings above, then comparing the generated summaries and predictions against the paper-level expectation rather than relying on unavailable copied DL prediction arrays.
