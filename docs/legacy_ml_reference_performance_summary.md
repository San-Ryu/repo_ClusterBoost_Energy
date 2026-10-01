# Legacy ML Reference Performance Summary

Date: 2026-08-10

## Source

The paper-style ML comparison performance was copied from:

`/Users/labq_s.h.ryu/repo_5Energy_Clustering/Src_Energy_M02`

The duplicate files under `Src_Energy_M02_KIRBS Ver` were checked with SHA-256 and matched exactly for all copied ML comparison txt files.

Current-project artifacts:

- `results/reference_legacy_ml/comparison_raw/`
- `results/reference_legacy_ml/legacy_ml_comparison_performance_long.csv`
- `results/reference_legacy_ml/legacy_ml_comparison_performance_wide.csv`
- `results/reference_legacy_ml/legacy_ml_comparison_provenance.json`

Completeness check:

- Models: `CB`, `DT`, `LGBM`, `RF`, `XGB`
- Conditions per model: `K2M`, `K2W`, `K3M`, `K3W`
- Long table rows: `90`
- Wide table rows: `20`
- Raw copied comparison files: `20`
- Duplicate `Src_Energy_M02_KIRBS Ver` files were SHA-256 identical for all copied ML comparison files.

## Experiment Group Mapping

- `대조군 : 군집화 X` -> `control`
- `실험군 01-01/01-02/01-03 : C* 군집의 합` -> `experimental_1`, direct cluster prediction
- `실험군 02 : 모든 군집의 합` -> `experimental_2`, summed cluster predictions

Metric order copied from the raw txt files:

`MAE`, `MAPE`, `MSE`, `RMSE`, `MSLE`, `MBE`, `R2`

## R2/RMSE Summary

| Model | Condition | Cluster Sizes | Control R2 | Exp. 2 R2 | Delta R2 | Control RMSE | Exp. 2 RMSE | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CB | K2M | 206/142 | 0.9171 | 0.9258 | 0.0087 | 9.9126 | 9.3817 | -0.5309 |
| CB | K2W | 210/138 | 0.9171 | 0.9255 | 0.0084 | 9.9126 | 9.3982 | -0.5144 |
| CB | K3M | 173/80/95 | 0.9171 | 0.9281 | 0.0110 | 9.9126 | 9.2315 | -0.6811 |
| CB | K3W | 93/83/172 | 0.9171 | 0.9280 | 0.0109 | 9.9126 | 9.2373 | -0.6753 |
| DT | K2M | 209/139 | 0.7412 | 0.8245 | 0.0833 | 17.5149 | 14.4239 | -3.0910 |
| DT | K2W | 210/138 | 0.7396 | 0.8172 | 0.0776 | 17.5706 | 14.7188 | -2.8518 |
| DT | K3M | 173/80/95 | 0.7421 | 0.8583 | 0.1162 | 17.4858 | 12.9592 | -4.5266 |
| DT | K3W | 49/136/163 | 0.7399 | 0.8636 | 0.1237 | 17.5579 | 12.7169 | -4.8410 |
| LGBM | K2M | 209/139 | 0.9254 | 0.9338 | 0.0084 | 9.4028 | 8.8554 | -0.5474 |
| LGBM | K2W | 138/210 | 0.9254 | 0.9322 | 0.0068 | 9.4028 | 8.9669 | -0.4359 |
| LGBM | K3M | 171/80/97 | 0.9254 | 0.9363 | 0.0109 | 9.4028 | 8.6923 | -0.7105 |
| LGBM | K3W | 103/78/167 | 0.9254 | 0.9363 | 0.0109 | 9.4028 | 8.6865 | -0.7163 |
| RF | K2M | 206/142 | 0.8526 | 0.8837 | 0.0311 | 13.2191 | 11.7393 | -1.4798 |
| RF | K2W | 206/142 | 0.8495 | 0.8832 | 0.0337 | 13.3587 | 11.7673 | -1.5914 |
| RF | K3M | 121/167/60 | 0.8533 | 0.8927 | 0.0394 | 13.1877 | 11.2761 | -1.9116 |
| RF | K3W | 163/136/49 | 0.8531 | 0.8947 | 0.0416 | 13.1947 | 11.1741 | -2.0206 |
| XGB | K2M | 142/206 | 0.8954 | 0.9119 | 0.0165 | 11.1367 | 10.2208 | -0.9159 |
| XGB | K2W | 138/210 | 0.8954 | 0.9152 | 0.0198 | 11.1367 | 10.0270 | -1.1097 |
| XGB | K3M | 120/165/63 | 0.8954 | 0.9238 | 0.0284 | 11.1367 | 9.5063 | -1.6304 |
| XGB | K3W | 103/78/167 | 0.8954 | 0.9230 | 0.0276 | 11.1367 | 9.5519 | -1.5848 |

## DL Alignment

DL should use the same interpretation:

- `dataset=ALL` -> control
- `dataset=K*` -> experimental 1
- `dataset=cluster_ensemble` -> experimental 2

Use `src/legacy_aggregate_experiment.py` for DL, because it already writes per-dataset predictions and computes `cluster_ensemble` by summing the cluster predictions on matched timestamps.

## Why This Differs From Current Stage 6 Scores

The reference ML comparison is not the same task as the current Stage 6 panel score.

- Previous paper-style ML: aggregate target with wide household inputs. In practice, the model sees weather/time features plus the other household usage columns on the same resampled time grid.
- Current Stage 6 panel: stricter per-household forecast evaluation, chronological split, lag/rolling features, and naive-baseline diagnostics.
- Current DL single-household path: per-household sequence forecasting with only that household's series plus time features.

Therefore a low current Stage 6 or DL-single R2 does not directly refute the paper ML result. It usually means the current experiment is solving a harder and less information-rich forecasting problem. For paper-equivalent comparison, use the copied ML tables here and run DL through `src/legacy_aggregate_experiment.py`.
