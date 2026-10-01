# Stage 6 LightGBM Panel Experiment Conclusion

Updated: 2026-08-04 KST

## Scope

This conclusion covers only the Stage 6 panel-based LightGBM experiment.

- Control: `STAGE6_PANEL_MODE=global`
- Experiment: `STAGE6_PANEL_MODE=cluster`
- Model: `LightGBM`
- Target: `level`
- Scoring during HPO: `r2`
- Cluster input: `KIER_4_Clustered/seasonal_naive/kmeans`

## Completed Inputs

- Global trials: `global/ALL 60/60`
- Cluster trials: `cluster/K0 60/60`, `cluster/K1 60/60`
- Global summary: `results/stage6_panel_global/ml_lightgbm_global_summary.csv`
- Cluster summary: `results/stage6_panel_cluster/ml_lightgbm_cluster_summary.csv`
- Comparison CSV: `results/stage6_panel_lightgbm_global_vs_cluster_comparison.csv`

## Overall Result

The cluster panel LightGBM result is broadly similar to the global panel baseline.

| Metric | Global | Cluster | Direction |
|---|---:|---:|---|
| MAE mean | 0.0204 | 0.0204 | Tie |
| MSE mean | 0.0075 | 0.0075 | Tie |
| RMSE mean | 0.0805 | 0.0804 | Cluster slightly better |
| SMAPE mean | 34.7053 | 34.7292 | Global slightly better |
| R2 mean | 0.1617 | 0.1635 | Cluster slightly better |
| MASE mean | 0.5396 | 0.5435 | Global slightly better |
| RMSE improvement vs best naive | 0.2580 | 0.2587 | Cluster slightly better |

## Interpretation

- Cluster training gives a very small improvement in RMSE, R2, and RMSE improvement against the best naive baseline.
- Global training is slightly better on SMAPE and MASE.
- The differences are small enough that the result should be interpreted as near parity rather than a strong ClusterBoost win.
- Because some individual CV trial scores were unstable, final interpretation should prioritize the held-out summary metrics over individual trial scores.

## Next Recommendation

- Treat the LightGBM-only Stage 6 panel experiment as complete.
- Use the result as evidence that clustering does not substantially harm LightGBM performance, but also does not produce a large improvement under the current 10MIN panel setup.
- If broader model comparison is still required, repeat the same checkpointed panel protocol for `xgboost` or a lower-cost resolution such as `1H` before attempting heavier models.
