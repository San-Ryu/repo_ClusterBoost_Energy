# Experiment Stage Checklist (Updated)

Updated: 2026-08-11 KST

KIER 에너지 예측 실험의 최신 실행 체크리스트다.  
본 버전은 **Stage 6 모델 실행을 모델별 스크립트로 분리**한 흐름과 논문 기준 재현용 **legacy aggregate ML/DL 비교 경로**를 함께 반영한다.

---

## 0) 실행 전 준비

- [ ] 가상환경 활성화

```bash
source .venv/bin/activate
```

- [ ] (권장) DL 전용 가상환경 생성/활성화  
TensorFlow 충돌이 있으면 `.venv`와 분리해 `.venv-dl`에서 DL만 실행한다.

```bash
python3.10 -m venv .venv-dl
source .venv-dl/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2.0" pandas scikit-learn matplotlib seaborn tqdm
python -m pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0 keras==2.15.0
python -c "import tensorflow as tf; print(tf.__version__)"
```

- [ ] 의존성 설치/보정

```bash
python -m pip install jupyter nbconvert
python -m pip install numpy pandas scipy matplotlib seaborn statsmodels scikit-learn \
  catboost lightgbm xgboost tensorflow tensorflow-addons keras-flops \
  torch pytorch-lightning pyarrow fastparquet requests beautifulsoup4 \
  selenium webdriver-manager openpyxl xlrd tqdm ipykernel
```

- [ ] macOS LightGBM 의존성(libomp) 설치

```bash
brew install libomp
```

- [ ] Matplotlib 캐시 경고 방지(선택)

```bash
export MPLCONFIGDIR="$PWD/.mplconfig"
```

- [ ] 보간 입력 경로 정합 (`KIER_S2_interp` 링크)

```bash
ln -sfn "KIER_2_interpolation" ~/data/KIER_S2_interp"
```

---

## 1) Stage 1 — Raw -> House Dataset 생성

### 실행

```bash
source .venv/bin/activate
python core/data_02_build_dataset.py
```

### 예상 소요
- 약 25~60분

### 완료 확인
- [ ] ~/data/KIER_1_USAGE_House/KIER_HINDEX.csv`
- [ ] ~/data/KIER_1_USAGE_House/KIER_USAGE_*_{ACCU,INST}.csv`
- [ ] ~/data/KIER_1_USAGE_House/KIER_1_DIVIDED_BY_DOMAIN_HOUSE/`

---

## 2) Stage 2 — 결측 보간/평가

### 2-1) Phase 1: 보간 5안 생성

```bash
source .venv/bin/activate
python core/data_04_interp.py --phase 1
```

실측 참고:
- baseline 0.5s, pchip 0.9s, seasonal_naive 0.8s
- stl 7,293s, mstl 17,046s

완료 확인:
- [ ] ~/data/KIER_2_interpolation/phase1_log.csv`
- [ ] ~/data/KIER_2_interpolation/{method}/KIER_USAGE_ELEC_INST_INTERP.parquet`

### 2-2) Phase 2: 인공 마스킹 정량 평가

빠른 검증:

```bash
source .venv/bin/activate
python core/data_04_interp.py --phase 2 \
  --methods baseline pchip seasonal_naive \
  --n-trials 10 --sample-n 20 --workers 4
```

전체 정밀:

```bash
source .venv/bin/activate
python core/data_04_interp.py --phase 2 \
  --n-trials 10 --sample-n 30 --workers 4
```

완료 확인:
- [ ] `results/preprocessing/ELEC_interp_metrics.csv`
- [ ] `results/preprocessing/ELEC_interp_summary.csv`
- [ ] `results/preprocessing/ELEC_interp_rank.csv`

---

## 3) Stage 3 — 시간 해상도 리샘플링

```bash
source .venv/bin/activate
python core/data_05_resample.py
```

빠른 실행(10MIN 생략):

```bash
python core/data_05_resample.py --skip-10min
```

완료 확인:
- [ ] ~/data/KIER_3_Temporal_Resolution/resample_log.csv`
- [ ] ~/data/KIER_3_Temporal_Resolution/{method}/KIER_USAGE_ELEC_INST_INTERP_{RES}.csv`

---

## 4) Stage 4 — 군집화 계수 기반 최적 K 선정

권장 실행:

```bash
source .venv/bin/activate
python core/data_08_optimal_k.py --resolutions 1D 1W 1M
```

전체 해상도:

```bash
python core/data_08_optimal_k.py
```

완료 확인:
- [ ] `results/clustering/optimal_k.json`
- [ ] `results/clustering/result_clustering.md`

---

## 5) Stage 5 — 최적 K 라벨링 및 군집 분리

권장 실행:

```bash
source .venv/bin/activate
python core/data_09_cluster_label.py --resolutions 1D 1W 1M
```

전체 해상도:

```bash
python core/data_09_cluster_label.py
```

완료 확인:
- [ ] `results/clustering/label_cluster.md`
- [ ] ~/data/KIER_4_Clustered/seasonal_naive/{algo}/KIER_USAGE_ELEC_INST_K*_{res}.csv`

---

## 6) Stage 6 — 모델 학습/비교 (모델별 분리 실행)

기존 노트북 일괄 실행 대신, 아래처럼 **모델별 스크립트 개별 실행** 권장.

중요 구분:

- 논문 기준 ML 성능표: `results/reference_legacy_ml/legacy_ml_comparison_performance_long.csv`, `results/reference_legacy_ml/legacy_ml_comparison_performance_wide.csv`
- 논문 기준 재실험 경로: `src/legacy_aggregate_experiment.py`, `src/run_legacy_aggregate.sh`
- 현재 Stage 6 panel 경로: `src/model_ml_panel_checkpointed.py`, `src/run_stage6_panel.sh`
- 현재 single-household DL 경로: `src/model_dl_single.py`, `src/run_dl_*.py`

현재 Stage 6 panel/DL-single 결과가 논문 ML보다 낮게 나오는 것은 자연스럽다. 이전 실험은 날씨/시간값과 다수 세대의 동시점 전력 사용량을 aggregate target 예측 입력으로 사용했고, 현재 panel/DL-single은 leakage를 피한 lag/sequence 기반 개별 세대 예측에 가깝다.

### 6-0) 배치 실행 스크립트 (에이전트 공용)

세션 충돌 방지를 위해 ML/DL은 분리 실행한다.

```bash
# ML 전용 배치
bash src/run_ml_batch.sh

# DL 전용 배치
bash src/run_dl_batch.sh

# (호환) 기존 래퍼
bash src/run_all_models_batch.sh ml
bash src/run_all_models_batch.sh dl
```

실행 중단/정리:

```bash
bash src/stop_batches.sh all
# 또는
bash src/stop_batches.sh ml
bash src/stop_batches.sh dl
```

런타임 로그/PID:
- `results/runtime/ml_batch_*.log`, `results/runtime/ml_batch.pid`
- `results/runtime/dl_batch_*.log`, `results/runtime/dl_batch.pid`

### 6-1) ML 모델별 실행

#### 논문 기준 참조 ML 성능표

스모크 실행으로 성능을 추정하지 않는다. 이전 참조 실험에서 저장된 성능표를 사용한다.

```bash
ls -lah results/reference_legacy_ml
python - <<'PY'
import pandas as pd
wide = pd.read_csv("results/reference_legacy_ml/legacy_ml_comparison_performance_wide.csv")
print(wide[["model", "condition", "control_R2", "exp2_sum_R2", "exp2_minus_control_R2"]].to_string(index=False))
PY
```

완전성 기준:

- `legacy_ml_comparison_performance_long.csv`: 90 rows
- `legacy_ml_comparison_performance_wide.csv`: 20 rows
- 모델: `CB`, `DT`, `LGBM`, `RF`, `XGB`
- 조건: `K2M`, `K2W`, `K3M`, `K3W`
- 그룹: 대조군, 실험군 01, 실험군 02

#### 공통 실행기

```bash
python src/model_ml_single.py --model CatBoost
```

#### 모델별 런처

```bash
source .venv/bin/activate
python src/run_ml_catboost.py
python src/run_ml_decisiontree.py
python src/run_ml_lightgbm.py
python src/run_ml_randomforest.py
python src/run_ml_xgboost.py
```

#### Legacy aggregate ML 모델별 실행 스크립트

현재 데이터로 논문 실험 형상을 재실행할 때만 사용한다. 참조 성능표 자체는 위 CSV를 우선한다.

```bash
# CatBoost
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=CatBoost SCOPE=both RESOLUTION=1H \
  RESULTS_DIR=results/legacy_aggregate_ml_catboost_1h bash src/run_legacy_aggregate.sh

# DecisionTree
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=DecisionTree SCOPE=both RESOLUTION=1H \
  RESULTS_DIR=results/legacy_aggregate_ml_decisiontree_1h bash src/run_legacy_aggregate.sh

# LightGBM
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=LightGBM SCOPE=both RESOLUTION=1H \
  RESULTS_DIR=results/legacy_aggregate_ml_lightgbm_1h bash src/run_legacy_aggregate.sh

# RandomForest
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=RandomForest SCOPE=both RESOLUTION=1H \
  RESULTS_DIR=results/legacy_aggregate_ml_randomforest_1h bash src/run_legacy_aggregate.sh

# XGBoost
PYTHON_BIN=.venv/bin/python FAMILY=ml MODEL=XGBoost SCOPE=both RESOLUTION=1H \
  RESULTS_DIR=results/legacy_aggregate_ml_xgboost_1h bash src/run_legacy_aggregate.sh
```

#### 파라미터 조정 예시

```bash
python src/run_ml_lightgbm.py --n-households 10 --n-iter 30 --cv-splits 3 --n-jobs 1
```

#### 체크포인트/재개 실행 (권장)

ML도 세대별 checkpoint를 남긴다. 한 세대가 끝날 때마다
`results/models/ml_<model>_detail.csv`와
`results/models/ml_<model>_best_params.json`이 갱신된다.

```bash
# 1) 대상 세대/완료 여부 확인
python src/run_ml_catboost.py --step plan --n-households 30

# 2) 한 번에 1~2세대씩만 실행
python src/run_ml_catboost.py \
  --step train \
  --n-households 30 \
  --n-iter 120 \
  --cv-splits 5 \
  --max-households-per-run 1

# 3) 완료된 detail 파일 기준 summary 생성
python src/run_ml_catboost.py --step finalize --n-households 30
```

중단 후 같은 명령을 다시 실행하면 완료된 세대는 자동 skip된다.
기존 결과를 무시하고 재실행할 때만 `--no-resume`을 붙인다.

#### Trial 단위 checkpoint 실행 (장기 운영 권장)

`RandomizedSearchCV` 대신 동일한 hyperparameter 후보를 직접 trial loop로 실행한다.
각 trial이 끝날 때마다 `results/models/checkpoints/<model>/<household>_trials.csv`에
저장되므로, 세대 내부에서 중단되어도 완료 trial을 재사용할 수 있다.

```bash
# feature cache 선생성(선택, 정확도 영향 없음)
# feature 정의를 수정한 뒤 첫 실행은 --force로 기존 cache를 덮어쓴다.
source .venv/bin/activate
python src/build_ml_feature_cache.py --n-households 30 --force

# 누수 진단: exact target leakage가 있으면 실패 처리한다.
python src/diagnose_ml_leakage.py --n-households 30 --target-mode delta --fail-on exact

# 입력 feature mode:
# - single: 본인 세대 과거값만 사용(하한선 참고용)
# - all_households: 본인 + 전체 세대의 과거값 사용(강한 대조군)
# - cluster_households: 본인 + 동일 클러스터 세대의 과거값 사용(ClusterBoost 실험군)
#
# 핵심 비교는 all_households vs cluster_households다.




# 가장 안전한 장기 실행: 모델별/세대별/ trial별로 조금씩 저장
STAGE6_MODELS=catboost,decisiontree,lightgbm,randomforest,xgboost \
STAGE6_STEP=train \
STAGE6_N_HOUSEHOLDS=30 STAGE6_N_ITER=120 STAGE6_CV_SPLITS=5 \
STAGE6_MAX_HOUSEHOLDS_PER_RUN=1 STAGE6_MAX_TRIALS_PER_RUN=1 \
TARGET_MODE=delta SCORING=r2 STAGE6_PEAK_QUANTILE=0.95 \
STAGE6_FEATURE_MODE=all_households STAGE6_PEER_LAGS=1 \
bash src/run_stage6_checkpointed.sh

# ClusterBoost 실험군
STAGE6_MODELS=catboost,decisiontree,lightgbm,randomforest,xgboost \
STAGE6_STEP=train \
STAGE6_N_HOUSEHOLDS=30 STAGE6_N_ITER=120 STAGE6_CV_SPLITS=5 \
STAGE6_MAX_HOUSEHOLDS_PER_RUN=1 STAGE6_MAX_TRIALS_PER_RUN=1 \
TARGET_MODE=delta SCORING=r2 STAGE6_PEAK_QUANTILE=0.95 \
STAGE6_FEATURE_MODE=cluster_households STAGE6_PEER_LAGS=1 \
bash src/run_stage6_checkpointed.sh

# 완료된 detail 기준 summary 생성
STAGE6_STEP=finalize bash src/run_stage6_checkpointed.sh
```

<!-- STAGE6_CURRENT_STATUS_START -->

#### 현재 운영 스냅샷 (2026-07-16 KST)

- 기존 `all_households` / `cluster_households` feature-mode runner는 main comparison에서 폐기했다.
- 폐기 사유: 세대별 local model + peer lag feature 구조라서, 의도한 “전체 panel 학습 후 세대 예측” 대조군 및 “cluster panel 학습 후 예하 세대 예측” 실험군과 의미가 다르다.
- Active runner: `src/run_stage6_panel.sh`
- Active ML implementation: `src/model_ml_panel_checkpointed.py`
- 대조군: `STAGE6_PANEL_MODE=global`
- 실험군: `STAGE6_PANEL_MODE=cluster`
- 빠른 검증 모델: `catboost,lightgbm,xgboost`
- 최종 ML 비교 모델: `catboost,lightgbm,xgboost,decisiontree,randomforest`
- Metric: `MAE`, `MAPE`, `MSE`, `RMSE`, `SMAPE`, `R2`, `MASE`
- Smoke 검증: LightGBM global/cluster panel 4세대 1 trial 성공

<!-- STAGE6_CURRENT_STATUS_END -->


중단:

```bash
bash src/stop_batches.sh stage6
```

진행 확인:

```bash
tail -f results/runtime/stage6_checkpointed_*.log
ls results/models/checkpoints/catboost
```

평가 설계:

- `R2` 단독 해석 금지: `MAE`, `RMSE`, `SMAPE`, `MASE`를 함께 본다.
- baseline은 `lag_1`, `lag_144`(1일 전), `lag_1008`(1주 전)을 모두 기록한다.
- `rmse_improvement_vs_best_naive`로 가장 강한 naive 대비 개선율을 본다.
- train 구간 `STAGE6_PEAK_QUANTILE` 이상을 peak로 정의하고 `peak_*`, `nonpeak_*` 지표를 분리해 본다.

배치 스크립트 파라미터(권장):

```bash
ML_N_HOUSEHOLDS=30 ML_N_ITER=120 ML_CV_SPLITS=5 ML_N_JOBS=1 \
TARGET_MODE=delta SCORING=r2 \
bash src/run_ml_batch.sh
```

체크포인트 배치 예시:

```bash
ML_MODELS=catboost ML_STEP=train \
ML_N_HOUSEHOLDS=30 ML_N_ITER=120 ML_CV_SPLITS=5 \
ML_MAX_HOUSEHOLDS_PER_RUN=1 \
bash src/run_ml_batch.sh
```

특정 모델만 실행:

```bash
ML_MODELS=catboost,lightgbm \
bash src/run_ml_batch.sh
```

### 6-2) DL 모델별 실행 (TCN 포함)

#### Peer-household GridSearch 5CV DL 실행

논문 때 논의한 “날씨변수 + 타겟 제외 347개 세대” 입력 조건으로 타겟 세대를 예측하려면 아래 runner를 사용한다.

- Main runner: `src/model_dl_peer_grid_cv.py`
- Batch runner: `src/run_dl_peer_grid_batch.sh`
- 결과 문서: `docs/dl_peer_grid_cv_experiment.md`
- Smoke 결과: `results/dl_peer_grid_cv_smoke_all/dl_peer_smoke_summary.csv`

입력 검증 조건:

- 전력 세대 컬럼 `348`
- 입력 peer 세대 컬럼 `347`
- 날씨 변수 `17`
- 총 입력 변수 `364`
- 타겟 세대 컬럼은 입력에서 제외

전체 DL 모델 smoke:

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

모델별 실행 스크립트:

```bash
python src/run_dl_peer_legacy_cnnlstm.py --grid-preset small --cv-splits 5
python src/run_dl_peer_legacy_seq2seq.py --grid-preset small --cv-splits 5
python src/run_dl_peer_cnnlstm.py --grid-preset small --cv-splits 5
python src/run_dl_peer_gru.py --grid-preset small --cv-splits 5
python src/run_dl_peer_transformer.py --grid-preset small --cv-splits 5
python src/run_dl_peer_tcn.py --grid-preset small --cv-splits 5
python src/run_dl_peer_retnet.py --grid-preset small --cv-splits 5
```

Grid preset:

- `smoke`: 1 combo, 1 epoch, 실행성/계약 검증용
- `small`: 8 combos, 기본 탐색용
- `paper`: 48 combos, 장기 탐색용

#### Legacy aggregate DL 실행 원칙

논문 ML과 같은 단위로 DL을 비교하려면 single-household runner가 아니라 aggregate runner를 사용한다.

- `ALL`: 대조군
- `K*`: 실험군 01, 군집별 직접 예측
- `cluster_ensemble`: 실험군 02, 모든 군집 예측 합산

계획 확인:

```bash
python src/run_legacy_aggregate_dl_legacy_cnnlstm.py --step plan
```

Legacy aggregate DL 모델별 실행 스크립트:

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

Batch:

```bash
bash src/run_legacy_aggregate_dl_batch.sh --step plan
DL_AGG_MODELS=legacy_cnnlstm,tcn EPOCHS=200 SEQ_LEN=3 \
  bash src/run_legacy_aggregate_dl_batch.sh
```

#### 공통 실행 템플릿 (복붙용)

아래 템플릿에서 `RUN_CMD`, `EPOCHS`, `BATCH`, `N_HOUSEHOLDS`, `CHUNK`만 바꿔서 실행한다.  
기본 resume 모드이므로 이미 완료된 세대는 자동 skip된다.

```bash
RUN_CMD="python src/run_dl_tcn.py"    # run_dl_cnnlstm.py / run_dl_gru.py / run_dl_transformer.py / run_dl_tcn.py / run_dl_retnet.py
N_HOUSEHOLDS=10
EPOCHS=50
BATCH=128
CHUNK=2                  # 1회 실행당 처리 세대 수
START=0                  # pending offset (0, 2, 4, ...)

# 1) plan (최초 1회)
$RUN_CMD --step plan \
  --n-households "$N_HOUSEHOLDS"

# 2) train (여러 번 반복 실행)
$RUN_CMD --step train \
  --n-households "$N_HOUSEHOLDS" \
  --epochs "$EPOCHS" --batch-size "$BATCH" \
  --start-index "$START" \
  --max-households-per-run "$CHUNK" \
  --fit-verbose 2

# 3) finalize (마지막 1회)
$RUN_CMD --step finalize \
  --n-households "$N_HOUSEHOLDS"
```

> `run_dl_*.py`는 내부에서 `.venv-dl`가 있으면 자동 사용하고,  
> `MPLCONFIGDIR / OMP_NUM_THREADS / TF_NUM_INTRAOP_THREADS / TF_NUM_INTEROP_THREADS`를 자동 세팅한다.

#### 모델별 대입 하이퍼파라미터 (권장 시작점)

- `MODEL=1D_CNN_LSTM` : `EPOCHS=80`, `BATCH=64`,  `CHUNK=1`
- `MODEL=GRU`         : `EPOCHS=60`, `BATCH=128`, `CHUNK=2`
- `MODEL=Transformer` : `EPOCHS=60`, `BATCH=64`,  `CHUNK=1`
- `MODEL=TCN`         : `EPOCHS=50`, `BATCH=128`, `CHUNK=2`
- `MODEL=RetNet`      : `EPOCHS=60`, `BATCH=64`,  `CHUNK=1`

#### 모델별 빠른 예시 (직접 실행용)

```bash
# CNN-LSTM
python src/run_dl_cnnlstm.py --step train --n-households 10 --epochs 80 --batch-size 64 --max-households-per-run 1 --fit-verbose 2

# GRU
python src/run_dl_gru.py --step train --n-households 10 --epochs 60 --batch-size 128 --max-households-per-run 2 --fit-verbose 2

# Transformer
python src/run_dl_transformer.py --step train --n-households 10 --epochs 60 --batch-size 64 --max-households-per-run 1 --fit-verbose 2

# TCN
python src/run_dl_tcn.py --step train --n-households 10 --epochs 50 --batch-size 128 --max-households-per-run 2 --fit-verbose 2

# RetNet
python src/run_dl_retnet.py --step train --n-households 10 --epochs 60 --batch-size 64 --max-households-per-run 1 --fit-verbose 2
```

배치 스크립트 파라미터(권장):

```bash
DL_STEP=train DL_N_HOUSEHOLDS=10 DL_EPOCHS=60 DL_PATIENCE=20 \
DL_SEQ_LEN=288 DL_BATCH_SIZE=64 DL_MAX_HOUSEHOLDS_PER_RUN=1 \
TARGET_MODE=delta \
bash src/run_dl_batch.sh
```

특정 모델만 실행:

```bash
DL_MODELS=gru,tcn bash src/run_dl_batch.sh
```

### Stage 6 결과 파일

<!-- STAGE6_RESULT_PATHS_START -->

현재 checkpoint 기반 Stage 6 결과 파일:

- Control (`all_households`):
  - `results/stage6_all_households/ml_<model>_all_households_detail.csv`
  - `results/stage6_all_households/ml_<model>_all_households_summary.csv`
  - `results/stage6_all_households/checkpoints/<model>/all_households/<household>_trials.csv`
  - `results/stage6_all_households/predictions/<model>/all_households/<household>_predictions.csv`
- Experiment (`cluster_households`):
  - `results/stage6_cluster_households/ml_<model>_cluster_households_detail.csv`
  - `results/stage6_cluster_households/ml_<model>_cluster_households_summary.csv`
  - `results/stage6_cluster_households/checkpoints/<model>/cluster_households/<household>_trials.csv`
  - `results/stage6_cluster_households/predictions/<model>/cluster_households/<household>_predictions.csv`

<!-- STAGE6_RESULT_PATHS_END -->

- ML:
  - `results/models/ml_<model>_detail.csv`
  - `results/models/ml_<model>_summary.csv`
  - `results/models/ml_<model>_best_params.json`
- DL:
  - `results/models/dl_<model>_plan.csv`
  - `results/models/dl_<model>_detail.csv`
  - `results/models/dl_<model>_summary.csv`

---

## 7) Stage 7 — 결과 취합 및 최종 모델 선정

### 산출물 확인

```bash
ls -lah results/preprocessing
ls -lah results/clustering
ls -lah results/models
```

### 체크포인트
- [ ] 보간법 최종 선정 근거(지표/그래프)
- [ ] 군집화 최적 K 및 라벨 품질 확인
- [ ] ML/DL 모델별 성능 표 비교
- [ ] 최종 운영 후보 1~2개 선정 및 근거 정리

---

## 전체 진행 요약 템플릿

- [ ] Stage 1 완료
- [ ] Stage 2 완료
- [ ] Stage 3 완료
- [ ] Stage 4 완료
- [ ] Stage 5 완료
- [ ] Stage 6-1 (ML) 완료
- [ ] Stage 6-2 (DL) 완료
- [ ] Stage 7 완료
