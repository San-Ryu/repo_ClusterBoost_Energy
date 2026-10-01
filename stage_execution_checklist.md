# Stage별 실행 체크리스트 (KIER 에너지 예측)

본 문서는 데이터 정제부터 클러스터링/모델 비교까지 실제 실행 가능한 순서, 예상 소요 시간, 실행 코드를 정리한 운영용 체크리스트다.

---

## 0) 실행 전 준비

- [ ] 가상환경 생성 및 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] 보간 폴더 경로 정합 확인  
현재 코드의 기본 입력 경로와 실제 폴더명이 다를 수 있으므로 심볼릭 링크를 맞춘다.

```bash
ln -sfn "KIER_2_interpolation" ~/data/KIER_S2_interp"
```

---

## 1) Stage 1 — Raw -> House Dataset 생성

### 목적
- 원시 CSV에서 공통 세대 인덱스(`KIER_HINDEX.csv`) 생성
- 도메인별 Wide-format `ACCU/INST` 생성
- 세대별 분리 파일 생성

### 입력 데이터 규모 (참고)
- ~/data/KIER_0_USAGE_Raw/KIER_0_Raw`
- 주요 파일 크기: ELEC 2.8G, GAS 4.3G, WATER 2.7G, HEAT 2.3G, HOT 3.0G

### 예상 소요 시간
- **약 25~60분** (디스크 I/O 속도 영향 큼)

### 실행 코드

```bash
source .venv/bin/activate
python core/data_02_build_dataset.py
```

### 완료 확인
- [ ] ~/data/KIER_1_USAGE_House/KIER_HINDEX.csv` 생성
- [ ] ~/data/KIER_1_USAGE_House/KIER_USAGE_ELEC_INST.csv` 생성(약 604MB)
- [ ] ~/data/KIER_1_USAGE_House/KIER_USAGE_*_{ACCU,INST}.csv` 생성
- [ ] ~/data/KIER_1_USAGE_House/KIER_1_DIVIDED_BY_DOMAIN_HOUSE/` 생성

---

## 2) Stage 2 — 결측 보간

### 2-1) Phase 1: 전체 보간 파일 생성

### 목적
- baseline / pchip / stl / mstl / seasonal_naive 5가지 보간 적용

### 실측 소요 (로그 기준)
- baseline: 0.5초
- pchip: 0.9초
- seasonal_naive: 0.8초
- stl: 7,293초 (약 2.0시간)
- mstl: 17,046초 (약 4.7시간)
- **총합: 약 6.8~7.2시간**

### 실행 코드

```bash
source .venv/bin/activate
python core/data_04_interp.py --phase 1
```

### 완료 확인
- [ ] ~/data/KIER_2_interpolation/phase1_log.csv` 생성
- [ ] ~/data/KIER_2_interpolation/{method}/KIER_USAGE_ELEC_INST_INTERP.parquet` 생성

---

### 2-2) Phase 2: 인공 마스킹 기반 정량 평가

### 목적
- 보간법별 RMSE/ACF/Wasserstein 등 지표 비교
- 최종 보간법 선정 근거 확보

### 예상 소요 시간
- 빠른 검증(3개 방법): **5~20분**
- 전체 정밀평가(5개 방법): **2~8시간**

### 실행 코드 (빠른 검증)

```bash
source .venv/bin/activate
python core/data_04_interp.py --phase 2 \
  --methods baseline pchip seasonal_naive \
  --n-trials 10 --sample-n 20 --workers 4
```

### 실행 코드 (전체 정밀)

```bash
source .venv/bin/activate
python core/data_04_interp.py --phase 2 \
  --n-trials 10 --sample-n 30 --workers 4
```

### 완료 확인
- [ ] `results/preprocessing/ELEC_interp_metrics.csv`
- [ ] `results/preprocessing/ELEC_interp_summary.csv`
- [ ] `results/preprocessing/ELEC_interp_rank.csv`

---

## 3) Stage 3 — 시간 해상도 리샘플링

### 목적
- 10MIN/1H/1D/1W/1M 해상도로 변환

### 실측 소요
- 방법 1개 기준: 10MIN 13.6초 + 1H 2.5초 + 기타 < 0.5초
- 5개 방법 전체:
  - 10MIN 포함: **1.5~3분**
  - 10MIN 제외: **30~90초**

### 실행 코드

```bash
source .venv/bin/activate
python core/data_05_resample.py
```

### 빠른 실행 (권장)

```bash
python core/data_05_resample.py --skip-10min
```

### 완료 확인
- [ ] ~/data/KIER_3_Temporal_Resolution/resample_log.csv`
- [ ] ~/data/KIER_3_Temporal_Resolution/{method}/KIER_USAGE_ELEC_INST_INTERP_{RES}.csv`

---

## 4) Stage 4 — 군집화 계수 기반 최적 K 선정 (필수 게이트)

### 목적
- K=2~15 범위에서 Elbow/Silhouette/CH/Dunn/DBI 산출
- 다수결 기반 `K*` 결정

### 실측 소요
- 1D/1W/1M 실행 기준: **2.5초**
- 전체 해상도 기준 예상: **5~20초**

### 실행 코드

```bash
source .venv/bin/activate
python core/data_08_optimal_k.py --resolutions 1D 1W 1M
```

### 전체 해상도 실행

```bash
python core/data_08_optimal_k.py
```

### 완료 확인
- [ ] `results/clustering/optimal_k.json`
- [ ] `results/clustering/result_clustering.md`

---

## 5) Stage 5 — 최적 K 기반 라벨링 및 군집 분리

### 목적
- KMeans/MeanShift/DBSCAN/L1L2/GMM으로 세대 라벨 생성
- 군집별 데이터셋 분리 저장

### 실측/예상 소요
- 1D만 실행 시: 1.6초
- 전체 해상도+알고리즘: **10~90초**

### 실행 코드

```bash
source .venv/bin/activate
python core/data_09_cluster_label.py --resolutions 1D 1W 1M
```

### 전체 실행

```bash
python core/data_09_cluster_label.py
```

### 완료 확인
- [ ] `results/clustering/label_cluster.md`
- [ ] ~/data/KIER_4_Clustered/seasonal_naive/{algo}/KIER_USAGE_ELEC_INST_K*_{res}.csv`

---

## 6) Stage 6 — Panel 기반 ML checkpoint 실험 및 성능 비교

현재 Stage 6의 핵심 비교는 세대별 local model이 아니라 panel 기반 비교다.
대조군과 실험군은 모두 시간순 split을 사용하며, shuffle은 사용하지 않는다.

### 현재 운영 기준

- Repository: `/Users/labq_s.h.ryu/repo_ClusterBoost_Energy`
- Shared data root: `/Users/labq_s.h.ryu/data`
- 대조군: `STAGE6_PANEL_MODE=global`
  - 전체 348개 세대의 train 기간 데이터를 하나의 panel dataset으로 학습
  - test 기간에서 각 세대별 예측값과 성능을 산출
- 실험군: `STAGE6_PANEL_MODE=cluster`
  - cluster별 train 기간 데이터를 학습
  - 해당 cluster 예하 세대별 예측값과 성능을 산출
- Target: `level` 기본값
- Scoring: `r2`
- HPO: group 단위 checkpoint
  - global: 모델별 1회 search
  - cluster: 모델별 cluster 단위 search
- 검증: `TimeSeriesSplit`, `shuffle=False`
- Metric: `MAE`, `MAPE`, `MSE`, `RMSE`, `SMAPE`, `R2`, `MASE`, best naive 대비 RMSE 개선률
- 빠른 검증 모델: `catboost,lightgbm,xgboost`
- 최종 ML 비교 모델: `catboost,lightgbm,xgboost,decisiontree,randomforest`
  - DecisionTree/RandomForest는 절대 성능이 낮더라도 이전 연구에서 개선률 해석에 필요하므로 최종 비교에는 포함한다.
- DL 모델은 같은 panel/global-vs-cluster 의미로 별도 runner를 추가한 뒤 비교한다. 기존 local DL runner 결과와 직접 비교하지 않는다.

### 6-0) 사전 확인

```bash
cd "/Users/labq_s.h.ryu/repo_ClusterBoost_Energy"
source .venv/bin/activate
screen -ls
```

### 6-1) 빠른 대조군 검증: global panel / 3개 ML

```bash
cd "/Users/labq_s.h.ryu/repo_ClusterBoost_Energy"
STAGE6_PANEL_MODE=global STAGE6_MODELS=catboost,lightgbm,xgboost STAGE6_STEP=all STAGE6_N_HOUSEHOLDS=348 STAGE6_N_ITER=60 STAGE6_CV_SPLITS=5 STAGE6_MAX_GROUPS_PER_RUN=1 STAGE6_MAX_TRIALS_PER_RUN=0 TARGET_MODE=level SCORING=r2 STAGE6_EXTRA_FLAGS="--results-dir results/stage6_panel_global --checkpoint-dir results/stage6_panel_global/checkpoints --predictions-dir results/stage6_panel_global/predictions" bash src/run_stage6_panel.sh
```

### 6-2) 빠른 실험군 검증: cluster panel / 3개 ML

```bash
cd "/Users/labq_s.h.ryu/repo_ClusterBoost_Energy"
STAGE6_PANEL_MODE=cluster STAGE6_MODELS=catboost,lightgbm,xgboost STAGE6_STEP=all STAGE6_N_HOUSEHOLDS=348 STAGE6_N_ITER=60 STAGE6_CV_SPLITS=5 STAGE6_MAX_GROUPS_PER_RUN=1 STAGE6_MAX_TRIALS_PER_RUN=0 TARGET_MODE=level SCORING=r2 STAGE6_EXTRA_FLAGS="--results-dir results/stage6_panel_cluster --checkpoint-dir results/stage6_panel_cluster/checkpoints --predictions-dir results/stage6_panel_cluster/predictions" bash src/run_stage6_panel.sh
```

### 6-3) 최종 ML 비교

빠른 검증에서 이전 연구 수준의 성능 범위가 회복되는지 확인한 뒤, 최종 ML 비교에는 5개 모델을 모두 포함한다.

```bash
# global control
STAGE6_PANEL_MODE=global STAGE6_MODELS=catboost,lightgbm,xgboost,decisiontree,randomforest \
STAGE6_STEP=all STAGE6_N_HOUSEHOLDS=348 STAGE6_N_ITER=120 STAGE6_CV_SPLITS=5 \
STAGE6_MAX_GROUPS_PER_RUN=1 STAGE6_MAX_TRIALS_PER_RUN=0 \
TARGET_MODE=level SCORING=r2 \
STAGE6_EXTRA_FLAGS="--results-dir results/stage6_panel_global --checkpoint-dir results/stage6_panel_global/checkpoints --predictions-dir results/stage6_panel_global/predictions" \
bash src/run_stage6_panel.sh

# cluster experiment
STAGE6_PANEL_MODE=cluster STAGE6_MODELS=catboost,lightgbm,xgboost,decisiontree,randomforest \
STAGE6_STEP=all STAGE6_N_HOUSEHOLDS=348 STAGE6_N_ITER=120 STAGE6_CV_SPLITS=5 \
STAGE6_MAX_GROUPS_PER_RUN=1 STAGE6_MAX_TRIALS_PER_RUN=0 \
TARGET_MODE=level SCORING=r2 \
STAGE6_EXTRA_FLAGS="--results-dir results/stage6_panel_cluster --checkpoint-dir results/stage6_panel_cluster/checkpoints --predictions-dir results/stage6_panel_cluster/predictions" \
bash src/run_stage6_panel.sh
```

### 진행 확인

```bash
screen -ls
tail -f results/runtime/stage6_panel_*.log
find results/stage6_panel_global/checkpoints -name '*_trials.csv' | wc -l
find results/stage6_panel_cluster/checkpoints -name '*_trials.csv' | wc -l
```

### 현재 체크포인트 스냅샷 (2026-07-16 KST)

- [x] 기존 single-household 의미의 `stage6_all_households` 프로세스 종료
- [x] 기존 single-household 의미의 Stage 6 결과/로그 폐기
- [x] 새 ML runner 추가: `src/model_ml_panel_checkpointed.py`
- [x] 새 실행 wrapper 추가: `src/run_stage6_panel.sh`
- [x] 기존 ML/DL metric에 `MSE` 추가
- [x] smoke 검증: LightGBM global panel 4세대 1 trial 성공
- [x] smoke 검증: LightGBM cluster panel 4세대 1 trial 성공
- [ ] global panel 348세대 / 3개 ML 빠른 검증
- [ ] cluster panel 348세대 / 3개 ML 빠른 검증
- [ ] 최종 ML 5개 모델 비교
- [ ] panel 의미에 맞는 DL runner 추가 및 비교

### 완료 확인

- [ ] `results/stage6_panel_global/ml_<model>_global_detail.csv`
- [ ] `results/stage6_panel_global/ml_<model>_global_summary.csv`
- [ ] `results/stage6_panel_cluster/ml_<model>_cluster_detail.csv`
- [ ] `results/stage6_panel_cluster/ml_<model>_cluster_summary.csv`
- [ ] 대조군/실험군 성능 비교표 작성

---

## 7) Stage 7 — 결과 취합 및 최종 모델 선정

### 목적
- 모델별 성능/시간/안정성 비교
- 클러스터링 유무 효과 분석
- 운영 배포 후보 1~2개 선정

### 실행 코드 (산출물 확인)

```bash
ls -lah results/clustering
ls -lah results/models
```

### 체크포인트
- [ ] MAE/RMSE/MAPE/R2 비교표 작성
- [ ] 추론시간/학습시간 포함 총평 작성
- [ ] 최종 1순위/2순위 모델 선정 및 근거 문서화

---

## 전체 예상 시간 요약

- 빠른 검증 루트 (seasonal_naive 중심 + 핵심 해상도 + 대표 모델): **약 1~4시간**
- 전체 풀스케일 루트 (5보간법 + 전체 해상도 + ML/DL 전체): **약 10~24시간**

