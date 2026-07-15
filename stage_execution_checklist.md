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

## 6) Stage 6 — 모델 학습 및 성능 비교 (TCN 포함)

현재 모델 비교는 노트북 실행 중심이다.

### 6-1) ML 비교
- 대상: CatBoost, XGBoost, LightGBM, DecisionTree, RandomForest
- 예상 소요: **30분~3시간**

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
  src/model_ml_02_comparison.ipynb
```

### 6-2) DL 비교
- 대상: 1D-CNN LSTM, GRU, Transformer, **TCN**, RetNet
- 예상 소요: **2~12시간** (GPU 유무 영향 큼)

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
  src/model_dl_02_comparison.ipynb
```

### 완료 확인
- [ ] `results/models/` 내 비교 결과 파일 생성

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

