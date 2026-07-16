# 코드 구조 및 소스 파일 설명

## 디렉터리 구조

```
WS_TimeSerial_ClusterBoost_Energy/
├── core/                  # 공용 Python 모듈 (18개 .py)
├── src/                   # 실험 워크플로우 (Jupyter 노트북, 9개 .ipynb)
├── data/
│   ├── KMA_ASOS/          # 기상청 ASOS 기상 관측 CSV
│   └── KIER_Usage/        # KIER 세대별 에너지 사용량 원시 데이터
├── results/               # 실험 결과물 (텍스트·이미지)
├── paper/                 # 논문 참고 자료 (PDF 제외, .gitignore 처리)
├── cursor/                # 리팩토링 변경 이력
├── requirements.txt
└── README.md
```

---

## core/ — 공용 모듈

### 데이터 처리

| 파일 | 역할 |
|---|---|
| `data_pipeline.py` | Raw long-format → Wide-format pivot (`raw_to_wide`), 3단계 결측치 보간 (`fill_missing_wide`) (단, 결측치 보간은 각 시간대별 전 세대 평균 사용량으로 대치 및 선형 보간 방법을 사용함. 보간 방법 선정 과정의 경우 논문에 포함되어있지 않음 (향후 과제에서 추가 예정)) |
| `data_preprocessing.py` | 시간 단위별 리샘플링(`resample_by_last`), IQR 기반 이상치 제거 |
| `data_analysis.py` | 기술통계, 이상치 탐지, OLS 회귀 잔차, 예측 지표(MAE/MSE/RMSE/R²) 계산 |
| `data_datetime.py` | 날짜·시간 파싱·검증, 공휴일 포함 달력 특성 생성 |
| `data_clustering.py` | K-Means 단일·다회차 실행, Elbow/Silhouette/CH/Dunn 평가, 클러스터 시각화 |
| `data_visualization.py` | 시계열 라인 플롯, 클러스터별 히트맵 등 공용 시각화 |

### 데이터 제공처 연동

| 파일 | 제공처 | 주요 기능 |
|---|---|---|
| `provider_kma.py` | 기상청 KMA | ASOS API 수집, CSV 다운로드, 결측치 보간 |
| `provider_kier.py` | 에너지기술연구원 KIER | 적산(누적) → 순시(차분) 사용량 변환 |
| `provider_kier_m02.py` | KIER M02 전용 | 도메인(전기·열·가스 등) 및 파일 경로 매핑 상수 |
| `provider_kasi.py` | 천문연구원 KASI | 공휴일·국경일·기념일 OpenAPI 수집 |
| `provider_keco.py` | 환경공단 KECO | AirKorea 대기질 데이터 정제·보간 |
| `provider_kdhc.py` | 지역난방공사 KDHC | 시간대별 열공급량 OpenAPI 수집 |
| `provider_korex.py` | 한국도로공사 KorEx | 톨게이트 교통량 API 수집 |

### 모델

| 파일 | 역할 |
|---|---|
| `model_ml.py` | CB / DT / LGBM / RF / XGB 학습·예측·KFold CV, 지표·시각화 공용 함수 (DL 모듈에서도 import) |
| `model_dl.py` | 1D-CNN LSTM · Seq2Seq · TCN · Transformer · RetNet 학습·예측·KFold CV, `model_ml`의 공용 함수 재사용 |

### 외부 참조

| 파일 | 역할 |
|---|---|
| `ref_cluster_eval.py` | Dunn Index 계산 (완전 연결 직경 기반, `scipy.spatial.distance` 벡터화) |
| `ref_deepant.py` | DeepAnT 시계열 이상치 탐지 모델 (외부 코드 기반, PyTorch Lightning) |
| `ref_deepant_utils.py` | DeepAnT 예측 Loss 분포 시각화 유틸리티 |

---

## src/ — 실험 노트북

노트북은 데이터 처리 → 클러스터링 → 모델링 순서로 실행한다.

| 파일 | 단계 | 내용 |
|---|---|---|
| `data_01_pipeline.ipynb` | 전처리 | hList 구성 → Wide 변환 → INST 계산 → 보간 → IQR → ACCU 복원 → 리샘플링 → 통합 저장 |
| `data_02_interp_eval.ipynb` | 전처리 | 보간법 정량 비교 (현행 행평균+선형 vs. PCHIP vs. STL 분해 재합성), 선정 근거 확보 |
| `data_03_stats.ipynb` | 탐색 | 세대별 전력 사용량 기술통계, 분포 시각화, 이상치 분석 |
| `data_04_weather.ipynb` | 전처리 | KMA·KDHC 기상/열공급 데이터 수집·정제·병합 |
| `clustering_01_kmeans.ipynb` | 클러스터링 | 최적 K 탐색 (4종 평가지표 × 5개 시간 단위) → K-Means 군집화 |
| `clustering_02_integration.ipynb` | 클러스터링 | 클러스터 레이블 기반 세대 데이터 통합·저장 |
| `model_00_index.ipynb` | 모델 진입점 | `model_ml_01_cv.ipynb` / `model_dl_01_cnnlstm.ipynb` 실행 진입점, 파라미터 안내 |
| `model_ml_01_cv.ipynb` | 모델 학습 | CB/XGB/LGBM/DT/RF 단일 학습 → KFold CV → 클러스터별 CV → 앙상블 비교 |
| `model_dl_01_cnnlstm.ipynb` | 모델 학습 | 1D-CNN LSTM · Seq2Seq · TCN · Transformer · RetNet 단일·군집화 KFold CV · 전체 비교 |

---

## data/

| 경로 | 내용 |
|---|---|
| ~/data/KMA_ASOS/` | 기상청 ASOS 시간별 관측값 CSV (기온·강수·풍속·습도 등) |
| ~/data/KIER_Usage/` | KIER M02 프로젝트 세대별 10분 단위 에너지 사용량 원시 데이터 |

---

## results/

실험 결과 파일. 파일명 규칙:

```
ELEC_Comparison_{모델}_{K수}{시간단위}[_{클러스터 구성}].txt   # KFold CV 비교 결과 (텍스트)
ELEC_Comparison of Model Pred_{모델}_{K수}{시간단위}_{구성}.png  # 예측 비교 시각화
ELEC_Visualized_{시간단위}_{K수}_{클러스터 구성}.png            # 클러스터 시각화
[BU] Statistical Analysis_*/                                   # 통계 분석 백업 (KFold 이력 포함)
```

예시: `ELEC_Comparison_CB_K2M.txt` → CatBoost, K=2, 월별 리샘플 결과

---

## 환경 설정

```bash
pip install -r requirements.txt
```

주요 의존성: `pandas`, `numpy`, `scikit-learn`, `catboost`, `lightgbm`, `xgboost`,
`tensorflow`/`keras`, `torch`, `pytorch-lightning`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`
