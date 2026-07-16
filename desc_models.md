# 예측 모델 설명 및 최적화 하이퍼파라미터

**작성일**: 2026-05-02  
**데이터**: ~/data/KIER_3_Temporal_Resolution/seasonal_naive/KIER_USAGE_ELEC_INST_INTERP_10MIN.csv`  
**예측 태스크**: 348세대 가정용 전기 소비량 단기 예측 (1-step ahead, 10분 후)

---

## 목차

1. [실험 설계](#1-실험-설계)
2. [ML 모델 (5종)](#2-ml-모델-5종)
   - CatBoost
   - Decision Tree
   - LightGBM
   - RandomForest
   - XGBoost
3. [DL 모델 (5종)](#3-dl-모델-5종)
   - 1D CNN-LSTM
   - GRU
   - Transformer
   - TCN
   - RetNet
4. [피처 공학](#4-피처-공학)
5. [하이퍼파라미터 최적화 전략](#5-하이퍼파라미터-최적화-전략)
6. [평가 지표](#6-평가-지표)

---

## 1. 실험 설계

### 데이터 개요

| 항목 | 값 |
|---|---|
| 데이터 경로 | ~/data/KIER_3_Temporal_Resolution/seasonal_naive/` |
| 보간 방식 | Seasonal Naive (일간 144스텝 주기 복사) |
| 시간 해상도 | 10분 |
| 기간 | 2022-07-17 ~ 2024-06-05 (약 23개월) |
| 총 행수 | 97,438행 |
| 세대 수 | 348세대 |
| 실험 세대 | 10세대 (랜덤 선택, seed=42) |

### 예측 태스크

```
X(t-seq:t) → ŷ(t+1)
```

- **입력**: 과거 관측값 + 시간 특징
- **출력**: 1스텝 후(10분 후) 전력 소비량 (kWh)
- **평가**: MAE, RMSE, MAPE, R² (실제 단위 kWh 기준)

### Train/Val/Test 분할

```
ML 모델: Train 85% | Test 15%         (시간순, shuffle=False)
DL 모델: Train 70% | Val 15% | Test 15% (시간순, shuffle=False)
```

> 전기 소비량은 강한 일간·주간 주기성을 가지므로 **시간순 분할**을 사용한다.
> 랜덤 분할 시 미래 정보가 훈련 데이터에 누출(data leakage)되어 과적합된다.

---

## 2. ML 모델 (5종)

ML 모델의 입력은 **태뷸러(tabular) 특징 벡터**로, 23개의 래그·롤링·시간 피처를 사용한다.  
각 타임스텝을 독립적인 행(row)으로 취급하며, 슬라이딩 윈도우 없이 단일 샘플로 예측한다.

### 2-1. CatBoost

**이론적 배경**  
CatBoost(Categorical Boosting, Prokhorenkova et al. 2018)는 그래디언트 부스팅 계열 앙상블 학습 알고리즘이다.  
기존 XGBoost/LightGBM의 **타겟 통계(target statistics)를 활용한 범주형 처리**와 **Ordered Boosting**을 핵심 기법으로 채택한다.

- **Ordered Boosting**: 잔차(residual) 계산 시 현재 관측보다 이전 데이터만 사용 → 과적합 억제
- **Symmetric Trees**: 모든 분기에서 동일한 분기 조건 적용 → 정규화 효과, 빠른 예측
- **내장 L2 정규화** (`l2_leaf_reg`): 리프 값의 L2 패널티

**구조**: KIER 전력 데이터의 연속 수치형 피처에 대해 범주형 인코딩 없이 적용.

**최적화 탐색 공간**

| 하이퍼파라미터 | 범위 | 의미 |
|---|---|---|
| `iterations` | 300 ~ 1000 | 트리 수 (앙상블 크기) |
| `depth` | 4 ~ 8 | 단일 트리 최대 깊이 |
| `learning_rate` | 0.01 ~ 0.15 | 그래디언트 스텝 크기 |
| `l2_leaf_reg` | 1 ~ 7 | 리프 값 L2 정규화 계수 |
| `bagging_temperature` | 0 ~ 1 | 베이지안 부트스트래핑 강도 |
| `random_strength` | 0 ~ 2 | 스플릿 선택 시 랜덤 노이즈 강도 |

**기타 고정 파라미터**

```python
od_type='Iter', od_wait=20   # Overfitting Detector
verbose=0
random_seed=42
```

---

### 2-2. Decision Tree

**이론적 배경**  
의사결정 나무(Breiman et al. 1984)는 피처 공간을 재귀적으로 이진 분할하여 예측값을 결정하는 비모수 모델이다.

- **분기 기준**: MSE 최소화 (회귀)
- **장점**: 해석 가능성 높음, 학습 빠름
- **단점**: 높은 분산(과적합), 부스팅·배깅 앙상블의 기본 학습기로 사용

**최적화 탐색 공간**

| 하이퍼파라미터 | 범위 | 의미 |
|---|---|---|
| `max_depth` | 4, 6, 8, 10, 12, None | 최대 트리 깊이 (None=제한 없음) |
| `min_samples_split` | 2 ~ 20 | 노드 분기 최소 샘플 수 |
| `min_samples_leaf` | 1 ~ 16 | 리프 노드 최소 샘플 수 |
| `max_features` | sqrt, log2, None, 0.5, 0.7 | 분기 탐색 시 고려할 피처 비율 |

> `max_depth`와 `min_samples_leaf`가 과적합 조절의 핵심 파라미터이다.

---

### 2-3. LightGBM

**이론적 배경**  
LightGBM(Ke et al. 2017)은 Microsoft가 개발한 그래디언트 부스팅 프레임워크로, **Leaf-wise Tree Growth**와 **Gradient-based One-Side Sampling(GOSS)**을 사용해 대용량 데이터에서 높은 속도를 달성한다.

- **Leaf-wise 성장**: 가장 손실 감소가 큰 리프를 우선 분기 → XGBoost(Level-wise)보다 정확도 높지만 과적합 위험
- **GOSS**: 작은 그래디언트 샘플을 하위 샘플링 → 데이터 효율 향상
- **Exclusive Feature Bundling(EFB)**: 희소 피처 번들링 → 차원 감소

**최적화 탐색 공간**

| 하이퍼파라미터 | 범위 | 의미 |
|---|---|---|
| `n_estimators` | 200 ~ 1000 | 부스팅 라운드 수 |
| `learning_rate` | 0.005 ~ 0.15 | 학습률 |
| `num_leaves` | 20 ~ 127 | 단일 트리 최대 리프 수 |
| `max_depth` | -1, 5, 7, 9, 11 | 트리 깊이 (-1=제한 없음) |
| `min_child_samples` | 10 ~ 100 | 리프 최소 데이터 수 |
| `subsample` | 0.6 ~ 1.0 | 행 샘플링 비율 |
| `colsample_bytree` | 0.6 ~ 1.0 | 열(피처) 샘플링 비율 |
| `reg_alpha` | 0 ~ 0.5 | L1 정규화 |
| `reg_lambda` | 0 ~ 0.5 | L2 정규화 |

> `num_leaves`와 `min_child_samples`가 Leaf-wise 과적합 방지의 핵심이다.

---

### 2-4. RandomForest

**이론적 배경**  
랜덤 포레스트(Breiman 2001)는 독립적으로 학습된 결정 트리 앙상블의 예측을 평균하는 **배깅(Bagging)** 알고리즘이다.

- **Bootstrap Sampling**: 훈련 데이터를 복원 추출하여 트리 다양성 확보
- **Random Feature Selection**: 각 노드에서 `max_features`개의 피처만 고려
- **장점**: 과적합에 강함, 변수 중요도 해석 가능
- **단점**: 메모리 사용량 큼, 예측 속도 느림

**최적화 탐색 공간**

| 하이퍼파라미터 | 범위 | 의미 |
|---|---|---|
| `n_estimators` | 100 ~ 400 | 트리 수 |
| `max_depth` | 5, 8, 12, 16, None | 트리 최대 깊이 |
| `min_samples_split` | 2 ~ 20 | 노드 분기 최소 샘플 수 |
| `min_samples_leaf` | 1 ~ 16 | 리프 최소 샘플 수 |
| `max_features` | sqrt, log2, 0.3, 0.5, 0.7 | 피처 샘플링 비율 |

---

### 2-5. XGBoost

**이론적 배경**  
XGBoost(Chen & Guestrin 2016)는 **Regularized Gradient Boosting**을 구현한 오픈소스 라이브러리로, L1·L2 정규화 항을 목적함수에 명시적으로 추가한다.

$$\mathcal{L} = \sum_{i} l(y_i, \hat{y}_i) + \sum_k \left[\gamma T_k + \frac{1}{2}\lambda \|w_k\|^2\right]$$

- **Exact/Histogram-based Greedy**: 효율적인 최적 분기점 탐색
- **Column Subsampling**: 트리·노드 단위 피처 샘플링
- **Early Stopping**: 검증 손실 모니터링으로 자동 조기 종료

**최적화 탐색 공간**

| 하이퍼파라미터 | 범위 | 의미 |
|---|---|---|
| `n_estimators` | 200 ~ 1000 | 부스팅 라운드 수 |
| `max_depth` | 3 ~ 8 | 트리 최대 깊이 |
| `learning_rate` | 0.01 ~ 0.15 | 학습률 (eta) |
| `subsample` | 0.6 ~ 1.0 | 행 샘플링 비율 |
| `colsample_bytree` | 0.6 ~ 1.0 | 트리당 열 샘플링 비율 |
| `reg_alpha` | 0 ~ 0.5 | L1 정규화 (λ₁) |
| `reg_lambda` | 0.5 ~ 2.0 | L2 정규화 (λ₂) |
| `min_child_weight` | 1 ~ 10 | 리프 최소 Hessian 합 |

---

## 3. DL 모델 (5종)

DL 모델의 입력은 **슬라이딩 윈도우 시퀀스**: `(batch, seq_len=144, n_features=7)`.  
7개 특징 = 에너지 소비량(MinMax 정규화) + 시간 인코딩 6개 (hour/dow/month × sin·cos).

### 공통 학습 설정

| 항목 | 값 |
|---|---|
| 손실 함수 | Huber Loss (이상치 강건성) |
| 옵티마이저 | Adam (lr=1e-3, clipnorm=1.0) |
| EarlyStopping | monitor=val_loss, patience=20, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=10, min_lr=1e-6 |
| Batch Size | 128 |
| Max Epochs | 200 |
| 정규화 | 세대별 MinMaxScaler (훈련 구간 fit) |

---

### 3-1. 1D CNN-LSTM

**이론적 배경**  
1D 합성곱(Conv1D)으로 지역 패턴을 추출한 후 LSTM으로 장기 시간 의존성을 포착하는 하이브리드 구조이다.

```
Input (B, 144, 7)
  → Conv1D(64, k=3, causal) → BN
  → Conv1D(128, k=3, causal) → BN
  → LSTM(128, return_seq=True, dropout=0.15)
  → LSTM(64, return_seq=False, dropout=0.15)
  → Dense(32, swish)
  → Dense(1)
Output (B, 1)
```

**설계 포인트**
- **Causal Padding**: 미래 정보 누출 방지 (t 시점이 t+k 정보를 보지 못함)
- **BatchNormalization**: 내부 공변량 이동(internal covariate shift) 완화
- **LSTM Dropout**: 순환 연결에 드롭아웃 → 과적합 억제

**하이퍼파라미터 (최적화 적용값)**

| 파라미터 | 값 |
|---|---|
| CNN 필터 | [64, 128] |
| CNN 커널 | 3 (causal padding) |
| LSTM 유닛 | [128, 64] |
| LSTM Dropout | 0.15 |
| FC 유닛 | 32 |
| 활성화 | swish |

---

### 3-2. GRU (Gated Recurrent Unit)

**이론적 배경**  
GRU(Cho et al. 2014)는 LSTM의 경량화 변형으로, 셀 상태(cell state)를 제거하고 **리셋 게이트(reset gate)**와 **업데이트 게이트(update gate)** 두 개만 사용한다.

$$r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$z_t = \sigma(W_z [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**장점**: LSTM 대비 파라미터 수 ~33% 적음, 학습 속도 빠름, 단기 패턴에 효과적

```
Input (B, 144, 7)
  → GRU(128, return_seq=True, dropout=0.15)
  → LayerNormalization
  → GRU(64, return_seq=False, dropout=0.15)
  → Dense(32, relu)
  → Dense(1)
Output (B, 1)
```

**하이퍼파라미터 (최적화 적용값)**

| 파라미터 | 값 |
|---|---|
| GRU 유닛 | [128, 64] |
| Dropout | 0.15 |
| 활성화 | tanh (GRU 내부), relu (FC) |
| LayerNorm | 중간 층 사이 적용 |

---

### 3-3. Transformer (시계열 Encoder)

**이론적 배경**  
Transformer(Vaswani et al. 2017)의 인코더 블록을 시계열 예측에 적용한다.  
**Multi-Head Self-Attention**으로 전체 시퀀스 내 임의 거리의 의존성을 O(1)에 포착한다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Sinusoidal Positional Encoding**으로 시간적 위치 정보를 주입:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```
Input (B, 144, 7)
  → Dense(d_model=64) + Positional Encoding
  → Dropout(0.1)
  → [TransformerEncoderBlock × 2]
       MultiHeadAttention(heads=4) → LayerNorm
       FFN(Dense(128)→Dense(64)) → LayerNorm
  → GlobalAveragePooling1D
  → Dense(32, relu)
  → Dense(1)
Output (B, 1)
```

**하이퍼파라미터 (최적화 적용값)**

| 파라미터 | 값 |
|---|---|
| d_model | 64 |
| num_heads | 4 |
| ff_dim | 128 |
| num_layers | 2 |
| Dropout | 0.1 |
| 위치 인코딩 | Sinusoidal (고정) |

---

### 3-4. TCN (Temporal Convolutional Network)

**이론적 배경**  
TCN(Bai et al. 2018)은 **팽창 인과 합성곱(Dilated Causal Convolution)**과 **잔차 연결(Residual Connection)**을 사용해 장기 의존성을 포착하는 완전 합성곱 아키텍처이다.

**수용 영역(Receptive Field)**:
$$RF = 1 + 2(k-1) \sum_{d \in D} d$$

dilations = [1, 2, 4, 8, 16], k=3일 때: RF = 1 + 2×2×31 = **125 스텝** ≈ 20.8시간

```
Input (B, 144, 7)
  → [TCN Block × 5] (dilations: 1,2,4,8,16)
       Conv1D(64, k=3, causal, dilation=d) × 2
       LayerNorm → Activation → Dropout(0.1)
       Residual Projection (필요 시)
  → GlobalAveragePooling1D
  → Dense(32, relu)
  → Dense(1)
Output (B, 1)
```

**하이퍼파라미터 (최적화 적용값)**

| 파라미터 | 값 |
|---|---|
| nb_filters | 64 |
| kernel_size | 3 |
| dilations | [1, 2, 4, 8, 16] |
| Dropout | 0.1 |
| 활성화 | relu |
| 수용 영역 | 125 스텝 (≈20.8시간) |

---

### 3-5. RetNet (Retentive Network)

**이론적 배경**  
RetNet(Sun et al. 2023)은 Transformer의 Self-Attention을 **Multi-Scale Retention**으로 대체하여 학습 시 병렬 모드, 추론 시 O(1) 순환 모드를 지원한다.

**감쇠 마스크(Decay Mask)**: 헤드 h의 감쇠 계수 $\gamma_h = 1 - 2^{-(5+h)}$

$$D[m, n] = \gamma^{m-n} \cdot \mathbb{1}[m \geq n]$$

**Retention 출력**:
$$\text{Ret}_h(X) = \left(Q_h K_h^T \odot D\right) V_h / \sqrt{d_h}$$
$$\text{MSR}(X) = W_O\left(\text{LayerNorm}\left(\text{concat}(\text{Ret}_h)\right) \odot \text{swish}(W_G X)\right)$$

**장점**: Transformer 대비 O(T) 추론 복잡도, 다중 스케일 시간 감쇠로 단기·장기 패턴 동시 포착

```
Input (B, 144, 7)
  → Dense(d_model=64)
  → Dropout(0.1)
  → [RetNet Block × 2]
       MultiScaleRetention → Dropout → LayerNorm + Residual
       FFN(Dense(128,gelu)→Dense(64)) → Dropout → LayerNorm + Residual
  → GlobalAveragePooling1D
  → Dense(32, gelu)
  → Dense(1)
Output (B, 1)
```

**하이퍼파라미터 (최적화 적용값)**

| 파라미터 | 값 |
|---|---|
| d_model | 64 |
| num_heads | 4 |
| ff_dim | 128 |
| num_layers | 2 |
| Dropout | 0.1 |
| 활성화 | gelu (FFN), swish (게이트) |
| 감쇠 계수 γ_h | 1-2^(-5-h), h=0..3 → [0.9688, 0.9844, 0.9922, 0.9961] |

---

## 4. 피처 공학

### ML 모델용 — 태뷸러 피처 (23개)

```
에너지(target) → lag 피처 + 롤링 통계 + 시간 인코딩
```

| 그룹 | 변수명 | 설명 |
|---|---|---|
| 단기 래그 | `lag_1` ~ `lag_6` | 10~60분 전 소비량 |
| 중기 래그 | `lag_12`, `lag_18`, `lag_24`, `lag_36`, `lag_48` | 2H~8H 전 소비량 |
| 일간 래그 | `lag_144`, `lag_288` | 24H, 48H 전 (전날 동시간대) |
| 롤링 평균 | `rolling_mean_12`, `rolling_mean_144` | 2H, 24H 이동 평균 |
| 롤링 표준편차 | `rolling_std_12`, `rolling_std_144` | 2H, 24H 이동 표준편차 |
| 시간 인코딩 | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | 주기성 보존 인코딩 |

> **래그 선택 근거**: 전기 소비량의 강한 일간 주기성(lag_144)과 단기 자기상관(lag_1~6)을 동시에 반영한다.

### DL 모델용 — 시퀀스 피처 (7개 × 144 스텝)

| 피처 | 설명 |
|---|---|
| `energy_scaled` | MinMaxScaler(훈련 구간 fit) 정규화 에너지 |
| `hour_sin/cos` | 시간 주기 인코딩 |
| `dow_sin/cos` | 요일 주기 인코딩 |
| `month_sin/cos` | 월 주기 인코딩 |

> **Cyclical 인코딩**: `sin(2π × value / period)` + `cos(2π × value / period)` → 24→0 경계에서의 불연속성 방지

---

## 5. 하이퍼파라미터 최적화 전략

### ML 모델: RandomizedSearchCV + TimeSeriesSplit

```python
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

tscv   = TimeSeriesSplit(n_splits=3)
search = RandomizedSearchCV(
    estimator   = base_model,
    param_distributions = param_space,
    n_iter      = 30,           # 30가지 랜덤 조합 탐색
    scoring     = 'neg_root_mean_squared_error',
    cv          = tscv,         # 시계열 순서 유지 CV
    refit       = True,         # 최적 파라미터로 전체 훈련 데이터 재학습
    n_jobs      = -1,
)
```

> **TimeSeriesSplit 사용 이유**: 랜덤 KFold는 미래 데이터가 훈련 폴드에 포함되어 leakage 발생.  
> TimeSeriesSplit은 항상 이전 시점 데이터로 학습하고 이후 시점으로 검증한다.

### DL 모델: 사전 정의 하이퍼파라미터 + EarlyStopping

DL 모델은 탐색 공간이 넓고 단일 훈련 비용이 크기 때문에, **도메인 지식 기반의 최적 초기값**을 사용하고 EarlyStopping + ReduceLROnPlateau로 세밀 조정한다.

| 항목 | 값 | 근거 |
|---|---|---|
| seq_len = 144 | 24H 룩백 | 일간 주기성 완전 포착 |
| Huber Loss | δ=1.0 | kWh 이상치(정전 후 급등) 강건성 |
| Adamax → Adam | lr=1e-3 | 희소 그래디언트에 강건한 Adam 계열 |
| EarlyStopping patience=20 | — | 조기 종료로 과적합 방지 |
| ReduceLROnPlateau ×0.5 | patience=10 | 학습률 감쇠로 수렴 가속 |

---

## 6. 평가 지표

모든 지표는 MinMaxScaler 역변환 후 **실제 단위(kWh)**로 계산한다.

| 지표 | 수식 | 해석 |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 예측 오차 절댓값 평균 (단위: kWh) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | 큰 오차에 민감 (단위: kWh) |
| **MAPE** | $\frac{100}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|+\epsilon}$ | 상대 오차 (단위: %) |
| **R²** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 분산 설명력 (1=완벽, 0=평균예측) |

> MAPE에서 $\epsilon = 10^{-8}$를 추가하여 소비량이 0에 가까울 때의 수치 불안정성을 방지한다.

### 종합 순위 산정

각 지표의 순위를 평균하여 모델 종합 순위를 산출한다:

```python
avg_rank = mean([rank_MAE, rank_RMSE, rank_MAPE, rank_R2_desc])
```

- MAE, RMSE, MAPE: 낮을수록 좋음 → 오름차순 순위
- R²: 높을수록 좋음 → 내림차순 순위

---

## 결과 파일 위치

```
results/models/
├── dl_comparison_detail.csv      # DL 세대별 상세 결과
├── dl_comparison_summary.csv     # DL 모델별 집계 (평균±표준편차)
├── dl_comparison_rank.csv        # DL 종합 순위
├── dl_comparison_bar.png         # DL 지표별 Bar Chart
├── dl_comparison_loss_curves.png # DL 학습 곡선
├── dl_comparison_prediction.png  # DL 72H 예측 비교
├── dl_comparison_radar.png       # DL Radar Chart
├── ml_comparison_detail.csv      # ML 세대별 상세 결과
├── ml_comparison_summary.csv     # ML 모델별 집계
├── ml_comparison_rank.csv        # ML 종합 순위
├── ml_best_params.json           # ML HPO 대표 Best Params (중앙값)
├── ml_comparison_bar.png         # ML Bar Chart
├── ml_comparison_boxplot.png     # ML 세대별 분포 Boxplot
├── ml_comparison_prediction.png  # ML 72H 예측 비교
└── ml_comparison_radar.png       # ML Radar Chart
```

## 참고 문헌

1. Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
2. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
5. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder. *EMNLP*.
6. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
7. Bai, S. et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
8. Sun, Y. et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models. *arXiv:2307.08621*.
