# KIER ELEC 결측치 보간법 비교 — 실험 결과 문서

> 작성일: 2026-05-02  
> 적용 도메인: ELEC (전기)  
> 데이터: `KIER_USAGE_ELEC_INST.csv` — 10분 간격 순시 사용량, 348세대, 97,438행  
> 기간: 2022-07-17 23:00 ~ 2024-06-05 15:30

---

## 1. 개요

전기 순시 사용량(INST) 데이터의 결측치에 대해 5가지 보간법을 적용하고  
처리 시간·결측 충전률·파일 크기를 실측하였으며, 정성적 적합도를 분석하였다.

---

## 2. 비교 대상 (5안)

| 안 | 방법 | 핵심 원리 | 전기 도메인 적합 이유 |
|----|------|----------|--------------------|
| 기존안 | 행 평균 대치 + 선형 보간 | 전 세대 시각별 평균 → 선형 | Baseline. 세대 개별 패턴 소실, 피크 희석 |
| 1안 | **PCHIP** | Piecewise Cubic Hermite Interpolation | 단조성·비음수 보존, 오버슈트 없음, 단기 결측 안정 |
| 2안 | **STL 재합성** | `STL(period=144)` trend+seasonal | 24H 일주기 구조 보존 |
| 3안 | **MSTL 재합성** | `MSTL(periods=[144, 1008])` 이중 주기 | 24H + 7D(평일/주말) 이중 주기 동시 포착 |
| 4안 | **Seasonal Naive** | 24H lag → 7D lag fallback → 선형 | 전기 주기 직접 활용, 빠름, 해석 용이 |

---

## 3. Phase 1 실측 결과 — 전체 데이터 보간 및 저장

### 3-1. 처리 시간 · 결측 충전률

| 방법 | 처리 시간 | 결측 (전) | 결측 (후) | 충전률 | 저장 형식 |
|------|-----------|-----------|-----------|--------|----------|
| 기존안 (baseline) | **0.5 초** | 2,094,874 | 0 | **100%** | Parquet |
| 1안 (PCHIP) | **0.9 초** | 2,094,874 | 0 | **100%** | Parquet |
| 4안 (Seasonal Naive) | **0.8 초** | 2,094,874 | 0 | **100%** | Parquet |
| 2안 (STL) | 7,293 초 (≈ 2.0 시간) | 2,094,874 | 0 | **100%** | Parquet |
| 3안 (MSTL) | 17,046 초 (≈ 4.7 시간) | 2,094,874 | 0 | **100%** | Parquet |

> 전체 결측 수: 2,094,874 / (97,438 × 348) = **약 6.2% 결측률**

### 3-2. 출력 Parquet 파일 크기

| 방법 | 파일 크기 | 비고 |
|------|-----------|------|
| baseline | 61 MB | — |
| pchip | 45 MB | — |
| stl | 61 MB | — |
| mstl | 61 MB | — |
| seasonal_naive | **36 MB** | 최소 — 이산값(전날 동일값 복사) 특성상 압축률 우수 |

> 원시 CSV(`KIER_USAGE_ELEC_INST.csv`) 대비 Parquet으로 약 **90% 용량 절감**  
> (원본 추정 ~604 MB → 평균 ~53 MB/방법)

### 3-3. 처리 시간 분석

```
Seasonal Naive  ████░░░░░░░░░░░░░░░░░░░░░  0.8초   (정렬/이동 연산만)
PCHIP           ████░░░░░░░░░░░░░░░░░░░░░  0.9초   (보간 다항식 구성)
baseline        ████░░░░░░░░░░░░░░░░░░░░░  0.5초   (행 평균 + 선형)
STL             ████████████████████████░  7,293초  (세대별 반복 분해 × 348)
MSTL            █████████████████████████  17,046초 (이중 주기 분해 × 348)
```

STL 대비 MSTL이 **2.3배** 느림 → 실시간/반복 파이프라인에서 비용 고려 필요.

---

## 4. Phase 2 정량 평가 계획

Phase 2는 완전 세대(complete cases)에 인공 마스킹을 주입하여 보간법을 정량 비교한다.  
*(Phase 2 실행 후 이 섹션에 결과를 추가할 것)*

### 4-1. 평가 방법론

```
완전 세대 선별 (NaN = 0인 세대) → 샘플 N개 선택
    ↓
N_TRIALS × 3종 마스킹 주입:
    ① Random 10%   — 통신 오류 등 산발적 결측
    ② Block 6H     — 단기 단전 (연속 36 스텝)
    ③ Block 24H    — 장기 단전 (연속 144 스텝)  ← 전기 특화
    ↓
5안 보간 수행 → 원본 대비 오차 측정 (마스킹 위치만)
    ↓
6종 지표 계산 → Z-score 정규화 → 가중 합산 Composite Score
```

### 4-2. 평가 지표 (6종)

| 지표 | 의미 | 가중치 |
|------|------|--------|
| RMSE | 마스킹 구간 재현 오차 | **0.40** |
| Roughness | 2차 차분 절댓값 평균 (매끄러움) | 0.15 |
| ACF Divergence | 자기상관 구조 보존 (주기성) | 0.15 |
| Wasserstein Distance | 분포 보존 | 0.10 |
| Peak Error | 상위 5% 피크 구간 RMSE (전기 특화) | **0.10** |
| Daily Pattern Div | 1 − Pearson r (24H 평균 프로파일, 전기 특화) | **0.10** |

### 4-3. Phase 2 실행 명령

```bash
# 빠른 평가 (3안 × 10 trials × 20세대, 수 분)
python core/data_04_interp.py --phase 2 \
  --methods baseline pchip seasonal_naive \
  --n-trials 10 --sample-n 20

# STL·MSTL 포함 전체 평가 (수 시간)
python core/data_04_interp.py --phase 2 --n-trials 10 --sample-n 20
```

결과 저장 위치: `results/preprocessing/`
- `ELEC_interp_rank.csv` — 종합 순위
- `ELEC_interp_summary.csv` — method × mask_type 집계
- `ELEC_interp_bar.png` — 6종 지표 bar chart (2×3)
- `ELEC_score_heatmap.png` — Composite Score 히트맵

---

## 5. 이론적 적합도 분석 (Phase 2 선행 예측)

### 5-1. 마스크 유형별 예상 우위

| 결측 유형 | 예상 1위 | 예상 2위 | 기대 이유 |
|----------|---------|---------|----------|
| Random 10% | PCHIP | MSTL | 단기·산발적 → 이웃값 보간이 유리 |
| Block 6H | MSTL | Seasonal Naive | 반일 블록 → 주기 구조 참조 효과적 |
| **Block 24H** | **Seasonal Naive** | MSTL | 정확히 1일 주기 → 전날 동일 시각 직접 참조 |

### 5-2. 지표별 예상 열위 방법

| 지표 | 예상 최하위 | 이유 |
|------|-----------|------|
| RMSE | 기존안 | 세대 평균 대치 → 개별 오차 증폭 |
| Peak Error | 기존안 | 피크가 평균으로 희석됨 |
| Daily Pattern Div | 기존안 | 주기 구조를 전혀 활용하지 않음 |
| Roughness | Seasonal Naive | 이산 복사 특성으로 계단식 불연속 가능 |

### 5-3. 파일 크기로 유추한 특성

`seasonal_naive`의 Parquet이 **36 MB (타 방법 대비 41% 작음)**:  
동일 시각 전날 값을 직접 복사 → 반복 패턴이 강해 **Snappy 압축 효율 최고**.  
이는 보간된 값이 원본 측정값과 동일 범위에서 분포함을 시사한다.

---

## 6. 결론 및 선정 지침

### Phase 2 결과 전 잠정 권고

> **Seasonal Naive (4안)** 를 우선 선정 후 Phase 2로 검증

| 기준 | Seasonal Naive | MSTL |
|------|---------------|------|
| 처리 속도 | ✅ 0.8초 | ❌ 4.7시간 |
| 파일 크기 | ✅ 36 MB | ❌ 61 MB |
| 주기 보존 | ✅ 24H+7D 직접 활용 | ✅ 24H+7D 분리 추정 |
| 장기 결측 | ✅ 매우 강함 | ✅ 강함 |
| 단기 결측 | ⚠ PCHIP 대비 불연속 가능 | ✅ 매끄러움 |
| 논문 해석 | ✅ 직관적 설명 용이 | ⚠ 모델 복잡성 설명 필요 |

**파이프라인 적용**: `core/data_pipeline.py`의 `fill_missing_wide()`를  
`interp_seasonal_naive()` 또는 `interp_mstl()`로 교체.

---

## 7. 출력 데이터 구조

### 7-1. Stage 2 보간 결과 (Parquet)

```
~/data/KIER_S2_interp/
├── phase1_log.csv                                  ← Phase 1 실행 로그
├── baseline/       KIER_USAGE_ELEC_INST_INTERP.parquet  (61 MB)
├── pchip/          KIER_USAGE_ELEC_INST_INTERP.parquet  (45 MB)
├── stl/            KIER_USAGE_ELEC_INST_INTERP.parquet  (61 MB)
├── mstl/           KIER_USAGE_ELEC_INST_INTERP.parquet  (61 MB)
└── seasonal_naive/ KIER_USAGE_ELEC_INST_INTERP.parquet  (36 MB)
```

스키마: `METER_DATE (datetime64[ns]) | ELEC_{동}-{층}-{호} (float64) × 348`

### 7-2. 시간 해상도별 CSV (Temporal Resampling)

```
~/data/KIER_2_Temporal_Resolution/
└── {method}/
    ├── KIER_USAGE_ELEC_INST_INTERP_10MIN.csv   97,438행  ≈ 600 MB (방법별 상이)
    ├── KIER_USAGE_ELEC_INST_INTERP_1H.csv      16,529행  ≈ 103 MB
    ├── KIER_USAGE_ELEC_INST_INTERP_1D.csv         690행  ≈   4 MB
    ├── KIER_USAGE_ELEC_INST_INTERP_1W.csv         100행  ≈ 0.6 MB
    └── KIER_USAGE_ELEC_INST_INTERP_1M.csv          24행  ≈ 0.2 MB
```

집계 방법: **SUM** (`min_count=1`) — 순시 사용량 합산 = 해당 기간 총 에너지 소비량  
기간: 2022-07-17 23:00 ~ 2024-06-05 15:30 (약 23.6개월)  
생성 스크립트: `python core/build_kier_temporal_resample.py`

```bash
# 전체 5안 × 10MIN 제외 4해상도 생성 (권장, 빠름)
python core/build_kier_temporal_resample.py --skip-10min

# 10MIN 포함 전체 생성 (방법당 ≈600MB CSV, 총 ≈3GB)
python core/build_kier_temporal_resample.py

# 특정 방법/해상도만
python core/build_kier_temporal_resample.py --methods seasonal_naive mstl --resolutions 1H 1D
```

---

## 8. 참조 파일

| 파일 | 설명 |
|------|------|
| `core/data_04_interp.py` | 5안 보간 적용 + Parquet 저장 + 정량 평가 |
| `core/data_05_resample.py` | 시간 해상도별 CSV 재생성 |
| `src/data_02_interp_eval.ipynb` | 탐색적 보간법 비교 실험 (샘플 20세대) |
| `.cursor/methodology_interp.md` | 방법론 설계 근거 상세 |
| `results/preprocessing/` | Phase 2 정량 평가 결과 (실행 후 생성) |
