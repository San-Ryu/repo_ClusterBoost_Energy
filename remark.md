# KIER 데이터 파이프라인 방법론 검토

> 작성 목적: 논문 방법론 섹션 보조 자료 / 파이프라인 설계 의사결정 근거 기록  
> 작성일: 2026-04-23

---

## 1. 현황 요약

### 처리된 데이터 규모

| 구분 | 내용 |
|------|------|
| 원시 데이터 | 5개 도메인 CSV (ELEC·WATER·HEAT·HOT·GAS), 각 27~52M 행 |
| 공통 세대 | 3개 동 × 24개 층 × 최대 5호 = **348 세대** |
| 시간 해상도 | 10분 간격 (2022-07 ~ 2024-06 기준) |
| 현재 저장 방식 | Wide-format CSV (METER_DATE × 세대 컬럼) |
| 파일 수 / 용량 | 도메인 CSV 12개(3.1 GB) + 세대별 분리 2,088개(5.0 GB) = **≈ 8 GB** |

### 파이프라인 단계

```
[Stage 0] 원시 CSV (Raw)
    ↓
[Stage 1] ACCU / INST 변환  ← 현재 위치
    ↓
[Stage 2] 결측 보간 (방법론 미정)
    ↓
[Stage 3] 클러스터링 + 군집 레이블 생성
    ↓
[Stage 4] 5~6개 모델 학습 및 성능 평가
```

---

## 2. CSV 방식 평가 (시니어 엔지니어 관점)

### 2-1. 장점

- **이식성(Portability)**: 별도 라이브러리 없이 어디서나 열람 가능.  
- **투명성(Transparency)**: 텍스트 기반으로 스키마 확인이 즉각적.  
- **범용 협업**: 비개발자(연구자, 논문 리뷰어)와 데이터 공유 시 진입 장벽 없음.

### 2-2. 문제점

#### (a) I/O 비효율
- 단일 열(1세대)만 필요할 때도 파일 전체(270 MB+)를 읽어야 함.  
- Wide-format 12개 × 반복 학습 = 매 실험마다 3 GB 이상 직렬 읽기.  
- 세대별 분리 파일 2,088개는 파일시스템 메타데이터 오버헤드가 크고,  
  병렬 I/O 시 OS 파일 디스크립터 한도에 근접할 수 있음.

#### (b) 타입 손실
- `float32`로 생성한 값이 텍스트로 저장되고 재로딩 시 `float64`로 역변환됨  
  → 메모리 사용량 2배 증가, 정밀도 표현 불일치.
- `METER_DATE`가 문자열로 저장되어 매번 `pd.to_datetime()` 파싱 필요.

#### (c) 메타데이터 부재
- 어떤 보간 방법을 적용했는지, 어떤 파라미터를 썼는지 파일 내에 기록 불가.  
- 스테이지별 버전 관리 체계가 없어 실험 재현 시 설정을 별도 기억해야 함.

#### (d) 압축 부재
- CSV는 기본 비압축 → INST 파일의 경우 NaN이 문자열 `nan`으로 저장되어  
  ACCU보다 오히려 2배 이상 큰 파일이 생성됨 (ELEC: ACCU 247 MB → INST 604 MB).

---

## 3. 권장 개선 방향

### 3-1. Parquet 전환 (1순위 권장)

| 항목 | CSV | Parquet (snappy) |
|------|-----|-----------------|
| 파일 크기 | 247 MB (ELEC ACCU) | ≈ 40~60 MB |
| 타입 보존 | ✗ (모두 string→재파싱) | ✓ (float32·datetime64 그대로) |
| 열 선택 읽기 | ✗ (파일 전체 스캔) | ✓ (필요한 세대만 로딩) |
| pandas 로딩 | `pd.read_csv()` | `pd.read_parquet(columns=[...])` |
| 호환성 | 완전 범용 | Python·R·Spark·DuckDB 모두 지원 |

```python
# 기존
df = pd.read_csv("KIER_USAGE_ELEC_ACCU.csv")           # 270 MB 전체 읽기

# Parquet 전환 후 (단일 세대 필요 시)
df = pd.read_parquet("KIER_USAGE_ELEC_ACCU.parquet",
                     columns=["METER_DATE", "ELEC_1-1-1"])  # 해당 열만 읽기
```

**세대별 분리 파일(2,088개)은 Parquet 전환 시 불필요** — Wide-format Parquet에서  
열 선택 로딩이 가능하므로 KIER_1_DIVIDED_BY_DOMAIN_HOUSE 디렉터리 자체가 제거 가능.

### 3-2. HDF5 / Zarr (2순위, 대규모 실험 필요 시)

- `HDF5(h5py·PyTables)`: 계층 구조로 스테이지별 데이터를 단일 파일 관리 가능.  
  ex) `/stage1/elec_accu`, `/stage2/linear/elec`, `/stage3/clusters`  
- `Zarr`: 청크 단위 압축, 클라우드 스토리지(S3) 연동에 적합.  
- **단점**: 파일 손상 시 전체 데이터 손실 위험 있음 → 스테이지별 별도 파일 권장.

### 3-3. 체크포인트 네이밍 전략

논문 재현성(Reproducibility) 확보를 위해 스테이지 + 파라미터를 파일명에 인코딩:

```
~/data/data_Energy_KIER/
├── KIER_HINDEX.csv                              # 세대 인덱스 (경량, git 관리 가능)
│
├── KIER_S1_ACCU/                               # Stage 1: 적산
│   └── KIER_USAGE_{DOMAIN}_ACCU.parquet
│
├── KIER_S1_INST/                               # Stage 1: 순시
│   └── KIER_USAGE_{DOMAIN}_INST.parquet
│
├── KIER_S2_{METHOD}/                           # Stage 2: 보간 (방법론별 분기)
│   ├── linear/   KIER_USAGE_{DOMAIN}_INST_INTERP.parquet
│   ├── spline/   ...
│   └── knn/      ...
│
├── KIER_S3_CLUSTERED/                          # Stage 3: 클러스터 레이블
│   └── KIER_CLUSTER_{METHOD}_{K}.parquet
│
└── KIER_S4_FEATURES/                           # Stage 4: 모델 피처
    ├── {model}/  features.parquet
    └── {model}/  predictions.parquet
```

### 3-4. 메타데이터 병기 (파이프라인 재현성)

각 스테이지 완료 시 파라미터를 동반 저장:

```python
# Parquet 파일에 커스텀 메타데이터 삽입 예시
import pyarrow as pa, pyarrow.parquet as pq

meta = {
    b"stage":   b"S2_interpolation",
    b"method":  b"linear",
    b"window":  b"10T",
    b"created": b"2026-04-23",
}
table = pa.Table.from_pandas(df)
table = table.replace_schema_metadata({**table.schema.metadata, **meta})
pq.write_table(table, "KIER_USAGE_ELEC_INST_INTERP.parquet")
```

---

## 4. 현재 파이프라인 설계 종합 평가

| 기준 | 현재(CSV) | 권장(Parquet) |
|------|-----------|--------------|
| 저장 공간 | 8 GB (CSV 12 + 분리 2088개) | ≈ 1.2 GB (도메인 Parquet 12개) |
| 1개 도메인 로딩 | 270 MB 전체 | 필요 열만 (≈ 1 MB/세대) |
| 반복 실험 I/O 부담 | 높음 (매번 전체 스캔) | 낮음 (열 프루닝) |
| 스테이지 버전 관리 | 없음 | 파일명 + 메타데이터 |
| 논문 재현성 | 낮음 (파라미터 별도 기억) | 높음 (파일 자체에 기록 가능) |
| 즉시 전환 비용 | — | `df.to_parquet()` 1줄 |

### 결론

CSV 방식은 **탐색적 초기 단계**에서는 적절하나, 보간 → 클러스터링 → 다중 모델 평가로  
이어지는 파이프라인에서는 Parquet 전환이 실질적인 개발 생산성을 높인다.  
특히 논문 기고 시 재현성 문서화를 위해 **스테이지별 파라미터 메타데이터 병기**가  
추가적인 가치를 제공한다.

> **2026-05-02 core/ 파일명 정비**: `build_*` → `data_NN_*` 로 통일
> `data_01_pipeline`, `data_02_build_dataset`, `data_03_preprocessing`,
> `data_04_interp`, `data_05_resample`, `data_06_clustering`

---

## 5. 향후 파이프라인 의사결정 참고사항

- **보간 방법 선정 기준**: HEAT 도메인은 비가열기 NaN 비율이 ELEC 대비 높음.  
  단순 선형 보간은 계절성 왜곡 위험 → Forward-fill 또는 시계열 기반 보간 검토 필요.
- **클러스터링 입력**: INST 기반 사용 패턴이 ACCU 기반보다 세대 간 유사도 측정에 적합.
- **모델 성능 평가 체계**: 5~6개 모델을 동일 데이터 스플릿으로 비교하기 위해  
  Stage 3 이후 데이터는 단일 seed로 고정된 train/val/test 인덱스를 Parquet 메타데이터로 관리 권장.

---

*이 문서는 `core/build_kier_dataset.py` 파이프라인의 설계 결정 근거를 기록한 것으로,*  
*논문의 Data Preprocessing 및 Experimental Setup 섹션 작성 시 참조.*
