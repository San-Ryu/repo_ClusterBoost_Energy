# 세대별 군집 레이블 현황

**실행일시**: 2026-05-19 07:18:10  
**소요시간**: 90.1초  
**데이터**: `data/KIER_3_Temporal_Resolution/seasonal_naive/`  
**알고리즘**: K-Means · Mean-Shift · DBSCAN · L1L2(K-Medoids) · GMM  
**세대 수**: 348  

---

## 1. 알고리즘별 Silhouette Score 요약

| 해상도 | K-Means | Mean-Shift | DBSCAN | L1L2 | GMM |
|:---|:---:|:---:|:---:|:---:|:---:|
| 10MIN | 0.3355 | 0.3832 | 0.2422 | 0.3123 | 0.1587 |
| 1H | 0.3414 | 0.3892 | 0.3722 | 0.2984 | 0.1982 |
| 1D | 0.3325 | — | 0.4435 | 0.2746 | 0.2563 |
| 1W | 0.3475 | 0.2974 | 0.3768 | 0.2941 | 0.2003 |
| 1M | 0.3623 | 0.1610 | 0.2984 | 0.3191 | 0.1859 |

---

## 2. 10MIN — 알고리즘별 클러스터 분포

### 10MIN / K-Means
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3355

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 219 | ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2 ... (219세대) |
| K1 | 129 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-3-1, ELEC_1-4-2, ELEC_1-4-3, ELEC_1-5-1, ELEC_1-5-3 ... (129세대) |

### 10MIN / Mean-Shift
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3832

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 318 | ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-2, ELEC_1-4-3 ... (318세대) |
| K1 | 30 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-5-1, ELEC_1-7-1, ELEC_1-9-4, ELEC_1-10-1, ELEC_1-13-1 ... (30세대) |

### 10MIN / DBSCAN
> 군집 수: 2  |  노이즈: 227세대  |  Silhouette: 0.2422

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| 노이즈(-1) | 227 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3 ... (227세대) |
| K0 | 117 | ELEC_1-2-2, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-8-3, ELEC_1-9-3 ... (117세대) |
| K1 | 4 | ELEC_2-2-5, ELEC_2-15-4, ELEC_2-23-4, ELEC_3-11-5 |

### 10MIN / L1L2 (K-Medoids)
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3123

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 199 | ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3 ... (199세대) |
| K1 | 149 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3, ELEC_1-5-1 ... (149세대) |

### 10MIN / GMM
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.1587

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 224 | ELEC_1-2-2, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-5-2, ELEC_1-6-1, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-7-2 ... (224세대) |
| K1 | 124 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3 ... (124세대) |

## 2. 1H — 알고리즘별 클러스터 분포

### 1H / K-Means
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3414

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 218 | ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2 ... (218세대) |
| K1 | 130 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-3-1, ELEC_1-4-2, ELEC_1-4-3, ELEC_1-5-1, ELEC_1-5-3 ... (130세대) |

### 1H / Mean-Shift
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3892

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 319 | ELEC_1-1-3, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-2 ... (319세대) |
| K1 | 29 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-4-3, ELEC_1-5-1, ELEC_1-6-4, ELEC_1-7-1, ELEC_1-8-1, ELEC_1-11-4 ... (29세대) |

### 1H / DBSCAN
> 군집 수: 2  |  노이즈: 287세대  |  Silhouette: 0.3722

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| 노이즈(-1) | 287 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2 ... (287세대) |
| K0 | 54 | ELEC_1-4-1, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-8-3, ELEC_1-16-2, ELEC_1-16-3, ELEC_1-16-4, ELEC_1-17-2, ELEC_1-17-3 ... (54세대) |
| K1 | 7 | ELEC_1-9-3, ELEC_2-6-4, ELEC_2-12-4, ELEC_2-18-6, ELEC_2-22-1, ELEC_3-1-5, ELEC_3-5-3 |

### 1H / L1L2 (K-Medoids)
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.2984

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 168 | ELEC_1-2-2, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-8-2, ELEC_1-8-3 ... (168세대) |
| K1 | 180 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2 ... (180세대) |

### 1H / GMM
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.1982

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 235 | ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-1, ELEC_1-6-2 ... (235세대) |
| K1 | 113 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3, ELEC_1-5-1 ... (113세대) |

## 2. 1D — 알고리즘별 클러스터 분포

### 1D / K-Means
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3325

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 142 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3 ... (142세대) |
| K1 | 206 | ELEC_1-2-2, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-7-3 ... (206세대) |

### 1D / Mean-Shift
> 군집 수: 1  |  노이즈: 0세대  |  Silhouette: —

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 348 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2 ... (348세대) |

### 1D / DBSCAN
> 군집 수: 2  |  노이즈: 307세대  |  Silhouette: 0.4435

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| 노이즈(-1) | 307 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3 ... (307세대) |
| K0 | 7 | ELEC_1-8-2, ELEC_1-15-4, ELEC_2-7-2, ELEC_2-13-5, ELEC_3-12-1, ELEC_3-12-5, ELEC_3-16-4 |
| K1 | 34 | ELEC_1-2-2, ELEC_1-4-1, ELEC_1-12-2, ELEC_1-17-3, ELEC_2-2-5, ELEC_2-3-6, ELEC_2-4-6, ELEC_2-5-3, ELEC_2-8-2, ELEC_2-8-3 ... (34세대) |

### 1D / L1L2 (K-Medoids)
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.2746

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 132 | ELEC_1-2-2, ELEC_1-3-2, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-8-2, ELEC_1-8-3, ELEC_1-8-4 ... (132세대) |
| K1 | 216 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-3, ELEC_1-3-4 ... (216세대) |

### 1D / GMM
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.2563

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 72 | ELEC_1-1-1, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-4-3, ELEC_1-5-1, ELEC_1-5-4, ELEC_1-7-4, ELEC_1-8-1, ELEC_1-10-1 ... (72세대) |
| K1 | 276 | ELEC_1-1-2, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-2 ... (276세대) |

## 2. 1W — 알고리즘별 클러스터 분포

### 1W / K-Means
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3475

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 203 | ELEC_1-2-2, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-7-3 ... (203세대) |
| K1 | 145 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3 ... (145세대) |

### 1W / Mean-Shift
> 군집 수: 4  |  노이즈: 0세대  |  Silhouette: 0.2974

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 316 | ELEC_1-1-3, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-2 ... (316세대) |
| K1 | 4 | ELEC_1-17-2, ELEC_3-7-2, ELEC_3-15-4, ELEC_3-19-1 |
| K2 | 14 | ELEC_1-2-1, ELEC_1-5-1, ELEC_1-8-1, ELEC_1-13-1, ELEC_2-3-2, ELEC_2-7-4, ELEC_2-10-6, ELEC_2-15-3, ELEC_3-7-4, ELEC_3-8-2 ... (14세대) |
| K3 | 14 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-4, ELEC_1-9-4, ELEC_2-1-2, ELEC_2-9-4, ELEC_3-3-6, ELEC_3-4-6, ELEC_3-6-6, ELEC_3-8-3 ... (14세대) |

### 1W / DBSCAN
> 군집 수: 2  |  노이즈: 302세대  |  Silhouette: 0.3768

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| 노이즈(-1) | 302 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3 ... (302세대) |
| K0 | 36 | ELEC_1-2-2, ELEC_1-4-1, ELEC_1-12-2, ELEC_1-17-3, ELEC_1-18-4, ELEC_2-2-5, ELEC_2-3-6, ELEC_2-4-6, ELEC_2-5-3, ELEC_2-5-5 ... (36세대) |
| K1 | 10 | ELEC_1-8-2, ELEC_1-12-3, ELEC_1-15-4, ELEC_2-7-2, ELEC_2-13-5, ELEC_2-15-1, ELEC_3-5-5, ELEC_3-12-1, ELEC_3-12-5, ELEC_3-15-1 |

### 1W / L1L2 (K-Medoids)
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.2941

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 133 | ELEC_1-2-2, ELEC_1-3-2, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-8-2, ELEC_1-8-3, ELEC_1-8-4 ... (133세대) |
| K1 | 215 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-3, ELEC_1-3-4 ... (215세대) |

### 1W / GMM
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.2003

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 263 | ELEC_1-1-2, ELEC_1-1-3, ELEC_1-2-2, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-2 ... (263세대) |
| K1 | 85 | ELEC_1-1-1, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-4-3, ELEC_1-5-1, ELEC_1-5-4, ELEC_1-7-4, ELEC_1-8-1, ELEC_1-9-1 ... (85세대) |

## 2. 1M — 알고리즘별 클러스터 분포

### 1M / K-Means
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3623

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 207 | ELEC_1-2-2, ELEC_1-2-4, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-7-3 ... (207세대) |
| K1 | 141 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2, ELEC_1-4-3 ... (141세대) |

### 1M / Mean-Shift
> 군집 수: 3  |  노이즈: 0세대  |  Silhouette: 0.1610

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 341 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2 ... (341세대) |
| K1 | 2 | ELEC_3-1-3, ELEC_3-21-3 |
| K2 | 5 | ELEC_1-17-2, ELEC_2-7-3, ELEC_3-7-2, ELEC_3-15-4, ELEC_3-19-1 |

### 1M / DBSCAN
> 군집 수: 2  |  노이즈: 285세대  |  Silhouette: 0.2984

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| 노이즈(-1) | 285 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-2, ELEC_1-3-3 ... (285세대) |
| K0 | 58 | ELEC_1-2-2, ELEC_1-4-1, ELEC_1-8-2, ELEC_1-12-2, ELEC_1-12-3, ELEC_1-13-3, ELEC_1-15-4, ELEC_1-17-3, ELEC_1-18-4, ELEC_2-1-5 ... (58세대) |
| K1 | 5 | ELEC_1-7-3, ELEC_3-1-1, ELEC_3-9-4, ELEC_3-14-5, ELEC_3-18-2 |

### 1M / L1L2 (K-Medoids)
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.3191

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 154 | ELEC_1-2-2, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-4-1, ELEC_1-4-4, ELEC_1-5-2, ELEC_1-6-2, ELEC_1-6-3, ELEC_1-8-2, ELEC_1-8-3 ... (154세대) |
| K1 | 194 | ELEC_1-1-1, ELEC_1-1-2, ELEC_1-1-3, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-3, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-3-4, ELEC_1-4-2 ... (194세대) |

### 1M / GMM
> 군집 수: 2  |  노이즈: 0세대  |  Silhouette: 0.1859

| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |
|:---:|:---:|:---|
| K0 | 265 | ELEC_1-1-2, ELEC_1-1-3, ELEC_1-2-2, ELEC_1-2-3, ELEC_1-3-2, ELEC_1-3-3, ELEC_1-3-4, ELEC_1-4-1, ELEC_1-4-2, ELEC_1-4-3 ... (265세대) |
| K1 | 83 | ELEC_1-1-1, ELEC_1-1-4, ELEC_1-2-1, ELEC_1-2-4, ELEC_1-3-1, ELEC_1-5-1, ELEC_1-8-1, ELEC_1-10-4, ELEC_1-11-2, ELEC_1-13-1 ... (83세대) |

---

## 3. 세대별 레이블 전체 목록

> 세대 수가 많아 아래 테이블은 최대 50세대만 표시합니다.
> 전체는 `data/KIER_4_Clustered/` 경로의 CSV 분리 파일을 참조하세요.

### 10MIN 세대별 레이블

| 세대 | K-Means | Mean-Shift | DBSCAN | L1L2 (K-Medoids) | GMM |
|:---|:---:|:---:|:---:|:---:|:---:|
| ELEC_1-1-1 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-1-2 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-1-3 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-1-4 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-2-1 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-2-2 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-2-3 | K0 | K0 | noise | K0 | K1 |
| ELEC_1-2-4 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-3-1 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-3-2 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-3-3 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-3-4 | K0 | K0 | noise | K1 | K1 |
| ELEC_1-4-1 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-4-2 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-4-3 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-4-4 | K0 | K0 | noise | K0 | K1 |
| ELEC_1-5-1 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-5-2 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-5-3 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-5-4 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-6-1 | K1 | K0 | noise | K1 | K0 |
| ELEC_1-6-2 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-6-3 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-6-4 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-7-1 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-7-2 | K1 | K0 | noise | K1 | K0 |
| ELEC_1-7-3 | K0 | K0 | noise | K0 | K1 |
| ELEC_1-7-4 | K1 | K0 | noise | K1 | K0 |
| ELEC_1-8-1 | K1 | K0 | noise | K1 | K0 |
| ELEC_1-8-2 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-8-3 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-8-4 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-9-1 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-9-2 | K0 | K0 | noise | K1 | K1 |
| ELEC_1-9-3 | K1 | K0 | K0 | K1 | K0 |
| ELEC_1-9-4 | K1 | K1 | noise | K1 | K0 |
| ELEC_1-10-1 | K1 | K1 | noise | K1 | K0 |
| ELEC_1-10-2 | K0 | K0 | K0 | K0 | K0 |
| ELEC_1-10-3 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-10-4 | K0 | K0 | noise | K0 | K1 |
| ELEC_1-11-1 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-11-2 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-11-3 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-11-4 | K1 | K0 | noise | K1 | K1 |
| ELEC_1-12-1 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-12-2 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-12-3 | K0 | K0 | noise | K0 | K1 |
| ELEC_1-12-4 | K0 | K0 | noise | K0 | K0 |
| ELEC_1-13-1 | K1 | K1 | noise | K1 | K1 |
| ELEC_1-13-2 | K0 | K0 | noise | K0 | K0 |
| ... (348세대 중 50세대 표시) | ... | ... | ... | ... | ... |

---
*생성: 2026-05-19 07:18:10 by `core/data_09_cluster_label.py`*