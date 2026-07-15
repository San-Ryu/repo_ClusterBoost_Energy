"""
data_01_pipeline.py
시계열 에너지 데이터 파이프라인 — Stage 1 핵심 함수 라이브러리
– Wide-format 변환 및 결측치 보간 (벡터화)

History
  2026-04-13  Created
  2026-05-02  Renamed from data_pipeline.py → data_01_pipeline.py
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Step 2+3 : Raw(long-format) → Wide-format Pivot
# --------------------------------------------------------------------------- #

def raw_to_wide(df_raw: pd.DataFrame,
                col_accu: str,
                list_hid) -> pd.DataFrame:
    """
    원시 데이터(long-format) → wide-format DataFrame 단일 pivot 변환.

    기존 방식: 세대별 loop × 2회 (개별 저장 → 순차 merge) – O(N·M), 348회 I/O
    개선 방식: groupby + unstack 단일 연산 – O(N), 1회 연산

    Parameters
    ----------
    df_raw   : 'HOUSE_ID', 'METER_DATE', col_accu 컬럼을 가진 long-format DataFrame
    col_accu : 적산 컬럼명 (예: 'ELEC_ACTUAL_ACCU_EFF')
    list_hid : 공통 세대 HOUSE_ID 목록 (list / Series / Index)

    Returns
    -------
    wide-format DataFrame
      columns: ['METER_DATE', f'{col_accu}_{house_id}', ...]
      dtype  : METER_DATE → datetime64[ns], 세대 컬럼 → float64
    """
    df = df_raw[['METER_DATE', 'HOUSE_ID', col_accu]].copy()

    # 10분 단위로 floor (기존 코드의 math.floor(minute/10)*10 로직을 벡터화)
    df['METER_DATE'] = pd.to_datetime(df['METER_DATE']).dt.floor('10min')

    # 공통 세대만 필터
    df = df[df['HOUSE_ID'].isin(list_hid)]

    # 같은 세대·시각에 중복 값이 있을 경우 last 유지
    df = df.groupby(['METER_DATE', 'HOUSE_ID'])[col_accu].last()

    # HOUSE_ID → columns으로 pivot
    df_wide = df.unstack('HOUSE_ID').reset_index()

    # MultiIndex columns 정리: 'HOUSE_ID' index name 제거 후 flat rename
    # unstack 후: columns = Index([house_id1, house_id2, ...], name='HOUSE_ID')
    # reset_index 후: columns[0] = 'METER_DATE', columns[1:] = house IDs
    df_wide.columns.name = None
    df_wide.columns = ['METER_DATE'] + [
        f"{col_accu}_{h}" for h in df_wide.columns[1:]
    ]

    return df_wide


# --------------------------------------------------------------------------- #
#  Step 4 : 결측치 보간 (벡터화 3단계)
# --------------------------------------------------------------------------- #

def fill_missing_wide(df: pd.DataFrame,
                      house_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wide-format 결측치 3단계 대치 (완전 벡터화).

    Stage 1: 각 시각의 전체 세대 평균으로 NaN 대치
    Stage 2: all-NaN 시각(평균도 NaN)은 평균 선형보간 후 재대치
    Stage 3: 잔여 NaN 세대별 선형보간 (양방향)

    기존 row-loop(for i in range(len(df)) 방식, ~96K 회) 대비
    수십~수백 배 빠름.

    Parameters
    ----------
    df         : wide-format DataFrame (METER_DATE + 세대 컬럼 등)
    house_cols : 세대 컬럼명 리스트 (메타 컬럼 제외)

    Returns
    -------
    (df_stage1, df_final)
      df_stage1 : Stage 1 완료 후 (1차 평균 대치만 적용, 중간 저장용)
      df_final  : Stage 3 완료 후 (최종 결과)
    """
    df = df.copy()

    # ── Stage 1: 시각별 전체 평균으로 NaN 대치 ────────────────────────────
    # row_mean: 각 행(시각)의 세대 평균값 Series (index 동일)
    row_mean = df[house_cols].mean(axis=1)
    # apply는 column-wise → 각 컬럼의 NaN을 해당 행의 평균으로 fillna
    df[house_cols] = df[house_cols].apply(lambda col: col.fillna(row_mean))
    df_stage1 = df.copy()

    # ── Stage 2: all-NaN 시각 처리 ────────────────────────────────────────
    # Stage 1 이후에도 row_mean이 NaN인 시각(전 세대가 NaN이었던 경우)은
    # 시간 순 선형보간된 평균값으로 재대치
    row_mean_interp = df[house_cols].mean(axis=1).interpolate(method='linear')
    df[house_cols] = df[house_cols].apply(lambda col: col.fillna(row_mean_interp))

    # ── Stage 3: 세대별 선형보간 ──────────────────────────────────────────
    # Stage 2 이후 엣지(시작/끝) 잔여 NaN 처리
    df[house_cols] = df[house_cols].interpolate(
        method='linear', axis=0, limit_direction='both'
    )

    return df_stage1, df
