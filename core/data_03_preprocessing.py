"""
data_03_preprocessing.py
데이터 전처리 (리샘플링, 이상치 제거) 공통 모듈 — Stage 2 지원

History
  2023-11-14  Created
  2026-03-30  Refactored - chained indexing 제거, 벡터화, import 정리
  2026-05-02  Renamed from data_preprocessing.py → data_03_preprocessing.py
"""

import numpy as np
import pandas as pd

from core import data_analysis as com_Analysis


# --------------------------------------------------------------------------- #
#  Resampling
# --------------------------------------------------------------------------- #

def resample_by_last(df: pd.DataFrame, domain: str,
                     col: str, interval: str) -> pd.DataFrame:
    """
    날짜 컬럼 기준으로 지정 간격(interval)의 마지막 값으로 리샘플.
    """
    resample_col = f"{domain}_{col}"
    df[resample_col] = pd.to_datetime(df[col])
    df = df.dropna()
    df = df.resample(interval, on=resample_col).last()
    return df


# --------------------------------------------------------------------------- #
#  Outlier Removal (IQR)
# --------------------------------------------------------------------------- #

def del_outlier_usages(df: pd.DataFrame, col: str,
                       max_iter: int = 10) -> pd.DataFrame:
    """
    IQR 방식으로 이상치를 반복 보정.
    - 상/하한 초과 → 인접 두 값의 평균으로 대체
    - 음수 → NaN 처리

    Parameters
    ----------
    max_iter : 최대 반복 횟수 (이상치가 없으면 조기 종료)
    """
    s = df[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.90)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    total_fixed = 0

    for iteration in range(1, max_iter + 1):
        outlier_rows = com_Analysis.find_outlier_iqr(df, col)
        if not outlier_rows:
            break

        values = df[col].values.copy()
        fixed_this_round = 0

        for row in reversed(outlier_rows):
            val = values[row]
            if val > upper or val < lower:
                if 0 < row < len(values) - 1:
                    values[row] = (values[row - 1] + values[row + 1]) / 2
                fixed_this_round += 1
            if val < 0:
                values[row] = np.nan
                fixed_this_round += 1

        df[col] = values
        total_fixed += fixed_this_round

    print(f"  Iterations: {iteration}, Total fixed: {total_fixed}")
    return df
