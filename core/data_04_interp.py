"""
data_04_interp.py
=================
Stage 2: KIER ELEC INST 데이터에 5가지 보간법을 적용하고 결과를 저장한다.

Phase 1 — Preprocessing:
    각 보간법으로 전체 ELEC INST 데이터 처리 → 기본 ~/data/KIER_S2_interp/{method}/

Phase 2 — Evaluation:
    완전 세대(complete cases)에 인공 마스킹 → 6종 지표 계산 → results/preprocessing/

Usage:
    python core/data_04_interp.py
    python core/data_04_interp.py --methods baseline pchip stl  # 선택 실행
    python core/data_04_interp.py --workers 8 --n-trials 5      # 병렬/빠른 평가

History:
    2026-04-29  Created
    2026-05-02  Renamed from build_kier_s2_interp.py → data_04_interp.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from scipy.stats import wasserstein_distance
from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

try:
    from core.project_paths import data_path
except ModuleNotFoundError:
    from project_paths import data_path

warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent          # 프로젝트 루트
_DATA_IN     = data_path("KIER_1_USAGE_House")
_DATA_OUT    = data_path("KIER_S2_interp")
_RESULTS_DIR = _ROOT / "results" / "preprocessing"

# ── 상수 ───────────────────────────────────────────────────────────────────────
STL_PERIOD   = 144          # 10분 × 144 = 24H
MSTL_PERIODS = [144, 1008]  # 24H + 7D
N_TRIALS_DEF = 10
SAMPLE_N     = 30           # 평가에 사용할 완전 세대 수
RANDOM_SEED  = 42
META_COLS    = {'METER_DATE', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE',
                'MEAN_OF_INST', 'SUM_OF_INST'}


# ══════════════════════════════════════════════════════════════════════════════
#  보간 함수
# ══════════════════════════════════════════════════════════════════════════════

def interp_baseline(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """기존안: 행 평균 대치 + 선형 보간 (3단계 벡터화)."""
    df = df.copy()
    row_mean = df[cols].mean(axis=1)
    df[cols] = df[cols].apply(lambda c: c.fillna(row_mean))
    row_mean_interp = df[cols].mean(axis=1).interpolate(method='linear')
    df[cols] = df[cols].apply(lambda c: c.fillna(row_mean_interp))
    df[cols] = df[cols].interpolate(method='linear', axis=0, limit_direction='both')
    return df


def _pchip_col(s: pd.Series) -> pd.Series:
    valid = s.notna()
    if valid.sum() < 2:
        return s.interpolate(method='linear', limit_direction='both')
    idx = np.arange(len(s))
    fn = PchipInterpolator(idx[valid], s[valid], extrapolate=True)
    result = s.copy()
    result[~valid] = fn(idx[~valid]).clip(min=0)
    return result

def interp_pchip(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """1안: PCHIP — 단조성 보존, 오버슈트 없음."""
    df = df.copy()
    df[cols] = df[cols].apply(_pchip_col)
    return df


def _stl_col(s: pd.Series, period: int = STL_PERIOD) -> pd.Series:
    if s.isna().sum() == 0:
        return s
    s_lin = s.interpolate(method='linear', limit_direction='both')
    try:
        res = STL(s_lin, period=period, robust=True).fit()
        recon = pd.Series(res.trend + res.seasonal, index=s.index)
    except Exception:
        return s_lin
    result = s.copy()
    result[s.isna()] = recon[s.isna()].clip(lower=0)
    return result

def interp_stl(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """2안: STL 분해 후 재합성 (24H 일주기)."""
    df = df.copy()
    df[cols] = df[cols].apply(lambda c: _stl_col(c, STL_PERIOD))
    return df


def _mstl_col(s: pd.Series) -> pd.Series:
    if s.isna().sum() == 0:
        return s
    s_lin = s.interpolate(method='linear', limit_direction='both')
    try:
        res = MSTL(s_lin, periods=MSTL_PERIODS).fit()
        recon = pd.Series(res.trend + res.seasonal.sum(axis=1), index=s.index)
    except Exception:
        return s_lin
    result = s.copy()
    result[s.isna()] = recon[s.isna()].clip(lower=0)
    return result

def interp_mstl(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """3안: MSTL 분해 후 재합성 (24H + 7D 이중 주기)."""
    df = df.copy()
    df[cols] = df[cols].apply(_mstl_col)
    return df


def _sn_col(s: pd.Series, lag_day: int = 144, lag_week: int = 1008) -> pd.Series:
    result = s.copy()
    lag1 = s.shift(lag_day)
    f1 = result.isna() & lag1.notna()
    result[f1] = lag1[f1]
    lag2 = s.shift(lag_week)
    f2 = result.isna() & lag2.notna()
    result[f2] = lag2[f2]
    result = result.interpolate(method='linear', limit_direction='both')
    return result.clip(lower=0)

def interp_seasonal_naive(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """4안: Seasonal Naive — 24H lag → 7D lag fallback → 선형보간."""
    df = df.copy()
    df[cols] = df[cols].apply(_sn_col)
    return df


METHODS: dict[str, callable] = {
    'baseline':       interp_baseline,
    'pchip':          interp_pchip,
    'stl':            interp_stl,
    'mstl':           interp_mstl,
    'seasonal_naive': interp_seasonal_naive,
}

METHOD_LABELS = {
    'baseline':       '기존안 (대치+선형)',
    'pchip':          '1안 (PCHIP)',
    'stl':            '2안 (STL)',
    'mstl':           '3안 (MSTL)',
    'seasonal_naive': '4안 (Seasonal Naive)',
}


# ══════════════════════════════════════════════════════════════════════════════
#  평가 함수
# ══════════════════════════════════════════════════════════════════════════════

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))

def _roughness(s: pd.Series) -> float:
    return float(s.diff().diff().abs().mean())

def _acf_divergence(s_orig: pd.Series, s_interp: pd.Series,
                    nlags: int = STL_PERIOD) -> float:
    a1 = acf(s_orig.dropna(),   nlags=nlags, fft=True)
    a2 = acf(s_interp.dropna(), nlags=nlags, fft=True)
    return float(np.sqrt(np.mean((a1 - a2) ** 2)))

def _wass(s_orig: pd.Series, s_interp: pd.Series) -> float:
    return float(wasserstein_distance(s_orig.dropna().values,
                                      s_interp.dropna().values))

def _peak_error(s_full: pd.Series, s_filled: pd.Series,
                mask_idx: np.ndarray, q: float = 0.95) -> float:
    thr = np.quantile(s_full.dropna().values, q)
    t = s_full.iloc[mask_idx].values
    p = s_filled.iloc[mask_idx].values
    sel = t >= thr
    if sel.sum() == 0:
        return np.nan
    return _rmse(t[sel], p[sel])

def _daily_div(s_orig: pd.Series, s_interp: pd.Series,
               period: int = STL_PERIOD) -> float:
    idx = np.arange(len(s_orig)) % period
    op = s_orig.groupby(idx).mean()
    ip = s_interp.groupby(idx).mean()
    denom = op.std() * ip.std()
    if denom < 1e-9:
        return 0.0
    return float(1.0 - op.corr(ip))


def make_masks(length: int, seed: int):
    rng = np.random.default_rng(seed)
    rand_idx      = rng.choice(length, size=int(length * 0.10), replace=False)
    bs6           = rng.integers(0, length - 36)
    block_idx6    = np.arange(bs6, bs6 + 36)
    bs24          = rng.integers(0, length - 144)
    block_idx24   = np.arange(bs24, bs24 + 144)
    return rand_idx, block_idx6, block_idx24


def _valid_block_starts(valid_mask: np.ndarray, block_len: int) -> np.ndarray:
    """
    관측치(valid=True)만으로 연속 block_len 구간을 만들 수 있는 시작 인덱스 반환.
    """
    if len(valid_mask) < block_len:
        return np.array([], dtype=int)
    conv = np.convolve(valid_mask.astype(int), np.ones(block_len, dtype=int), mode='valid')
    return np.where(conv == block_len)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: 전체 데이터 보간 + 저장
# ══════════════════════════════════════════════════════════════════════════════

def _apply_and_save_method(method_key: str, df: pd.DataFrame,
                            house_cols: list[str]) -> dict:
    """단일 보간법 적용 후 Parquet 저장. 멀티프로세싱 worker로도 사용 가능."""
    out_dir = _DATA_OUT / method_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "KIER_USAGE_ELEC_INST_INTERP.parquet"

    t0 = time.time()
    fn = METHODS[method_key]
    df_filled = fn(df, house_cols)
    elapsed = time.time() - t0

    nan_before = int(df[house_cols].isna().sum().sum())
    nan_after  = int(df_filled[house_cols].isna().sum().sum())

    try:
        df_filled.to_parquet(out_path, index=False)
        fmt = 'parquet'
    except Exception:
        csv_path = out_path.with_suffix('.csv')
        df_filled.to_csv(csv_path, index=False)
        fmt = 'csv'
        out_path = csv_path

    return {
        'method':     method_key,
        'path':       str(out_path),
        'format':     fmt,
        'elapsed_s':  round(elapsed, 1),
        'nan_before': nan_before,
        'nan_after':  nan_after,
        'fill_rate':  round((nan_before - nan_after) / max(nan_before, 1) * 100, 2),
    }


def run_phase1(df: pd.DataFrame, house_cols: list[str],
               methods_to_run: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Phase 1: 전체 ELEC INST 보간 + 저장")
    print(f"  데이터 shape : {df.shape}")
    print(f"  보간법       : {methods_to_run}")
    print("=" * 60)

    records = []
    for mk in methods_to_run:
        print(f"\n  [{mk}] 처리 중...", end=' ', flush=True)
        info = _apply_and_save_method(mk, df, house_cols)
        records.append(info)
        print(f"완료 ({info['elapsed_s']}s) | "
              f"NaN {info['nan_before']:,} → {info['nan_after']} | "
              f"충전률 {info['fill_rate']}% | 저장: {info['path']}")

    df_log = pd.DataFrame(records)
    log_path = _DATA_OUT / "phase1_log.csv"
    df_log.to_csv(log_path, index=False)
    print(f"\n  Phase 1 로그 저장: {log_path}")
    return df_log


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: 정량 평가 + 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

def _eval_one_house(house: str, s_full: pd.Series, methods_to_run: list[str],
                    n_trials: int, df_eval: pd.DataFrame) -> list[dict]:
    """단일 세대 평가 (멀티프로세싱 worker용)."""
    s_full = s_full.astype(float)
    valid_mask = s_full.notna().values
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size < 200:
        return []

    rows = []
    for trial in range(n_trials):
        rng = np.random.default_rng(RANDOM_SEED + trial)

        # random 마스킹은 관측치에서만 샘플링
        rand_n = max(1, int(valid_idx.size * 0.10))
        rand_idx = rng.choice(valid_idx, size=rand_n, replace=False)
        masks = [('random', rand_idx)]

        # block 마스킹도 관측치 연속 구간에서만 생성
        for mask_type, block_len in [('block_6h', 36), ('block_24h', 144)]:
            starts = _valid_block_starts(valid_mask, block_len)
            if starts.size > 0:
                st = int(starts[rng.integers(0, starts.size)])
                masks.append((mask_type, np.arange(st, st + block_len)))

        for mask_type, mask_idx in masks:
            s_masked = s_full.copy()
            s_masked.iloc[mask_idx] = np.nan
            df_tmp = df_eval[['METER_DATE', house]].copy()
            df_tmp[house] = s_masked

            for mk in methods_to_run:
                fn = METHODS[mk]
                s_f = fn(df_tmp, [house])[house]
                rows.append({
                    'house':      house,
                    'trial':      trial,
                    'mask_type':  mask_type,
                    'method':     mk,
                    'rmse':       _rmse(s_full.iloc[mask_idx].values,
                                        s_f.iloc[mask_idx].values),
                    'roughness':  _roughness(s_f),
                    'acf_div':    _acf_divergence(s_full, s_f),
                    'wass_dist':  _wass(s_full, s_f),
                    'peak':       _peak_error(s_full, s_f, mask_idx),
                    'daily_corr': _daily_div(s_full, s_f),
                })
    return rows


def run_phase2(df: pd.DataFrame, house_cols: list[str],
               methods_to_run: list[str], n_trials: int,
               workers: int = 1) -> None:
    print("\n" + "=" * 60)
    print("Phase 2: 정량 평가 (인공 마스킹)")
    print(f"  완전 세대 샘플: {SAMPLE_N}개 | N_TRIALS={n_trials} | workers={workers}")
    print("=" * 60)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 완전 세대 선별 및 샘플링
    complete = [c for c in house_cols if df[c].isna().sum() == 0]
    print(f"  완전 세대 수: {len(complete)} / {len(house_cols)}")

    # 완전 세대가 없으면 결측이 적은 세대로 대체 샘플 구성
    if len(complete) == 0:
        print("  완전 세대가 없어, 결측이 적은 세대로 대체 평가를 수행합니다.")
        ratio = 1.0 - (df[house_cols].isna().sum() / len(df))
        pool, thr_used = [], None
        for thr in [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
            cand = ratio[ratio >= thr].index.tolist()
            if cand:
                pool, thr_used = cand, thr
                break
        if not pool:
            pool = ratio.sort_values(ascending=False).head(max(SAMPLE_N, 1)).index.tolist()
            print(f"  대체 후보 부족: 결측률 최저 상위 {len(pool)}세대를 사용합니다.")
        else:
            print(f"  대체 후보: 관측 비율 >= {thr_used:.0%} 세대 {len(pool)}개")
        sample_pool = pool
    else:
        sample_pool = complete

    if len(sample_pool) == 0:
        raise ValueError("평가 가능한 세대가 없습니다. 입력 데이터 결측 상태를 확인하세요.")

    rng = np.random.default_rng(RANDOM_SEED)
    sample = list(rng.choice(sample_pool,
                              size=min(SAMPLE_N, len(sample_pool)),
                              replace=False))
    print(f"  평가 샘플 수: {len(sample)}")
    df_eval = df[['METER_DATE'] + sample].copy()

    # 평가 실행
    all_records = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_eval_one_house, h, df_eval[h],
                          methods_to_run, n_trials, df_eval): h
                for h in sample
            }
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc='평가 진행'):
                all_records.extend(fut.result())
    else:
        for h in tqdm(sample, desc='세대 평가'):
            all_records.extend(
                _eval_one_house(h, df_eval[h], methods_to_run,
                                n_trials, df_eval))

    if len(all_records) == 0:
        raise ValueError(
            "평가 레코드가 생성되지 않았습니다. "
            "세대별 결측이 과도해 마스킹 가능한 관측 구간이 없는 상태입니다."
        )

    df_results = pd.DataFrame(all_records)
    df_results['method_label'] = df_results['method'].map(METHOD_LABELS)

    metrics_path = _RESULTS_DIR / "ELEC_interp_metrics.csv"
    df_results.to_csv(metrics_path, index=False)
    print(f"\n  원시 지표 저장: {metrics_path}")

    # 집계
    METRIC_COLS = ['rmse', 'roughness', 'acf_div', 'wass_dist', 'peak', 'daily_corr']
    summary = (df_results.groupby(['method', 'mask_type'])[METRIC_COLS]
               .mean().reset_index())
    summary['method_label'] = summary['method'].map(METHOD_LABELS)

    for col in METRIC_COLS:
        mu, sigma = summary[col].mean(), summary[col].std()
        summary[col + '_z'] = (summary[col] - mu) / (sigma + 1e-9)

    W = {'rmse_z': 0.40, 'roughness_z': 0.15, 'acf_div_z': 0.15,
         'wass_dist_z': 0.10, 'peak_z': 0.10, 'daily_corr_z': 0.10}
    summary['score'] = sum(summary[k] * v for k, v in W.items())

    summary_path = _RESULTS_DIR / "ELEC_interp_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  집계 요약 저장: {summary_path}")

    _save_rank(summary)
    _plot_bar(summary, METRIC_COLS)
    _plot_heatmap(summary)
    _plot_curves(df_eval, sample, methods_to_run)


def _save_rank(summary: pd.DataFrame) -> None:
    rank = (summary.groupby(['method', 'method_label'])['score']
            .mean().reset_index()
            .sort_values('score')
            .reset_index(drop=True))
    rank.index += 1
    rank.columns = ['method', 'method_label', 'composite_score']
    rank['rank'] = rank.index

    rank_path = _RESULTS_DIR / "ELEC_interp_rank.csv"
    rank.to_csv(rank_path, index=False)

    print("\n" + "=" * 50)
    print("  보간법 종합 순위 (composite score ↓ 우수)")
    print("=" * 50)
    best = rank.iloc[0]['method']
    for _, row in rank.iterrows():
        marker = ' ◀ 선정' if row['method'] == best else ''
        print(f"  {int(row['rank'])}위  {row['method_label']:<25}  "
              f"score={row['composite_score']:+.4f}{marker}")
    print(f"\n  순위 저장: {rank_path}")


def _plot_bar(summary: pd.DataFrame, metric_cols: list[str]) -> None:
    metric_info = [
        ('rmse',       'RMSE (↓)'),
        ('roughness',  'Roughness (↓)'),
        ('acf_div',    'ACF Divergence (↓)'),
        ('wass_dist',  'Wasserstein (↓)'),
        ('peak',       'Peak Error (↓)'),
        ('daily_corr', 'Daily Pattern Div (↓)'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(21, 9))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metric_info):
        pivot = summary.pivot(index='method_label', columns='mask_type',
                              values=metric)
        pivot.plot(kind='bar', ax=ax, rot=20)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('')
        ax.legend(title='mask type', fontsize=8)
    plt.suptitle(f'보간법 비교 — ELEC (n={SAMPLE_N}세대 × N_TRIALS)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = _RESULTS_DIR / "ELEC_interp_bar.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  bar chart 저장: {out}")


def _plot_heatmap(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index='method_label', columns='mask_type',
                          values='score')
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                linewidths=0.5,
                cbar_kws={'label': 'Composite Score (↓ 우수)'})
    ax.set_title('Composite Score 히트맵 (낮을수록 우수)', fontsize=12)
    ax.set_xlabel('마스킹 유형')
    ax.set_ylabel('')
    plt.tight_layout()
    out = _RESULTS_DIR / "ELEC_score_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  score heatmap 저장: {out}")


def _plot_curves(df_eval: pd.DataFrame, sample: list[str],
                 methods_to_run: list[str]) -> None:
    house_ex = sample[0]
    s_full   = df_eval[house_ex]
    _, block_idx6, block_idx24 = make_masks(len(s_full), RANDOM_SEED)

    styles = {
        'baseline':       ('b--',  0.80),
        'pchip':          ('r-',   0.90),
        'stl':            ('g-.',  0.90),
        'mstl':           ('m:',   0.90),
        'seasonal_naive': ('c-',   0.90),
    }

    for mask_label, mask_idx in [('6H_block', block_idx6),
                                   ('24H_block', block_idx24)]:
        s_masked = s_full.copy()
        s_masked.iloc[mask_idx] = np.nan
        df_tmp = df_eval[['METER_DATE', house_ex]].copy()
        df_tmp[house_ex] = s_masked

        view = slice(max(0, mask_idx[0] - 30), mask_idx[-1] + 30)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(s_full.iloc[view].values, 'k-', lw=1.8, label='원본', alpha=0.8)

        for mk in methods_to_run:
            if mk not in styles:
                continue
            s_f = METHODS[mk](df_tmp, [house_ex])[house_ex]
            ls, alpha = styles[mk]
            ax.plot(s_f.iloc[view].values, ls, lw=1.2, alpha=alpha,
                    label=METHOD_LABELS.get(mk, mk))

        ax.axvspan(30, 30 + len(mask_idx), alpha=0.10, color='gray',
                   label='마스킹 구간')
        ax.set_title(f'보간 곡선 비교 — {house_ex} ({mask_label})')
        ax.legend(fontsize=8)
        ax.set_xlabel('10분 스텝')
        ax.set_ylabel('INST (kWh)')
        plt.tight_layout()
        out = _RESULTS_DIR / f"ELEC_interp_curve_{mask_label}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  곡선 시각화 저장: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global SAMPLE_N

    parser = argparse.ArgumentParser(
        description='KIER ELEC INST Stage 2 보간법 비교 파이프라인')
    parser.add_argument('--methods', nargs='+',
                        default=list(METHODS.keys()),
                        choices=list(METHODS.keys()),
                        help='적용할 보간법 (기본: 전체 5안)')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS_DEF,
                        help=f'마스킹 반복 횟수 (기본: {N_TRIALS_DEF})')
    parser.add_argument('--workers', type=int, default=1,
                        help='병렬 프로세스 수 (기본: 1, 순차 처리)')
    parser.add_argument('--phase', choices=['all', '1', '2'], default='all',
                        help='실행 Phase 선택 (기본: all)')
    parser.add_argument('--sample-n', type=int, default=SAMPLE_N,
                        help=f'평가 샘플 세대 수 (기본: {SAMPLE_N})')
    args = parser.parse_args()

    SAMPLE_N = args.sample_n

    # 데이터 로드
    in_path = _DATA_IN / "KIER_USAGE_ELEC_INST.csv"
    print(f"\n데이터 로드: {in_path}")
    t0 = time.time()
    df = pd.read_csv(in_path)
    df['METER_DATE'] = pd.to_datetime(df['METER_DATE'])
    house_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  shape: {df.shape} | 로드 시간: {time.time()-t0:.1f}s")
    print(f"  세대 수: {len(house_cols)}")
    print(f"  전체 NaN 수: {df[house_cols].isna().sum().sum():,}")

    # Phase 실행
    if args.phase in ('all', '1'):
        run_phase1(df, house_cols, args.methods)

    if args.phase in ('all', '2'):
        run_phase2(df, house_cols, args.methods,
                   n_trials=args.n_trials, workers=args.workers)

    print("\n✓ 완료")


if __name__ == '__main__':
    main()
