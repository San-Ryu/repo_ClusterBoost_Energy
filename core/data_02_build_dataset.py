"""
KIER 에너지 사용량 도메인별 Wide-format CSV 생성

출력 파일 (기본 ~/data/KIER_1_USAGE_House/):
  KIER_HINDEX.csv         : HINDEX, BLD, FLOOR, HOUSE, HASH  (348 세대)
  KIER_USAGE_ELEC.csv     : METER_DATE, ELEC_1-1-1, ...,     ELEC_3-{F}-{H}
  KIER_USAGE_WATER.csv    : METER_DATE, WATER_1-1-1, ...,    WATER_3-{F}-{H}
  KIER_USAGE_HEAT.csv     : METER_DATE, HEAT_1-1-1, ...,     HEAT_3-{F}-{H}
  KIER_USAGE_HOT_HEAT.csv : METER_DATE, HOT_HEAT_1-1-1, ..., HOT_HEAT_3-{F}-{H}
  KIER_USAGE_HOT_FLOW.csv : METER_DATE, HOT_FLOW_1-1-1, ..., HOT_FLOW_3-{F}-{H}
  KIER_USAGE_GAS.csv      : METER_DATE, GAS_1-1-1, ...,      GAS_3-{F}-{H}

컬럼 ID 규칙: {DOMAIN}_{BLD_SEQ}-{FLOOR_INT}-{HOUSE_NUM}
  - BLD_SEQ  : BLD 알파벳 오름차순 순번 (0561→1, 0562→2, 0563→3)
  - FLOOR_INT: FLOOR 정수값 (01→1, 24→24)
  - HOUSE_NUM: (BLD, FLOOR) 내 세대 순번

Usage:
    .venv/bin/python core/data_02_build_dataset.py

History:
  2026-04-22  Created
  2026-05-02  Renamed from build_kier_dataset.py → data_02_build_dataset.py
"""
import gc
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from core.project_paths import data_path
except ModuleNotFoundError:
    from project_paths import data_path

# ─── 경로 ──────────────────────────────────────────────────────────────────────
def _resolve_raw_dir() -> Path:
    """
    환경별 원시 데이터 폴더명을 순차 탐색해 실제 경로를 반환한다.
    """
    candidates = [
        data_path("KIER_0_USAGE_Raw", "KIER_0_Raw"),  # 현재 저장 구조
        data_path("KIER_0_USAGE_Raw"),                # 상위 폴더 직접 사용 구조
        data_path("KIER_0_USAGERaw"),                 # 과거/오타 경로 호환
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "원시 데이터 경로를 찾을 수 없습니다. "
        "확인 경로: " + ", ".join(str(p) for p in candidates)
    )


RAW_DIR  = _resolve_raw_dir()
OUT_DIR  = data_path("KIER_1_USAGE_House")

# ─── 도메인 정의 ──────────────────────────────────────────────────────────────
# (출력명, CSV 파일명, [CSV 값 컬럼명, ...])
# HOT은 1회 읽기 → HOT_HEAT / HOT_FLOW 2개 파일 출력
DOMAINS: list[tuple[str, str, list[str]]] = [
    ("ELEC",     "KIER_RAW_ELEC_2024-06-07.csv",  ["ELEC_ACTUAL_ACCU_EFF"]),
    ("WATER",    "KIER_RAW_WATER_2024-06-07.csv", ["WATER_ACCU_FLOW"]),
    ("HEAT",     "KIER_RAW_HEAT_2024-06-07.csv",  ["HEAT_ACCU_HEAT"]),
    ("HOT",      "KIER_RAW_HOT_2024-06-07.csv",   ["HOT_ACCU_HEAT", "HOT_ACCU_FLOW"]),
    ("GAS",      "KIER_RAW_GAS_2024-06-07.csv",   ["GAS_ACCU_FLOW"]),
]

# CSV 값 컬럼명 → 출력 도메인명
_VAL_TO_OUTPUT: dict[str, str] = {
    "ELEC_ACTUAL_ACCU_EFF": "ELEC",
    "WATER_ACCU_FLOW":      "WATER",
    "HEAT_ACCU_HEAT":       "HEAT",
    "HOT_ACCU_HEAT":        "HOT_HEAT",
    "HOT_ACCU_FLOW":        "HOT_FLOW",
    "GAS_ACCU_FLOW":        "GAS",
}


# ─── hIndex 생성 (Step 1) ──────────────────────────────────────────────────────
def build_hindex() -> None:
    """
    5개 도메인 공통 (BLD, FLOOR, HASH) 교집합 → KIER_HINDEX.csv 저장.
    컬럼: HINDEX, BLD, FLOOR, HOUSE, HASH
    """
    print("[ Step 1 ] hIndex 계산", flush=True)
    common: set | None = None

    for _, filename, _ in DOMAINS:
        path = RAW_DIR / filename
        ids: set[tuple] = set()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ids.add((row[1].strip(), row[2].strip(), row[3].strip()))
        common = ids if common is None else common & ids
        print(f"  {filename[:30]}: {len(ids):,} 세대", flush=True)

    print(f"  공통: {len(common):,} 세대", flush=True)

    house_ctr: dict = defaultdict(int)
    rows = []
    for hindex, (bld, floor, hash_val) in enumerate(sorted(common), 1):
        house_ctr[(bld, floor)] += 1
        rows.append((hindex, bld, floor, house_ctr[(bld, floor)], hash_val))

    out = OUT_DIR / "KIER_HINDEX.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["HINDEX", "BLD", "FLOOR", "HOUSE", "HASH"])
        w.writerows(rows)
    print(f"  → {out} ({len(rows):,} 행)\n", flush=True)


# ─── col_map 로드 ──────────────────────────────────────────────────────────────
def load_col_map() -> pd.DataFrame:
    """
    KIER_HINDEX.csv → merge 키 + col_id 열 추가한 DataFrame 반환.
    col_id 예시: '1-1-1' (BLD_SEQ-FLOOR_INT-HOUSE_NUM)
    """
    df = pd.read_csv(OUT_DIR / "KIER_HINDEX.csv", dtype=str)
    bld_seq = {b: str(i + 1) for i, b in enumerate(sorted(df["BLD"].unique()))}
    df["col_id"] = (
        df["BLD"].map(bld_seq)
        + "-"
        + df["FLOOR"].astype(int).astype(str)
        + "-"
        + df["HOUSE"]
    )
    return df[["BLD", "FLOOR", "HASH", "col_id"]]


# ─── 타임스탬프 정규화 ─────────────────────────────────────────────────────────
def _floor_10min(s: pd.Series) -> pd.Series:
    """문자열 Series 'YYYY-MM-DD HH:MM:SS' → 10분 내림 'YYYY-MM-DD HH:M0:00'
    형식 불량 행(길이 < 19, 비숫자 분 필드)은 원본 값 유지."""
    m_str = s.str[14:16]
    m_int = pd.to_numeric(m_str, errors="coerce")          # 파싱 불가 → NaN
    m_floored = (m_int // 10 * 10).fillna(0).astype(int)   # NaN → 0으로 폴백
    result = s.str[:14] + m_floored.astype(str).str.zfill(2) + ":00"
    return result.where(s.str.len() >= 19, other=s)        # 짧은 행은 원본 유지


# ─── 도메인 처리 (Step 2) ──────────────────────────────────────────────────────
def _col_sort_key(col_name: str) -> tuple[int, int, int]:
    """'DOMAIN_B-F-H' → (B, F, H) int 정렬 키 (도메인명에 _ 포함 가능: HOT_HEAT)"""
    bfh = col_name[col_name.rfind("_") + 1:]   # 마지막 _ 이후: '1-1-1'
    return tuple(int(x) for x in bfh.split("-"))


def process_domain(domain_key: str, path: Path, val_cols: list[str],
                   col_map: pd.DataFrame) -> None:
    """
    원시 CSV → wide-format CSV 변환·저장.

    - domain_key가 "HOT"이면 HOT_HEAT, HOT_FLOW 두 파일 출력
    - 동일 (METER_DATE, col_id) 중복은 마지막 값 유지 (원시 데이터 기준)
    """
    print(f"[ Step 2 ] {domain_key} 읽는 중...", flush=True)

    df = pd.read_csv(
        path,
        usecols=["METER_DATE", "HOUSE_ID_DONG", "HOUSE_ID_HO_PRE", "HOUSE_ID_HO"] + val_cols,
        dtype={"HOUSE_ID_DONG": "category", "HOUSE_ID_HO_PRE": "category",
               "HOUSE_ID_HO": "category"},
    )
    for col in val_cols:
        df[col] = df[col].astype("float32")
    print(f"  읽기 완료: {len(df):,} 행", flush=True)

    # METER_DATE 10분 단위 내림 → category로 메모리 절약
    df["METER_DATE"] = _floor_10min(df["METER_DATE"]).astype("category")

    # hIndex 세대 필터링 + col_id 매핑
    df = df.merge(
        col_map,
        left_on=["HOUSE_ID_DONG", "HOUSE_ID_HO_PRE", "HOUSE_ID_HO"],
        right_on=["BLD", "FLOOR", "HASH"],
        how="inner",
    )
    df = df.drop(columns=["HOUSE_ID_DONG", "HOUSE_ID_HO_PRE", "HOUSE_ID_HO",
                           "BLD", "FLOOR", "HASH"])

    # 값 컬럼별 wide-format 파일 생성
    for val_col in val_cols:
        output_name = _VAL_TO_OUTPUT[val_col]
        print(f"  pivot → KIER_USAGE_{output_name}.csv ...", flush=True)

        sub = df[["METER_DATE", "col_id", val_col]].copy()
        sub = sub.drop_duplicates(subset=["METER_DATE", "col_id"], keep="last")

        wide = sub.pivot_table(
            index="METER_DATE",
            columns="col_id",
            values=val_col,
            aggfunc="last",
            observed=True,
        )

        # 컬럼명: 'col_id' → '{OUTPUT_NAME}_{col_id}'  ex) ELEC_1-1-1
        wide.columns = [f"{output_name}_{c}" for c in wide.columns]
        wide.columns.name = None
        wide.index.name = "METER_DATE"

        # BLD-FLOOR-HOUSE 정수 순 정렬
        wide = wide[sorted(wide.columns, key=_col_sort_key)]
        wide = wide.sort_index().reset_index()

        out_path = OUT_DIR / f"KIER_USAGE_{output_name}.csv"
        wide.to_csv(out_path, index=False)
        print(f"  → {out_path}  ({wide.shape[0]:,} 행 × {wide.shape[1]} 열)", flush=True)

        del sub, wide

    del df
    gc.collect()
    print(flush=True)


# ─── ACCU 개명 + INST 생성 (Step 3) ──────────────────────────────────────────
DIVIDED_DIR = OUT_DIR / "KIER_1_DIVIDED_BY_DOMAIN_HOUSE"

# KIER_USAGE_{DOMAIN}.csv 또는 KIER_USAGE_{DOMAIN}_ACCU.csv 둘 다 지원
_DOMAIN_NAMES: list[str] = [
    "ELEC", "WATER", "HEAT", "HOT_HEAT", "HOT_FLOW", "GAS",
]


def _accu_path(domain: str) -> Path:
    return OUT_DIR / f"KIER_USAGE_{domain}_ACCU.csv"


def _inst_path(domain: str) -> Path:
    return OUT_DIR / f"KIER_USAGE_{domain}_INST.csv"


def rename_accu_and_create_inst() -> None:
    """
    KIER_USAGE_{DOMAIN}.csv → KIER_USAGE_{DOMAIN}_ACCU.csv  (rename)
    KIER_USAGE_{DOMAIN}_INST.csv 생성  (forward diff: INST[t] = ACCU[t+1] - ACCU[t])

    [검사] 348세대 컬럼 외 초과 컬럼 발견 시 제거 후 진행.
    [전처리 제외] NaN 대체·이상치 처리 없이 원시 diff값 그대로 저장.
    """
    print("[ Step 3 ] ACCU 개명 + INST 생성", flush=True)

    # hIndex 기반 기대 col_id 집합
    hindex = pd.read_csv(OUT_DIR / "KIER_HINDEX.csv", dtype=str)
    bld_seq = {b: str(i + 1) for i, b in enumerate(sorted(hindex["BLD"].unique()))}
    hindex["col_id"] = (
        hindex["BLD"].map(bld_seq)
        + "-"
        + hindex["FLOOR"].astype(int).astype(str)
        + "-"
        + hindex["HOUSE"]
    )

    for domain in _DOMAIN_NAMES:
        # 원본 파일 (이미 _ACCU인 경우 스킵)
        src_raw  = OUT_DIR / f"KIER_USAGE_{domain}.csv"
        src_accu = _accu_path(domain)
        src_inst = _inst_path(domain)

        if not src_raw.exists() and not src_accu.exists():
            print(f"  [{domain}] 파일 없음 — 스킵", flush=True)
            continue

        if src_raw.exists():
            # ── 검사: 예상 외 컬럼 제거 ──────────────────────────────────────
            df = pd.read_csv(src_raw, index_col="METER_DATE")
            expected = sorted(
                [f"{domain}_{cid}" for cid in hindex["col_id"]],
                key=_col_sort_key,
            )
            extra = set(df.columns) - set(expected)
            if extra:
                print(f"  [{domain}] 초과 컬럼 {len(extra)}개 제거: {list(extra)[:3]}...", flush=True)
                df = df[[c for c in expected if c in df.columns]]

            n_cols = len(df.columns)
            print(f"  [{domain}] {n_cols}세대 컬럼 확인 ✓", flush=True)

            # ── ACCU 저장 (rename) ────────────────────────────────────────────
            df.sort_index().reset_index().to_csv(src_accu, index=False)
            src_raw.unlink()                         # 원본 삭제
            print(f"  → {src_accu.name}", flush=True)
        else:
            # 이미 ACCU 파일이 존재하는 경우
            df = pd.read_csv(src_accu, index_col="METER_DATE")
            df = df.sort_index()
            print(f"  [{domain}] ACCU 이미 존재, INST만 생성", flush=True)

        # ── INST 생성: INST[t] = ACCU[t+1] - ACCU[t] ────────────────────────
        if not src_inst.exists():
            inst_df = df.shift(-1) - df          # forward diff, 마지막 행 NaN
            inst_df.reset_index().to_csv(src_inst, index=False)
            print(f"  → {src_inst.name}  ({inst_df.shape[0]:,} 행 × {inst_df.shape[1]} 열)", flush=True)
        else:
            print(f"  [{domain}] INST 이미 존재 — 스킵", flush=True)

        del df
        gc.collect()
        print(flush=True)


# ─── 도메인·세대별 분류 (Step 4) ──────────────────────────────────────────────
def split_by_domain_house() -> None:
    """
    KIER_USAGE_{DOMAIN}_{ACCU|INST}.csv → KIER_1_DIVIDED_BY_DOMAIN_HOUSE/{DOMAIN}_{ACCU|INST}/{col}.csv

    각 파일: METER_DATE, {DOMAIN}_{BLD}-{FLOOR}-{HOUSE}  (단일 세대)
    """
    print("[ Step 4 ] 도메인·세대별 분류", flush=True)

    targets = (
        [f"KIER_USAGE_{d}_ACCU.csv" for d in _DOMAIN_NAMES]
        + [f"KIER_USAGE_{d}_INST.csv" for d in _DOMAIN_NAMES]
    )

    for fname in targets:
        src = OUT_DIR / fname
        if not src.exists():
            print(f"  {fname} 없음 — 스킵", flush=True)
            continue

        # KIER_USAGE_{DOMAIN}_{ACCU|INST}.csv → '{DOMAIN}_{ACCU|INST}'
        stem  = fname.removeprefix("KIER_USAGE_").removesuffix(".csv")
        domain_dir = DIVIDED_DIR / stem
        domain_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {stem} 읽는 중...", flush=True)
        df = pd.read_csv(src, index_col="METER_DATE")

        for col in df.columns:
            df[[col]].reset_index().to_csv(domain_dir / f"{col}.csv", index=False)

        print(f"  → {domain_dir}  ({len(df.columns)} 파일)", flush=True)
        del df
        gc.collect()

    print(f"  완료: {DIVIDED_DIR}\n", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (OUT_DIR / "KIER_HINDEX.csv").exists():
        build_hindex()
    else:
        print("[ Step 1 ] KIER_HINDEX.csv 이미 존재 — 스킵\n", flush=True)

    col_map = load_col_map()

    for domain_key, filename, val_cols in DOMAINS:
        process_domain(domain_key, RAW_DIR / filename, val_cols, col_map)

    rename_accu_and_create_inst()
    split_by_domain_house()
    print("✓ 완료", flush=True)


if __name__ == "__main__":
    main()
