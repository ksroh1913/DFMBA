# -*- coding: utf-8 -*-
"""
[전처리] 주민등록 인구·세대·청년층 인구·인구이동 - 17개 시도 + 서울 25개 구, 월별

입력: raw:kosis/DT_1B040A3.csv     시군구별 주민등록인구            (V009)
      raw:kosis/DT_1B040B3.csv     시군구별 주민등록세대수          (V010/V039/V073)
      raw:kosis/DT_1B04006.csv     시군구/1세별 주민등록인구 20~39세 (V011)
      raw:kosis/DT_1B26001_A01.csv 시군구별 순이동자수               (V012)
출력: processed/주민등록_인구세대이동_월별.csv  (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 네 표 모두 행정표준코드를 쓴다: 2자리=시도, 5자리=시군구. 그래서 한 스크립트로 묶었다.
  - 서울은 시도 합계(11)와 25개 구를 모두 낸다. (기존 주민등록세대수_월별.csv는 서울 합계가
    빠져 있었다)
  - 2026-07부터 광주(29)·전남(46)이 통합코드(12)로만 공표된다. 통합코드 아래 광주 5개 구
    코드가 남아 있어 광주 = 5개 구 합계, 전남 = 통합 - 광주 로 복원한다.
    이 복원은 저량(인구·세대)과 순이동에만 성립한다(광주↔전남 이동은 두 값에서 상쇄된다).
  - 20~39세 비중(%)은 20~39세 인구 / 총인구 x 100 으로 계산한 파생값이다.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    ADMIN_SIDO, SEOUL_GU, check_unique_keys, load_raw, restore_gwangju_jeonnam, to_ym,
    write_processed,
)

HEADER = ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"]


def to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect(tbl, item_of):
    """raw를 {(항목키, 연월): {행정코드: 값}} 로 모은다. item_of(row) 가 None이면 버린다."""
    grid = defaultdict(dict)
    for row in load_raw("kosis", tbl):
        key = item_of(row)
        val = to_num(row.get("DT"))
        if key is None or val is None:
            continue
        grid[(key, row["PRD_DE"])][row["C1"]] = grid[(key, row["PRD_DE"])].get(row["C1"], 0) + val
    return grid


def emit(grid, stat, unit, restorable=True):
    """행정코드 격자를 (시도/서울구) 행으로 펼친다."""
    out = []
    for (item, prd), by_code in grid.items():
        ym = to_ym(prd)
        seen = set()
        for code, val in by_code.items():
            if len(code) == 2 and code in ADMIN_SIDO:
                sido = ADMIN_SIDO[code]
                if sido in seen:  # 강원 42/51 등 신·구 코드가 한 시점에 같이 오는 경우
                    continue
                seen.add(sido)
                out.append([sido, "", ym, stat, item, val, unit])
            elif code in SEOUL_GU:
                out.append(["서울", SEOUL_GU[code], ym, stat, item, val, unit])
        if restorable:
            for sido, val in restore_gwangju_jeonnam(by_code).items():
                out.append([sido, "", ym, stat, item, val, unit])
    return out


def age_band(row):
    code = row.get("C2", "")
    if code == "000":
        return "계"
    if code[:2] in ("12", "13"):
        return "20-29세"
    if code[:2] in ("15", "16"):
        return "30-39세"
    return None


def main():
    rows = []

    grid = collect("DT_1B040A3", lambda r: "총인구수" if r.get("ITM_ID") == "T20" else None)
    rows += emit(grid, "주민등록인구", "명")

    grid = collect("DT_1B040B3", lambda r: "세대수" if r.get("ITM_ID") == "T1" else None)
    rows += emit(grid, "주민등록세대수", "세대")

    # 연령: 20대·30대 합계를 먼저 만든 뒤 20~39세와 비중을 파생
    grid = collect("DT_1B04006", age_band)
    derived = defaultdict(dict)
    for (item, prd), by_code in grid.items():
        if item in ("20-29세", "30-39세"):
            for code, v in by_code.items():
                derived[("20-39세", prd)][code] = derived[("20-39세", prd)].get(code, 0) + v
    grid.update(derived)
    age_rows = emit(grid, "주민등록인구_연령", "명")
    total = {(r[0], r[1], r[2]): r[5] for r in age_rows if r[4] == "계"}
    for r in list(age_rows):
        if r[4] == "20-39세" and total.get((r[0], r[1], r[2])):
            age_rows.append([r[0], r[1], r[2], "주민등록인구_연령", "20-39세 비중",
                             r[5] / total[(r[0], r[1], r[2])] * 100, "%"])
    rows += age_rows

    grid = collect("DT_1B26001_A01", lambda r: "순이동" if r.get("ITM_ID") == "T25" else None)
    rows += emit(grid, "인구이동", "명")

    # 정수값은 정수로
    for r in rows:
        if isinstance(r[5], float) and r[5].is_integer():
            r[5] = int(r[5])
    rows.sort(key=lambda r: (r[3], r[4], r[0], r[1], r[2]))
    check_unique_keys(rows, [0, 1, 2, 3, 4], "주민등록_인구세대이동")
    path = write_processed("주민등록_인구세대이동_월별", HEADER, rows)

    by_stat = defaultdict(int)
    for r in rows:
        by_stat[(r[3], r[4])] += 1
    for k, n in sorted(by_stat.items()):
        print(f"- {k[0]} / {k[1]}: {n}행")
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
