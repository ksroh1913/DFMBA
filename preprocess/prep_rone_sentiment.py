# -*- coding: utf-8 -*-
"""
[전처리] 주택시장 소비심리지수

입력: raw/rone/T232543129897499.csv
출력: processed/부동산원_주택시장소비심리지수.csv

이 데이터의 특성:
  - 국토연구원 부동산시장 소비자동향조사 기반 지수(0~200, 100=보합)
  - 지역 계층이 '수도권>서울', '비수도권>강원'처럼 권역이 앞에 붙고 시도가 마지막에 온다
    (parse_rone_region이 마지막 조각으로 판정하므로 그대로 처리됨)
  - 자치구 단위는 제공되지 않아 시도 단위 행만 남는다
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

STATBL_ID = "T232543129897499"
OUT_NAME = "주택시장_소비심리지수"


def main():
    rows = []
    for row in load_raw("rone", STATBL_ID):
        region = parse_rone_region(row.get("CLS_FULLNM"))
        if region is None:
            continue  # '수도권>소계', '전국>소계' 등 집계행 제외
        sido, gu = region
        rows.append([sido, gu, to_ym(row.get("WRTTIME_IDTFR_ID", "")), OUT_NAME,
                     row.get("ITM_NM", ""), row.get("DTA_VAL", ""), row.get("UI_NM", "")])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    path = write_processed(
        "부동산원_주택시장소비심리지수",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
