# -*- coding: utf-8 -*-
"""
[전처리] 수급동향 3종 - 매매/전세/월세 (아파트, 현행 조사)

입력: raw/rone/A_2024_00076.csv (매매), A_2024_00077.csv (전세), A_2024_00078.csv (월세)
출력: processed/부동산원_수급동향_아파트.csv

이 데이터의 특성:
  - 0~200 범위의 수급지수(100=균형, >100 수요우위)
  - 지역 세분화 수준이 가격지수보다 낮아 서울은 권역(강남지역/강북지역)까지만 있고
    자치구 단위는 없다. 따라서 결과에는 시도 단위 행만 남는다.
  - 2010~2015년 (구)조사분은 산출 방식이 달라 prep_rone_supply_old.py에서 별도 처리
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

SERIES = [
    ("매매수급동향_아파트", "A_2024_00076"),
    ("전세수급동향_아파트", "A_2024_00077"),
    ("월세수급동향_아파트", "A_2024_00078"),
]


def main():
    rows = []
    for name, statbl_id in SERIES:
        before = len(rows)
        for row in load_raw("rone", statbl_id):
            region = parse_rone_region(row.get("CLS_FULLNM"))
            if region is None:
                continue
            sido, gu = region
            rows.append([sido, gu, to_ym(row.get("WRTTIME_IDTFR_ID", "")), name,
                         row.get("ITM_NM", ""), row.get("DTA_VAL", ""), row.get("UI_NM", "")])
        print(f"- {name}: {len(rows) - before}행")

    rows.sort(key=lambda r: (r[3], r[0], r[1], r[2]))
    path = write_processed(
        "부동산원_수급동향_아파트",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
