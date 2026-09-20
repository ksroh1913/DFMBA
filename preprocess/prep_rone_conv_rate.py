# -*- coding: utf-8 -*-
"""
[전처리] 지역별 전월세 전환율 - 아파트

입력: raw/rone/A_2024_00156.csv
출력: processed/부동산원_전월세전환율_아파트.csv

이 데이터의 특성:
  - 전세보증금을 월세로 돌릴 때 적용되는 연 환산 이율(%)이라 지수 계열과 단위가 다름
  - 항목이 '전월세 전환율' 하나뿐
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

STATBL_ID = "A_2024_00156"
OUT_NAME = "지역별전월세전환율_아파트"


def main():
    rows = []
    for row in load_raw("rone", STATBL_ID):
        region = parse_rone_region(row.get("CLS_FULLNM"))
        if region is None:
            continue
        sido, gu = region
        rows.append([sido, gu, to_ym(row.get("WRTTIME_IDTFR_ID", "")), OUT_NAME,
                     row.get("ITM_NM", ""), row.get("DTA_VAL", ""), row.get("UI_NM", "")])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    path = write_processed(
        "부동산원_전월세전환율_아파트",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
