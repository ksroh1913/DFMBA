# -*- coding: utf-8 -*-
"""
[전처리] 행정구역별 아파트 거래현황 / 아파트 매매거래현황

입력: raw/rone/A_2024_00549.csv (전체 거래), A_2024_00554.csv (매매 거래)
출력: processed/부동산원_아파트거래현황.csv

이 데이터의 특성:
  - 지수가 아니라 실적치이고, 한 시점에 '동(호)수'와 '면적' 두 항목이 함께 나온다
    (단위가 각각 동(호)수 / 천㎡ 로 달라 항목명을 그대로 남긴다)
  - 시도 합계가 '경기>계' 형태의 별도 행으로 제공되고, 서울은 '서울>강남구'처럼
    2단 계층이라 가격지수(4단 계층)와 구조가 다르다
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

SERIES = [
    ("행정구역별_아파트거래현황", "A_2024_00549"),
    ("행정구역별_아파트매매거래현황", "A_2024_00554"),
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

    rows.sort(key=lambda r: (r[3], r[0], r[1], r[2], r[4]))
    path = write_processed(
        "부동산원_아파트거래현황",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
