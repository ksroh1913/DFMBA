# -*- coding: utf-8 -*-
"""
[전처리] 아파트 전세가율 (매매가격 대비 전세가격 비율, %)

입력: raw:rone/A_2024_00072.csv   (V003)
출력: processed/부동산원_전세가율_아파트.csv  (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 지수가 아니라 가격 '수준'의 비율(%)이라 가격지수 파일과 분리했다.
  - 지역 계층이 가격지수 계열과 같다(서울>강북지역>도심권>종로구). 권역 중간 집계행은 제외.
  - 2012.01 시작, 지역별로 시작월이 다를 수 있다(세종 등).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check_unique_keys, load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

STATBL = "A_2024_00072"
STAT = "전세가율_아파트"


def main():
    rows = []
    for r in load_raw("rone", STATBL):
        region = parse_rone_region(r.get("CLS_FULLNM"))
        if region is None or r.get("DTA_VAL") in ("", None):
            continue
        rows.append([*region, to_ym(r["WRTTIME_IDTFR_ID"]), STAT, r.get("ITM_NM", ""),
                     r["DTA_VAL"], r.get("UI_NM", "%")])
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    check_unique_keys(rows, [0, 1, 2, 3, 4], STAT)
    path = write_processed("부동산원_전세가율_아파트",
                           ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"], rows)
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
