# -*- coding: utf-8 -*-
"""
[전처리] 미분양주택현황

입력: raw/rone/T237973129847263.csv
출력: processed/부동산원_미분양주택현황.csv

이 데이터의 특성:
  - 공급 과잉/부족을 직접 보여주는 재고 지표(단위: 호)
  - 서울은 '서울>강남구'처럼 자치구까지, 시도 합계는 '서울>계' 형태로 제공
  - 지수 계열과 달리 0이 정상값이다(미분양이 없는 달). 값이 비어 있는 것과 구분 필요
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

STATBL_ID = "T237973129847263"
OUT_NAME = "미분양주택현황"


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
        "부동산원_미분양주택현황",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
