# -*- coding: utf-8 -*-
"""
[전처리] 지역별 주택관련대출 (예금은행, 말잔) - 시도별 월간

입력: raw/ecos/지역별_주택관련대출_예금은행.csv
출력: processed/ECOS_지역별_주택관련대출.csv

이 데이터의 특성:
  - ECOS 지표 중 유일하게 지역(시도) 축이 있어 지역별 분석에 바로 쓸 수 있다
  - 지역이 ITEM_CODE2/ITEM_NAME2에 들어 있고, ITEM_CODE1은 대출 종류를 뜻한다
    (수집 단계에서 11110A0 = 주택관련대출-예금은행 으로 고정해 받음)
  - 말잔(월말 잔액)이라 유량이 아닌 저량 지표다. 증감을 쓰려면 차분해야 한다
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import REGION_SET, load_raw, write_processed  # noqa: E402

OUT_NAME = "지역별_주택관련대출_예금은행"


def main():
    rows = []
    skipped = set()
    for row in load_raw("ecos", "지역별_주택관련대출_예금은행"):
        region = row.get("ITEM_NAME2", "")
        if region not in REGION_SET:
            skipped.add(region)  # '전국' 등 집계행
            continue
        value = row.get("DATA_VALUE", "")
        if value in ("", None):
            continue
        time_value = str(row.get("TIME", ""))
        rows.append([region, "", f"{time_value[:4]}-{time_value[4:6]}", OUT_NAME,
                     row.get("ITEM_NAME1", ""), value, row.get("UNIT_NAME", "")])

    rows.sort(key=lambda r: (r[0], r[2]))
    path = write_processed(
        "ECOS_지역별_주택관련대출",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")
    if skipped:
        print(f"※ 집계행 제외: {', '.join(sorted(s for s in skipped if s))}")


if __name__ == "__main__":
    main()
