# -*- coding: utf-8 -*-
"""
[전처리] 아파트 가격지수 5종 - 매매/전세/월세/월세통합/전월세통합

입력: raw/rone/A_2024_00045.csv (매매), A_2024_00050.csv (전세),
      raw/rone/A_2024_00164.csv (월세 구버전), A_2024_00055.csv (월세 신버전),
      raw/rone/A_2024_00054.csv (월세통합), A_2024_00049.csv (전월세통합)
출력: processed/부동산원_가격지수_아파트.csv

이 데이터의 특성:
  - 5종 모두 "지수" 한 항목뿐이고 구조가 같아 동일한 처리로 묶임
  - 월세가격지수만 2015년 조사 개편으로 표가 갈려서, 구버전(2010~2015)과
    신버전(2016~)을 같은 통계명으로 이어붙인다 (기간이 겹치지 않아 충돌 없음)
  - 서울은 자치구까지, 그 외는 시도 단위까지 제공
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

# 통계명 -> [(STATBL_ID, 사용할 기간 상한 None=제한없음)]
SERIES = {
    "매매가격지수_아파트": [("A_2024_00045", None)],
    "전세가격지수_아파트": [("A_2024_00050", None)],
    "월세가격지수_아파트": [("A_2024_00164", "201512"), ("A_2024_00055", None)],
    "월세통합가격지수_아파트": [("A_2024_00054", None)],
    "전월세통합지수_아파트": [("A_2024_00049", None)],
}


def main():
    rows = []
    for name, segments in SERIES.items():
        before = len(rows)
        for statbl_id, until in segments:
            for row in load_raw("rone", statbl_id):
                wrttime = row.get("WRTTIME_IDTFR_ID", "")
                if until and wrttime > until:
                    continue  # 구버전 표에 신버전 기간이 섞여 있어도 잘라낸다
                region = parse_rone_region(row.get("CLS_FULLNM"))
                if region is None:
                    continue
                sido, gu = region
                rows.append([sido, gu, to_ym(wrttime), name,
                             row.get("ITM_NM", ""), row.get("DTA_VAL", ""), row.get("UI_NM", "")])
        print(f"- {name}: {len(rows) - before}행")

    rows.sort(key=lambda r: (r[3], r[0], r[1], r[2]))
    path = write_processed(
        "부동산원_가격지수_아파트",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
