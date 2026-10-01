# -*- coding: utf-8 -*-
"""
[전처리] 아파트 가격지수 3종 - 월세통합/매매/전세

입력: raw:rone/A_2024_00054.csv (월세통합), A_2024_00045.csv (매매), A_2024_00050.csv (전세)
출력: processed/부동산원_가격지수_아파트.csv
데이터사전: V001 월세통합 / V002 매매 / V026 전세

이 데이터의 특성:
  - 모두 "지수" 한 항목뿐이고 구조가 같아 동일한 처리로 묶임
  - 서울은 자치구까지, 그 외는 시도 단위까지 제공
  - 기준시점 2026.06=100. 월세통합가격지수는 2015.06부터 공표된다(그 이전 공식 지수 없음).
  - 통계명은 R-ONE 통계표와 1:1로 대응시키고 서로 다른 표를 이어 붙이지 않는다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

# 통계명 -> STATBL_ID (표 하나당 통계명 하나. 서로 다른 표를 합치지 않는다)
SERIES = {
    "월세통합가격지수_아파트": "A_2024_00054",
    "매매가격지수_아파트": "A_2024_00045",
    "전세가격지수_아파트": "A_2024_00050",
}


def main():
    rows = []
    for name, statbl_id in SERIES.items():
        before = len(rows)
        periods = []
        for row in load_raw("rone", statbl_id):
            region = parse_rone_region(row.get("CLS_FULLNM"))
            if region is None:
                continue  # 권역/집계행 제외
            sido, gu = region
            wrttime = row.get("WRTTIME_IDTFR_ID", "")
            periods.append(wrttime)
            rows.append([sido, gu, to_ym(wrttime), name,
                         row.get("ITM_NM", ""), row.get("DTA_VAL", ""), row.get("UI_NM", "")])
        span = f"{to_ym(min(periods))}~{to_ym(max(periods))}" if periods else "데이터 없음"
        print(f"- {name} ({statbl_id}): {len(rows) - before}행, {span}")

    rows.sort(key=lambda r: (r[3], r[0], r[1], r[2]))
    path = write_processed(
        "부동산원_가격지수_아파트",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
