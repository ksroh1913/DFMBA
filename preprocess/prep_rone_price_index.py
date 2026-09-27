# -*- coding: utf-8 -*-
"""
[전처리] 아파트 가격지수 5종 - 매매/전세/월세/월세통합/전월세통합

입력: raw/rone/A_2024_00045.csv (매매), A_2024_00050.csv (전세),
      raw/rone/A_2024_00164.csv (월세 구버전), A_2024_00055.csv (월세 신버전),
      raw/rone/A_2024_00054.csv (월세통합), A_2024_00049.csv (전월세통합)
출력: processed/부동산원_가격지수_아파트.csv

이 데이터의 특성:
  - 모두 "지수" 한 항목뿐이고 구조가 같아 동일한 처리로 묶임
  - 서울은 자치구까지, 그 외는 시도 단위까지 제공 (단, 구버전 표는 아래 참고)

통계명은 R-ONE 통계표와 1:1로 대응시키고, 서로 다른 표를 이어 붙이지 않는다.
연결 여부는 분석 단계에서 판단할 문제이고, 여기서 합치면 아래 함정이 값에 묻힌다.

  통계명                      STATBL_ID       기간              기준시점
  매매가격지수_아파트           A_2024_00045    2003.01~         2026.06=100
  전세가격지수_아파트           A_2024_00050    2003.01~         2026.06=100
  전월세통합지수_아파트         A_2024_00049    2015.06~         2026.06=100
  월세통합가격지수_아파트       A_2024_00054    2015.06~         2026.06=100
  월세가격지수_아파트           A_2024_00055    2015.06~         2026.06=100
  월세가격지수(구)_아파트       A_2024_00164    2010.06~2015.06  2012.06=100

월세 계열을 이어 붙이려 할 때 걸리는 문제:
  - 기준시점이 다르다. 구버전 2012.06=100 vs 현행 2026.06=100. 그대로 이으면
    수준이 튀므로 리베이싱이 필요하다.
  - 다행히 2015.06 한 달이 겹친다(구버전 종료월 = 신버전 시작월). 이 시점의
    비율을 쓰면 리베이싱 기준점을 잡을 수 있다. 단 구버전은 조사 설계가 달라
    같은 대상을 재신 것이 아니므로, 접합값은 근사치로 봐야 한다.
  - 구버전은 지역 세분화가 얕다. 서울이 강남/강북 권역까지만이고 시도는 8개뿐,
    자치구 단위는 전 기간 없다. 따라서 자치구 단위 시계열은 2015.06 이후만 가능.
  - 구버전에는 '월세통합' 개념이 없어 월세통합가격지수(2015.06~)의 이전 구간에
    대응하는 원본 통계가 아예 존재하지 않는다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

# 통계명 -> STATBL_ID (표 하나당 통계명 하나. 서로 다른 표를 합치지 않는다)
SERIES = {
    "매매가격지수_아파트": "A_2024_00045",
    "전세가격지수_아파트": "A_2024_00050",
    "전월세통합지수_아파트": "A_2024_00049",
    "월세통합가격지수_아파트": "A_2024_00054",
    "월세가격지수_아파트": "A_2024_00055",
    "월세가격지수(구)_아파트": "A_2024_00164",
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
