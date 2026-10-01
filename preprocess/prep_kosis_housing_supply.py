# -*- coding: utf-8 -*-
"""
[전처리] 주택건설실적(인허가/착공/준공) - 아파트, 광역지자체·월별

입력: raw:kosis/DT_MLTM_1948.csv (인허가), DT_MLTM_5387.csv (착공), DT_MLTM_5373.csv (준공)
출력: processed/주택건설실적_월별.csv  (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)

  통계명                     항목명            데이터사전
  주택건설실적_아파트         인허가/착공/준공   V021/V022/V023

이 데이터의 특성:
  - 지역 단위가 시도까지만 제공되고 자치구 단위는 없음
  - 인허가 표만 "월별 누계"라서 전월을 빼야 월별 실적이 됨 (1월은 누계=당월)
  - 착공/준공은 "월계"라 그대로 사용
  - 공표 후 수정으로 차분값이 음수가 되는 달이 있다. 원자료 그대로 두고 보정하지 않는다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import REGION_SET, REGIONS, check_unique_keys, load_raw, to_ym, write_processed  # noqa: E402

SOURCES = [
    ("인허가", "DT_MLTM_1948", True),   # True = 월별 누계 -> 차감 필요
    ("착공", "DT_MLTM_5387", False),
    ("준공", "DT_MLTM_5373", False),
]
# 소분류(C4) 명칭 -> 통계명
HOUSING_TYPES = {"아파트": "주택건설실적_아파트"}


def extract_series(tbl_id):
    """raw에서 {(통계명, 지역): {연월: 값}} 로 정리"""
    out = {}
    for row in load_raw("kosis", tbl_id):
        stat = HOUSING_TYPES.get(row.get("C4_NM"))
        region = row.get("C1_NM")
        if stat is None or region not in REGION_SET:  # 전국/수도권/지방소계 등 집계행 제외
            continue
        try:
            value = float(row["DT"])
        except (KeyError, ValueError):
            continue
        out.setdefault((stat, region), {})[row["PRD_DE"]] = value
    return out


def cumulative_to_monthly(series):
    """연중 누계 -> 월별 실적. 1월은 누계가 곧 당월 실적."""
    monthly = {}
    first = min(series) if series else None
    for prd in sorted(series):
        year, month = prd[:4], prd[4:6]
        current = series[prd]
        if month == "01" or prd == first:
            # 연중에 새로 생긴 지역(세종 2012-07)은 첫 관측월 누계가 곧 당월 실적
            monthly[prd] = current
        else:
            prev = series.get(f"{year}{int(month) - 1:02d}")
            monthly[prd] = None if prev is None else current - prev
    return monthly


def main():
    rows = []
    for item, tbl_id, is_cumulative in SOURCES:
        series = extract_series(tbl_id)
        for (stat, region), s in series.items():
            if is_cumulative:
                s = cumulative_to_monthly(s)
            for prd, v in s.items():
                if v is None:
                    continue
                rows.append([region, "", to_ym(prd), stat, item,
                             int(v) if float(v).is_integer() else v, "호"])
        print(f"- {item} ({tbl_id}): {len(series)}개 (통계명x지역)")

    rows.sort(key=lambda r: (r[3], r[4], REGIONS.index(r[0]), r[2]))
    check_unique_keys(rows, [0, 1, 2, 3, 4], "주택건설실적")
    path = write_processed("주택건설실적_월별",
                           ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"], rows)
    print(f"완료: {path} ({len(rows)}행)")
    print("※ 시도 단위까지만 제공되어 자치구 컬럼은 비어 있음")


if __name__ == "__main__":
    main()
