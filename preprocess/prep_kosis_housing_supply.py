# -*- coding: utf-8 -*-
"""
[전처리] 주택건설실적(인허가/착공/준공) - 아파트, 광역지자체·월별

입력: raw/kosis/DT_MLTM_1948.csv (인허가), DT_MLTM_5387.csv (착공), DT_MLTM_5373.csv (준공)
출력: processed/주택건설실적_아파트_월별.csv

이 데이터의 특성:
  - 지역 단위가 시도(광역지자체)까지만 제공되고 자치구 단위는 없음
  - 인허가 표만 "월별 누계"라서 전월을 빼야 월별 실적이 됨 (1월은 누계=당월)
  - 착공/준공은 "월계"라 그대로 사용
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import REGION_SET, REGIONS, load_raw, to_ym, write_processed  # noqa: E402

SOURCES = [
    ("인허가", "DT_MLTM_1948", True),   # True = 월별 누계 -> 차감 필요
    ("착공", "DT_MLTM_5387", False),
    ("준공", "DT_MLTM_5373", False),
]


def extract_region_series(tbl_id):
    """raw에서 아파트·대상지역 행만 골라 {지역: {연월: 값}} 로 정리"""
    by_region = {}
    for row in load_raw("kosis", tbl_id):
        if row.get("C4_NM") != "아파트":
            continue
        region = row.get("C1_NM")
        if region not in REGION_SET:  # 전국/수도권/지방소계 등 집계행 제외
            continue
        try:
            value = float(row["DT"])
        except (KeyError, ValueError):
            continue
        by_region.setdefault(region, {})[row["PRD_DE"]] = value
    return by_region


def cumulative_to_monthly(series):
    """연중 누계 -> 월별 실적. 1월은 누계가 곧 당월 실적."""
    monthly = {}
    for prd in sorted(series):
        year, month = prd[:4], prd[4:6]
        current = series[prd]
        if month == "01":
            monthly[prd] = current
        else:
            prev = series.get(f"{year}{int(month) - 1:02d}")
            monthly[prd] = None if prev is None else current - prev
    return monthly


def main():
    result = {}  # 통계명 -> {지역: {연월: 값}}
    for name, tbl_id, is_cumulative in SOURCES:
        series = extract_region_series(tbl_id)
        if is_cumulative:
            series = {r: cumulative_to_monthly(s) for r, s in series.items()}
        result[name] = series
        print(f"- {name}: {len(series)}개 지역")

    all_periods = sorted({
        prd for series in result.values() for s in series.values() for prd in s
    })

    rows = []
    for region in REGIONS:
        for prd in all_periods:
            values = [result[name].get(region, {}).get(prd) for name, _, _ in SOURCES]
            if all(v is None for v in values):
                continue
            fmt = lambda v: "" if v is None else (int(v) if float(v).is_integer() else v)
            rows.append([region, "", to_ym(prd)] + [fmt(v) for v in values])

    path = write_processed(
        "주택건설실적_아파트_월별",
        ["광역지자체", "자치구", "연월", "인허가(호)", "착공(호)", "준공(호)"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")
    print("※ 이 통계는 시도 단위까지만 제공되어 자치구 컬럼은 비어 있음")


if __name__ == "__main__":
    main()
