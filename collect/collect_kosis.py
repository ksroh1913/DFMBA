# -*- coding: utf-8 -*-
"""
[수집] KOSIS(국가통계포털) Open API -> raw/kosis/<통계표ID>.csv

응답 필드를 가공 없이 그대로 저장한다. 지역 필터링·누계 차감·시도 합산 등
모든 가공은 preprocess/ 단계에서 수행.

수집 통계표:
  DT_MLTM_1948  주택유형별 주택건설 인허가실적(월별 누계)   - 아파트
  DT_MLTM_5387  주택유형별 주택건설 착공실적(월계)          - 아파트
  DT_MLTM_5373  주택유형별 주택건설 준공실적(월계)          - 아파트
  DT_1B040B3    행정구역(시군구)별 주민등록세대수
  DT_1ES1B01S   시도/성별 경제활동인구 총괄 (연간, ~2010)
  DT_1ES2B01S   시도 경제활동인구 (분기, 2011~2012)
  DT_1ES3A01S   시군구 경제활동인구 총괄 (반기, 2013~)
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    http_get_json, require_key, save_raw, raw_max_value, year_chunks,
)

API_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
API_KEY = require_key("KOSIS_API_KEY")
THIS_YEAR = date.today().year

# 재수집 시 최근 N개월(분기/반기 포함)은 확정치 반영을 위해 다시 받는다.
REFETCH_BUFFER_YEARS = 1

KEY_FIELDS = ["TBL_ID", "PRD_DE", "C1", "C2", "C3", "C4", "ITM_ID"]

# (통계표ID, orgId, itmId, 고정 obj 파라미터, prdSe, 시작연도, 종료연도(None=현재))
TABLES = [
    # --- 주택건설실적 (아파트 소분류 코드로 고정) ---
    dict(tbl="DT_MLTM_1948", org="116", itm="13103871090T1+", prd="M", start=2007, end=None,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL", "objL4": "13102871090D.0009"},
         desc="인허가실적_아파트"),
    dict(tbl="DT_MLTM_5387", org="116", itm="13103766969T1+", prd="M", start=2007, end=None,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL", "objL4": "13102766969D.0008"},
         desc="착공실적_아파트"),
    dict(tbl="DT_MLTM_5373", org="116", itm="13103766973T1+", prd="M", start=2007, end=None,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL", "objL4": "13102766973D.0008"},
         desc="준공실적_아파트"),
    # --- 주민등록세대수 ---
    dict(tbl="DT_1B040B3", org="101", itm="T1+", prd="M", start=2011, end=None,
         obj={"objL1": "ALL"}, desc="주민등록세대수"),
    # --- 경제활동인구 (표가 시기별로 교체됨) ---
    dict(tbl="DT_1ES1B01S", org="101", itm="T1+T2+T3+T4+T5+T6+T7+T8+", prd="Y", start=2000, end=2010,
         obj={"objL1": "ALL", "objL2": "ALL"}, desc="경제활동인구_연간"),
    dict(tbl="DT_1ES2B01S", org="101", itm="T1+T2+T3+T4+T5+T6+T7+T8+", prd="Q", start=2011, end=2012,
         obj={"objL1": "ALL", "objL2": "ALL"}, desc="경제활동인구_분기"),
    dict(tbl="DT_1ES3A01S", org="101", itm="T1+T2+T3+T4+T9+T5+T6+T7+T11+T8+T10+", prd="H",
         start=2013, end=None, obj={"objL1": "ALL"}, desc="경제활동인구_반기"),
]


def period_bounds(prd, year, is_start):
    """주기별 PRD 파라미터 문자열. 연간은 연도만, 그 외는 연도+기수."""
    if prd == "Y":
        return str(year)
    if prd == "M":
        return f"{year}01" if is_start else f"{year}12"
    if prd == "Q":
        return f"{year}01" if is_start else f"{year}04"
    if prd == "H":
        return f"{year}01" if is_start else f"{year}02"
    raise ValueError(prd)


def fetch_table(t):
    end_year = t["end"] or THIS_YEAR
    start_year = t["start"]

    # 종료 연도가 고정된 과거 표는 이미 받아뒀으면 건너뛴다 (원본이 더 이상 변하지 않음).
    last_prd = raw_max_value("kosis", t["tbl"], "PRD_DE")
    if t["end"] is not None and last_prd:
        print(f"- {t['tbl']} ({t['desc']}): 이미 수집 완료 (~{last_prd}), 건너뜀")
        return
    if last_prd:
        resume_year = max(start_year, int(last_prd[:4]) - REFETCH_BUFFER_YEARS)
        start_year = resume_year

    collected = []
    for cs, ce in year_chunks(start_year, end_year, years_per_chunk=3):
        params = {
            "method": "getList", "apiKey": API_KEY, "format": "json", "jsonVD": "Y",
            "orgId": t["org"], "tblId": t["tbl"], "itmId": t["itm"], "prdSe": t["prd"],
            "startPrdDe": period_bounds(t["prd"], cs, True),
            "endPrdDe": period_bounds(t["prd"], ce, False),
        }
        params.update(t["obj"])
        data = http_get_json(API_URL, params)
        if not isinstance(data, list):
            if isinstance(data, dict) and data.get("err") == "30":
                continue  # 해당 구간 데이터 없음
            raise RuntimeError(f"[{t['tbl']} {cs}-{ce}] KOSIS 오류: {data}")
        collected.extend(data)

    total = save_raw("kosis", t["tbl"], collected, KEY_FIELDS)
    print(f"- {t['tbl']} ({t['desc']}): {len(collected)}행 수신, raw 누적 {total}행")


def main():
    print("[수집] KOSIS -> raw/kosis/")
    for t in TABLES:
        fetch_table(t)
    print("완료")


if __name__ == "__main__":
    main()
