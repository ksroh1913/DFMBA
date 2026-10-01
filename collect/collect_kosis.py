# -*- coding: utf-8 -*-
"""
[수집] KOSIS(국가통계포털) Open API -> raw/<동인 폴더>/ (위치: raw_layout.py)<통계표ID>.csv

응답 필드를 가공 없이 그대로 저장한다. 지역 필터링·누계 차감·시도 합산 등
모든 가공은 preprocess/ 단계에서 수행.

수집 통계표 (괄호 안은 데이터사전 Variable_Master ID). 사전이 지정한 항목만 받는다.
  [월] DT_MLTM_1948  주택건설 인허가실적(월별 누계) 아파트   V021
  [월] DT_MLTM_5387  주택건설 착공실적(월계) 아파트          V022
  [월] DT_MLTM_5373  주택건설 준공실적(월계) 아파트          V023
  [월] DT_1B040B3    시군구별 주민등록세대수                V010
  [월] DT_1B040A3    시군구별 주민등록인구                  V009
  [월] DT_1B04006    시군구/1세별 주민등록인구 20~39세      V011
  [월] DT_1B26001_A01 시군구별 순이동자수                   V012
  [월] DT_1DA7004S   시도별 고용률·취업자                   V013
  [반기] DT_1ES3A01S 시군구별 고용률·취업자 (서울 구)        V013
  [연] DT_1C96       1인당 GRDP·가계총처분가능소득         V015/V017
  [연] DT_201004_O090005  서울 구별 주거용 신축허가         V024
  [연] DT_1IN1502    인구주택총조사 가구·주택 수            V072
  [연] DT_MLTM_6827 / 7174  임대주택공급현황                V076
  [연] DT_MLTM_5560  임대주택건설공급현황                   V077
  [분기] DT_1L9I002  가계동향 도시2인이상 소득              V074

광주·전남은 2026-07부터 '12 전남광주통합특별시'로만 공표된다. 통합 후에도
광주 5개 구 코드(12210~12330)가 남아 있으므로, 구 단위 표는 그 코드까지 받아
전처리에서 광주·전남을 복원한다.

사용법:
  python collect/collect_kosis.py                  # 증분 (미수집분 + 최근 1년)
  python collect/collect_kosis.py --full           # 전 기간 재수집 (표 정의를 바꿨을 때)
  python collect/collect_kosis.py DT_1B040A3 ...   # 지정 표만
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    SEOUL_GU, http_get_json, require_key, save_raw, raw_max_value, year_chunks,
)

API_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
API_KEY = require_key("KOSIS_API_KEY")
THIS_YEAR = date.today().year

# 재수집 시 최근 N개월(분기/반기 포함)은 확정치 반영을 위해 다시 받는다.
REFETCH_BUFFER_YEARS = 1

KEY_FIELDS = ["TBL_ID", "PRD_DE", "C1", "C2", "C3", "C4", "ITM_ID"]

# 행정표준코드 기준 요청 지역: 전국·시도 (과거 강원 42 / 전북 45 코드 포함)
SIDO_CODES = ["00", "11", "12", "26", "27", "28", "29", "30", "31", "36", "41",
              "42", "43", "44", "45", "46", "47", "48", "50", "51", "52"]
# 통합 후 광주 5개 구 (전남광주통합특별시 하위코드) - 광주·전남 복원용
GWANGJU_GU_2026 = ["12210", "12240", "12270", "12300", "12330"]
# 20~39세 1세 코드: 20~24세 1201~1205 / 25~29세 1301~ / 30~34세 1501~ / 35~39세 1601~
AGE_20_39 = [f"{band}{i:02d}" for band in ("12", "13", "15", "16") for i in range(1, 6)]


def plus(codes):
    return "+".join(codes) + "+"


# chunk = 요청 1회당 연수 (셀 4만개 제한에 맞춤, 기본 3)
TABLES = [
    # --- 주택건설실적: 주택유형 소분류 '아파트'만 ---
    dict(tbl="DT_MLTM_1948", org="116", chunk=1, itm="13103871090T1+", prd="M", start=2007, end=None,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL",
              "objL4": "13102871090D.0009+"},
         desc="인허가실적_아파트"),
    dict(tbl="DT_MLTM_5387", org="116", chunk=1, itm="13103766969T1+", prd="M", start=2007, end=None,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL",
              "objL4": "13102766969D.0008+"},
         desc="착공실적_아파트"),
    dict(tbl="DT_MLTM_5373", org="116", chunk=1, itm="13103766973T1+", prd="M", start=2007, end=None,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL",
              "objL4": "13102766973D.0008+"},
         desc="준공실적_아파트"),
    # --- 주민등록 (행안부 원자료를 KOSIS가 재공표) ---
    dict(tbl="DT_1B040B3", org="101", itm="T1+", prd="M", start=2011, end=None,
         obj={"objL1": "ALL"}, desc="주민등록세대수"),
    dict(tbl="DT_1B040A3", org="101", itm="T20+", prd="M", start=2011, end=None,
         obj={"objL1": "ALL"}, desc="주민등록인구"),
    dict(tbl="DT_1B04006", org="101", itm="T2+", prd="M", start=2011, end=None, chunk=2,
         obj={"objL1": plus(SIDO_CODES + GWANGJU_GU_2026 + list(SEOUL_GU)),
              "objL2": plus(["000"] + AGE_20_39)},
         desc="주민등록인구_20-39세_1세별"),
    dict(tbl="DT_1B26001_A01", org="101", itm="T25+", prd="M", start=2011, end=None,
         obj={"objL1": "ALL"}, desc="인구이동_순이동"),
    # --- 고용 (V013: 시도 월별 + 서울 구 반기) ---
    dict(tbl="DT_1DA7004S", org="101", itm="T30+T90+", prd="M",
         start=2011, end=None, obj={"objL1": "ALL"}, desc="고용률취업자_시도_월"),
    dict(tbl="DT_1ES3A01S", org="101", itm="T3+T7+", prd="H",
         start=2021, end=None, obj={"objL1": "ALL"}, desc="고용률취업자_시군구_반기"),
    # --- 지역소득 (연간) ---
    dict(tbl="DT_1C96", org="101", itm="T1+T3+", prd="Y", start=2010, end=None,
         obj={"objL1": "ALL"}, desc="1인당GRDP_가계총처분가능소득"),
    # --- 공급·재고 (연간) ---
    dict(tbl="DT_201004_O090005", org="201", itm="T001+T002+", prd="Y", start=2010, end=None,
         obj={"objL1": "ALL", "objL2": "002+", "objL3": "001001+", "objL4": "001+"},
         desc="서울구_주거용_신축허가"),
    dict(tbl="DT_1IN1502", org="101", itm="T200+T310+", prd="Y", start=2015,
         end=None, chunk=1, obj={"objL1": "ALL"}, desc="총조사_가구주택"),
    dict(tbl="DT_MLTM_5560", org="116", itm="ALL", prd="Y", start=2012, end=2019, chunk=1,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "13102866436C.0001+",
              "objL4": "13102866436D.0001+", "objL5": "ALL"},
         # 분류2·3은 총계만 (전체 조합은 4만셀 초과). 분류1(유형)·분류4는 전부
         desc="임대주택건설공급현황_2012-2019"),
    dict(tbl="DT_MLTM_6827", org="116", itm="ALL", prd="Y", start=2020, end=None, chunk=1,
         obj={"objL1": "ALL", "objL2": "ALL"}, desc="민간임대주택공급현황_2020-"),
    dict(tbl="DT_MLTM_7174", org="116", itm="ALL", prd="Y", start=2024, end=None, chunk=1,
         obj={"objL1": "ALL", "objL2": "ALL", "objL3": "ALL"}, desc="임대주택공급현황_2024-"),
    # --- 가계동향 (분기, 전국): V074는 사전 수록기간대로 구계열(1990~2019)만 ---
    dict(tbl="DT_1L9I002", org="101", itm="T1+", prd="Q", start=1990, end=2019,
         obj={"objL1": "A+"}, desc="가계동향_도시2인이상_소득_구계열"),
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


def fetch_table(t, full=False):
    end_year = t["end"] or THIS_YEAR
    start_year = t["start"]

    # 종료 연도가 고정된 과거 표는 이미 받아뒀으면 건너뛴다 (원본이 더 이상 변하지 않음).
    last_prd = None if full else raw_max_value("kosis", t["tbl"], "PRD_DE")
    if t["end"] is not None and last_prd:
        print(f"- {t['tbl']} ({t['desc']}): 이미 수집 완료 (~{last_prd}), 건너뜀")
        return
    if last_prd:
        resume_year = max(start_year, int(last_prd[:4]) - REFETCH_BUFFER_YEARS)
        start_year = resume_year

    collected = []
    for cs, ce in year_chunks(start_year, end_year, years_per_chunk=t.get("chunk", 3)):
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
    args = sys.argv[1:]
    full = "--full" in args
    wanted = {a for a in args if not a.startswith("--")}
    print(f"[수집] KOSIS -> raw/<동인 폴더>/ (위치: raw_layout.py) ({'전 기간' if full else '증분'})")
    for t in TABLES:
        if wanted and t["tbl"] not in wanted:
            continue
        fetch_table(t, full=full)
    print("완료")


if __name__ == "__main__":
    main()
