# -*- coding: utf-8 -*-
"""
[수집] 전처리 보완용 원자료 (데이터사전 변수 아님) -> raw/<동인 폴더>/ (위치: raw_layout.py)<이름>.csv

데이터사전(Variable_Master) 변수의 빈칸을 채우는 데만 쓰고, 그 자체로는 변수가 아니다.
그래서 사전 변수 raw와 섞이지 않게 raw/7_보완용_사전외/ 에 둔다.

  이름                         출처                              보완 대상
  rone_월세가격지수_구          R-ONE A_2024_00164 (2010.06~2015.06) V001 2015.06 이전 연결
  kosis_가계동향_신계열         KOSIS DT_1L9V121 (2019Q1~)         V074 2020년 이후 연결
  ecos_주택상가가치전망CSI      ECOS 511Y002 FMDE (2008.07~2012.12) V061~V063 2013.01 이전
  kosis_임금총액_산업9차        KOSIS 118/DT_118N_MON041 (2011.01~2019.12) V066 2020.01 이전 연결
  kosis_주택건설_소분류_<표ID>  KOSIS DT_MLTM_1948/5387/5373 전체 소분류
                                -> 합계와 다른 주택유형으로 아파트 빈 달이 0인지 검증 (V021~V023)

사용법: python collect/collect_supplement.py
"""

import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    http_get, http_get_json, month_range, require_key, save_raw, year_chunks,
)

KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
RONE_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"
ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
KOSIS_KEYS = ["TBL_ID", "PRD_DE", "C1", "C2", "C3", "C4", "C5", "ITM_ID"]


def kosis(name, org, tbl, itm, prd, y0, y1, obj, chunk=1, split=None):
    """split=(obj키, [코드...]) 이면 코드마다 따로 요청 (조합이 많아 셀 4만 제한을 넘는 표)"""
    key = require_key("KOSIS_API_KEY")
    rows = []
    variants = [dict(obj, **{split[0]: c + "+"}) for c in split[1]] if split else [obj]
    for ob in variants:
        rows += _kosis_range(key, org, tbl, itm, prd, y0, y1, ob, chunk)
    n = save_raw("supplement", name, rows, KOSIS_KEYS)
    print(f"- {name}: {len(rows)}행 수신, raw {n}행")


def _kosis_range(key, org, tbl, itm, prd, y0, y1, obj, chunk):
    rows = []
    for a, b in year_chunks(y0, y1, chunk):
        s, e = (str(a), str(b)) if prd == "Y" else ((f"{a}01", f"{b}04") if prd == "Q" else (f"{a}01", f"{b}12"))
        p = {"method": "getList", "apiKey": key, "format": "json", "jsonVD": "Y", "orgId": org,
             "tblId": tbl, "itmId": itm, "prdSe": prd, "startPrdDe": s, "endPrdDe": e, **obj}
        d = http_get_json(KOSIS_URL, p)
        if isinstance(d, dict):
            if d.get("err") == "30":
                continue
            raise RuntimeError(f"[{tbl} {a}-{b}] {d}")
        rows += d
    return rows


def rone(name, statbl, start, end):
    key = require_key("RONE_API_KEY")
    rows = []
    # 1,000행 이하 창으로 나눠 받는다 (페이지 넘김 누락 방지, collect_rone.py 와 같은 이유)
    months = month_range(start, end)
    for i in range(0, len(months), 6):
        p = {"KEY": key, "STATBL_ID": statbl, "DTACYCLE_CD": "MM", "START_WRTTIME": months[i],
             "END_WRTTIME": months[min(i + 5, len(months) - 1)], "Type": "json", "pSize": 1000, "pIndex": 1}
        d = json.loads(http_get(RONE_URL + "?" + urllib.parse.urlencode(p)))
        top = d.get("SttsApiTblData")
        if not top:
            continue
        total = top[0]["head"][0]["list_total_count"]
        part = top[1]["row"]
        if len(part) != total:
            raise RuntimeError(f"[{statbl} {months[i]}] 행수 불일치 {len(part)}/{total}")
        rows += part
    n = save_raw("supplement", name, rows, ["STATBL_ID", "WRTTIME_IDTFR_ID", "GRP_ID", "CLS_ID", "ITM_ID"])
    print(f"- {name}: {len(rows)}행 수신, raw {n}행")


def ecos(name, stat, cycle, start, end, item_sets):
    key = require_key("ECOS_API_KEY")
    rows = []
    for items in item_sets:
        url = f"{ECOS_URL}/{key}/json/kr/1/1000/{stat}/{cycle}/{start}/{end}/" + "/".join(items)
        rows += json.loads(http_get(url)).get("StatisticSearch", {}).get("row", [])
    n = save_raw("supplement", name, rows, ["STAT_CODE", "TIME", "ITEM_CODE1", "ITEM_CODE2", "ITEM_CODE3"])
    print(f"- {name}: {len(rows)}행 수신, raw {n}행")


def main():
    print("[수집] 보완용 원자료 -> raw/<동인 폴더>/ (위치: raw_layout.py)")
    rone("rone_월세가격지수_구", "A_2024_00164", "201006", "201506")
    kosis("kosis_가계동향_신계열", "101", "DT_1L9V121", "T210+", "Q", 2019, 2026, {"objL1": "A+"}, chunk=8)
    ecos("ecos_주택상가가치전망CSI", "511Y002", "M", "200807", "201212",
         [["FMDE", "F0001"], ["FMDE", "F0002"], ["FMDE", "F0003"]])
    # 사업체노동력조사 산업분류 9차 표: 전체 산업 · 전규모(1인이상) · 전체임금총액
    kosis("kosis_임금총액_산업9차", "118", "DT_118N_MON041", "13103110311MD_12+", "M", 2011, 2019,
          {"objL1": "15118INDUSTRY_9S0+", "objL2": "size01+"}, chunk=9)
    # 소분류: 합계(동수기준), 단독, 다가구 동수, 다세대, 연립, 아파트
    for tbl, itm, pre, codes in [
            ("DT_MLTM_1948", "13103871090T1+", "13102871090D.", ["0001", "0003", "0004", "0007", "0008", "0009"]),
            ("DT_MLTM_5387", "13103766969T1+", "13102766969D.", ["0001", "0003", "0004", "0006", "0007", "0008"]),
            ("DT_MLTM_5373", "13103766973T1+", "13102766973D.", ["0001", "0003", "0004", "0006", "0007", "0008"])]:
        kosis(f"kosis_주택건설_소분류_{tbl}", "116", tbl, itm, "M", 2011, 2026,
              {"objL1": "ALL", "objL2": "ALL", "objL3": "ALL"}, chunk=1,
              split=("objL4", [pre + c for c in codes]))
    print("완료")


if __name__ == "__main__":
    main()
