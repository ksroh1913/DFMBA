# -*- coding: utf-8 -*-
"""
[수집] 한국부동산원 R-ONE Open API -> raw/rone/<STATBL_ID>.csv

응답 행(CLS_FULLNM, ITM_NM, DTA_VAL 등)을 그대로 저장한다.
지역 계층 파싱·지수 환산 등은 preprocess/ 단계에서 수행.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    http_get, require_key, save_raw, raw_max_value, shift_ym,
)

import json  # noqa: E402
import urllib.parse  # noqa: E402

API_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"
API_KEY = require_key("RONE_API_KEY")
PAGE_SIZE = 1000  # R-ONE 1회 최대 요청 건수
REFETCH_BUFFER_MONTHS = 2

KEY_FIELDS = ["STATBL_ID", "WRTTIME_IDTFR_ID", "GRP_ID", "CLS_ID", "ITM_ID"]

TODAY_YM = date.today().strftime("%Y%m")

# (STATBL_ID, 설명, 시작연월, 종료연월 None=현재진행)
TABLES = [
    ("A_2024_00054", "월세통합가격지수_아파트", "201501", None),
    ("A_2024_00055", "월세가격지수_아파트(신)", "201601", None),
    ("A_2024_00164", "주택유형별 월세가격지수(구)_아파트", "201001", "201512"),
    ("A_2024_00045", "매매가격지수_아파트", "200301", None),
    ("A_2024_00049", "전월세통합지수_아파트", "201501", None),
    ("A_2024_00050", "전세가격지수_아파트", "200301", None),
    ("A_2024_00156", "지역별 전월세 전환율_아파트", "201101", None),
    ("A_2024_00076", "매매수급동향_아파트", "201201", None),
    ("A_2024_00077", "전세수급동향_아파트", "201201", None),
    ("A_2024_00078", "월세수급동향_아파트", "201501", None),
    ("A_2024_00549", "행정구역별 아파트거래현황", "200601", None),
    ("A_2024_00554", "행정구역별 아파트매매거래현황", "200601", None),
    ("T232543129897499", "주택시장 소비심리지수", "201101", None),
    ("T237973129847263", "미분양주택현황", "200701", None),
    ("T242093131851096", "지역별 수급동향(구)", "201001", "201512"),
]


def fetch_all_rows(statbl_id, start_wrttime, end_wrttime):
    """지정 기간 데이터를 페이지네이션하며 모두 가져온다."""
    rows = []
    p_index = 1
    while True:
        params = {
            "KEY": API_KEY, "STATBL_ID": statbl_id, "DTACYCLE_CD": "MM",
            "START_WRTTIME": start_wrttime, "END_WRTTIME": end_wrttime,
            "Type": "json", "pSize": PAGE_SIZE, "pIndex": p_index,
        }
        text = http_get(API_URL + "?" + urllib.parse.urlencode(params))
        data = json.loads(text)
        top = data.get("SttsApiTblData")
        if not top:
            break  # RESULT만 반환 = 데이터 없음
        head = top[0]["head"]
        total = head[0]["list_total_count"]
        if head[1]["RESULT"]["CODE"] != "INFO-000" or total == 0:
            break
        page_rows = top[1]["row"]
        rows.extend(page_rows)
        if p_index * PAGE_SIZE >= total or not page_rows:
            break
        p_index += 1
    return rows


def fetch_table(statbl_id, desc, start, end):
    last = raw_max_value("rone", statbl_id, "WRTTIME_IDTFR_ID")

    if end is not None and last and last >= end:
        print(f"- {statbl_id} ({desc}): 이미 수집 완료 (~{last}), 건너뜀")
        return

    fetch_start = start
    if last:
        candidate = shift_ym(last, -REFETCH_BUFFER_MONTHS)
        if candidate > fetch_start:
            fetch_start = candidate
    fetch_end = end or TODAY_YM

    rows = fetch_all_rows(statbl_id, fetch_start, fetch_end)
    total = save_raw("rone", statbl_id, rows, KEY_FIELDS)
    print(f"- {statbl_id} ({desc}): {fetch_start}~{fetch_end} {len(rows)}행 수신, raw 누적 {total}행")


def main():
    print("[수집] R-ONE -> raw/rone/")
    for statbl_id, desc, start, end in TABLES:
        fetch_table(statbl_id, desc, start, end)
    print("완료")


if __name__ == "__main__":
    main()
