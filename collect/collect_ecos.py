# -*- coding: utf-8 -*-
"""
[수집] 한국은행 ECOS Open API -> raw/ecos/<시리즈명>.csv

아파트 전월세 가격에 영향을 줄 수 있는 거시지표를 시리즈 단위로 저장한다.
응답 행(TIME, DATA_VALUE, ITEM_NAME1 등)을 그대로 보존.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import http_get, require_key, save_raw  # noqa: E402

import json  # noqa: E402

API_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"
API_KEY = require_key("ECOS_API_KEY")

START_M, END_M = "199001", "202612"
START_Q, END_Q = "1990Q1", "2026Q4"

KEY_FIELDS = ["STAT_CODE", "TIME", "ITEM_CODE1", "ITEM_CODE2"]

# 지역별 주택관련대출(151Y003)의 지역 항목코드
REGION_ITEMS = {
    "서울": "A00", "부산": "B00", "대구": "C00", "인천": "D00", "대전": "E00",
    "광주": "F00", "울산": "G00", "세종": "H00", "경기": "L00", "강원": "M00",
    "충북": "N00", "충남": "P00", "전북": "Q00", "전남": "R00", "경북": "S00",
    "경남": "T00", "제주": "U00",
}

# (raw 파일명, STAT_CODE, 주기, 시작, 종료, 항목코드들)
SERIES = [
    ("기준금리", "722Y001", "M", START_M, END_M, ["0101000"]),
    ("시장금리_CD91일", "721Y001", "M", START_M, END_M, ["2010000"]),
    ("시장금리_국고채3년", "721Y001", "M", START_M, END_M, ["5020000"]),
    ("M2_통화량", "161Y006", "M", START_M, END_M, ["BBHA00"]),
    ("소비자물가지수", "901Y009", "M", START_M, END_M, ["0"]),
    ("가계신용_총액", "151Y001", "Q", START_Q, END_Q, ["1000000"]),
    ("시간당명목임금지수", "901Y102", "Q", START_Q, END_Q, ["A10001"]),
]


def fetch_series(stat_code, cycle, start, end, item_codes, count=1000):
    items = "/".join(item_codes)
    url = f"{API_BASE}/{API_KEY}/json/kr/1/{count}/{stat_code}/{cycle}/{start}/{end}/{items}"
    data = json.loads(http_get(url))
    result = data.get("StatisticSearch")
    if not result or "row" not in result:
        return []
    return result["row"]


def main():
    print("[수집] ECOS -> raw/ecos/")

    for name, stat_code, cycle, start, end, items in SERIES:
        rows = fetch_series(stat_code, cycle, start, end, items)
        total = save_raw("ecos", name, rows, KEY_FIELDS)
        print(f"- {name}: {len(rows)}행 수신, raw 누적 {total}행")

    # 지역별 주택관련대출(예금은행) - 지역코드를 순회해 한 파일에 모음
    all_rows = []
    for region, item2 in REGION_ITEMS.items():
        rows = fetch_series("151Y003", "M", START_M, END_M, ["11110A0", item2])
        all_rows.extend(rows)
    total = save_raw("ecos", "지역별_주택관련대출_예금은행", all_rows, KEY_FIELDS)
    print(f"- 지역별_주택관련대출_예금은행: {len(all_rows)}행 수신, raw 누적 {total}행")

    print("완료")


if __name__ == "__main__":
    main()
