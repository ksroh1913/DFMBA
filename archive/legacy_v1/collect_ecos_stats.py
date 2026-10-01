# -*- coding: utf-8 -*-
"""
한국은행 ECOS Open API 거시경제 지표 취합 스크립트

아파트 월세/전세 가격에 영향을 줄 수 있는 거시지표를 월별로 취합.
기존 collect_housing_stats.py / collect_population_stats.py / collect_realestate_stats.py 와
동일하게 "광역지자체, 자치구, 연월" 키를 사용하는 long format으로 저장.
전국 단위 지표는 광역지자체="전국"으로 표기.

취합 항목:
  - 한국은행 기준금리                (722Y001)
  - 시장금리: CD(91일), 국고채(3년)   (721Y001)
  - M2 통화량(평잔, 원계열)           (161Y006)
  - 소비자물가지수(총지수)            (901Y009)
  - 가계신용(총액, 분기)              (151Y001)
  - 지역별 주택관련대출(예금은행, 시도별) (151Y003)
  - 시간당 명목임금지수(분기)         (901Y102)

산출물: ECOS_거시경제_시계열.csv
  컬럼: 광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위
"""

import csv
import json
import os
import re
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
API_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"


def load_env(path):
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m:
                env[m.group(1)] = m.group(2)
    return env


ENV = load_env(ENV_PATH)
API_KEY = ENV.get("ECOS_API_KEY") or os.environ.get("ECOS_API_KEY")
if not API_KEY:
    raise SystemExit("ECOS_API_KEY가 .env에 없습니다.")

START_M = "199001"
END_M = "202612"
START_Q = "1990Q1"
END_Q = "2026Q4"

REGION_ITEMS = {
    "서울": "A00", "부산": "B00", "대구": "C00", "인천": "D00", "대전": "E00",
    "광주": "F00", "울산": "G00", "세종": "H00", "경기": "L00", "강원": "M00",
    "충북": "N00", "충남": "P00", "전북": "Q00", "전남": "R00", "경북": "S00",
    "경남": "T00", "제주": "U00",
}


def call_api(stat_code, cycle, start_time, end_time, item_codes, count=1000, retries=3):
    items = "/".join(item_codes)
    url = f"{API_BASE}/{API_KEY}/json/kr/1/{count}/{stat_code}/{cycle}/{start_time}/{end_time}/{items}"
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as res:
                data = json.loads(res.read().decode("utf-8"))
            return data
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def ym_from_time(t, cycle):
    """TIME 값을 연월(YYYY-MM)로 정규화. 분기(Q)는 분기 시작월로 변환."""
    if cycle == "M":
        return f"{t[:4]}-{t[4:6]}"
    if cycle == "Q":
        # 형식 예: 2012Q1
        year, q = t[:4], t[5]
        month = {"1": "01", "2": "04", "3": "07", "4": "10"}.get(q, "01")
        return f"{year}-{month}"
    if cycle == "A":
        return f"{t[:4]}-01"
    return t


def fetch_series(name, stat_code, cycle, start_time, end_time, item_codes, out_rows,
                  region=None, region_short=None):
    data = call_api(stat_code, cycle, start_time, end_time, item_codes)
    result = data.get("StatisticSearch")
    if not result or "row" not in result:
        print(f"  ! {name}: 데이터 없음 ({data})")
        return
    for row in result["row"]:
        ym = ym_from_time(row["TIME"], cycle)
        val = row["DATA_VALUE"]
        if val in (None, ""):
            continue
        sido = region_short if region_short else "전국"
        item_label = row.get("ITEM_NAME1") or name
        out_rows.append((sido, "", ym, name, item_label, float(val), row.get("UNIT_NAME", "")))


def main():
    print("ECOS에서 거시경제 지표(월별/분기별)를 수집합니다...")
    out_rows = []

    series_plan = [
        ("한국은행_기준금리", "722Y001", "M", START_M, END_M, ["0101000"]),
        ("시장금리_CD91일", "721Y001", "M", START_M, END_M, ["2010000"]),
        ("시장금리_국고채3년", "721Y001", "M", START_M, END_M, ["5020000"]),
        ("M2_통화량", "161Y006", "M", START_M, END_M, ["BBHA00"]),
        ("소비자물가지수", "901Y009", "M", START_M, END_M, ["0"]),
        ("가계신용_총액", "151Y001", "Q", START_Q, END_Q, ["1000000"]),
        ("시간당명목임금지수", "901Y102", "Q", START_Q, END_Q, ["A10001"]),
    ]

    for name, stat_code, cycle, s, e, items in series_plan:
        before = len(out_rows)
        fetch_series(name, stat_code, cycle, s, e, items, out_rows)
        print(f"- {name}: {len(out_rows) - before}행 수집")

    # 지역별 주택관련대출(예금은행, 시도별) - 서울 등 17개 시도
    before = len(out_rows)
    for region, item2 in REGION_ITEMS.items():
        fetch_series(
            "지역별_주택관련대출_예금은행", "151Y003", "M", START_M, END_M,
            ["11110A0", item2], out_rows, region_short=region,
        )
    print(f"- 지역별_주택관련대출_예금은행: {len(out_rows) - before}행 수집")

    out_path = os.path.join(BASE_DIR, "ECOS_거시경제_시계열.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"])
        for r in sorted(out_rows, key=lambda x: (x[3], x[0], x[1], x[2])):
            w.writerow(r)

    print(f"\n완료: {out_path} (총 {len(out_rows)}행)")
    print("※ 지역별 주택관련대출을 제외한 나머지 지표는 전국 단위(광역지자체='전국')입니다.")


if __name__ == "__main__":
    main()
