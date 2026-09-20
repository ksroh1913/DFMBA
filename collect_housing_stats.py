# -*- coding: utf-8 -*-
"""
광역지자체별 주택(아파트) 인허가·착공·준공 실적 월별 취합 스크립트
데이터 출처: KOSIS(국가통계포털) Open API - 국토교통부 주택건설실적통계
주의: 이 통계는 광역지자체(시도) 단위까지만 제공되며, 자치구별 데이터는 존재하지 않음.
"""

import csv
import json
import os
import re
import time
import urllib.request
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
API_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


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
API_KEY = ENV.get("KOSIS_API_KEY") or os.environ.get("KOSIS_API_KEY")
if not API_KEY:
    raise SystemExit("KOSIS_API_KEY가 .env에 없습니다.")

START_PRD = "200701"  # 데이터 시작 시점
END_PRD = "202612"  # 필요시 조정 (미래 시점은 응답에 자동으로 없음)

# 취합 대상 광역지자체 (전국 총계/수도권소계 등 집계행은 제외)
REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

# 3개 통계표 정의: 소분류 코드(objL4)=아파트, 광역지자체(objL1)=ALL
TABLES = {
    "인허가": {
        "tblId": "DT_MLTM_1948",
        "itmId": "13103871090T1",
        "objL4": "13102871090D.0009",  # 아파트
        "cumulative": True,  # "월별 누계" 통계이므로 월별 실적으로 환산 필요
    },
    "착공": {
        "tblId": "DT_MLTM_5387",
        "itmId": "13103766969T1",
        "objL4": "13102766969D.0008",
        "cumulative": False,
    },
    "준공": {
        "tblId": "DT_MLTM_5373",
        "itmId": "13103766973T1",
        "objL4": "13102766973D.0008",
        "cumulative": False,
    },
}


def call_api(params, retries=3):
    query = urllib.parse.urlencode(params, safe="+")
    url = API_URL + "?" + query
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def year_chunks(start_prd, end_prd, years_per_chunk=3):
    start_year = int(start_prd[:4])
    end_year = int(end_prd[:4])
    chunks = []
    y = start_year
    while y <= end_year:
        chunk_end_year = min(y + years_per_chunk - 1, end_year)
        chunks.append((f"{y}01", f"{chunk_end_year}12"))
        y += years_per_chunk
    return chunks


def build_params(table_def, start_prd, end_prd):
    return {
        "method": "getList",
        "apiKey": API_KEY,
        "itmId": table_def["itmId"] + "+",
        "objL1": "ALL",
        "objL2": "ALL",
        "objL3": "ALL",
        "objL4": table_def["objL4"],
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": start_prd,
        "endPrdDe": end_prd,
        "orgId": "116",
        "tblId": table_def["tblId"],
    }


def fetch_table(name, table_def):
    """지역명 -> {연월: 값} 딕셔너리를 반환"""
    by_region = {}
    for s, e in year_chunks(START_PRD, END_PRD):
        data = call_api(build_params(table_def, s, e))
        if not isinstance(data, list):
            if isinstance(data, dict) and data.get("err") == "30":
                continue  # 해당 구간에 데이터 없음
            raise RuntimeError(f"[{name} {s}-{e}] KOSIS 응답 오류: {data}")
        for row in data:
            if row["C4_NM"] != "아파트":
                continue
            if row["C1_NM"] not in REGIONS:
                continue  # 총계/소계 등 제외
            by_region.setdefault(row["C1_NM"], {})[row["PRD_DE"]] = float(row["DT"])
    return by_region


def to_monthly_from_cumulative(cum_map):
    monthly = {}
    for ym in sorted(cum_map.keys()):
        year, month = ym[:4], ym[4:6]
        cur = cum_map[ym]
        if month == "01":
            monthly[ym] = cur
        else:
            prev_ym = f"{year}{int(month) - 1:02d}"
            prev = cum_map.get(prev_ym)
            monthly[ym] = None if prev is None else cur - prev
    return monthly


def main():
    print("KOSIS에서 광역지자체별 아파트 인허가/착공/준공 실적(월별)을 수집합니다...")

    # results[통계명] = {지역: {연월: 값}}
    results = {}
    for name, table_def in TABLES.items():
        by_region = fetch_table(name, table_def)
        if table_def["cumulative"]:
            by_region = {region: to_monthly_from_cumulative(m) for region, m in by_region.items()}
        results[name] = by_region
        print(f"- {name}: {len(by_region)}개 지역 수집 완료")

    # 전체 연월 집합
    all_months = set()
    for by_region in results.values():
        for m in by_region.values():
            all_months.update(m.keys())
    sorted_months = sorted(all_months)

    out_path = os.path.join(BASE_DIR, "광역지자체_아파트_인허가착공준공_월별.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["광역지자체", "자치구", "연월", "인허가(호)", "착공(호)", "준공(호)"])
        for region in REGIONS:
            for ym in sorted_months:
                permit = results["인허가"].get(region, {}).get(ym)
                start = results["착공"].get(region, {}).get(ym)
                done = results["준공"].get(region, {}).get(ym)
                if permit is None and start is None and done is None:
                    continue
                fmt = lambda v: "" if v is None else int(v) if float(v).is_integer() else v
                w.writerow([region, "", f"{ym[:4]}-{ym[4:6]}", fmt(permit), fmt(start), fmt(done)])

    print(f"\n완료: {out_path}")
    print("※ 자치구별 데이터는 KOSIS/data.go.kr에서 제공되지 않아 광역지자체(시도) 단위로 취합되었습니다.")


if __name__ == "__main__":
    main()
