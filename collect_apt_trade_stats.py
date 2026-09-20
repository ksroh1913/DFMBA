# -*- coding: utf-8 -*-
"""
국토교통부 아파트 매매 실거래가(상세자료) Open API 취합 스크립트 - 서울 25개 자치구

건별 원자료(연간 수만 건)를 그대로 쌓지 않고, 자치구·월별로 바로 집계하여 저장한다.
기존 스크립트들과 동일하게 "광역지자체, 자치구, 연월" 키를 사용하며,
재실행 시 마지막 수집월 이후만 다시 받는 증분 수집 방식을 사용한다.

산출물: 서울_아파트_실거래가_월별집계.csv
  컬럼: 광역지자체, 자치구, 연월, 거래건수, 평균거래금액(만원), 중위거래금액(만원),
        평균전용면적(㎡), 평균단가(만원_전용㎡당)
"""

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
OUT_CSV_NAME = "서울_아파트_실거래가_월별집계.csv"

START_YM = "200601"
REFETCH_BUFFER_MONTHS = 3  # 실거래 신고 지연(계약 후 최대 30일 등)을 감안해 최근 N개월 재수집


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
API_KEY = ENV.get("DATA_GO_KR_API_KEY") or os.environ.get("DATA_GO_KR_API_KEY")
if not API_KEY:
    raise SystemExit("DATA_GO_KR_API_KEY가 .env에 없습니다.")

SEOUL_GU = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}


def call_api(lawd_cd, deal_ymd, page_no, num_of_rows=1000, retries=3):
    params = {
        "serviceKey": API_KEY,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as res:
                return res.read().decode("utf-8")
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def fetch_month(lawd_cd, deal_ymd):
    """해당 시군구·계약년월의 전체 거래 item 목록(dict list)을 페이지네이션하며 가져온다."""
    items = []
    page_no = 1
    while True:
        xml_text = call_api(lawd_cd, deal_ymd, page_no)
        root = ET.fromstring(xml_text)
        result_code = root.findtext("./header/resultCode")
        if result_code != "000":
            msg = root.findtext("./header/resultMsg")
            if result_code in ("03", "004"):  # 데이터 없음류
                break
            raise RuntimeError(f"[{lawd_cd} {deal_ymd}] API 오류 {result_code}: {msg}")
        body_items = root.findall("./body/items/item")
        for item in body_items:
            items.append({child.tag: (child.text or "").strip() for child in item})
        total_count = int(root.findtext("./body/totalCount") or 0)
        if page_no * 1000 >= total_count or not body_items:
            break
        page_no += 1
    return items


def aggregate_month(items):
    amounts = []
    areas = []
    unit_prices = []
    for it in items:
        try:
            amt = float(it["dealAmount"].replace(",", ""))  # 만원
            area = float(it["excluUseAr"])
        except (KeyError, ValueError):
            continue
        amounts.append(amt)
        areas.append(area)
        if area > 0:
            unit_prices.append(amt / area)

    if not amounts:
        return None

    amounts.sort()
    n = len(amounts)
    median = amounts[n // 2] if n % 2 == 1 else (amounts[n // 2 - 1] + amounts[n // 2]) / 2

    return {
        "거래건수": n,
        "평균거래금액(만원)": round(sum(amounts) / n, 1),
        "중위거래금액(만원)": round(median, 1),
        "평균전용면적(㎡)": round(sum(areas) / len(areas), 2) if areas else "",
        "평균단가(만원_전용㎡당)": round(sum(unit_prices) / len(unit_prices), 2) if unit_prices else "",
    }


def month_range(start_ym, end_ym):
    y, m = int(start_ym[:4]), int(start_ym[4:6])
    ey, em = int(end_ym[:4]), int(end_ym[4:6])
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def shift_ym(ym, months):
    y, m = int(ym[:4]), int(ym[4:6])
    total = y * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}{nm + 1:02d}"


def load_existing(path):
    rows = {}  # (구, 연월) -> row(list)
    max_ym_by_gu = {}
    if not os.path.exists(path):
        return rows, max_ym_by_gu
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            gu, ym = row[1], row[2].replace("-", "")
            rows[(gu, ym)] = row
            if ym > max_ym_by_gu.get(gu, ""):
                max_ym_by_gu[gu] = ym
    return rows, max_ym_by_gu


def main():
    today_ym = date.today().strftime("%Y%m")
    out_path = os.path.join(BASE_DIR, OUT_CSV_NAME)
    existing_rows, max_ym_by_gu = load_existing(out_path)
    if existing_rows:
        print(f"기존 결과 {len(existing_rows)}행 로드 (증분 수집 모드)")

    merged = dict(existing_rows)
    new_count = 0

    for gu_code, gu_name in SEOUL_GU.items():
        last_ym = max_ym_by_gu.get(gu_name)
        start = START_YM
        if last_ym:
            candidate = shift_ym(last_ym, -REFETCH_BUFFER_MONTHS)
            if candidate > start:
                start = candidate
        months = month_range(start, today_ym)

        gu_new = 0
        for ym in months:
            items = fetch_month(gu_code, ym)
            agg = aggregate_month(items)
            if agg is None:
                continue
            row = [
                "서울", gu_name, f"{ym[:4]}-{ym[4:6]}",
                agg["거래건수"], agg["평균거래금액(만원)"], agg["중위거래금액(만원)"],
                agg["평균전용면적(㎡)"], agg["평균단가(만원_전용㎡당)"],
            ]
            merged[(gu_name, ym)] = row
            gu_new += 1
        new_count += gu_new
        print(f"- {gu_name}: {len(months)}개월 조회, {gu_new}개월 갱신")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["광역지자체", "자치구", "연월", "거래건수", "평균거래금액(만원)",
                    "중위거래금액(만원)", "평균전용면적(㎡)", "평균단가(만원_전용㎡당)"])
        for (gu, ym), row in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            w.writerow(row)

    print(f"\n완료: {out_path} (전체 {len(merged)}행, 신규/갱신 {new_count}행)")
    print("※ 서울 25개 자치구만 취합됨. 다른 시도는 추후 별도 작업 필요.")


if __name__ == "__main__":
    main()
