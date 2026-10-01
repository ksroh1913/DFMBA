# -*- coding: utf-8 -*-
"""
국토교통부 아파트 전월세 실거래가 Open API 취합 스크립트 - 서울 25개 자치구

갱신계약(contractType='갱신')은 제외하고 신규계약만 집계한다.
(2021.06 이전 자료는 계약구분 필드 자체가 없어 구분 불가 - 전부 포함하고 비고에 표시)
전세(월세=0)와 월세를 구분해 구·월별로 집계.

산출물: 서울_아파트_전월세_실거래가_월별집계.csv
  컬럼: 광역지자체, 자치구, 연월,
        전세건수, 전세평균보증금(만원), 전세중위보증금(만원),
        월세건수, 월세평균보증금(만원), 월세평균월세(만원),
        평균전용면적(㎡), 계약구분기록여부
"""

import csv
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent"
OUT_CSV_NAME = "서울_아파트_전월세_실거래가_월별집계.csv"

START_YM = "200601"
REFETCH_BUFFER_MONTHS = 3


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
    items = []
    page_no = 1
    while True:
        xml_text = call_api(lawd_cd, deal_ymd, page_no)
        root = ET.fromstring(xml_text)
        result_code = root.findtext("./header/resultCode")
        if result_code != "000":
            msg = root.findtext("./header/resultMsg")
            if result_code in ("03", "004"):
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
    jeonse_deposits, jeonse_areas = [], []
    wolse_deposits, wolse_rents, wolse_areas = [], [], []
    has_contract_type_field = False

    for it in items:
        contract_type = it.get("contractType", "")
        if contract_type:
            has_contract_type_field = True
        if contract_type == "갱신":
            continue  # 갱신계약 제외, 신규계약만 집계 (구분 없는 과거 자료는 포함)

        try:
            deposit = float(it["deposit"].replace(",", ""))
            area = float(it["excluUseAr"])
            monthly_rent = float(it.get("monthlyRent", "0").replace(",", "") or 0)
        except (KeyError, ValueError):
            continue

        if monthly_rent > 0:
            wolse_deposits.append(deposit)
            wolse_rents.append(monthly_rent)
            wolse_areas.append(area)
        else:
            jeonse_deposits.append(deposit)
            jeonse_areas.append(area)

    if not jeonse_deposits and not wolse_deposits:
        return None

    def median(vals):
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return ""
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

    all_areas = jeonse_areas + wolse_areas
    return {
        "전세건수": len(jeonse_deposits),
        "전세평균보증금(만원)": round(sum(jeonse_deposits) / len(jeonse_deposits), 1) if jeonse_deposits else "",
        "전세중위보증금(만원)": round(median(jeonse_deposits), 1) if jeonse_deposits else "",
        "월세건수": len(wolse_deposits),
        "월세평균보증금(만원)": round(sum(wolse_deposits) / len(wolse_deposits), 1) if wolse_deposits else "",
        "월세평균월세(만원)": round(sum(wolse_rents) / len(wolse_rents), 1) if wolse_rents else "",
        "평균전용면적(㎡)": round(sum(all_areas) / len(all_areas), 2) if all_areas else "",
        "계약구분기록여부": "Y" if has_contract_type_field else "N",
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
    rows = {}
    max_ym_by_gu = {}
    if not os.path.exists(path):
        return rows, max_ym_by_gu
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            gu, ym = row[1], row[2].replace("-", "")
            rows[(gu, ym)] = row
            if ym > max_ym_by_gu.get(gu, ""):
                max_ym_by_gu[gu] = ym
    return rows, max_ym_by_gu


HEADER = ["광역지자체", "자치구", "연월", "전세건수", "전세평균보증금(만원)", "전세중위보증금(만원)",
          "월세건수", "월세평균보증금(만원)", "월세평균월세(만원)", "평균전용면적(㎡)", "계약구분기록여부"]


def write_csv(out_path, merged):
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for (gu, ym), row in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            w.writerow(row)


def main():
    t0 = time.time()
    today_ym = date.today().strftime("%Y%m")
    out_path = os.path.join(BASE_DIR, OUT_CSV_NAME)
    existing_rows, max_ym_by_gu = load_existing(out_path)
    if existing_rows:
        print(f"기존 결과 {len(existing_rows)}행 로드 (증분 수집 모드)", flush=True)

    merged = dict(existing_rows)
    new_count = 0
    gu_list = list(SEOUL_GU.items())

    for i, (gu_code, gu_name) in enumerate(gu_list, 1):
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
                agg["전세건수"], agg["전세평균보증금(만원)"], agg["전세중위보증금(만원)"],
                agg["월세건수"], agg["월세평균보증금(만원)"], agg["월세평균월세(만원)"],
                agg["평균전용면적(㎡)"], agg["계약구분기록여부"],
            ]
            merged[(gu_name, ym)] = row
            gu_new += 1
        new_count += gu_new

        # 구 단위로 즉시 저장 -> 중단돼도 데이터 보존 + 진행률을 CSV로 바로 확인 가능
        write_csv(out_path, merged)

        elapsed = time.time() - t0
        avg_per_gu = elapsed / i
        remaining = avg_per_gu * (len(gu_list) - i)
        print(f"- [{i}/{len(gu_list)}] {gu_name}: {len(months)}개월 조회, {gu_new}개월 갱신 "
              f"| 경과 {elapsed/60:.1f}분, 예상잔여 {remaining/60:.1f}분", flush=True)

    print(f"\n완료: {out_path} (전체 {len(merged)}행, 신규/갱신 {new_count}행)", flush=True)
    print("※ 갱신계약(contractType='갱신')은 제외, 신규계약만 집계.")
    print("  계약구분기록여부='N'인 행은 원본에 구분 필드가 없어(2021.06 이전 등) 전량 포함된 것.")
    print("※ 서울 25개 자치구만 취합됨.")


if __name__ == "__main__":
    main()
