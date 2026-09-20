# -*- coding: utf-8 -*-
"""
서울 자치구 + 광역지자체 인구 관련 데이터 취합 스크립트 (KOSIS Open API)

기존 collect_housing_stats.js(아파트 인허가·착공·준공)와 동일한 스키마(광역지자체,자치구,연월)로
결과를 저장하여, 이후 머신러닝을 위한 데이터 취합 시 바로 병합(merge)할 수 있도록 함.

산출물:
  1) 주민등록세대수_월별.csv        - 서울 25개 자치구 + 나머지 16개 광역지자체, 월별
  2) 경제활동인구_시계열.csv         - 연간(~2010)/분기(2011~2012)/반기(2013~) 조합.
      반기 구간은 지역별로 커버리지 시작 시점이 다름:
        - 9개 도(경기/강원/충북/충남/전북/전남/경북/경남/제주): 2013년부터
        - 서울 및 6개 광역시(부산/대구/인천/광주/대전/울산): 2021년 상반기부터
        - 세종: 데이터 없음 (전 기간)
      즉 17개 시도 완전 패널이 아니며, 2013~2020년은 9개 도만 존재함.
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


def call_api(params, retries=3):
    query = urllib.parse.urlencode(params, safe="+")
    url = API_URL + "?" + query
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def year_chunks(start_year, end_year, years_per_chunk=2):
    chunks = []
    y = start_year
    while y <= end_year:
        end = min(y + years_per_chunk - 1, end_year)
        chunks.append((y, end))
        y += years_per_chunk
    return chunks


# ---------------------------------------------------------------------------
# 서울 25개 자치구 + 16개 광역지자체 (기존 인허가/착공/준공 스크립트와 동일 범위)
# ---------------------------------------------------------------------------
SIDO_CODE_TO_NAME = {
    "11": "서울",
    "26": "부산",
    "27": "대구",
    "28": "인천",
    "29": "광주",
    "30": "대전",
    "31": "울산",
    "36": "세종",
    "41": "경기",
    "43": "충북",
    "44": "충남",
    "46": "전남",
    "47": "경북",
    "48": "경남",
    "50": "제주",
    "51": "강원",
    "52": "전북",
}


def fmt_ym(prd_de):
    """PRD_DE(YYYYMM) -> YYYY-MM"""
    return f"{prd_de[:4]}-{prd_de[4:6]}"


# ---------------------------------------------------------------------------
# 1) 주민등록세대수 (행정구역(시군구)별, 월별) - DT_1B040B3
#    서울은 자치구(5자리 코드, 11xxx) 단위, 나머지는 시도(2자리 코드) 단위만 취합
# ---------------------------------------------------------------------------
def collect_household_count():
    print("[주민등록세대수] 수집 시작...")
    rows = []  # (광역지자체, 자치구, 연월, 세대수)
    for start_year, end_year in year_chunks(2011, 2026, years_per_chunk=3):
        params = {
            "method": "getList",
            "apiKey": API_KEY,
            "itmId": "T1+",
            "objL1": "ALL",
            "format": "json",
            "jsonVD": "Y",
            "prdSe": "M",
            "startPrdDe": f"{start_year}01",
            "endPrdDe": f"{end_year}12",
            "orgId": "101",
            "tblId": "DT_1B040B3",
        }
        data = call_api(params)
        if not isinstance(data, list):
            if isinstance(data, dict) and data.get("err") == "30":
                continue
            raise RuntimeError(f"[주민등록세대수 {start_year}-{end_year}] API 오류: {data}")

        for row in data:
            code = row["C1"]
            if len(code) == 2:
                # 시도 단위 (전국 '00' 제외, 서울 '11'도 시도합계로 별도 처리하지 않고 구단위로만 취합)
                if code == "00" or code == "11":
                    continue
                if code not in SIDO_CODE_TO_NAME:
                    continue
                region = SIDO_CODE_TO_NAME[code]
                rows.append((region, "", fmt_ym(row["PRD_DE"]), row["DT"]))
            elif len(code) == 5 and code.startswith("11"):
                # 서울 자치구
                rows.append(("서울", row["C1_NM"], fmt_ym(row["PRD_DE"]), row["DT"]))
        print(f"  - {start_year}~{end_year} 처리 완료 (누적 {len(rows)}행)")

    out_path = os.path.join(BASE_DIR, "주민등록세대수_월별.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["광역지자체", "자치구", "연월", "세대수"])
        for r in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
            w.writerow(r)
    print(f"[주민등록세대수] 완료: {out_path} ({len(rows)}행)\n")


# ---------------------------------------------------------------------------
# 2) 경제활동인구
#    - 연간 DT_1ES1B01S (~2010, 시도 단위)
#    - 분기 DT_1ES2B01S (2011~2012, 시도 단위)
#    - 반기 DT_1ES3A01S (시군구 단위만 제공, 시도 합계 없어 합산 산출.
#      9개 도는 2013~, 서울/6개 광역시는 2021.H1~ 만 존재. 세종은 데이터 없음)
#    각 표의 ITM_NM(지표명)을 그대로 사용해 long format으로 저장
# ---------------------------------------------------------------------------
SIDO_FULLNAME_TO_SHORT = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "전라북도": "전북",
    "전북특별자치도": "전북",
}


def collect_economic_activity():
    print("[경제활동인구] 수집 시작...")
    rows = []  # (광역지자체, 자치구, 연월, 주기, 지표명, 값, 단위)

    # --- 연간 (~2010) ---
    params = {
        "method": "getList",
        "apiKey": API_KEY,
        "itmId": "T1+T2+T3+T4+T5+T6+T7+T8+",
        "objL1": "ALL",
        "objL2": "ALL",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "Y",
        "startPrdDe": "2000",
        "endPrdDe": "2010",
        "orgId": "101",
        "tblId": "DT_1ES1B01S",
    }
    data = call_api(params)
    if isinstance(data, list):
        for row in data:
            if row.get("C2_NM") not in (None, "계"):
                continue  # 성별(남/여) 분리 행 제외, 계(전체)만 사용
            name = SIDO_FULLNAME_TO_SHORT.get(row["C1_NM"])
            if not name:
                continue  # '계' 등 합계행 제외
            ym = f"{row['PRD_DE']}-01"
            rows.append((name, "", ym, "연간", row["ITM_NM"], row["DT"], row.get("UNIT_NM", "")))
    print(f"  - 연간(DT_1ES1B01S) {len([r for r in rows if r[3]=='연간'])}행 수집")

    # --- 분기 (2011~2012) ---
    params = {
        "method": "getList",
        "apiKey": API_KEY,
        "itmId": "T1+T2+T3+T4+T5+T6+T7+T8+",
        "objL1": "ALL",
        "objL2": "ALL",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "Q",
        "startPrdDe": "201101",
        "endPrdDe": "201204",
        "orgId": "101",
        "tblId": "DT_1ES2B01S",
    }
    data = call_api(params)
    before = len(rows)
    if isinstance(data, list):
        for row in data:
            if row.get("C2_NM") not in (None, "계"):
                continue  # 성별(남/여) 분리 행 제외, 계(전체)만 사용
            name = SIDO_FULLNAME_TO_SHORT.get(row["C1_NM"])
            if not name:
                continue
            prd = row["PRD_DE"]  # 예: 201202 -> 2012년 2분기
            year, q = prd[:4], prd[4:]
            month = {"01": "01", "02": "04", "03": "07", "04": "10"}.get(q, "01")
            ym = f"{year}-{month}"
            rows.append((name, "", ym, "분기", row["ITM_NM"], row["DT"], row.get("UNIT_NM", "")))
    print(f"  - 분기(DT_1ES2B01S) {len(rows) - before}행 수집")

    # --- 반기 (2013~, 시군구 전국 단위 표) ---
    # 이 표(DT_1ES3A01S)는 시도 합계 행이 없고 시군구(행정구역) 단위로만 제공된다.
    # C1 코드의 앞 2자리가 표 내부적으로 시도를 구분하는 접두코드이며(11=서울, 21=부산, ...,
    # 31=경기, ..., 39=제주 / 세종은 이 표에 항목 자체가 없음), 실제 데이터로 검증하여 확인함.
    # - 서울(개별 자치구)과 부산/대구/인천/광주/대전/울산(구 단위 합산): 실제로는
    #   2021년 상반기부터만 이 표에 존재. 그 이전(2013~2020)은 원본에 구 자체가 없어 공백.
    # - 경기/강원/충북/충남/전북/전남/경북/경남/제주(9개 도): 시군구 데이터가 2013년부터
    #   존재해 시군구 합산으로 2013년부터 시도값 산출 가능.
    # - 세종: 이 표에 행정구역 항목 자체가 없어 전 기간 데이터 없음.
    # 결과적으로 시도 패널이 완전한 것은 아니며, 지역에 따라 커버리지 시작 시점이 다르다.
    PREFIX_TO_SIDO = {
        "11": "서울", "21": "부산", "22": "대구", "23": "인천", "24": "광주",
        "25": "대전", "26": "울산", "31": "경기", "32": "강원", "33": "충북",
        "34": "충남", "35": "전북", "36": "전남", "37": "경북", "38": "경남", "39": "제주",
    }

    before = len(rows)
    district_sums = {}  # (시도, 연월, 지표명) -> 합계 (인구수 계열만, 시도 단위 산출용)
    for start_year, end_year in year_chunks(2013, 2026, years_per_chunk=3):
        params = {
            "method": "getList",
            "apiKey": API_KEY,
            "itmId": "T1+T2+T3+T4+T9+T5+T6+T7+T11+T8+T10+",
            "objL1": "ALL",
            "format": "json",
            "jsonVD": "Y",
            "prdSe": "H",
            "startPrdDe": f"{start_year}01",
            "endPrdDe": f"{end_year}02",
            "orgId": "101",
            "tblId": "DT_1ES3A01S",
        }
        data = call_api(params)
        if not isinstance(data, list):
            if isinstance(data, dict) and data.get("err") == "30":
                continue
            raise RuntimeError(f"[경제활동인구 반기 {start_year}-{end_year}] API 오류: {data}")
        for row in data:
            prefix = row["C1"][:2]
            sido = PREFIX_TO_SIDO.get(prefix)
            if sido is None:
                continue
            prd = row["PRD_DE"]  # 예: 202501 -> 2025년 상반기, 202502 -> 하반기
            year, h = prd[:4], prd[4:]
            month = {"01": "01", "02": "07"}.get(h, "01")
            ym = f"{year}-{month}"
            itm = row["ITM_NM"]

            if sido == "서울":
                gu = row["C1_NM"].replace("서울 ", "")
                rows.append(("서울", gu, ym, "반기", itm, row["DT"], row.get("UNIT_NM", "")))

            if "천명" in itm:  # 비율(%) 지표는 단순 합산 불가하므로 인구수 계열만 시도 합계 산출
                try:
                    key = (sido, ym, itm)
                    district_sums[key] = district_sums.get(key, 0.0) + float(row["DT"])
                except (TypeError, ValueError):
                    pass
    print(f"  - 반기(DT_1ES3A01S, 서울 자치구) {len(rows) - before}행 수집")

    # 시군구 값을 합산해 시도 단위 값을 산출 (지역별 시작 시점 상이 - 위 주석 참고)
    before = len(rows)
    for (sido, ym, itm), total in district_sums.items():
        rows.append((sido, "", ym, "반기(구합산)", itm, round(total, 1), ""))
    print(f"  - 반기 시도 전체(시군구 합산) {len(rows) - before}행 산출")

    out_path = os.path.join(BASE_DIR, "경제활동인구_시계열.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["광역지자체", "자치구", "연월", "주기", "지표명", "값", "단위"])
        for r in sorted(rows, key=lambda x: (x[0], x[1], x[2], x[4])):
            w.writerow(r)
    print(f"[경제활동인구] 완료: {out_path} ({len(rows)}행)\n")


if __name__ == "__main__":
    collect_household_count()
    collect_economic_activity()
    print("모든 인구 관련 데이터 취합이 완료되었습니다.")
