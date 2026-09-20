# -*- coding: utf-8 -*-
"""
한국부동산원 R-ONE(부동산통계정보시스템) Open API 아파트 관련 통계 취합 스크립트

취합 항목(모두 "_아파트" 기준):
  1. 월세통합가격지수      2. 월세가격지수(2010~15 구버전 포함)   3. 매매가격지수
  4. 전월세통합지수        5. 전세가격지수                        6. 지역별 전월세전환율
  7. 매매/전세/월세 수급동향(2010~15 구버전 포함)
  8. 행정구역별 아파트거래현황 / 아파트매매거래현황
  9. 주택시장 소비심리지수

기존 collect_housing_stats.py / collect_population_stats.py 와 동일하게
"광역지자체, 자치구, 연월" 키를 사용하는 long format으로 저장하여
이후 머신러닝용 데이터셋 병합이 쉽도록 함.

산출물: 부동산원_아파트_통계_시계열.csv
  컬럼: 광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위
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
API_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"


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
API_KEY = ENV.get("RONE_API_KEY") or os.environ.get("RONE_API_KEY")
if not API_KEY:
    raise SystemExit("RONE_API_KEY가 .env에 없습니다.")

REGIONS = {
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
}

PAGE_SIZE = 1000


def call_api(params, retries=3):
    query = urllib.parse.urlencode(params)
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


def fetch_all_rows(statbl_id, start_wrttime, end_wrttime):
    """지정 기간의 모든 데이터 행을 페이지네이션하며 가져온다."""
    rows = []
    p_index = 1
    while True:
        params = {
            "KEY": API_KEY,
            "STATBL_ID": statbl_id,
            "DTACYCLE_CD": "MM",
            "START_WRTTIME": start_wrttime,
            "END_WRTTIME": end_wrttime,
            "Type": "json",
            "pSize": PAGE_SIZE,
            "pIndex": p_index,
        }
        data = call_api(params)
        top = data.get("SttsApiTblData")
        if not top:
            # RESULT 최상위 오류 (예: 데이터 없음)
            break
        head = top[0]["head"]
        total = head[0]["list_total_count"]
        code = head[1]["RESULT"]["CODE"]
        if code != "INFO-000" or total == 0:
            break
        page_rows = top[1]["row"]
        rows.extend(page_rows)
        if p_index * PAGE_SIZE >= total:
            break
        p_index += 1
    return rows


def parse_region(cls_fullnm):
    """CLS_FULLNM(예: '서울>강북지역>도심권>종로구', '경기', '전남광주>광주')을
    (광역지자체, 자치구) 로 변환. 대상 밖이면 None 반환."""
    if not cls_fullnm:
        return None
    parts = cls_fullnm.split(">")
    last = parts[-1]
    first = parts[0]
    if last in REGIONS:
        return last, ""
    if last == "계" and first in REGIONS and len(parts) == 2:
        # 예: '경기>계' -> 시도 합계 행 (거래현황/미분양현황 계열)
        return first, ""
    if first == "서울" and last.endswith(("구", "군")) and len(parts) >= 2:
        return "서울", last
    return None


def ym_from_wrttime(wrttime):
    return f"{wrttime[:4]}-{wrttime[4:6]}"


def wrttime_from_ym(ym):
    return ym.replace("-", "")


def shift_ym(ym, months):
    """ym('YYYY-MM')에서 months만큼 이동한 'YYYYMM' 문자열 반환 (음수면 과거로)"""
    y, m = int(ym[:4]), int(ym[5:7])
    total = y * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}{nm + 1:02d}"


OUT_CSV_PATH_NAME = "부동산원_아파트_통계_시계열.csv"
REFETCH_BUFFER_MONTHS = 2  # 마지막 수집월 기준 최근 N개월은 확정치 반영을 위해 재수집


def load_existing(base_dir):
    """이전 실행 결과 CSV를 불러온다. 없으면 빈 상태 반환."""
    path = os.path.join(base_dir, OUT_CSV_PATH_NAME)
    rows_by_key = {}
    max_ym_by_name = {}
    if not os.path.exists(path):
        return rows_by_key, max_ym_by_name
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) != 7:
                continue
            sido, gu, ym, name, item, val, unit = row
            key = (sido, gu, ym, name, item)
            rows_by_key[key] = (sido, gu, ym, name, item, val, unit)
            if ym > max_ym_by_name.get(name, ""):
                max_ym_by_name[name] = ym
    return rows_by_key, max_ym_by_name


# ---------------------------------------------------------------------------
# 통계표 정의
# ---------------------------------------------------------------------------
# 각 항목: (출력용 통계명, [(STATBL_ID, 시작연월, 종료연월), ...])  여러 구간을 이어붙임
TABLE_DEFS = [
    ("월세통합가격지수_아파트", [("A_2024_00054", "201501", "202612")]),
    ("월세가격지수_아파트", [
        ("A_2024_00164", "201001", "201512"),  # 구: 주택유형별 월세가격지수(구)_아파트
        ("A_2024_00055", "201601", "202612"),  # 신: 월세가격지수_아파트
    ]),
    ("매매가격지수_아파트", [("A_2024_00045", "200301", "202612")]),
    ("전월세통합지수_아파트", [("A_2024_00049", "201501", "202612")]),
    ("전세가격지수_아파트", [("A_2024_00050", "200301", "202612")]),
    ("지역별전월세전환율_아파트", [("A_2024_00156", "201101", "202612")]),
    ("매매수급동향_아파트", [("A_2024_00076", "201201", "202612")]),
    ("전세수급동향_아파트", [("A_2024_00077", "201201", "202612")]),
    ("월세수급동향_아파트", [("A_2024_00078", "201501", "202612")]),
    ("행정구역별_아파트거래현황", [("A_2024_00549", "200601", "202612")]),
    ("행정구역별_아파트매매거래현황", [("A_2024_00554", "200601", "202612")]),
    ("주택시장_소비심리지수", [("T232543129897499", "201101", "202612")]),
    ("미분양주택현황", [("T237973129847263", "200701", "202612")]),
]

# 2010~2015년 구버전 "지역별 수급동향(구)" - (구)월세가격동향조사 계열로, 현행
# 매매수급동향_아파트(A_2024_00076, 2012~)와는 별개 통계이며 기간도 겹친다.
# 수요우위/비슷함/공급우위 응답비율을 표준 수급지수(100+수요우위-공급우위)로 환산하되,
# 통계명을 별도로 두어 신버전과 절대 같은 (지역,연월,통계명,항목명) 키로 겹치지 않게 한다.
OLD_SUPPLY_STATBL_ID = "T242093131851096"
OLD_SUPPLY_RANGE = ("201001", "201512")
OLD_SUPPLY_OUT_NAME = "수급동향(구)_통합"


def process_price_style_table(statbl_id, start_wrttime, end_wrttime, out_name, out_rows):
    raw_rows = fetch_all_rows(statbl_id, start_wrttime, end_wrttime)
    for row in raw_rows:
        region = parse_region(row["CLS_FULLNM"])
        if region is None:
            continue
        sido, gu = region
        ym = ym_from_wrttime(row["WRTTIME_IDTFR_ID"])
        out_rows.append((sido, gu, ym, out_name, row["ITM_NM"], row["DTA_VAL"], row.get("UI_NM", "")))


def process_old_supply_demand(out_rows):
    """구버전 수급동향(구): GRP_NM=지역, CLS_NM=수요우위/비슷함/공급우위 -> 수급지수로 환산"""
    raw_rows = fetch_all_rows(OLD_SUPPLY_STATBL_ID, *OLD_SUPPLY_RANGE)
    grouped = {}  # (지역, 연월) -> {수요우위:v, 비슷함:v, 공급우위:v}
    for row in raw_rows:
        region_name = row.get("GRP_NM")
        if region_name is None:
            continue
        region = parse_region(region_name)  # GRP_NM은 이미 '서울', '경기' 등 단일 지역명
        if region is None:
            continue
        sido, gu = region
        ym = ym_from_wrttime(row["WRTTIME_IDTFR_ID"])
        key = (sido, gu, ym)
        grouped.setdefault(key, {})[row["CLS_NM"]] = row["DTA_VAL"]

    for (sido, gu, ym), vals in grouped.items():
        demand = vals.get("수요우위")
        supply = vals.get("공급우위")
        if demand is None or supply is None:
            continue
        index = 100 + demand - supply
        out_rows.append((sido, gu, ym, OLD_SUPPLY_OUT_NAME, "지수", round(index, 2), "지수"))


def main():
    print("R-ONE에서 아파트 관련 부동산 통계(월별)를 수집합니다...")

    existing_rows, existing_max_ym = load_existing(BASE_DIR)
    if existing_rows:
        print(f"기존 결과 {len(existing_rows)}행 로드 (증분 수집 모드)")

    out_rows = []  # 새로 fetch한 행만 담김 (기존 rows_by_key에 병합됨)

    for out_name, segments in TABLE_DEFS:
        before = len(out_rows)
        last_ym = existing_max_ym.get(out_name)
        for statbl_id, s, e in segments:
            if e != "202612":
                # 종료 시점이 고정된 과거 구간(예: 2010~2015 구버전) - 이미 그 끝까지
                # 수집되어 있으면 원본이 더 이상 변하지 않으므로 재요청하지 않는다.
                if last_ym and last_ym >= f"{e[:4]}-{e[4:6]}":
                    continue
                process_price_style_table(statbl_id, s, e, out_name, out_rows)
            else:
                # 현재까지 계속 갱신되는 구간 - 마지막 수집월 근처부터만 다시 받는다.
                start = s
                if last_ym:
                    candidate = shift_ym(last_ym, -REFETCH_BUFFER_MONTHS)
                    if candidate > s:
                        start = candidate
                process_price_style_table(statbl_id, start, e, out_name, out_rows)
        print(f"- {out_name}: {len(out_rows) - before}행 신규 수집" + (" (스킵)" if len(out_rows) == before else ""))

    before = len(out_rows)
    if not (existing_max_ym.get(OLD_SUPPLY_OUT_NAME, "") >= f"{OLD_SUPPLY_RANGE[1][:4]}-{OLD_SUPPLY_RANGE[1][4:6]}"):
        process_old_supply_demand(out_rows)
    print(f"- {OLD_SUPPLY_OUT_NAME}(2010~2015, (구)월세가격동향조사 계열): {len(out_rows) - before}행 신규 수집")

    # 기존 결과에 새로 받은 행을 병합 (동일 키는 새 값으로 덮어씀 -> 확정치 갱신 반영)
    merged = dict(existing_rows)
    for r in out_rows:
        key = (r[0], r[1], r[2], r[3], r[4])
        merged[key] = r

    # 키(광역지자체+자치구+연월+통계명+항목명) 유일성 검증
    dup_keys = []
    seen_for_dupcheck = {}
    for r in out_rows:
        key = (r[0], r[1], r[2], r[3], r[4])
        if key in seen_for_dupcheck and seen_for_dupcheck[key] != r[5]:
            dup_keys.append((key, seen_for_dupcheck[key], r[5]))
        seen_for_dupcheck[key] = r[5]
    if dup_keys:
        print(f"\n!! 경고: 신규 수집분 내 키 중복 {len(dup_keys)}건 발견 (동일 키에 서로 다른 값):")
        for key, v1, v2 in dup_keys[:10]:
            print(f"   {key}: {v1} vs {v2}")
        if len(dup_keys) > 10:
            print(f"   ... 외 {len(dup_keys) - 10}건")

    out_path = os.path.join(BASE_DIR, OUT_CSV_PATH_NAME)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"])
        for r in sorted(merged.values(), key=lambda x: (x[3], x[0], x[1], x[2], x[4])):
            w.writerow(r)

    print(f"\n완료: {out_path} (전체 {len(merged)}행, 신규/갱신 {len(out_rows)}행)")
    print("※ 서울은 자치구 단위까지, 그 외 지역은 광역지자체(시도) 단위로 취합되었습니다.")
    print("  (표별로 자치구 데이터 유무가 달라, 자치구 데이터가 없는 표는 시도 단위만 존재)")


if __name__ == "__main__":
    main()
