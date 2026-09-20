# -*- coding: utf-8 -*-
"""
[수집] 국토교통부 실거래가 Open API(data.go.kr) -> raw/molit/<유형>_<시군구코드>.csv

거래 "건별" 원자료를 그대로 저장한다. 갱신계약 제외 여부, 전세/월세 구분,
집계 방식 등은 preprocess/ 단계에서 결정하므로, 기준이 바뀌어도 재호출이 불필요하다.

수집 대상:
  apt_trade : 아파트 매매 실거래가 상세자료 (RTMSDataSvcAptTradeDev)
  apt_rent  : 아파트 전월세 실거래가        (RTMSDataSvcAptRent)
  지역 범위 : 서울 25개 자치구

주의: data.go.kr은 API별 일일 호출 한도가 있다. 한도 초과(HTTP 429) 시
      그때까지 받은 분을 저장하고 종료하므로, 다음 날 다시 실행하면 이어서 받는다.
"""

import csv
import os
import sys
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    RAW_DIR, SEOUL_GU, http_get, month_range, raw_path, require_key, shift_ym,
)

API_KEY = require_key("DATA_GO_KR_API_KEY")
TODAY_YM = date.today().strftime("%Y%m")
START_YM = "200601"  # 실거래가 공개 시작 시점
REFETCH_BUFFER_MONTHS = 3  # 신고 지연(계약 후 30일 이내 신고) 감안

ENDPOINTS = {
    "apt_trade": "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev",
    "apt_rent": "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
}

# 어느 조회월 요청으로 받은 행인지 표시 -> 월 단위 교체 갱신에 사용
SRC_COL = "_dealYmd"


class QuotaExceeded(Exception):
    pass


def fetch_month(endpoint, lawd_cd, deal_ymd):
    """해당 시군구·계약년월의 전체 거래 건을 dict 리스트로 반환."""
    items = []
    page_no = 1
    while True:
        params = {
            "serviceKey": API_KEY, "LAWD_CD": lawd_cd, "DEAL_YMD": deal_ymd,
            "numOfRows": 1000, "pageNo": page_no,
        }
        url = endpoint + "?" + urllib.parse.urlencode(params)
        try:
            text = http_get(url)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise QuotaExceeded(f"{lawd_cd} {deal_ymd}")
            raise

        root = ET.fromstring(text)
        code = root.findtext("./header/resultCode")
        if code != "000":
            if code in ("03", "004"):  # 데이터 없음
                break
            raise RuntimeError(f"[{lawd_cd} {deal_ymd}] API 오류 {code}: {root.findtext('./header/resultMsg')}")

        page_items = root.findall("./body/items/item")
        for item in page_items:
            row = {child.tag: (child.text or "").strip() for child in item}
            row[SRC_COL] = deal_ymd
            items.append(row)

        total = int(root.findtext("./body/totalCount") or 0)
        if page_no * 1000 >= total or not page_items:
            break
        page_no += 1
    return items


def load_gu_raw(kind, lawd_cd):
    path = raw_path("molit", f"{kind}_{lawd_cd}")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_gu_raw(kind, lawd_cd, rows):
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    path = raw_path("molit", f"{kind}_{lawd_cd}")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get(SRC_COL, ""), x.get("aptSeq", ""))):
            w.writerow(r)
    return len(rows)


def collect_kind(kind):
    endpoint = ENDPOINTS[kind]
    print(f"\n[{kind}] 수집 시작")

    for i, (lawd_cd, gu_name) in enumerate(SEOUL_GU.items(), 1):
        existing = load_gu_raw(kind, lawd_cd)
        last_ymd = max((r.get(SRC_COL, "") for r in existing), default="")

        start = START_YM
        if last_ymd:
            candidate = shift_ym(last_ymd, -REFETCH_BUFFER_MONTHS)
            if candidate > start:
                start = candidate
        months = month_range(start, TODAY_YM)

        # 다시 받는 월의 기존 행은 제거하고 새로 받은 것으로 교체
        refetch = set(months)
        kept = [r for r in existing if r.get(SRC_COL, "") not in refetch]

        new_rows = []
        try:
            for ym in months:
                new_rows.extend(fetch_month(endpoint, lawd_cd, ym))
        except QuotaExceeded as e:
            save_gu_raw(kind, lawd_cd, kept + new_rows)
            print(f"  ! 일일 호출 한도 초과({e}). 여기까지 저장하고 중단합니다.")
            print(f"    다음 날 다시 실행하면 {gu_name}부터 이어서 받습니다.")
            raise SystemExit(1)

        total = save_gu_raw(kind, lawd_cd, kept + new_rows)
        print(f"- [{i}/{len(SEOUL_GU)}] {gu_name}({lawd_cd}): {len(months)}개월 조회, "
              f"{len(new_rows)}건 수신, raw 누적 {total}건", flush=True)


def main():
    os.makedirs(os.path.join(RAW_DIR, "molit"), exist_ok=True)
    print("[수집] 국토부 실거래가 -> raw/molit/ (건별 원자료)")
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    kinds = [target] if target in ENDPOINTS else list(ENDPOINTS)
    for kind in kinds:
        collect_kind(kind)
    print("\n완료")


if __name__ == "__main__":
    main()
