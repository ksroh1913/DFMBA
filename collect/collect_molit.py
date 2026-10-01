# -*- coding: utf-8 -*-
"""
[수집] 국토교통부 실거래가 Open API(data.go.kr) -> raw/<동인 폴더>/ (위치: raw_layout.py)<유형>_<시군구코드>.csv

거래 "건별" 원자료를 가공 없이 저장한다. 전세/월세 구분, 갱신계약 제외, 집계 방식은
모두 preprocess/ 단계에서 결정하므로 기준이 바뀌어도 재호출이 불필요하다.

수집 대상:
  apt_trade : 아파트 매매 실거래가 상세자료 (RTMSDataSvcAptTradeDev), 2006.01~   V007
  apt_rent  : 아파트 전월세 실거래가        (RTMSDataSvcAptRent),      2011.01~   V006
  지역 범위 : 서울 25개 자치구

--------------------------------------------------------------------------------
과거 계약의 취소·해제를 어떻게 다루는가 (실측 근거)
--------------------------------------------------------------------------------
두 API의 취소 표현 방식이 다르다.
  - 매매  : 해제된 건도 응답에 남아 있고 cdealType='O', cdealDay(해제일)로 표시된다.
            강남구 7개월 표본에서 2,125건 중 178건(8.4%)이 해제였고, 특정 월은 14%였다.
  - 전월세: 해제 관련 필드가 아예 없다. 취소되면 응답에서 그냥 빠지는 것으로 보이며,
            같은 월을 다시 받아 행을 대조하는 것 외에는 감지할 방법이 없다.

해제는 계약월로부터 시차를 두고 발생한다(같은 표본의 해제일 기준):
  3개월 내 89.9% / 4개월 내 96.1% / 5개월 내 98.9% / 최대 11개월.
즉 지난달 데이터는 이미 확정된 게 아니라 몇 달간 계속 바뀐다.

그래서 이렇게 설계했다.
  1) 최근 REVERIFY_MONTHS(기본 6개월)만 매번 다시 받는다. 위 분포상 99% 이상을 잡고,
     호출은 25구 x 6개월 = 150회로 끝난다. 전 기간 재수집(약 5천회)은 불필요하다.
  2) 행마다 _firstSeen / _lastSeen 을 남긴다. 다시 받았는데 사라진 행은 지우지 않고
     _lastSeen 을 과거 날짜로 남겨둔다. 그래서 "언제부터 응답에서 빠졌는지"가 보존되고,
     전월세처럼 해제 플래그가 없는 경우에도 취소를 추적할 수 있다.
     전처리에서는 월별 최신 _lastSeen 과 같은 행만 쓰면 현재 유효한 스냅샷이 된다.
  3) 11개월을 넘겨 해제되는 건(0.6%)이나 원본 소급정정까지 반영하려면 주기적으로
     --full 로 전 기간을 재검증한다(연 1회 정도).

사용법:
  python collect/collect_molit.py                      # 전체 유형, 미수집분 + 최근 6개월
  python collect/collect_molit.py apt_rent             # 전월세만
  python collect/collect_molit.py apt_rent --full      # 전월세 전 기간 재검증
  python collect/collect_molit.py apt_rent --verify-from 202301
"""

import csv
import hashlib
import os
import sys
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    SEOUL_GU, http_get, month_range, raw_path, require_key, shift_ym,
)

API_KEY = require_key("DATA_GO_KR_API_KEY")
TODAY = date.today().isoformat()
TODAY_YM = date.today().strftime("%Y%m")

# 해제 시차 분포(5개월 내 98.9%)를 감안한 재검증 창
REVERIFY_MONTHS = 6

KINDS = {
    "apt_trade": {
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev",
        "start": "200601",
        "desc": "아파트 매매",
    },
    "apt_rent": {
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
        "start": "201101",  # 전월세 실거래 공개 시작
        "desc": "아파트 전월세",
    },
}

# 수집 메타 컬럼 (원본 필드와 구분하기 위해 밑줄 접두)
M_YMD, M_FIRST, M_LAST = "_dealYmd", "_firstSeen", "_lastSeen"
META = (M_YMD, M_FIRST, M_LAST)


class QuotaExceeded(Exception):
    pass


def fetch_month(url, lawd_cd, deal_ymd):
    """해당 시군구·계약년월의 전체 거래 건을 dict 리스트로 반환"""
    items = []
    page_no = 1
    while True:
        params = {"serviceKey": API_KEY, "LAWD_CD": lawd_cd, "DEAL_YMD": deal_ymd,
                  "numOfRows": 1000, "pageNo": page_no}
        try:
            text = http_get(url + "?" + urllib.parse.urlencode(params))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise QuotaExceeded(f"{lawd_cd} {deal_ymd}")
            raise

        root = ET.fromstring(text)
        code = root.findtext("./header/resultCode")
        if code != "000":
            if code in ("03", "004"):  # 데이터 없음
                break
            raise RuntimeError(f"[{lawd_cd} {deal_ymd}] API 오류 {code}: "
                               f"{root.findtext('./header/resultMsg')}")

        page_items = root.findall("./body/items/item")
        for item in page_items:
            items.append({c.tag: (c.text or "").strip() for c in item})

        total = int(root.findtext("./body/totalCount") or 0)
        if page_no * 1000 >= total or not page_items:
            break
        page_no += 1
    return items


def row_identity(row):
    """원본 필드만으로 계산하는 행 지문. 메타 컬럼은 제외해야 재실행 시 값이 흔들리지 않는다."""
    payload = "|".join(f"{k}={row[k]}" for k in sorted(row) if k not in META)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def keyed(rows):
    """같은 내용의 행이 여러 건일 수 있으므로(동일 단지·면적·층·가격의 별개 계약)
    지문에 출현 순번을 붙여 건수를 잃지 않게 한다."""
    seen = defaultdict(int)
    out = {}
    for r in rows:
        h = row_identity(r)
        out[(h, seen[h])] = r
        seen[h] += 1
    return out


def load_raw(kind, lawd_cd):
    path = raw_path("molit", f"{kind}_{lawd_cd}")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_raw(kind, lawd_cd, rows):
    fieldnames = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in META]
    fieldnames += list(META)
    path = raw_path("molit", f"{kind}_{lawd_cd}")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get(M_YMD, ""), x.get("aptSeq", ""))):
            w.writerow(r)


def months_to_fetch(existing, start, mode, verify_from):
    """이미 받은 월은 건너뛰고, 재검증 대상 월만 다시 받는다."""
    all_months = month_range(start, TODAY_YM)
    have = {r.get(M_YMD) for r in existing}

    if mode == "full":
        return all_months
    if mode == "verify-from":
        return [m for m in all_months if m >= verify_from or m not in have]

    window_start = shift_ym(TODAY_YM, -(REVERIFY_MONTHS - 1))
    return [m for m in all_months if m not in have or m >= window_start]


def collect_kind(kind, mode, verify_from):
    cfg = KINDS[kind]
    print(f"\n[{kind}] {cfg['desc']} 수집 (모드: {mode})")

    for i, (lawd_cd, gu_name) in enumerate(SEOUL_GU.items(), 1):
        existing = load_raw(kind, lawd_cd)
        targets = months_to_fetch(existing, cfg["start"], mode, verify_from)
        if not targets:
            print(f"- [{i}/25] {gu_name}: 재수집 대상 없음 (보유 {len(existing)}건)")
            continue

        # 대상 월만 갈아끼우고 나머지는 그대로 유지
        target_set = set(targets)
        kept = [r for r in existing if r.get(M_YMD) not in target_set]
        old_by_month = defaultdict(list)
        for r in existing:
            if r.get(M_YMD) in target_set:
                old_by_month[r[M_YMD]].append(r)

        merged, added, vanished = [], 0, 0
        try:
            for ym in targets:
                new_rows = fetch_month(cfg["url"], lawd_cd, ym)
                for r in new_rows:
                    r[M_YMD] = ym

                old_keyed = keyed(old_by_month.get(ym, []))
                new_keyed = keyed(new_rows)

                for key, r in new_keyed.items():
                    prev = old_keyed.get(key)
                    r[M_FIRST] = prev.get(M_FIRST, TODAY) if prev else TODAY
                    r[M_LAST] = TODAY
                    if not prev:
                        added += 1
                    merged.append(r)

                # 응답에서 빠진 행: 삭제하지 않고 _lastSeen을 과거로 남겨 취소 흔적을 보존
                for key, prev in old_keyed.items():
                    if key not in new_keyed:
                        merged.append(prev)
                        vanished += 1
        except QuotaExceeded as e:
            save_raw(kind, lawd_cd, kept + merged)
            print(f"  ! 일일 호출 한도 초과({e}). 여기까지 저장하고 중단합니다.")
            print(f"    다음 날 같은 명령을 실행하면 {gu_name}부터 이어받습니다.")
            raise SystemExit(1)

        save_raw(kind, lawd_cd, kept + merged)
        note = f", 신규 {added}건" if added else ""
        note += f", 응답에서 사라짐 {vanished}건" if vanished else ""
        print(f"- [{i}/25] {gu_name}({lawd_cd}): {len(targets)}개월 조회, "
              f"누적 {len(kept) + len(merged)}건{note}", flush=True)


def main():
    args = sys.argv[1:]

    mode, verify_from = "incremental", None
    if "--full" in args:
        mode = "full"
        args.remove("--full")
    if "--verify-from" in args:
        idx = args.index("--verify-from")
        mode, verify_from = "verify-from", args[idx + 1]
        del args[idx:idx + 2]

    kinds = [a for a in args if a in KINDS] or list(KINDS)

    print("[수집] 국토부 실거래가 -> raw/<동인 폴더>/ (위치: raw_layout.py) (건별 원자료)")
    for kind in kinds:
        collect_kind(kind, mode, verify_from)
    print("\n완료")


if __name__ == "__main__":
    main()
