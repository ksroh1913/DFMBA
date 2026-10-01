# -*- coding: utf-8 -*-
"""
[수집] 한국부동산원 R-ONE Open API -> raw/<동인 폴더>/ (위치: raw_layout.py)<STATBL_ID>.csv

응답 행(CLS_FULLNM, ITM_NM, DTA_VAL 등)을 그대로 저장한다.
지역 계층 파싱·지수 환산 등은 preprocess/ 단계에서 수행.

페이지 넘김을 쓰지 않는다:
  R-ONE은 여러 페이지로 나눠 받을 때 정렬이 고정되지 않아, 같은 행이 두 페이지에
  중복으로 오고 다른 행 하나가 빠지는 일이 있다(실측: 월세수급동향 2017-10 전남 누락).
  그래서 조회 기간을 한 번에 1,000행 이하가 되도록 반씩 쪼개 받고, 받은 고유 행 수가
  API가 알려준 총건수와 같은지 확인한다. 한 달치가 1,000행을 넘는 경우에만 페이지를
  넘기며, 그때도 총건수가 맞을 때까지 다시 받는다.

사용법:
  python collect/collect_rone.py                  # 증분 (최근 2개월 재수집 + 신규)
  python collect/collect_rone.py --full           # 전 기간 재수집
  python collect/collect_rone.py --verify         # 연도별 건수 대조 후 누락 연도만 재수집
  python collect/collect_rone.py A_2024_00078     # 지정 표만
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    http_get, load_raw, month_range, require_key, save_raw, raw_max_value, shift_ym,
)

import json  # noqa: E402
import urllib.parse  # noqa: E402

API_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"
API_KEY = require_key("RONE_API_KEY")
PAGE_SIZE = 1000  # R-ONE 1회 최대 요청 건수
REFETCH_BUFFER_MONTHS = 2

KEY_FIELDS = ["STATBL_ID", "WRTTIME_IDTFR_ID", "GRP_ID", "CLS_ID", "ITM_ID"]

TODAY_YM = date.today().strftime("%Y%m")

# (STATBL_ID, 설명, 시작연월, 종료연월 None=현재진행)  주석 = 데이터사전 Master ID
TABLES = [
    ("A_2024_00054", "월세통합가격지수_아파트", "201501", None),  # V001
    ("A_2024_00045", "매매가격지수_아파트", "200301", None),  # V002
    ("A_2024_00050", "전세가격지수_아파트", "200301", None),  # V026
    ("A_2024_00156", "지역별 전월세 전환율_아파트", "201101", None),  # V004
    ("A_2024_00072", "매매가격 대비 전세가격 비율(전세가율)_아파트", "201201", None),  # V003
    ("A_2024_00076", "매매수급동향_아파트", "201201", None),  # V031
    ("A_2024_00077", "전세수급동향_아파트", "201201", None),  # V032
    ("A_2024_00078", "월세수급동향_아파트", "201501", None),  # V005 (=V033)
    ("A_2024_00549", "행정구역별 아파트거래현황", "200601", None),  # V035
    ("A_2024_00554", "행정구역별 아파트매매거래현황", "200601", None),  # V036
    ("T232543129897499", "주택시장 소비심리지수", "201101", None),  # V037
    ("T237973129847263", "미분양주택현황", "200701", None),  # V038
    ("T242093131851096", "지역별 수급동향(구)", "201001", "201512"),  # V034
]


def request(statbl_id, start, end, p_index=1, p_size=PAGE_SIZE):
    """한 번 호출. (총건수, 행 목록) 반환. 데이터가 없으면 (0, [])."""
    params = {
        "KEY": API_KEY, "STATBL_ID": statbl_id, "DTACYCLE_CD": "MM",
        "START_WRTTIME": start, "END_WRTTIME": end,
        "Type": "json", "pSize": p_size, "pIndex": p_index,
    }
    data = json.loads(http_get(API_URL + "?" + urllib.parse.urlencode(params)))
    top = data.get("SttsApiTblData")
    if not top:
        return 0, []  # RESULT만 반환 = 데이터 없음
    head = top[0]["head"]
    if head[1]["RESULT"]["CODE"] != "INFO-000":
        return 0, []
    return head[0]["list_total_count"], top[1]["row"]


def count_rows(statbl_id, start, end):
    """건수만 조회 (pSize=1). 1,000행 조회의 약 1/30 시간."""
    return request(statbl_id, start, end, p_size=1)[0]


def row_key(r):
    return tuple(str(r.get(k)) for k in KEY_FIELDS)


def fetch_single_month_paged(statbl_id, ym, total, retries=3):
    """한 달치가 1,000행을 넘는 경우만: 페이지를 넘기되 고유 행 수가 총건수와 맞는지 검증."""
    for _ in range(retries):
        uniq = {}
        for p in range(1, -(-total // PAGE_SIZE) + 1):
            for r in request(statbl_id, ym, ym, p)[1]:
                uniq[row_key(r)] = r
        if len(uniq) == total:
            return list(uniq.values())
    raise RuntimeError(f"[{statbl_id} {ym}] 페이지 수집 불일치: {len(uniq)} / {total}")


def fetch_all_rows(statbl_id, start_wrttime, end_wrttime):
    """기간을 반씩 쪼개 한 번에 1,000행 이하로 받는다 (페이지 넘김 불안정 회피).
    쪼갤지는 건수 조회로 먼저 판단해 대용량 호출을 낭비하지 않는다."""
    total = count_rows(statbl_id, start_wrttime, end_wrttime)
    if total == 0:
        return []
    if total <= PAGE_SIZE:
        total, rows = request(statbl_id, start_wrttime, end_wrttime)
        if len({row_key(r) for r in rows}) != total:
            raise RuntimeError(f"[{statbl_id} {start_wrttime}~{end_wrttime}] 행수 불일치")
        return rows
    months = month_range(start_wrttime, end_wrttime)
    if len(months) == 1:
        return fetch_single_month_paged(statbl_id, start_wrttime, total)
    mid = len(months) // 2
    return (fetch_all_rows(statbl_id, months[0], months[mid - 1])
            + fetch_all_rows(statbl_id, months[mid], months[-1]))


def fetch_table(statbl_id, desc, start, end, full=False):
    last = None if full else raw_max_value("rone", statbl_id, "WRTTIME_IDTFR_ID")

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


def verify_table(statbl_id, desc, start, end):
    """연도별로 API 총건수와 raw 행수를 비교해, 다른 연도만 다시 받는다."""
    from collections import Counter
    have = Counter(r.get("WRTTIME_IDTFR_ID", "")[:4] for r in load_raw("rone", statbl_id))
    fixed = []
    for y in range(int(start[:4]), int((end or TODAY_YM)[:4]) + 1):
        s, e = max(f"{y}01", start), min(f"{y}12", end or TODAY_YM)
        api = count_rows(statbl_id, s, e)
        if api == have.get(str(y), 0):
            continue
        rows = fetch_all_rows(statbl_id, s, e)
        save_raw("rone", statbl_id, rows, KEY_FIELDS)
        fixed.append(f"{y}(raw {have.get(str(y), 0)} -> api {api})")
    after = len(load_raw("rone", statbl_id))
    note = ", ".join(fixed) if fixed else "누락 없음"
    print(f"- {statbl_id} ({desc}): {note} / raw {after}행", flush=True)


def main():
    args = sys.argv[1:]
    if "--verify" in args:
        wanted = {a for a in args if not a.startswith("--")}
        print("[검증] R-ONE 연도별 총건수 vs raw")
        for statbl_id, desc, start, end in TABLES:
            if not wanted or statbl_id in wanted:
                verify_table(statbl_id, desc, start, end)
        print("완료")
        return
    full = "--full" in args
    wanted = {a for a in args if not a.startswith("--")}
    print(f"[수집] R-ONE -> raw/<동인 폴더>/ (위치: raw_layout.py) ({'전 기간' if full else '증분'})")
    for statbl_id, desc, start, end in TABLES:
        if wanted and statbl_id not in wanted:
            continue
        fetch_table(statbl_id, desc, start, end, full=full)
    print("완료")


if __name__ == "__main__":
    main()
