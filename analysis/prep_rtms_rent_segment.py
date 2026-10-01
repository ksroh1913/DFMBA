# -*- coding: utf-8 -*-
"""
※ 데이터사전 변수가 아닌 분석용 산출물 (V006·V007 실거래 원본을 가공). 결과는 analysis/output/ 에 저장.
[전처리] 아파트 전월세 실거래 -> 계약유형(전세/월세/준월세/준전세)별 월별 집계

입력: raw:molit/apt_rent_<시군구코드>.csv          (건별 원자료)
      processed/부동산원_전월세전환율_아파트.csv     (자치구별 월별 전환율, 연 %)
출력: analysis/output/서울_아파트_전월세_유형별_월별집계.csv

계약유형 구분 (한국부동산원 공식 기준, 보증금/월세 배수)
  전세   : 월세 = 0
  월세   : 보증금 <= 월세 x 12        (보증금이 1년치 월세 이하 = 사실상 순수월세)
  준월세 : 월세 x 12 < 보증금 <= 월세 x 240
  준전세 : 보증금 > 월세 x 240        (사실상 전세에 가까운 반전세)

기존 집계(prep_rtms_rent.py)는 월세를 한 덩어리로 봤다. 그런데 월세 계약의 39%가
준전세라서, '월세 평균가격'이 사실은 전세에 가까운 계약의 가격 흐름에 끌려간다.
유형을 나누면 각 시장이 서로 다르게 움직이는지 확인할 수 있다.

환산월세 = 월세 + 보증금 x (전환율/100) / 12   <- 계약 건별
㎡당      = 환산월세 / 전용면적                 <- 계약 건별로 구한 뒤 평균/중위

자치구 행과 서울 전체 행(자치구 빈칸)을 함께 출력한다.
갱신계약(contractType='갱신') 제외, 응답에서 사라진 건(취소 추정) 제외는
prep_rtms_rent.py 와 동일하다.
"""

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PROCESSED_DIR, SEOUL_GU, raw_path, write_analysis  # noqa: E402

CONV_CSV = os.path.join(PROCESSED_DIR, "부동산원_전월세전환율_아파트.csv")
TYPES = ["전세", "월세", "준월세", "준전세"]


def load_conversion_rates():
    rates = {}
    with open(CONV_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r["자치구"]:
                continue
            try:
                rates[(r["자치구"], r["연월"])] = float(r["값"])
            except (TypeError, ValueError):
                pass
    return rates


def classify(deposit, rent):
    """보증금/월세 배수로 계약유형 판정"""
    if rent <= 0:
        return "전세"
    ratio = deposit / rent
    if ratio <= 12:
        return "월세"
    if ratio <= 240:
        return "준월세"
    return "준전세"


def mean(v):
    return sum(v) / len(v) if v else None


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def rnd(v, d=1):
    return "" if v is None else round(v, d)


def new_bucket():
    return {"dep": [], "rent": [], "area": [], "eq": [], "unit": [], "depeq": []}


def main():
    rates = load_conversion_rates()
    # (자치구, 연월, 유형) -> 값 리스트 ; 자치구 "" 는 서울 전체
    buckets = defaultdict(new_bucket)
    renewals = stale = 0

    for lawd_cd, gu in SEOUL_GU.items():
        path = raw_path("molit", f"apt_rent_{lawd_cd}")
        if not os.path.exists(path):
            print(f"  ! raw 없음: {gu}")
            continue

        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        newest = defaultdict(str)
        for r in rows:
            ym = r.get("_dealYmd", "")
            newest[ym] = max(newest[ym], r.get("_lastSeen", ""))

        for r in rows:
            if r.get("_lastSeen", "") != newest[r.get("_dealYmd", "")]:
                stale += 1
                continue
            if (r.get("contractType") or "").strip() == "갱신":
                renewals += 1
                continue
            try:
                deposit = float(r["deposit"].replace(",", ""))
                area = float(r["excluUseAr"])
                rent = float((r.get("monthlyRent") or "0").replace(",", "") or 0)
                ym = f"{int(r['dealYear']):04d}-{int(r['dealMonth']):02d}"
            except (KeyError, ValueError):
                continue
            if area <= 0:
                continue

            kind = classify(deposit, rent)
            rate = rates.get((gu, ym))
            depeq = deposit * (rate / 100) / 12 if rate is not None else None
            eq = rent + depeq if depeq is not None else None

            for scope in (gu, ""):
                b = buckets[(scope, ym, kind)]
                b["dep"].append(deposit)
                b["rent"].append(rent)
                b["area"].append(area)
                if eq is not None:
                    b["eq"].append(eq)
                    b["unit"].append(eq / area)
                    b["depeq"].append(depeq)

        print(f"  - {gu}: {len(rows):,}건 처리", flush=True)

    # 유형별 비중을 내려면 (자치구, 연월) 전체 건수가 필요하다
    total_n = defaultdict(int)
    for (scope, ym, kind), b in buckets.items():
        total_n[(scope, ym)] += len(b["dep"])

    out = []
    for (scope, ym, kind), b in buckets.items():
        n = len(b["dep"])
        if not n:
            continue
        rate = rates.get((scope, ym)) if scope else None
        eq_mean = mean(b["eq"])
        area_mean = mean(b["area"])
        # 환산월세 중 보증금에서 온 몫
        dep_share = mean(b["depeq"]) / eq_mean * 100 if eq_mean else None
        out.append([
            "서울", scope, ym, kind, n,
            round(n / total_n[(scope, ym)] * 100, 2),
            rnd(mean(b["dep"])), rnd(mean(b["rent"])), rnd(area_mean, 2),
            rnd(rate, 3), rnd(eq_mean, 2),
            rnd(mean(b["unit"]), 4), rnd(median(b["unit"]), 4),
            rnd(dep_share),
        ])

    out.sort(key=lambda x: (x[1], x[2], TYPES.index(x[3])))
    path = write_analysis(
        "서울_아파트_전월세_유형별_월별집계",
        ["광역지자체", "자치구", "연월", "계약유형", "건수", "비중(%)",
         "평균보증금(만원)", "평균월세(만원)", "평균전용면적(㎡)",
         "전월세전환율(연%)", "평균환산월세(만원)",
         "㎡당환산월세(만원)", "중위㎡당환산월세(만원)", "보증금기여율(%)"],
        out,
    )
    print(f"\n완료: {path} ({len(out)}행)")
    print(f"※ 갱신 {renewals:,}건 / 취소추정 {stale:,}건 제외")
    print("※ 자치구 빈칸 = 서울 전체. 전세 행의 환산월세는 보증금 전액을 월세화한 값")


if __name__ == "__main__":
    main()
