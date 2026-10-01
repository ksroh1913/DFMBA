# -*- coding: utf-8 -*-
"""
※ 데이터사전 변수가 아닌 분석용 산출물 (V006·V007 실거래 원본을 가공). 결과는 analysis/output/ 에 저장.
[전처리] 아파트 전월세 실거래 건별 -> 자치구·월별 집계 + 환산월세

입력: raw:molit/apt_rent_<시군구코드>.csv          (건별 원자료)
      processed/부동산원_전월세전환율_아파트.csv     (자치구별 월별 전환율, 연 %)
출력: analysis/output/서울_아파트_전월세실거래_월별집계.csv

계약 구분:
  전세 = monthlyRent 가 0        (원자료에 전세/월세 구분 필드가 없어 월세액으로 판정)
  월세 = monthlyRent 가 0보다 큼  (보증금이 있으면 전월세전환율로 월세화해 합산)

  환산월세 = 월세 + 보증금 x (전환율/100) / 12      <- 계약 건별로 계산
  ㎡당 환산월세 = 평균환산월세 / 월세계약 평균전용면적

집계본이 아니라 건별에서 계산하므로 이전 산출물의 두 가지 결함이 해소된다.
  - 면적: 전세·월세를 섞어 평균내지 않고 월세 계약의 면적만 쓴다.
  - 환산: 평균을 환산하는 게 아니라 계약마다 환산한 뒤 평균낸다.
    (보증금이 크면 월세가 작다는 계약별 역관계가 보존된다)

갱신계약(contractType='갱신')은 직전 계약 조건에 묶여 시세를 반영하지 않으므로 제외한다.
다만 이 필드는 2021.6 계약갱신청구권 도입 이후에만 채워진다. 그 이전 자료는 갱신 여부를
알 수 없어 전량 포함되므로 '계약구분기록여부'로 표시해 둔다(N인 구간은 신규·갱신 혼재).

취소·해제 처리: 전월세 API에는 해제 플래그가 없어 취소되면 응답에서 사라진다.
수집기가 남긴 _lastSeen 으로 월별 최신 확인분만 골라 현재 유효한 스냅샷을 쓴다.
"""

import csv
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PROCESSED_DIR, SEOUL_GU, raw_path, write_analysis  # noqa: E402

CONV_CSV = os.path.join(PROCESSED_DIR, "부동산원_전월세전환율_아파트.csv")


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


def main():
    rates = load_conversion_rates()
    buckets = defaultdict(lambda: {
        "jd": [], "ja": [],            # 전세 보증금 / 면적
        "wd": [], "wr": [], "wa": [],  # 월세 보증금 / 월세 / 면적
        "eq": [], "unit": [],          # 건별 환산월세 / 건별 ㎡당
        "typed": False,
    })
    renewals = stale = 0

    for lawd_cd, gu in SEOUL_GU.items():
        path = raw_path("molit", f"apt_rent_{lawd_cd}")
        if not os.path.exists(path):
            print(f"  ! raw 없음: {gu}")
            continue

        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        # 월별 최신 확인분만 사용 (그보다 오래된 _lastSeen = 응답에서 사라진 건 = 취소 추정)
        newest = defaultdict(str)
        for r in rows:
            ym = r.get("_dealYmd", "")
            newest[ym] = max(newest[ym], r.get("_lastSeen", ""))

        for r in rows:
            if r.get("_lastSeen", "") != newest[r.get("_dealYmd", "")]:
                stale += 1
                continue

            ctype = (r.get("contractType") or "").strip()
            if ctype == "갱신":
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

            b = buckets[(gu, ym)]
            if ctype:
                b["typed"] = True

            if rent > 0:
                b["wd"].append(deposit)
                b["wr"].append(rent)
                b["wa"].append(area)
                rate = rates.get((gu, ym))
                if rate is not None:
                    eq = rent + deposit * (rate / 100) / 12
                    b["eq"].append(eq)
                    b["unit"].append(eq / area)
            else:
                b["jd"].append(deposit)
                b["ja"].append(area)

        print(f"  - {gu}: {len(rows):,}건 처리")

    out = []
    for (gu, ym), b in buckets.items():
        if not b["jd"] and not b["wd"]:
            continue
        rate = rates.get((gu, ym))
        eq_mean = mean(b["eq"])
        wa_mean = mean(b["wa"])
        unit = eq_mean / wa_mean if eq_mean is not None and wa_mean else None
        dep_share = None
        if eq_mean and b["wd"] and rate is not None:
            dep_share = (mean(b["wd"]) * (rate / 100) / 12) / eq_mean * 100

        out.append([
            "서울", gu, ym,
            len(b["jd"]), rnd(mean(b["jd"])), rnd(median(b["jd"])), rnd(mean(b["ja"]), 2),
            len(b["wd"]), rnd(mean(b["wd"])), rnd(mean(b["wr"])), rnd(mean(b["wa"]), 2),
            rnd(rate, 3), rnd(eq_mean, 2), rnd(unit, 4), rnd(median(b["unit"]), 4), rnd(dep_share),
            "Y" if b["typed"] else "N",
        ])

    out.sort(key=lambda x: (x[1], x[2]))
    path = write_analysis(
        "서울_아파트_전월세실거래_월별집계",
        ["광역지자체", "자치구", "연월",
         "전세건수", "전세평균보증금(만원)", "전세중위보증금(만원)", "전세평균전용면적(㎡)",
         "월세건수", "월세평균보증금(만원)", "월세평균월세(만원)", "월세평균전용면적(㎡)",
         "전월세전환율(연%)", "평균환산월세(만원)", "㎡당환산월세(만원)", "중위㎡당환산월세(만원)",
         "보증금기여율(%)", "계약구분기록여부"],
        out,
    )
    print(f"\n완료: {path} ({len(out)}행)")
    print(f"※ 갱신계약 {renewals:,}건 제외 / 응답에서 사라진(취소 추정) {stale:,}건 제외")
    print("※ 면적은 월세 계약만, 환산은 계약 건별로 계산")


if __name__ == "__main__":
    main()
