# -*- coding: utf-8 -*-
"""
[전처리] 실거래 기반 환산월세 - 보증금을 전월세전환율로 월세화해 합산

입력: 서울_아파트_전월세_실거래가_월별집계.csv  (실거래 월별 집계)
      processed/부동산원_전월세전환율_아파트.csv (자치구별 월별 전환율, 연 %)
출력: processed/서울_아파트_환산월세_월별.csv

계산식:
  환산월세 = 실제월세 + 보증금 x (전환율/100) / 12
  단가     = 환산월세 / 평균전용면적            (만원/㎡/월)

보증금과 월세를 그대로 비교하면 반전세(보증금 크고 월세 적음)와 순수월세를
같은 척도로 놓을 수 없다. 전월세전환율은 부동산원이 지역·시점별로 산출하는
'보증금을 월세로 바꿀 때 적용되는 연이율'이라, 이걸로 보증금을 월세로 환산하면
계약 구조가 달라도 비교 가능한 단일 월세 지표가 된다.

--------------------------------------------------------------------------------
이 산출물의 한계 (집계본을 입력으로 쓰기 때문에 생기는 것)
--------------------------------------------------------------------------------
  - 평균전용면적이 전세·월세 계약을 함께 평균한 값이다. 월세만의 면적이 아니라
    단가에 편의가 있다. 건별 raw(raw/molit/)가 준비되면 월세 계약만으로 다시 계산해야 한다.
  - 보증금 평균과 월세 평균을 따로 환산해 더하므로, 계약별 '보증금 높으면 월세 낮다'는
    역관계가 사라진다. 평균의 환산이지 환산의 평균이 아니다.
  - 월세로 분류된 건에 준전세(보증금/월세 240배 초과)가 15% 이상 섞여 있다.
    환산월세는 이 문제를 상당 부분 흡수하지만, 유형별 분리 분석은 raw가 필요하다.
  - 전환율은 2026-07까지만 공표되어 그 이후 월(2026-08~09)은 산출 불가로 제외된다.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BASE_DIR, PROCESSED_DIR, write_processed  # noqa: E402

RENT_AGG = os.path.join(BASE_DIR, "서울_아파트_전월세_실거래가_월별집계.csv")
CONV_CSV = os.path.join(PROCESSED_DIR, "부동산원_전월세전환율_아파트.csv")


def load_conversion_rates():
    """(자치구, 연월) -> 전월세전환율(연 %)"""
    rates = {}
    with open(CONV_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r["자치구"]:
                continue  # 시도 합계행 제외
            try:
                rates[(r["자치구"], r["연월"])] = float(r["값"])
            except (TypeError, ValueError):
                continue
    return rates


def main():
    rates = load_conversion_rates()
    rows, skipped = [], 0

    with open(RENT_AGG, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            gu, ym = r["자치구"], r["연월"]
            rate = rates.get((gu, ym))
            if rate is None:
                skipped += 1
                continue
            try:
                n = int(r["월세건수"] or 0)
                deposit = float(r["월세평균보증금(만원)"])
                rent = float(r["월세평균월세(만원)"])
                area = float(r["평균전용면적(㎡)"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            if n == 0 or area <= 0:
                skipped += 1
                continue

            deposit_as_rent = deposit * (rate / 100) / 12
            equivalent = rent + deposit_as_rent

            rows.append([
                "서울", gu, ym, n,
                round(deposit, 1), round(rent, 1), round(rate, 3),
                round(deposit_as_rent, 2), round(equivalent, 2),
                round(area, 2), round(equivalent / area, 4),
                round(deposit_as_rent / equivalent * 100, 1),
            ])

    rows.sort(key=lambda x: (x[1], x[2]))
    path = write_processed(
        "서울_아파트_환산월세_월별",
        ["광역지자체", "자치구", "연월", "월세건수",
         "월세평균보증금(만원)", "실제월세(만원)", "전월세전환율(연%)",
         "보증금환산월세(만원)", "환산월세(만원)",
         "평균전용면적(㎡)", "㎡당환산월세(만원)", "보증금기여율(%)"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")
    print(f"※ 전환율 미공표 등으로 제외된 행: {skipped}건")
    print("※ 평균전용면적이 전세·월세 혼합 평균이라 단가에 편의 있음. raw 확보 후 재계산 필요")


if __name__ == "__main__":
    main()
