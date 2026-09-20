# -*- coding: utf-8 -*-
"""
[전처리] 아파트 매매 실거래가 - 건별 원자료를 자치구·월별로 집계

입력: raw/molit/apt_trade_<시군구코드>.csv (건별)
출력: processed/서울_아파트_매매실거래_월별집계.csv

이 데이터의 특성:
  - 유일하게 "건별" 원자료라서 집계 기준을 바꿔도 재수집이 불필요하다
  - dealAmount는 만원 단위이고 천 단위 쉼표가 들어 있어 숫자 변환이 필요
  - 금액 분포가 오른쪽으로 길게 치우쳐 평균만으로는 대표성이 떨어지므로 중위값을 함께 낸다
  - 해제된 거래(cdealType='O')는 실제 성사되지 않은 계약이라 집계에서 제외
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEOUL_GU, load_raw, write_processed  # noqa: E402


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return ""
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    monthly = {}  # (자치구, 연월) -> {amounts, areas, unit_prices}
    cancelled = 0

    for lawd_cd, gu_name in SEOUL_GU.items():
        for row in load_raw("molit", f"apt_trade_{lawd_cd}"):
            if row.get("cdealType", "").strip() == "O":
                cancelled += 1
                continue  # 해제된 거래 제외

            try:
                amount = float(row["dealAmount"].replace(",", ""))  # 만원
                area = float(row["excluUseAr"])
                ym = f"{int(row['dealYear']):04d}-{int(row['dealMonth']):02d}"
            except (KeyError, ValueError):
                continue

            bucket = monthly.setdefault((gu_name, ym), {"amounts": [], "areas": [], "units": []})
            bucket["amounts"].append(amount)
            bucket["areas"].append(area)
            if area > 0:
                bucket["units"].append(amount / area)

    rows = []
    for (gu_name, ym), b in monthly.items():
        n = len(b["amounts"])
        if n == 0:
            continue
        rows.append([
            "서울", gu_name, ym, n,
            round(sum(b["amounts"]) / n, 1),
            round(median(b["amounts"]), 1),
            round(sum(b["areas"]) / len(b["areas"]), 2) if b["areas"] else "",
            round(sum(b["units"]) / len(b["units"]), 2) if b["units"] else "",
        ])

    rows.sort(key=lambda r: (r[1], r[2]))
    path = write_processed(
        "서울_아파트_매매실거래_월별집계",
        ["광역지자체", "자치구", "연월", "거래건수", "평균거래금액(만원)",
         "중위거래금액(만원)", "평균전용면적(㎡)", "평균단가(만원_전용㎡당)"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")
    print(f"※ 해제 거래 {cancelled}건 제외")


if __name__ == "__main__":
    main()
