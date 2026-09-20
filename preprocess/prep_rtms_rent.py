# -*- coding: utf-8 -*-
"""
[전처리] 아파트 전월세 실거래가 - 건별 원자료를 자치구·월별로 집계 (갱신계약 제외)

입력: raw/molit/apt_rent_<시군구코드>.csv (건별)
출력: processed/서울_아파트_전월세실거래_월별집계.csv

이 데이터의 특성:
  - 전세와 월세가 한 표에 섞여 있고, monthlyRent=0이면 전세로 구분한다
  - contractType으로 신규/갱신을 구분하는데, 갱신계약은 직전 계약 조건에 묶여 있어
    시세를 반영하지 않으므로 제외한다
  - 단, 이 필드는 2021.6 계약갱신청구권 도입 이후에만 채워진다. 그 이전 자료는
    갱신 여부를 알 수 없어 전량 포함되므로 '계약구분기록여부'로 표시해 둔다
    (이 값이 N인 구간은 신규/갱신이 섞여 있다고 보고 해석해야 함)
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
    monthly = {}
    renewals = 0

    for lawd_cd, gu_name in SEOUL_GU.items():
        for row in load_raw("molit", f"apt_rent_{lawd_cd}"):
            contract_type = row.get("contractType", "").strip()
            if contract_type == "갱신":
                renewals += 1
                continue

            try:
                deposit = float(row["deposit"].replace(",", ""))
                area = float(row["excluUseAr"])
                monthly_rent = float((row.get("monthlyRent") or "0").replace(",", "") or 0)
                ym = f"{int(row['dealYear']):04d}-{int(row['dealMonth']):02d}"
            except (KeyError, ValueError):
                continue

            b = monthly.setdefault((gu_name, ym), {
                "jeonse": [], "wolse_dep": [], "wolse_rent": [], "areas": [], "typed": False,
            })
            if contract_type:
                b["typed"] = True
            b["areas"].append(area)
            if monthly_rent > 0:
                b["wolse_dep"].append(deposit)
                b["wolse_rent"].append(monthly_rent)
            else:
                b["jeonse"].append(deposit)

    rows = []
    for (gu_name, ym), b in monthly.items():
        if not b["jeonse"] and not b["wolse_dep"]:
            continue
        rows.append([
            "서울", gu_name, ym,
            len(b["jeonse"]),
            round(sum(b["jeonse"]) / len(b["jeonse"]), 1) if b["jeonse"] else "",
            round(median(b["jeonse"]), 1) if b["jeonse"] else "",
            len(b["wolse_dep"]),
            round(sum(b["wolse_dep"]) / len(b["wolse_dep"]), 1) if b["wolse_dep"] else "",
            round(sum(b["wolse_rent"]) / len(b["wolse_rent"]), 1) if b["wolse_rent"] else "",
            round(sum(b["areas"]) / len(b["areas"]), 2) if b["areas"] else "",
            "Y" if b["typed"] else "N",
        ])

    rows.sort(key=lambda r: (r[1], r[2]))
    path = write_processed(
        "서울_아파트_전월세실거래_월별집계",
        ["광역지자체", "자치구", "연월", "전세건수", "전세평균보증금(만원)", "전세중위보증금(만원)",
         "월세건수", "월세평균보증금(만원)", "월세평균월세(만원)", "평균전용면적(㎡)", "계약구분기록여부"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")
    print(f"※ 갱신계약 {renewals}건 제외 (신규계약만 집계)")


if __name__ == "__main__":
    main()
