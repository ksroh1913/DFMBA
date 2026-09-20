# -*- coding: utf-8 -*-
"""
[전처리] 전국 거시경제 지표 - 금리/통화량/물가/가계신용/임금

입력: raw/ecos/기준금리.csv, 시장금리_CD91일.csv, 시장금리_국고채3년.csv,
      M2_통화량.csv, 소비자물가지수.csv, 가계신용_총액.csv, 시간당명목임금지수.csv
출력: processed/ECOS_거시경제_전국.csv

이 데이터의 특성:
  - 지역 구분이 없는 전국 단일 계열이라 광역지자체='전국'으로 고정
  - 주기가 섞여 있음(월/분기). 다른 데이터와 연월로 병합할 수 있도록
    분기값은 해당 분기 시작월(1Q->01, 2Q->04, 3Q->07, 4Q->10)로 정규화한다
  - 단위가 제각각(연%, 지수, 십억원)이라 값 스케일 조정은 하지 않고 단위를 함께 남긴다
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, write_processed  # noqa: E402

QUARTER_MONTH = {"1": "01", "2": "04", "3": "07", "4": "10"}

# (raw 파일명, 출력 통계명, 주기)
SERIES = [
    ("기준금리", "한국은행_기준금리", "M"),
    ("시장금리_CD91일", "시장금리_CD91일", "M"),
    ("시장금리_국고채3년", "시장금리_국고채3년", "M"),
    ("M2_통화량", "M2_통화량", "M"),
    ("소비자물가지수", "소비자물가지수", "M"),
    ("가계신용_총액", "가계신용_총액", "Q"),
    ("시간당명목임금지수", "시간당명목임금지수", "Q"),
]


def to_ym(time_value, cycle):
    t = str(time_value)
    if cycle == "M":
        return f"{t[:4]}-{t[4:6]}"
    if cycle == "Q":
        # ECOS 분기 표기는 '2012Q2' 형태
        quarter = t[5] if len(t) > 5 else "1"
        return f"{t[:4]}-{QUARTER_MONTH.get(quarter, '01')}"
    return f"{t[:4]}-01"


def main():
    rows = []
    for raw_name, out_name, cycle in SERIES:
        before = len(rows)
        for row in load_raw("ecos", raw_name):
            value = row.get("DATA_VALUE", "")
            if value in ("", None):
                continue
            rows.append(["전국", "", to_ym(row.get("TIME", ""), cycle), out_name,
                         row.get("ITEM_NAME1", out_name), value, row.get("UNIT_NAME", "")])
        print(f"- {out_name}: {len(rows) - before}행")

    rows.sort(key=lambda r: (r[3], r[2]))
    path = write_processed(
        "ECOS_거시경제_전국",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
