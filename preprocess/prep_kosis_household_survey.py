# -*- coding: utf-8 -*-
"""
[전처리] 가계동향조사 - 가구당 월평균 소득 (도시, 2인 이상, 전체가구), 분기

입력: raw:kosis/DT_1L9I002.csv   1990Q1~2019Q4  (V074)
출력: processed/가계동향_소득_분기.csv
      (광역지자체, 자치구, 기간, 주기, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 데이터사전 V074의 수록 표는 2019Q4에서 끝나는 구계열이다. 2019년 표본개편 이후
    신계열(DT_1L9V121)은 조사 방식이 달라 사전 최종목록에 없으므로 받지 않는다.
  - 전국 단일값. 지역 패널에는 공통변수로만 쓸 수 있다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check_unique_keys, load_raw, write_processed  # noqa: E402

SOURCES = [
    ("DT_1L9I002", "가계동향_도시2인이상_소득"),
]


def main():
    rows = []
    for tbl, stat in SOURCES:
        n = 0
        for r in load_raw("kosis", tbl):
            if r.get("PRD_SE") not in ("Q", "분기") or r.get("DT") in ("", None, "-"):
                continue
            prd = r["PRD_DE"]  # 'YYYY0Q'
            period = f"{prd[:4]}Q{int(prd[4:6])}"
            rows.append(["전국", "", period, "분기", stat, r.get("C1_NM", "소득").strip(),
                         r["DT"], r.get("UNIT_NM", "")])
            n += 1
        print(f"- {stat}: {n}행")
    rows.sort(key=lambda x: (x[4], x[2]))
    check_unique_keys(rows, [0, 1, 2, 4, 5], "가계동향")
    path = write_processed("가계동향_소득_분기",
                           ["광역지자체", "자치구", "기간", "주기", "통계명", "항목명", "값", "단위"],
                           rows)
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
