# -*- coding: utf-8 -*-
"""
[전처리] 시도별 경제활동인구(월) - 고용률·취업자 등 9개 지표

입력: raw:kosis/DT_1DA7004S.csv   행정구역(시도)별 경제활동인구 (경제활동인구조사, 월)  (V013)
출력: processed/경제활동인구_시도_월별.csv  (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 시도 단위 월별 통계. 서울 구 단위는 없다(구 단위는 반기 지역별고용조사 = 경제활동인구_시계열.csv).
  - 시도 코드가 행정표준코드가 아니라 통계청 구코드(21 부산, 24 광주, 32 강원 ...)라서
    코드 대신 시도명으로 맞춘다.
  - 광주·전남 통합(2026-07) 이후에도 이 표는 '전남광주통합특별시' 아래에 광주·전남을
    따로 두므로 복원이 필요 없다. 통합 합계행은 제외한다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check_unique_keys, load_raw, sido_short, to_ym, write_processed  # noqa: E402

TBL = "DT_1DA7004S"
STAT = "경제활동인구_시도월별"


def main():
    rows = []
    for r in load_raw("kosis", TBL):
        sido = sido_short(r.get("C1_NM"))
        if sido is None or r.get("DT") in ("", None, "-"):
            continue
        rows.append([sido, "", to_ym(r["PRD_DE"]), STAT, r.get("ITM_NM", ""), r["DT"],
                     r.get("UNIT_NM", "")])
    rows.sort(key=lambda x: (x[4], x[0], x[2]))
    check_unique_keys(rows, [0, 1, 2, 3, 4], STAT)
    path = write_processed("경제활동인구_시도_월별",
                           ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"], rows)
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
