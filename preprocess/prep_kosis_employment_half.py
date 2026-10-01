# -*- coding: utf-8 -*-
"""
[전처리] 서울 구별 고용률·취업자 (지역별고용조사, 반기)

입력: raw:kosis/DT_1ES3A01S.csv   시군구 경제활동인구 총괄 (반기)
출력: processed/경제활동인구_서울구_반기.csv
      (광역지자체, 자치구, 기간, 주기, 통계명, 항목명, 값, 단위)
데이터사전: V013 고용률·취업자의 서울 구 단위 (시도 월별은 prep_kosis_employment_monthly)

이 데이터의 특성:
  - 서울 25개 구는 2021 상반기부터 있다.
  - 상반기(4월 조사)는 8월, 하반기(10월 조사)는 익년 2월 공표. 월별 패널에 붙이면
    공표 전 정보가 되므로 '2021H1'처럼 기준기간 그대로 둔다.
  - 표 내부 지역명이 '서울 종로구' 형태라 앞의 시도명을 뗀다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEOUL_GU, check_unique_keys, load_raw, write_processed  # noqa: E402

STAT = "지역별고용조사_서울구반기"
ITEMS = {"T3": ("취업자", "천명"), "T7": ("고용률", "%")}
GU_NAMES = set(SEOUL_GU.values())


def main():
    rows = []
    for r in load_raw("kosis", "DT_1ES3A01S"):
        name = (r.get("C1_NM") or "").replace("서울", "", 1).strip()
        item = ITEMS.get(r.get("ITM_ID"))
        if name not in GU_NAMES or item is None or r.get("DT") in ("", None, "-"):
            continue
        prd = r["PRD_DE"]  # 'YYYY01' 상반기 / 'YYYY02' 하반기
        rows.append(["서울", name, f"{prd[:4]}H{int(prd[4:6])}", "반기", STAT, item[0], r["DT"], item[1]])
    rows.sort(key=lambda x: (x[5], x[1], x[2]))
    check_unique_keys(rows, [0, 1, 2, 4, 5], STAT)
    path = write_processed("경제활동인구_서울구_반기",
                           ["광역지자체", "자치구", "기간", "주기", "통계명", "항목명", "값", "단위"], rows)
    print(f"완료: {path} ({len(rows)}행, {rows[0][2]}~{max(r[2] for r in rows)})")


if __name__ == "__main__":
    main()
