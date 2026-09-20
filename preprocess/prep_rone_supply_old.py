# -*- coding: utf-8 -*-
"""
[전처리] 지역별 수급동향(구) - 2010~2015, (구)월세가격동향조사 계열

입력: raw/rone/T242093131851096.csv
출력: processed/부동산원_수급동향구_통합.csv

이 데이터의 특성:
  - 현행 수급동향과 달리 완성된 지수가 아니라, 응답 비율(수요우위/비슷함/공급우위)로
    제공되므로 표준 수급지수 = 100 + 수요우위 - 공급우위 로 환산해야 비교 가능하다.
  - 지역이 CLS가 아니라 GRP_NM에 들어 있고, CLS_NM에는 응답 구분이 들어 있다.
  - (구)월세가격동향조사 계열이라 현행 매매수급동향과 조사 모집단·설계가 다르다.
    기간(2012~2015)이 겹치지만 같은 통계로 이어붙이면 안 되므로 통계명을 분리한다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, parse_rone_region, to_ym, write_processed  # noqa: E402

STATBL_ID = "T242093131851096"
OUT_NAME = "수급동향(구)_통합"


def main():
    grouped = {}  # (시도, 자치구, 연월) -> {응답구분: 비율}
    for row in load_raw("rone", STATBL_ID):
        region = parse_rone_region(row.get("GRP_NM"))  # 이 표는 지역이 GRP에 있음
        if region is None:
            continue
        sido, gu = region
        key = (sido, gu, to_ym(row.get("WRTTIME_IDTFR_ID", "")))
        try:
            grouped.setdefault(key, {})[row.get("CLS_NM", "")] = float(row["DTA_VAL"])
        except (KeyError, TypeError, ValueError):
            continue  # 해당 시점에 값이 없는 지역(공란)은 건너뜀

    rows = []
    for (sido, gu, ym), vals in grouped.items():
        demand, supply = vals.get("수요우위"), vals.get("공급우위")
        if demand is None or supply is None:
            continue
        index = 100 + demand - supply
        rows.append([sido, gu, ym, OUT_NAME, "지수", round(index, 2), "지수"])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    path = write_processed(
        "부동산원_수급동향구_통합",
        ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")
    print("※ 현행 매매수급동향과는 별개 통계이므로 단순 연결하지 말 것")


if __name__ == "__main__":
    main()
