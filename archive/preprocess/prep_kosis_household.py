# -*- coding: utf-8 -*-
"""
[전처리] 주민등록세대수 - 서울 25개 자치구 + 16개 광역지자체, 월별

입력: raw/kosis/DT_1B040B3.csv
출력: processed/주민등록세대수_월별.csv

이 데이터의 특성:
  - 행정표준코드 체계를 그대로 사용: 2자리=시도, 5자리=시군구
  - 서울은 자치구(11xxx)까지 제공되므로 구 단위로 저장
  - 그 외 지역은 시도 합계행(2자리)만 사용 (시군구까지 내려가지 않음)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, to_ym, write_processed  # noqa: E402

TBL_ID = "DT_1B040B3"

# 행정표준코드 2자리 -> 약칭. '12 전남광주통합특별시'처럼 최근 신설된 통합코드는
# 기존 시도(29 광주, 46 전남)와 중복되므로 제외하고 기존 체계를 유지한다.
SIDO_CODE = {
    "11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주",
    "30": "대전", "31": "울산", "36": "세종", "41": "경기", "43": "충북",
    "44": "충남", "46": "전남", "47": "경북", "48": "경남", "50": "제주",
    "51": "강원", "52": "전북",
}


def main():
    rows = []
    for row in load_raw("kosis", TBL_ID):
        code = row.get("C1", "")
        prd = row.get("PRD_DE", "")
        value = row.get("DT", "")
        if not code or not prd or value == "":
            continue

        if len(code) == 2:
            # 전국(00)은 제외. 서울(11)은 자치구 행으로만 담으므로 시도 합계는 건너뜀.
            if code in ("00", "11") or code not in SIDO_CODE:
                continue
            rows.append([SIDO_CODE[code], "", to_ym(prd), value])
        elif len(code) == 5 and code.startswith("11"):
            rows.append(["서울", row.get("C1_NM", ""), to_ym(prd), value])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    path = write_processed(
        "주민등록세대수_월별",
        ["광역지자체", "자치구", "연월", "세대수"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
