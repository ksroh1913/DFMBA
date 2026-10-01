# -*- coding: utf-8 -*-
"""
[전처리] 전국 거시경제 지표 - 금리/대출금리/통화·신용/물가·임금/경기/금융시장/심리

입력: raw:ecos/<시리즈명>.csv  (collect_ecos.py 의 SERIES 참조)
출력: processed/ECOS_거시경제_전국.csv       월별 (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)
데이터사전: Variable_Master 중 ECOS 전국 계열 (시리즈 대응은 variables.py)
      processed/ECOS_거시경제_전국_분기.csv  분기 (광역지자체, 자치구, 기간, 주기, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 지역 구분이 없는 전국 단일 계열이라 광역지자체='전국'으로 고정
    (주택가격전망CSI는 서울/6대광역시/기타도시 권역값이지만 역시 한 계열씩이다)
  - 월별과 분기를 파일로 분리한다. 예전에는 분기값을 분기 시작월(1Q->01)에 넣었는데,
    분기값은 분기 종료 1.5개월 뒤에야 공표되므로 시작월에 두면 미래정보가 섞인다.
    분기는 '2026Q2'처럼 기준기간 그대로 두고, 월별 패널에 붙일지는 병합 단계에서 정한다.
  - 사업체노동력조사 임금은 산업분류 개정으로 표가 바뀐다(10차 2020~2025, 11차 2026~).
    '전체 산업·전규모' 합계라 분류 개정 영향이 작아 한 계열로 잇되, 2019년 이전은 ECOS에 없다.
  - 단위가 제각각(연%, 지수, 십억원, 천불)이라 스케일 조정 없이 단위를 함께 남긴다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check_unique_keys, load_raw, write_processed  # noqa: E402

# (raw 파일명, 출력 통계명, 주기)
SERIES = [
    ("기준금리", "한국은행_기준금리", "M"),
    ("시장금리_CD91일", "시장금리_CD91일", "M"),
    ("시장금리_국고채3년", "시장금리_국고채3년", "M"),
    ("예금은행대출금리_주택담보", "예금은행대출금리_주택담보", "M"),
    ("예금은행대출금리_전세자금", "예금은행대출금리_전세자금", "M"),
    ("비은행대출금리_주택담보", "비은행대출금리_주택담보", "M"),
    ("M2_통화량", "M2_통화량", "M"),
    ("예금취급기관_가계대출", "예금취급기관_가계대출", "M"),
    ("예금취급기관_주택관련대출", "예금취급기관_주택관련대출", "M"),
    ("정책대출_주금공_주택도시기금", "정책대출_주금공_주택도시기금", "M"),
    ("소비자물가지수", "소비자물가지수", "M"),
    ("임금총액_산업10차", "임금총액_사업체노동력", "M"),
    ("임금총액_산업11차", "임금총액_사업체노동력", "M"),
    ("선행지수순환변동치", "선행지수순환변동치", "M"),
    ("수출금액", "수출금액", "M"),
    ("KOSPI_종가", "KOSPI_종가", "M"),
    ("주택가격전망CSI_서울", "주택가격전망CSI_서울", "M"),
    ("주택가격전망CSI_6대광역시", "주택가격전망CSI_6대광역시", "M"),
    ("주택가격전망CSI_기타도시", "주택가격전망CSI_기타도시", "M"),
    ("가계신용_총액", "가계신용_총액", "Q"),
    ("대출태도_국내은행_가계주택", "대출태도_국내은행_가계주택", "Q"),
    ("시간당명목임금지수", "시간당명목임금지수", "Q"),
]


def main():
    monthly, quarterly = [], []
    for raw_name, out_name, cycle in SERIES:
        n = 0
        for row in load_raw("ecos", raw_name):
            value = row.get("DATA_VALUE", "")
            if value in ("", None):
                continue
            t = str(row.get("TIME", ""))
            item = row.get("ITEM_NAME2") or row.get("ITEM_NAME1") or out_name
            unit = (row.get("UNIT_NAME") or "").strip()
            if cycle == "M":
                monthly.append(["전국", "", f"{t[:4]}-{t[4:6]}", out_name, item, value, unit])
            else:
                quarterly.append(["전국", "", t, "분기", out_name, item, value, unit])
            n += 1
        print(f"- {out_name} ({raw_name}): {n}행")

    monthly.sort(key=lambda r: (r[3], r[2]))
    quarterly.sort(key=lambda r: (r[4], r[2]))
    check_unique_keys(monthly, [0, 1, 2, 3], "ECOS 월별")
    check_unique_keys(quarterly, [0, 1, 2, 4], "ECOS 분기")
    p1 = write_processed("ECOS_거시경제_전국",
                         ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"], monthly)
    p2 = write_processed("ECOS_거시경제_전국_분기",
                         ["광역지자체", "자치구", "기간", "주기", "통계명", "항목명", "값", "단위"],
                         quarterly)
    print(f"완료: {p1} ({len(monthly)}행)")
    print(f"완료: {p2} ({len(quarterly)}행)")


if __name__ == "__main__":
    main()
