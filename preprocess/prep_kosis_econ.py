# -*- coding: utf-8 -*-
"""
[전처리] 경제활동인구 - 연간/분기/반기 3개 표를 하나의 시계열로 결합

입력: raw/kosis/DT_1ES1B01S.csv (연간 ~2010, 시도)
      raw/kosis/DT_1ES2B01S.csv (분기 2011~2012, 시도)
      raw/kosis/DT_1ES3A01S.csv (반기 2013~, 시군구)
출력: processed/경제활동인구_시계열.csv

이 데이터의 특성:
  - KOSIS가 시기별로 표를 교체해서, 하나의 지표를 보려면 3개 표를 이어붙여야 함
  - 연간/분기 표는 성별(계/남자/여자) 축이 있어 '계'만 사용
  - 반기 표는 시군구 단위만 있고 시도 합계행이 없어, 시군구를 직접 합산해 시도값을 만듦
    (비율 지표는 단순 합산이 불가능하므로 인구수 계열 '천명'만 합산)
  - 지역별 커버리지 시작 시점이 다름:
      9개 도(경기/강원/충북/충남/전북/전남/경북/경남/제주) : 2013년~
      서울 및 6개 광역시(부산/대구/인천/광주/대전/울산)    : 2021년 상반기~
      세종                                                : 이 표에 항목 자체가 없어 전 기간 없음
    따라서 17개 시도 완전 패널이 아니다.
  - 주기가 서로 다르므로 연월은 해당 기간의 시작월로 정규화한다
    (1분기/상반기 -> 01월, 2분기 -> 04월, 3분기/하반기 -> 07월, 4분기 -> 10월)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_raw, write_processed  # noqa: E402

SIDO_FULLNAME = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
    "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
    "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
    "충청북도": "충북", "충청남도": "충남", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
    "강원도": "강원", "강원특별자치도": "강원",
    "전라북도": "전북", "전북특별자치도": "전북",
}

# 반기 표(DT_1ES3A01S)의 C1 코드 앞 2자리가 시도를 구분한다.
# 표 내부 전용 코드라 행정표준코드와 다르며, 실제 데이터로 검증해 확정함.
PREFIX_TO_SIDO = {
    "11": "서울", "21": "부산", "22": "대구", "23": "인천", "24": "광주",
    "25": "대전", "26": "울산", "31": "경기", "32": "강원", "33": "충북",
    "34": "충남", "35": "전북", "36": "전남", "37": "경북", "38": "경남", "39": "제주",
}

QUARTER_MONTH = {"01": "01", "02": "04", "03": "07", "04": "10"}
HALF_MONTH = {"01": "01", "02": "07"}


def prep_annual(rows):
    out = []
    for row in load_raw("kosis", "DT_1ES1B01S"):
        if row.get("C2_NM") not in (None, "", "계"):
            continue  # 성별 분리행 제외
        region = SIDO_FULLNAME.get(row.get("C1_NM", ""))
        if not region:
            continue  # '계' 등 전국 집계행 제외
        out.append((region, "", f"{row['PRD_DE']}-01", "연간",
                    row["ITM_NM"], row["DT"], row.get("UNIT_NM", "")))
    rows.extend(out)
    return len(out)


def prep_quarterly(rows):
    out = []
    for row in load_raw("kosis", "DT_1ES2B01S"):
        if row.get("C2_NM") not in (None, "", "계"):
            continue
        region = SIDO_FULLNAME.get(row.get("C1_NM", ""))
        if not region:
            continue
        prd = row["PRD_DE"]  # 예: 201202 = 2012년 2분기
        ym = f"{prd[:4]}-{QUARTER_MONTH.get(prd[4:], '01')}"
        out.append((region, "", ym, "분기", row["ITM_NM"], row["DT"], row.get("UNIT_NM", "")))
    rows.extend(out)
    return len(out)


def prep_halfyear(rows):
    """서울은 자치구 개별 행으로, 모든 시도는 시군구 합산으로 시도값을 산출"""
    gu_rows = []
    district_sums = {}  # (시도, 연월, 지표) -> 합계

    for row in load_raw("kosis", "DT_1ES3A01S"):
        sido = PREFIX_TO_SIDO.get(row.get("C1", "")[:2])
        if sido is None:
            continue
        prd = row["PRD_DE"]  # 예: 202501 = 2025년 상반기
        ym = f"{prd[:4]}-{HALF_MONTH.get(prd[4:], '01')}"
        itm = row["ITM_NM"]

        if sido == "서울":
            gu = row.get("C1_NM", "").replace("서울 ", "")
            gu_rows.append(("서울", gu, ym, "반기", itm, row["DT"], row.get("UNIT_NM", "")))

        if "천명" in itm:  # 비율(%)은 합산 불가하므로 인구수 계열만
            try:
                key = (sido, ym, itm)
                district_sums[key] = district_sums.get(key, 0.0) + float(row["DT"])
            except (TypeError, ValueError):
                pass

    sum_rows = [(sido, "", ym, "반기(구합산)", itm, round(total, 1), "")
                for (sido, ym, itm), total in district_sums.items()]

    rows.extend(gu_rows)
    rows.extend(sum_rows)
    return len(gu_rows), len(sum_rows)


def main():
    rows = []
    print(f"- 연간(DT_1ES1B01S): {prep_annual(rows)}행")
    print(f"- 분기(DT_1ES2B01S): {prep_quarterly(rows)}행")
    gu_n, sum_n = prep_halfyear(rows)
    print(f"- 반기(DT_1ES3A01S) 서울 자치구: {gu_n}행 / 시도 합산 산출: {sum_n}행")

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[4]))
    path = write_processed(
        "경제활동인구_시계열",
        ["광역지자체", "자치구", "연월", "주기", "지표명", "값", "단위"],
        rows,
    )
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
