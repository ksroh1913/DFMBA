# -*- coding: utf-8 -*-
"""
[전처리] 연간 통계 - 지역소득·건축허가·주택보급률·임대주택 (17개 시도 / 서울 25개 구)

입력: raw:kosis/DT_1C96.csv            1인당 GRDP·가계총처분가능소득 (지역소득)     V015/V017
      raw:kosis/DT_201004_O090005.csv  서울 구별 주거용 신축 건축허가 (서울통계)    V024
      raw:kosis/DT_MLTM_2100.csv       국토부 (新)주택보급률 (시도, 2010~)           V072
      raw:kosis/DT_MLTM_6827·7174.csv  임대주택공급현황 (2020~ 민간 / 2024~ 전체)  V076
      raw:kosis/DT_MLTM_5560.csv       임대주택건설공급현황 (2012~2019)            V077
출력: processed/연간_소득생산공급.csv
      (광역지자체, 자치구, 기간, 주기, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 모두 연간이고 공표 시차가 1~2년이다. 월별 패널에 복제하지 않고 별도 시트로 병합한다.
    (기준연도에 값을 채우면 공표 전 정보를 쓰는 look-ahead가 된다)
  - 기관마다 지역 코드 체계가 달라(서울통계 001.., 총조사 11010..) 이름으로 맞춘다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEOUL_GU, check_unique_keys, load_raw, sido_short, write_processed  # noqa: E402

HEADER = ["광역지자체", "자치구", "기간", "주기", "통계명", "항목명", "값", "단위"]
GU_NAMES = set(SEOUL_GU.values())


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def seoul_stat_region(code, name):
    """서울통계(orgId 201) 코드: 001 = 서울시, 001xxx = 구. 본청(001026) 등은 제외."""
    if code == "001":
        return "서울", ""
    if code.startswith("001") and name in GU_NAMES:
        return "서울", name
    return None


def main():
    rows = []

    # V015/V017 지역소득 (시도)
    for r in load_raw("kosis", "DT_1C96"):
        sido, v = sido_short(r.get("C1_NM")), num(r.get("DT"))
        if sido and v is not None:
            rows.append([sido, "", r["PRD_DE"], "연", "지역소득_1인당", r.get("ITM_NM", ""), v,
                         r.get("UNIT_NM", "")])

    # V024 서울 구 주거용 신축허가
    for tbl, stat in [("DT_201004_O090005", "서울구_주거용신축허가")]:
        for r in load_raw("kosis", tbl):
            region, v = seoul_stat_region(r.get("C1", ""), r.get("C1_NM", "")), num(r.get("DT"))
            if region and v is not None:
                rows.append([*region, r["PRD_DE"], "연", stat, r.get("ITM_NM", ""), v,
                             r.get("UNIT_NM", "")])

    # V072 국토부 (新)주택보급률 (시도). 수도권·지방 등 집계행은 sido_short 에서 걸러진다
    for r in load_raw("kosis", "DT_MLTM_2100"):
        sido, v = sido_short(r.get("C1_NM")), num(r.get("DT"))
        if sido and v:  # 0 은 미작성 항목
            item = r.get("ITM_NM", "")
            unit = "%" if "보급률" in item else ("천가구" if "가구" in item else "천호")
            rows.append([sido, "", r["PRD_DE"], "연", "주택보급률_국토부", item, v, unit])

    # V077 임대주택건설공급현황 (2012~2019): 분류가 모두 '총계'인 시도 행만 수집했다
    for r in load_raw("kosis", "DT_MLTM_5560"):
        sido, v = sido_short(r.get("C1_NM")), num(r.get("DT"))
        if sido and v is not None:
            rows.append([sido, "", r["PRD_DE"], "연", "임대주택건설공급현황", "총계", v, "호"])

    # V076 임대주택공급현황: 표마다 포괄 범위가 달라 항목을 나눠 둔다(이어붙이지 않음)
    #   2020~ DT_MLTM_6827 민간임대만 -> 임대사업자 유형 x 주택유형 전체를 합산
    #   2024~ DT_MLTM_7174 공공+민간  -> 표가 제공하는 총계/계 행을 그대로 사용
    private = {}
    for r in load_raw("kosis", "DT_MLTM_6827"):
        sido, v = sido_short(r.get("C1_NM")), num(r.get("DT"))
        if sido and v is not None:
            private[(sido, r["PRD_DE"])] = private.get((sido, r["PRD_DE"]), 0) + v
    for (sido, y), v in private.items():
        rows.append([sido, "", y, "연", "임대주택공급현황", "민간임대(DT_MLTM_6827 합계)", v, "호"])
    total_rows = {
        ("임대주택 총계 (공공+민간)", "총계", "총계"): "공공+민간 총계(DT_MLTM_7174)",
        ("공공임대주택", "계", "계"): "공공임대 계(DT_MLTM_7174)",
        ("민간임대주택", "계", "계"): "민간임대 계(DT_MLTM_7174)",
    }
    for r in load_raw("kosis", "DT_MLTM_7174"):
        item = total_rows.get((r.get("C2_NM"), r.get("C3_NM"), r.get("ITM_NM")))
        sido, v = sido_short(r.get("C1_NM")), num(r.get("DT"))
        if item and sido and v is not None:
            rows.append([sido, "", r["PRD_DE"], "연", "임대주택공급현황", item, v, "호"])

    for r in rows:
        if isinstance(r[6], float) and r[6].is_integer():
            r[6] = int(r[6])
    rows.sort(key=lambda x: (x[4], x[5], x[0], x[1], x[2]))
    check_unique_keys(rows, [0, 1, 2, 4, 5], "연간_소득생산공급")
    path = write_processed("연간_소득생산공급", HEADER, rows)
    stats = {}
    for r in rows:
        stats.setdefault(r[4], set()).add(r[2])
    for k, ys in sorted(stats.items()):
        print(f"- {k}: {min(ys)}~{max(ys)}")
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
