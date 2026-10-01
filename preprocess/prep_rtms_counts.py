# -*- coding: utf-8 -*-
"""
[전처리] 서울 아파트 실거래 신고건수 - 전월세·매매, 자치구·월별

입력: raw:molit/apt_rent_<시군구코드>.csv   (V006)
      raw:molit/apt_trade_<시군구코드>.csv  (V007)
출력: processed/서울_아파트_실거래_월별건수.csv  (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)

이 데이터의 특성:
  - 신고 원본의 행수이지 공식 거래량 통계가 아니다(부동산원 거래현황과 다를 수 있다).
  - 전월세: 월세액>0 이면 월세(보증부 포함), 0이면 전세.
    취소되면 응답에서 사라지므로 월별 최신 확인분(_lastSeen)만 센다.
  - 신규/갱신 건수는 취합하지 않는다. 계약구분(contractType) 칸이 2021.6 이전에는 계약의
    3.5~5.5%만 적혀 있고 이후에도 2023년까지 50~83%라, 건수가 실제 신규·갱신 건수를 나타내지 못한다.
  - 매매: 해제 건도 응답에 남아 cdealType='O'로 표시된다. 해제 건은 빼고 센다.
    해제 건수 자체는 월세 연구 주제와 관련이 낮아 취합하지 않는다.
  - 최근 2~3개월은 신고기한(30일)과 해제 시차 때문에 계속 늘어나거나 줄어든다.
  - 자치구 합계로 서울 행(자치구 빈칸)을 만든다.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEOUL_GU, load_raw, write_processed  # noqa: E402


def month_of(r):
    try:
        return f"{int(r['dealYear']):04d}-{int(r['dealMonth']):02d}"
    except (KeyError, ValueError):
        return None


def current_snapshot(rows):
    """월별 최신 _lastSeen 과 같은 행만 = 지금 유효한 계약"""
    newest = defaultdict(str)
    for r in rows:
        newest[r.get("_dealYmd", "")] = max(newest[r.get("_dealYmd", "")], r.get("_lastSeen", ""))
    return [r for r in rows if r.get("_lastSeen", "") == newest[r.get("_dealYmd", "")]]


def main():
    counts = defaultdict(int)  # (자치구, 연월, 통계명, 항목명) -> 건수

    for lawd_cd, gu in SEOUL_GU.items():
        for r in current_snapshot(load_raw("molit", f"apt_rent_{lawd_cd}")):
            ym = month_of(r)
            if not ym:
                continue
            rent = (r.get("monthlyRent") or "0").replace(",", "").strip() or "0"
            items = ["전체", "월세(보증부 포함)" if float(rent) > 0 else "전세"]
            for it in items:
                counts[(gu, ym, "전월세실거래_건수", it)] += 1

        for r in current_snapshot(load_raw("molit", f"apt_trade_{lawd_cd}")):
            ym = month_of(r)
            if not ym:
                continue
            if (r.get("cdealType") or "").strip() == "O":
                continue  # 해제된 거래
            counts[(gu, ym, "매매실거래_건수", "거래(해제 제외)")] += 1

    seoul = defaultdict(int)
    for (gu, ym, stat, item), n in counts.items():
        seoul[(ym, stat, item)] += n

    rows = [["서울", gu, ym, stat, item, n, "건"] for (gu, ym, stat, item), n in counts.items()]
    rows += [["서울", "", ym, stat, item, n, "건"] for (ym, stat, item), n in seoul.items()]
    rows.sort(key=lambda x: (x[3], x[4], x[1], x[2]))
    path = write_processed("서울_아파트_실거래_월별건수",
                           ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"], rows)
    stats = defaultdict(set)
    for r in rows:
        stats[(r[3], r[4])].add(r[2])
    for k, ms in sorted(stats.items()):
        print(f"- {k[0]} / {k[1]}: {min(ms)}~{max(ms)}")
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
