# -*- coding: utf-8 -*-
"""
[전처리] 아파트 실거래 신고건수 - 전월세·매매, 17개 시도 + 서울 25개 구, 월별

입력: raw/4_임대시장수급전환구조/V006_아파트전월세실거래/<시도>/V006_<시군구코드>_<시군구>.csv   (V006)
      raw/3_금융여건상대가격/V007_아파트매매실거래/<시도>/V007_<시군구코드>_<시군구>.csv       (V007)
      시군구 목록은 sgg_codes.py
출력: processed/서울_아파트_실거래_월별건수.csv  (광역지자체, 자치구, 연월, 통계명, 항목명, 값, 단위)
      (파일명은 예전 그대로. 지금은 서울 구 행 + 17개 시도 행[자치구 빈칸]을 모두 담는다)

이 데이터의 특성:
  - 신고 원본의 행수이지 공식 거래량 통계가 아니다(부동산원 거래현황과 다를 수 있다).
  - 전월세: 월세액>0 이면 월세(보증부 포함), 0이면 전세.
    취소되면 응답에서 사라지므로 월별 최신 확인분(_lastSeen)만 센다.
  - 신규/갱신 건수는 취합하지 않는다. 계약구분(contractType) 칸이 2021.6 이전에는 계약의
    3.5~5.5%만 적혀 있고 이후에도 2023년까지 50~83%라, 건수가 실제 신규·갱신 건수를 나타내지 못한다.
  - 매매: 해제 건도 응답에 남아 cdealType='O'로 표시된다. 해제 건은 빼고 센다.
    해제 건수 자체는 월세 연구 주제와 관련이 낮아 취합하지 않는다.
  - 최근 2~3개월은 신고기한(30일)과 해제 시차 때문에 계속 늘어나거나 줄어든다.
  - 시도 행 = 그 시도 시군구 합계. API 가 과거 거래도 현재 시군구 코드로 돌려주므로(광주·전남 통합,
    강원·전북 코드 변경 포함) 현재 코드표(sgg_codes.py)의 시도로 묶으면 된다.
  - 시도는 소속 시군구를 전부 받은 경우에만 낸다(일부만 받은 시도는 합계가 작게 나오므로 빼고 경고).
    서울 외 시도는 2011.01부터 수집했다(서울 매매는 2006.01부터).
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "collect"))
from common import SEOUL_GU, load_raw, month_range, write_processed  # noqa: E402
from sgg_codes import SGG  # noqa: E402

KINDS = {"apt_rent": "전월세실거래_건수", "apt_trade": "매매실거래_건수"}
START = "201101"


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


def complete_codes(kind):
    """START~최근 수집월까지 모든 달을 조회한 시군구 코드 (조회기록 + 행이 있는 달 기준)"""
    log = load_raw("molit", f"{kind}_수집기록")
    asked = defaultdict(set)
    for r in log:
        asked[r["LAWD_CD"]].add(r["DEAL_YMD"])
    last = max((m for s in asked.values() for m in s), default=START)
    need = set(month_range(START, last))
    return {cd for cd in SGG if need <= asked.get(cd, set())}, last


def count_kind(kind, stat, counts):
    done, last = complete_codes(kind)
    for cd, (sido, _) in SGG.items():
        rows = load_raw("molit", f"{kind}_{cd}")
        if sido == "서울" and not rows:
            done.discard(cd)
        for r in current_snapshot(rows):
            ym = month_of(r)
            if not ym or ym < f"{START[:4]}-{START[4:]}":
                continue
            if kind == "apt_rent":
                rent = (r.get("monthlyRent") or "0").replace(",", "").strip() or "0"
                items = ["전체", "월세(보증부 포함)" if float(rent) > 0 else "전세"]
            else:
                if (r.get("cdealType") or "").strip() == "O":
                    continue  # 해제된 거래
                items = ["거래(해제 제외)"]
            gu = SEOUL_GU.get(cd, "")
            for it in items:
                counts[(sido, gu, ym, stat, it)] += 1
    # 서울은 기존 수집분(로그 이전)이 있어 행 존재로 완료 판단
    done |= {cd for cd in SEOUL_GU if load_raw("molit", f"{kind}_{cd}")}
    ok = {s for s in {v[0] for v in SGG.values()} if all(c in done for c, v in SGG.items() if v[0] == s)}
    return ok, last


def main():
    counts = defaultdict(int)  # (시도, 구, 연월, 통계명, 항목명) -> 건수
    rows = []
    for kind, stat in KINDS.items():
        ok, last = count_kind(kind, stat, counts)
        missing = sorted({v[0] for v in SGG.values()} - ok)
        print(f"- {stat}: 완료 시도 {len(ok)}/17 (최근 조회월 {last})" + (f", 미완료(제외) {missing}" if missing else ""))
        sido_tot = defaultdict(int)
        for (sido, gu, ym, st, it), n in counts.items():
            if st != stat:
                continue
            if sido == "서울" and gu:
                rows.append(["서울", gu, ym, st, it, n, "건"])
            if sido in ok:
                sido_tot[(sido, ym, it)] += n
        rows += [[s, "", ym, stat, it, n, "건"] for (s, ym, it), n in sido_tot.items()]

    rows.sort(key=lambda x: (x[3], x[4], x[0], x[1], x[2]))
    path = write_processed("서울_아파트_실거래_월별건수",
                           ["광역지자체", "자치구", "연월", "통계명", "항목명", "값", "단위"], rows)
    print(f"완료: {path} ({len(rows)}행)")


if __name__ == "__main__":
    main()
