# -*- coding: utf-8 -*-
"""
데이터사전 변수 점검 - variables.py 의 모든 Master ID가 processed 에 실제로 있는지 확인한다.

  python check_variables.py              # 공표 시차 점검 기준일 = 오늘
  python check_variables.py 2026-09-29   # 기준일 = raw 수집일

변수별로 행수, 시도 수, 서울 구 수, 기간을 출력하고, 연결이 끊긴 변수(파일 없음 /
통계명 불일치 / 행 0개)가 있으면 종료코드 1로 끝난다. 전처리를 다시 돌린 뒤 실행한다.
"""

import os
import sys

import pandas as pd

from common import BASE_DIR
from variables import PENDING, VARIABLES

_cache = {}


def load(rel):
    if rel not in _cache:
        path = os.path.join(BASE_DIR, rel)
        _cache[rel] = (pd.read_csv(path, encoding="utf-8-sig", dtype=str)
                       if os.path.exists(path) else None)
    return _cache[rel]


def select(v):
    df = load(v["processed"])
    if df is None:
        return None, "파일 없음"
    if v["stat"]:
        stats = v["stat"] if isinstance(v["stat"], list) else [v["stat"]]
        df = df[df["통계명"].isin(stats)]
    if v["items"]:
        df = df[df["항목명"].isin(v["items"])]
    if df.empty:
        return None, "행 없음(통계명/항목명 불일치)"
    return df, ""


def period_end_month(period):
    """'2026-08' / '2026Q2' / '2026H1' / '2024' -> 기준기간 마지막 달의 월 번호(연*12+월)"""
    y = int(period[:4])
    if len(period) == 7 and period[4] == "-":
        return y * 12 + int(period[5:7])
    if "Q" in period:
        return y * 12 + int(period[-1]) * 3
    if "H" in period:
        return y * 12 + int(period[-1]) * 6
    return y * 12 + 12


def released(period, rule, asof):
    """rule(lag, day)에 따라 기준일 asof 에 이 기간 값이 공표돼 있었는가"""
    now = asof.year * 12 + asof.month
    due = period_end_month(period) + rule["lag"]
    if due < now:
        return True
    day = rule.get("day", 1)
    return due == now and asof.day >= (31 if day is None else day)  # 일자 미상 = 말일로 보수 처리


def expected_latest(sample, rule, asof):
    """수집 여부와 무관하게, 규칙상 기준일에 공표돼 있어야 할 가장 최근 기간 (sample 과 같은 표기)"""
    y, m = asof.year, asof.month
    if len(sample) == 7 and sample[4] == "-":
        cands = [f"{(y * 12 + m - 1 - k) // 12:04d}-{(y * 12 + m - 1 - k) % 12 + 1:02d}" for k in range(-3, 36)]
    elif "Q" in sample:
        cands = [f"{yy}Q{q}" for yy in range(y + 1, y - 4, -1) for q in (4, 3, 2, 1)]
    elif "H" in sample:
        cands = [f"{yy}H{h}" for yy in range(y + 1, y - 4, -1) for h in (2, 1)]
    else:
        cands = [str(yy) for yy in range(y + 1, y - 5, -1)]
    cands.sort(key=period_end_month, reverse=True)
    return next((c for c in cands if released(c, rule, asof)), "-")


def check_release_lags(asof=None):
    """공표 시차 규칙으로 '기준일에 알 수 있는 최신 기간'을 계산해 실제 수집된 최신 기간과 대조.
    어긋나면 규칙이 틀렸거나(값이 규칙보다 먼저 있음 = 시차 과대추정) 수집이 밀린 것이다.
    asof 는 raw 수집일로 준다 (공표 당일 이후 수집하지 않았으면 당연히 아직 없음)."""
    from datetime import date
    from variables import BY_ID, RELEASE, RELEASE_MONTHLY
    asof = asof or date.today()
    print(f"\n[공표 시차 점검] 기준일(수집일) {asof}")
    bad = 0
    for vid, r in {**RELEASE_MONTHLY, **RELEASE}.items():
        df, err = select(BY_ID[vid])
        if df is None:
            print(f"  {vid:5} !! {err}")
            continue
        tcol = "연월" if "연월" in df.columns else "기간"
        periods = sorted(df[tcol].unique(), key=period_end_month)
        latest = periods[-1]
        if r.get("kind") == "수시":
            print(f"  {vid:5} --  수시 갱신 자료. 최신 {latest}은 신고기한(30일) 동안 계속 늘어남 ({r['verdict']})")
            continue
        exp_latest = expected_latest(latest, r, asof)
        if r.get("day", 1) is None:
            # 공식 자료가 '월'까지만 정한 경우: 공표일이 그 달 1일~말일 어디든 될 수 있으므로 두 경우 모두 허용
            early = expected_latest(latest, dict(r, day=1), asof)
            late = expected_latest(latest, dict(r, day=31), asof)
            exp_latest = latest if latest in (early, late) else late
        if r.get("ended"):
            print(f"  {vid:5} --  종료된 통계 (최신 {latest}). {r['verdict']}")
            continue
        ok = exp_latest == latest
        if not ok and r.get("known_delay"):
            print(f"  {vid:5} ~~  규칙상 최신 {exp_latest:7}  실제 최신 {latest:7}  알려진 수집 지연(수집원 게시가 공표보다 늦음)")
            continue
        bad += not ok
        note = ""
        if not ok:
            note = "  <- 규칙상 공표됐는데 수집본에 없음(수집 지연)" if period_end_month(exp_latest) > period_end_month(latest) \
                else "  <- 규칙보다 먼저 값이 있음(시차 과대추정)"
        d = r.get("day", "")
        rule = f"lag {r['lag']:>2}" + (f", {d:>2}일" if isinstance(d, int) else (", 월기준" if "day" in r else "     "))
        print(f"  {vid:5} {'OK' if ok else '!!'}  {rule}  규칙상 최신 {exp_latest:7}  실제 최신 {latest:7}  ({r['verdict']}){note}")
    print(f"  => 불일치 {bad}건")


def main():
    broken = []
    print(f"{'ID':5} {'상태':4} {'주기':2} {'행수':>8} {'시도':>4} {'서울구':>5}  기간             변수명")
    for v in VARIABLES:
        if v["status"] != "수집":
            tag = f"-> {v['same_as']}" if v["status"] == "중복" else v["note"]
            print(f"{v['id']:5} {v['status']:4} {v['freq']:2} {'':>8} {'':>4} {'':>5}  {tag}  {v['name']}")
            continue
        df, err = select(v)
        if df is None:
            broken.append((v["id"], err))
            print(f"{v['id']:5} !!   {v['freq']:2} {err}  {v['name']}")
            continue
        tcol = "연월" if "연월" in df.columns else "기간"
        gu = df["자치구"].fillna("")
        sido_n = df.loc[gu == "", "광역지자체"].nunique()
        gu_n = df.loc[gu != "", "자치구"].nunique()
        span = f"{df[tcol].min()}~{df[tcol].max()}"
        print(f"{v['id']:5} 수집 {v['freq']:2} {len(df):>8} {sido_n:>4} {gu_n:>5}  {span:16} {v['name']}")

    asof = None
    if len(sys.argv) > 1:
        from datetime import date
        asof = date.fromisoformat(sys.argv[1])
    check_release_lags(asof)
    print(f"\n미정의(추가검토) {len(PENDING)}건: " + " / ".join(PENDING))
    if broken:
        print(f"\n!! 연결 안 된 변수 {len(broken)}개: {broken}")
        sys.exit(1)
    ids = {v["id"][:4] for v in VARIABLES}
    n = len({v["id"][:4] for v in VARIABLES if v["status"] == "수집"})
    print(f"\n데이터사전 {len(ids)}개 중 {n}개 연결 확인 (나머지는 다른 변수와 같은 원자료인 중복 행)")


if __name__ == "__main__":
    main()
