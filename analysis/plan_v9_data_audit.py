# -*- coding: utf-8 -*-
"""9차 보완 설계안 1장의 자료 점검 수치를 다시 계산한다 (데이터취합_전처리_20261004 → 20261005 의 차이, 실거래 열의 성질).

출력: analysis/output/plan_v9_data_audit.csv (항목, 값, 비고) 와 화면 출력.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "analysis", "output")
OLD, NEW = "데이터취합_전처리_20261004.xlsx", "데이터취합_전처리_20261005.xlsx"
SHEETS = ["17시도_1차_결측보완", "17시도_2차_공표시점반영(ML용)", "서울25구_1차_결측보완", "서울25구_2차_공표시점반영(ML용)"]
V = {"월세": "V006_국토부실거래_아파트전월세_월세(보증부포함)_건", "전세": "V006_국토부실거래_아파트전월세_전세_건",
     "전체": "V006_국토부실거래_아파트전월세_전체_건", "매매": "V007_국토부실거래_아파트매매(해제제외)_건"}
rows = []


def rec(item, value, note=""):
    rows.append({"항목": item, "값": value, "비고": note})
    print(f"{item}: {value}  {note}")


def main():
    # 1. 시트별 차이 (region, month 기준 정렬)
    for sh in SHEETS:
        o = pd.read_excel(os.path.join(BASE, OLD), sh)
        n = pd.read_excel(os.path.join(BASE, NEW), sh)
        m = o.merge(n, on=["region", "month"], how="outer", suffixes=("_o", "_n"), indicator=True)
        both = m[m["_merge"] == "both"]
        changed = []
        for c in n.columns:
            if c in ("region", "month") or c not in o.columns:
                continue
            a, b = both[c + "_o"], both[c + "_n"]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                d = ~((a == b) | (a.isna() & b.isna()))
                if d.sum():
                    changed.append(f"{c.split('_')[0]}{'·' + c.split('_')[-2] if c.startswith('V006') else ''}: {int(d.sum())}셀, "
                                   f"결측 {a.isna().mean():.3f}→{b.isna().mean():.3f} (겹치는 행), 새 파일 전체 {n[c].isna().mean():.3f} ({int(n[c].isna().sum())}셀)")
        new_months = sorted(m.loc[m["_merge"] == "right_only", "month"].unique())
        rec(f"[{sh}] 행", f"{len(o)} → {len(n)}", f"추가 월 {new_months}")
        rec(f"[{sh}] 값이 바뀐 열", "; ".join(changed) if changed else "없음")

    # 2. 실거래 열의 성질 (17개 시도 1차 시트)
    n1 = pd.read_excel(os.path.join(BASE, NEW), "17시도_1차_결측보완")
    n2 = pd.read_excel(os.path.join(BASE, NEW), "17시도_2차_공표시점반영(ML용)")
    g1 = pd.read_excel(os.path.join(BASE, NEW), "서울25구_1차_결측보완")
    cols = list(V.values())
    na = n1[cols].isna()
    rec("1차 실거래 결측 셀", int(na.sum().sum()), f"기간 {n1.month.min()}~{n1.month.max()}; 결측 행의 월 {sorted(n1.loc[na.any(axis=1), 'month'].unique())}; "
        f"열별 {dict(zip(['월세','전세','전체','매매'], na.sum().tolist()))}")
    rec("1차 실거래 0건 셀", int((n1[cols] == 0).sum().sum()))
    rec("1차 실거래 시도별 첫 관측월", str(sorted(n1.dropna(subset=[V['월세']]).groupby('region').month.min().unique())))
    # 2차 = 1차 를 몇 달 밀었나
    for k in ("월세", "매매"):
        p1 = n1.pivot(index="month", columns="region", values=V[k]); p2 = n2.pivot(index="month", columns="region", values=V[k])
        match = {s: float(np.nanmean((p1.shift(s).values == p2.values) | (np.isnan(p1.shift(s).values) & np.isnan(p2.values)))) for s in range(4)}
        rec(f"2차 밀림 일치율 ({k})", ", ".join(f"{s}개월 {v:.3f}" for s, v in match.items()))
    # 서울 = 구 합
    gs = g1.groupby("month")[cols].sum(); se = n1[n1.region == "서울"].set_index("month")[cols]
    ratio = (se / gs.reindex(se.index)).replace([np.inf, -np.inf], np.nan).dropna()
    rec("서울 시도값 / 25구 합 (월세, 전 기간 min~max)", f"{ratio[V['월세']].min():.3f}~{ratio[V['월세']].max():.3f}", f"n={len(ratio)}")
    # 절단: 최신 3개월 / 전년 동월
    for k in ("월세", "매매"):
        p = n1.pivot(index="month", columns="region", values=V[k])
        r = (p / p.shift(12)).tail(3)
        for mth, row in r.iterrows():
            row = row.dropna()
            rec(f"{k} 건수 {mth} / 전년 동월", f"{row.min():.2f}({row.idxmin()})~{row.max():.2f}({row.idxmax()})",
                f"중앙값 {row.median():.2f}, 1 미만 시도 {int((row < 1).sum())}/{len(row)}")
    # 신고제 단절: 2021H2 / 2020H2
    for k in ("월세", "전세"):
        p = n1.pivot(index="month", columns="region", values=V[k])
        r = p.loc[202107:202112].mean() / p.loc[202007:202012].mean()
        rec(f"{k} 건수 2021H2/2020H2", f"{r.min():.2f}({r.idxmin()})~{r.max():.2f}({r.idxmax()})", f"중앙값 {r.median():.2f}")
    tot = n1[n1.month <= 202608].groupby("month")[cols].sum()
    yr = tot.groupby(tot.index // 100).mean()
    rec("전국 월세 건수 월평균 2020/2021/2022", "/".join(f"{yr.loc[y, V['월세']]:,.0f}" for y in (2020, 2021, 2022)))
    rec("전국 매매 건수 월평균 2020/2022", f"{yr.loc[2020, V['매매']]:,.0f}/{yr.loc[2022, V['매매']]:,.0f}")
    # 월세 비중 추세
    share = n1.pivot(index="month", columns="region", values=V["월세"]) / n1.pivot(index="month", columns="region", values=V["전체"])
    sy = share.groupby(share.index // 100).mean()
    for y in (2012, 2025):
        rec(f"월세 비중 {y} 연평균", f"{sy.loc[y].min():.2f}({sy.loc[y].idxmin()})~{sy.loc[y].max():.2f}({sy.loc[y].idxmax()})")
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "plan_v9_data_audit.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    sys.exit(main())
