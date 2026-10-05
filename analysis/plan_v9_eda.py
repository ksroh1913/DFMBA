# -*- coding: utf-8 -*-
"""9차(보완) 설계용 자료 탐색. 수업자료(FDA 13a)의 절차대로 '분할 먼저, 탐색은 훈련 자료에서만' 을 지킨다.

입력: 데이터취합_전처리_20261005.xlsx (17개 시도 실거래 V006·V007 전국 취합 완료본). 8차 설정(config/plan_v8_settings.yaml)의
      처리 순서 ①~④(마스킹 → 시차 → 변환 → 기준 지역 분해)를 그대로 쓰고, 입력 파일만 20261005 로 바꾼다.
탐색 창: 결정월 <= 2017-12 (8차 최초 훈련기간 끝). 목표와의 관계는 정답 확인 시점 <= 2017-12 인 행만 쓴다.
         공식 지수 목표는 2016-01 이후만 있으므로(18~24개월) 구지수 연결 계열(2012.07~2017, 옛 지수가 있는 8개 시도 평균과 그 편차)을
         '참고' 로 함께 계산한다. 2015.06 이전 옛 지수는 서울·부산·대구·인천·광주·대전·울산·경기에만 있다.
         2018~2025 평가 구간의 목표값은 쓰지 않는다.

계산
  1. 변수별: 공통 성분(기준 지역 평균)과 전국 G_h 의 Spearman, 지역 성분(월내 편차)과 r_h 의 Spearman(풀링)과 월내 평균 순위상관,
     관측률, 사전 부호와의 일치.
  2. 변수 간: 공통 블록·지역 블록 각각의 |Spearman| 행렬(쌍마다 겹치는 관측 24개월 이상) → |ρ| >= 0.90 쌍 목록, >= 0.95 연결 성분(군집)과 대표 후보.
  3. 실거래 후보(RT_*): 월세 비중, 월세·전체·매매 건수의 로그Δ12, 천세대당 12개월합, 매매/전월세 비율 — 같은 통계와
     부동산원 거래현황(V035·V036)과의 중복 여부.
출력: analysis/output/plan_v9_eda_변수별.csv, _상관쌍.csv, _군집.csv, _실거래.csv, _요약.txt
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import settings as S, data as Dm, targets as T  # noqa: E402

OUT = os.path.join(BASE, "analysis", "output")
NEW_PRE, NEW_MERGED = "데이터취합_전처리_20261005.xlsx", "데이터취합_20261005.xlsx"
TRAIN_END = pd.Period("2017-12", "M")
PAIR_HI, CLUSTER, MIN_OVERLAP = 0.90, 0.95, 24
RT_COLS = {"rent": "V006_국토부실거래_아파트전월세_월세(보증부포함)_건", "all": "V006_국토부실거래_아파트전월세_전체_건",
           "sale": "V007_국토부실거래_아파트매매(해제제외)_건"}
REPORT_BREAK = pd.Period("2021-07", "M")   # 임대차 신고제(2021.06 계약분) 가 2차 행(1개월 밀림)에 처음 나타나는 달


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 12:
        return np.nan, int(m.sum())
    return float(stats.spearmanr(a[m], b[m])[0]), int(m.sum())


def within_month_rank(df, x, y):
    """월별 17개 시도 순위상관의 평균·표준편차·양수 비율 (월당 10개 이상 관측 → 공식 구간 2015.07 이후만 해당)"""
    vals = []
    for _, g in df.groupby("P"):
        m = g[x].notna() & g[y].notna()
        if m.sum() >= 10:
            vals.append(stats.spearmanr(g.loc[m, x], g.loc[m, y])[0])
    v = np.array(vals, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan, float(np.mean(v > 0)), len(v)


def rt_inputs(ml, hh):
    """실거래 후보 입력 (2차 값: 이미 1개월 밀림). 신고제 경계를 가로지르는 12개월 변화는 결측."""
    g = ml["region"]
    rent, allc, sale = (ml[RT_COLS[k]].astype(float) for k in ("rent", "all", "sale"))
    X = ml[["region", "P"]].copy()
    spec = []

    def add(name, val, vid, group, sign, tr):
        X[name] = val.values
        spec.append({"입력": name, "ID": vid, "원열": RT_COLS.get(vid.split("_")[1], ""), "변환": tr, "block": "regional",
                     "동인": 4 if vid != "RT_sale" and vid != "RT_saleratio" else 3, "하위묶음": group, "예상부호": sign})

    share = rent / allc.where(allc > 0)
    add("RT_share|수준", share, "RT_share", "전환구조", "±", "level_and_d12")
    add("RT_share|Δ12", share - share.groupby(g).shift(12), "RT_share", "전환구조", "±", "level_and_d12")
    for k, x in (("rent", rent), ("all", allc), ("sale", sale)):
        lx = np.log(x.where(x > 0))
        grp = "거래" if k != "sale" else "매매수급거래"
        add(f"RT_{k}|로그Δ12", 100 * (lx - lx.groupby(g).shift(12)), f"RT_{k}", grp, "±", "log12")
        add(f"RT_{k}|12개월합_천세대당", 1000 * Dm._roll12(x, g) / hh, f"RT_{k}", grp, "±", "flow_per_1000hh")
        # 단절에 강한 형태: 3개월합의 로그 12개월 변화 (수준을 쓰지 않으므로 지역별 영구 이동이 12개월 뒤 사라진다)
        s3 = x.groupby(g).transform(lambda v: v.rolling(3, min_periods=3).sum())
        l3 = np.log(s3.where(s3 > 0))
        add(f"RT_{k}|3개월합_로그Δ12", 100 * (l3 - l3.groupby(g).shift(12)), f"RT_{k}", grp, "±", "sum3_log12")
    ratio = sale / allc.where(allc > 0)
    add("RT_saleratio|수준", ratio, "RT_saleratio", "매매수급거래", "±", "level_and_d12")
    add("RT_saleratio|Δ12", ratio - ratio.groupby(g).shift(12), "RT_saleratio", "매매수급거래", "±", "level_and_d12")
    cross12 = (ml["P"] >= REPORT_BREAK) & (ml["P"] < REPORT_BREAK + 12)          # 12개월 변화·12개월합이 경계를 가로지르는 12행
    cross15 = (ml["P"] >= REPORT_BREAK) & (ml["P"] < REPORT_BREAK + 14)          # 3개월합의 12개월 변화: 비교 창 t-14~t 가 경계를 포함하는 14행
    for c in X.columns:
        if not c.startswith("RT_"):
            continue
        if "3개월합_로그Δ12" in c:
            X.loc[cross15, c] = np.nan
        elif "Δ12" in c or "12개월합" in c:
            X.loc[cross12, c] = np.nan
    return X, pd.DataFrame(spec)


def main():
    s = S.load("config/plan_v8_settings.yaml")   # 커밋된 plan_v9_eda_*.csv 는 8차 설정(V044 포함, 추가 입력 없음)에 20261005 입력으로 만든 것
    s["inputs"]["preprocessed_xlsx"], s["inputs"]["merged_xlsx"] = NEW_PRE, NEW_MERGED
    b = Dm.build(s, "sido")
    ml, X, spec, C, D = b["ml"], b["X"], b["spec"], b["C"], b["D"]
    hh = ml[s["inputs"]["households_col"]].astype(float)
    Xr, spec_r = rt_inputs(ml, hh)
    refs = {n: sorted(ml["region"].unique()) for n in spec_r["입력"]}
    Cr, Dr = Dm.decompose(s, Xr, spec_r, refs)
    C = C.join(Cr)
    D = D.merge(Dr, on=["region", "P"])
    spec_all = pd.concat([spec[spec["block"] != "price_trend"], spec_r], ignore_index=True)

    # ---- 목표. 공식(플래그) 과 구지수 연결(참고)
    pt = T.regional(s, ml)
    ycol = s["inputs"]["target"]["sido"]
    Rp = ml[ycol].astype(float)
    gp = Rp.groupby(ml["region"])
    proxy = ml[["region", "P"]].copy()
    hs = s["timing"]["horizons"]
    for h in hs:
        proxy[f"G{h}"] = (100 * (gp.shift(1 - h) / gp.shift(1) - 1)).values
        wide = proxy.pivot(index="P", columns="region", values=f"G{h}")
        # 구지수 연결 계열은 2015.06 이전에 8개 시도(서울·부산·대구·인천·광주·대전·울산·경기)만 있다. 그 8개가 모두 관측된 달의
        # 8개 평균을 참고 목표로 쓰고, 지역 성분은 그 8개 시도에서만 비교한다(2012.07~). 공식 구간(2015.07~)에서는 17개 평균
        proxy[f"Gbar{h}"] = proxy["P"].map(wide.mean(axis=1).where(wide.notna().sum(axis=1) >= 8)).values
        proxy[f"r{h}"] = proxy[f"G{h}"] - proxy[f"Gbar{h}"]
    nat = T.national(s)

    # ---- 1. 변수별 통계 (탐색 창)
    rows = []
    trainC = C[C.index <= TRAIN_END]
    for _, v in spec_all.iterrows():
        name = v["입력"]
        rec = {k: v[k] for k in ("입력", "ID", "변환", "block", "동인", "하위묶음", "예상부호")}
        rec["첫관측"] = str(C[name].first_valid_index()) if name in C and C[name].notna().any() else ""
        rec["관측률_탐색창_공통"] = round(float(trainC[name].notna().mean()), 3) if name in trainC else np.nan
        # 추세 지배 여부: 탐색 창에서 시간(월 순번)과의 Spearman. |ρ| 가 높으면 공통 블록의 수준값이 시간 추세와 구분되지 않는다
        rec["ρ공통_시간추세"], _ = sp(trainC[name].values, np.arange(len(trainC), dtype=float)) if name in trainC else (np.nan, 0)
        for h in hs:
            lab_ok = C.index + h <= TRAIN_END
            # 공통 성분 vs 전국 G_h (R-ONE 전국 공식) 과 Ḡ_h 구지수연결 참고
            cc = C.loc[lab_ok, name]
            gn = nat[f"G{h}"].reindex(cc.index)
            rec[f"ρ공통_G전국_h{h}"], rec[f"n공통_h{h}"] = sp(cc.values, gn.values)
            gpx = proxy.drop_duplicates("P").set_index("P")[f"Gbar{h}"].reindex(cc.index)
            rec[f"ρ공통_Ḡ구지수_h{h}"], rec[f"n공통_구지수_h{h}"] = sp(cc.values, gpx.values)
        if v["block"] == "regional":
            dd = D[["region", "P", name]]
            rec["관측률_탐색창_지역"] = round(float(dd.loc[dd["P"] <= TRAIN_END, name].notna().mean()), 3)
            for h in hs:
                m = dd.merge(pt[["region", "P", f"r{h}"]], on=["region", "P"]).merge(
                    proxy[["region", "P", f"r{h}"]].rename(columns={f"r{h}": f"r{h}_px"}), on=["region", "P"])
                m = m[m["P"] + h <= TRAIN_END]
                rec[f"ρ지역_r_h{h}"], rec[f"n지역_h{h}"] = sp(m[name].values, m[f"r{h}"].values)
                rec[f"ρ지역_r구지수_h{h}"], rec[f"n지역_구지수_h{h}"] = sp(m[name].values, m[f"r{h}_px"].values)
                (rec[f"월내순위ρ_h{h}"], rec[f"월내순위ρ_SD_h{h}"], rec[f"월내순위ρ_양수비율_h{h}"],
                 rec[f"월수_h{h}"]) = within_month_rank(m, name, f"r{h}")      # 공식 r (17개 시도, 2015.07~)
        r6 = rec.get("ρ공통_Ḡ구지수_h6", np.nan)
        rec["부호일치_공통_h6"] = "" if v["예상부호"] == "±" or not np.isfinite(r6) else ("일치" if np.sign(r6) == (1 if v["예상부호"] == "+" else -1) else "불일치")
        rows.append(rec)
    var = pd.DataFrame(rows)

    # ---- 2. 변수 간 상관 (탐색 창)
    def pair_table(frame, cols, block):
        sub = frame[cols]
        # 관측률로 입력을 거르지 않는다(짧은 계열 V005·V015·V017·V057·V059·V071 도 포함). 쌍마다 겹치는 관측이 MIN_OVERLAP 이상일 때만 계산
        rho = sub.corr(method="spearman", min_periods=MIN_OVERLAP).abs()
        pairs = []
        cs = list(rho.columns)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                r = rho.iloc[i, j]
                if np.isfinite(r) and r >= PAIR_HI:
                    a, bb = cs[i], cs[j]
                    da, db = spec_all.set_index("입력").loc[a, "동인"], spec_all.set_index("입력").loc[bb, "동인"]
                    n_ov = int((sub[a].notna() & sub[bb].notna()).sum())
                    pairs.append({"블록": block, "입력1": a, "입력2": bb, "|ρ|": round(float(r), 3), "n겹침": n_ov, "동인1": da, "동인2": db,
                                  "같은변수": a.split("|")[0] == bb.split("|")[0], "같은동인": da == db})
        # 연결 성분 (>= CLUSTER)
        parent = {c: c for c in cs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for p in pairs:
            if p["|ρ|"] >= CLUSTER:
                parent[find(p["입력1"])] = find(p["입력2"])
        comp = {}
        for c in cs:
            comp.setdefault(find(c), []).append(c)
        clusters = []
        first = {c: (sub[c].first_valid_index() if sub[c].notna().any() else None) for c in cs}
        for k, members in comp.items():
            if len(members) > 1:
                rep = sorted(members, key=lambda c: (str(first[c]), c))[0]      # 기계적 대표: 관측 시작이 이른 것 → 이름순 (최종 대표는 설계안 R2 기준)
                clusters.append({"블록": block, "크기": len(members), "구성": ", ".join(sorted(members)),
                                 "대표(기계적: 시작월→이름순)": rep, "동인": ", ".join(sorted({str(spec_all.set_index("입력").loc[c, "동인"]) for c in members}))})
        return pd.DataFrame(pairs).sort_values("|ρ|", ascending=False) if pairs else pd.DataFrame(), pd.DataFrame(clusters)

    ccols = [c for c in spec_all["입력"] if c in C.columns]
    pc, kc = pair_table(C[C.index <= TRAIN_END], ccols, "공통")
    dcols = [c for c in spec_all.loc[spec_all["block"] == "regional", "입력"] if c in D.columns]
    pdm, kd = pair_table(D[D["P"] <= TRAIN_END], dcols, "지역")
    pairs = pd.concat([pc, pdm], ignore_index=True)
    clusters = pd.concat([kc, kd], ignore_index=True)

    # ---- 3. 실거래 후보 요약 + 부동산원 거래현황과의 중복 (탐색 창과 전체 기간)
    rt = var[var["ID"].str.startswith("RT_")].copy()
    dup_rows = []
    for a, bb in (("RT_sale|12개월합_천세대당", "V036|12개월합_천세대당"), ("RT_all|12개월합_천세대당", "V035|12개월합_천세대당"),
                  ("RT_sale|로그Δ12", "V036|12개월합_천세대당"), ("RT_all|로그Δ12", "V035|12개월합_천세대당")):
        for nm, frame, mask in (("공통_탐색창", C, C.index <= TRAIN_END), ("공통_전체", C, C.index == C.index),
                                ("지역_탐색창", D, D["P"] <= TRAIN_END), ("지역_전체", D, D["P"] == D["P"])):
            f = frame.loc[mask]
            r, n = sp(f[a].values, f[bb].values)
            dup_rows.append({"입력1": a, "입력2": bb, "구간": nm, "ρ": round(r, 3) if np.isfinite(r) else np.nan, "n": n})
    dup = pd.DataFrame(dup_rows)

    os.makedirs(OUT, exist_ok=True)
    var.to_csv(os.path.join(OUT, "plan_v9_eda_변수별.csv"), index=False, encoding="utf-8-sig")
    pairs.to_csv(os.path.join(OUT, "plan_v9_eda_상관쌍.csv"), index=False, encoding="utf-8-sig")
    clusters.to_csv(os.path.join(OUT, "plan_v9_eda_군집.csv"), index=False, encoding="utf-8-sig")
    pd.concat([rt.assign(표="변수별"), dup.assign(표="중복")], ignore_index=True).to_csv(os.path.join(OUT, "plan_v9_eda_실거래.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 70)
    lines = []
    lines.append(f"입력 파일: {NEW_PRE} (sha {S.sha256(S.path(s, 'preprocessed_xlsx'))}); 탐색 창 결정월 <= {TRAIN_END}; 정답 확인 <= {TRAIN_END}")
    lines.append(f"입력 수: 공통 {len(ccols)}, 지역 {len(dcols)} (실거래 후보 {len(spec_r)} 포함)")
    keep = ["입력", "동인", "예상부호", "관측률_탐색창_공통", "ρ공통_시간추세", "ρ공통_G전국_h6", "n공통_h6", "ρ공통_Ḡ구지수_h6", "n공통_구지수_h6", "부호일치_공통_h6",
            "ρ지역_r_h6", "n지역_h6", "ρ지역_r구지수_h6", "n지역_구지수_h6", "월내순위ρ_h6", "월내순위ρ_SD_h6", "월내순위ρ_양수비율_h6", "월수_h6"]
    lines.append("\n[변수별, h=6] 공통 성분 vs 전국 G6 (공식, 2016~2017) / Ḡ6 구지수 연결(2012.07~2017, 8개 시도 평균, 참고); 지역 성분 vs r6 (공식, 풀링) / r6 구지수(8개 시도 편차 포함, 풀링) / 월내 순위상관(공식 r6, 월당 17개 시도, 2015.07~2017.06 의 24개월: 평균·SD·양수 비율)\n" + var[keep].round(2).to_string(index=False))
    lines.append(f"\n[상관쌍 |ρ| >= {PAIR_HI}] {len(pairs)} 쌍 (같은 변수의 수준·Δ12 쌍 {int(pairs['같은변수'].sum()) if len(pairs) else 0} 포함)\n" + (pairs.to_string(index=False) if len(pairs) else "없음"))
    lines.append(f"\n[군집 |ρ| >= {CLUSTER}]\n" + (clusters.to_string(index=False) if len(clusters) else "없음"))
    lines.append("\n[실거래 vs 부동산원 거래현황 중복]\n" + dup.to_string(index=False))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "plan_v9_eda_요약.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
