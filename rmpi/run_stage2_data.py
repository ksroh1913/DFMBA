# -*- coding: utf-8 -*-
"""
[2단계 자료 진단] 원자료·보완값 구분, 보완 셀 마스킹, 변환·동인 배정, 대용계열 접합 (연구계획 8차 확정본 11장 2단계).

출력 (rmpi/output/)
  stage2_변수표.csv        입력 열별 원열·변환·block·동인·하위묶음·예상부호·기준 지역·최초 훈련기간 관측률·처리
  stage2_제외열.csv        제외한 열과 이유
  stage2_마스킹기록.csv    미래정보 보완 셀의 기준기간 범위와 2차 결정월 범위
  stage2_자료기간표.csv    단위별 자료 기간 (전국·수도권·지방권·17개 시도·서울 구·매매·전세·옛 지수)
  stage2_접합판정.csv      h별 결정월 접합 판정과 첫 평가 시점 훈련 월 수
  stage2_타깃요약.csv      h별 G 행 수, Ḡ 월 수, 개발 성격 구간 사건 수 (plan_v5_numbers 와 대조)
  stage2_manifest.csv      입력 해시·커밋·패키지 버전
예측 성능은 계산하지 않는다.
"""

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402

EXCLUDE_REASON = {
    "V006": "실거래 신고 스냅숏(시도 94% 결측). 서울 구 보조에서만 검토",      # 8차
    "V007": "실거래 신고 스냅숏(시도 94% 결측)",                               # 8차
    "V007_v9": "국토부 매매 건수: 부동산원 매매거래현황(V036)과 0.98 중복(S2). 9차 주 분석 미사용",
    "V044": "CD91일 금리: 기준금리(V043)와 0.96, 같은 단기금리 측정(S2)",
    "V024": "서울 구 전용(시도 단위 없음)",
    "V034": "구 수급동향, 2015.06 종료",
    "V076": "2020년 이후만 (기간 짧음)",
    "V077": "2012~2019년만 (기간 짧음)",
    "V067": "순환변동치는 양방향 추세 제거로 과거 값이 다시 계산됨(구성상 미래정보)",
    "D_V0": "CSI 전신계열 더미: 공식 구간에서 상수",
}


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    os.makedirs(out, exist_ok=True)
    b = D.build(s, "sido")
    X, spec, C, Dd = b["X"], b["spec"], b["C"], b["D"]
    lo, hi = S.per(s["timing"]["official_first_decision"]), S.per(s["timing"]["first_train_end"])
    ev0, ev1 = S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"])

    # ---- 변수표
    Csub, Dsub = C.loc[lo:hi], Dd[(Dd.P >= lo) & (Dd.P <= hi)]
    Cev, Dev = C.loc[ev0:ev1], Dd[(Dd.P >= ev0) & (Dd.P <= ev1)]
    rows = []
    for _, r in spec.iterrows():
        name = r["입력"]
        rec = r.to_dict()
        rec["동인명"] = s["driver_names"][r["동인"]]
        if r["block"] == "regional":
            refs = b["ref_regions"][name]
            rec["기준지역수"] = len(refs)
            rec["기준지역제외"] = ",".join(g for g in s["inputs"]["regions"] if g not in refs)
            rec["관측률_최초훈련_공통성분"] = round(Csub[name].notna().mean(), 3)
            rec["관측률_최초훈련_지역성분"] = round(Dsub[name].notna().mean(), 3)
            rec["관측률_평가구간_지역성분"] = round(Dev[name].notna().mean(), 3)
            rec["사용"] = "공통 블록(기준 지역 평균) + 지역 블록(월내 편차)"
        elif r["block"] == "common":
            rec["기준지역수"] = ""
            rec["관측률_최초훈련_공통성분"] = round(Csub[name].notna().mean(), 3)
            rec["관측률_평가구간_공통성분"] = round(Cev[name].notna().mean(), 3)
            rec["사용"] = "공통 블록"
        elif r["block"] == "extra":
            refs = b["ref_regions"][name]
            rec["기준지역수"] = len(refs)
            rec["관측률_최초훈련_지역성분"] = round(Dsub[name].notna().mean(), 3)
            rec["관측률_평가구간_지역성분"] = round(Dev[name].notna().mean(), 3)
            rec["사용"] = "별도 입력(지역 블록 처리, RMPI 밖). M3·B+실거래 전용"
        else:
            rec["사용"] = "A(가격 추세) 전용. RMPI 제외"
        obs = X[(X.P >= lo)][name]
        rec["개정위험"] = "예" if r["ID"] in s["revision_risk"]["columns"] else ""
        rec["첫관측결정월"] = str(X.loc[X[name].notna(), "P"].min())
        rows.append(rec)
    vt = pd.DataFrame(rows)
    vt.to_csv(os.path.join(out, "stage2_변수표.csv"), index=False, encoding="utf-8-sig")

    # ---- 제외 열
    ex = []
    for c in s["excluded_columns"]:
        key = next((k for k in EXCLUDE_REASON if c.startswith(k)), "")
        ex.append({"열": c, "이유": EXCLUDE_REASON.get(key, "")})
    pd.DataFrame(ex).to_csv(os.path.join(out, "stage2_제외열.csv"), index=False, encoding="utf-8-sig")

    # ---- 마스킹 기록
    b["mask_log"].to_csv(os.path.join(out, "stage2_마스킹기록.csv"), index=False, encoding="utf-8-sig")

    # ---- 자료 기간표
    R = T.rone(S.path(s, "raw_rent"), "전국")
    rng = []

    def add(unit, series_name, ser, note=""):
        ser = ser.dropna()
        rng.append({"단위": unit, "계열": series_name, "시작": str(ser.index.min()), "끝": str(ser.index.max()),
                    "개월": int(len(ser)), "비고": note})
    add("전국", "월세통합지수(공식)", R, "주 분석 타깃 (RQ1)")
    for n in ("수도권", "지방권"):
        add(n, "월세통합지수(공식)", T.rone(S.path(s, "raw_rent"), n))
    add("전국", "매매가격지수", T.rone(S.path(s, "raw_sale"), "전국"), "A 입력(6개월 변화율)")
    add("전국", "전세가격지수", T.rone(S.path(s, "raw_jeonse"), "전국"), "A 입력(6개월 변화율)")
    for n in s["timing"].get("proxy_sensitivity", []):
        add(n, "옛 월세가격지수(집계)", T.rone(S.path(s, "raw_old_rent"), n), "대용계열(확장 실험, 학습만)")
    pt = T.regional(s, b["ml"])
    off = pt[pt.R_last.notna()]
    add("17개 시도", "월세통합지수(공식) R(t-1)", off.set_index("P")["R_last"].groupby(level=0).first(),
        f"{off.region.nunique()}개 시도, 결정월 기준")
    pd.DataFrame(rng).to_csv(os.path.join(out, "stage2_자료기간표.csv"), index=False, encoding="utf-8-sig")

    # ---- 접합 판정 (8차 확장 실험용. v9 설정에는 extension_first_decision 이 없어 건너뛴다)
    sp = []
    summ = []
    T0 = ev0
    for h in (s["timing"]["horizons"] if "extension_first_decision" in s["timing"] else []):
        months = pd.period_range("2011-01", T0 - h, freq="M")
        tb = T.splice_table(s, h, months)
        tb_ext = tb[tb.P >= S.per(s["timing"]["extension_first_decision"])]
        sp.append(tb_ext)
        n_new, n_old = int((tb.판정 == "공식").sum()), int((tb.판정 == "옛").sum())
        exc = tb_ext[tb_ext.판정 == "제외"].P
        kept15 = tb_ext[(tb_ext.판정 == "옛") & (tb_ext.P.dt.year == 2015)].P
        ef = T.extension_frame(s, h, "sensitivity")
        n_sens = int(((ef.index <= T0 - h) & (ef["판정"] == "옛")).sum())
        summ.append({"h": h, "공식 훈련월(2018.01)": n_new, "옛 체계 훈련월": n_old, "합계": n_new + n_old,
                     "제외 결정월": D._span(exc), "옛 체계로 남는 2015년": D._span(kept15),
                     "민감도 대용계열 옛 체계 훈련월": n_sens})
    if sp:
        pd.concat(sp).to_csv(os.path.join(out, "stage2_접합판정.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(summ).to_csv(os.path.join(out, "stage2_접합요약.csv"), index=False, encoding="utf-8-sig")

    # ---- v9 실거래 진단: 지역별 첫 관측, 경계 결측, 절단, 2021.06 단절 비율
    if "extra_inputs" in s:
        rc = s["rt_columns"]
        ml = b["ml"]
        rows_rt = []
        first = S.per(s["masking"]["rt_break_first_row"]); n_mask = s["masking"]["rt_mask_rows"]["sum3_d12"]
        for reg, g in ml.groupby("region"):
            g = g.set_index("P").sort_index()
            rent, tot = g[rc["월세"]].astype(float), g[rc["전체"]].astype(float)
            post = rent.loc["2021-08":"2022-01"].mean(); pre = rent.loc["2020-08":"2021-01"].mean()
            xr = X[X.region == reg].set_index("P")
            rows_rt.append({"region": reg, "월세건수_첫관측": str(rent.dropna().index.min()), "월세건수_2021H2/2020H2(2차행)": round(post / pre, 2) if pre else np.nan,
                            "월세비중_2016": round(float((rent / tot).loc["2016-01":"2016-12"].mean()), 3), "월세비중_2025": round(float((rent / tot).loc["2025-01":"2025-12"].mean()), 3),
                            "RT_mix_경계결측행": int(xr.loc[first: first + n_mask - 1, "RT_mix|Δ12"].isna().sum()),
                            "RT_mix_관측률_평가": round(float(xr.loc[ev0:ev1, "RT_mix|Δ12"].notna().mean()), 3), "RT_act_관측률_평가": round(float(xr.loc[ev0:ev1, "RT_act|Δ12"].notna().mean()), 3)})
        pd.DataFrame(rows_rt).to_csv(os.path.join(out, "stage2_실거래진단.csv"), index=False, encoding="utf-8-sig")

    # ---- 타깃 요약 (개발 성격 구간: 결정월 2016.01~, 정답 2020.12까지)
    dev_end = pd.Period("2020-12", "M")
    tr = []
    for h in s["timing"]["horizons"]:
        d = pt[(pt.P >= lo) & (pt[f"label_month{h}"] - 1 <= dev_end) & pt[f"G{h}"].notna()]
        n_months_gbar = int(pt.loc[pt[f"Gbar{h}"].notna(), "P"].nunique())
        ev_months = pt[(pt.P >= ev0) & (pt.P <= ev1) & pt[f"G{h}"].notna()].groupby("P").region.nunique()
        tr.append({"h": h, "G 행수(공식, 결정월 2016.01~)": int(pt[(pt.P >= lo) & pt[f"G{h}"].notna()].shape[0]),
                   "Ḡ 월수": n_months_gbar, "평가 결정월 중 17개 시도 모두 정답": f"{int((ev_months == 17).sum())}/{len(ev_months)}",
                   "개발구간 급락 비율": round(d[f"down{h}"].mean(), 3), "개발구간 급락 행": int(d[f"down{h}"].sum()),
                   "개발구간 급등 비율": round(d[f"up{h}"].mean(), 3), "개발구간 급등 행": int(d[f"up{h}"].sum()),
                   "급락 경계": round(T.thresholds(s)[h][0], 3), "급등 경계": round(T.thresholds(s)[h][1], 3)})
    pd.DataFrame(tr).to_csv(os.path.join(out, "stage2_타깃요약.csv"), index=False, encoding="utf-8-sig")

    # ---- 전국 계열 점검: 결정월 2016.01~ 의 past6 와 G 가 모두 있는지
    nat = T.national(s)
    chk = nat.loc[lo:ev1]
    S.manifest(s, {"stage": 2, "입력열": len(spec), "공통블록열": C.shape[1], "지역블록열": Dd.shape[1] - 2,
                   "전국표 past6 결측(2016.01~2025.12)": int(chk.past6.isna().sum()),
                   "전국표 G6 결측(2016.01~2025.12)": int(chk.G6.isna().sum())}).to_csv(
        os.path.join(out, "stage2_manifest.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).to_string(index=False))
    print(pd.DataFrame(tr).to_string(index=False))
    print(pd.DataFrame(rng).to_string(index=False))
    print(vt[["입력", "block", "동인", "하위묶음", "예상부호", "기준지역수", "관측률_최초훈련_공통성분"]].to_string(index=False))


if __name__ == "__main__":
    main()
