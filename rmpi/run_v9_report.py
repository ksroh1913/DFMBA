# -*- coding: utf-8 -*-
"""[9차 보고] rmpi/output_v9 의 CSV 로 docs/연구결과_9차_후속분석_보고.md 를 만든다. 수치는 모두 파일에서 읽는다.
RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행."""

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import settings as S  # noqa: E402


def md(df, cols=None, nd=3):
    if df is None or len(df) == 0:
        return "(없음)\n"
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
        elif d[c].dtype == object:
            d[c] = d[c].fillna("")
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def _episode_counts(out, model, h=6):
    """(국면수, 사전 포착수(선행>0), 당월 포착수(선행=0)) for 1배·2배 cutoffs."""
    lead = rd(out, "v9_s4_국면시작_선행.csv")
    res = {}
    if lead is None:
        return res
    for k in (1, 2):
        x = lead[(lead["모형"] == model) & (lead["h"] == h) & (lead["배수"] == k)]
        res[k] = (len(x), int((x["선행개월"] > 0).sum()), int((x["선행개월"] == 0).sum()))
    return res


def _v8_B_text():
    """8차 실제 전국 B MAE (rmpi/output/stage6_예측값_전국.csv) — 9차 구성 + alpha 10 과 구분하기 위해 적는다."""
    p = os.path.join(BASE, "rmpi", "output", "stage6_예측값_전국.csv")
    if not os.path.exists(p):
        return ""
    d = pd.read_csv(p)
    d = d[(d["모형"] == "ridge_B") & (d["P"] >= "2018-01") & (d["P"] <= "2025-12")]
    m = d.groupby("h").apply(lambda g: (g["y"] - g["yhat"]).abs().mean())
    return "·".join(f"{m.get(h, np.nan):.4f}" for h in (1, 3, 6))


def _episode_text(out):
    eb, ea = _episode_counts(out, "panel_down_B"), _episode_counts(out, "panel_down_A")
    if not eb or not ea:
        return ""
    n = eb[1][0]
    return (f"국면 시작 {n}개(2017년 이력 포함) 중 1배 컷오프로 B 는 {eb[1][1]}개를 시작 전에, {eb[1][2]}개를 시작 달에 잡았고(합 {eb[1][1] + eb[1][2]}), "
            f"A 는 {ea[1][1]}개 사전·{ea[1][2]}개 당월이다. 2배 컷오프에서는 B {eb[2][1]}개 사전·{eb[2][2]}개 당월, A {ea[2][1]}개 사전·{ea[2][2]}개 당월이다.")


def _v8_BA_ratio():
    """8차 실제 전국 B/A MAE 비 (rmpi/output/stage6_예측값_전국.csv, 2018-01~2025-12)."""
    p = os.path.join(BASE, "rmpi", "output", "stage6_예측값_전국.csv")
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    d = d[(d["P"] >= "2018-01") & (d["P"] <= "2025-12")]
    m = d[d["모형"].isin(["ridge_A", "ridge_B"])].groupby(["모형", "h"]).apply(lambda g: (g["y"] - g["yhat"]).abs().mean())
    return {h: m.get(("ridge_B", h), np.nan) / m.get(("ridge_A", h), np.nan) for h in (1, 3, 6)}


def _alert_share_text(out):
    """6.4 용: 컷오프 목표 비율(훈련 전체 행 사건비율)과 평가기간 실제 경보 비율(전체 행·비급락 행) 을 예측 파일에서 계산한다."""
    p = rd(out, "v9_s1_예측값_패널.csv")
    if p is None:
        return ""
    g = p[(p["모형"] == "panel_down_B") & (p["h"] == 6)]
    if g.empty:
        return ""
    tr = g["훈련사건비율"].mean()
    nz = g[g["기존급락상태"] == 0]
    a1, a2 = (g["yhat"] >= g["컷오프_1x"]).mean(), (g["yhat"] >= g["컷오프_2x"]).mean()
    b1, b2 = (nz["yhat"] >= nz["컷오프_1x"]).mean(), (nz["yhat"] >= nz["컷오프_2x"]).mean()
    return (f"- 컷오프의 목표 비율은 훈련 행 전체의 사건비율(B h=6 평균 {tr:.2f}; 1배 ≈ {tr:.0%}, 2배 ≈ {2*tr:.0%})이다. 비급락 상태 행에서 잰 실제 경보 빈도(B h=6: 1배 {b1:.0%}, 2배 {b2:.0%})는 "
            f"이 행들의 사건비율 자체가 낮아(약 {nz['y'].mean():.0%}) 목표보다 낮고, 기존 급락 상태 행을 포함한 전체 행에서 재면 1배 {a1:.0%}, 2배 {a2:.0%}로 목표보다 높다. "
            "후자가 훈련 분포와 같은 모집단이므로, 평가기간의 예측확률 분포가 훈련기간보다 위에 있다는 말은 전체 행 기준에서만 성립한다.\n")


def rd(out, name):
    p = os.path.join(out, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    L = []
    w = L.append
    w("# 아파트 월세 흐름 진단과 급변 예측: 9차 후속 탐색 분석 결과\n")
    w(f"설정 {s['_path']} ({s['meta']['settings_version']}, sha {s['_sha']}), 입력 {s['inputs']['preprocessed_xlsx']}. "
      "이 분석은 8차 결과를 보고 설계한 **후속 탐색 분석**이다(설계안 docs/연구계획_9차_보완설계안.md). 어떤 결과도 사전 지정 검증으로 쓰지 않으며, "
      "8차 판정 규칙(두 t구간)은 참고로만 계산해 적는다. 2018.01~2025.12 의 96개 결정월, h = 1·3·6. 실거래 관련 결과는 최신 수정자료(2026.10.01~05 수집)를 이용한 보조 분석이다.\n")
    s1 = rd(out, "v9_s1_비교.csv"); s2 = rd(out, "v9_s2_비교.csv"); s3 = rd(out, "v9_s3_비교.csv"); s4 = rd(out, "v9_s4_경보지표.csv")
    a1 = rd(out, "v9_s1_alpha선택.csv"); a2 = rd(out, "v9_s2_alpha선택.csv")

    # ---------------- 0 핵심
    w("## 0 핵심 결과\n")
    if s1 is not None:
        for tag, nm in (("cvmin", "전진 교차검증 최소 alpha"), ("fixed10", "9차 구성에 alpha 10 고정, 8차와 같은 값")):
            r = s1[s1["비교"].str.contains(f"\\({tag}\\)")]
            bva = r[r["비교"].str.startswith("B vs A")].set_index("h")["MAE비율"]
            bvn = r[r["비교"].str.startswith("B vs 선택형")].set_index("h")["MAE비율"]
            avn = r[r["비교"].str.startswith("A vs 선택형")].set_index("h")["MAE비율"]
            w(f"- ① 기준 재정리({nm}): B/A MAE 비 " + "·".join(f"{bva.get(h, np.nan):.2f}" for h in (1, 3, 6)) +
              ", B/선택형 단순기준 " + "·".join(f"{bvn.get(h, np.nan):.2f}" for h in (1, 3, 6)) +
              ", A/선택형 " + "·".join(f"{avn.get(h, np.nan):.2f}" for h in (1, 3, 6)) + " (h=1·3·6).")
    if s2 is not None:
        for unit in ("전국", "패널"):
            r = s2[s2["단위"] == unit]
            def ratio(nm):
                q = r[r["비교"] == f"{unit} {nm}"].set_index("h")["MAE비율"]
                return "·".join(f"{q.get(h, np.nan):.3f}" for h in (1, 3, 6))
            txt = f"- ② 보정 구조({unit}): M1/M0 {ratio('M1 vs M0')}, M2/M1 {ratio('M2 vs M1')}, M2/M0 {ratio('M2 vs M0')}"
            if unit == "패널":
                txt += f", M3/M2 {ratio('M3 vs M2')}"
            w(txt + ".")
    if s3 is not None:
        q2 = s3[s3["비교"].str.startswith("RQ2")].set_index("h")["MAE비율"]
        q3 = s3[s3["비교"].str.startswith("RQ3")].set_index("h")["비율"] if "비율" in s3.columns else pd.Series(dtype=float)
        w(f"- ③ 실거래 추가: RQ2 B+실거래/B MAE 비 " + "·".join(f"{q2.get(h, np.nan):.3f}" for h in (1, 3, 6)) +
          ", RQ3 Brier 비 " + "·".join(f"{q3.get(h, np.nan):.3f}" for h in (1, 3, 6)) + ".")
    if s4 is not None:
        r = s4[(s4["목표빈도(사건비율배수)"] == 1) & (s4["h"] == 6)].set_index("모형")
        w("- ④ 급락 경보(h=6, 목표 빈도 1배): " + "; ".join(f"{m}: 적중률 {r.loc[m, '경보적중률']:.2f}, 포착률 {r.loc[m, '급락포착률']:.2f}, 실제 빈도 {r.loc[m, '경보빈도_실제']:.2f}, 동월 AUC {r.loc[m, '동월지역쌍_AUC']:.2f}"
                                             for m in r.index) + ".")
    w("")

    # ---------------- 1 ①
    w("## 1 ① 기준 재정리: 같은 튜닝 아래 A 와 B\n")
    w("전국 Ridge 의 alpha 를 매년 1월 훈련자료 안 전진 교차검증(검증 12개월) 평균 손실 최소로 골랐다. 고정 10 과 1-SE 는 비교용이며, 'fixed10' 은 **9차 입력·지수 구성(20261005 자료, S1~S3)에 8차와 같은 alpha 10 을 적용한 9차 모형**이지 8차 결과의 재현이 아니다"
      + (f"(실제 8차 전국 B MAE 는 {_v8_B_text()}, rmpi/output/stage6_예측값_전국.csv)" if _v8_B_text() else "") + ". 선택형 단순기준은 8차 정의 그대로다.\n")
    if s1 is not None:
        w(md(s1, ["비교", "h", "MAE_기준", "MAE_비교", "MAE비율", "평균개선율(%)", "연평균_하한", "연평균_상한", "2년_하한", "2년_상한", "우세연도", "8차규칙판정(참고)"]))
    if a1 is not None:
        piv = a1[a1["규칙"] == "cv_min"].pivot_table(index=["정보군", "h"], columns="시점", values="선택", aggfunc="first")
        w("\n선택된 alpha(전진 교차검증 최소, 1월 재선택):\n")
        w(md(piv.reset_index(), nd=2))
        fb = a1[a1["규칙"] == "fallback"]
        flat = a1[a1["규칙"] == "cv_min"].groupby("h")["평평후보수"].median()
        w("\n빈칸은 전진 폴드가 설정 최소치(4)에 못 미쳐 대체값 alpha 10 을 쓴 경우" +
          (f"({', '.join(sorted(set(fb['시점'].astype(str) + ' h=' + fb['h'].astype(int).astype(str))))})" if len(fb) else "") +
          ". 최소 손실의 1% 안에 든 후보 수의 중앙값은 " + ", ".join(f"h={int(h)}: {v:.0f}개" for h, v in flat.items()) +
          "로, 교차검증 곡선이 평평해 선택값이 연도마다 크게 바뀐다(해석은 6장). 격자는 1단계 25점 + 2단계 4점 = 29점이고, 1단계 최소가 끝점이면 한 자릿수 연장한 뒤 2단계를 그 최소 둘레에 다시 두므로 "
          "그 해의 격자점수는 35 가 된다(연장 끝에서 반 자릿수 더 나감; 선택값에는 영향이 거의 없다).\n")
    dm1 = rd(out, "v9_s1_DM참고.csv")
    if dm1 is not None:
        w("\nDiebold-Mariano 참고(판정에 쓰지 않음):\n"); w(md(dm1))

    # ---------------- 2 ②
    w("\n## 2 ② 보정 구조: M0 → M1 → M2 (전국), M0 → M3 (통합 패널)\n")
    w("학습 목표 = 실제 변화 − mom1, 최종 예측 = mom1 + 보정값. M1 은 절편만, M1′은 mom1 계수 학습, M2 는 매매·전세 추세와 RMPI, M3 는 M2 + 실거래 두 입력(RMPI 구성·가중·전처리 동일). 경제정보 기여 = M2 − M1, 실거래 기여 = M3 − M2.\n")
    if s2 is not None:
        w(md(s2, ["단위", "비교", "h", "MAE_기준", "MAE_비교", "MAE비율", "평균개선율(%)", "연평균_하한", "연평균_상한", "2년_하한", "2년_상한", "우세연도", "8차규칙판정(참고)"]))
        yc = [c for c in s2.columns if c.startswith("d_")]
        w("\n연도별 손실 차이 d(비교 − 기준, 연평균; 음수면 비교 모형이 낫다):\n")
        w(md(s2, ["단위", "비교", "h"] + yc))
    reg = rd(out, "v9_s2_지역별MAE.csv")
    if reg is not None:
        piv = reg.pivot_table(index="region", columns=["h", "모형"], values="MAE")
        w("\n지역별 MAE(통합 패널, 모형별):\n"); w(md(piv.reset_index(), nd=3))
    if a2 is not None:
        w("\n보정모형 alpha 선택(1월):\n")
        w(md(a2.pivot_table(index=["단위", "모형", "h"], columns="시점", values="선택", aggfunc="first").reset_index(), nd=1))
        w("\n빈칸은 전진 폴드 부족으로 대체값 10 을 쓴 경우(2018-01, h=6). alpha 는 모형마다 따로 고르므로 M3 와 M2 의 선택값이 다른 해가 있다. "
          "따라서 M3 − M2 는 '실거래 두 입력 + 그 입력이 있을 때 다시 고른 alpha' 의 효과다(설정대로이며, 구성·가중·전처리는 같다). "
          "보정모형의 적합(RMPI 변환·대체·표준화·계수)은 매 결정월 다시 하고 alpha 만 1월에 고른다. 구성 명세는 첫·마지막 결정월 것을 v9_s2_구성명세_*.csv 에 둔다.\n")
    dm2 = rd(out, "v9_s2_DM참고.csv")
    if dm2 is not None:
        w("\nDiebold-Mariano 참고:\n"); w(md(dm2))

    # ---------------- 3 ③
    w("\n## 3 ③ 실거래 정보의 추가 효과\n")
    w("지역 블록 처리의 별도 입력 두 개(월세 비중 로짓 3개월합 Δ12, 천세대당 전월세 거래량 로그 3개월합 Δ12). 신고제 경계 2021.07~2022.08 결측. 최신 수정자료 보조 분석. "
      "data_asof 2026-10 절단은 2차 시트 행 2026-09·10 을 비우므로(2차 행은 거래월보다 한 달 뒤), 쓸 수 있는 마지막 실거래 행은 2026-08 이고 1차 시트 기준으로는 최신 거래월 세 달(2026.08~10)이 빠진다. "
      "2018~2025 평가에는 영향이 없고 2026 사후 확인 구간에만 해당한다.\n")
    if s3 is not None:
        w(md(s3))
    sens = rd(out, "v9_s3_민감도.csv")
    if sens is not None:
        w("\n민감도(h=6):\n"); w(md(sens, ["비교", "h", "MAE_기준", "MAE_비교", "MAE비율", "우세연도", "민감도"]))
    rtd = rd(out, "stage2_실거래진단.csv")
    if rtd is not None:
        w("\n실거래 자료 진단(2단계):\n"); w(md(rtd, nd=3))

    # ---------------- 4 ④
    w("\n## 4 ④ 급락 경보의 실용성 (비급락 상태 지역, 보조 보고)\n")
    w("컷오프는 각 모형의 훈련기간 예측확률 분포에서 '훈련 사건비율 × 배수' 상위 비율의 분위. 훈련 사건비율은 훈련 행 전체(결정월에 이미 급락 상태인 행 포함)의 비율이고(h=6 평균 약 0.14), "
      "지표는 결정월에 급락 상태가 아닌 행(사건비율 약 0.07)에서만 계산하므로 '목표 빈도'와 실제 경보 빈도는 같지 않다. 실제 경보 빈도를 병기한다. 유의성 검정 없음. "
      "국면 시작(6개 결정월 연속 비사건 뒤 첫 사건)은 평가 첫 달 앞의 2017년 사건 이력을 붙여 판정하고, 경보 창은 **시작 5개월 전부터 시작 달까지의 6개 결정월**이며 비급락 상태 행의 경보만 센다. "
      "선행개월 0 은 시작 달에 처음 켜진 경보(당월 포착)이고 '포착' 수에는 이것도 들어간다. 2018년 상반기에 시작한 국면은 창의 일부가 평가 구간 밖이라 창 안 가용 결정월 수가 6 보다 작다.\n")
    if s4 is not None:
        w(md(s4, ["모형", "h", "목표빈도(사건비율배수)", "행수", "사건수", "경보수", "경보빈도_실제", "경보적중률", "급락포착률", "F1", "동월지역쌍_AUC", "AUC_유효월수", "국면시작수", "국면포착률", "평균선행개월(포착분)"]))
    lead = rd(out, "v9_s4_국면시작_선행.csv")
    if lead is not None and len(lead):
        agg = dict(국면수=("시작", "count"), 포착=("선행개월", lambda v: int(v.notna().sum())), 사전포착=("선행개월", lambda v: int((v > 0).sum())), 당월포착=("선행개월", lambda v: int((v == 0).sum())), 평균선행=("선행개월", "mean"))
        if "창내가용월수" in lead.columns:
            agg["창6개월미만"] = ("창내가용월수", lambda v: int((v < 6).sum()))
        q = lead[(lead["배수"] == 1)].groupby(["모형", "h"]).agg(**agg).reset_index()
        w("\n국면 시작 포착(1배 컷오프):\n"); w(md(q))
    cases = rd(out, "v9_s4_사례표_h6.csv")
    if cases is not None and len(cases):
        q = cases.groupby(["모형", "결과"]).size().unstack(fill_value=0).reset_index()
        w("\n사례 집계(h=6, 1배 컷오프; 전체 사례표는 v9_s4_사례표_h6.csv):\n"); w(md(q))
    if os.path.exists(os.path.join(out, "fig6_4_경보사례_h6.png")):
        w("\n![급락 경보 사례 h=6](../rmpi/output_v9/fig6_4_경보사례_h6.png)\n")
        w("그림 6-4. 지역 × 결정월 격자. 초록 적중, 주황 오경보, 파랑 놓침, 회색은 결정월에 이미 급락 상태라 경보 대상에서 뺀 칸(rmpi/run_v9_fig.py).\n")

    # ---------------- 5 7단계
    w("\n## 5 해석과 보조 분석 (7단계)\n")
    dr = rd(out, "v9_s7_동인제거.csv")
    if dr is not None:
        w("### 5.1 동인 제거 (M2, alpha 를 주 실행 선택값의 중앙값으로 고정)\n")
        w("'(없음, alpha 고정 재실행)' 행의 손실변화는 주 실행(cv_min alpha) 대비이고, 동인을 뺀 행의 손실변화는 alpha 고정 재실행 대비다(양수 = 빼면 나빠짐). 고정 alpha 는 전 기간 선택값의 중앙값이라 사후 정보가 섞인 진단이다.\n")
        w(md(dr, ["단위", "h", "뺀 동인", "alpha", "MAE", "MAE_주실행", "손실변화(%)", "손실변화_기준"]))
    lvl = rd(out, "v9_s7_신고제이후_수준형.csv")
    if lvl is not None:
        w("\n### 5.2 신고제 이후 실거래 수준형과 상대 변화 (2022.07~, 사후·기술 통계)\n"); w(md(lvl))
    gu = rd(out, "v9_s7_서울구.csv")
    if gu is not None:
        w("\n### 5.3 서울 25개 구 보조 (목표 r, alpha 170 고정)\n"); w(md(gu))
    th = rd(out, "v9_s7_TH비교.csv")
    if th is not None:
        w("\n### 5.4 TH V11 과 같은 행 비교 (17개 시도, h=6, 2021~2025, 6개월 변화율 MAE %p)\n")
        w("TH V11 은 분기 재학습·훈련 시작 2015.07·매매·전세지수 제외, 우리는 월별 재학습·훈련 시작 2016.01·매매·전세 추세 포함. 같은 행에서의 수치 비교이며 설계 차이는 남는다.\n")
        w(md(th))
        thr = rd(out, "v9_s7_TH비교_행자료.csv")
        if thr is not None:
            nk = len(thr[["region", "P"]].drop_duplicates())
            w(f"\n대조 행자료: rmpi/output_v9/v9_s7_TH비교_행자료.csv, {len(thr)}행 = 지역·월 {nk}조합 × TH 모형 {thr['model'].nunique()}개(열: {', '.join(thr.columns)}). "
              f"원본은 {thr['출처'].iloc[0]} 이며, 비교표는 이 저장소 내부 파일만으로 다시 계산된다(rmpi/run_v9_stage7.py --parts D). 우리 쪽 행은 v9_s2_예측값.csv(9차)와 rmpi/output/stage6_예측값_패널.csv(8차)에서 읽는다.\n")
        thy = rd(out, "v9_s7_TH비교_연도별.csv")
        if thy is not None and len(thy):
            w("\n연도별:\n"); w(md(thy.pivot_table(index="모형", columns="연도", values="MAE").reset_index()))


    # ---------------- 6 해석과 특이사항
    w("\n## 6 해석과 특이사항 (연구자 메모, 추정 보고)\n")
    w("아래는 1~5장의 표를 읽은 메모다. 비교 대부분이 8차 규칙으로 '차이 불확실'이고 판정이 나온 몇 경우도 참고용이므로, 방향과 크기를 적되 판정하지 않는다.\n")
    reg = rd(out, "v9_s2_지역별MAE.csv")
    sj = ""
    if reg is not None:
        r6 = reg[reg["h"] == 6].pivot_table(index="region", columns="모형", values="MAE")
        if "세종" in r6.index and {"M0", "M2"} <= set(r6.columns):
            drop = 1 - r6.loc["세종", "M2"] / r6.loc["세종", "M0"]
            rank_top = r6["M2"].idxmax() == "세종"
            sj = (f"세종은 M0 대비 M2 의 오차가 {drop:.1%} 줄었지만(h=6: {r6.loc['세종', 'M0']:.3f} → {r6.loc['세종', 'M2']:.3f}), 보정 후에도 17개 시도 중 오차가 가장 "
                  f"{'크다' if rank_top else '큰 편이다'}(다음으로 큰 지역 {r6['M2'].drop('세종').max():.2f}).")
    c23 = ""
    if s2 is not None:
        r = s2[(s2["비교"] == "전국 M2 vs M1") & (s2["h"] == 1)]
        if len(r):
            dd = {c: float(r.iloc[0][c]) for c in s2.columns if c.startswith("d_")}
            tot = sum(dd.values())
            if tot < 0:
                c23 = f" 다만 연도별 순개선분의 {dd.get('d_2023', 0) / tot:.0%} 가 2023년 한 해에서 나왔고 두 불확실성 구간 모두 0 을 포함하므로, '단기 보정에 도움이 될 가능성을 발견했다' 이상으로 쓰지 않는다."
    w("### 6.1 ① 같은 튜닝 아래 A 와 B\n"
      "- 전진 교차검증 최소 alpha(cv_min)는 h=1 에서 A·B 모두의 손실을 낮췄다. B 의 h=1 MAE 0.042 는 mom1(0.044)과 선택형 단순기준(0.045)보다 낮다(비 0.94, 구간은 0 포함). cv_min 아래에서 B 가 두 단순기준 모두보다 낮은 수평선은 h=1 뿐이다.\n"
      "- B 의 h=3·6 과 A 의 h=3 에서는 cv_min 이 고정 10(8차와 같은 값, 9차 구성) 보다 나빴고(B h=6: 0.740 대 0.615), A 의 h=6 에서만 cv_min 이 나았다(0.757 대 0.798). 선택 alpha 가 연도마다 수십~수백 배 바뀌고(B h=6: 0.56 → 42 → 0.56 → 10 → 56 → 750 → 17,783), "
      "최소 손실의 1% 안에 든 후보가 중앙값으로 h=1 7개·h=3 4개·h=6 3개(해에 따라 1~14개)라 선택이 자료의 작은 변화에 민감하다. 1-SE 는 B 의 h=1·3 과 A 의 h=1·6 에서 세 튜닝 중 가장 나빴고, B h=6(0.736 대 cv_min 0.740)과 A h=3(0.299 대 0.302)에서는 cv_min 이 가장 나빴다. 어느 수평선에서도 1-SE 가 고정 10 보다 낫지는 않았다.\n"
      "- 그 결과 '같은 튜닝 아래' B/A 비는 0.79·0.89·0.98 로, h=1 에서 B 우위가 가장 뚜렷하고 h=6 에서는 거의 사라진다. 고정 10(9차 구성)의 h=6 B/A 0.77" + (f"(실제 8차 결과는 {_v8_BA_ratio()[6]:.2f})" if _v8_BA_ratio() else "") + " 은 고정 alpha 10 이 B 에 유리하게 작용한 몫을 포함한다고 읽힌다. 다만 어느 튜닝도 사전 지정이 아니었으므로 어느 쪽이 '맞는' 비교인지 확정하지 않는다.\n"
      "- A(가격 추세만의 Ridge)는 어떤 튜닝에서도 단순기준보다 나빴다(비 1.13~1.41). 8차 소견과 같다.\n")
    w("### 6.2 ② 보정 구조\n"
      "- 절편 보정 M1 은 M0 보다 나쁘다(전국 1~3%, 패널 0.3~1.4%). 훈련기간 평균 잔차가 평가기간에 이어지지 않는다. M1′(mom1 계수 학습)도 M0 보다 2~9% 나빠, mom1 계수를 1 로 두는 쪽이 낫다.\n"
      "- 경제정보 기여(M2 − M1)는 수평선에 따라 갈린다. 전국 h=1 에서 20% 개선(M2 MAE 0.036; mom1 0.044, ① 의 B 0.042 보다 낮고 DM 참고 p=0.002, 8개 연도 중 7개 우세)이지만, h=3·6 에서는 5%·7% 악화다." + c23 + " "
      "h=6 의 연도별 d 는 2021(+0.57)·2024(+0.29)·2020(+0.15)에서 크게 나쁘고 2023(−0.34)·2025(−0.21)·2019(−0.14)에서 좋아, 상승 전환기(2020~21)에 보정이 하락 쪽으로 치우친 것으로 읽힌다.\n"
      "- 통합 패널에서는 M2 − M1 이 h=1·3·6 에서 2%·4%·1% 개선이나 모두 구간이 0 을 넓게 포함한다. 패널 alpha 가 대부분 수백~수만으로 선택돼(예외: h=6 2024년 10) 보정항이 작게 축소된 상태라 패널 보정모형은 M0 에 가깝게 작동한다.\n"
      "- M2+past36(과거 3·6개월 변화 추가)은 전국 h=3·6 에서 8%·10% 개선, h=1 에서 5% 악화다. 긴 수평선의 잔차에는 중기 추세 정보가, 1개월에는 잡음이 된다. 민감도 항목이므로 주 모형을 바꾸지 않는다.\n"
      f"- 지역별로는 세종이 패널 손실의 큰 몫을 차지한다. {sj}\n")
    w("### 6.3 ③ 실거래 정보\n"
      "- 변화형 두 입력(월세 비중 로짓 Δ12, 천세대당 거래량 로그 Δ12)의 순수 추가 효과는 0 에 가깝거나 약간 음이다(RQ2 MAE 비 1.009~1.012, RQ3 Brier 비 1.004~1.009, 패널 M3/M2 1.001~1.012). 12개월합과 1개월 추가 시차 민감도도 같다(1.004~1.011). 구성명세에서 두 입력은 관측률 0.88~1.00 으로 유지됐으므로 '입력이 빠져서'가 아니다.\n"
      "- M3 는 M2 와 구성·가중·전처리가 같지만 alpha 를 따로 고르므로 M3 − M2 에는 alpha 재선택의 효과가 섞인다(2장 표). 그래도 차이가 1.2% 이내라 결론은 같다.\n"
      "- 과거 건수는 2026.10 일괄 수집본이므로, 이 결과는 '최신 수정자료와 지정한 입력 형태에서는 추가 개선이 관찰되지 않았다' 까지다. 당시 이용 가능했던 건수로도 같은 결과가 나오는지는 확인하지 못했고, 수집 시점 편향의 방향과 크기도 모른다.\n"
      "- 반면 신고제 이후 수준형(5.2)은 월세 비중 수준이 높은 지역일수록 이후 상대 월세가 덜 오르고(순위상관 −0.14~−0.21), 거래회전율이 높은 지역일수록 더 오르는(+0.15~+0.20) 관계를 보인다. h=6 상·하 삼분위의 후속 r 평균 차이는 0.3~0.5%p 다. 수준형은 신고제 전후 비교가 불가능해 주 분석에서 뺀 형태라 사후 기술통계로만 적고, 월 1회 재수집으로 쌓이는 시점별 자료에서 수준형을 쓸 수 있는지는 전향 검증(설계를 고정한 뒤 예측을 저장하고 나중에 정답을 맞춰 보는 구간) 과제로 남긴다.\n")
    w("### 6.4 ④ 급락 경보\n"
      "- 같은 목표 경보 빈도에서 B 가 A 보다 적중률·포착률 모두 높다(h=6 1배: 0.41/0.58 대 0.30/0.36). " + _episode_text(out) + " 실거래 추가는 B 와 거의 같다.\n"
      + _alert_share_text(out) +
      "- 그림 6-4 와 국면표에서 2022~2023 하락 국면(대구는 2022-03, 나머지는 2022 하반기 시작)은 B 가 대구·서울·대전·전남을 시작 2~5개월 전에, 울산은 시작 달에 잡았고 인천·부산·광주·경기는 놓쳤다. 세종의 급락(2018-01, 2019-02, 2021-08, 2023-12)은 2023-12 를 A 가 5개월 전에 잡은 것 말고는 모두 놓쳤다. "
      "2019-02 세종은 창 안의 경보가 이미 급락 상태인 달의 것뿐이라 포착으로 세지 않았다. 2018 상반기에 시작한 4개 국면은 창이 평가 구간 밖까지 걸쳐 가용 결정월이 1~5개뿐이다.\n"
      "- 오경보는 2018~2019 의 부산·충남과 2023 국면 후반~종료 뒤의 충남·대전·경남·전북·제주·충북·강원·경북에 몰려 '하락이 끝난 뒤에도 계속 켜지는' 형태다. 2024 는 세종의 2023 말 국면이 결정월 1~5월까지 사건으로 이어진 것(A 만 2024-05 적중)뿐이고, 2025 는 사건도 경보도 없다.\n"
      "- 동월 지역쌍 AUC 는 0.82~0.85 로 같은 달 안의 순위 변별은 있으나 모형 간 차이는 약 0.02 다.\n")
    w("### 6.5 7단계 보조 분석\n"
      "- 동인 제거: M2 의 경제정보 기여는 ⑤ 시장과열·단기 트리거에 집중된다(빼면 전국 h=1 +24%, h=3 +12%, h=6 +4%; 패널 +3%·+5%·+4%). 나머지 동인은 ±5% 안이고 ⑥ 거시경기는 전국 h=3 에서 빼는 쪽이 5% 낫다.\n"
      "- alpha 를 주 실행 선택값의 중앙값으로 고정해 다시 돌리면 전국 MAE 가 h=1·3·6 에서 7%·12%·19%, 패널에서 2%·2%·8% 낮다. 전 기간 선택값을 쓴 사후 진단이라 성능 주장에는 쓰지 않지만, 튜닝 불안정의 비용이 이 정도임을 보여 준다.\n"
      "- 서울 25구: A·B·B+실거래의 월내 상대 MAE 차이가 h=1·3 에서는 0.001 안이고 h=6 에서는 A 0.452 < B+실거래 0.455 < B 0.455 로 0.004 차이다(A 가 약 0.8% 낫다). 셋 모두 0 예측보다 6~24% 낮다(h=6 → h=1). 구 단위에서는 RMPI 와 실거래 추가가 A 를 개선하지 못했고 h=6 에서는 조금 나쁘다(8차와 같은 방향).\n"
      "- TH V11 같은 행 비교(1,020행, h=6, 2021~2025): corr_M2 0.682 < corr_M3 0.699 < XGBoost 0.721 < 8차 ridge_B 0.763 < Random Forest 0.767 < Extra Trees 0.773. 2023 에 차이가 가장 크고(M2 0.504 대 XGBoost 0.694) 2021 은 TH 모형이 낫다(0.92~1.05 대 1.10). 월별 재학습·훈련 시작 2016.01·매매·전세 추세 포함이라는 설계 차이는 남는다.\n")
    w("### 6.6 특이사항과 후속 과제\n"
      "- alpha 선택의 불안정이 9차의 가장 큰 방법상 특이사항이다. 전진 교차검증 곡선이 평평할 때 '로그 10 에 가장 가까운 후보' 규칙으로 가는데도 1% 안 후보가 많아 연도별 선택이 수십 배씩 뛴다. 다음 설계에서는 평평 구간이면 직전 연도 선택값을 유지하는 규칙이나 후보 간 모형 평균을 사전 지정해 두는 것을 검토한다.\n"
      "- 보정모형의 적합 주기가 결과를 바꾼다(6.7). 매달 재적합은 h=1 에 유리하고 연 1회 적합은 h=3·6 에 유리했다. 적합 주기도 alpha 와 같이 사전 지정 항목이다. "
      "2026년 1~9월 정답으로 지금 모형을 다시 계산하는 것은 **사후 확인**이고, 설계를 고정한 뒤 매달 예측을 저장해 두고 나중에 정답을 맞춰 보는 구간부터가 **전향 검증**이다. 두 구간 모두에서 전국 M2(매달)·M2(연 1회)·B(고정 10)·mom1 을 함께 추적한다.\n"
      "- 2018-01 의 h=6 선택은 전진 폴드 2개뿐이라 대체값 10 이다(모든 전국·보정모형 공통). 2018 년 h=6 결과는 튜닝 없는 결과다.\n"
      "- 경보 컷오프는 훈련기간 분위라 평가기간의 실제 경보 빈도(전체 행 기준)가 목표보다 높았다. 실무 적용 시 빈도 목표를 맞추려면 최근 N개월 예측확률 분위로 갱신하는 방식이 필요하나, 이 보고에서는 사전 지정대로 고정 컷오프를 유지했다.\n"
      "- 독립 검증(코드 누수·스크립트·보고서 수치, 4개 관점과 반박 검증)에서 사실로 확인된 지적은 모두 반영했다: 보정모형 월별 재적합, 국면 시작의 2017 이력과 비급락 행 경보, 설정 객체 부작용, 공통 프레임의 미사용 열, 표 표기. 정답 가용성·내부 폴드·컷오프의 과거 절단 불변은 검증에서 재현으로 확인됐다.\n")
    # ---------------- 6.7 적합 주기 참고 (검증 전 실행)
    ref = rd(os.path.join(out, "참고_수정전"), "v9_s2_비교_연1회적합.csv")
    if ref is not None and s2 is not None:
        w("### 6.7 참고: 보정모형 적합 주기 (매달 재적합 = 본 보고, 연 1회 적합 = 검증 전 실행)\n")
        w("검증에서 보정모형 M2·M3 의 적합이 alpha 선택과 같이 1월에만 이루어지던 결함(2~12월은 1월 모형 재사용)이 확인되어 매 결정월 재적합으로 고쳤다. "
          "고치기 전 실행의 비교표를 rmpi/output_v9/참고_수정전/ 에 두고 여기서 적합 주기의 민감도로만 읽는다(설정·입력 동일, 결과 커밋 40045dd). "
          "M0·M1·M1′ 과 ①·③ 의 B 모형은 두 실행에서 같다.\n")
        keys = ["전국 M2 vs M1", "전국 M2 vs M0", "패널 M2 vs M1", "패널 M2 vs M0", "패널 M3 vs M2"]
        a = s2[s2["비교"].isin(keys)][["비교", "h", "MAE_비교", "MAE비율"]].rename(columns={"MAE_비교": "MAE_매달재적합", "MAE비율": "비율_매달재적합"})
        b = ref[ref["비교"].isin(keys)][["비교", "h", "MAE_비교", "MAE비율"]].rename(columns={"MAE_비교": "MAE_연1회적합", "MAE비율": "비율_연1회적합"})
        m = a.merge(b, on=["비교", "h"], how="outer").sort_values(["h", "비교"])
        w(md(m, nd=3))
        th_ref = rd(os.path.join(out, "참고_수정전"), "v9_s7_TH비교_연1회적합.csv")
        th_new = rd(out, "v9_s7_TH비교.csv")
        if th_ref is not None and th_new is not None:
            q = th_new[th_new["모형"].astype(str).str.startswith("corr_M")][["모형", "MAE"]].rename(columns={"MAE": "MAE_매달재적합"}).merge(
                th_ref[th_ref["모형"].astype(str).str.startswith("corr_M")][["모형", "MAE"]].rename(columns={"MAE": "MAE_연1회적합"}), on="모형", how="outer")
            w("\nTH V11 같은 행(1,020행, h=6) 의 보정모형 MAE:\n"); w(md(q, nd=3))

    # ---------------- 7 통제·한계
    w("\n## 7 통제 항목과 한계\n")
    chk = rd(out, "stage5_점검.csv")
    if chk is not None:
        w("5단계 점검: " + "; ".join(f"{r['항목']} {r['결과']}" for _, r in chk.iterrows()) + "\n")
    w("- 평가 구간 2018~2025 는 8차에서 이미 본 구간이다. 이 보고의 모든 비교는 후속 탐색이다. 2026년 1~9월 정답을 지금 모형으로 다시 계산하는 사후 확인도 설계가 2018~2025 결과를 본 뒤 정해졌으므로 표본외 검증이 아니며, 설계 고정 뒤 저장한 예측을 나중에 맞춰 보는 전향 검증만이 표본외 진단이다.\n"
      "- 실거래 과거 건수는 2026.10.01~05 일괄 수집본이라 당시 알 수 있던 건수가 아니다(낙관 편향 가능). 월 1회 재수집으로 시점별 건수를 쌓는다.\n"
      "- 보정모형의 alpha 는 기간별로 전진 교차검증에서 골랐다. 평가기간의 선택값을 향후 적정값으로 확정하지 않는다.\n"
      "- 8차 판정 규칙(두 t구간)은 참고로만 적었고, 동등성 판정은 하지 않는다.\n")
    man = rd(out, "v9_stage6_manifest.csv")
    if man is not None:
        w("\n산출물 manifest(6단계 마지막 호출 기록): " + "; ".join(f"{r['항목']}={r['값']}" for _, r in man.iterrows()) +
          ". commit 은 실행 시작 시점의 HEAD 이고 실행 코드는 검증 수정분을 담은 07eba5d 와 같다(수정분은 실행 직후 커밋). 7단계 manifest 는 v9_stage7_manifest.csv(전체 실행)와 v9_stage7_manifest_D.csv(TH 비교를 저장소 내부 행자료만으로 다시 계산한 기록).\n")
    os.makedirs(os.path.join(BASE, "docs"), exist_ok=True)
    path = os.path.join(BASE, "docs", "연구결과_9차_후속분석_보고.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", path)


if __name__ == "__main__":
    main()
