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
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


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
        for tag, nm in (("cvmin", "전진 교차검증 최소 alpha"), ("fixed10", "8차 고정 alpha 10")):
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
    w("전국 Ridge 의 alpha 를 매년 1월 훈련자료 안 전진 교차검증(검증 12개월) 평균 손실 최소로 골랐다. 고정 10(8차)과 1-SE(8차 민감도)는 비교용이다. 선택형 단순기준은 8차 정의 그대로다.\n")
    if s1 is not None:
        w(md(s1, ["비교", "h", "MAE_기준", "MAE_비교", "MAE비율", "평균개선율(%)", "연평균_하한", "연평균_상한", "2년_하한", "2년_상한", "우세연도", "8차규칙판정(참고)"]))
    if a1 is not None:
        piv = a1[a1["규칙"] == "cv_min"].pivot_table(index=["정보군", "h"], columns="시점", values="선택", aggfunc="first")
        w("\n선택된 alpha(전진 교차검증 최소, 1월 재선택):\n")
        w(md(piv.reset_index(), nd=2))
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
    dm2 = rd(out, "v9_s2_DM참고.csv")
    if dm2 is not None:
        w("\nDiebold-Mariano 참고:\n"); w(md(dm2))

    # ---------------- 3 ③
    w("\n## 3 ③ 실거래 정보의 추가 효과\n")
    w("지역 블록 처리의 별도 입력 두 개(월세 비중 로짓 3개월합 Δ12, 천세대당 전월세 거래량 로그 3개월합 Δ12). 신고제 경계 2021.07~2022.08 결측. 최신 수정자료 보조 분석.\n")
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
    w("컷오프는 각 모형의 훈련기간 예측확률 분포에서 '훈련 사건비율 × 배수' 상위 비율의 분위. 실제 경보 빈도를 병기한다. 유의성 검정 없음.\n")
    if s4 is not None:
        w(md(s4, ["모형", "h", "목표빈도(사건비율배수)", "행수", "사건수", "경보수", "경보빈도_실제", "경보적중률", "급락포착률", "F1", "동월지역쌍_AUC", "AUC_유효월수", "국면시작수", "국면포착률", "평균선행개월(포착분)"]))
    lead = rd(out, "v9_s4_국면시작_선행.csv")
    if lead is not None and len(lead):
        q = lead[(lead["배수"] == 1)].groupby(["모형", "h"]).agg(국면수=("시작", "count"), 포착=("선행개월", lambda v: int(v.notna().sum())), 평균선행=("선행개월", "mean")).reset_index()
        w("\n국면 시작 포착(1배 컷오프):\n"); w(md(q))
    cases = rd(out, "v9_s4_사례표_h6.csv")
    if cases is not None and len(cases):
        q = cases.groupby(["모형", "결과"]).size().unstack(fill_value=0).reset_index()
        w("\n사례 집계(h=6, 1배 컷오프; 전체 사례표는 v9_s4_사례표_h6.csv):\n"); w(md(q))

    # ---------------- 5 7단계
    w("\n## 5 해석과 보조 분석 (7단계)\n")
    dr = rd(out, "v9_s7_동인제거.csv")
    if dr is not None:
        w("### 5.1 동인 제거 (M2, alpha 를 주 실행 선택값의 중앙값으로 고정; 손실변화 양수 = 빼면 나빠짐)\n"); w(md(dr, ["단위", "h", "뺀 동인", "alpha", "MAE", "손실변화(%)"]))
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
        thy = rd(out, "v9_s7_TH비교_연도별.csv")
        if thy is not None and len(thy):
            w("\n연도별:\n"); w(md(thy.pivot_table(index="모형", columns="연도", values="MAE").reset_index()))

    # ---------------- 6 통제·한계
    w("\n## 6 통제 항목과 한계\n")
    chk = rd(out, "stage5_점검.csv")
    if chk is not None:
        w("5단계 점검: " + "; ".join(f"{r['항목']} {r['결과']}" for _, r in chk.iterrows()) + "\n")
    w("- 평가 구간 2018~2025 는 8차에서 이미 본 구간이다. 이 보고의 모든 비교는 후속 탐색이며, 2026년 사후 확인과 전향 검증만이 표본외 진단이다.\n"
      "- 실거래 과거 건수는 2026.10.01~05 일괄 수집본이라 당시 알 수 있던 건수가 아니다(낙관 편향 가능). 월 1회 재수집으로 시점별 건수를 쌓는다.\n"
      "- 보정모형의 alpha 는 기간별로 전진 교차검증에서 골랐다. 평가기간의 선택값을 향후 적정값으로 확정하지 않는다.\n"
      "- 8차 판정 규칙(두 t구간)은 참고로만 적었고, 동등성 판정은 하지 않는다.\n")
    man = rd(out, "v9_stage6_manifest.csv")
    if man is not None:
        w("\n산출물 manifest: " + "; ".join(f"{r['항목']}={r['값']}" for _, r in man.iterrows()) + "\n")
    os.makedirs(os.path.join(BASE, "docs"), exist_ok=True)
    path = os.path.join(BASE, "docs", "연구결과_9차_후속분석_보고.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", path)


if __name__ == "__main__":
    main()
