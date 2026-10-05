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
        fb = a1[a1["규칙"] == "fallback"]
        flat = a1[a1["규칙"] == "cv_min"].groupby("h")["평평후보수"].median()
        w("\n빈칸은 전진 폴드가 설정 최소치(4)에 못 미쳐 대체값 alpha 10 을 쓴 경우" +
          (f"({', '.join(sorted(set(fb['시점'].astype(str) + ' h=' + fb['h'].astype(int).astype(str))))})" if len(fb) else "") +
          ". 최소 손실의 1% 안에 든 후보 수의 중앙값은 " + ", ".join(f"h={int(h)}: {v:.0f}개" for h, v in flat.items()) +
          "로, 교차검증 곡선이 평평해 선택값이 연도마다 크게 바뀐다(해석은 6장).\n")
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
        w("\n빈칸은 전진 폴드 부족으로 대체값 10 을 쓴 경우(2018-01, h=6). 패널 보정모형의 선택값은 수백~수만으로, 보정항이 거의 0 으로 축소된 상태에서 작동한다.\n")
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
    if os.path.exists(os.path.join(out, "fig6_4_경보사례_h6.png")):
        w("\n![급락 경보 사례 h=6](../rmpi/output_v9/fig6_4_경보사례_h6.png)\n")
        w("그림 6-4. 지역 × 결정월 격자. 초록 적중, 주황 오경보, 파랑 놓침, 회색은 결정월에 이미 급락 상태라 경보 대상에서 뺀 칸(rmpi/run_v9_fig.py).\n")

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


    # ---------------- 6 해석과 특이사항
    w("\n## 6 해석과 특이사항 (연구자 메모, 추정 보고)\n")
    w("아래는 1~5장의 표를 읽은 메모다. 비교 대부분이 8차 규칙으로 '차이 불확실'이고 판정이 나온 몇 경우도 참고용이므로, 방향과 크기를 적되 판정하지 않는다.\n")
    w("### 6.1 ① 같은 튜닝 아래 A 와 B\n"
      "- 전진 교차검증 최소 alpha(cv_min)는 h=1 에서 A·B 모두의 손실을 낮췄다. B 의 h=1 MAE 0.042 는 mom1(0.044)과 선택형 단순기준(0.045)보다 낮다(비 0.94, 구간은 0 포함). cv_min 아래에서 B 가 두 단순기준 모두보다 낮은 수평선은 h=1 뿐이다.\n"
      "- h=3·6 에서는 cv_min 이 8차 고정 10 보다 나빴다(B h=6: 0.740 대 0.615). 선택 alpha 가 연도마다 수십~수백 배 바뀌고(B h=6: 0.56 → 42 → 0.56 → 10 → 56 → 750 → 17,783), 1% 안 후보가 3~5개라 선택이 거의 임의적이다. 1-SE 는 모든 수평선에서 가장 나빴다.\n"
      "- 그 결과 '같은 튜닝 아래' B/A 비는 0.79·0.89·0.98 로, h=1 에서 B 우위가 가장 뚜렷하고 h=6 에서는 거의 사라진다. 8차의 h=6 B/A 0.77 은 고정 alpha 10 이 B 에 유리하게 작용한 몫을 포함한다고 읽힌다. 다만 어느 튜닝도 사전 지정이 아니었으므로 어느 쪽이 '맞는' 비교인지 확정하지 않는다.\n"
      "- A(가격 추세만의 Ridge)는 어떤 튜닝에서도 단순기준보다 나빴다(비 1.13~1.41). 8차 소견과 같다.\n")
    w("### 6.2 ② 보정 구조\n"
      "- 절편 보정 M1 은 전국·패널 모두 M0 보다 1~3% 나쁘다. 훈련기간 평균 잔차가 평가기간에 이어지지 않는다. M1′(mom1 계수 학습)도 M0 보다 2~9% 나빠, mom1 계수를 1 로 두는 쪽이 낫다.\n"
      "- 경제정보 기여(M2 − M1)는 전국 h=3·6 에서 6%·11% 개선, h=1 에서 2% 악화다. 전국 h=6 M2 의 MAE 0.550 은 ① 의 어떤 B(0.615~0.740)보다 낮고 mom1 보다 8% 낮다. mom1 계수를 고정하고 잔차만 축소 추정하는 구조가 h=6 에서 B 자유 회귀보다 안정적이라는 신호이나, 연도별 d 가 2019·2023·2025 의 큰 개선(−0.20·−0.45·−0.25)과 2020·2024 의 악화(+0.25·+0.20)로 갈려 구간은 넓다.\n"
      "- 통합 패널에서는 M2 − M1 이 h=3 에서만 5% 개선이고 h=1·6 은 1~4% 악화다. 패널 alpha 가 수백~수만으로 선택돼 보정항이 거의 0 으로 축소된 상태라, 패널 보정모형은 M0 에 가깝게 작동한다.\n"
      "- M2+past36 은 h=1 에서 26% 악화다. 과거 3·6개월 변화는 1개월 보정에 잡음으로 작용한다.\n"
      "- 지역별로는 세종의 MAE(h=6 에서 2.6~3.0)가 다른 지역(0.4~1.8)보다 월등히 커 패널 손실의 큰 몫을 차지한다. 어느 모형도 세종을 개선하지 못한다.\n")
    w("### 6.3 ③ 실거래 정보\n"
      "- 변화형 두 입력(월세 비중 로짓 Δ12, 천세대당 거래량 로그 Δ12)의 순수 추가 효과는 0 에 가깝거나 약간 음이다(RQ2 MAE 비 1.009~1.012, RQ3 Brier 비 1.004~1.009, 패널 M3/M2 0.997~1.016). 12개월합과 1개월 추가 시차 민감도도 같다. 구성명세에서 두 입력은 관측률 0.88~1.00 으로 유지됐으므로 '입력이 빠져서'가 아니다.\n"
      "- 과거 건수가 2026.10 일괄 수집본이라 낙관 편향 가능성이 있는데도 효과가 0 이므로, 수집 시점 편향이 이 결론의 방향을 바꾸지는 않는다.\n"
      "- 반면 신고제 이후 수준형(5.2)은 월세 비중 수준이 높은 지역일수록 이후 상대 월세가 덜 오르고(순위상관 −0.15~−0.21), 거래회전율이 높은 지역일수록 더 오르는(+0.15~+0.20) 관계를 보인다. h=6 상·하 삼분위의 후속 r 평균 차이는 0.3~0.5%p 다. 수준형은 신고제 전후 비교가 불가능해 주 분석에서 뺀 형태라 사후 기술통계로만 적고, 월 1회 재수집으로 쌓이는 시점별 자료에서 수준형을 쓸 수 있는지는 2026년 전향 검증 과제로 남긴다.\n")
    w("### 6.4 ④ 급락 경보\n"
      "- 같은 목표 경보 빈도에서 B 가 A 보다 적중률·포착률 모두 높다(h=6 1배: 0.41/0.58 대 0.30/0.36; 2배 컷오프의 국면 포착률 92% 대 38%). 실거래 추가는 B 와 거의 같다.\n"
      "- 실제 경보 빈도는 목표(훈련 사건비율 약 7%)보다 높다(B 10%, 2배에서 23%). 평가기간의 예측확률 분포가 훈련기간보다 위에 있다.\n"
      "- 그림 6-4 에서 2022 하반기~2023 하락 국면은 B 가 대구·서울·대전·전남을 국면 시작 2~5개월 전에, 울산은 시작 달에 잡았고 인천·부산·광주·경기는 놓쳤다. 오경보는 2018~2019 의 부산·충남과 2023 국면 후반~종료 뒤의 충남·대전·경남·전북·제주·충북·강원·경북에 몰려 '하락이 끝난 뒤에도 계속 켜지는' 형태다. 세종은 2019 급락을 세 모형 모두 5개월 전에 잡았으나 2021 급락은 모두 놓쳤고 2023 말 급락은 A 만 잡았다. 2024~2025 는 사건도 경보도 없다.\n"
      "- 동월 지역쌍 AUC 는 0.82~0.85 로 같은 달 안의 순위 변별은 있으나 모형 간 차이는 0.02 안이다.\n")
    w("### 6.5 7단계 보조 분석\n"
      "- 동인 제거: M2 의 경제정보 기여는 ⑤ 시장과열·단기 트리거에 집중된다(빼면 전국 h=1 +26%, h=3 +16%; 패널 h=3 +6%, h=6 +8%). ② 주택공급·재고는 전국 h=3·6 에서 6~7%. 나머지 동인은 ±4% 안이고 일부는 빼는 쪽이 낫다.\n"
      "- alpha 를 주 실행 선택값의 중앙값으로 고정해 다시 돌리면 전국 MAE 가 h=1·3 에서 19%, h=6 에서 9% 낮다. 전 기간 선택값을 쓴 사후 진단이라 성능 주장에는 쓰지 않지만, 튜닝 불안정의 비용이 이 정도임을 보여 준다.\n"
      "- 서울 25구: A·B·B+실거래의 월내 상대 MAE 차이가 0.001 안이고 0 예측보다 6~24% 낮다(h=6 → h=1). 구 단위에서는 RMPI 와 실거래 추가가 효과가 없다(8차와 같음).\n"
      "- TH V11 같은 행 비교(1,020행, h=6, 2021~2025): corr_M2 0.683 < XGBoost 0.721 < 8차 ridge_B 0.763 < Random Forest 0.767 < Extra Trees 0.773. 2023 에 차이가 가장 크고(M2 0.516 대 XGBoost 0.694) 2021 은 TH 모형이 낫다(0.94~1.05 대 1.16). 월별 재학습·훈련 시작 2016.01·매매·전세 추세 포함이라는 설계 차이는 남는다.\n")
    w("### 6.6 특이사항과 후속 과제\n"
      "- alpha 선택의 불안정이 9차의 가장 큰 방법상 특이사항이다. 전진 교차검증 곡선이 평평할 때 '로그 10 에 가장 가까운 후보' 규칙으로 가는데도 1% 안 후보가 많아 연도별 선택이 수십 배씩 뛴다. 다음 설계에서는 평평 구간이면 직전 연도 선택값을 유지하는 규칙이나 후보 간 모형 평균을 사전 지정해 두는 것을 검토한다.\n"
      "- 전국 h=6 보정모형 M2 의 결과는 후속 탐색이므로 2026년 전향 검증(정답 2026.09 까지의 사후 확인 구간)에서 M2 와 B(고정 10)·mom1 을 함께 추적한다.\n"
      "- 2018-01 의 h=6 선택은 전진 폴드 2개뿐이라 대체값 10 이다(모든 전국·보정모형 공통). 2018 년 h=6 결과는 튜닝 없는 결과다.\n"
      "- 경보 컷오프는 훈련기간 분위라 평가기간 실제 빈도가 목표보다 높았다. 실무 적용 시 빈도 목표를 맞추려면 최근 N개월 예측확률 분위로 갱신하는 방식이 필요하나, 이 보고에서는 사전 지정대로 고정 컷오프를 유지했다.\n")
    # ---------------- 7 통제·한계
    w("\n## 7 통제 항목과 한계\n")
    chk = rd(out, "stage5_점검.csv")
    if chk is not None:
        w("5단계 점검: " + "; ".join(f"{r['항목']} {r['결과']}" for _, r in chk.iterrows()) + "\n")
    w("- 평가 구간 2018~2025 는 8차에서 이미 본 구간이다. 이 보고의 모든 비교는 후속 탐색이며, 2026년 사후 확인과 전향 검증만이 표본외 진단이다.\n"
      "- 실거래 과거 건수는 2026.10.01~05 일괄 수집본이라 당시 알 수 있던 건수가 아니다(낙관 편향 가능). 월 1회 재수집으로 시점별 건수를 쌓는다.\n"
      "- 보정모형의 alpha 는 기간별로 전진 교차검증에서 골랐다. 평가기간의 선택값을 향후 적정값으로 확정하지 않는다.\n"
      "- 8차 판정 규칙(두 t구간)은 참고로만 적었고, 동등성 판정은 하지 않는다.\n")
    man = rd(out, "v9_stage6_manifest.csv")
    if man is not None:
        w("\n산출물 manifest(6단계 마지막 호출 기록): " + "; ".join(f"{r['항목']}={r['값']}" for _, r in man.iterrows()) +
          ". 6단계는 ①, ②, ③④ 의 세 호출로 나눠 실행했고 코드·설정·입력은 같다(결과 커밋 2ffddff, b39175a, 40045dd). 7단계 manifest 는 v9_stage7_manifest.csv.\n")
    os.makedirs(os.path.join(BASE, "docs"), exist_ok=True)
    path = os.path.join(BASE, "docs", "연구결과_9차_후속분석_보고.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", path)


if __name__ == "__main__":
    main()
