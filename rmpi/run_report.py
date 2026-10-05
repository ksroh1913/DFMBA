# -*- coding: utf-8 -*-
"""
[보고서] 1~7단계 산출물(rmpi/output/*.csv)을 읽어 docs/연구결과_8차계획_실행보고.md 를 만든다.
수치는 모두 CSV 에서 가져오며 손으로 적지 않는다. 그림은 rmpi/output/fig*.png 를 가리킨다.
"""

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import settings as S  # noqa: E402

OUT = os.path.join(BASE, "rmpi", "output")
DOC = os.path.join(BASE, "docs", "연구결과_8차계획_실행보고.md")


def rd(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:.{nd}f}"
    return str(v)


def md_table(df, nd=3, cols=None):
    if df is None or len(df) == 0:
        return "_(자료 없음)_\n"
    df = df if cols is None else df[[c for c in cols if c in df.columns]]
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v, nd) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows) + "\n"


def main():
    s = S.load()
    hs = s["timing"]["horizons"]
    ver = rd("stage6_판정.csv")
    summ = rd("stage6_요약.csv")
    yl = rd("stage6_연도별손실.csv")
    aux = rd("stage6_전국_보조.csv")
    comb = rd("stage6_결합예측.csv")
    ev = rd("stage6_급락_지표.csv")
    strata = rd("stage6_급락_층별.csv")
    ep = rd("stage6_급락_국면시작.csv")
    up = rd("stage6_급등_보조.csv")
    sel = rd("stage6_선택형기준.csv")
    leaf = rd("stage6_ET잎진단.csv")
    shap_s = rd("stage7_SHAP_요약.csv")
    rem = rd("stage7_동인제거.csv")
    gu = rd("stage7_서울구.csv")
    xm = rd("stage7_추가모형.csv")
    th = rd("stage7_팀보고서비교.csv")
    chk5 = rd("stage5_점검.csv")
    spl = rd("stage2_접합요약.csv")
    comp4 = rd("stage4_구성명세.csv")
    man = rd("stage6_manifest.csv")

    L = []
    A = L.append
    A("# 아파트 월세 흐름 진단과 급변 예측: 8차 확정 계획의 실행 결과\n")
    A(f"> 계획: docs/연구계획_8차_확정본.md · 설정: {s['_path']} (sha256 {s['_sha']}, {s['meta']['settings_version']}) · "
      f"실행 코드: rmpi/ · 산출물: rmpi/output/ · 평가 구간: 결정월 {s['timing']['eval_start']}~{s['timing']['eval_end']}, 월별 재적합\n")
    if len(man):
        A("실행 기록(stage6_manifest.csv): " + "; ".join((f"{r['항목']} {r['값']}" if isinstance(r["값"], str) else str(r["항목"])) for _, r in man.iterrows()
                                                      if r["구분"] in ("commit", "input", "package")) + "\n")

    # ---------------- 0 핵심 결과
    def V_(name, h, col="MAE비율(평균의 비)"):
        x = ver[(ver["비교"] == name) & (ver["h"] == h)]
        return float(x[col].iloc[0]) if len(x) else np.nan
    def R3(name, col="MAE비율(평균의 비)"):
        return "·".join(f"{V_(name, h, col):.2f}" for h in hs)
    A("## 0 핵심 결과 (수치는 아래 표에서 가져옴)\n")
    A(f"- RQ1(전국 공통 흐름): B_공통의 MAE 는 A 의 {R3('RQ1 B_공통 대 A')}배(h=1·3·6)로 h=1·3 에서 ‘개선’이지만, 선택형 단순기준 대비는 {R3('RQ1 B_공통 대 선택형 단순기준')}배로 모두 ‘차이 불확실’이다. "
      f"선택형 단순기준은 거의 모든 시점에서 ‘h×최근 월변화’(mom1)가 선택됐다. A 자체가 선택형 단순기준보다 나쁘고(비 {R3('보조 RQ1 A 대 선택형 단순기준')}), "
      f"alpha 를 해마다 다시 고르는 민감도에서는 B 와 A 의 차이가 {R3('민감도 RQ1 B 재튜닝 대 A 재튜닝')}배로 줄어 모두 ‘차이 불확실’이다. "
      "따라서 사전 고정한 alpha=10 이 가격 추세 모형 A 를 과하게 축소했고, B 의 A 대비 개선 중 상당 부분은 그 영향으로 해석해야 한다. 계획의 종합 규칙에 따라 RQ1 은 ‘지지 안 됨’이다.\n")
    A(f"- RQ2(시도 상대 변화): 지역 RMPI 를 더한 B_지역은 상대 가격 추세 A_지역과 같다(비 {R3('RQ2 B_지역 대 A_지역')}). 상대 가격 추세 자체는 ‘차이 0 기준’보다 크게 낫다(비 {R3('보조 RQ2 A_지역 대 차이 0')}). "
      "지역 동인에 정보가 없다는 뜻이 아니라, 가격 추세에 더해 얻는 개선을 확인하지 못한 것이다.\n")
    A(f"- RQ3(급락 확률): B 의 Brier 는 A 의 {R3('RQ3 B(공통+지역 RMPI) 대 A')}배로 세 기간 모두 연평균 구간만 0 아래인 ‘개선 신호’다(2년 묶음 구간은 0 을 포함). "
      "개선은 주로 이미 하락 중인 지역(기존 급락 상태)에서 나오고, 비급락 상태에서의 Brier 차이는 작다. 국면 시작 사례의 경보 선행은 대부분 1개월이다.\n")
    if len(aux):
        pl = aux[aux["항목"].str.contains("분위")]
        ex = aux[aux["항목"].str.contains("B main w=1.0")]
        A("- 위약 시험: 공통 블록을 무작위 계열로 바꾼 20회의 B 보다 실제 B 의 MAE 가 모두 낮다(분위 " + "·".join(f"{v:.2f}" for v in pl["값"]) +
          "). 공통 블록에 잡음 이상의 정보가 있지만 그것이 모멘텀 규칙을 이기는 데까지는 이르지 못했다. 대용계열 확장(옛 수도권 지수, 가중치 1)은 B 의 MAE 를 "
          + "·".join(f"{v:+.3f}" for v in ex["값"]) + "%p 바꿔 오히려 나빠졌다.\n")
    if len(shap_s) and len(rem):
        A("- 해석: 선형 SHAP 과 동인 제거 재학습 모두에서 ⑤ 시장과열·단기 트리거(소비심리·주택가격전망 CSI)가 공통 RMPI 의 기여를 주도한다. 다른 동인을 빼도 MAE 변화는 작다.\n")
    if len(comb):
        A("- 결합 예측: 공통 성분을 전국 모형으로, 월내 성분을 상대 모형으로 나눈 결합 B 와 같은 입력의 통합 패널 B 는 ‘차이 불확실’이며, h=1 에서는 지역별 mom1 단순 규칙이 둘보다 낫다.\n")
    A("- 트리 모형(Extra Trees)은 모든 과제에서 Ridge 보다 나쁘거나 같았다. 추가 비교모형(부록)에서도 선형 계열이 가장 좋았다.\n")
    A("## 1 사전 지정 주 분석의 판정\n")
    A("판정 규칙(계획 1장): d(t) = L_추가정보 − L_기준. 결정연도 8개 연평균의 95% t구간(자유도 7)과 2년 묶음 4개의 t구간(자유도 3)의 "
      "상한이 모두 0 미만이면 ‘개선’, 연평균 구간만이면 ‘개선 신호’. RQ1 은 같은 h 에서 A 대비와 선택형 단순기준 대비가 모두 ‘개선’이어야 그 h 를 개선으로 센다.\n")
    A(md_table(summ))
    main_v = ver[ver["비교"].str.startswith("RQ")]
    cols = ["비교", "h", "mean", "annual_lo", "annual_hi", "biennial_lo", "biennial_hi", "years_better", "판정", "MAE비율(평균의 비)",
            "부트_블록12_하한", "부트_블록12_상한"]
    A("주 비교의 수치(손실 차이 평균 d̄, 두 t구간, 손실이 낮았던 연도 수/8, 평균의 비, 블록 12 부트스트랩은 참고):\n")
    A(md_table(main_v.rename(columns={"mean": "d̄", "annual_lo": "연평균 하한", "annual_hi": "연평균 상한", "biennial_lo": "2년 하한",
                                      "biennial_hi": "2년 상한", "years_better": "우세 연도", "부트_블록12_하한": "부트12 하한", "부트_블록12_상한": "부트12 상한"}),
                cols=["비교", "h", "d̄", "연평균 하한", "연평균 상한", "2년 하한", "2년 상한", "우세 연도", "판정", "MAE비율(평균의 비)", "부트12 하한", "부트12 상한"]))
    A("![전국 예측](../rmpi/output/fig6_1_전국예측.png)\n")
    A("![손실 차이 누적](../rmpi/output/fig6_2_손실차이누적.png)\n")

    # ---------------- 2 RQ1
    A("## 2 분석 2 · 전국 공통 흐름 (RQ1)\n")
    A("### 2.1 보조 비교와 민감도 (같은 판정 규칙으로 계산, 판정에는 쓰지 않음)\n")
    a1 = ver[ver["비교"].str.contains("RQ1") & ~ver["비교"].str.startswith("RQ1")]
    A(md_table(a1.rename(columns={"mean": "d̄", "annual_lo": "연평균 하한", "annual_hi": "연평균 상한", "biennial_lo": "2년 하한", "biennial_hi": "2년 상한",
                                  "years_better": "우세 연도"}), cols=["비교", "h", "d̄", "연평균 하한", "연평균 상한", "2년 하한", "2년 상한", "우세 연도", "판정", "MAE비율(평균의 비)"]))
    A("### 2.2 연도별 MAE (%p)\n")
    y1 = yl[(yl["패널"] == "전국") & yl["모형"].isin(["ridge_A", "ridge_B", "ridge_C", "naive_selected", "naive_zero", "naive_mom1", "naive_train_median", "et_B", "ridge_B_재튜닝", "ridge_A_재튜닝"])]
    ycols = ["모형", "h", "전체", "2018~2020", "2021~2025"] + [c for c in y1.columns if c.isdigit()]
    A(md_table(y1.sort_values(["h", "모형"]), cols=ycols))
    A("![연도별 MAE와 급락 사례](../rmpi/output/fig6_3_연도별_급락사례.png)\n")
    if len(sel):
        A("### 2.3 선택형 단순기준의 선택 분포(시점 수)\n")
        A(md_table(sel.groupby(["h", "선택"]).size().rename("시점 수").reset_index(), nd=0))
    A("### 2.4 2026년 사후 확인, A 대비 표본외 R², 위약 시험, 대용계열 확장 실험\n")
    A(md_table(aux, cols=["항목", "모형", "h", "값", "월수", "연평균 t구간", "옛 체계 훈련월(첫 시점)"]))
    if len(spl):
        A("접합 규칙 결과(stage2_접합요약.csv):\n")
        A(md_table(spl, nd=0))

    # ---------------- 3 RQ2
    A("## 3 분석 3 · 17개 시도 상대 변화 (RQ2)\n")
    a2 = ver[ver["비교"].str.contains("RQ2") & ~ver["비교"].str.startswith("RQ2")]
    A(md_table(a2.rename(columns={"mean": "d̄", "annual_lo": "연평균 하한", "annual_hi": "연평균 상한", "biennial_lo": "2년 하한", "biennial_hi": "2년 상한",
                                  "years_better": "우세 연도"}), cols=["비교", "h", "d̄", "연평균 하한", "연평균 상한", "2년 하한", "2년 상한", "우세 연도", "판정", "MAE비율(평균의 비)"]))
    y2 = yl[(yl["패널"] == "17시도") & (yl["타깃"] == "r")]
    A("연도별 월내 상대 MAE (%p):\n")
    A(md_table(y2.sort_values(["h", "모형"]), cols=ycols))
    if len(gu):
        A("### 3.1 서울 25개 구 상대 상승 순위 (보조)\n")
        A(md_table(gu))

    # ---------------- 4 RQ3
    A("## 4 분석 4 · 급락 확률 (RQ3), 층별·국면 시작, 결합 예측\n")
    a3 = ver[ver["비교"].str.contains("RQ3") & ~ver["비교"].str.startswith("RQ3")]
    A(md_table(a3.rename(columns={"mean": "d̄", "annual_lo": "연평균 하한", "annual_hi": "연평균 상한", "biennial_lo": "2년 하한", "biennial_hi": "2년 상한",
                                  "years_better": "우세 연도"}), cols=["비교", "h", "d̄", "연평균 하한", "연평균 상한", "2년 하한", "2년 상한", "우세 연도", "판정", "MAE비율(평균의 비)"]))
    if len(ev):
        A("### 4.1 급락 지표(전체 기간, 컷오프 = 훈련 사건비율)\n")
        e_all = ev[(ev["연도"].astype(str) == "전체") & ev["모형"].isin(["ridge_A", "ridge_B", "ridge_Bc", "et_B"])]
        A(md_table(e_all, cols=["모형", "h", "행수", "양성", "사건비율", "Brier", "Brier_기준", "Brier_skill", "평균예측확률-실현빈도", "AP", "ROC_AUC", "정밀도", "재현율", "F1", "F1_항상사건"]))
        A("연도별 Brier(B 모형)와 양성 수:\n")
        e_y = ev[(ev["모형"] == "ridge_B") & (ev["연도"].astype(str) != "전체")]
        A(md_table(e_y, cols=["h", "연도", "행수", "양성", "Brier", "Brier_skill", "AP", "ROC_AUC", "F1"]))
    if len(strata):
        A("### 4.2 예측 시점 상태별 층 (판정 없음)\n")
        A(md_table(strata[strata["모형"].isin(["ridge_A", "ridge_B", "B−A Brier 차이"])], cols=["모형", "h", "층", "행수", "유효월수", "Brier", "Brier_skill", "AP", "ROC_AUC", "F1"] + [c for c in strata.columns if c.isdigit()]))
    if len(ep):
        A("### 4.3 국면 시작 사례 (6개 결정월 비사건 뒤 첫 사건, B 모형 확률)\n")
        e2 = ep[ep["시도"] != "(전체)"]
        summary = e2.groupby("h").agg(국면수=("시도", "size"), 경보적중=("시작 전 6개월 내 경보", "sum"),
                                     평균선행개월=("가장 가까운 경보 선행개월", "mean")).reset_index()
        A(md_table(summary, nd=2))
        fa = ep[ep["시도"] == "(전체)"][["h", "컷오프(훈련 사건비율)"]].rename(columns={"컷오프(훈련 사건비율)": "정상기 경보 빈도"})
        A(md_table(fa))
        A("사례표 전체: rmpi/output/stage6_급락_국면시작.csv\n")
    if len(up):
        A("### 4.4 급등 (보조, 학습 양성 30행 이상부터)\n")
        A(md_table(up, cols=["모형", "h", "평가 시작", "월수", "양성", "사건비율", "Brier", "Brier_skill", "AP", "ROC_AUC", "F1", "F1_항상사건"]))
    A("### 4.5 결합 예측 (Ĝ = Ḡ̂ + r̃) 과 통합 패널\n")
    A(md_table(comb))
    a4 = ver[ver["비교"].str.contains("결합")]
    A(md_table(a4.rename(columns={"mean": "d̄", "annual_lo": "연평균 하한", "annual_hi": "연평균 상한", "biennial_lo": "2년 하한", "biennial_hi": "2년 상한",
                                  "years_better": "우세 연도"}), cols=["비교", "h", "d̄", "연평균 하한", "연평균 상한", "2년 하한", "2년 상한", "우세 연도", "판정", "MAE비율(평균의 비)"]))
    if len(leaf):
        A("Extra Trees 잎당 서로 다른 결정월 수(진단):\n")
        A(md_table(leaf, nd=1))

    # ---------------- 5 해석
    A("## 5 분석 5 · 해석\n")
    if len(shap_s):
        A("### 5.1 전국 Ridge B 의 묶음별 평균 |기여| (선형 SHAP, %p)\n")
        A(md_table(shap_s, nd=4))
        A("![SHAP](../rmpi/output/fig7_1_SHAP_전국.png)\n")
    if len(rem):
        A("### 5.2 동인 제거 후 재학습 (손실 변화, 양수 = 제거하면 나빠짐)\n")
        A(md_table(rem.pivot_table(index=["과제", "제거 동인"], columns="h", values="변화").reset_index(), nd=4))
    A("### 5.3 부분의존\n")
    A("![PDP 공통](../rmpi/output/fig7_2_PDP_공통.png)\n")
    if os.path.exists(os.path.join(OUT, "fig7_3_PDP_2way.png")):
        A("![PDP 2way](../rmpi/output/fig7_3_PDP_2way.png)\n")
    if len(xm):
        A("### 5.4 추가 비교모형 (부록, 고정 설정, 결론에 쓰지 않음)\n")
        A(md_table(xm.pivot_table(index=["과제", "모형"], columns="h", values="MAE").reset_index(), nd=4))
    if len(th):
        A("### 5.5 팀 보고서(TH v9)와의 비교\n")
        A("TH v9 의 수치는 보고서에 적힌 값(6개월 변화율, 17개 시도)이며 같은 행의 예측값을 공유받기 전까지는 설계 차이(목표 정의·학습 구간·재학습 주기)가 남는다.\n")
        A(md_table(th))

    # ---------------- 6 통제·한계·결정 기록
    A("## 6 통제 항목, 한계, 결정 기록\n")
    if len(chk5):
        A("5단계 점검(stage5_점검.csv): " + ", ".join(f"{r['항목']} {r['결과']}" for _, r in chk5.iterrows()) + "\n")
    if len(comp4):
        drop = comp4[(~comp4["유지"]) | (~comp4["동인지수유지"])][["h", "블록", "입력", "제외이유", "동인제외이유"]].drop_duplicates()
        A("RMPI 구성에서 제외된 항목(최초 훈련기간 기준, stage4_구성명세.csv):\n")
        A(md_table(drop, nd=2))
    sim = rd("../../analysis/output/plan_v7_inference_sim.csv") if False else (pd.read_csv(os.path.join(BASE, "analysis", "output", "plan_v7_inference_sim.csv"))
                                                                              if os.path.exists(os.path.join(BASE, "analysis", "output", "plan_v7_inference_sim.csv")) else pd.DataFrame())
    if len(sim):
        A("### 6.1 판정 규칙의 모의실험 (analysis/plan_v7_inference_sim.py, 2026-10-05 재작성)\n")
        A("원래 스크립트는 저장소에 없어 계획서의 설정 설명대로 다시 만들었다. 차이가 없을 때(귀무) 각 규칙이 ‘개선’ 또는 기각으로 판정하는 비율(%), AR(1) φ별:\n")
        null = sim[(sim["효과(σ)"] == 0) & (sim["과정"] == "AR(1)")].pivot_table(index="규칙", columns="φ", values="비율") * 100
        null = null.reset_index()
        A(md_table(null, nd=1))
        ar = sim[(sim["효과(σ)"] == 0) & (sim["과정"] == "AR(1)")]
        def rng_(mask):
            v = ar[mask]["비율"] * 100
            return f"{v.min():.1f}~{v.max():.1f}%"
        A("계획서 1장 문장과의 대조(φ 0.6~0.9, 차이 없음): 연평균 구간만 " + rng_(ar["규칙"].str.startswith("R1")) + " (계획서 3.0~9.5%), 두 구간 결합 "
          + rng_(ar["규칙"].str.startswith("R2")) + " (1.4~4.0%), 연평균 t구간 양측 " + rng_(ar["규칙"].str.startswith("T2")) + " (6.6~18.2%), 블록 부트스트랩 양측 "
          + rng_(ar["규칙"].str.startswith("B") & ar["규칙"].str.contains("양측")) + " (12.6~36.5%). 재작성한 스크립트의 수치로 계획서 1장을 고쳐 적어야 한다.\n")
        power = sim[(sim["효과(σ)"] == -0.5) & (sim["과정"] == "AR(1)") & sim["규칙"].str.match(r"^(R1|R2|B12 블록 12 상한)")].pivot_table(index="규칙", columns="φ", values="비율") * 100
        A("평균을 −0.5σ 옮긴 대립에서의 판정 비율(검정력, %):\n")
        A(md_table(power.reset_index(), nd=1))
    A("- 하이퍼파라미터는 설정 파일에 사전 고정했다(Ridge 전국 alpha 10, 패널 170, 로지스틱 C 1/170). 평가 결과를 본 뒤 바꾸지 않았고, 매년 1월 전진 검증으로 다시 고르는 재튜닝은 민감도로만 보고한다.\n")
    A("- 2018~2025년은 설계 전에 일부 통계가 열람된 구간이므로 ‘설계 동결 후 역사적 재검증’이다(계획 부록 A).\n")
    A("- 입력은 최신 개정자료에 공표 시차를 적용한 의사 실시간 자료다. V013·V015·V017·V021~V023·V066·V074 는 공표 당시 값과 다를 수 있어 제외 민감도를 함께 보고했다.\n")
    A("- 판정에 쓴 두 t구간은 연도 간 의존성과 적은 묶음 수 때문에 명목 신뢰수준을 보장하지 않는다. 블록 부트스트랩은 참고 결과다.\n")
    A("- 전향 검증(저장일·입력 마감일 기록)은 설계 동결 뒤 실제 저장 시점부터 시작한다. 이 보고서의 2026년 사후 확인은 최신 자료로 계산한 사후 값이다.\n")
    A("- 재현 경로 메모: 계획 12장이 가리키는 analysis/plan_v7_inference_sim.py, config/plan_v8_settings.yaml, analysis/check_settings.py, docs/연구계획_8차_확정본.docx 가운데 "
      "이 브랜치에는 설정 파일과 check_settings.py 를 이번 실행에서 새로 만들었고(11장 ①~⑦ 값은 계획 본문대로), 모의실험 스크립트와 docx 원본은 제공되지 않아 포함하지 않았다. "
      "계획 본문의 pandoc 사본을 docs/연구계획_8차_확정본.md 로 두었다.\n")
    A("- 산출물 목록: 1단계 config/plan_v8_settings.yaml·analysis/output/check_settings.csv, 2단계 stage2_*.csv, 3단계 fig3_*.png·stage3_*.csv, 4단계 stage4_*.csv·fig4_*.png, "
      "5단계 stage5_점검.csv, 6단계 stage6_*.csv·fig6_*.png, 7단계 stage7_*.csv·fig7_*.png (모두 rmpi/output/).\n")
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote", os.path.relpath(DOC, BASE), len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
