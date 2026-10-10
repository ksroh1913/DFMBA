# -*- coding: utf-8 -*-
"""
롤링 모형 결과 카드 — rolling_models.py 의 1차(고정 파라미터) 결과와 2차(튜닝) 결과를 한 html 로.
한눈에 보기(최선 모형·개선·평가 설계의 한계)  ① 설정(학습·검증·평가 집합 구성 그림 포함)  ② 사건 수(창별)  ③ 학습창 민감도(창별 전체 유효 기간 + 창 비교용 공통 기간)
④ 베이스라인 대비 개선(창별)  ⑤ 튜닝 전후 비교(같은 결정월에서)와 선택된 파라미터  ⑥ 시간에 따른 롤링 성능  ⑦ 과적합  ⑧ 기여도  ⑨ 결과 분석(--notes 파일)  ⑩ 읽는 법
평가 결정월: 학습창 W 는 창이 처음 꽉 차는 달 t_W = y0 + 12W − 1 + h 부터(rolling_models.first_full). 최선 창 선택은 공통 기간(가장 긴 창의 첫 달~)에서, 수치는 저장된 예측값에서 계산
사용: PYTHONUTF8=1 python analysis/model_card.py [--dir 결과폴더] [--tuned-dir 튜닝결과폴더] [--notes 분석글.html]
"""
import argparse
import base64
import datetime as dt
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rolling_models import summarize, first_full  # noqa: E402
from card_style import page  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALUES = os.path.join(BASE, "analysis", "output", "모형입력표_10차_values.csv")
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#4a3aa7")
MODEL_COLOR = {"AR(Ridge)": C["ink"], "AR(Logit)": C["ink"], "Ridge": C["blue"], "Logit": C["blue"], "RF": C["orange"], "ET": C["aqua"], "XGB": C["violet"],
               "Zero": C["muted"], "Mom(past_h)": C["muted"], "Mom1(h×past1)": C["ink2"], "Clim": C["muted"]}
MODEL_STYLE = {"AR(Ridge)": "--", "AR(Logit)": "--", "Zero": ":", "Mom(past_h)": "-.", "Mom1(h×past1)": (0, (1, 1)), "Clim": ":"}
ORDER = ["AR(Ridge)", "AR(Logit)", "Mom1(h×past1)", "Ridge", "Logit", "RF", "ET", "XGB", "RF(튜닝)", "ET(튜닝)", "XGB(튜닝)", "Mom(past_h)", "Zero"]   # Clim(기준율 예측)은 BSS 계산에만 쓰고 표·그림에는 표시하지 않음(사용자 요청)


def add_mom1(pred):
    """예측 파일에 9차·TH 와 같은 단순 기준 mom1 = h × past1 (최근 1개월 변화의 h개월 연장) 참고 행을 더한다 (없을 때만)"""
    if "Mom1(h×past1)" in set(pred["model"]):
        return pred
    v = pd.read_csv(VALUES, encoding="utf-8-sig")[["region", "결정월", "past1"]]
    base = pred[pred["model"] == "Zero"].merge(v, left_on=["region", "t"], right_on=["region", "결정월"], how="left")
    base["model"] = "Mom1(h×past1)"
    base["pred"] = base["h"] * base["past1"]
    return pd.concat([pred, base.drop(columns=["결정월", "past1"])], ignore_index=True)


# 평가 규칙(2026-10-10 결정): 학습창 W 는 자료의 첫 정답 y0(2015-07)으로부터 창이 처음 꽉 차는 달 t_W = y0 + 12W − 1 + h 부터 평가하고, 그 전 달은 그 창으로
# 학습·평가하지 않는다(rolling_models.first_full). 창끼리 비교(최선 창 선택)는 네 창이 모두 유효한 공통 기간(가장 긴 창의 첫 달~)으로 한다.
# 평가를 2018-01 부터 모든 창에 적용했던 이전 실행(커밋 c0003b3)의 수치 — 한계 상자에서 비교용으로만 인용(달마다 독립 학습이라 겹치는 달의 예측은 동일, 재실행으로 확인)
PREV_RUN = dict(start="2018-01", n_months={6: 99, 3: 102}, et=0.791, ar=0.872, mom1=0.790, up_best="RF 0.789 / +0.511", up_ar="0.690 / +0.373", dn_best="XGB 0.655 / +0.377", dn_ar="0.589 / +0.353",
                regimes="5년 창 ET vs mom1: 2018~19 급락기 0.74 vs 0.71, 2020 급등기 1.55 vs 0.88")


def _y0(h):
    """정답 G_h 가 있는 첫 결정월"""
    v = pd.read_csv(VALUES, encoding="utf-8-sig")
    return pd.Period(v.dropna(subset=[f"G{h}"])["결정월"].min(), "M")


def eval_starts(h, windows):
    """학습창별 첫 평가 결정월 {W: Period}"""
    y0 = _y0(h)
    return {int(W): first_full(h, int(W), y0) for W in windows}


def common_start(h, windows):
    """창끼리 비교하는 공통 기간의 시작 = 가장 긴 창의 첫 평가 결정월"""
    return max(eval_starts(h, windows).values())


def restrict_common(pred, windows):
    """예측 기간 h 마다 모든 창이 유효한 공통 기간만 남긴다"""
    parts = []
    for h, g in pred.groupby("h"):
        parts.append(g[pd.PeriodIndex(g["t"], freq="M") >= common_start(int(h), windows)])
    return pd.concat(parts, ignore_index=True)


def load_pred(d):
    """예측 파일(모든 h)을 읽고 mom1 참고 행을 더한다"""
    files = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("predictions_h") and f.endswith(".csv")]
    return add_mom1(pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True))


def load_results(d, common=False):
    """저장된 예측값에서 지표를 다시 계산. common=True 면 창 비교용 공통 기간만"""
    pred = load_pred(d)
    if common:
        pred = restrict_common(pred, sorted(pred["W"].unique()))
    return summarize(pred)
TASK_KR = {"reg": "회귀: 변화율(%p)", "up": "급등 예측(h개월 뒤 +1% 이상)", "dn": "급락 예측(h개월 뒤 −1% 이하)"}
# 2차 튜닝 결과가 1차와 다른 자료 버전으로 계산됐을 때 ⑤ 에 붙이는 주석 (튜닝 재실행 뒤 빈 문자열로)
TUNED_NOTE = ("2차 튜닝은 V078 을 '최근 12개월 내 변경(상태화, ±1 자름)' 으로 넣은 이전 자료로 계산된 결과다(2026-10-10 V078 을 '최근 12개월 변경 합' 으로 바꾼 뒤 1차·기여도만 재실행, 튜닝은 효과가 작아 보류). "
              "1차와 자료가 V078 한 열만 다르며, 이 절의 '고정' 행도 당시 1차 예측으로 같은 자료 기준이다")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"], "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlecolor": C["ink"], "legend.fontsize": 8, "legend.frameon": False})


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    return "<img src='data:image/png;base64," + base64.b64encode(buf.getvalue()).decode() + "'/>"


def style_of(model):
    base = model.replace("(튜닝)", "")
    return dict(color=MODEL_COLOR.get(base, C["ink"]), ls=(0, (4, 1.5)) if model.endswith("(튜닝)") else MODEL_STYLE.get(model, "-"),
                lw=2 if base in ("RF", "ET", "XGB", "Ridge", "Logit") else 1.6)


def mline(ax, x, y, model, **kw):
    ax.plot(x, y, label=model, **style_of(model), **kw)


def shared_legend(fig, axes, right=0.9):
    handles = {}
    for ax in np.ravel(axes):
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    order = [m for m in ORDER if m in handles]
    fig.legend([handles[m] for m in order], order, loc="center left", bbox_to_anchor=(right + 0.005, 0.5), frameon=False)
    fig.subplots_adjust(right=right)


def tbl(df, **kw):
    return df.to_html(float_format=lambda x: f"{x:.3f}", na_rep="", **kw)


def glossary_table():
    rows = [
        ("결정월 t", "예측을 내리는 시점(그 달 말). 이때까지 공표된 자료만 쓴다"),
        ("h개월 뒤 예측 (h=6 주, 3 보조)", "결정월에 아는 마지막 월세지수 R(t−1)에서 h개월 뒤까지의 변화율 G_h = 100·[R(t+h−1)/R(t−1) − 1] 을 맞히는 것. 영어로는 forecast horizon"),
        ("회귀 / 분류", "회귀 = 변화율(%) 자체를 맞힘. 분류 = 급등(G_h ≥ +1%)·급락(G_h ≤ −1%)이 일어날 확률을 맞힘"),
        ("베이스라인", "설명변수 없이 월세지수 자신의 과거 1·3·6개월 변화율만 쓰는 단순 모형(AR). 비교 모형이 이걸 못 넘으면 설명변수가 쓸모없다는 뜻"),
        ("mom1 / Mom(past_h) / Zero", "학습 없이 규칙으로 내는 참고 값: 최근 1개월 변화율 × h / 최근 h개월 변화율 그대로 / 0% (변화 없음)"),
        ("학습창(W년) · 롤링", "결정월마다 직전 W년 자료로 다시 학습하고 다음 달로 한 칸씩 옮기는 방식. W를 2·3·4·5년으로 바꿔 가며 어느 길이가 좋은지 봄"),
        ("정답 확정 행", "h개월 뒤 결과가 결정월 시점에 이미 공표된 행(s ≤ t−h). 학습에는 이런 행만 써서 미래 정보가 섞이지 않게 함"),
        ("MAE", "예측과 실제 변화율의 차이(절댓값)의 평균, 단위 %p. 작을수록 좋음"),
        ("F1", "'급등이다'라고 찍은 것 중 맞힌 비율(정밀도)과 실제 급등 중 잡아낸 비율(재현율)의 조화평균. 0~1, 클수록 좋음"),
        ("PR-AUC", "확률 순서만으로 급등·급락을 얼마나 잘 가려내는지. 아무 정보 없으면 사건 비율(예: 0.30)과 같고, 1이면 완벽"),
        ("BSS", "확률 예측의 정확도(Brier)를 '학습창의 사건 비율만 말하는 예측'과 비교한 점수. 0이면 그와 같고, 양수면 더 낫고, 음수면 더 못함(과신)"),
        ("적합 vs 평가", "적합 = 모형이 학습한 자료에서의 성능, 평가 = 학습에 안 쓴 다음 달 자료에서의 성능. 둘의 격차로 과적합을 봄"),
        ("SHAP", "한 예측을 특성별 기여로 쪼갠 값(합하면 예측값). 평균 |SHAP|가 크면 그 변수가 예측을 많이 움직였다는 뜻"),
    ]
    return pd.DataFrame(rows, columns=["용어", "뜻"]).to_html(index=False)


# ------------------------------------------------------------ ① 설정: 기준·모형 정의 표
def definitions_table():
    rows = [
        ("Zero", "참고 기준", "Ĝ_h = 0 (변화 없음)", "학습 없음", "zero"),
        ("Mom1(h×past1)", "참고 기준", "Ĝ_h = h × past1 (최근 1개월 변화율을 h개월 연장)", "학습 없음", "mom1 — 9차 주 분석의 기준(M0), TH_v19 의 mom1 과 같음"),
        ("Mom(past_h)", "참고 기준", "Ĝ_h = past_h (최근 h개월 변화율이 그대로 이어진다)", "학습 없음", "lasth"),
        ("AR(Ridge)", "베이스라인(회귀)", "Ĝ_h = b0 + b1·past1 + b3·past3 + b6·past6", "학습창(직전 W년, 17개 시도 풀링)에서 매월 Ridge 로 추정. alpha 는 학습창 안 LOO 로 0.01~1000 중 선택. 설명변수·지역 더미 없음", "없음(9차 A 정보군은 past1·3·6 에 V002·V026 6개월 변화를 더한 것)"),
        ("AR(Logit)", "베이스라인(분류)", "P(사건) = σ(b0 + b1·past1 + b3·past3 + b6·past6)", "학습창에서 매월 로지스틱(C=1) 추정", "없음"),
        ("Ridge / Logit", "비교 모형", "모멘텀 3열 + 설명변수 47열 + 지역 더미 17열의 선형 모형", "Ridge: alpha LOO 선택 / Logit: C=1", "9차 B·C 정보군에 해당(단, 9차는 동인 지수로 압축)"),
        ("RF / ET / XGB", "비교 모형", "같은 특성의 트리 모형(랜덤포레스트·엑스트라트리·XGBoost)", "1차 고정, 2차 '(튜닝)' 은 학습창 안 중첩 시계열 CV 로 격자 선택", "TH_v19 의 RF·ET·XGB 와 같은 계열(설정값은 다름)"),
    ]
    return pd.DataFrame(rows, columns=["카드 이름", "구분", "예측식", "학습·파라미터", "9차·TH 대응"]).to_html(index=False)


# ------------------------------------------------------------ ① 설정: 집합의 실제 기간 표
def period_table(roll, h, windows, tuned_windows=()):
    """예측 기간 h: 창별 첫 평가 결정월과, 창마다 대표 결정월의 학습 구간(실제 자료 기준)·튜닝 검증 겹 기간을 표로"""
    y0 = _y0(h)
    starts = eval_starts(h, windows)
    r = roll[(roll["h"] == h) & (roll["task"] == "reg")]
    t_last = max(pd.PeriodIndex(r["t"].unique(), freq="M"))
    rows = []
    for W in windows:
        tW = starts[W]
        n_W = int(r[r["W"] == W]["t"].nunique())
        for t in [tW] + [p for p in (pd.Period("2022-01", "M"), pd.Period("2024-01", "M")) if tW < p < t_last] + [t_last]:
            s_hi = t - h
            s_lo = max(s_hi - 12 * W + 1, y0)
            n = (s_hi - s_lo).n + 1
            folds = ""
            if W in tuned_windows:
                parts = []
                for k in range(3):
                    v_end = s_hi - 3 * (2 - k)
                    v_lo = v_end - 2
                    parts.append(f"겹{k + 1} 학습 {s_lo}~{v_lo - h} / 검증 {v_lo}~{v_end}")
                folds = "; ".join(parts)
            rows.append({"학습창 W": f"{W}년 (평가 {tW}~{t_last}, {n_W}개월)", "평가 결정월 t": str(t), "학습 구간(결정월 s)": f"{s_lo}~{s_hi} ({n}개월 × 17 = {n * 17:,}행)" + (" ← W년보다 짧음" if n < 12 * W else ""),
                         "비움(정답 미확정)": f"{s_hi + 1}~{t - 1} ({h - 1}개월)", "검증 겹(튜닝 모형만)": folds, "평가": f"{t} 의 17개 시도, 정답은 {t + h - 1} 지수 공표 뒤"})
    head = (f"<p class='note'><b>h={h}개월</b>: 정답이 있는 첫 결정월 {y0}. 학습창 W 는 창이 처음 꽉 차는 달 t_W = {y0} + 12W − 1 + {h} 부터 평가 → "
            + ", ".join(f"{W}년 {starts[W]}~" for W in windows) + f" (마지막 {t_last}). 그 전 달은 그 창으로 학습·평가하지 않으며(자료는 학습 행으로만), "
            f"창끼리 비교하는 공통 기간은 {common_start(h, windows)}~ 이다</p>")
    return head + pd.DataFrame(rows).to_html(index=False)


# ------------------------------------------------------------ ⑦ 과적합 진단
def overfit_section(d, summ, tuned_dir=None, summ_c=None, st_t=None, start=None):
    path = os.path.join(d, "fit_metrics.csv")
    if not os.path.exists(path):
        return "<p class='note'>fit_metrics.csv 가 없어 적합 성능을 표시하지 못함 (rolling_models.py 재실행 필요)</p>"
    fit = pd.read_csv(path, encoding="utf-8-sig")
    if start:
        fit = fit[fit["t"] >= start]                       # 적합 성능도 지표 집계 기간의 결정월만 평균
    agg = fit.groupby(["h", "W", "task", "model"]).agg(MAE_train=("MAE_train", "mean"), Brier_train=("Brier_train", "mean"), PR_AUC_train=("PR_AUC_train", "mean")).reset_index()
    m = agg.merge(summ[["h", "W", "task", "model", "MAE", "Brier", "PR_AUC", "BSS"]], on=["h", "W", "task", "model"], how="left")
    reg = m[m["task"] == "reg"].copy()
    reg["평가/적합 비율"] = reg["MAE"] / reg["MAE_train"]
    reg_t = reg.rename(columns={"MAE_train": "MAE 적합(학습창 안)", "MAE": "MAE 평가"})[["h", "W", "model", "MAE 적합(학습창 안)", "MAE 평가", "평가/적합 비율"]]
    clf = m[m["task"] != "reg"].copy()
    clf["과제"] = clf["task"].map({"up": "급등", "dn": "급락"})
    clf_t = clf.rename(columns={"PR_AUC_train": "PR-AUC 적합", "PR_AUC": "PR-AUC 평가", "Brier_train": "Brier 적합", "Brier": "Brier 평가"})[["h", "W", "과제", "model", "PR-AUC 적합", "PR-AUC 평가", "Brier 적합", "Brier 평가", "BSS"]]
    # 그림: 예측 기간별로 적합(점선·빈 표식) vs 평가(실선·채운 표식)
    figs = ""
    for h in sorted(m["h"].unique(), reverse=True):
        panels = [("reg", "MAE_train", "MAE", "MAE: 적합(점선) vs 평가(실선)"), ("up", "PR_AUC_train", "PR_AUC", "급등 PR-AUC: 적합 vs 평가"), ("dn", "PR_AUC_train", "PR_AUC", "급락 PR-AUC: 적합 vs 평가")]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
        for ax, (task, mtr, mte, title) in zip(axes, panels):
            g = m[(m["h"] == h) & (m["task"] == task)]
            for model in [x for x in ORDER if x in set(g["model"])]:
                gg = g[g["model"] == model].sort_values("W")
                st_ = style_of(model)
                ax.plot(gg["W"], gg[mtr], color=st_["color"], ls=":", lw=1.4, marker="o", mfc="none", ms=5)
                ax.plot(gg["W"], gg[mte], color=st_["color"], ls=st_["ls"], lw=st_["lw"], marker="o", ms=4, label=model)
            ax.set_title(title, loc="left")
            ax.set_xticks(sorted(g["W"].unique()))
            ax.set_xlabel("학습창(년)")
        fig.suptitle(f"{h}개월 뒤 예측 — 학습 자료에서의 성능(적합)과 처음 보는 자료에서의 성능(평가)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
        fig.tight_layout()
        shared_legend(fig, axes, right=0.86)
        figs += img(fig)
    html = ("<p class='note'><b>읽는 법</b>: 적합(train)은 모형이 학습한 바로 그 행에서의 성능, 평가(test)는 매월 전진한 표본 외 성능" + (f"(둘 다 결정월 {start} 이후 평균)" if start else "") + ". 트리 모형은 적합 오차가 매우 작은 것이 정상이므로 격차 자체보다 "
            "(1) 평가 성능이 베이스라인을 넘는지, (2) 학습창이 길어질 때 평가 성능이 나아지고 격차가 줄어드는지, (3) 적합은 좋은데 평가가 베이스라인보다 못한 모형(전형적 과적합)이 무엇인지를 본다.</p>"
            + figs + "<h3>회귀</h3>" + tbl(reg_t.sort_values(["h", "W", "model"]).reset_index(drop=True), index=False)
            + "<h3>분류</h3>" + tbl(clf_t.sort_values(["h", "W", "과제", "model"]).reset_index(drop=True), index=False))
    if tuned_dir and os.path.exists(os.path.join(tuned_dir, "tuned_params.csv")) and summ_c is not None:
        p = pd.read_csv(os.path.join(tuned_dir, "tuned_params.csv"), encoding="utf-8-sig")
        if start:
            p = p[p["t"] >= start]
        cv = p.groupby(["h", "W", "task", "model"])["cv_score"].mean().reset_index()
        cv["model"] = cv["model"] + "(튜닝)"
        cv = cv.merge(summ_c[["h", "W", "task", "model", "MAE", "PR_AUC"]], on=["h", "W", "task", "model"], how="left")
        cv["검증(CV) 점수"] = np.where(cv["task"] == "reg", -cv["cv_score"], cv["cv_score"])
        cv["평가 점수"] = np.where(cv["task"] == "reg", cv["MAE"], cv["PR_AUC"])
        cv["지표"] = np.where(cv["task"] == "reg", "MAE(낮을수록)", "PR-AUC(높을수록)")
        cv["과제"] = cv["task"].map({"reg": "회귀", "up": "급등", "dn": "급락"})
        html += ("<h3>튜닝 모형: 학습창 안 검증(CV) 점수 vs 평가 점수</h3><p class='note'>검증 점수 = 격자 선택에 쓴 3겹 평균(결정월 평균). 검증이 평가보다 많이 좋으면 검증 겹에 맞춘 선택(선택 과적합)을 의심</p>"
                 + tbl(cv[["h", "W", "과제", "model", "지표", "검증(CV) 점수", "평가 점수"]].sort_values(["h", "W", "과제"]).reset_index(drop=True), index=False))
    return html


# ------------------------------------------------------------ ① 설정: 창별 평가 기간 그림
def eval_design_fig(hs, windows, last):
    """학습창마다 창이 처음 꽉 차는 달부터 평가하는 설계를 한 장으로 (last = {h: 마지막 평가 결정월})"""
    ts = lambda p: p.to_timestamp()  # noqa: E731
    fig, axes = plt.subplots(len(hs), 1, figsize=(12, 3.2 * len(hs)), sharex=True, squeeze=False)
    for ax, h in zip(axes[:, 0], hs):
        y0, st = _y0(h), eval_starts(h, windows)
        cs = max(st.values())
        for k, W in enumerate(windows):
            y = len(windows) - 1 - k
            tW = st[W]
            ax.broken_barh([(ts(y0), ts(tW) - ts(y0))], (y - 0.3, 0.6), color=C["grid"], alpha=0.9)
            ax.broken_barh([(ts(tW), ts(last[h] + 1) - ts(tW))], (y - 0.3, 0.6), color=C["orange"], alpha=0.85)
            ax.text(ts(tW) + pd.Timedelta(days=15), y, f"평가 {tW}~{last[h]} ({(last[h] - tW).n + 1}개월, 매월 롤링)", va="center", fontsize=8.5, color="white", fontweight="bold")
            s_hi = tW - h
            ax.broken_barh([(ts(y0), ts(s_hi + 1) - ts(y0))], (y - 0.3, 0.6), facecolor="none", edgecolor=C["blue"], lw=1.6)
            ax.text(ts(y0) + pd.Timedelta(days=15), y, f"첫 학습창 {y0}~{s_hi} ({12 * W}개월)", va="center", fontsize=8, color=C["blue"])
            ax.text(ts(y0) - pd.Timedelta(days=40), y, f"{W}년 창", ha="right", va="center", fontsize=10, color=C["ink"], fontweight="bold")
        ax.axvline(ts(cs), color=C["violet"], ls="--", lw=1.2)
        ax.text(ts(cs) + pd.Timedelta(days=10), len(windows) - 0.38, f"창끼리 비교하는 공통 기간 {cs}~ (모든 창 유효)", fontsize=8.5, color=C["violet"])
        ax.set_ylim(-0.7, len(windows) - 0.1)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{h}개월 뒤 예측 — 첫 평가 결정월 t_W = {y0} + 12W - 1 + {h}", loc="left")
    ax = axes[-1, 0]
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(ts(pd.Period("2015-01", "M")), ts(max(last.values()) + 4))
    fig.suptitle("평가 설계: 학습창은 창이 처음 꽉 차는 달부터 평가하고, 그 전 자료는 학습 행으로만 쓴다", x=0.01, y=1.0, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=C["grid"]), plt.Rectangle((0, 0), 1, 1, color=C["orange"]), plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=C["blue"], lw=1.6)]
    fig.legend(handles, ["자료 있음(학습 행으로만 쓰임)", "평가 기간(매월 직전 W년으로 새로 학습 → 그 달 예측)", "첫 평가월의 학습창(12W개월)"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return img(fig)


# ------------------------------------------------------------ ① 설정: 집합 구성 그림
def split_fig(h=6, W=4, t="2022-06"):
    t = pd.Period(t, "M")
    s_hi, s_lo = t - h, t - h - 12 * W + 1
    fig, ax = plt.subplots(figsize=(10.5, 2.4))
    ts = lambda p: p.to_timestamp()  # noqa: E731
    ax.broken_barh([(ts(s_lo), ts(s_hi + 1) - ts(s_lo))], (2.6, 0.8), color=C["blue"], alpha=0.85)
    ax.text(ts(s_lo), 3.45, f"학습(train): 결정월 {s_lo}~{s_hi} × 17개 시도 = {12 * W * 17:,}행 (W={W}년, 정답 G{h} 가 t 시점에 확정된 행만)", fontsize=8.5, color=C["ink"], va="bottom")
    ax.broken_barh([(ts(s_hi + 1), ts(t) - ts(s_hi + 1))], (2.6, 0.8), color=C["muted"], alpha=0.45)
    ax.text(ts(s_hi + 1), 2.5, f"정답 미확정 {h - 1}개월(제외)", fontsize=8, color=C["ink2"], va="top")
    ax.broken_barh([(ts(t), ts(t + 1) - ts(t))], (2.6, 0.8), color=C["orange"])
    ax.text(ts(t + 1), 3.0, f"평가(test): 결정월 t={t} 의 17개 시도 (정답은 {t + h - 1} 지수 공표 뒤 확정)", fontsize=8.5, color=C["ink"], va="center")
    # 중첩 CV (튜닝 모형만): 학습 블록 끝 쪽 3겹, 검증 3개월, 간격 h−1
    for k in range(3):
        v_end = s_hi - 3 * (2 - k)
        v_lo = v_end - 2
        tr_end = v_lo - h
        y0 = 1.0 - 0.45 * k
        ax.broken_barh([(ts(s_lo), ts(tr_end + 1) - ts(s_lo))], (y0, 0.32), color=C["blue"], alpha=0.35)
        ax.broken_barh([(ts(v_lo), ts(v_end + 1) - ts(v_lo))], (y0, 0.32), color=C["violet"], alpha=0.8)
        ax.text(ts(s_lo) - pd.Timedelta(days=20), y0 + 0.16, f"겹 {k + 1}", fontsize=7.5, color=C["ink2"], ha="right", va="center")
    ax.text(ts(s_lo), 1.45, "검증(validation, 튜닝 모형만): 학습 블록 안 시간 순 3겹. 연한 파랑 = 겹 학습, 보라 = 검증 3개월, 그 사이 h-1개월 간격. 격자 선택 뒤 전체 학습 블록으로 재학습",
            fontsize=8, color=C["ink2"], va="bottom")
    ax.set_ylim(-0.4, 4.2)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(ts(s_lo) - pd.Timedelta(days=200), ts(t + 8))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"예시: {h}개월 뒤 예측, 학습 기간 W={W}년, 결정월 t={t} 의 학습·검증·평가 구성 (매월 t 를 한 칸씩 옮기며 반복)", loc="left")
    return img(fig)


# ------------------------------------------------------------ ② 사건 수
def event_tables(values_path, start, horizons, thr):
    df = pd.read_csv(values_path, encoding="utf-8-sig")
    a = df[df["결정월"] >= start]
    rows, yearly = [], []
    for h in horizons:
        s = a[f"G{h}"].dropna()
        up, dn = (s >= thr), (s <= -thr)
        rows.append({"예측 기간": f"{h}개월", "평가 행(시도×결정월)": len(s), "결정월 범위": f"{a.loc[s.index, '결정월'].min()}~{a.loc[s.index, '결정월'].max()}",
                     f"급등(≥+{thr:.0f}%) 건수": int(up.sum()), "급등 비율": f"{up.mean():.1%}", f"급락(≤−{thr:.0f}%) 건수": int(dn.sum()), "급락 비율": f"{dn.mean():.1%}",
                     "둘 중 하나": f"{(up | dn).mean():.1%}"})
        g = a.loc[s.index].assign(연도=a.loc[s.index, "결정월"].str[:4], up=up.values, dn=dn.values).groupby("연도").agg(행=("up", "size"), 급등=("up", "sum"), 급락=("dn", "sum"))
        g.columns = [f"h{h} {c}" for c in g.columns]
        yearly.append(g)
    return pd.DataFrame(rows), pd.concat(yearly, axis=1)


def event_tables_by_window(values_path, h, starts, thr):
    """예측 기간 h: 학습창별 평가 기간의 행 수·사건 수 (창마다 첫 평가 결정월이 다르므로)"""
    df = pd.read_csv(values_path, encoding="utf-8-sig")
    rows = []
    for W, tW in sorted(starts.items()):
        a = df[df["결정월"] >= str(tW)]
        s = a[f"G{h}"].dropna()
        up, dn = (s >= thr), (s <= -thr)
        rows.append({"학습창": f"{W}년", "평가 기간": f"{tW}~{a.loc[s.index, '결정월'].max()}", "평가 행(시도×결정월)": len(s),
                     f"급등(≥+{thr:.0f}%) 건수": int(up.sum()), "급등 비율": f"{up.mean():.1%}", f"급락(≤−{thr:.0f}%) 건수": int(dn.sum()), "급락 비율": f"{dn.mean():.1%}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ ③ 민감도
def sensitivity_figs(summ, h, period=""):
    s = summ[summ["h"] == h]
    panels = [("reg", "MAE", "MAE (낮을수록 좋음)"), ("up", "PR_AUC", "급등 PR-AUC (높을수록)"), ("up", "BSS", "급등 BSS (높을수록)"),
              ("dn", "PR_AUC", "급락 PR-AUC"), ("dn", "BSS", "급락 BSS")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2))
    for ax, (task, met, title) in zip(axes, panels):
        g = s[s["task"] == task]
        for model in [m for m in ORDER if m in set(g["model"])]:
            if model == "Clim" and met == "BSS":
                continue
            gg = g[g["model"] == model].sort_values("W")
            mline(ax, gg["W"], gg[met], model, marker="o", ms=4)
        ax.set_title(title, loc="left")
        ax.set_xticks(sorted(s["W"].unique()))
        ax.set_xlabel("학습창(년)")
        if met == "BSS":
            ax.axhline(0, color=C["axis"], lw=0.8)
    fig.suptitle(f"{h}개월 뒤 예측 — 학습 기간(2~5년)별 성능{period}", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    shared_legend(fig, axes, right=0.9)
    return img(fig)


def sensitivity_table(summ, h):
    s = summ[summ["h"] == h]
    out = []
    reg = s[s["task"] == "reg"].pivot_table(index="model", columns="W", values="MAE").round(3)
    reg.columns = [f"MAE W={w}" for w in reg.columns]
    out.append(reg.reindex([m for m in ORDER if m in reg.index]))
    for ev in ("up", "dn"):
        g = s[s["task"] == ev]
        for met in ("PR_AUC", "BSS", "F1_기준율"):
            p = g.pivot_table(index="model", columns="W", values=met).round(3)
            p.columns = [f"{TASK_KR[ev][:2]} {met} W={w}" for w in p.columns]
            out.append(p.reindex([m for m in ORDER if m in p.index]))
    return pd.concat(out, axis=1)


# ------------------------------------------------------------ ④ 개선
def improvement_table(summ):
    rows = []
    cands = [m for m in ORDER if m not in ("AR(Ridge)", "AR(Logit)", "Zero", "Mom(past_h)", "Mom1(h×past1)", "Clim")]
    for (h, W), g in summ.groupby(["h", "W"]):
        reg = g[g["task"] == "reg"].set_index("model")
        if "AR(Ridge)" in reg.index:
            base = reg.at["AR(Ridge)", "MAE"]
            for m in cands:
                if m in reg.index:
                    rows.append(dict(h=h, W=W, 과제="회귀", 모형=m, 지표="MAE", 베이스라인=round(base, 3), 모형값=round(reg.at[m, "MAE"], 3), 개선=f"{100 * (base - reg.at[m, 'MAE']) / base:+.1f}%"))
        for ev in ("up", "dn"):
            c = g[g["task"] == ev].set_index("model")
            if "AR(Logit)" not in c.index:
                continue
            for m in cands:
                if m in c.index:
                    for met in ("PR_AUC", "BSS"):
                        b, v = c.at["AR(Logit)", met], c.at[m, met]
                        rows.append(dict(h=h, W=W, 과제=TASK_KR[ev][:2], 모형=m, 지표=met, 베이스라인=round(b, 3), 모형값=round(v, 3), 개선=f"{v - b:+.3f}"))
    return pd.DataFrame(rows)


def best_table(summ):
    rows = []
    for h, g in summ.groupby("h"):
        r = g[(g["task"] == "reg") & (~g["model"].isin(["Zero", "Mom(past_h)", "Mom1(h×past1)"]))].sort_values("MAE")
        ar = g[(g["task"] == "reg") & (g["model"] == "AR(Ridge)")]["MAE"].min()
        m1 = g[(g["task"] == "reg") & (g["model"] == "Mom1(h×past1)")]["MAE"].min()
        rows.append(dict(h=h, 과제="회귀 MAE", 최선=f"{r.iloc[0]['model']} (W={r.iloc[0]['W']}년) {r.iloc[0]['MAE']:.3f}", AR베이스라인=f"{ar:.3f} (최선 창)", 개선=f"{100 * (ar - r.iloc[0]['MAE']) / ar:+.1f}%",
                         **{"mom1(h×past1) 대비": f"mom1 {m1:.3f} → {100 * (m1 - r.iloc[0]['MAE']) / m1:+.1f}%" if pd.notna(m1) else ""}))
        for ev in ("up", "dn"):
            c = g[(g["task"] == ev) & (g["model"] != "Clim")].sort_values("PR_AUC", ascending=False)
            ar = g[(g["task"] == ev) & (g["model"] == "AR(Logit)")]["PR_AUC"].max()
            rows.append(dict(h=h, 과제=f"{TASK_KR[ev][:2]} PR-AUC", 최선=f"{c.iloc[0]['model']} (W={c.iloc[0]['W']}년) {c.iloc[0]['PR_AUC']:.3f} (BSS {c.iloc[0]['BSS']:+.3f})",
                             AR베이스라인=f"{ar:.3f} (최선 창)", 개선=f"{c.iloc[0]['PR_AUC'] - ar:+.3f}"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ ⑤ 튜닝 비교
def tuned_comparison(fixed_dir, tuned_dir, st_t):
    """튜닝 실행이 평가한 (결정월, 창) 표본만 골라 고정·튜닝 모형을 같은 행에서 비교"""
    preds = []
    for h in st_t["horizons"]:
        pt = pd.read_csv(os.path.join(tuned_dir, f"predictions_h{h}.csv"), encoding="utf-8-sig")
        pt = pt[pt["model"].isin(st_t["tune"])].copy()
        pt["model"] = pt["model"] + "(튜닝)"
        pf = pd.read_csv(os.path.join(fixed_dir, f"predictions_h{h}.csv"), encoding="utf-8-sig")
        keys = set(zip(pt["t"], pt["W"]))
        pf = pf[[(t, w) in keys for t, w in zip(pf["t"], pf["W"])]]
        preds += [pf, pt]
    pred = add_mom1(pd.concat(preds, ignore_index=True))
    summ, roll = summarize(pred)
    return summ, roll


def tuned_params_table(tuned_dir):
    p = pd.read_csv(os.path.join(tuned_dir, "tuned_params.csv"), encoding="utf-8-sig")
    keys = [c for c in p.columns if c not in ("h", "W", "t", "task", "model", "cv_score")]
    rows = []
    for (h, W, task, model), g in p.groupby(["h", "W", "task", "model"]):
        cols = [k for k in keys if g[k].notna().any()]
        combo = g[cols].astype(str).agg(", ".join, axis=1)
        vc = combo.value_counts()
        last = g.sort_values("t").iloc[-1]
        rows.append({"h": h, "W": W, "과제": TASK_KR[task][:2], "모형": model, "격자": ", ".join(cols), "가장 많이 뽑힌 값": vc.index[0], "빈도": f"{vc.iloc[0] / len(g):.0%} ({vc.iloc[0]}/{len(g)}개월)",
                     "마지막 결정월": last["t"], "마지막 결정월 선택값": ", ".join(str(last[c]) for c in cols), "CV 점수 평균": round(float(g["cv_score"].mean()), 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ ⑥ 롤링 곡선
def rolling_figs(roll, h, reg_win=12, clf_win=24, title_extra="", mark_start=None):
    """시기별 이동 창 성능. mark_start 를 주면 그 전 기간(지표 집계 제외)을 회색으로 표시"""
    r = roll[roll["h"] == h].copy()
    r["tP"] = pd.PeriodIndex(r["t"], freq="M")
    Ws = sorted(r["W"].unique())
    cols = [("reg", "MAE", f"{reg_win}개월 이동 MAE"), ("up", "BSS", f"급등 {clf_win}개월 이동 BSS"), ("up", "F1", f"급등 {clf_win}개월 이동 F1(기준율 임계)"),
            ("dn", "BSS", f"급락 {clf_win}개월 이동 BSS"), ("dn", "F1", f"급락 {clf_win}개월 이동 F1(기준율 임계)")]
    fig, axes = plt.subplots(len(Ws), len(cols), figsize=(3.4 * len(cols), 2.5 * len(Ws)), sharex=True, squeeze=False)
    for i, W in enumerate(Ws):
        for j, (task, met, title) in enumerate(cols):
            ax = axes[i, j]
            g = r[(r["W"] == W) & (r["task"] == task)]
            for model in [m for m in ORDER if m in set(g["model"])]:
                if model in ("Clim", "Zero"):
                    continue
                gg = g[g["model"] == model].sort_values("tP").set_index("tP")
                if task == "reg":
                    y = gg["MAE"].rolling(reg_win, min_periods=reg_win).mean()
                elif met == "BSS":
                    y = 1 - (gg["Brier"] * gg["n"]).rolling(clf_win, min_periods=clf_win).sum() / (gg["Brier_clim"] * gg["n"]).rolling(clf_win, min_periods=clf_win).sum()
                else:
                    hit = gg["hitB"].rolling(clf_win, min_periods=clf_win).sum()
                    y = 2 * hit / (gg["predB"].rolling(clf_win, min_periods=clf_win).sum() + gg["n_pos"].rolling(clf_win, min_periods=clf_win).sum())
                mline(ax, y.index.to_timestamp(), y.values, model)
            if met == "BSS":
                ax.axhline(0, color=C["axis"], lw=0.8)
                ax.set_ylim(-1, 1)
            if mark_start:
                ax.axvspan(r["tP"].min().to_timestamp(), pd.Period(mark_start, "M").to_timestamp(), color=C["grid"], alpha=0.55, lw=0, zorder=0)
            if i == 0:
                ax.set_title(title, loc="left")
            if j == 0:
                ax.set_ylabel(f"학습창 {W}년", fontsize=9, color=C["ink"])
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"{h}개월 뒤 예측 — 시기별 성능 변화{title_extra} (가로축 = 결정월, 이동 창 끝 기준" + (f"; 회색 = {mark_start} 전, 지표 집계 제외)" if mark_start else ")"),
                 x=0.01, y=1.01, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    shared_legend(fig, axes, right=0.92)
    return img(fig)


# ------------------------------------------------------------ 맨 위 요약
def _bests(g):
    """한 예측 기간의 요약 행에서 과제별 최선 모형·베이스라인·단순 규칙 mom1 을 뽑는다"""
    r = g[(g["task"] == "reg") & (~g["model"].isin(["Zero", "Mom(past_h)", "Mom1(h×past1)"]))].sort_values("MAE").iloc[0]
    ar = g[(g["task"] == "reg") & (g["model"] == "AR(Ridge)")]["MAE"].min()
    m1 = g[(g["task"] == "reg") & (g["model"] == "Mom1(h×past1)")]["MAE"].min()
    out = dict(r=r, ar=ar, m1=m1, gain=100 * (ar - r["MAE"]) / ar, dm=100 * (m1 - r["MAE"]) / m1 if pd.notna(m1) else np.nan, ev={})
    for ev in ("up", "dn"):
        c = g[(g["task"] == ev) & (g["model"] != "Clim")].sort_values("PR_AUC", ascending=False).iloc[0]
        b = g[(g["task"] == ev) & (g["model"] == "AR(Logit)")].sort_values("PR_AUC", ascending=False).iloc[0]
        out["ev"][ev] = (c, b, float(g[g["task"] == ev]["양성비율"].iloc[0]))
    return out


def caveat_box(summ, summ_c, starts, hs, thr):
    """'창별 첫 유효월부터 평가' 설계의 한계 — 사건 수는 자료에서 직접 계산, 이전 실행(2018-01 부터 모든 창)의 지표는 PREV_RUN 에서 인용"""
    h = hs[0]
    st = starts[h]
    Ws = sorted(st)
    cs = max(st.values())
    pv = PREV_RUN
    ev = event_tables_by_window(VALUES, h, st, thr).set_index("학습창")
    col_n = "평가 행(시도×결정월)"
    col_dn = [c for c in ev.columns if c.startswith("급락") and c.endswith("건수")][0]
    n_by = {W: int(summ[(summ["h"] == h) & (summ["task"] == "reg") & (summ["W"] == W)]["n"].iloc[0] // 17) for W in Ws}
    _, yearly = event_tables(VALUES, str(st[Ws[0]]), [h], thr)                          # 가장 이른 창(2년)의 평가 기간부터 연도별
    dn_col = f"h{h} 급락"
    yrs = yearly.index.astype(str)
    dn_pre, dn_all = int(yearly.loc[yrs < str(cs)[:4], dn_col].sum()), int(yearly[dn_col].sum())
    dn_in = yearly.loc[yrs >= str(cs)[:4], dn_col].sort_values(ascending=False)
    top2 = dn_in.head(2)
    bc = _bests(summ_c[summ_c["h"] == h])
    W5 = int(bc["r"]["W"])
    g = summ[(summ["h"] == h) & (summ["task"] == "reg") & (summ["W"] == W5)].set_index("model")
    items = [
        f"<b>창마다 평가 기간이 다르다.</b> " + ", ".join(f"{W}년 {st[W]}~({n_by[W]}개월)" for W in Ws) + f". 긴 창일수록 평가가 짧고 국면 구성도 다르므로 창 사이의 절대 수치 비교는 공통 기간 {cs}~({n_by[Ws[-1]]}개월)으로만 한다(③ 둘째 표, 맨 위의 최선 창 선택). "
        "같은 창 안의 모형 비교(베이스라인 대비)는 기간이 같으므로 공정하다.",
        f"<b>긴 창은 2018~19 급락기를 평가하지 못한다.</b> {cs} 전의 급락 {dn_pre}건({str(st[Ws[0]])[:4]}년 이후 전체 {dn_all}건의 {dn_pre / dn_all:.0%})은 2·3년 창에서만 평가되고 4·5년 창에서는 학습 행으로만 쓰인다. "
        f"4·5년 창의 급락 성능은 사실상 {top2.index[0]}·{top2.index[1]}년(공통 기간 급락 {int(dn_in.sum())}건의 {top2.sum() / dn_in.sum():.0%}) 하나의 국면으로 판단하는 셈이며, 성격이 다른 급락(지방 공급 과잉형)에 대한 성능은 2·3년 창 결과(④·⑥)로만 가늠할 수 있다.",
        f"<b>표본이 작고 국면이 적다.</b> 공통 기간은 {ev.loc[f'{Ws[-1]}년', col_n]:,}행, 급락 {ev.loc[f'{Ws[-1]}년', col_dn]}건이다. 17개 시도가 같은 달에 함께 움직이므로 유효 표본은 행 수보다 훨씬 작고, 개선폭의 유의성 검정은 하지 않았다.",
        f"<b>설명변수의 이득은 국면 전환기에 집중된다.</b> 평가를 {pv['start']}부터 모든 창에 적용했던 이전 실행(커밋 c0003b3)에서는 ET 5년 창 MAE {pv['et']:.3f} 이 단순 규칙 mom1 {pv['mom1']:.3f} 과 동률이었는데, 창이 덜 찬 2018~2020년이 섞인 결과다({pv['regimes']}). "
        f"지금 규칙에서는 5년 창이 {st[W5]}부터만 평가되므로 ET {g.at['ET', 'MAE']:.3f} vs mom1 {g.at['Mom1(h×past1)', 'MAE']:.3f} 으로 차이가 커 보인다. "
        "반면 2020 급등기를 평가에 포함하는 짧은 창에서는 mom1 이 ET 보다 낫다(" + ", ".join(
            f"{W}년 ET {summ[(summ['h'] == h) & (summ['task'] == 'reg') & (summ['W'] == W) & (summ['model'] == 'ET')]['MAE'].iloc[0]:.3f} vs mom1 {summ[(summ['h'] == h) & (summ['task'] == 'reg') & (summ['W'] == W) & (summ['model'] == 'Mom1(h×past1)')]['MAE'].iloc[0]:.3f}"
            for W in Ws if W != W5) + "). 설명변수의 이득은 2021~23 전환기에 몰려 있고, 급등 초입처럼 모멘텀이 강한 시기에는 단순 규칙이 낫다는 뜻이다(⑥).",
        f"<b>이전 실험과의 비교.</b> 9차(KS)·TH_v19 는 {pv['start']}~ 평가라 2년 창 결과({st[Ws[0]]}~)와만 기간이 비슷하다. 달마다 독립 학습이라 겹치는 달의 예측은 이전 실행과 동일하고, 바뀐 것은 창이 안 찬 달을 평가에서 뺀 것뿐이다(학습창·자료·모형 불변).",
    ]
    return ("<div class='caveat'><span class='t'>평가 기간 설계의 한계 (각 창을 처음 꽉 차는 달부터 평가)</span><ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul></div>")


def _pick(g, task, model, W):
    """요약 표에서 (과제, 모형, 창) 한 행"""
    x = g[(g["task"] == task) & (g["model"] == model) & (g["W"] == W)]
    return x.iloc[0] if len(x) else None


def summary_box(summ, summ_c, d, starts, extra=""):
    """과제별 최선 모형·창은 공통 기간(summ_c, 모든 창 유효)에서 고르고, 그 창의 전체 유효 기간(summ) 값을 보인다 (카드 맨 위: 숫자 타일 + 핵심 메시지 + 한계 상자(extra) + 표)"""
    rows, msgs, main = [], [], None
    for h in sorted(summ["h"].unique(), reverse=True):
        bc = _bests(summ_c[summ_c["h"] == h])                                   # 공통 기간에서 최선(모형, 창)
        g = summ[summ["h"] == h]                                                 # 창별 전체 유효 기간
        W = int(bc["r"]["W"])
        r = _pick(g, "reg", bc["r"]["model"], W)
        ar, m1 = float(_pick(g, "reg", "AR(Ridge)", W)["MAE"]), float(_pick(g, "reg", "Mom1(h×past1)", W)["MAE"])
        gain, dm = 100 * (ar - r["MAE"]) / ar, 100 * (m1 - r["MAE"]) / m1
        rows.append({"예측": f"{h}개월 뒤 변화율(회귀)", "지표": "MAE(%p, 낮을수록 좋음)", "최선 모형(창은 공통 기간에서 선택)": f"{r['model']} (학습 {W}년)", "평가 기간": f"{starts[h][W]}~ ({int(r['n'] // 17)}개월)",
                     "최선 값": f"{r['MAE']:.3f}", "베이스라인 AR(같은 창·기간)": f"{ar:.3f}", "개선": f"오차 {abs(gain):.0f}% {'감소' if gain >= 0 else '증가'}",
                     "단순 규칙 mom1": f"{m1:.3f} (동률)" if abs(dm) < 1 else f"{m1:.3f} (최선 모형이 {abs(dm):.0f}% {'낫다' if dm > 0 else '못하다'})",
                     "공통 기간 값": f"{bc['r']['MAE']:.3f} vs AR {bc['ar']:.3f}"})
        ev_best = {}
        for ev, nm in (("up", "급등"), ("dn", "급락")):
            cc, _, _ = bc["ev"][ev]
            Wc = int(cc["W"])
            c, bb = _pick(g, ev, cc["model"], Wc), _pick(g, ev, "AR(Logit)", Wc)
            rate = float(c["양성비율"])
            ev_best[ev] = (c, bb, rate)
            rows.append({"예측": f"{h}개월 뒤 {nm}(분류)", "지표": "PR-AUC / BSS (높을수록 좋음)", "최선 모형(창은 공통 기간에서 선택)": f"{c['model']} (학습 {Wc}년)", "평가 기간": f"{starts[h][Wc]}~ ({int(c['n'] // 17)}개월)",
                         "최선 값": f"{c['PR_AUC']:.2f} / {c['BSS']:+.2f}", "베이스라인 AR(같은 창·기간)": f"{bb['PR_AUC']:.2f} / {bb['BSS']:+.2f}",
                         "개선": f"PR-AUC {c['PR_AUC'] - bb['PR_AUC']:+.2f}, BSS {c['BSS'] - bb['BSS']:+.2f}", "단순 규칙 mom1": f"(사건 비율 {rate:.0%})",
                         "공통 기간 값": f"{cc['PR_AUC']:.2f} / {cc['BSS']:+.2f}"})
        if h == max(summ["h"]):
            main = dict(h=h, r=r, ar=ar, m1=m1, gain=gain, dm=dm, ev=ev_best)
    # 숫자 타일 (주 예측 기간)
    h, r, ar, m1, gain, dm = main["h"], main["r"], main["ar"], main["m1"], main["gain"], main["dm"]
    up, upb, upr = main["ev"]["up"]; dn, dnb, dnr = main["ev"]["dn"]
    per = lambda x: f"{starts[h][int(x['W'])]}~ {int(x['n'] // 17)}개월"  # noqa: E731
    tiles = ("<div class='tiles'>"
             f"<div class='tile'><div class='lab'>{h}개월 뒤 변화율 예측 오차 (MAE) — 최선 {r['model']}, 학습 {r['W']}년 (평가 {per(r)})</div><div class='num'>{r['MAE']:.2f}<span style='font-size:14px'> %p</span></div>"
             f"<div class='sub'>베이스라인(같은 창) {ar:.2f} → <span class='good'>오차 {abs(gain):.0f}% 감소</span>" + (f" · 단순 규칙 mom1 {m1:.2f}({'동률' if abs(dm) < 1 else f'{dm:+.0f}%'})" if pd.notna(m1) else "") + "</div></div>"
             f"<div class='tile'><div class='lab'>{h}개월 뒤 급등(+1% 이상) 예측 — 최선 {up['model']}, 학습 {up['W']}년 (평가 {per(up)})</div><div class='num'>PR-AUC {up['PR_AUC']:.2f}</div>"
             f"<div class='sub'>베이스라인 {upb['PR_AUC']:.2f} → <span class='good'>{up['PR_AUC'] - upb['PR_AUC']:+.2f}</span> · BSS {up['BSS']:+.2f} (베이스라인 {upb['BSS']:+.2f}) · 사건 비율 {upr:.0%}</div></div>"
             f"<div class='tile'><div class='lab'>{h}개월 뒤 급락(−1% 이하) 예측 — 최선 {dn['model']}, 학습 {dn['W']}년 (평가 {per(dn)})</div><div class='num'>PR-AUC {dn['PR_AUC']:.2f}</div>"
             f"<div class='sub'>베이스라인 {dnb['PR_AUC']:.2f} → <span class='good'>{dn['PR_AUC'] - dnb['PR_AUC']:+.2f}</span> · BSS {dn['BSS']:+.2f} (베이스라인 {dnb['BSS']:+.2f}) · 사건 비율 {dnr:.0%}</div></div>"
             "</div>")
    if pd.isna(dm):
        mom_txt = ""
    elif abs(dm) >= 3:
        mom_txt = f"학습 없는 단순 규칙 mom1(최근 1개월 변화 × {h})보다도 {abs(dm):.0f}% {'낫다' if dm > 0 else '못하다'}({r['MAE']:.3f} vs {m1:.3f})."
    else:
        mom_txt = f"다만 학습 없는 단순 규칙 mom1(최근 1개월 변화 × {h})과는 비슷하다({r['MAE']:.3f} vs {m1:.3f})."
    msgs.append(f"<b>{h}개월 뒤 변화율</b>은 {r['model']}(학습 {r['W']}년)이 가장 좋았고, 같은 창에서 월세지수의 과거 흐름만 쓴 베이스라인(AR)보다 오차가 {abs(gain):.0f}% 작다. " + mom_txt)
    msgs.append(f"<b>급등·급락 예측</b>에서는 설명변수의 효과가 뚜렷하다. 급등은 {up['model']}(PR-AUC {up['PR_AUC']:.2f} vs 베이스라인 {upb['PR_AUC']:.2f}), 급락은 {dn['model']}(PR-AUC {dn['PR_AUC']:.2f} vs {dnb['PR_AUC']:.2f}). "
                f"아무 정보가 없을 때의 PR-AUC 는 사건 비율(급등 {upr:.0%}, 급락 {dnr:.0%})이다.")
    msgs.append("학습 기간은 4~5년이 안정적이고, 2년 창은 급락 예측이 무너진다(2020~21년에 급락이 없어 짧은 창에는 급락 사례가 부족). "
                "모든 변수를 넣은 선형 모형(Ridge·Logit)은 베이스라인보다 못해(과적합) 트리 모형만 쓸 만하고, 하이퍼파라미터 튜닝의 효과는 작고 모형마다 방향이 엇갈린다(⑤).")
    cs = max(starts[h].values())
    msgs.append(f"<b>평가 기간</b>: 각 학습창은 창이 처음 꽉 차는 달부터 평가했다({', '.join(f'{W}년 {starts[h][W]}~' for W in sorted(starts[h]))}; h={h}). 그 전 달은 그 창으로 학습·평가하지 않았다. "
                f"창끼리 비교(최선 창 선택)는 네 창이 모두 유효한 공통 기간 {cs}~ 로 했고(③ 둘째 표), 위 수치는 선택된 창의 전체 유효 기간 값이다. 설계의 한계는 아래 상자.")
    cv, cvv = os.path.join(d, "contrib_driver.csv"), os.path.join(d, "contrib_var.csv")
    if os.path.exists(cv) and os.path.exists(cvv):
        drv, var = pd.read_csv(cv, encoding="utf-8-sig"), pd.read_csv(cvv, encoding="utf-8-sig")
    else:
        drv = pd.DataFrame()
    if len(drv):
        h0 = max(drv["h"])
        d6 = drv[(drv["h"] == h0) & (drv["task"] == "reg")].sort_values("share", ascending=False)
        v6 = var[(var["h"] == h0) & (var["task"] == "reg")].sort_values("rank").head(3)
        msgs.append(f"<b>예측에 가장 크게 기여한 것</b>({h0}개월 뒤 변화율, SHAP 기준): 동인은 {d6.iloc[0]['동인']}({d6.iloc[0]['share']:.0%})·{d6.iloc[1]['동인']}({d6.iloc[1]['share']:.0%})·{d6.iloc[2]['동인']}({d6.iloc[2]['share']:.0%}), "
                    f"변수는 {', '.join(v6['변수'])} (⑧).")
    return ("<h2>한눈에 보기</h2>" + tiles + "<ul>" + "".join(f"<li>{m}</li>" for m in msgs) + "</ul>" + extra
            + "<p class='note'>과제별 최선 모형과 베이스라인(월세지수 과거 흐름만 쓴 AR) 대비 개선폭. 자세한 내용은 ③~⑨.</p>" + pd.DataFrame(rows).to_html(index=False))


# ------------------------------------------------------------ html
def build(d, tuned_dir=None, notes=None):
    st = json.load(open(os.path.join(d, "settings.json"), encoding="utf-8"))
    pred = load_pred(d)
    summ, roll = summarize(pred.copy())                                          # 창별 전체 유효 기간(주)
    hs = sorted(summ["h"].unique(), reverse=True)
    Ws = sorted(int(w) for w in summ["W"].unique())
    starts = {h: eval_starts(h, Ws) for h in hs}
    cs = {h: common_start(h, Ws) for h in hs}
    summ_c, _ = summarize(restrict_common(pred, Ws))                              # 공통 기간(창 비교용)
    for h in hs:
        for W in Ws:
            t0 = pred[(pred["h"] == h) & (pred["W"] == W)]["t"].min()
            if str(starts[h][W]) != str(t0):
                print(f"[주의] h={h} W={W}: 예측 파일의 첫 결정월 {t0} 이 규칙(first_full) {starts[h][W]} 과 다름")
    has_tuned = bool(tuned_dir) and os.path.exists(os.path.join(tuned_dir, "metrics_summary.csv")) and os.path.exists(os.path.join(tuned_dir, "predictions_h6.csv"))
    if bool(tuned_dir) and os.path.exists(os.path.join(tuned_dir, "metrics_summary.csv")) and not has_tuned:
        print("[주의] 튜닝 예측값 파일(predictions_h*.csv)이 없어 ⑤ 튜닝 비교 절을 생략함")
    st_t = json.load(open(os.path.join(tuned_dir, "settings.json"), encoding="utf-8")) if has_tuned else None
    n_months = {h: {W: int(roll[(roll["h"] == h) & (roll["task"] == "reg") & (roll["W"] == W)]["t"].nunique()) for W in Ws} for h in hs}
    starts_txt = "; ".join(f"h={h}: " + ", ".join(f"{W}년 {starts[h][W]}" for W in Ws) for h in hs)
    n_txt = "; ".join(f"h={h}: " + ", ".join(f"{W}년 {n_months[h][W]}" for W in Ws) for h in hs)
    cs_txt = ", ".join(f"h={h} {cs[h]}~" for h in hs)
    caveat = caveat_box(summ, summ_c, starts, hs, st["event_thr"])
    parts = [summary_box(summ, summ_c, d, starts, extra=caveat), "<h2>① 설정</h2><p class='note'><b>이 카드가 답하려는 질문</b>:아파트 월세지수가 앞으로 3·6개월 동안 얼마나 변할지(회귀), 그리고 ±1% 넘게 급등·급락할지(분류)를 "
             "월세지수의 과거 흐름만으로 맞히는 것(베이스라인)보다 경제 설명변수 47개를 더하면 얼마나 더 잘 맞히는가. 학습 기간 길이(2~5년), 튜닝, 시기, 과적합, 변수 기여도까지 차례로 본다.</p>"
             "<h3>용어 정리</h3>" + glossary_table() + f"""<table class='kv'>
<tr><th>자료</th><td>analysis/output/모형입력표_10차_values.csv — 17개 시도 × 결정월(월말) 패널. 설명변수 {st['n_expl']}열 + 타깃 모멘텀 3열(past1·3·6) + 지역 더미 17열 = {len(st['features_full'])}개 특성</td></tr>
<tr><th>타깃</th><td>G_h = 100·[R(t+h−1)/R(t−1) − 1]: 결정월 t 에 아는 마지막 지수 R(t−1) 대비 앞으로 h개월 변화율(%). 회귀는 G_h, 분류는 급등(G_h ≥ +{st['event_thr']:.0f}%)·급락(G_h ≤ −{st['event_thr']:.0f}%) 이진 사건</td></tr>
<tr><th>집합 구성</th><td><b>평가(test)</b> = 결정월 t 의 17개 시도(한 달 17행). <b>학습(train)</b> = t 시점에 정답이 확정된 결정월 s ≤ t−h 중 직전 W년(12W개월 × 17 = {12 * 2 * 17:,}~{12 * 5 * 17:,}행).
t 와 학습 사이 h−1개월은 정답 미확정이라 비움(미래 정보 유출 차단). <b>학습창 W 는 창이 처음 꽉 차는 달 t_W = y0 + 12W − 1 + h 부터 평가</b>(y0 = 정답이 있는 첫 결정월 2015-07; {starts_txt}).
그 전 달은 그 창으로 학습·평가하지 않고 자료는 학습 행으로만 쓴다. t 를 거기서부터 매월 한 칸씩 옮겨 모든 평가 예측을 모아 지표 계산(평가 결정월 수 — {n_txt}). 설계의 한계는 맨 위 상자.
<b>검증(validation)</b>은 2차 튜닝 모형에만: 학습 블록 안을 시간 순 3겹(검증 3개월, 겹 사이 h−1개월 간격)으로 나눠 격자를 고르고 전체 학습 블록으로 재학습. 1차(고정 파라미터)는 검증 집합 없음(Ridge alpha 만 학습 블록 안 LOO)</td></tr>
<tr><th>롤링</th><td>학습창 W = {st['windows']}년 각각 전체 반복(민감도). 평가 기간이 창마다 다르므로 창끼리 비교는 네 창이 모두 유효한 공통 기간({cs_txt})으로 하고, 같은 창 안의 모형 비교(베이스라인 대비)는 기간이 같아 그대로 한다</td></tr>
<tr><th>모형</th><td>회귀: AR(Ridge) 베이스라인(모멘텀 3열만) / 참고 Zero·Mom(past_h) / Ridge·RF·ET·XGB(모든 특성). 분류: AR(Logit) 베이스라인 / Logit·RF·ET·XGB. 학습창 양성 {st['min_pos']}건 미만이면 분류기는 학습창 기준율 예측(표의 기준율예측비율)</td></tr>
<tr><th>지표</th><td>MAE(%p) / F1(임계 0.5·학습창 기준율) / PR-AUC(평균정밀도, 하한 = 양성 비율) / BSS = 1 − Brier ÷ Brier(기준율 예측). 기준율 예측 = 학습창의 사건 비율을 모든 시도에 같은 확률로 내는 것(표·그림에는 따로 표시하지 않음; BSS 0 = 기준율 예측과 같음, 음수 = 그보다 못함). 전체 평가 기간 값 + 이동 창 값(MAE 12개월, 분류 24개월)</td></tr>
<tr><th>1차 설정</th><td>{st['note']}. 트리 {('100' if st.get('quick') else '300')}개, RF·ET leaf 5·max_features 0.5, XGB depth 3·학습률 0.05·subsample 0.8, Logit C=1</td></tr>"""
             + (f"<tr><th>2차 튜닝</th><td>모형 {', '.join(st_t['tune'])}, 학습창 {st_t['windows']}년, 평가 결정월 간격 {st_t['step']}개월(시간 절약). 격자: " + "; ".join(f"{m} {g}" for m, g in st_t["grids"].items()) + "</td></tr>" if has_tuned else "")
             + "</table><h3>기준·모형 정의 (past_k = 100·[R(t−1)/R(t−1−k) − 1], 결정월 t 에 아는 타깃 자신의 최근 k개월 변화율)</h3>" + definitions_table()
             + "<h3>학습창별 평가 기간</h3>" + eval_design_fig(hs, Ws, {h: max(pd.PeriodIndex(roll[roll["h"] == h]["t"].unique(), freq="M")) for h in hs})
             + "<h3>한 시점의 학습·검증·평가 구성</h3>" + split_fig(h=hs[0], W=4)
             + "<p class='note'>위 그림은 한 시점(2022-06, 학습창 4년)의 예시다. 롤링이라 학습창은 '직전 W년'만 쓴다. 자료가 2015-07에 시작하므로 각 창은 처음 꽉 차는 달부터 평가한다(아래 표). "
               "결정월·예측 기간·학습창을 바꿔 가며 볼 수 있는 대화형 그림: <a href='학습구조_시각화_10차.html'>학습구조_시각화_10차.html</a></p>"
             + "<h3>집합의 실제 기간 (창별 첫 평가월과 대표 결정월)</h3>" + "".join(period_table(roll, h, Ws, tuple(st_t["windows"]) if has_tuned else ()) for h in hs)]
    # ② 사건 수 (창별 평가 기간 + 연도별)
    fb = summ[(summ["task"] != "reg") & (summ["model"] == "RF")].groupby(["h", "W", "task"])["기준율예측비율"].first().unstack("task")
    parts.append("<h2>② 급등·급락은 얼마나 자주 일어났나 (평가 기간의 사건 수)</h2><p class='note'>창마다 평가 기간이 다르므로 창별로 센다. 급등·급락 비율은 아무 정보가 없을 때의 PR-AUC(하한)</p>")
    for h in hs:
        _, yearly = event_tables(VALUES, str(starts[h][Ws[0]]), [h], st["event_thr"])
        parts.append(f"<h3>{h}개월 뒤 예측</h3>" + event_tables_by_window(VALUES, h, starts[h], st["event_thr"]).to_html(index=False)
                     + f"<p class='note'>연도별(결정월 기준, 가장 이른 창인 {Ws[0]}년 창의 평가 기간 {starts[h][Ws[0]]}~) 행 수와 사건 수 — 급락은 2018~19 와 2022~23 에 몰려 있고 2020~21 에는 거의 없어 짧은 학습창은 급락 양성이 부족한 달이 생김</p>" + yearly.to_html())
    parts.append("<p class='note'>학습창에 양성이 부족해 분류기가 기준율을 예측한 달의 비율(행 기준)</p>" + fb.rename(columns={"up": "급등", "dn": "급락"}).to_html(float_format=lambda x: f"{x:.1%}"))
    # ③ 민감도: 창별 전체 유효 기간 + 창 비교용 공통 기간
    parts.append("<h2>③ 학습 기간(2~5년)에 따라 성능이 어떻게 달라지나 (1차, 고정 파라미터)</h2>"
                 "<p class='note'><b>첫째 표·그림</b>은 각 창을 자기 전체 유효 기간(창이 처음 꽉 차는 달~)으로 평가한 값 — 창마다 기간이 달라 같은 창 안에서 모형끼리 비교할 때 쓴다. "
                 "<b>둘째 표·그림</b>은 네 창이 모두 유효한 공통 기간만으로 다시 계산한 값 — 창 길이끼리 비교하고 최선 창을 고를 때 쓴다.</p>")
    for h in hs:
        parts.append(f"<h3>{h}개월 뒤 예측 — 창별 전체 유효 기간 (" + ", ".join(f"{W}년 {starts[h][W]}~" for W in Ws) + ")</h3>" + sensitivity_figs(summ, h, " (창별 전체 유효 기간)") + tbl(sensitivity_table(summ, h)))
    for h in hs:
        parts.append(f"<h3>{h}개월 뒤 예측 — 창 비교용 공통 기간 {cs[h]}~ ({n_months[h][Ws[-1]]}개월)</h3>" + sensitivity_figs(summ_c, h, f" (공통 기간 {cs[h]}~)") + tbl(sensitivity_table(summ_c, h)))
    parts.append("<p class='note'>공통 기간에서 과제별 최선 모형·창 (맨 위 요약의 선택 근거)</p>" + best_table(summ_c).to_html(index=False))
    # ④ 개선
    parts.append("<h2>④ 설명변수를 넣으면 얼마나 나아지나 (베이스라인 대비, 1차)</h2><p class='note'>창별 전체 유효 기간에서 같은 창의 베이스라인과 비교. 회귀는 MAE가 몇 % 줄었는지, 분류는 PR-AUC·BSS가 얼마나 올랐는지(모형 − 베이스라인)</p>"
                 + improvement_table(summ).to_html(index=False))
    # ⑤ 튜닝
    if has_tuned:
        summ_t, roll_t = tuned_comparison(d, tuned_dir, st_t)
        n_c = {W: int(roll_t[(roll_t["h"] == hs[0]) & (roll_t["task"] == "reg") & (roll_t["W"] == W)]["t"].nunique()) for W in sorted(roll_t["W"].unique())}
        parts.append(f"<h2>⑤ 하이퍼파라미터 튜닝은 효과가 있었나 (2차, 같은 결정월 {st_t['step']}개월 간격 표본, 학습창 {st_t['windows']}년)</h2>"
                     f"<p class='note'>각 창의 유효 기간 안에서 {st_t['step']}개월 간격으로 튜닝한 표본(h={hs[0]}: " + ", ".join(f"{W}년 {n}개 결정월" for W, n in n_c.items()) + ")에서 고정·튜닝 모형을 같은 행으로 비교</p>"
                     + (f"<p class='note'><b>주의</b>: {TUNED_NOTE}</p>" if TUNED_NOTE else ""))
        for h in sorted(summ_t["h"].unique(), reverse=True):
            parts.append(f"<h3>{h}개월 뒤 예측</h3>" + sensitivity_figs(summ_t, h, " (튜닝 표본)") + tbl(sensitivity_table(summ_t, h)))
        parts.append("<h3>베이스라인 대비(튜닝 포함, 같은 표본, 창별)</h3>" + improvement_table(summ_t).to_html(index=False))
        parts.append("<h3>선택된 하이퍼파라미터</h3><p class='note'>롤링이라 결정월마다 학습창 안 CV 로 다시 고름 → 가장 자주 뽑힌 값(빈도)과 마지막 결정월의 값을 제시. CV 점수 = 회귀 −MAE, 분류 평균정밀도</p>"
                     + tuned_params_table(tuned_dir).to_html(index=False))
        for h in sorted(roll_t["h"].unique(), reverse=True):
            parts.append(rolling_figs(roll_t, h, reg_win=4, clf_win=8, title_extra=f" — 튜닝 비교(표본 {st_t['step']}개월 간격이라 이동 창 = 4·8개 평가점)"))
    # ⑥ 롤링
    parts.append("<h2>⑥ 시기별로 성능이 어떻게 변했나 (1차)</h2><p class='note'>각 점은 그 달까지의 최근 성능(MAE는 6개월, 분류는 12개월 이동 창). 창별 행은 그 창이 처음 꽉 차는 달부터 시작하므로 2년 창만 2018~19 급락기를 보여 준다. "
                 "2018~19 급락기, 2020~21 급등기, 2022~23 하락기, 2024~26 회복·재상승기에서 모형별 차이를 본다</p>")
    for h in hs:
        parts.append(rolling_figs(roll, h, reg_win=6, clf_win=12))
    # ⑦ 과적합 진단
    parts.append("<h2>⑦ 학습 자료에만 맞춘 것은 아닌가 (과적합 점검: 적합 성능 vs 평가 성능)</h2>" + overfit_section(d, summ, tuned_dir if has_tuned else None, summ_t if has_tuned else None, st_t))
    # ⑧ 변수·동인 기여도 (contrib_analysis.py 가 만든 본문)
    contrib = os.path.join(d, "기여도_section.html")
    if os.path.exists(contrib):
        parts.append("<h2>⑧ 어떤 변수·동인이 예측에 기여했나 (SHAP·순열 중요도)</h2>" + open(contrib, encoding="utf-8").read())
    # ⑨ 분석
    if notes and os.path.exists(notes):
        parts.append("<h2>⑨ 결과 해석 — 전반적 평가·한계·보완할 점</h2>" + open(notes, encoding="utf-8").read())
    parts.append(f"""<h2>⑩ 읽을 때 주의할 점</h2><ul>
<li>각 학습창은 창이 처음 꽉 차는 달부터 평가했다({starts_txt}). 창마다 평가 기간이 다르므로 창끼리의 절대 수치 비교는 ③ 둘째 표(공통 기간)로만 하고, 설계의 한계(긴 창은 2018~19 급락기 미평가, 표본 적음 등)는 맨 위 상자에 적었다.</li>
<li>베이스라인 AR 은 월세지수 자신의 과거 1·3·6개월 변화율만 쓴다. 비교 모형이 이보다 좋아야 설명변수에 정보가 있다는 뜻이다. 학습 없는 단순 규칙(mom1)도 같이 보라 — 6개월 예측에서는 AR 보다 mom1 이 더 강한 기준이다.</li>
<li>분류 BSS 의 기준은 학습창 기준율이라 0 이면 기준율과 같고, 음수면 기준율보다 못하다(과신). PR-AUC 는 임계값과 무관한 순위 성능이며 양성 비율(② 사건 수 표)이 하한.</li>
<li>하이퍼파라미터: 1차 고정, 2차는 RF·ET·XGB 만 학습창 안 중첩 시계열 CV. 창마다 평가 기간이 같으므로 창끼리 직접 비교 가능. 튜닝 비교는 같은 결정월 표본으로 다시 계산.</li></ul>""")
    subtitle = (f"생성 {dt.date.today().isoformat()} · 평가 결정월: 각 학습창이 처음 꽉 차는 달부터(h={hs[0]}: " + ", ".join(f"{W}년 {starts[hs[0]][W]}" for W in Ws) + f") · 학습 자료 analysis/output/모형입력표_10차_values.csv · 결과 {os.path.relpath(d, BASE).replace(os.sep, '/')}"
                + (f" · 튜닝 {os.path.relpath(tuned_dir, BASE).replace(os.sep, '/')}" if has_tuned else ""))
    html = page("10차 월세 변화율·급등급락 예측 모형 — 결과 카드", subtitle, "".join(parts))
    out = os.path.join(d, "모형결과카드_10차.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{os.path.relpath(out, BASE)} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(BASE, "analysis", "output", "모형결과_10차"))
    ap.add_argument("--tuned-dir", default=os.path.join(BASE, "analysis", "output", "모형결과_10차_tuned"))
    ap.add_argument("--notes", default=os.path.join(BASE, "analysis", "output", "모형결과_10차", "분석_notes.html"))
    a = ap.parse_args()
    build(a.dir, a.tuned_dir, a.notes)
