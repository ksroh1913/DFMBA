# -*- coding: utf-8 -*-
"""
롤링 모형 결과 카드 — rolling_models.py 의 1차(고정 파라미터) 결과와 2차(튜닝) 결과를 한 html 로.
① 설정(학습·검증·시험 집합 구성 그림 포함)  ② 사건 수  ③ 학습창 민감도  ④ 베이스라인 대비 개선  ⑤ 튜닝 전후 비교(같은 결정월에서)와 선택된 파라미터
⑥ 시간에 따른 롤링 성능  ⑦ 결과 분석(전반적 평가·한계·보완, --notes 파일)  ⑧ 읽는 법
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
from rolling_models import summarize  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALUES = os.path.join(BASE, "analysis", "output", "모형입력표_10차_values.csv")
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#4a3aa7")
MODEL_COLOR = {"AR(Ridge)": C["ink"], "AR(Logit)": C["ink"], "Ridge": C["blue"], "Logit": C["blue"], "RF": C["orange"], "ET": C["aqua"], "XGB": C["violet"],
               "Zero": C["muted"], "Mom(past_h)": C["muted"], "Mom1(h×past1)": C["ink2"], "Clim": C["muted"]}
MODEL_STYLE = {"AR(Ridge)": "--", "AR(Logit)": "--", "Zero": ":", "Mom(past_h)": "-.", "Mom1(h×past1)": (0, (1, 1)), "Clim": ":"}
ORDER = ["AR(Ridge)", "AR(Logit)", "Mom1(h×past1)", "Ridge", "Logit", "RF", "ET", "XGB", "RF(튜닝)", "ET(튜닝)", "XGB(튜닝)", "Mom(past_h)", "Zero", "Clim"]


def add_mom1(pred):
    """예측 파일에 9차·TH 와 같은 단순 기준 mom1 = h × past1 (최근 1개월 변화의 h개월 연장) 참고 행을 더한다 (없을 때만)"""
    if "Mom1(h×past1)" in set(pred["model"]):
        return pred
    v = pd.read_csv(VALUES, encoding="utf-8-sig")[["region", "결정월", "past1"]]
    base = pred[pred["model"] == "Zero"].merge(v, left_on=["region", "t"], right_on=["region", "결정월"], how="left")
    base["model"] = "Mom1(h×past1)"
    base["pred"] = base["h"] * base["past1"]
    return pd.concat([pred, base.drop(columns=["결정월", "past1"])], ignore_index=True)


def load_results(d):
    """예측 파일이 있으면 mom1 을 더해 지표를 다시 계산, 없으면 저장된 요약을 쓴다"""
    files = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("predictions_h") and f.endswith(".csv")]
    if files:
        pred = add_mom1(pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True))
        return summarize(pred)
    return (pd.read_csv(os.path.join(d, "metrics_summary.csv"), encoding="utf-8-sig"), pd.read_csv(os.path.join(d, "rolling_metrics.csv"), encoding="utf-8-sig"))
TASK_KR = {"reg": "회귀(G_h, %)", "up": "급등(G_h ≥ +1%)", "dn": "급락(G_h ≤ −1%)"}
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


# ------------------------------------------------------------ ① 설정: 기준·모형 정의 표
def definitions_table():
    rows = [
        ("Zero", "참고 기준", "Ĝ_h = 0 (변화 없음)", "학습 없음", "zero"),
        ("Mom1(h×past1)", "참고 기준", "Ĝ_h = h × past1 (최근 1개월 변화율을 h개월 연장)", "학습 없음", "mom1 — 9차 주 분석의 기준(M0), TH_v19 의 mom1 과 같음"),
        ("Mom(past_h)", "참고 기준", "Ĝ_h = past_h (최근 h개월 변화율이 그대로 이어진다)", "학습 없음", "lasth"),
        ("AR(Ridge)", "베이스라인(회귀)", "Ĝ_h = b0 + b1·past1 + b3·past3 + b6·past6", "학습창(직전 W년, 17개 시도 풀링)에서 매월 Ridge 로 추정. alpha 는 학습창 안 LOO 로 0.01~1000 중 선택. 설명변수·지역 더미 없음", "없음(9차 A 정보군은 past1·3·6 에 V002·V026 6개월 변화를 더한 것)"),
        ("AR(Logit)", "베이스라인(분류)", "P(사건) = σ(b0 + b1·past1 + b3·past3 + b6·past6)", "학습창에서 매월 로지스틱(C=1) 추정", "없음"),
        ("Clim", "참고 기준(분류)", "P(사건) = 학습창의 사건 비율", "학습 없음", "BSS 의 기준(9차도 학습 사건비율 기준)"),
        ("Ridge / Logit", "비교 모형", "모멘텀 3열 + 설명변수 47열 + 지역 더미 17열의 선형 모형", "Ridge: alpha LOO 선택 / Logit: C=1", "9차 B·C 정보군에 해당(단, 9차는 동인 지수로 압축)"),
        ("RF / ET / XGB", "비교 모형", "같은 특성의 트리 모형(랜덤포레스트·엑스트라트리·XGBoost)", "1차 고정, 2차 '(튜닝)' 은 학습창 안 중첩 시계열 CV 로 격자 선택", "TH_v19 의 RF·ET·XGB 와 같은 계열(설정값은 다름)"),
    ]
    return pd.DataFrame(rows, columns=["카드 이름", "구분", "예측식", "학습·파라미터", "9차·TH 대응"]).to_html(index=False)


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
    ax.text(ts(t + 1), 3.0, f"시험(test): 결정월 t={t} 의 17개 시도 (정답은 {t + h - 1} 지수 공표 뒤 확정)", fontsize=8.5, color=C["ink"], va="center")
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
    ax.set_title(f"예시: 지평 h={h}개월, 학습창 W={W}년, 결정월 t={t} 의 집합 구성 (매월 t 를 한 칸씩 옮기며 반복)", loc="left")
    return img(fig)


# ------------------------------------------------------------ ② 사건 수
def event_tables(values_path, start, horizons, thr):
    df = pd.read_csv(values_path, encoding="utf-8-sig")
    a = df[df["결정월"] >= start]
    rows, yearly = [], []
    for h in horizons:
        s = a[f"G{h}"].dropna()
        up, dn = (s >= thr), (s <= -thr)
        rows.append({"지평": f"{h}개월", "평가 행(시도×결정월)": len(s), "결정월 범위": f"{a.loc[s.index, '결정월'].min()}~{a.loc[s.index, '결정월'].max()}",
                     f"급등(≥+{thr:.0f}%) 건수": int(up.sum()), "급등 비율": f"{up.mean():.1%}", f"급락(≤−{thr:.0f}%) 건수": int(dn.sum()), "급락 비율": f"{dn.mean():.1%}",
                     "둘 중 하나": f"{(up | dn).mean():.1%}"})
        g = a.loc[s.index].assign(연도=a.loc[s.index, "결정월"].str[:4], up=up.values, dn=dn.values).groupby("연도").agg(행=("up", "size"), 급등=("up", "sum"), 급락=("dn", "sum"))
        g.columns = [f"h{h} {c}" for c in g.columns]
        yearly.append(g)
    return pd.DataFrame(rows), pd.concat(yearly, axis=1)


# ------------------------------------------------------------ ③ 민감도
def sensitivity_figs(summ, h):
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
    fig.suptitle(f"지평 h={h}개월 — 학습창 길이별 전체 평가 기간 성능", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
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
    """튜닝 실행이 평가한 결정월(간격 step)만 골라 고정·튜닝 모형을 같은 표본에서 비교"""
    preds = []
    for h in st_t["horizons"]:
        pt = pd.read_csv(os.path.join(tuned_dir, f"predictions_h{h}.csv"), encoding="utf-8-sig")
        pt = pt[pt["model"].isin(st_t["tune"])].copy()
        pt["model"] = pt["model"] + "(튜닝)"
        pf = pd.read_csv(os.path.join(fixed_dir, f"predictions_h{h}.csv"), encoding="utf-8-sig")
        pf = pf[pf["t"].isin(set(pt["t"])) & pf["W"].isin(set(pt["W"]))]
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
def rolling_figs(roll, h, reg_win=12, clf_win=24, title_extra=""):
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
            if i == 0:
                ax.set_title(title, loc="left")
            if j == 0:
                ax.set_ylabel(f"학습창 {W}년", fontsize=9, color=C["ink"])
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"지평 h={h}개월 — 롤링 평가 성능의 시간 변화{title_extra} (가로축 = 결정월, 이동 창 끝 기준)", x=0.01, y=1.01, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    shared_legend(fig, axes, right=0.92)
    return img(fig)


# ------------------------------------------------------------ html
def build(d, tuned_dir=None, notes=None):
    summ, roll = load_results(d)
    st = json.load(open(os.path.join(d, "settings.json"), encoding="utf-8"))
    hs = sorted(summ["h"].unique(), reverse=True)
    has_tuned = bool(tuned_dir) and os.path.exists(os.path.join(tuned_dir, "metrics_summary.csv"))
    st_t = json.load(open(os.path.join(tuned_dir, "settings.json"), encoding="utf-8")) if has_tuned else None
    n_months = {h: int(roll[(roll["h"] == h) & (roll["task"] == "reg")]["t"].nunique()) for h in hs}
    parts = [f"""<h2>① 설정</h2><table class='kv'>
<tr><th>자료</th><td>analysis/output/모형입력표_10차_values.csv — 17개 시도 × 결정월(월말) 패널. 설명변수 {st['n_expl']}열 + 타깃 모멘텀 3열(past1·3·6) + 지역 더미 17열 = {len(st['features_full'])}개 특성</td></tr>
<tr><th>타깃</th><td>G_h = 100·[R(t+h−1)/R(t−1) − 1]: 결정월 t 에 아는 마지막 지수 R(t−1) 대비 앞으로 h개월 변화율(%). 회귀는 G_h, 분류는 급등(G_h ≥ +{st['event_thr']:.0f}%)·급락(G_h ≤ −{st['event_thr']:.0f}%) 이진 사건</td></tr>
<tr><th>집합 구성</th><td><b>시험(test)</b> = 결정월 t 의 17개 시도(한 달 17행). <b>학습(train)</b> = t 시점에 정답이 확정된 결정월 s ≤ t−h 중 직전 W년(12W개월 × 17 = {12 * 2 * 17:,}~{12 * 5 * 17:,}행).
t 와 학습 사이 h−1개월은 정답 미확정이라 비움(미래 정보 유출 차단). t 를 {st['start']}부터 매월 한 칸씩 옮겨 반복(평가 결정월 {', '.join(f'h={h}: {n}개월' for h, n in n_months.items())}) → 모든 시험 예측을 모아 지표 계산.
<b>검증(validation)</b>은 2차 튜닝 모형에만: 학습 블록 안을 시간 순 3겹(검증 3개월, 겹 사이 h−1개월 간격)으로 나눠 격자를 고르고 전체 학습 블록으로 재학습. 1차(고정 파라미터)는 검증 집합 없음(Ridge alpha 만 학습 블록 안 LOO)</td></tr>
<tr><th>롤링</th><td>학습창 W = {st['windows']}년 각각 전체 반복(민감도). 평가 기간은 창과 무관하게 같음 → 창끼리 직접 비교 가능</td></tr>
<tr><th>모형</th><td>회귀: AR(Ridge) 베이스라인(모멘텀 3열만) / 참고 Zero·Mom(past_h) / Ridge·RF·ET·XGB(모든 특성). 분류: AR(Logit) 베이스라인 / Logit·RF·ET·XGB. 학습창 양성 {st['min_pos']}건 미만이면 분류기는 학습창 기준율 예측(표의 기준율예측비율)</td></tr>
<tr><th>지표</th><td>MAE(%p) / F1(임계 0.5·학습창 기준율) / PR-AUC(평균정밀도, 하한 = 양성 비율) / BSS = 1 − Brier ÷ Brier(기준율). 전체 평가 기간 값 + 이동 창 값(MAE 12개월, 분류 24개월)</td></tr>
<tr><th>1차 설정</th><td>{st['note']}. 트리 {('100' if st.get('quick') else '300')}개, RF·ET leaf 5·max_features 0.5, XGB depth 3·학습률 0.05·subsample 0.8, Logit C=1</td></tr>"""
             + (f"<tr><th>2차 튜닝</th><td>모형 {', '.join(st_t['tune'])}, 학습창 {st_t['windows']}년, 평가 결정월 간격 {st_t['step']}개월(시간 절약). 격자: " + "; ".join(f"{m} {g}" for m, g in st_t["grids"].items()) + "</td></tr>" if has_tuned else "")
             + "</table><h3>기준·모형 정의 (past_k = 100·[R(t−1)/R(t−1−k) − 1], 결정월 t 에 아는 타깃 자신의 최근 k개월 변화율)</h3>" + definitions_table() + split_fig(h=hs[0], W=4)]
    # ② 사건 수
    ev, yearly = event_tables(VALUES, st["start"], hs, st["event_thr"])
    fb = summ[(summ["task"] != "reg") & (summ["model"] == "RF")].groupby(["h", "W", "task"])["기준율예측비율"].first().unstack("task")
    parts.append("<h2>② 평가 기간의 사건 수</h2>" + ev.to_html(index=False) + "<p class='note'>연도별(결정월 기준) 행 수와 사건 수 — 사건이 특정 연도에 몰려 있어 짧은 학습창은 양성 없는 달이 생김</p>" + yearly.to_html()
                 + "<p class='note'>학습창에 양성이 부족해 분류기가 기준율을 예측한 달의 비율(행 기준)</p>" + fb.rename(columns={"up": "급등", "dn": "급락"}).to_html(float_format=lambda x: f"{x:.1%}"))
    # ③ 민감도
    parts.append("<h2>③ 학습창 민감도 (1차 고정 파라미터, 전체 평가 기간)</h2>")
    for h in hs:
        parts.append(f"<h3>지평 h={h}개월</h3>" + sensitivity_figs(summ, h) + tbl(sensitivity_table(summ, h)))
    # ④ 개선
    parts.append("<h2>④ 베이스라인(AR) 대비 개선 (1차)</h2><p class='note'>회귀는 MAE 감소율(%), 분류는 PR-AUC·BSS 차이(모형 − AR)</p>" + best_table(summ).to_html(index=False)
                 + "<details><summary>학습창·모형별 전체 표</summary>" + improvement_table(summ).to_html(index=False) + "</details>")
    # ⑤ 튜닝
    if has_tuned:
        summ_c, roll_c = tuned_comparison(d, tuned_dir, st_t)
        parts.append(f"<h2>⑤ 2차 튜닝 전후 비교 (같은 결정월 {st_t['step']}개월 간격 표본, 학습창 {st_t['windows']}년)</h2>")
        for h in sorted(summ_c["h"].unique(), reverse=True):
            parts.append(f"<h3>지평 h={h}개월</h3>" + sensitivity_figs(summ_c, h) + tbl(sensitivity_table(summ_c, h)))
        parts.append("<h3>베이스라인 대비(튜닝 포함, 같은 표본)</h3>" + best_table(summ_c).to_html(index=False)
                     + "<details><summary>학습창·모형별 전체 표</summary>" + improvement_table(summ_c).to_html(index=False) + "</details>")
        parts.append("<h3>선택된 하이퍼파라미터</h3><p class='note'>롤링이라 결정월마다 학습창 안 CV 로 다시 고름 → 가장 자주 뽑힌 값(빈도)과 마지막 결정월의 값을 제시. CV 점수 = 회귀 −MAE, 분류 평균정밀도</p>"
                     + tuned_params_table(tuned_dir).to_html(index=False))
        for h in sorted(roll_c["h"].unique(), reverse=True):
            parts.append(rolling_figs(roll_c, h, reg_win=4, clf_win=8, title_extra=f" — 튜닝 비교(표본 {st_t['step']}개월 간격이라 이동 창 = 4·8개 평가점)"))
    # ⑥ 롤링
    parts.append("<h2>⑥ 시간에 따른 롤링 성능 (1차)</h2><p class='note'>각 점은 그 달까지의 이동 창(MAE 12개월, 분류 24개월) 성능</p>")
    for h in hs:
        parts.append(rolling_figs(roll, h))
    # ⑦ 분석
    if notes and os.path.exists(notes):
        parts.append("<h2>⑦ 결과 분석 — 전반적 평가·한계·보완사항</h2>" + open(notes, encoding="utf-8").read())
    parts.append("""<h2>⑧ 읽는 법·주의</h2><ul>
<li>베이스라인 AR 은 타깃 자신의 과거 1·3·6개월 변화만 쓴다. 설명변수 모형이 이보다 좋아야 변수에 정보가 있다는 뜻.</li>
<li>분류 BSS 의 기준은 학습창 기준율이라 0 이면 기준율과 같고, 음수면 기준율보다 못하다(과신). PR-AUC 는 임계값과 무관한 순위 성능이며 양성 비율(Clim 행)이 하한.</li>
<li>하이퍼파라미터: 1차 고정, 2차는 RF·ET·XGB 만 학습창 안 중첩 시계열 CV. 창마다 평가 기간이 같으므로 창끼리 직접 비교 가능. 튜닝 비교는 같은 결정월 표본으로 다시 계산.</li></ul>""")
    css = ("<style>body{font-family:'Malgun Gothic',system-ui,sans-serif;font-size:13px;color:#0b0b0b;background:#f9f9f7;margin:24px;max-width:1500px}"
           "h1{font-size:21px}h2{font-size:16px;border-bottom:1px solid #c3c2b7;margin-top:32px;padding-bottom:4px}h3{font-size:14px;color:#52514e;margin-top:20px}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}th,td{border:1px solid #e1e0d9;padding:3px 7px;text-align:right;vertical-align:top}th{background:#f0efec;text-align:left}"
           "td:first-child{text-align:left}table.kv th{width:90px}table.kv td{text-align:left}img{display:block;margin:6px 0 14px 0;max-width:100%}p.note{color:#52514e}details{margin:8px 0}"
           "div.analysis{background:#fcfcfb;border:1px solid #e1e0d9;padding:10px 16px}div.analysis h4{margin:14px 0 6px 0;font-size:13.5px}</style>")
    html = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>롤링 모형 결과 10차</title>{css}</head><body>"
            f"<h1>10차 롤링 모형 결과 — 베이스라인 대비 개선, 학습창 민감도, 튜닝, 시간에 따른 성능</h1><p class='note'>생성 {dt.date.today().isoformat()} · 1차 {os.path.relpath(d, BASE)}"
            + (f" · 2차 {os.path.relpath(tuned_dir, BASE)}" if has_tuned else "") + "</p>" + "".join(parts) + "</body></html>")
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
