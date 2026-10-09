# -*- coding: utf-8 -*-
"""
롤링 모형 결과 카드 — rolling_models.py 의 출력(metrics_summary.csv, rolling_metrics.csv, settings.json)을 한 html 로.
① 설정  ② 학습창(2~5년) 민감도: 지표 표 + 선 그림  ③ 베이스라인(AR) 대비 개선  ④ 시간에 따른 롤링 성능(12개월 이동 MAE, 24개월 이동 BSS·F1)  ⑤ 요약·주의
사용: PYTHONUTF8=1 python analysis/model_card.py [--dir analysis/output/모형결과_10차]
"""
import argparse
import base64
import datetime as dt
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
MODEL_COLOR = {"AR(Ridge)": C["ink"], "AR(Logit)": C["ink"], "Ridge": "#2a78d6", "Logit": "#2a78d6", "RF": "#eb6834", "ET": "#1baf7a", "XGB": "#4a3aa7",
               "Zero": C["muted"], "Mom(past_h)": C["muted"], "Clim": C["muted"]}
MODEL_STYLE = {"AR(Ridge)": "--", "AR(Logit)": "--", "Zero": ":", "Mom(past_h)": "-.", "Clim": ":"}
ORDER = ["AR(Ridge)", "AR(Logit)", "Ridge", "Logit", "RF", "ET", "XGB", "Mom(past_h)", "Zero", "Clim"]
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


def mline(ax, x, y, model, **kw):
    ax.plot(x, y, color=MODEL_COLOR.get(model, C["ink"]), ls=MODEL_STYLE.get(model, "-"), lw=2 if model in ("RF", "ET", "XGB", "Ridge", "Logit") else 1.6, label=model, **kw)


def shared_legend(fig, axes, right=0.9):
    """모든 패널의 모형을 모아 그림 오른쪽에 범례 하나 (같은 이름은 한 번만)"""
    handles, labels = {}, []
    for ax in np.ravel(axes):
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in handles:
                handles[l_] = h_
    order = [m for m in ORDER if m in handles]
    fig.legend([handles[m] for m in order], order, loc="center left", bbox_to_anchor=(right + 0.005, 0.5), frameon=False)
    fig.subplots_adjust(right=right)


# ------------------------------------------------------------ ② 민감도
def sensitivity_figs(summ, h):
    s = summ[summ["h"] == h]
    panels = [("reg", "MAE", "MAE (낮을수록 좋음)"), ("up", "PR_AUC", "급등 PR-AUC (높을수록)"), ("up", "BSS", "급등 BSS (높을수록)"),
              ("dn", "PR_AUC", "급락 PR-AUC"), ("dn", "BSS", "급락 BSS")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2))
    for ax, (task, met, title) in zip(axes, panels):
        g = s[s["task"] == task]
        for model in [m for m in ORDER if m in set(g["model"])]:
            gg = g[g["model"] == model].sort_values("W")
            if model == "Clim" and met == "BSS":
                continue
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


# ------------------------------------------------------------ ③ 개선
def improvement_table(summ):
    rows = []
    for (h, W), g in summ.groupby(["h", "W"]):
        reg = g[g["task"] == "reg"].set_index("model")
        base = reg.at["AR(Ridge)", "MAE"] if "AR(Ridge)" in reg.index else np.nan
        for m in ("Ridge", "RF", "ET", "XGB"):
            if m in reg.index:
                rows.append(dict(h=h, W=W, 과제="회귀", 모형=m, 지표="MAE", 베이스라인=round(base, 3), 모형값=round(reg.at[m, "MAE"], 3), 개선=f"{100 * (base - reg.at[m, 'MAE']) / base:+.1f}%"))
        for ev in ("up", "dn"):
            c = g[g["task"] == ev].set_index("model")
            if "AR(Logit)" not in c.index:
                continue
            for m in ("Logit", "RF", "ET", "XGB"):
                if m in c.index:
                    for met in ("PR_AUC", "BSS"):
                        b, v = c.at["AR(Logit)", met], c.at[m, met]
                        rows.append(dict(h=h, W=W, 과제=TASK_KR[ev][:2], 모형=m, 지표=met, 베이스라인=round(b, 3), 모형값=round(v, 3), 개선=f"{v - b:+.3f}"))
    return pd.DataFrame(rows)


def best_table(summ):
    rows = []
    for h, g in summ.groupby("h"):
        r = g[(g["task"] == "reg") & (~g["model"].isin(["Zero", "Mom(past_h)"]))].sort_values("MAE")
        rows.append(dict(h=h, 과제="회귀 MAE", 최선=f"{r.iloc[0]['model']} (W={r.iloc[0]['W']}년) {r.iloc[0]['MAE']:.3f}",
                         AR베이스라인=f"{g[(g['task'] == 'reg') & (g['model'] == 'AR(Ridge)')]['MAE'].min():.3f} (최선 창)"))
        for ev in ("up", "dn"):
            c = g[(g["task"] == ev) & (g["model"] != "Clim")].sort_values("PR_AUC", ascending=False)
            rows.append(dict(h=h, 과제=f"{TASK_KR[ev][:2]} PR-AUC", 최선=f"{c.iloc[0]['model']} (W={c.iloc[0]['W']}년) {c.iloc[0]['PR_AUC']:.3f} (BSS {c.iloc[0]['BSS']:+.3f})",
                             AR베이스라인=f"{g[(g['task'] == ev) & (g['model'] == 'AR(Logit)')]['PR_AUC'].max():.3f} (최선 창)"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ ④ 롤링 곡선
def rolling_figs(roll, h, reg_win=12, clf_win=24):
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
    fig.suptitle(f"지평 h={h}개월 — 롤링 평가 성능의 시간 변화 (가로축 = 결정월, 이동 창 끝 기준)", x=0.01, y=1.01, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    shared_legend(fig, axes, right=0.92)
    return img(fig)


# ------------------------------------------------------------ html
def build(d):
    summ = pd.read_csv(os.path.join(d, "metrics_summary.csv"), encoding="utf-8-sig")
    roll = pd.read_csv(os.path.join(d, "rolling_metrics.csv"), encoding="utf-8-sig")
    st = json.load(open(os.path.join(d, "settings.json"), encoding="utf-8"))
    hs = sorted(summ["h"].unique(), reverse=True)
    parts = [f"""<h2>① 설정</h2><table class='kv'>
<tr><th>자료</th><td>analysis/output/모형입력표_10차_values.csv — 17개 시도 × 결정월, 설명변수 {st['n_expl']}열 + 모멘텀 3열(past1·3·6) + 지역 더미</td></tr>
<tr><th>롤링</th><td>결정월 t 마다 정답이 확정된 행(결정월 s ≤ t−h)만으로 직전 W년을 학습 → t 의 17개 시도 예측. 매월 전진. 학습창 W = {st['windows']}년, 평가 {st['start']}~</td></tr>
<tr><th>회귀</th><td>y = G_h. 베이스라인 AR(Ridge) = 모멘텀만. 참고 Zero(0 예측)·Mom(past_h). 비교 {', '.join(m for m in st['models_reg'] if not m.startswith('AR'))} (모멘텀 + 설명변수 + 지역)</td></tr>
<tr><th>분류</th><td>급등 = G_h ≥ +{st['event_thr']:.0f}%, 급락 = G_h ≤ −{st['event_thr']:.0f}% (각각 이진). 베이스라인 AR(Logit) = 모멘텀만. 비교 {', '.join(m for m in st['models_clf'] if not m.startswith('AR'))}. 학습창 양성 {st['min_pos']}건 미만이면 모든 분류기가 학습창 기준율을 예측</td></tr>
<tr><th>지표</th><td>MAE / F1(임계 0.5 와 학습창 기준율) / PR-AUC(평균정밀도) / BSS = 1 − Brier ÷ Brier(기준율). 전체 평가 기간 값과 이동 창 값(MAE 12개월, 분류 24개월)</td></tr>
<tr><th>전처리·설정</th><td>{st['note']}. Ridge 의 alpha 는 학습창 안에서 자동 선택(RidgeCV). 트리 {('100' if st.get('quick') else '300')}개</td></tr></table>"""]
    parts.append("<h2>② 학습창 민감도 (전체 평가 기간)</h2>")
    for h in hs:
        parts.append(f"<h3>지평 h={h}개월</h3>" + sensitivity_figs(summ, h) + sensitivity_table(summ, h).to_html(float_format=lambda x: f"{x:.3f}", na_rep=""))
    parts.append("<h2>③ 베이스라인(AR) 대비 개선</h2><p class='note'>회귀는 MAE 감소율(%), 분류는 PR-AUC·BSS 의 차이(모형 − AR)</p>" + best_table(summ).to_html(index=False)
                 + "<details><summary>학습창·모형별 전체 표</summary>" + improvement_table(summ).to_html(index=False) + "</details>")
    parts.append("<h2>④ 시간에 따른 롤링 성능</h2><p class='note'>각 점은 그 달까지의 이동 창(MAE 12개월, 분류 24개월) 성능. 2020~21 급등기와 2022~23 급락기에서 모형별 차이가 드러나는지, 창이 길어질수록 안정되는지를 본다</p>")
    for h in hs:
        parts.append(rolling_figs(roll, h))
    parts.append("""<h2>⑤ 읽는 법·주의</h2><ul>
<li>베이스라인 AR 은 타깃 자신의 과거 1·3·6개월 변화만 쓴다. 설명변수 모형이 이보다 좋아야 변수에 정보가 있다는 뜻.</li>
<li>분류 BSS 의 기준은 학습창 기준율(기후값)이라 0 이면 기준율과 같고, 음수면 기준율보다 못하다(과신). PR-AUC 는 임계값과 무관한 순위 성능이며 양성 비율(Clim 행)이 하한.</li>
<li>사건이 연도별로 쏠려(2020~21 급등, 2018~19·2022 급락) 짧은 학습창은 양성이 없는 달이 생긴다 — 그 달은 기준율 예측으로 대체됐고 표의 '기준율예측비율'에 기록.</li>
<li>하이퍼파라미터는 고정(튜닝 없음). 학습창 선택은 이 민감도 표로 하되, 창마다 평가 기간이 같으므로 직접 비교 가능.</li></ul>""")
    css = ("<style>body{font-family:'Malgun Gothic',system-ui,sans-serif;font-size:13px;color:#0b0b0b;background:#f9f9f7;margin:24px;max-width:1500px}"
           "h1{font-size:21px}h2{font-size:16px;border-bottom:1px solid #c3c2b7;margin-top:32px;padding-bottom:4px}h3{font-size:14px;color:#52514e;margin-top:20px}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}th,td{border:1px solid #e1e0d9;padding:3px 7px;text-align:right;vertical-align:top}th{background:#f0efec;text-align:left}"
           "td:first-child{text-align:left}table.kv th{width:90px}table.kv td{text-align:left}img{display:block;margin:6px 0 14px 0;max-width:100%}p.note{color:#52514e}details{margin:8px 0}</style>")
    html = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>롤링 모형 결과 10차</title>{css}</head><body>"
            f"<h1>10차 롤링 모형 결과 — 베이스라인 대비 개선, 학습창 민감도, 시간에 따른 성능</h1><p class='note'>생성 {dt.date.today().isoformat()} · 결과 폴더 {os.path.relpath(d, BASE)}</p>"
            + "".join(parts) + "</body></html>")
    out = os.path.join(d, "모형결과카드_10차.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{os.path.relpath(out, BASE)} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(BASE, "analysis", "output", "모형결과_10차"))
    build(ap.parse_args().dir)
