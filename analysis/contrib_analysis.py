# -*- coding: utf-8 -*-
"""
변수·동인별 기여도 — 롤링 전진 평가(rolling_models.py 와 같은 학습창·특성)를 재현하면서 매 결정월 평가 행(17개 시도)의 기여도를 모은다.
  - XGB(회귀·급등·급락): XGBoost 내장 TreeSHAP(pred_contribs) → 평가 행마다 특성별 SHAP (회귀 %p, 분류 로그오즈). shap 패키지 불필요
  - ET(회귀)·RF(급등): 표본 외 순열 중요도 — 평가 행의 특성값을 학습창에서 무작위로 뽑은 값으로 바꿨을 때 오차 증가(3회 평균). 전국 공통 변수도 평가 가능
출력(analysis/output/모형결과_10차/): contrib_shap_long.csv, contrib_var.csv(변수별 요약), contrib_driver.csv, contrib_driver_time.csv, contrib_perm.csv, 기여도_section.html(카드 ⑧ 본문)
사용: PYTHONUTF8=1 python analysis/contrib_analysis.py [--horizons 6 3] [--W 4] [--step 1] [--from-csv] [--start 2021-01]
  --start: 평가 시작 하한(선택). 기본은 학습창 W 가 처음 꽉 차는 달(rolling_models.first_full: y0 + 12W − 1 + h)부터 — 롤링·카드와 같은 규칙
"""
import argparse
import base64
import io
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import mean_absolute_error, average_precision_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import variable_card as vc  # noqa: E402
from rolling_models import EVENT_THR, MIN_POS, load, models_for  # noqa: E402
from rolling_models import first_full  # noqa: E402  — 창이 처음 꽉 차는 달부터 평가(롤링·카드와 같은 규칙)

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "analysis", "output", "모형결과_10차")
DRIVER_NAME = {"①": "① 임차수요 압력", "②": "② 주택공급·재고 여건", "③": "③ 금융여건·상대가격", "④": "④ 임대시장 수급·전환 구조", "⑤": "⑤ 시장과열·단기 트리거", "⑥": "⑥ 거시경기·금융시장 여건",
               "모": "타깃 모멘텀(past1·3·6)", "지": "지역 더미"}
DRIVER_COLOR = {"①": "#2a78d6", "②": "#eb6834", "③": "#1baf7a", "④": "#eda100", "⑤": "#e87ba4", "⑥": "#4a3aa7", "모": "#0b0b0b", "지": "#898781"}
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", red="#e34948")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"], "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlecolor": C["ink"], "legend.fontsize": 8, "legend.frameon": False})
TASK_KR = {"reg": "회귀: 변화율(%p)", "up": "급등 확률(로그오즈)", "dn": "급락 확률(로그오즈)"}


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    return "<img src='data:image/png;base64," + base64.b64encode(buf.getvalue()).decode() + "'/>"


def driver_of(feature, log):
    if feature.startswith("past"):
        return "모"
    if feature.startswith("R_"):
        return "지"
    vid = feature.split("_")[0]
    d = log.loc[log["ID"] == vid, "동인"]
    return str(d.iloc[0])[0] if len(d) and str(d.iloc[0]) else "?"


def short_name(feature):
    if feature.startswith("past") or feature.startswith("R_"):
        return feature
    parts = feature.split("_")
    return f"{parts[0]} {parts[1][:12]} · {parts[-1][:14]}".replace("−", "-")      # 글꼴에 없는 수학 마이너스(−) → 하이픈


def two_line(feature):
    if feature.startswith("past") or feature.startswith("R_"):
        return feature
    parts = feature.split("_")
    return f"{parts[0]} {parts[1][:10]}\n{parts[-1][:16]}".replace("−", "-")


# ------------------------------------------------------------ 계산
def run(h, W, step, df, X, feats, log, start=None):
    y_all = df[f"G{h}"]
    months = sorted(df["결정월"].unique())
    has_y = [m for m in months if y_all[df["결정월"] == m].notna().any()]
    t_W = first_full(h, W, has_y[0])                                                                  # 창이 처음 꽉 차는 달부터(rolling_models 와 같은 규칙)
    if start:
        t_W = max(t_W, pd.Period(start, "M"))
    test_months = [m for m in has_y if m >= t_W][::step]
    cols = feats["full"]
    reg_models, clf_models = models_for("reg", False), models_for("clf", False)
    shap_rows, perm_rows = [], []
    rng = np.random.default_rng(10)
    t0 = time.time()
    for i, t in enumerate(test_months):
        te = (df["결정월"] == t).values
        s_hi, s_lo = t - h, t - h - 12 * W + 1
        tr_mask = ((df["결정월"] >= s_lo) & (df["결정월"] <= s_hi)).values & y_all.notna().values
        if tr_mask.sum() < 17 * 12:
            continue
        tr = df.loc[df.index[tr_mask]].sort_values(["결정월", "region"]).index
        Xtr, Xte = X.loc[tr, cols].values, X.loc[te, cols].values
        regs = df.loc[te, "region"].values
        targets = {"reg": (y_all.loc[tr].values, y_all[te].values)}
        for ev, yb in (("up", y_all >= EVENT_THR), ("dn", y_all <= -EVENT_THR)):
            targets[ev] = (yb.loc[tr].values.astype(int), yb[te].values.astype(int))
        for task, (ytr, yte) in targets.items():
            if task != "reg" and (ytr.sum() < MIN_POS or (len(ytr) - ytr.sum()) < MIN_POS):
                continue
            mk = reg_models["XGB"][1] if task == "reg" else clf_models["XGB"][1]
            pipe = mk().fit(Xtr, ytr)
            imp, booster = pipe.steps[0][1], pipe.steps[-1][1].get_booster()
            Xte_i = imp.transform(Xte)
            contrib = booster.predict(xgb.DMatrix(Xte_i), pred_contribs=True)          # (17, n_feat + 1), 마지막 = 기준값
            for r in range(len(regs)):
                for j, f in enumerate(cols):
                    shap_rows.append((h, W, str(t), task, regs[r], f, float(Xte_i[r, j]), float(contrib[r, j])))
            # 순열 중요도: 회귀 ET, 급등 RF (각 과제의 1차 최선 트리 모형)
            pm = {"reg": ("ET", reg_models["ET"][1]), "up": ("RF", clf_models["RF"][1])}.get(task)
            if pm:
                name, mk2 = pm
                m2 = mk2().fit(Xtr, ytr)
                if task == "reg":
                    base_err = mean_absolute_error(yte, m2.predict(Xte))
                else:
                    if yte.sum() == 0:
                        continue
                    base_err = -average_precision_score(yte, m2.predict_proba(Xte)[:, 1])
                for j, f in enumerate(cols):
                    deltas = []
                    for _ in range(3):
                        Xp = Xte.copy()
                        Xp[:, j] = Xtr[rng.integers(0, len(Xtr), size=len(Xp)), j]            # 학습창에서 무작위로 뽑은 값으로 교체
                        err = mean_absolute_error(yte, m2.predict(Xp)) if task == "reg" else -average_precision_score(yte, m2.predict_proba(Xp)[:, 1])
                        deltas.append(err - base_err)
                    perm_rows.append((h, W, str(t), task, name, f, float(np.mean(deltas))))
        if i % 12 == 0:
            print(f"  h={h} W={W}: {t} ({i + 1}/{len(test_months)}) {time.time() - t0:.0f}s", flush=True)
    shap_df = pd.DataFrame(shap_rows, columns=["h", "W", "t", "task", "region", "feature", "x", "shap"])
    perm_df = pd.DataFrame(perm_rows, columns=["h", "W", "t", "task", "model", "feature", "delta"])
    return shap_df, perm_df


# ------------------------------------------------------------ 요약·그림
def summarize(shap_df, perm_df, log, start=None):
    """변수별·동인별 요약. start 를 주면 변수·동인 비중과 순열 중요도는 그 결정월 이후(지표 집계 기간)만 집계하고, 시간 변화(dt)는 전체 기간을 둔다"""
    shap_df["driver"] = shap_df["feature"].map(lambda f: driver_of(f, log))
    perm_df["driver"] = perm_df["feature"].map(lambda f: driver_of(f, log))
    s = shap_df[shap_df["t"] >= start] if start else shap_df
    perm_df = perm_df[perm_df["t"] >= start] if (start and not perm_df.empty) else perm_df
    # 변수별
    g = s.groupby(["h", "W", "task", "feature", "driver"])
    var = g.agg(mean_abs_shap=("shap", lambda v: float(np.abs(v).mean())), mean_shap=("shap", "mean"),
                sign_corr=("shap", lambda v: float(np.corrcoef(v, s.loc[v.index, "x"])[0, 1]) if v.std() > 0 and s.loc[v.index, "x"].std() > 0 else np.nan)).reset_index()
    var["share"] = var["mean_abs_shap"] / var.groupby(["h", "W", "task"])["mean_abs_shap"].transform("sum")
    var["rank"] = var.groupby(["h", "W", "task"])["mean_abs_shap"].rank(ascending=False).astype(int)
    var["변수"] = var["feature"].map(short_name)
    var["동인"] = var["driver"].map(DRIVER_NAME)
    # 동인별 (|SHAP| 합의 비중)
    drv = var.groupby(["h", "W", "task", "driver"]).agg(abs_sum=("mean_abs_shap", "sum"), n_vars=("feature", "size")).reset_index()
    drv["share"] = drv["abs_sum"] / drv.groupby(["h", "W", "task"])["abs_sum"].transform("sum")
    drv["동인"] = drv["driver"].map(DRIVER_NAME)
    # 동인별 시간 변화: 결정월마다 17개 시도 평균의 (부호 포함) SHAP 합
    dt = shap_df.groupby(["h", "W", "task", "t", "driver"])["shap"].mean().reset_index()
    dt["동인"] = dt["driver"].map(DRIVER_NAME)
    # 순열 중요도
    if perm_df is None or perm_df.empty:
        perm = pd.DataFrame(columns=["h", "W", "task", "model", "feature", "driver", "delta", "rank", "변수", "동인"])
    else:
        perm = perm_df.groupby(["h", "W", "task", "model", "feature", "driver"])["delta"].mean().reset_index()
        perm["rank"] = perm.groupby(["h", "W", "task", "model"])["delta"].rank(ascending=False).astype(int)
        perm["변수"] = perm["feature"].map(short_name)
        perm["동인"] = perm["driver"].map(DRIVER_NAME)
    return var, drv, dt, perm


def fig_top_vars(var, h, W, top=20):
    tasks = [t for t in ("reg", "up", "dn") if t in set(var["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 0.3 * top + 1.6))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        v = var[(var["h"] == h) & (var["W"] == W) & (var["task"] == task)].nsmallest(top, "rank").sort_values("mean_abs_shap")
        ax.barh(range(len(v)), v["mean_abs_shap"], color=[DRIVER_COLOR.get(d, C["muted"]) for d in v["driver"]], height=0.65)
        for k, (_, r) in enumerate(v.iterrows()):
            if pd.notna(r["sign_corr"]):
                ax.text(r["mean_abs_shap"], k, " +" if r["sign_corr"] > 0.2 else (" -" if r["sign_corr"] < -0.2 else " ±"), va="center", fontsize=8, color=C["ink2"])
        ax.set_yticks(range(len(v)))
        ax.set_yticklabels(v["변수"], fontsize=7.5)
        ax.set_title(f"{TASK_KR[task]} — 평균 |SHAP| 상위 {top}", loc="left")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=c, lw=6) for c in DRIVER_COLOR.values()]
    fig.suptitle(f"{h}개월 뒤 예측, 학습창 {W}년, XGB — 변수별 기여도 (막대 끝 부호: 변수값이 클 때 예측이 커지면 '+', 작아지면 '-', 비선형·혼합 '±')", x=0.01, y=1.0, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.legend(handles, [DRIVER_NAME[k] for k in DRIVER_COLOR], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    return img(fig)


def fig_driver_share(drv, h, W):
    tasks = [t for t in ("reg", "up", "dn") if t in set(drv["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.0))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        d = drv[(drv["h"] == h) & (drv["W"] == W) & (drv["task"] == task)].sort_values("share")
        ax.barh(range(len(d)), d["share"], color=[DRIVER_COLOR.get(x, C["muted"]) for x in d["driver"]], height=0.65)
        for k, (_, r) in enumerate(d.iterrows()):
            ax.text(r["share"], k, f" {r['share']:.0%} ({r['n_vars']}개)", va="center", fontsize=8, color=C["ink2"])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["동인"], fontsize=8)
        ax.set_xlim(0, max(0.5, d["share"].max() * 1.3))
        ax.set_title(f"{TASK_KR[task]}", loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"{h}개월 뒤 예측 — 동인별 기여 비중 (동인에 속한 변수들의 평균 |SHAP| 합 ÷ 전체, 괄호 = 변수 수)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    return img(fig)


def fig_driver_time(dt, h, W, start=None):
    tasks = [t for t in ("reg", "up", "dn") if t in set(dt["task"])]
    if start and pd.PeriodIndex(dt["t"], freq="M").min() >= pd.Period(start, "M"):
        start = None                                   # start 전 자료가 없으면 회색 표시 없음
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10.5, 2.6 * len(tasks)), sharex=True, squeeze=False)
    for ax, task in zip(axes[:, 0], tasks):
        d = dt[(dt["h"] == h) & (dt["W"] == W) & (dt["task"] == task)].pivot(index="t", columns="driver", values="shap").fillna(0)
        d.index = pd.PeriodIndex(d.index, freq="M").to_timestamp()
        order = [k for k in DRIVER_COLOR if k in d.columns]
        d = d[order].rolling(3, min_periods=1).mean()
        ax.stackplot(d.index, [d[k].clip(lower=0) for k in order], colors=[DRIVER_COLOR[k] for k in order], alpha=0.85)
        ax.stackplot(d.index, [d[k].clip(upper=0) for k in order], colors=[DRIVER_COLOR[k] for k in order], alpha=0.85)
        ax.axhline(0, color=C["axis"], lw=0.8)
        if start:
            ax.axvspan(d.index.min(), pd.Period(start, "M").to_timestamp(), color=C["grid"], alpha=0.55, lw=0, zorder=0)
        ax.set_title(f"{TASK_KR[task]} — 동인별 기여(17개 시도 평균 SHAP, 3개월 이동평균)의 시간 변화", loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=DRIVER_COLOR[k]) for k in DRIVER_COLOR]
    fig.legend(handles, [DRIVER_NAME[k] for k in DRIVER_COLOR], loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{h}개월 뒤 예측, 학습창 {W}년, XGB — 0 위는 예측을 올리는 기여, 0 아래는 내리는 기여 (기준값 제외" + (f"; 회색 = {start} 전, 집계 제외)" if start else ")"),
                 x=0.01, y=1.01, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    return img(fig)


def fig_dependence(shap_df, var, h, W, task="reg", top=6):
    v = var[(var["h"] == h) & (var["W"] == W) & (var["task"] == task)].nsmallest(top, "rank")
    fig, axes = plt.subplots(1, top, figsize=(3.0 * top, 2.8))
    for ax, (_, r) in zip(axes, v.iterrows()):
        s = shap_df[(shap_df["h"] == h) & (shap_df["W"] == W) & (shap_df["task"] == task) & (shap_df["feature"] == r["feature"])]
        ax.scatter(s["x"], s["shap"], s=5, alpha=0.35, color=DRIVER_COLOR.get(r["driver"], C["muted"]), edgecolors="none")
        ax.axhline(0, color=C["axis"], lw=0.8)
        ax.set_title(two_line(r["feature"]), loc="left", fontsize=8)
        ax.set_xlabel("변수값", fontsize=8)
    axes[0].set_ylabel("SHAP", fontsize=8)
    fig.suptitle(f"{h}개월 뒤 예측, {TASK_KR[task]} — 상위 {top}개 변수의 값 vs SHAP (평가 행 전부, 비선형 관계 확인)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    return img(fig)


def build_section(var, drv, dt, perm, shap_df, hs, W, start=None):
    parts = ["<p class='note'><b>방법</b>: 권고 설정(학습창 4년, 1차 고정 파라미터)의 롤링을 그대로 재현하며 매 결정월 평가 행(17개 시도)에서 XGB 의 TreeSHAP(모형이 그 예측을 낸 데 각 특성이 기여한 양, 회귀는 %p, 분류는 로그오즈)를 모았다. "
             + (f"평가 기간({start}~) 평가 행의 " if start else "전체 평가 기간의 ") + "평균 |SHAP| 가 변수 중요도, 동인별 합의 비중이 동인 기여도다. ET(회귀)·RF(급등)는 표본 외 순열 중요도(평가 행의 특성값을 학습창 값으로 바꿨을 때 오차 증가)로 보완했다.</p>"]
    for h in hs:
        parts.append(f"<h3>{h}개월 뒤 예측</h3>" + fig_driver_share(drv, h, W) + fig_top_vars(var, h, W) + fig_driver_time(dt, h, W, start))
        if ((shap_df["h"] == h) & (shap_df["task"] == "reg")).any():
            parts.append(fig_dependence(shap_df, var, h, W, "reg"))
        d = drv[(drv["h"] == h) & (drv["W"] == W)].pivot(index="동인", columns="task", values="share").rename(columns={"reg": "회귀", "up": "급등", "dn": "급락"})
        parts.append("<p class='note'>동인별 기여 비중(XGB |SHAP| 합 ÷ 전체)</p>" + d.to_html(float_format=lambda x: f"{x:.1%}", na_rep=""))
        v = var[(var["h"] == h) & (var["W"] == W)]
        piv = v.pivot_table(index=["동인", "변수"], columns="task", values="rank").rename(columns={"reg": "회귀 순위", "up": "급등 순위", "dn": "급락 순위"})
        piv["평균 순위"] = piv.mean(axis=1)
        piv = piv.sort_values("평균 순위").head(25)
        parts.append("<details><summary>변수별 순위 상위 25 (세 과제 평균)</summary>" + piv.to_html(float_format=lambda x: f"{x:.0f}") + "</details>")
        p = perm[(perm["h"] == h) & (perm["W"] == W)]
        for (task, model), g in p.groupby(["task", "model"]):
            g = g.nsmallest(15, "rank")[["변수", "동인", "delta"]].rename(columns={"delta": "오차 증가(MAE %p 또는 −PR-AUC)"})
            parts.append(f"<details><summary>{model} {TASK_KR[task][:2]} 표본 외 순열 중요도 상위 15</summary>" + g.to_html(index=False, float_format=lambda x: f"{x:.4f}") + "</details>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="*", default=[6, 3])
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--from-csv", action="store_true", help="계산을 건너뛰고 저장된 contrib_*.csv 로 요약·그림만 다시 만든다")
    ap.add_argument("--start", default=None, help="평가 시작 하한(선택). 기본은 창이 처음 꽉 차는 달부터(first_full)")
    args = ap.parse_args()
    log = vc.log_load()
    if args.from_csv:
        shap_df = pd.read_csv(os.path.join(OUT, "contrib_shap_long.csv"), encoding="utf-8-sig")
        perm_df = pd.read_csv(os.path.join(OUT, "contrib_perm_long.csv"), encoding="utf-8-sig") if os.path.exists(os.path.join(OUT, "contrib_perm_long.csv")) else pd.DataFrame(columns=["h", "W", "t", "task", "model", "feature", "delta"])
        args.horizons = sorted(shap_df["h"].unique(), reverse=True)
    else:
        df, X, feats, expl = load(args.start)
        shap_all, perm_all = [], []
        for h in args.horizons:
            s, p = run(h, args.W, args.step, df, X, feats, log, args.start)
            shap_all.append(s)
            perm_all.append(p)
        shap_df, perm_df = pd.concat(shap_all, ignore_index=True), pd.concat(perm_all, ignore_index=True)
        perm_df.to_csv(os.path.join(OUT, "contrib_perm_long.csv"), index=False, encoding="utf-8-sig")
    start = args.start or None
    var, drv, dt, perm = summarize(shap_df, perm_df, log, start)
    if args.from_csv and perm.empty and os.path.exists(os.path.join(OUT, "contrib_perm.csv")):      # 긴 형식이 없으면 저장된 요약을 그대로 씀
        perm = pd.read_csv(os.path.join(OUT, "contrib_perm.csv"), encoding="utf-8-sig")
        perm["변수"] = perm["변수"].str.replace("−", "-")
    shap_df.to_csv(os.path.join(OUT, "contrib_shap_long.csv"), index=False, encoding="utf-8-sig")        # 긴 형식은 전체 기간 그대로 저장
    var.to_csv(os.path.join(OUT, "contrib_var.csv"), index=False, encoding="utf-8-sig")                  # 요약 CSV 는 start 이후 집계
    drv.to_csv(os.path.join(OUT, "contrib_driver.csv"), index=False, encoding="utf-8-sig")
    dt.to_csv(os.path.join(OUT, "contrib_driver_time.csv"), index=False, encoding="utf-8-sig")
    perm.to_csv(os.path.join(OUT, "contrib_perm.csv"), index=False, encoding="utf-8-sig")
    html = build_section(var, drv, dt, perm, shap_df[shap_df["t"] >= start] if start else shap_df, args.horizons, args.W, start)
    with open(os.path.join(OUT, "기여도_section.html"), "w", encoding="utf-8") as f:
        f.write(html)
    pd.set_option("display.width", 220)
    for h in args.horizons:
        print(f"\n== h={h} 동인별 기여 비중")
        print(drv[drv["h"] == h].pivot(index="동인", columns="task", values="share").round(3).to_string())
        print(f"== h={h} 회귀 상위 10 변수")
        print(var[(var["h"] == h) & (var["task"] == "reg")].nsmallest(10, "rank")[["변수", "동인", "mean_abs_shap", "sign_corr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
