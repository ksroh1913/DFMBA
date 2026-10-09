# -*- coding: utf-8 -*-
"""
롤링 윈도우 전진 평가 — 베이스라인(타깃 자신의 모멘텀 AR) 대비 설명변수 추가 모형의 성능, 학습창(2~5년) 민감도, 시간에 따른 성능 변화.

자료: analysis/output/모형입력표_10차_values.csv (17개 시도 × 결정월, 타깃 G3·G6, 설명변수 47열, model_values.py 가 만듦)
설계
  - 결정월 t(월말)마다, t 시점에 정답이 확정된 행만 학습: 지평 h 의 타깃 G_h(s) 는 s ≤ t−h 인 결정월 s 에서만 앎
    (G_h(s) = 100·[R(s+h−1)/R(s−1) − 1], R 은 1개월 뒤 공표). 학습창 = 직전 W년(12W 개 결정월) × 17개 시도, 매월 전진.
  - 회귀: y = G_h (%, MAE). 베이스라인 AR(Ridge) = past1·past3·past6 만. 참고: 0 예측, 모멘텀(past_h). 비교: Ridge·RF·ET·XGB (모멘텀 + 설명변수 + 지역 더미)
  - 분류: 급등 = G_h ≥ +1, 급락 = G_h ≤ −1 (각각 이진). 베이스라인 AR(Logit) = 모멘텀만. 비교: Logit·RF·ET·XGB.
    지표 F1(임계 0.5 와 학습창 기준율), PR-AUC(평균정밀도), BSS = 1 − Brier/Brier(기준율). 학습창에 양성이 MIN_POS 미만이면 모든 분류기가 기준율을 예측(기록).
  - 선형 모형: 중앙값 대치 + 표준화. 트리: 중앙값 대치. 하이퍼파라미터 고정(튜닝 없음).
출력: analysis/output/모형결과_10차/predictions_h{h}.csv (긴 형식), metrics_summary.csv, rolling_metrics.csv, settings.json
사용: PYTHONUTF8=1 python analysis/rolling_models.py [--quick] [--horizons 6 3] [--windows 2 3 4 5] [--start 2018-01]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALUES = os.path.join(BASE, "analysis", "output", "모형입력표_10차_values.csv")
OUT = os.path.join(BASE, "analysis", "output", "모형결과_10차")
MIN_POS = 5
EVENT_THR = 1.0
SEED = 10


# 2차 튜닝용 격자 (학습창 안 중첩 시계열 CV 로 선택). 파이프라인 마지막 단계 이름 = 추정기 클래스 이름 소문자
GRIDS = {
    "RF": {"min_samples_leaf": [3, 5, 10], "max_features": [0.3, 0.5, 0.8]},
    "ET": {"min_samples_leaf": [3, 5, 10], "max_features": [0.3, 0.5, 0.8]},
    "XGB": {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "n_estimators": [200, 400]},
    "Logit": {"C": [0.03, 0.1, 0.3, 1.0, 3.0]},
}


def models_for(task, quick, tune=(), gap_rows=0, test_rows=None):
    """task: 'reg' | 'clf'. 반환 {이름: (특성집합, 추정기 생성 함수)}.
    tune 에 든 모형은 GridSearchCV(TimeSeriesSplit 3겹, 검증 겹 test_rows 행 = 3개월, 겹 사이 gap_rows 행 = h−1개월: 학습 겹의 정답이 검증 시점에 확정된 것만)로
    학습창 안에서 격자 선택 뒤 전체 창으로 재학습"""
    from xgboost import XGBClassifier, XGBRegressor
    n_tree = 100 if quick else 300
    tree_kw = dict(n_estimators=n_tree, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED)
    xgb_kw = dict(n_estimators=n_tree, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=4, verbosity=0)
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    alphas = np.logspace(-2, 3, 11)                                   # Ridge 의 alpha 는 학습창 안에서 LOO 로 고름(수동 튜닝 없음)
    if task == "reg":
        base = {"AR(Ridge)": ("ar", lambda: make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=alphas))),
                "Ridge": ("full", lambda: make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=alphas))),
                "RF": ("full", lambda: make_pipeline(imp(), RandomForestRegressor(**tree_kw))),
                "ET": ("full", lambda: make_pipeline(imp(), ExtraTreesRegressor(**tree_kw))),
                "XGB": ("full", lambda: make_pipeline(imp(), XGBRegressor(**xgb_kw)))}
        scoring = "neg_mean_absolute_error"
    else:
        base = {"AR(Logit)": ("ar", lambda: make_pipeline(imp(), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
                "Logit": ("full", lambda: make_pipeline(imp(), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
                "RF": ("full", lambda: make_pipeline(imp(), RandomForestClassifier(**tree_kw))),
                "ET": ("full", lambda: make_pipeline(imp(), ExtraTreesClassifier(**tree_kw))),
                "XGB": ("full", lambda: make_pipeline(imp(), XGBClassifier(**xgb_kw, eval_metric="logloss")))}
        scoring = "average_precision"
    out = {}
    for nm, (fs, mk) in base.items():
        if nm in tune and nm in GRIDS:
            def mk_tuned(mk=mk, nm=nm):
                pipe = mk()
                step = pipe.steps[-1][0]                                   # 예: 'randomforestregressor'
                grid = {f"{step}__{k}": v for k, v in GRIDS[nm].items()}
                return GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=3, test_size=test_rows, gap=gap_rows), scoring=scoring, n_jobs=1, refit=True)
            out[nm] = (fs, mk_tuned)
        else:
            out[nm] = (fs, mk)
    return out


def load(start):
    df = pd.read_csv(VALUES, encoding="utf-8-sig")
    df["결정월"] = pd.PeriodIndex(df["결정월"], freq="M")
    expl = [c for c in df.columns[11:]]
    mom = ["past1", "past3", "past6"]
    dummies = pd.get_dummies(df["region"], prefix="R", dtype=float)
    X = pd.concat([df[mom + expl], dummies], axis=1)
    feats = {"ar": mom, "full": mom + expl + list(dummies.columns)}
    return df, X, feats, expl


def run(h, windows, start, quick, df, X, feats, tune=(), step=1):
    y_all = df[f"G{h}"]
    months = sorted(df["결정월"].unique())
    test_months = [m for m in months if m >= pd.Period(start, "M") and y_all[df["결정월"] == m].notna().any()][::step]
    n_reg = df["region"].nunique()
    cv_kw = dict(gap_rows=n_reg * (h - 1), test_rows=n_reg * 3)        # 검증 겹 3개월, 간격 h−1개월 (학습 겹 s ≤ 검증 v − h)
    reg_models, clf_models = models_for("reg", quick, tune, **cv_kw), models_for("clf", quick, tune, **cv_kw)
    rows, params = [], []

    def note_params(m, task, nm, W, t):                                 # 튜닝 모형이면 그 결정월에 선택된 격자값과 CV 점수를 기록
        if hasattr(m, "best_params_"):
            params.append(dict(h=h, W=W, t=str(t), task=task, model=nm, cv_score=float(m.best_score_),
                               **{k.split("__")[-1]: v for k, v in m.best_params_.items()}))
    t0 = time.time()
    for W in windows:
        for i, t in enumerate(test_months):
            te = (df["결정월"] == t).values
            s_hi = t - h
            s_lo = s_hi - 12 * W + 1
            tr_mask = ((df["결정월"] >= s_lo) & (df["결정월"] <= s_hi)).values & y_all.notna().values
            if tr_mask.sum() < 17 * 12:
                continue
            tr = df.loc[df.index[tr_mask]].sort_values(["결정월", "region"]).index   # 학습 행 인덱스, 시간 순 (중첩 CV 의 TimeSeriesSplit 이 시간 블록이 되도록)
            ytr, yte = y_all.loc[tr].values, y_all[te].values
            base = dict(h=h, W=W, t=str(t), region=df.loc[te, "region"].values, y=yte, n_train=int(len(tr)))
            # 참고 예측(모형 아님)
            for nm, pred in (("Zero", np.zeros(te.sum())), ("Mom(past_h)", df.loc[te, f"past{h}"].values)):
                rows.append(pd.DataFrame({**base, "task": "reg", "model": nm, "pred": pred}))
            for nm, (fs, mk) in reg_models.items():
                m = mk().fit(X.loc[tr, feats[fs]].values, ytr)
                note_params(m, "reg", nm, W, t)
                rows.append(pd.DataFrame({**base, "task": "reg", "model": nm, "pred": m.predict(X.loc[te, feats[fs]].values)}))
            for ev, ybin_all in (("up", (y_all >= EVENT_THR)), ("dn", (y_all <= -EVENT_THR))):
                ybin_tr = ybin_all.loc[tr].values.astype(int)
                ybin_te = ybin_all[te].values.astype(int)
                clim = float(ybin_tr.mean())
                b = {**base, "task": ev, "y": ybin_te, "clim": clim}
                rows.append(pd.DataFrame({**b, "model": "Clim", "pred": np.full(te.sum(), clim)}))
                if ybin_tr.sum() < MIN_POS or (len(ybin_tr) - ybin_tr.sum()) < MIN_POS:
                    for nm in clf_models:
                        rows.append(pd.DataFrame({**b, "model": nm, "pred": np.full(te.sum(), clim), "fallback": True}))
                    continue
                for nm, (fs, mk) in clf_models.items():
                    m = mk().fit(X.loc[tr, feats[fs]].values, ybin_tr)
                    note_params(m, ev, nm, W, t)
                    rows.append(pd.DataFrame({**b, "model": nm, "pred": m.predict_proba(X.loc[te, feats[fs]].values)[:, 1], "fallback": False}))
            if i % 12 == 0:
                print(f"  h={h} W={W}: {t} ({i + 1}/{len(test_months)}) {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    if "fallback" not in out.columns:
        out["fallback"] = np.nan
    return out, pd.DataFrame(params)


def summarize(pred):
    """전체 평가 기간 지표(모형 × 학습창 × 지평 × 과제) + 시간에 따른 이동 지표"""
    pred["tP"] = pd.PeriodIndex(pred["t"], freq="M")
    summ, roll = [], []
    for (h, W, task, model), g in pred.groupby(["h", "W", "task", "model"]):
        g = g.dropna(subset=["y", "pred"])
        if task == "reg":
            summ.append(dict(h=h, W=W, task=task, model=model, n=len(g), MAE=mean_absolute_error(g["y"], g["pred"]),
                             t_first=str(g["tP"].min()), t_last=str(g["tP"].max())))
            for t, gg in g.groupby("tP"):
                roll.append(dict(h=h, W=W, task=task, model=model, t=str(t), MAE=mean_absolute_error(gg["y"], gg["pred"])))
        else:
            y, p = g["y"].astype(int).values, g["pred"].values
            clim = g["clim"].values
            bs, bs_clim = brier_score_loss(y, p), brier_score_loss(y, clim)
            thr_base = g["clim"].values
            summ.append(dict(h=h, W=W, task=task, model=model, n=len(g), 양성비율=float(y.mean()),
                             F1_05=f1_score(y, (p >= 0.5).astype(int), zero_division=0), F1_기준율=f1_score(y, (p >= thr_base).astype(int), zero_division=0),
                             PR_AUC=average_precision_score(y, p) if y.sum() else np.nan, Brier=bs, BSS=1 - bs / bs_clim if bs_clim > 0 else np.nan,
                             기준율예측비율=float(g["fallback"].fillna(False).astype(bool).mean()) if model != "Clim" else 0.0,
                             t_first=str(g["tP"].min()), t_last=str(g["tP"].max())))
            for t, gg in g.groupby("tP"):
                yy, pp = gg["y"].astype(int).values, gg["pred"].values
                roll.append(dict(h=h, W=W, task=task, model=model, t=str(t), Brier=brier_score_loss(yy, pp), Brier_clim=brier_score_loss(yy, gg["clim"].values),
                                 n_pos=int(yy.sum()), n=len(yy), hit05=int(((pp >= 0.5) & (yy == 1)).sum()), pred05=int((pp >= 0.5).sum()),
                                 hitB=int(((pp >= gg["clim"].values) & (yy == 1)).sum()), predB=int((pp >= gg["clim"].values).sum())))
    return pd.DataFrame(summ), pd.DataFrame(roll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="*", default=[6, 3])
    ap.add_argument("--windows", type=int, nargs="*", default=[2, 3, 4, 5])
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--quick", action="store_true", help="트리 100개, 학습창 2·5년, 빠른 점검")
    ap.add_argument("--tune", nargs="*", default=[], help="학습창 안 중첩 시계열 CV 로 격자 선택할 모형 이름 (RF ET XGB Logit)")
    ap.add_argument("--step", type=int, default=1, help="평가 결정월 간격(개월). 튜닝 실행 시간 절약용")
    ap.add_argument("--tag", default="", help="출력 폴더 접미사 (예: _tuned)")
    args = ap.parse_args()
    if args.quick:
        args.windows = [w for w in args.windows if w in (2, 5)] or args.windows
    out_dir = OUT + args.tag
    os.makedirs(out_dir, exist_ok=True)
    df, X, feats, expl = load(args.start)
    print(f"자료 {df.shape}, 설명변수 {len(expl)}, 특성(full) {len(feats['full'])}, 지평 {args.horizons}, 학습창 {args.windows}년, 평가 시작 {args.start}"
          + (f", 튜닝 {args.tune} (격자 {[GRIDS[m] for m in args.tune if m in GRIDS]}), 간격 {args.step}개월" if args.tune else ""))
    preds, params = [], []
    for h in args.horizons:
        p, prm = run(h, args.windows, args.start, args.quick, df, X, feats, tune=tuple(args.tune), step=args.step)
        p.to_csv(os.path.join(out_dir, f"predictions_h{h}.csv"), index=False, encoding="utf-8-sig")
        preds.append(p)
        params.append(prm)
    pred = pd.concat(preds, ignore_index=True)
    params = pd.concat(params, ignore_index=True)
    if len(params):
        params.to_csv(os.path.join(out_dir, "tuned_params.csv"), index=False, encoding="utf-8-sig")   # 결정월마다 선택된 격자값
    summ, roll = summarize(pred)
    summ.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")
    roll.to_csv(os.path.join(out_dir, "rolling_metrics.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(dict(horizons=args.horizons, windows=args.windows, start=args.start, quick=args.quick, event_thr=EVENT_THR, min_pos=MIN_POS, step=args.step,
                       tune=args.tune, grids={m: GRIDS[m] for m in args.tune if m in GRIDS},
                       n_expl=len(expl), features_full=feats["full"], features_ar=feats["ar"], models_reg=list(models_for("reg", True)), models_clf=list(models_for("clf", True)),
                       note="학습 행 = 결정월 s ≤ t−h (정답 확정) 인 직전 W년. 선형: 중앙값 대치+표준화, 트리: 중앙값 대치. "
                            + ("튜닝 모형은 학습창 안 TimeSeriesSplit(3겹, 지평만큼 간격) 격자 선택 뒤 전체 창 재학습, 나머지는 고정" if args.tune else "하이퍼파라미터 고정(Ridge alpha 만 RidgeCV)")),
                  f, ensure_ascii=False, indent=1)
    pd.set_option("display.width", 220)
    print("\n== 회귀 MAE (낮을수록 좋음)")
    print(summ[summ["task"] == "reg"].pivot_table(index=["h", "model"], columns="W", values="MAE").round(3).to_string())
    for ev in ("up", "dn"):
        print(f"\n== 분류 {'급등' if ev == 'up' else '급락'} PR-AUC / BSS")
        s = summ[summ["task"] == ev]
        print(s.pivot_table(index=["h", "model"], columns="W", values=["PR_AUC", "BSS"]).round(3).to_string())


if __name__ == "__main__":
    main()
