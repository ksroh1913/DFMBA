# -*- coding: utf-8 -*-
"""
[7단계 해석 · 분석 5] SHAP(시점 귀속), 동인 제거 후 재학습, PDP, 서울 구 상대 상승 순위(보조), 추가 비교모형(부록), 팀 보고서 비교
(연구계획 8차 확정본 6·10장).

  A 전국 Ridge B 의 선형 SHAP: 기여 = 계수 x 표준화 입력(훈련 평균 기준). 매달 재적합한 모형으로 그 달의 예측행을 설명한다(시점 귀속)
  B 전국 Extra Trees B 의 TreeSHAP (h=3, 보조)
  C 동인 제거 후 재학습: 공통 RMPI 에서 동인 하나를 빼고 RQ1 B 를 다시 평가(MAE), RQ3 B 에서 공통·지역 동인을 빼고 Brier
  D PDP: 전국 ET B(h=3, 마지막 시점)의 공통 하위지수 1-way, 급락 ET B 의 공통③ x 지역③ 2-way (관측된 값만 축에 표시)
  E 서울 25개 구 상대 상승 순위(보조): A_지역 / B_지역(지역 RMPI) / C_지역 Ridge, 월내 Spearman·월내 상대 MAE
  F 추가 비교모형(부록): 같은 분할, 고정 설정. 선형·Lasso·Elastic Net·KNN·SVR·Decision Tree·XGBoost·Random Forest
  G 팀 보고서(TH v9) 수치와의 비교표 (같은 행의 예측값 공유 전까지는 보고된 수치와 우리 결과를 나란히 둔다)
출력: rmpi/output/stage7_*.csv, fig7_*.png
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import engine as E  # noqa: E402
from rmpi import frames as F  # noqa: E402
from rmpi import models as M  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402
from rmpi import viz as V  # noqa: E402
from rmpi.index import DRIVER_MARK  # noqa: E402

plt = V.plt
t0 = time.time()
SMOKE = "--smoke" in sys.argv   # 코드 점검용: 처음 3개 시점만
PAL8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def log(m):
    print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def group_of(name):
    if name.startswith("RMPI공통|") or name.startswith("RMPI지역|"):
        parts = name.split("|")
        return ("결측표시" if len(parts) > 2 else f"{'공통' if '공통' in parts[0] else '지역'} {parts[1]}")
    if name == "옛체계":
        return "전환 더미"
    return "A(가격 추세)"


def monthly_loss(df, kind="abs"):
    d = df.dropna(subset=["y", "yhat"]).copy()
    d["loss"] = (d.yhat - d.y) ** 2 if kind == "sq" else (d.yhat - d.y).abs()
    o = d.groupby("P")["loss"].mean()
    o.index = pd.PeriodIndex(o.index, freq="M")
    return o


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    V.setup()
    hs = s["timing"]["horizons"]
    first = S.per(s["timing"]["official_first_decision"])
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    if SMOKE:
        origins = origins[:3]
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat)
    pf = F.panel_frame(s, b, pt, nat)
    ccols = [c for c in spec.loc[spec.block.isin(["regional", "common"]), "입력"] if c in nf.columns]
    rev = s["revision_risk"]["columns"]

    # ============================================================ A. 선형 SHAP (전국 Ridge B)
    shap_rows = []
    for h in hs:
        ct = M.features(s, spec, F.A_NAT, common_cols=ccols)
        pipe = M.ridge(s, ct, "national")
        frame = nf.copy()
        frame[F.SIGN_C] = frame[f"G{h}"]
        ycol = f"G{h}"
        for Tn in origins:
            tr = frame[(frame.index >= first) & (frame.index <= Tn - h) & frame[ycol].notna() & frame[F.A_NAT].notna().all(axis=1)]
            te = frame.loc[[Tn]]
            p = clone(pipe).fit(tr.drop(columns=[ycol]), tr[ycol])
            z = p[:-1].transform(te.drop(columns=[ycol]))
            coef = p.named_steps["model"].coef_
            names = list(z.columns)
            contrib = coef * z.values[0]
            for n_, c_ in zip(names, contrib):
                shap_rows.append({"h": h, "P": str(Tn), "입력": n_, "묶음": group_of(n_), "기여": float(c_), "모형": "ridge_B"})
            shap_rows.append({"h": h, "P": str(Tn), "입력": "(절편=훈련 평균)", "묶음": "절편", "기여": float(p.named_steps["model"].intercept_), "모형": "ridge_B"})
        log(f"선형 SHAP h={h}")
    SH = pd.DataFrame(shap_rows)
    SH.to_csv(os.path.join(out, "stage7_SHAP_전국.csv"), index=False, encoding="utf-8-sig")
    imp = SH[SH["묶음"] != "절편"].assign(a=lambda d: d["기여"].abs()).groupby(["h", "묶음"])["a"].mean().unstack(0).round(4)
    imp.to_csv(os.path.join(out, "stage7_SHAP_요약.csv"), encoding="utf-8-sig")

    # ============================================================ B. TreeSHAP (전국 ET B, h=3)
    try:
        import shap
        h = 3
        ct = M.features(s, spec, F.A_NAT, common_cols=ccols)
        pipe = M.extra_trees(s, ct, "national", h, "reg")
        frame = nf.copy()
        frame[F.SIGN_C] = frame[f"G{h}"]
        rows = []
        for Tn in origins:
            tr = frame[(frame.index >= first) & (frame.index <= Tn - h) & frame[f"G{h}"].notna() & frame[F.A_NAT].notna().all(axis=1)]
            te = frame.loc[[Tn]]
            p = clone(pipe).fit(tr.drop(columns=[f"G{h}"]), tr[f"G{h}"])
            Z = p[:-1].transform(te.drop(columns=[f"G{h}"]))
            sv = shap.TreeExplainer(p.named_steps["model"]).shap_values(Z.values, check_additivity=False)
            for n_, c_ in zip(Z.columns, np.asarray(sv).ravel()):
                rows.append({"h": h, "P": str(Tn), "입력": n_, "묶음": group_of(n_), "기여": float(c_), "모형": "et_B"})
        ETS = pd.DataFrame(rows)
        ETS.to_csv(os.path.join(out, "stage7_SHAP_전국_ET.csv"), index=False, encoding="utf-8-sig")
        log("TreeSHAP 완료")
    except Exception as e:   # noqa: BLE001
        ETS = pd.DataFrame()
        log(f"TreeSHAP 생략: {e}")

    # ============================================================ C. 동인 제거 후 재학습
    rem_rows = []
    byd = {d: spec.loc[spec["동인"] == d, "ID"].tolist() for d in range(1, 7)}
    for h in hs:
        full, _ = E.national_run(s, nf, spec, h, origins, info="B")
        base = monthly_loss(full).mean()
        rem_rows.append({"과제": "RQ1 B_공통(전국 MAE)", "h": h, "제거 동인": "(없음)", "손실": base, "변화": 0.0})
        for d, ids in byd.items():
            p, _ = E.national_run(s, nf, spec, h, origins, info="B", exclude_ids=ids, label=f"B_-{d}")
            L = monthly_loss(p).mean()
            rem_rows.append({"과제": "RQ1 B_공통(전국 MAE)", "h": h, "제거 동인": DRIVER_MARK[d], "손실": L, "변화": L - base})
        pb, _, _ = E.panel_run(s, pf, spec, h, origins, target="down", info="B")
        base = monthly_loss(pb, "sq").mean()
        rem_rows.append({"과제": "RQ3 B(패널 Brier)", "h": h, "제거 동인": "(없음)", "손실": base, "변화": 0.0})
        for d, ids in byd.items():
            p, _, _ = E.panel_run(s, pf, spec, h, origins, target="down", info="B", exclude_ids=ids, label=f"B_-{d}")
            L = monthly_loss(p, "sq").mean()
            rem_rows.append({"과제": "RQ3 B(패널 Brier)", "h": h, "제거 동인": DRIVER_MARK[d], "손실": L, "변화": L - base})
        log(f"동인 제거 h={h}")
    pd.DataFrame(rem_rows).to_csv(os.path.join(out, "stage7_동인제거.csv"), index=False, encoding="utf-8-sig")

    # ============================================================ D. PDP
    h = 3
    Tl = origins[-1]
    ct = M.features(s, spec, F.A_NAT, common_cols=ccols)
    pipe = M.extra_trees(s, ct, "national", h, "reg")
    frame = nf.copy()
    frame[F.SIGN_C] = frame[f"G{h}"]
    tr = frame[(frame.index >= first) & (frame.index <= Tl - h) & frame[f"G{h}"].notna() & frame[F.A_NAT].notna().all(axis=1)]
    p = clone(pipe).fit(tr.drop(columns=[f"G{h}"]), tr[f"G{h}"])
    Ztr = p[:-1].transform(tr.drop(columns=[f"G{h}"]))
    est = p.named_steps["model"]
    feats = [c for c in Ztr.columns if c.startswith("RMPI공통|") and not c.endswith("결측")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6))
    pdp_rows = []
    for ax, c in zip(axes.ravel(), feats):
        xs = np.quantile(Ztr[c].dropna(), np.linspace(0.02, 0.98, 25))
        ys = []
        for x in xs:
            Zc = Ztr.copy()
            Zc[c] = x
            ys.append(float(est.predict(Zc).mean()))
        ax.plot(xs, ys, color=V.SERIES[0])
        ax.plot(Ztr[c].values, np.full(len(Ztr), min(ys) - 0.05 * (max(ys) - min(ys) + 1e-9)), "|", color=V.GRAY, ms=6)
        ax.set_title(c.replace("RMPI공통|", "공통 "), fontsize=9.5)
        ax.set_xlabel("표준화 단위 (눈금: 관측된 월)", fontsize=8)
        pdp_rows += [{"입력": c, "x": float(x), "pdp": y} for x, y in zip(xs, ys)]
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"전국 Extra Trees B 의 부분의존(h=3, 2025.12 적합). 세로축: 예측 G(t,3) %. 관측 범위 안에서만 해석", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_2_PDP_공통.png"))
    plt.close(fig)
    pd.DataFrame(pdp_rows).to_csv(os.path.join(out, "stage7_PDP_공통.csv"), index=False, encoding="utf-8-sig")
    # 2-way: 급락 ET B, 공통③ x 지역③
    dcols = [f"D|{c}" for c in spec.loc[spec.block == "regional", "입력"] if f"D|{c}" in pf.columns]
    ccols_p = [f"C|{c}" for c in spec.loc[spec.block.isin(["regional", "common"]), "입력"] if f"C|{c}" in pf.columns]
    ctp = M.features(s, spec, F.A_REG, common_cols=ccols_p, regional_cols=dcols, prefix_c="C|", prefix_d="D|")
    pipe_p = M.extra_trees(s, ctp, "panel", h, "clf")
    fr = pf.copy()
    fr[F.SIGN_R], fr[F.SIGN_C] = fr[f"r{h}"], fr[f"Gnat{h}"]
    trp = fr[(fr.P >= first) & (fr.P <= Tl - h) & fr[f"down{h}"].notna() & fr[F.A_REG].notna().all(axis=1)]
    pp = clone(pipe_p).fit(trp.drop(columns=[f"down{h}"]), trp[f"down{h}"])
    Zp = pp[:-1].transform(trp.drop(columns=[f"down{h}"]))
    cx, cy = "RMPI공통|③", "RMPI지역|③"
    if cx in Zp.columns and cy in Zp.columns:
        gx = np.quantile(Zp[cx], np.linspace(0.03, 0.97, 15))
        gy = np.quantile(Zp[cy], np.linspace(0.03, 0.97, 15))
        grid = np.zeros((len(gy), len(gx)))
        for i, yv in enumerate(gy):
            for j, xv in enumerate(gx):
                Zc = Zp.copy()
                Zc[cx], Zc[cy] = xv, yv
                grid[i, j] = pp.named_steps["model"].predict_proba(Zc)[:, 1].mean()
        fig, ax = plt.subplots(figsize=(6.4, 5))
        im = ax.pcolormesh(gx, gy, grid, cmap="Blues", shading="nearest")
        ax.scatter(Zp[cx], Zp[cy], s=4, color=V.INK2, alpha=0.35, lw=0)
        ax.set_xlabel("공통 ③ 금융여건·상대가격 (표준화)")
        ax.set_ylabel("지역 ③ 금융여건·상대가격 월내 편차 (표준화)")
        ax.set_title("급락 확률의 2-way 부분의존 (h=3, Extra Trees B, 2025.12 적합). 점은 관측된 훈련행", fontsize=9.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="평균 예측 확률")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig7_3_PDP_2way.png"))
        plt.close(fig)
    log("PDP 완료")

    # ============================================================ E. 서울 25개 구 (보조)
    bg = D.build(s, "gu")
    spec_g, ml_g = bg["spec"], bg["ml"]
    pg = T.regional(s, ml_g, "gu")
    pfg = F.panel_frame(s, bg, pg, nat)
    gu_rows, gu_preds = [], []
    for h in hs:
        res = {}
        for info in ("A", "B", "C"):
            p, _, _ = E.panel_run(s, pfg, spec_g, h, origins, target="r", info=info, label=f"gu_{info}")
            res[info] = p
            gu_preds.append(p)
        zero = E.panel_naive(s, pfg, h, origins, target="r")
        res["zero"] = zero
        for k, p in res.items():
            L = monthly_loss(p).mean()
            sp_ = []
            for _, g in p.dropna(subset=["y", "yhat"]).groupby("P"):
                if len(g) >= 10 and g.yhat.nunique() > 1:
                    sp_.append(spearmanr(g.yhat, g.y).statistic)
            gu_rows.append({"h": h, "모형": k, "월내 상대 MAE": L, "월내 Spearman(평균)": float(np.nanmean(sp_)) if sp_ else np.nan,
                            "Spearman 유효월": len(sp_)})
        log(f"서울 구 h={h}")
    GU = pd.DataFrame(gu_rows)
    GU.to_csv(os.path.join(out, "stage7_서울구.csv"), index=False, encoding="utf-8-sig")
    pd.concat(gu_preds).to_csv(os.path.join(out, "stage7_서울구_예측값.csv"), index=False, encoding="utf-8-sig")
    comp_g = M.compositions  # noqa: F841

    # ============================================================ F. 추가 비교모형 (부록)
    def extra_models(kind, h):
        leaf = 6 if kind == "national" else {1: 34, 3: 68, 6: 136}[h]
        seed = s["meta"]["random_seed"]
        import xgboost as xgb
        return {
            "선형회귀": LinearRegression(),
            "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=20000),
            "ElasticNet(0.01, l1=0.5)": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000),
            f"KNN(k={2 if kind == 'national' else 34})": KNeighborsRegressor(n_neighbors=2 if kind == "national" else 34),
            "SVR(C=1, eps=0.1 표준화)": TransformedTargetRegressor(SVR(C=1.0, epsilon=0.1), transformer=StandardScaler()),
            f"DecisionTree(depth3, leaf{leaf})": DecisionTreeRegressor(max_depth=3, min_samples_leaf=leaf, random_state=seed),
            "XGBoost(200, depth2, lr0.05)": xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                            colsample_bytree=0.8, random_state=seed, n_jobs=4, verbosity=0),
            f"RandomForest(300, depth4, leaf{leaf})": RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=leaf,
                                                                            random_state=seed, n_jobs=4),
        }

    xm_rows = []
    for h in hs:
        # 전국 B 입력
        frame = nf.copy()
        frame[F.SIGN_C] = frame[f"G{h}"]
        ycol = f"G{h}"
        for name, est in extra_models("national", h).items():
            ct = M.features(s, spec, F.A_NAT, common_cols=ccols)
            low_dim = name.startswith(("KNN", "SVR"))
            pipe = Pipeline([("features", ct), ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", StandardScaler()), ("model", est)]).set_output(transform="pandas")
            preds = []
            for Tn in origins:
                tr = frame[(frame.index >= first) & (frame.index <= Tn - h) & frame[ycol].notna() & frame[F.A_NAT].notna().all(axis=1)]
                te = frame.loc[[Tn]]
                Xtr, Xte = tr.drop(columns=[ycol]), te.drop(columns=[ycol])
                p = clone(pipe).fit(Xtr, tr[ycol])
                preds.append({"P": Tn, "y": float(te[ycol].iloc[0]), "yhat": float(p.predict(Xte)[0])})
            L = monthly_loss(pd.DataFrame(preds)).mean()
            xm_rows.append({"과제": "RQ1 전국 B 입력", "h": h, "모형": name + (" (저차원: A+RMPI)" if low_dim else ""), "MAE": L})
        log(f"추가 모형 전국 h={h}")
        # 패널 r, B_지역 입력
        fr = pf.copy()
        fr[F.SIGN_R], fr[F.SIGN_C] = fr[f"r{h}"], fr[f"Gnat{h}"]
        ycol = f"r{h}"
        for name, est in extra_models("panel", h).items():
            ct = M.features(s, spec, F.A_REL, regional_cols=dcols, prefix_d="D|")
            pipe = Pipeline([("features", ct), ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", StandardScaler()), ("model", est)]).set_output(transform="pandas")
            preds = []
            for Tn in origins:
                tr = fr[(fr.P >= first) & (fr.P <= Tn - h) & fr[ycol].notna() & fr[F.A_REL].notna().all(axis=1)]
                te = fr[(fr.P == Tn) & fr[F.A_REL].notna().all(axis=1)]
                p = clone(pipe).fit(tr.drop(columns=[ycol]), tr[ycol])
                yh = p.predict(te.drop(columns=[ycol]))
                yh = yh - yh.mean()
                preds.append(pd.DataFrame({"P": te.P.values, "y": te[ycol].values, "yhat": yh}))
            L = monthly_loss(pd.concat(preds)).mean()
            xm_rows.append({"과제": "RQ2 패널 B_지역 입력(월내 평균 0 보정)", "h": h, "모형": name, "MAE": L})
        log(f"추가 모형 패널 h={h}")
    XM = pd.DataFrame(xm_rows)
    XM.to_csv(os.path.join(out, "stage7_추가모형.csv"), index=False, encoding="utf-8-sig")

    # ============================================================ G. 팀 보고서 비교
    NAT = pd.read_csv(os.path.join(out, "stage6_예측값_전국.csv"))
    PAN = pd.read_csv(os.path.join(out, "stage6_예측값_패널.csv"))
    for d_ in (NAT, PAN):
        d_["P"] = pd.PeriodIndex(d_["P"], freq="M")
    th_rows = []
    o21 = origins[origins.year >= 2021]
    G6 = pf[["region", "P", "G6", "Gbar6"]]
    def panel_stats(d, label, po):
        d = d[d.P.isin(po)].merge(G6, on=["region", "P"]).dropna(subset=["y", "yhat"])
        L = d.assign(l=(d.yhat - d.y).abs()).groupby("P")["l"].mean().mean()
        Lc = (d.groupby("P")["yhat"].mean() - d.groupby("P")["Gbar6"].first()).abs().mean()
        Lw = d.assign(w=((d.yhat - d.groupby("P")["yhat"].transform("mean")) - (d.y - d.Gbar6)).abs()).groupby("P")["w"].mean().mean()
        return {"모형": label, "시도별 MAE": L, "지역평균(공통) MAE": Lc, "월내차이 MAE": Lw}
    for lab, po in (("2021~2025", o21), ("2018~2025", origins)):
        for m, name in (("ridge_B", "통합 패널 Ridge B(A+공통·지역 RMPI)"), ("ridge_A", "통합 패널 Ridge A"), ("naive_mom1", "단순 mom1(지역별)"),
                        ("naive_zero", "단순 변화율 0")):
            d = PAN[(PAN["모형"] == m) & (PAN.h == 6) & (PAN["타깃"] == "G")]
            th_rows.append({"기간": lab, "출처": "본 연구(h=6)", **panel_stats(d, name, po)})
        rB = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == 6) & (PAN["타깃"] == "r")]
        gB = NAT[(NAT["모형"] == "ridge_B_Gbar") & (NAT.h == 6)].set_index("P")["yhat"]
        d = rB[["region", "P", "yhat"]].merge(G6[["region", "P", "G6"]], on=["region", "P"])
        d["yhat"] = d["P"].map(gB) + d["yhat"]
        d = d.rename(columns={"G6": "y"})[["region", "P", "y", "yhat"]]
        th_rows.append({"기간": lab, "출처": "본 연구(h=6)", **panel_stats(d, "결합 B(Ḡ̂_B + r̃_B)", po)})
    th_rows += [
        {"기간": "2021~2025", "출처": "TH v9 보고(17개 시도, 6개월 변화율, 기준 전체형)", "모형": "Extra Trees", "시도별 MAE": 0.824, "지역평균(공통) MAE": 0.473, "월내차이 MAE": 0.772},
        {"기간": "2021~2025", "출처": "TH v9 보고", "모형": "Ridge", "시도별 MAE": 0.929},
        {"기간": "2021~2025", "출처": "TH v9 보고", "모형": "Random Forest", "시도별 MAE": 0.831},
        {"기간": "2021~2025", "출처": "TH v9 보고", "모형": "XGBoost", "시도별 MAE": 0.857},
        {"기간": "2021~2025", "출처": "TH v9 보고", "모형": "전체 학습 중앙값 기준", "시도별 MAE": 1.170},
        {"기간": "2021~2025", "출처": "TH v9 보고", "모형": "차이 0 기준(월내차이)", "월내차이 MAE": 0.791},
        {"기간": "2018~2025", "출처": "TH v9 보고(시차 추가형)", "모형": "Extra Trees", "시도별 MAE": 0.903},
        {"기간": "2018~2025", "출처": "TH v9 보고(시차 추가형)", "모형": "Ridge", "시도별 MAE": 0.953},
    ]
    TH = pd.DataFrame(th_rows)
    TH.to_csv(os.path.join(out, "stage7_팀보고서비교.csv"), index=False, encoding="utf-8-sig")

    # ============================================================ 그림 7-1: 선형 SHAP 묶음별 기여 (h=3)
    h = 3
    d = SH[(SH.h == h) & (SH["묶음"] != "절편")].copy()
    d["P"] = pd.PeriodIndex(d["P"], freq="M")
    piv = d.groupby(["P", "묶음"])["기여"].sum().unstack().fillna(0)
    order = ["A(가격 추세)"] + [f"공통 {DRIVER_MARK[k]}" for k in range(1, 7)] + ["결측표시"]
    piv = piv[[c for c in order if c in piv.columns]]
    y_act = NAT[(NAT["모형"] == "ridge_B") & (NAT.h == h) & NAT.P.isin(origins)].set_index("P")[["y", "yhat"]]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = piv.index.to_timestamp()
    pos = np.zeros(len(piv))
    neg = np.zeros(len(piv))
    for i, c in enumerate(piv.columns):
        v = piv[c].values
        up = np.where(v > 0, v, 0)
        dn = np.where(v < 0, v, 0)
        ax.fill_between(x, pos, pos + up, color=PAL8[i % 8], alpha=0.85, lw=0, label=c)
        ax.fill_between(x, neg, neg + dn, color=PAL8[i % 8], alpha=0.85, lw=0)
        pos, neg = pos + up, neg + dn
    base_line = SH[(SH.h == h) & (SH["묶음"] == "절편")].set_index(pd.PeriodIndex(SH[(SH.h == h) & (SH["묶음"] == "절편")]["P"], freq="M"))["기여"]
    ax.plot(x, (y_act["yhat"] - base_line.reindex(y_act.index)).values, color=V.INK, lw=1.4, label="예측 − 훈련 평균")
    ax.axhline(0, color=V.BASE, lw=0.8)
    ax.set_title("전국 Ridge B 예측의 묶음별 기여 (h=3, 선형 SHAP = 계수 x 표준화 입력, 매달 재적합 모형 기준)", fontsize=10)
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_1_SHAP_전국.png"))
    plt.close(fig)

    S.manifest(s, {"stage": 7}).to_csv(os.path.join(out, "stage7_manifest.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(imp.to_string())
    print(pd.DataFrame(rem_rows).pivot_table(index=["과제", "제거 동인"], columns="h", values="변화").round(4).to_string())
    print(GU.round(3).to_string(index=False))
    print(XM.pivot_table(index=["과제", "모형"], columns="h", values="MAE").round(4).to_string())
    print(TH.round(3).to_string(index=False))
    log("완료")


if __name__ == "__main__":
    main()
