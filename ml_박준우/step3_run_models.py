# -*- coding: utf-8 -*-
"""
[3단계] 기준모형 대비 회귀·머신러닝 모형 비교 (17시도, 6개월 누적 월세 변화율)

입력: output/02_모델입력패널.csv
검증: 시간순 확장 창. 시험연도 Y 의 학습 = 결정월 t <= (Y-1)년 7월 (목표 구간이 시험 구간과 겹치지 않게)
지역 고정효과: 학습 구간의 지역 평균을 y 와 설명변수에서 뺀다(within 변환). OLS 에서는 지역 더미와 같은 결과이고,
              모든 모형에 같은 방식으로 적용돼 규제·트리 모형에서도 고정효과가 벌점 없이 들어간다
결측: 지역 평균 제거 후 0(=지역 평균)으로 채움
조정: 하이퍼파라미터는 학습 구간 안에서만 시간순 3겹 검증(검증 블록 8개월, 간격 6개월)으로 고른다

비교 대상
  참고 0  변화없음          : 예측 = 0
  참고 1  관성 그대로        : 예측 = 직전 6개월 변화율
  기준    모멘텀+지역FE (OLS): 이후 모든 비교의 기준
  확장    전체 변수 + 지역FE : OLS, Ridge, Lasso, ElasticNet, PCR, PLS, SVR, RandomForest, ExtraTrees, GBM
  참고    평균 앙상블         : 확장 10개 모형 예측의 단순 평균
  민감도  기준+1개월 변화율   : 기준모형에 직전 1개월 변화율을 더했을 때

출력: output/03_*.csv, output/03_*.png
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from spec import H, MOM, OUT, TEST_YEARS

warnings.filterwarnings("ignore")
MOM1 = "월세_직전1개월변화율"
BASELINE = "기준: 모멘텀+지역FE"
SEED = 0


def scaled(est):
    return make_pipeline(StandardScaler(), est)


# 이름 -> (모형 생성 함수, 탐색 격자)
MODELS = {
    "OLS": (lambda: LinearRegression(), {}),
    "Ridge": (lambda alpha: scaled(Ridge(alpha=alpha)), {"alpha": list(np.logspace(0, 4, 13))}),
    "Lasso": (lambda alpha: scaled(Lasso(alpha=alpha, max_iter=50000)), {"alpha": list(np.logspace(-3, 0, 13))}),
    "ElasticNet": (lambda alpha, l1_ratio: scaled(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000)),
                   {"alpha": list(np.logspace(-3, 0.5, 8)), "l1_ratio": [0.2, 0.5, 0.8]}),
    "PCR": (lambda n: make_pipeline(StandardScaler(), PCA(n_components=n, random_state=SEED), LinearRegression()),
            {"n": [1, 2, 3, 5, 8, 12, 16]}),
    "PLS": (lambda n: PLSRegression(n_components=n, scale=True), {"n": [1, 2, 3, 4, 6, 8]}),
    "SVR": (lambda C, epsilon: scaled(SVR(kernel="rbf", C=C, epsilon=epsilon, gamma="scale")),
            {"C": [0.3, 1, 3, 10], "epsilon": [0.05, 0.2]}),
    "RandomForest": (lambda max_features, min_samples_leaf: RandomForestRegressor(
        n_estimators=300, max_features=max_features, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=SEED),
        {"max_features": [0.3, 0.6], "min_samples_leaf": [5, 20]}),
    "ExtraTrees": (lambda max_features, min_samples_leaf: ExtraTreesRegressor(
        n_estimators=300, max_features=max_features, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=SEED),
        {"max_features": [0.3, 0.6], "min_samples_leaf": [5, 20]}),
    "GBM": (lambda n_estimators, max_depth: GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05, subsample=0.8, min_samples_leaf=10,
        random_state=SEED), {"n_estimators": [100, 300], "max_depth": [2, 3]}),
}


def predict(model, X):
    return np.ravel(model.predict(X))


def inner_folds(months, n_folds=3, block=8, gap=H):
    """학습 구간 안의 시간순 검증. 마지막 n_folds x block 개월을 블록으로 나누고, 각 블록 앞 gap 개월은 학습에서 뺀다"""
    u = np.sort(np.unique(months))
    for i in range(n_folds):
        a = len(u) - (n_folds - i) * block
        val = u[a:a + block]
        tr_last = u[a - gap]              # 이 달까지의 목표값은 검증 블록 시작 전에 확정
        yield np.where(months <= tr_last)[0], np.where(np.isin(months, val))[0]


def tune(make, grid, X, y, months):
    if not grid:
        return {}
    folds = list(inner_folds(months))
    best, best_mse = None, np.inf
    for params in ParameterGrid(grid):
        se, n = 0.0, 0
        for tr, va in folds:
            m = make(**params).fit(X[tr], y[tr])
            se += ((y[va] - predict(m, X[va])) ** 2).sum()
            n += len(va)
        if se / n < best_mse:
            best, best_mse = params, se / n
    return best


def within(tr, te, cols):
    """학습 구간 지역 평균 제거. 반환: (X_tr, X_te, y_tr, 시험 행의 y 지역 평균)"""
    mu = tr.groupby("region")[cols + ["y"]].mean()
    Xtr = (tr[cols] - mu.loc[tr.region, cols].values).fillna(0).values
    Xte = (te[cols] - mu.loc[te.region, cols].values).fillna(0).values
    return Xtr, Xte, (tr["y"] - mu.loc[tr.region, "y"].values).values, mu.loc[te.region, "y"].values


def perm_importance(model, X, y_dm, cols, rng, repeats=5):
    """시험 구간에서 변수 하나를 섞었을 때 RMSE 증가분"""
    base = np.sqrt(np.mean((y_dm - predict(model, X)) ** 2))
    out = {}
    for j, c in enumerate(cols):
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            inc.append(np.sqrt(np.mean((y_dm - predict(model, Xp)) ** 2)) - base)
        out[c] = float(np.mean(inc))
    return out


def dm_test(e_base, e_model, months, h=H):
    """Diebold-Mariano (제곱오차). 월별로 지역 평균 손실차를 내고, 겹치는 예측 기간을 Newey-West(h-1)로 보정.
    Harvey-Leybourne-Newbold 소표본 보정. 통계량 > 0 이면 모형이 기준보다 나음"""
    d = pd.Series(e_base ** 2 - e_model ** 2).groupby(np.asarray(months)).mean().values
    T, dbar = len(d), d.mean()
    u = d - dbar
    var = (u @ u) / T
    for k in range(1, h):
        var += 2 * (1 - k / h) * (u[k:] @ u[:-k]) / T
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), T - 1))


def metrics(y, p):
    e = y - p
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "평균오차(실제-예측)": e.mean(),
            "상관계수": np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan,
            "방향일치율": np.mean(np.sign(y) == np.sign(p)) if np.std(p) > 0 else np.nan}


def main():
    d = pd.read_csv(os.path.join(OUT, "02_모델입력패널.csv"))
    feats = [c for c in d.columns if c not in ("region", "month", "y", MOM, MOM1)]
    full = [MOM] + feats
    rng = np.random.default_rng(SEED)
    print(f"표본 {len(d)}행 ({d.month.min()}~{d.month.max()}), 설명변수 {len(feats)}개 + 모멘텀")

    preds, tuned, imps, coefs = [], [], [], []
    for yr in TEST_YEARS:
        tr = d[d.month <= (yr - 1) * 100 + 7].reset_index(drop=True)
        te = d[(d.month >= yr * 100 + 1) & (d.month <= yr * 100 + 12)].reset_index(drop=True)
        rec = te[["region", "month", "y"]].copy()
        rec.insert(0, "시험연도", yr)
        rec["참고: 변화없음(0)"] = 0.0
        rec["참고: 관성 그대로"] = te[MOM].values

        Xtr, Xte, ytr, mu_te = within(tr, te, [MOM])
        m = LinearRegression().fit(Xtr, ytr)
        rec[BASELINE] = predict(m, Xte) + mu_te
        coefs.append({"시험연도": yr, "학습 행수": len(tr), "학습 구간": f"{tr.month.min()}~{tr.month.max()}",
                      "모멘텀 계수": m.coef_[0]})
        Xtr, Xte, ytr, mu_te = within(tr, te, [MOM, MOM1])
        rec["민감도: 기준+1개월변화율"] = predict(LinearRegression().fit(Xtr, ytr), Xte) + mu_te

        Xtr, Xte, ytr, mu_te = within(tr, te, full)
        yte_dm = te["y"].values - mu_te
        for name, (make, grid) in MODELS.items():
            params = tune(make, grid, Xtr, ytr, tr.month.values)
            m = make(**params).fit(Xtr, ytr)
            rec[name] = predict(m, Xte) + mu_te
            tuned.append({"시험연도": yr, "모형": name, "선택값": str({k: (round(float(v), 4)) for k, v in params.items()})})
            for c, v in perm_importance(m, Xte, yte_dm, full, rng).items():
                imps.append({"시험연도": yr, "모형": name, "변수": c, "RMSE 증가": v})
            print(f"  {yr} {name:13s} {params}", flush=True)
        rec["참고: 평균 앙상블"] = rec[list(MODELS)].mean(axis=1)
        preds.append(rec)

    P = pd.concat(preds, ignore_index=True)
    P.to_csv(os.path.join(OUT, "03_예측값.csv"), index=False, encoding="utf-8-sig")
    names = [c for c in P.columns if c not in ("시험연도", "region", "month", "y")]
    y = P["y"].values
    sse_b = ((y - P[BASELINE].values) ** 2).sum()

    rows = []
    for n in names:
        p = P[n].values
        r = {"모형": n, **metrics(y, p)}
        r["기준 대비 RMSE 비율"] = r["RMSE"] / np.sqrt(sse_b / len(y))
        r["기준 대비 R2(표본외)"] = 1 - ((y - p) ** 2).sum() / sse_b
        r["DM 통계량"], r["DM p값"] = (np.nan, np.nan) if n == BASELINE else dm_test(y - P[BASELINE].values, y - p, P.month)
        yr_rmse = P.assign(e=(y - p) ** 2).groupby("시험연도").e.mean() ** 0.5
        yr_base = P.assign(e=(y - P[BASELINE].values) ** 2).groupby("시험연도").e.mean() ** 0.5
        r["기준보다 나은 연도 수"] = int((yr_rmse < yr_base).sum()) if n != BASELINE else np.nan
        # 세종은 신도시 입주기라 변동이 유독 커서, 세종을 빼고 채점한 값을 따로 둔다(학습에는 포함)
        k = (P.region != "세종").values
        r["RMSE(세종 제외 채점)"] = np.sqrt(np.mean((y[k] - p[k]) ** 2))
        r["기준 대비 RMSE 비율(세종 제외 채점)"] = r["RMSE(세종 제외 채점)"] / np.sqrt(
            np.mean((y[k] - P[BASELINE].values[k]) ** 2))
        r["DM p값(세종 제외 채점)"] = np.nan if n == BASELINE else dm_test(
            (y - P[BASELINE].values)[k], (y - p)[k], P.month[k])[1]
        rows.append(r)
    S = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    S.insert(1, "시험 행수", len(y))
    S.to_csv(os.path.join(OUT, "03_성과_요약.csv"), index=False, encoding="utf-8-sig")

    def by(key):
        t = pd.DataFrame({n: P.assign(e=(y - P[n].values) ** 2).groupby(key).e.mean() ** 0.5 for n in names})
        t.insert(0, "y 표준편차", P.groupby(key).y.std())
        t.insert(0, "y 평균", P.groupby(key).y.mean())
        return t[["y 평균", "y 표준편차"] + list(S["모형"])]
    by("시험연도").to_csv(os.path.join(OUT, "03_성과_연도별RMSE.csv"), encoding="utf-8-sig")
    by("region").to_csv(os.path.join(OUT, "03_성과_지역별RMSE.csv"), encoding="utf-8-sig")
    pd.DataFrame(tuned).to_csv(os.path.join(OUT, "03_선택_하이퍼파라미터.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(coefs).to_csv(os.path.join(OUT, "03_기준모형_계수.csv"), index=False, encoding="utf-8-sig")
    imp = pd.DataFrame(imps).groupby(["모형", "변수"])["RMSE 증가"].mean().unstack("모형")[list(MODELS)]
    imp.insert(0, "모형 평균", imp.mean(axis=1))
    imp.sort_values("모형 평균", ascending=False).to_csv(os.path.join(OUT, "03_변수중요도_순열.csv"), encoding="utf-8-sig")
    plot(P, S)

    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + S.round(3).to_string(index=False))
    print("\n연도별 RMSE\n" + by("시험연도").round(3).T.to_string())


def plot(P, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    best = [n for n in S["모형"] if n in MODELS][0]
    m = P.groupby("month")[["y", BASELINE, best]].mean()
    x = pd.to_datetime(m.index.astype(str), format="%Y%m")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(x, m["y"], color="black", lw=2, label="실제")
    ax[0].plot(x, m[BASELINE], color="#888888", lw=1.5, ls="--", label=BASELINE)
    ax[0].plot(x, m[best], color="#c0392b", lw=1.5, label=f"확장: {best}")
    ax[0].axhline(0, color="#cccccc", lw=0.8)
    ax[0].set_title("17개 시도 평균: 6개월 누적 월세 변화율 (결정월 기준, 표본외)")
    ax[0].set_ylabel("%")
    ax[0].legend()
    t = S.sort_values("RMSE", ascending=False)
    colors = ["#888888" if n == BASELINE else "#c0392b" if n in MODELS else "#bbbbbb" for n in t["모형"]]
    ax[1].barh(t["모형"], t["RMSE"], color=colors)
    ax[1].axvline(float(S.loc[S["모형"] == BASELINE, "RMSE"].iloc[0]), color="black", lw=0.8, ls=":")
    ax[1].set_title("표본외 RMSE (2021~2025, 점선 = 기준모형)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_실제vs예측_RMSE.png"), dpi=130)


if __name__ == "__main__":
    main()
