# -*- coding: utf-8 -*-
"""손실, 두 t구간 판정, 블록 부트스트랩(참고), 사건 지표 (연구계획 8차 확정본 1·9장).

d(t) = L_추가정보(t) − L_기준(t). 음수일수록 추가정보 모형이 낫다. 패널은 같은 달 17개 시도 평균을 먼저 낸 월 단위 d.
구간 1: 결정연도 8개(2018~2025, 해마다 12개월)의 연평균 d 에 자유도 7 의 t분포.
구간 2: 2년 묶음 4개(2018~19, 20~21, 22~23, 24~25)의 평균에 자유도 3 의 t분포.
판정: 개선(두 상한 < 0) / 개선 신호(연평균 상한만 < 0) / 악화(두 하한 > 0) / 악화 신호(연평균 하한만 > 0) / 차이 불확실.
질문 요약: 세 기간 중 두 개 이상 개선이고 악화가 없으면 '지지', 하나만 개선이면 '기간 한정 개선(탐색)'. 개선 신호는 세지 않는다.
블록 부트스트랩(순환, 블록 6·12·18, 2,000회, 고정 난수)은 참고 결과이며 판정에 쓰지 않는다.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


# ------------------------------------------------------------ t구간
def _t_ci(group_means, level):
    k = len(group_means)
    m = float(np.mean(group_means))
    if k < 2:
        return m, np.nan, np.nan, k
    se = float(np.std(group_means, ddof=1) / np.sqrt(k))
    q = stats.t.ppf(0.5 + level / 2, k - 1)
    return m, m - q * se, m + q * se, k


def t_intervals(d, level=0.95):
    """d: 월 단위 손실 차이(Series, PeriodIndex 결정월). 반환 dict: mean, annual(lo,hi,k), biennial(lo,hi,k)"""
    d = pd.Series(d).dropna()
    yrs = d.index.year
    ann = d.groupby(yrs).mean()
    bi = d.groupby((yrs - yrs.min()) // 2).mean()
    m, alo, ahi, ka = _t_ci(ann.values, level)
    _, blo, bhi, kb = _t_ci(bi.values, level)
    return {"mean": m, "n_months": int(len(d)), "annual_lo": alo, "annual_hi": ahi, "annual_k": ka,
            "biennial_lo": blo, "biennial_hi": bhi, "biennial_k": kb,
            "years_better": int((ann < 0).sum()), "years_total": int(len(ann))}


def verdict(ti):
    a_lo, a_hi, b_lo, b_hi = ti["annual_lo"], ti["annual_hi"], ti["biennial_lo"], ti["biennial_hi"]
    if np.isnan(a_hi):
        return "판정 불가"
    if a_hi < 0 and b_hi < 0:
        return "개선"
    if a_hi < 0:
        return "개선 신호"
    if a_lo > 0 and b_lo > 0:
        return "악화"
    if a_lo > 0:
        return "악화 신호"
    return "차이 불확실"


def summarize_question(verdicts_by_h):
    v = list(verdicts_by_h.values())
    n_imp = sum(x == "개선" for x in v)
    if any(x == "악화" for x in v):
        return "지지 안 됨(악화 포함)"
    if n_imp >= 2:
        return "지지"
    if n_imp == 1:
        return "기간 한정 개선(탐색)"
    return "지지 안 됨"


# ------------------------------------------------------------ 블록 부트스트랩 (참고)
def circular_block_bootstrap(d, block, n_boot=2000, seed=0, level=0.95):
    d = pd.Series(d).dropna().values
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        means[b] = d[idx].mean()
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi), float((means < 0).mean())


# ------------------------------------------------------------ 서술 지표
def mae_ratio(loss_model, loss_ref):
    return float(np.nanmean(loss_model) / np.nanmean(loss_ref))


def yearly_table(d):
    d = pd.Series(d).dropna()
    return d.groupby(d.index.year).agg(["mean", "count"]).rename(columns={"mean": "연평균 d", "count": "월수"})


# ------------------------------------------------------------ 사건 지표
def event_metrics(y, p, cutoff, p_ref=None):
    """y: 0/1, p: 확률, cutoff: 사전 지정 컷오프(훈련 사건비율), p_ref: 기준 확률(훈련 사건비율). 정의 불가는 NaN."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    out = {"행수": int(len(y)), "양성": int(y.sum()), "사건비율": float(y.mean()) if len(y) else np.nan}
    if not len(y):
        return out
    out["Brier"] = float(np.mean((p - y) ** 2))
    if p_ref is not None:
        bref = float(np.mean((p_ref - y) ** 2))
        out["Brier_기준"] = bref
        out["Brier_skill"] = 1 - out["Brier"] / bref if bref > 0 else np.nan
    out["평균예측확률-실현빈도"] = float(p.mean() - y.mean())
    out["AP"] = float(average_precision_score(y, p)) if y.sum() > 0 else np.nan
    out["ROC_AUC"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    pred = (p >= cutoff).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    out["정밀도"] = tp / (tp + fp) if tp + fp > 0 else np.nan
    out["재현율"] = tp / (tp + fn) if tp + fn > 0 else np.nan
    out["F1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan
    out["F1_항상사건"] = 2 * y.sum() / (y.sum() + len(y)) if y.sum() > 0 else np.nan
    return out


# ------------------------------------------------------------ v9 ④ 경보 실용성 지표
def within_month_auc(df, ycol="y", pcol="yhat", month="P", min_regions=10):
    """같은 달 안에서 사건 지역과 비사건 지역의 예측확률 순위를 가르는 정도: 월별 ROC-AUC 의 평균(두 계급이 모두 있는 달만)"""
    vals = []
    for _, g in df.groupby(month):
        y = g[ycol].astype(float)
        if len(g) >= min_regions and 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, g[pcol]))
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def episode_lead_times(df, cut_col, ycol="y", pcol="yhat", quiet=6, window=6):
    """국면 시작(지역별로 quiet 개 결정월 연속 비사건 뒤 첫 사건) 마다, 시작 전 window 개월 안에서 처음 경보가 켜진 시점까지의 개월 수.
    반환 DataFrame(region, 시작, 경보첫시점, 선행개월; 경보 없으면 NaN)"""
    rows = []
    for reg, g in df.sort_values("P").groupby("region"):
        g = g.set_index("P")
        y = g[ycol].astype(float)
        prev = y.shift(1).rolling(quiet, min_periods=quiet).max()
        starts = g.index[(y == 1) & (prev == 0)]
        alert = (g[pcol] >= g[cut_col])
        for s0 in starts:
            win = alert.loc[s0 - (window - 1): s0]
            on = win[win].index
            rows.append({"region": reg, "시작": str(s0), "경보첫시점": str(on.min()) if len(on) else "",
                         "선행개월": int((s0 - on.min()).n) if len(on) else np.nan})
    return pd.DataFrame(rows)


def alert_metrics(df, cut_col, ycol="y", pcol="yhat", state_col="기존급락상태"):
    """비급락 상태 지역을 대상으로 한 경보 지표. df: 한 모형의 패널 예측(P, region, y, yhat, 컷오프 열, 기존급락상태)."""
    d = df[df[state_col] == 0] if state_col in df.columns else df
    d = d[d[ycol].notna() & d[pcol].notna() & d[cut_col].notna()]
    y = d[ycol].astype(float).values
    a = (d[pcol].values >= d[cut_col].values).astype(float)
    tp = float(((a == 1) & (y == 1)).sum()); fp = float(((a == 1) & (y == 0)).sum()); fn = float(((a == 0) & (y == 1)).sum())
    out = {"행수": int(len(d)), "사건수": int(y.sum()), "경보수": int(a.sum()), "경보빈도_실제": float(a.mean()) if len(d) else np.nan,
           "경보적중률": tp / (tp + fp) if tp + fp > 0 else np.nan, "급락포착률": tp / (tp + fn) if tp + fn > 0 else np.nan}
    out["F1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else np.nan
    auc, n_m = within_month_auc(d, ycol, pcol)
    out["동월지역쌍_AUC"], out["AUC_유효월수"] = auc, n_m
    lead = episode_lead_times(df, cut_col, ycol, pcol)
    if len(lead):
        out["국면시작수"] = int(len(lead)); out["국면포착률"] = float(lead["선행개월"].notna().mean())
        out["평균선행개월(포착분)"] = float(lead["선행개월"].mean()) if lead["선행개월"].notna().any() else np.nan
    return out, lead
