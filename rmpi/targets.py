# -*- coding: utf-8 -*-
"""타깃·사건·대용계열 (연구계획 8차 확정본 4·5장).

결정월 t: t월 말. 그때 관측되는 마지막 월세 수준은 R(t-1).
  G(i,t,h) = 100 x [R(i,t-1+h) / R(i,t-1) - 1]        정답은 t+h 월 말에 확인
  Ḡ(t,h)  = 17개 시도 G 의 단순평균 (17개가 모두 있는 달만)
  r(i,t,h) = G - Ḡ
  과거 월세 입력 A: 100 x [R(t-1)/R(t-1-k) - 1], k = 1, 3, 6
사건: 급락 G <= 100x[(0.98)^(h/12)-1], 급등 G >= 100x[(1.03)^(h/12)-1]
상태: 직전 h개월 변화 100x[R(t-1)/R(t-1-h)-1] 이 급락 경계 이하이면 '기존 급락 상태'
"""

import numpy as np
import pandas as pd

from . import settings as S


def thresholds(s, mult=1.0):
    """h -> (급락 경계, 급등 경계). mult 는 연율 경계의 배수(민감도 0.75·1.25)."""
    ev = s["events"]
    return {h: (100 * ((1 + ev["down_annual"] * mult) ** (h / 12) - 1),
                100 * ((1 + ev["up_annual"] * mult) ** (h / 12) - 1)) for h in s["timing"]["horizons"]}


# ============================================================ 시도 패널
def regional(s, ml, panel="sido"):
    """ml(2차, region·P 정렬) -> 패널 표: R(공식), G·past·사건·상태 (h별)."""
    ycol, flag = s["inputs"]["target"][panel], s["inputs"]["official_flag"]
    R = ml[ycol].where(ml[flag] == 1).astype(float)
    g = R.groupby(ml["region"])
    out = ml[["region", "P"]].copy()
    out["R_last"] = g.shift(1).values                       # R(t-1)
    for k in (1, 3, 6):
        out[f"past{k}"] = (100 * (g.shift(1) / g.shift(1 + k) - 1)).values
    th = thresholds(s)
    n_reg = out["region"].nunique()
    for h in s["timing"]["horizons"]:
        G = 100 * (g.shift(1 - h) / g.shift(1) - 1)
        out[f"G{h}"] = G.values
        wide = out.pivot(index="P", columns="region", values=f"G{h}")
        gbar = wide.mean(axis=1).where(wide.notna().sum(axis=1) == n_reg)
        out[f"Gbar{h}"] = out["P"].map(gbar).values
        out[f"r{h}"] = out[f"G{h}"] - out[f"Gbar{h}"]
        dn, up = th[h]
        out[f"down{h}"] = np.where(out[f"G{h}"].notna(), (out[f"G{h}"] <= dn).astype(float), np.nan)
        out[f"up{h}"] = np.where(out[f"G{h}"].notna(), (out[f"G{h}"] >= up).astype(float), np.nan)
        prior = 100 * (g.shift(1) / g.shift(1 + h) - 1)
        out[f"state_down{h}"] = np.where(prior.notna(), (prior <= dn).astype(float), np.nan)
        out[f"state_up{h}"] = np.where(prior.notna(), (prior >= up).astype(float), np.nan)      # 결정월에 이미 급등 상태(직전 h개월 변화가 경계 이상)
        out[f"label_month{h}"] = (out["P"] + h).values     # 정답 확인 시점(그 달 말)
    return out


def gbar_frame(panel_targets, s):
    """월 단위 Ḡ 표 (결합 예측용 전국 모형의 타깃)."""
    cols = [f"Gbar{h}" for h in s["timing"]["horizons"]]
    return panel_targets.drop_duplicates("P").set_index("P")[cols].sort_index()


# ============================================================ 전국 계열 (R-ONE 원자료)
def rone(path, name):
    df = pd.read_csv(path)
    x = df[df.CLS_NM == name]
    cnt = x.groupby("WRTTIME_IDTFR_ID").size()
    if (cnt > 1).any():
        raise ValueError(f"{name}: 한 달에 행 {cnt.max()}개")
    s_ = x.set_index("WRTTIME_IDTFR_ID")["DTA_VAL"].astype(float)
    s_.index = pd.PeriodIndex([f"{int(m) // 100}-{int(m) % 100:02d}" for m in s_.index], freq="M")
    return s_.sort_index()


def national(s, months=None):
    """결정월 기준 전국 표: R_last, G{h}, past1·3·6, 매매·전세 전국 6개월 변화율(t-1 기준).
    months: 결정월 PeriodIndex (기본 2011-01 ~ 공식 정답 끝 + 6)."""
    R = rone(S.path(s, "raw_rent"), "전국")
    sale = rone(S.path(s, "raw_sale"), "전국")
    jeon = rone(S.path(s, "raw_jeonse"), "전국")
    if months is None:
        months = pd.period_range("2011-01", R.index.max() + 6, freq="M")
    out = pd.DataFrame(index=months)
    out.index.name = "P"
    lag1 = out.index - 1
    out["R_last"] = R.reindex(lag1).values
    for k in (1, 3, 6):
        out[f"past{k}"] = 100 * (R.reindex(lag1).values / R.reindex(lag1 - k).values - 1)
    out["V002|6개월%"] = 100 * (sale.reindex(lag1).values / sale.reindex(lag1 - 6).values - 1)
    out["V026|6개월%"] = 100 * (jeon.reindex(lag1).values / jeon.reindex(lag1 - 6).values - 1)
    th = thresholds(s)
    for h in s["timing"]["horizons"]:
        out[f"G{h}"] = 100 * (R.reindex(lag1 + h).values / R.reindex(lag1).values - 1)
        out[f"down{h}"] = np.where(np.isfinite(out[f"G{h}"]), (out[f"G{h}"] <= th[h][0]).astype(float), np.nan)
        out[f"label_month{h}"] = out.index + h
        # 민감도 타깃 g: R(t) 기준 순수 미래 변화. 정답은 t+h+1 월 말에 확인 (훈련행 t <= T-h-1)
        out[f"g{h}"] = 100 * (R.reindex(out.index + h).values / R.reindex(out.index).values - 1)
    return out


# ============================================================ 대용계열 (옛 지수 접합)
def proxy_level(s, which="main"):
    """옛 지수 집계를 2015.06 공식 전국 수준에 맞춰 이은 월세 수준 계열(달력 월 기준).
    main: 옛 수도권 집계. sensitivity: 수도권·5대광역시·8개시도 로그변화의 동일 가중 평균(2012.05~)."""
    R = rone(S.path(s, "raw_rent"), "전국")
    J = S.per(s["timing"]["splice_month"])
    old = {n: rone(S.path(s, "raw_old_rent"), n) for n in s["timing"]["proxy_sensitivity"]}
    if which == "main":
        o = old[s["timing"]["proxy_main"]]
        lvl = o * (R.loc[J] / o.loc[J])
    else:
        dl = pd.concat([np.log(v).diff() for v in old.values()], axis=1).dropna()
        cum = dl.mean(axis=1)[::-1].cumsum()[::-1]            # J 까지 거꾸로 누적
        lvl = pd.Series(np.nan, index=dl.index.union([J]))
        lvl.loc[J] = R.loc[J]
        # level(t) = level(J) / exp(sum of mean log-changes from t+1..J)
        back = dl.mean(axis=1)
        acc = 0.0
        for p in sorted(back.index[back.index <= J], reverse=True):
            acc += back.loc[p]
            lvl.loc[p - 1] = R.loc[J] / np.exp(acc)
        lvl = lvl.dropna().sort_index()
    out = R.copy()
    pre = lvl[lvl.index < J]
    return pd.concat([pre, out]).sort_index()


def system_of(a, b, J):
    """지수 수준 두 시점 a<b 로 계산하는 구간 [a,b] 의 체계"""
    return "old" if b <= J else ("new" if a >= J else "cross")


def splice_table(s, h, months):
    """결정월별 접합 판정: 목표 구간 [t-1, t-1+h], A 구간 [t-7, t-1] 이 같은 체계일 때만 사용."""
    J = S.per(s["timing"]["splice_month"])
    off0, ext0 = S.per(s["timing"]["official_first_decision"]), S.per(s["timing"]["extension_first_decision"])
    rows = []
    for t in months:
        tgt, mom = system_of(t - 1, t - 1 + h, J), system_of(t - 7, t - 1, J)
        if tgt == mom == "new" and t >= off0:
            k = "공식"
        elif tgt == mom == "old" and t >= ext0:
            k = "옛"
        else:
            k = "제외"
        rows.append({"h": h, "P": t, "목표구간": tgt, "A구간": mom, "판정": k})
    return pd.DataFrame(rows)


def extension_frame(s, h, which="main"):
    """확장 실험용 전국 표(결정월 기준): 대용계열로 R_last·past·G 를 만들고 접합 판정을 붙인다.
    공식 체계 행의 값은 national() 과 같다."""
    lvl = proxy_level(s, which)
    J = S.per(s["timing"]["splice_month"])
    months = pd.period_range(S.per(s["timing"]["extension_first_decision"]) - 12, lvl.index.max() + 6, freq="M")
    out = pd.DataFrame(index=months)
    out.index.name = "P"
    lag1 = out.index - 1
    out["R_last"] = lvl.reindex(lag1).values
    for k in (1, 3, 6):
        out[f"past{k}"] = 100 * (lvl.reindex(lag1).values / lvl.reindex(lag1 - k).values - 1)
    out[f"G{h}"] = 100 * (lvl.reindex(lag1 + h).values / lvl.reindex(lag1).values - 1)
    sp = splice_table(s, h, months).set_index("P")
    out["판정"] = sp["판정"].values
    out["옛체계"] = (out["판정"] == "옛").astype(int)
    if which != "main":   # 지방 집계는 2012.05 부터 -> A 구간 때문에 결정월 2012.12 부터
        first = pd.Period("2012-05", "M") + 7
        out.loc[(out.index < first) & (out["판정"] == "옛"), "판정"] = "제외"
        out["옛체계"] = (out["판정"] == "옛").astype(int)
    out["label_month"] = out.index + h
    return out


# ============================================================ 국면 시작
def episode_starts(y, quiet):
    """사건 열(0/1, NaN 가능)의 시계열에서 quiet 개 결정월 연속 비사건 뒤 처음 사건이 된 위치(bool Series)."""
    y = pd.Series(y).astype(float)
    prev = y.shift(1).rolling(quiet, min_periods=quiet).sum()
    return (y == 1) & (prev == 0)
