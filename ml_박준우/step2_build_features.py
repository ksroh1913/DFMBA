# -*- coding: utf-8 -*-
"""
[2단계] 목표값과 설명변수 만들기 (변수 1개 = 변환 1개)

출력
  output/02_모델입력패널.csv   region, month, y, 모멘텀, 설명변수 (분석 표본만)
  output/02_사용변수_정리.csv  변수명·기본 정보·전처리 방법·사유·기초통계·정상성 점검·상관 제거 여부
"""
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from spec import (CSI, H, HH, MAX_CORR, METHOD_LABEL, MOM, OUT, POP, SPEC, START, TEST_YEARS, YCOL, load,
                  region_type)

warnings.filterwarnings("ignore")


def transform(df, col, method):
    g = df.groupby("region")
    x = df[col].astype(float) if col in df else None
    if method == "level":
        return x
    if method in ("d6", "d12"):
        return x - g[col].shift(int(method[1:]))
    if method in ("dl6", "dl12"):
        lx = np.log(x.where(x > 0))
        return 100 * (lx - lx.groupby(df["region"]).shift(int(method[2:])))
    if method in ("sum12_per_hh", "sum12_per_pop"):
        s12 = g[col].transform(lambda s: s.rolling(12, min_periods=12).sum())
        return 1000 * s12 / df[HH if method.endswith("hh") else POP]
    if method == "sum3_yoy":
        s3 = np.log(g[col].transform(lambda s: s.rolling(3, min_periods=3).sum()).where(lambda v: v > 0))
        return 100 * (s3 - s3.groupby(df["region"]).shift(12))
    if method == "per_hh":
        return 1000 * x / df[HH]
    if method == "csi_match":
        kind = df["region"].map(region_type)
        return sum(df[c].where(kind == k, 0) for k, c in CSI.items()).astype(float)
    raise ValueError(method)


def adf_p(s):
    s = s.dropna()
    if len(s) < 40 or s.nunique() < 20:      # 연간 자료처럼 값이 거의 안 바뀌면 검정하지 않음
        return np.nan
    try:
        return adfuller(s, autolag="AIC")[1]
    except Exception:
        return np.nan


def main():
    raw, meta, fname = load()
    df = raw[raw.month >= START].sort_values(["region", "month"]).reset_index(drop=True)
    R = df[YCOL].where(df["Y_평가사용가능"] == 1)
    Rg = R.groupby(df["region"])
    out = df[["region", "month"]].copy()
    out["y"] = 100 * (Rg.shift(-(H - 1)) / Rg.shift(1) - 1)       # R(t+5)/R(t-1)
    out[MOM] = 100 * (Rg.shift(1) / Rg.shift(H + 1) - 1)          # R(t-1)/R(t-7)
    out["월세_직전1개월변화율"] = 100 * (Rg.shift(1) / Rg.shift(2) - 1)   # 기준모형 민감도 점검용
    for name, col, method, _ in SPEC:
        out[name] = transform(df, col, method)
    feats = [s[0] for s in SPEC]

    # 표본: 15개 이상 지역에서 모든 변수가 계산되는 첫 달부터, 목표값이 있는 마지막 달까지.
    # 세종(고용률·주택보급률·소비심리)과 제주(소비심리)는 초기 값이 없어, 그 빈칸은 모델 단계에서 지역 평균으로 채운다
    ok = out.groupby("month")[feats + [MOM]].apply(lambda d: (d.notna().sum() >= 15).all())
    first = ok[ok].index.min()
    panel = out[(out.month >= first) & out["y"].notna() & out[MOM].notna()].reset_index(drop=True)

    # 변환 후 상관이 지나치게 높은 변수 제거 (첫 학습 구간, 지역 평균 제거 후)
    tr = panel[panel.month <= (TEST_YEARS[0] - 1) * 100 + 7]
    w = tr[feats] - tr.groupby("region")[feats].transform("mean")
    corr = w.corr().abs()
    kept, dropped = [], {}
    for f in feats:
        hit = [k for k in kept if corr.loc[f, k] > MAX_CORR]
        if hit:
            dropped[f] = f"{hit[0]} 와 상관 {corr.loc[f, hit[0]]:.3f}"
        else:
            kept.append(f)

    rows = []
    for name, col, method, why in SPEC:
        m = meta.loc[col] if col in meta.index else meta.loc[CSI["서울"]]
        national = (panel.groupby("month")[name].nunique(dropna=True) <= 1).all()
        p = panel.groupby("region")[name].apply(adf_p)
        v = panel[name]
        rows.append({
            "모델 변수명": name, "원 컬럼": col, "Master ID": "V061~V063" if method == "csi_match" else m["Master ID"],
            "동인": m["동인"], "변수명(데이터사전)": m["변수명(데이터사전)"], "주기": m["주기"],
            "값의 단위": "전국 공통" if national else "지역별", "출처": m["출처(기관·표)"],
            "공표 시차(2차 시트)": m["2차 시트 시차 규칙"], "전처리 방법": METHOD_LABEL[method], "전처리 사유": why,
            "모델 사용": "제외" if name in dropped else "사용", "제외 사유": dropped.get(name, ""),
            "평균": round(v.mean(), 3), "표준편차": round(v.std(), 3), "최소": round(v.min(), 3), "최대": round(v.max(), 3),
            "결측률": round(v.isna().mean(), 4),
            "ADF p값(지역 중앙값)": round(p.median(), 3) if p.notna().any() else "",
            "ADF 10% 기각 지역 비율": round((p < 0.10).mean(), 2) if p.notna().any() else "",
        })
    head = {"모델 변수명": MOM, "원 컬럼": YCOL, "Master ID": "V001", "동인": meta.loc[YCOL, "동인"],
            "변수명(데이터사전)": meta.loc[YCOL, "변수명(데이터사전)"], "주기": "월", "값의 단위": "지역별",
            "출처": meta.loc[YCOL, "출처(기관·표)"], "공표 시차(2차 시트)": "목표값은 밀지 않음. 모델에서 R(t-1)까지만 사용",
            "전처리 방법": "직전 6개월 변화율(%) = 100 x (R(t-1)/R(t-7) - 1)",
            "전처리 사유": "가격 모멘텀. 목표와 같은 정의를 한 기간 앞당긴 값", "모델 사용": "사용(기준모형)", "제외 사유": "",
            "평균": round(panel[MOM].mean(), 3), "표준편차": round(panel[MOM].std(), 3),
            "최소": round(panel[MOM].min(), 3), "최대": round(panel[MOM].max(), 3), "결측률": 0.0,
            "ADF p값(지역 중앙값)": round(panel.groupby("region")[MOM].apply(adf_p).median(), 3),
            "ADF 10% 기각 지역 비율": round((panel.groupby("region")[MOM].apply(adf_p) < 0.10).mean(), 2)}
    table = pd.DataFrame([head] + rows)
    os.makedirs(OUT, exist_ok=True)
    table.to_csv(os.path.join(OUT, "02_사용변수_정리.csv"), index=False, encoding="utf-8-sig")
    panel[["region", "month", "y", MOM, "월세_직전1개월변화율"] + kept].to_csv(
        os.path.join(OUT, "02_모델입력패널.csv"), index=False, encoding="utf-8-sig")

    print(f"입력: {fname}")
    print(f"표본: {len(panel)}행, 결정월 {panel.month.min()}~{panel.month.max()}, {panel.region.nunique()}개 시도")
    print(f"설명변수 {len(feats)}개 중 사용 {len(kept)}개, 상관 제거 {len(dropped)}개")
    for k, v in dropped.items():
        print(f"  제외: {k}  ({v})")
    miss = panel[kept].isna().mean()
    print("남은 결측:", miss[miss > 0].round(4).to_dict())
    print(f"y: 평균 {panel.y.mean():.3f}, 표준편차 {panel.y.std():.3f}, 범위 {panel.y.min():.2f}~{panel.y.max():.2f}")


if __name__ == "__main__":
    main()
