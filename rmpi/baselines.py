# -*- coding: utf-8 -*-
"""단순 기준과 선택형 단순기준 (연구계획 8차 확정본 5·9장).

무학습 규칙(결정월 t, 마지막 수준 R(t-1)):
  zero   변화율 0
  mom1   h x 최근 월변화            = h x 100[R(t-1)/R(t-2) - 1]
  mom3   h x 최근 3개월 평균 월변화   = h x 평균{월변화(t-1), 월변화(t-2), 월변화(t-3)}
  lasth  최근 h개월 변화             = 100[R(t-1)/R(t-1-h) - 1]
학습 중앙값(train_median): 결정월 s 에 그때 확인된 공식 구간 정답(결정월 first ~ s-h)의 중앙값. 정답이 12개 이상일 때부터.
선택형(select_naive): 시점 T 의 평가창 = 정답이 확인된 공식 결정월 중 최근 최대 24개월. 평가창의 모든 달에 예측이 있는
  후보만 같은 달 집합에서 MAE 를 비교해 가장 낮은 후보. 동점이면 zero, mom1, mom3, lasth, train_median 순.
  평가창이 12개월 미만이면 기본 규칙(zero).
"""

import numpy as np
import pandas as pd


def rules(frame, h, by=None):
    """frame: past1·past3·past6·R_last 가 있는 표(결정월 순). by: 패널이면 'region'. 반환: zero·mom1·mom3·lasth 열."""
    out = pd.DataFrame(index=frame.index)
    p1 = frame["past1"]
    if by is None:
        m3 = p1.rolling(3, min_periods=3).mean()
    else:
        m3 = p1.groupby(frame[by]).transform(lambda v: v.rolling(3, min_periods=3).mean())
    out["zero"] = 0.0
    out["mom1"] = h * p1
    out["mom3"] = h * m3
    out["lasth"] = frame[f"past{h}"]
    return out


def train_median(G, P, h, first, min_labels=12):
    """G: 정답(결정월 순, 전국 계열), P: 결정월 PeriodIndex. 결정월 s 마다 그때 확인된 정답의 중앙값."""
    G = pd.Series(np.asarray(G, dtype=float), index=pd.PeriodIndex(P))
    out = pd.Series(np.nan, index=G.index)
    off = G[(G.index >= first)]
    for s_ in G.index:
        lab = off[off.index <= s_ - h].dropna()
        if len(lab) >= min_labels:
            out.loc[s_] = lab.median()
    return out


def select_naive(G, cand, T, h, first, window=24, min_window=12, tie_order=("zero", "mom1", "mom3", "lasth", "train_median"),
                 default="zero"):
    """G·cand 는 같은 결정월 index(PeriodIndex). 반환: (선택 후보명, {후보: 창 MAE}, 창 크기)"""
    idx = pd.PeriodIndex(G.index)
    hi = T - h
    lo = max(first, hi - (window - 1))
    w = (idx >= lo) & (idx <= hi) & pd.Series(G.values, index=idx).notna().values
    n = int(w.sum())
    if n < min_window:
        return default, {}, n
    maes = {}
    for c in cand.columns:
        f = cand.loc[w, c]
        if f.notna().all():
            maes[c] = float((G[w] - f).abs().mean())
    if not maes:
        return default, maes, n
    best = min(maes.values())
    for c in tie_order:
        if c in maes and abs(maes[c] - best) < 1e-12:
            return c, maes, n
    return min(maes, key=maes.get), maes, n
