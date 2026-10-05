# -*- coding: utf-8 -*-
"""전진 날짜 분할 (연구계획 8차 확정본 9장).

예측 시점 T, 목표 기간 h: 훈련행은 정답이 공표된 결정월 t <= T-h (t >= 공식 첫 결정월). 예측행은 t = T.
내부 검증(재튜닝·C 정보군의 규제 강도)도 같은 규칙의 전진 분할기(ForwardMonthSplit)를 쓴다.
무작위 split·shuffle KFold·월 묶음 GroupKFold 는 쓰지 않는다.
"""

import numpy as np
import pandas as pd

from . import settings as S


def origins(s, start=None, end=None):
    return pd.period_range(S.per(start or s["timing"]["eval_start"]), S.per(end or s["timing"]["eval_end"]), freq="M")


def train_mask(P, T, h, first):
    """결정월 P(Series/Index of Period) 중 시점 T 에 정답이 확인된 행"""
    P = pd.PeriodIndex(P)
    return np.asarray((P >= first) & (P <= T - h))


class ForwardMonthSplit:
    """sklearn 호환 분할기. groups 에 결정월(Period)을 넘긴다.
    검증 월 v(마지막 n_val 개 결정월) 마다 훈련 = 결정월 <= v - h, 검증 = 결정월 == v."""

    def __init__(self, h, n_val=12, min_train_months=12):
        self.h = h
        self.n_val = n_val
        self.min_train_months = min_train_months

    def split(self, X, y=None, groups=None):
        P = pd.PeriodIndex(groups)
        uniq = P.unique().sort_values()
        for v in uniq[-self.n_val:]:
            tr = np.where(P <= v - self.h)[0]
            te = np.where(P == v)[0]
            if len(np.unique(P[tr])) >= self.min_train_months and len(te):
                yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None):
        return sum(1 for _ in self.split(X, y, groups))
