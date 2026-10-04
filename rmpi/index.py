# -*- coding: utf-8 -*-
"""RMPI 변환기 (연구계획 8차 확정본 4장). 훈련자료로만 적합하고 Pipeline 안에서 쓴다.

  ⑤ 학습 구간 관측률 필터: 관측률 70% 이상이고 상수가 아닌 구성변수만
  ⑥ 지수 구성용 표준화: 공통 블록은 훈련기간 월 단위 평균·표준편차, 지역 블록은 모든 지역의 훈련기간 월내 편차를
     모아 계산한 표준편차로 나눈다(중심화 없음, 지역마다 따로 계산하지 않음)
  ⑦ 방향: 데이터사전 예상 부호. ± 는 훈련기간 목표(공통 G_전국, 지역 r)와의 순위상관 부호로 정해 기록
  ⑧ 집계: 동인 안의 하위 묶음을 먼저 평균(남은 구성변수의 절반 이상 관측될 때만), 묶음 간 동일 가중 평균
     (남은 묶음의 절반 이상 계산될 때만). 제외 변수와 빈 묶음은 빼고 남은 항목에 동일 가중치. 0으로 채우지 않는다
  ⑨ 결측 대체: 동인 지수 단위로 훈련 중앙값 + 결측 표시. 훈련기간 전체가 결측이거나 상수인 지수는 제외하고 기록
개별 변수 모형(C)은 ⑥~⑧ 없이 ⑤ 다음에 변수 단위로 ⑨를 적용한다(IndividualInputs).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin

DRIVER_MARK = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥"}


def _sign_of(expected, x, y):
    """예상 부호 -> (+1/-1, 근거). ± 는 훈련기간 순위상관 부호."""
    if expected == "+":
        return 1.0, "사전 +"
    if expected == "-":
        return -1.0, "사전 −"
    m = x.notna() & pd.Series(y, index=x.index).notna()
    if m.sum() < 8:
        return 1.0, "순위상관 불가(표본 부족) → +"
    rho = spearmanr(x[m], np.asarray(y)[m.values]).statistic
    if not np.isfinite(rho) or rho == 0:
        return 1.0, "순위상관 0 → +"
    return (1.0 if rho > 0 else -1.0), f"순위상관 {rho:+.2f}"


class RMPI(BaseEstimator, TransformerMixin):
    def __init__(self, spec, block, prefix, obs_rate_min=0.7, group_min_share=0.5, driver_min_share=0.5, sign_col=None):
        self.spec = spec
        self.block = block
        self.prefix = prefix
        self.obs_rate_min = obs_rate_min
        self.group_min_share = group_min_share
        self.driver_min_share = driver_min_share
        self.sign_col = sign_col     # ± 부호 결정용 목표가 든 열(공통 G_전국, 지역 r). 있으면 y 대신 쓰고 출력에서 뺀다

    def _split_sign(self, X, y):
        X = pd.DataFrame(X)
        if self.sign_col is not None and self.sign_col in X.columns:
            y = X[self.sign_col].values
            X = X.drop(columns=[self.sign_col])
        return X, y

    # ------------------------------------------------------------ 적합
    def fit(self, X, y=None):
        X, y = self._split_sign(X, y)
        sp = self.spec[self.spec["입력"].isin(X.columns)].copy()
        obs = X[sp["입력"]].notna().mean()
        nun = X[sp["입력"]].nunique()
        self.obs_rate_ = obs
        keep, dropped = [], {}
        for c in sp["입력"]:
            if obs[c] < self.obs_rate_min:
                dropped[c] = f"관측률 {obs[c]:.2f} < {self.obs_rate_min}"
            elif nun[c] <= 1:
                dropped[c] = "상수"
            else:
                keep.append(c)
        # 표준화
        if self.block == "common":
            mu = X[keep].mean()
            sd = X[keep].std(ddof=0)
        else:
            mu = pd.Series(0.0, index=keep)
            sd = X[keep].std(ddof=0)
        for c in list(keep):
            if not np.isfinite(sd[c]) or sd[c] <= 0:
                dropped[c] = "표준편차 0"
                keep.remove(c)
        self.keep_, self.dropped_, self.mu_, self.sd_ = keep, dropped, mu[keep], sd[keep]
        # 방향
        signs, src = {}, {}
        exp = dict(zip(sp["입력"], sp["예상부호"]))
        for c in keep:
            signs[c], src[c] = _sign_of(exp[c], X[c], y) if y is not None else (1.0 if exp[c] != "-" else -1.0, "사전")
        self.sign_, self.sign_source_ = pd.Series(signs), src
        # 구조: 동인 -> 묶음 -> 구성변수 (남은 것만)
        struct = {}
        for _, r in sp[sp["입력"].isin(keep)].iterrows():
            struct.setdefault(int(r["동인"]), {}).setdefault(str(r["하위묶음"]), []).append(r["입력"])
        self.struct_ = struct
        idx = self._indices(X)
        med = idx.median()
        drivers, dropped_d = [], {}
        for d in idx.columns:
            if idx[d].notna().sum() == 0:
                dropped_d[d] = "훈련기간 전체 결측(중앙값 계산 불가)"
            elif idx[d].nunique() <= 1:
                dropped_d[d] = "훈련기간 상수(변별력 없음)"
            else:
                drivers.append(d)
        self.drivers_, self.dropped_drivers_, self.median_ = drivers, dropped_d, med[drivers]
        self.n_train_ = len(X)
        self.feature_names_out_ = [f"{self.prefix}|{d}" for d in drivers] + [f"{self.prefix}|{d}|결측" for d in drivers]
        return self

    # ------------------------------------------------------------ 지수 계산 (적합·변환 공용)
    def _indices(self, X):
        Z = (X[self.keep_] - self.mu_) / self.sd_ * self.sign_
        out = {}
        for d, groups in self.struct_.items():
            gvals = []
            for g, cols in groups.items():
                sub = Z[cols]
                n_av = sub.notna().sum(axis=1)
                need = max(1, int(np.ceil(self.group_min_share * len(cols))))
                gvals.append(sub.mean(axis=1).where(n_av >= need))
            G = pd.concat(gvals, axis=1)
            n_av = G.notna().sum(axis=1)
            need = max(1, int(np.ceil(self.driver_min_share * G.shape[1])))
            out[DRIVER_MARK[d]] = G.mean(axis=1).where(n_av >= need)
        return pd.DataFrame(out, index=X.index)

    def transform(self, X):
        X, _ = self._split_sign(X, None)
        idx = self._indices(X)
        out = pd.DataFrame(index=X.index)
        for d in self.drivers_:
            out[f"{self.prefix}|{d}"] = idx[d].fillna(self.median_[d])
        for d in self.drivers_:
            out[f"{self.prefix}|{d}|결측"] = idx[d].isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    # ------------------------------------------------------------ 구성 명세
    def composition(self):
        rows = []
        sp = self.spec.set_index("입력")
        for c in list(self.keep_) + list(self.dropped_):
            r = sp.loc[c]
            kept = c in self.keep_
            d = DRIVER_MARK[int(r["동인"])]
            n_g = len(self.struct_.get(int(r["동인"]), {}).get(str(r["하위묶음"]), []))
            n_groups = len(self.struct_.get(int(r["동인"]), {}))
            rows.append({"블록": self.block, "동인": d, "하위묶음": r["하위묶음"], "입력": c, "유지": kept,
                         "제외이유": self.dropped_.get(c, ""), "관측률": round(float(self.obs_rate_.get(c, np.nan)), 3),
                         "예상부호": r["예상부호"], "적용부호": (self.sign_[c] if kept else np.nan),
                         "부호근거": self.sign_source_.get(c, ""), "표준편차": (round(float(self.sd_[c]), 4) if kept else np.nan),
                         "묶음내가중": (round(1 / n_g, 3) if kept and n_g else np.nan),
                         "동인내묶음가중": (round(1 / n_groups, 3) if kept and n_groups else np.nan),
                         "동인지수유지": (d in self.drivers_), "동인제외이유": self.dropped_drivers_.get(d, "")})
        return pd.DataFrame(rows)


class IndividualInputs(BaseEstimator, TransformerMixin):
    """C 정보군: 관측률 필터(⑤) → 변수 단위 훈련 중앙값 대체 + 결측 표시(⑨). 스케일링은 뒤 단계."""

    def __init__(self, columns, prefix, obs_rate_min=0.7):
        self.columns = columns
        self.prefix = prefix
        self.obs_rate_min = obs_rate_min

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = [c for c in self.columns if c in X.columns]
        obs = X[cols].notna().mean()
        nun = X[cols].nunique()
        self.keep_ = [c for c in cols if obs[c] >= self.obs_rate_min and nun[c] > 1]
        self.dropped_ = {c: ("상수" if nun[c] <= 1 else f"관측률 {obs[c]:.2f}") for c in cols if c not in self.keep_}
        self.median_ = X[self.keep_].median()
        self.has_missing_ = [c for c in self.keep_ if X[c].isna().any()]
        self.feature_names_out_ = [f"{self.prefix}|{c}" for c in self.keep_] + [f"{self.prefix}|{c}|결측" for c in self.has_missing_]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for c in self.keep_:
            out[f"{self.prefix}|{c}"] = X[c].fillna(self.median_[c])
        for c in self.has_missing_:
            out[f"{self.prefix}|{c}|결측"] = X[c].isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class Exposure(BaseEstimator, TransformerMixin):
    """노출항(보조): 공통 충격(훈련 표준화) x 지역 특성 월내 편차(훈련 표준편차로 나눔). 결측은 0(= 정보 없음)."""

    def __init__(self, pairs):
        self.pairs = pairs   # [(공통열, 지역열, 이름)]

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.stats_ = {}
        for c, r, _ in self.pairs:
            self.stats_[c] = (X[c].mean(), X[c].std(ddof=0) or 1.0)
            self.stats_[r] = (0.0, X[r].std(ddof=0) or 1.0)
        self.feature_names_out_ = [f"노출|{n}" for _, _, n in self.pairs]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for c, r, n in self.pairs:
            zc = (X[c] - self.stats_[c][0]) / self.stats_[c][1]
            zr = X[r] / self.stats_[r][1]
            out[f"노출|{n}"] = (zc * zr).fillna(0.0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
