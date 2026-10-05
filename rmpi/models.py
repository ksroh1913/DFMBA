# -*- coding: utf-8 -*-
"""사전 지정 모형과 보조 모형의 파이프라인 (연구계획 8차 확정본 8장).

Pipeline = ColumnTransformer(A 통과 / RMPI 공통 / RMPI 지역 / C 개별 변수 / 노출항) → 중앙값 대체 → 표준화(트리 생략) → 모형.
RMPI 변환·결측 대체·스케일링은 모두 훈련자료로만 적합된다. 하이퍼파라미터는 설정 파일에 고정한다.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .frames import SIGN_C, SIGN_R
from .index import RMPI, Exposure, ExtraInputs, IndividualInputs


def _in_common(spec):
    """v9: 출력 단위 공통 블록 포함 플래그(없으면 모두 포함)"""
    if "in_common" in spec.columns:
        return spec["in_common"].fillna(True).astype(bool)
    return pd.Series(True, index=spec.index)


def features(s, spec, A_cols, common_cols=None, regional_cols=None, c_common=None, c_regional=None,
             exposure_pairs=None, prefix_c="", prefix_d="", extra_cols=None):
    """특징 변환기. spec 의 '입력' 이름은 열 이름과 같아야 한다(패널표는 prefix 로 맞춘다).
    extra_cols(v9): RMPI 동인 지수 밖의 별도 입력(실거래). 지역 블록 처리(풀링 표준편차, 훈련 중앙값 대체, 결측 표시)."""
    kw = dict(obs_rate_min=s["rmpi"]["obs_rate_min"], group_min_share=s["rmpi"]["group_min_share"],
              driver_min_share=s["rmpi"]["driver_min_share"])
    parts = [("A", "passthrough", list(A_cols))]
    if common_cols:
        sp = spec[spec["block"].isin(["regional", "common"]) & _in_common(spec)].copy()
        sp["입력"] = prefix_c + sp["입력"]
        common_cols = [c for c in common_cols if c in set(sp["입력"])]
        parts.append(("RMPI공통", RMPI(sp, "common", "RMPI공통", sign_col=SIGN_C, **kw), list(common_cols) + [SIGN_C]))
    if regional_cols:
        sp = spec[spec["block"] == "regional"].copy()
        sp["입력"] = prefix_d + sp["입력"]
        parts.append(("RMPI지역", RMPI(sp, "regional", "RMPI지역", sign_col=SIGN_R, **kw), list(regional_cols) + [SIGN_R]))
    if c_common:
        parts.append(("C공통", IndividualInputs(list(c_common), "C공통", s["rmpi"]["obs_rate_min"]), list(c_common)))
    if c_regional:
        parts.append(("C지역", IndividualInputs(list(c_regional), "C지역", s["rmpi"]["obs_rate_min"]), list(c_regional)))
    if exposure_pairs:
        cols = sorted({c for c, r, _ in exposure_pairs} | {r for c, r, _ in exposure_pairs})
        parts.append(("노출", Exposure(exposure_pairs), cols))
    if extra_cols:
        parts.append(("별도입력", ExtraInputs(list(extra_cols), "별도", s["rmpi"]["obs_rate_min"]), list(extra_cols)))
    ct = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def ridge(s, ct, kind="national", alpha=None):
    a = alpha if alpha is not None else s["models"][f"ridge_{kind}"]["alpha"]
    if not isinstance(a, (int, float)):            # v9: alpha: nested -> 초기값은 fallback, 실제 값은 choose_alpha_cv 가 정한다
        a = float(s["models"][f"ridge_{kind}"]["nested"]["fallback_alpha"])
    return Pipeline([("features", ct), ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("model", Ridge(alpha=a))]).set_output(transform="pandas")


def logistic(s, ct, C=None):
    c = C if C is not None else s["models"]["logistic_panel"]["C"]
    return Pipeline([("features", ct), ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=c, max_iter=5000))]).set_output(transform="pandas")


def extra_trees(s, ct, kind="national", h=1, task="reg"):
    p = s["models"][f"extra_trees_{kind}"]
    leaf = p["min_samples_leaf"][h] if isinstance(p["min_samples_leaf"], dict) else p["min_samples_leaf"]
    mf = p["max_features"]
    common = dict(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=leaf, max_features=mf,
                  random_state=s["meta"]["random_seed"], n_jobs=4)
    est = ExtraTreesRegressor(**common) if task == "reg" else ExtraTreesClassifier(**common)
    return Pipeline([("features", ct), ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("model", est)]).set_output(transform="pandas")


def feature_names(pipe):
    return list(pipe.named_steps["features"].get_feature_names_out())


def compositions(pipe):
    """적합된 파이프라인의 RMPI 구성 명세(공통·지역)"""
    out = []
    ct = pipe.named_steps["features"] if hasattr(pipe, "named_steps") else pipe
    for name, tr, _ in ct.transformers_:
        if isinstance(tr, ExtraInputs):
            c = tr.composition()
            if len(c):
                out.append(c)
            continue
        if isinstance(tr, RMPI):
            out.append(tr.composition())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def leaf_month_diagnostic(pipe, X, months):
    """트리 모형: 잎마다 서로 다른 결정월 수 (최소·중앙값). 트리 모형이 아니면 None."""
    est = pipe.named_steps["model"]
    if not hasattr(est, "estimators_"):
        return None
    Z = pipe[:-1].transform(X)
    months = np.asarray(months)
    mins, meds = [], []
    for t in est.estimators_[:50]:
        leaves = t.apply(Z.values if hasattr(Z, "values") else Z)
        cnt = pd.Series(months).groupby(leaves).nunique()
        mins.append(cnt.min())
        meds.append(cnt.median())
    return {"잎당_월수_최소(50트리 중앙)": float(np.median(mins)), "잎당_월수_중앙(50트리 중앙)": float(np.median(meds))}
