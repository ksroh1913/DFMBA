# -*- coding: utf-8 -*-
"""전진 확장창 평가 엔진 (연구계획 8차 확정본 5~9장).

예측 시점 T 마다 정답이 확인된 결정월 t <= T-h 로 적합하고 결정월 T 의 행을 예측한다. 예측값은 채점 전에 저장한다.
  national_run : 전국 모형(월 단위). 정보군 A / B_공통 / C_공통. 확장 실험(옛 체계 행 추가, 전환 더미, 가중치) 지원.
  panel_run    : 시도 패널. 타깃 r(상대, 예측은 월내 평균 0 보정) / G(통합 패널) / down(급락 로지스틱).
  naive_run    : 단순 기준(zero·mom1·mom3·lasth·train_median)과 시점별 선택형 단순기준.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from . import baselines as B
from . import frames as F
from . import models as M
from . import settings as S
from .split import ForwardMonthSplit


def _alpha_grid(s):
    g = s["models"]["retune"]["grid"]["ridge_alpha"]
    return np.logspace(np.log10(g["min"]), np.log10(g["max"]), g["n"])


def choose_alpha(pipe, X, y, months, h, s, metric="mae"):
    """내부 전진 검증(마지막 12개 결정월)으로 Ridge alpha 를 고른다. 1-표준오차 규칙: 최소 평균 손실의 1 SE 안에서 가장 큰 alpha."""
    grid = _alpha_grid(s)
    cv = list(ForwardMonthSplit(h, n_val=12).split(X, y, groups=months))
    if len(cv) < 4:
        return pipe.named_steps["model"].alpha
    scores = np.full((len(grid), len(cv)), np.nan)
    for j, (tr, te) in enumerate(cv):
        base = clone(pipe)
        base.fit(X.iloc[tr], y.iloc[tr])
        Z_tr = base[:-1].transform(X.iloc[tr])
        Z_te = base[:-1].transform(X.iloc[te])
        for i, a in enumerate(grid):
            m = clone(pipe.named_steps["model"]).set_params(alpha=a).fit(Z_tr, y.iloc[tr])
            scores[i, j] = np.abs(m.predict(Z_te) - y.iloc[te].values).mean()
    mean, se = scores.mean(axis=1), scores.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    k = int(np.nanargmin(mean))
    ok = np.where(mean <= mean[k] + se[k])[0]
    return float(grid[ok.max()])


def _placebo(X, cols, rng):
    """위약 시험: 공통 블록 열을 월별 무작위 계열(같은 평균·표준편차)로 바꾼다."""
    X = X.copy()
    for c in cols:
        v = X[c]
        X[c] = rng.normal(v.mean(), v.std(ddof=0) if v.std(ddof=0) > 0 else 1.0, len(v)) if v.notna().any() else v
    return X


# ============================================================ 전국 모형
def national_run(s, nf, spec, h, origins, info="B", model="ridge", alpha=None, retune=False, refit_months=None,
                 extension=None, ext_weight=1.0, placebo_rng=None, exclude_ids=(), target="G", label=None, alpha_log=None):
    """반환: 예측 DataFrame(P, y, yhat, 모형, 정보군, h, 훈련행수, alpha), 구성 명세 목록(첫·마지막 시점).
    target 'g'(R(t) 기준 민감도 타깃)는 정답이 t+h+1 에 확인되므로 훈련행이 한 달 더 짧다."""
    first = S.per(s["timing"]["official_first_decision"])
    ycol = f"{target}{h}"
    lab = h + 1 if target == "g" else h
    sp = spec[~spec["ID"].isin(exclude_ids)]
    ccols = [c for c in sp.loc[sp.block.isin(["regional", "common"]) & M._in_common(sp), "입력"] if c in nf.columns]
    frame = nf.copy()
    frame[F.SIGN_C] = frame[f"G{h}"]
    if extension is not None:   # 옛 체계 행 추가: R_last·past·G 를 대용계열 값으로 바꾸고 더미를 붙인다
        ext = extension.copy()
        old = ext[ext["판정"] == "옛"].index
        frame = frame.reindex(frame.index.union(old))
        for c in ("R_last", "past1", "past3", "past6", f"G{h}"):
            frame.loc[old, c] = ext.loc[old, c]
        frame.loc[old, F.SIGN_C] = ext.loc[old, f"G{h}"]
        frame["옛체계"] = 0.0
        frame.loc[old, "옛체계"] = 1.0
        # 옛 행에 공통 블록·매매·전세 입력 붙이기 (nf 에 2012.01 부터 값이 있음)
        for c in ccols + ["V002|6개월%", "V026|6개월%"]:
            miss = frame[c].isna() & frame.index.isin(old)
            frame.loc[miss, c] = nf[c].reindex(frame.index[miss]).values
    A = list(F.A_NAT) + (["옛체계"] if extension is not None else [])
    if info == "A":
        ct = M.features(s, sp, A)
    elif info == "B":
        ct = M.features(s, sp, A, common_cols=ccols)
    elif info == "C":
        ct = M.features(s, sp, A, c_common=ccols)
    else:
        raise ValueError(info)
    pipe = M.ridge(s, ct, "national", alpha) if model == "ridge" else M.extra_trees(s, ct, "national", h, "reg")
    preds, comps = [], []
    fitted, cur_alpha = None, None
    for T in origins:
        tr_mask = (frame.index >= first) & (frame.index <= T - lab) & frame[ycol].notna() & frame[A].notna().all(axis=1)
        if extension is not None:
            tr_mask |= (frame["옛체계"] == 1) & (frame.index <= T - lab) & frame[ycol].notna() & frame[A].notna().all(axis=1)
        tr = frame[tr_mask]
        te = frame.loc[[T]] if T in frame.index else None
        if te is None or te[A].isna().any(axis=1).iloc[0]:
            continue
        Xtr, ytr = tr.drop(columns=[ycol]), tr[ycol]
        if placebo_rng is not None:
            Xtr = _placebo(Xtr, ccols, placebo_rng)
        need_fit = fitted is None or refit_months is None or T.month in refit_months
        if need_fit:
            p = clone(pipe)
            nested = alpha is None and not isinstance(s["models"]["ridge_national"].get("alpha"), (int, float))
            rule = "one_se" if (retune or info == "C") else ("cv_min" if nested else None)     # v9: 주 사양 cv_min, 8차 1-SE 는 민감도
            if model == "ridge" and rule and (T.month == s["models"]["ridge_national"].get("nested", {}).get("refit_month", 1) or cur_alpha is None):
                if rule == "one_se" and "nested" not in s["models"]["ridge_national"]:
                    cur_alpha = choose_alpha(p, Xtr, ytr, Xtr.index, h, s)
                else:
                    cur_alpha, ainfo = choose_alpha_cv(p, Xtr, ytr, Xtr.index, h, s, rule=rule)
                    if alpha_log is not None:
                        alpha_log.append({"시점": str(T), "h": h, "정보군": info, "모형": label or f"{model}_{info}", **ainfo})
            if cur_alpha is not None and model == "ridge":
                p.set_params(model__alpha=cur_alpha)
            fit_kw = {}
            if extension is not None and ext_weight != 1.0:
                fit_kw["model__sample_weight"] = np.where(Xtr["옛체계"] == 1, ext_weight, 1.0)
            p.fit(Xtr, ytr, **fit_kw)
            fitted = p
            if T == origins[0] or T == origins[-1]:
                c = M.compositions(p)
                if len(c):
                    c.insert(0, "시점", str(T))
                    comps.append(c)
        yhat = float(fitted.predict(te.drop(columns=[ycol]))[0])
        preds.append({"P": T, "h": h, "정보군": info, "모형": label or f"{model}_{info}", "y": float(te[ycol].iloc[0]), "yhat": yhat,
                      "훈련행수": int(len(tr)), "훈련월_옛체계": int(tr["옛체계"].sum()) if extension is not None else 0,
                      "alpha": float(fitted.named_steps["model"].alpha) if model == "ridge" else np.nan,
                      "label_month": str(T + h)})
    return pd.DataFrame(preds), comps


# ============================================================ 시도 패널 모형
def panel_run(s, pf, spec, h, origins, target="r", info="B", model="ridge", refit_months=None, placebo_rng=None,
              exclude_ids=(), label=None, prior_state=False, extra=False, alert_rates=None):
    """target: 'r'(상대 변화, 월내 평균 0 보정) / 'G'(통합 패널) / 'down'(급락 확률).
    정보군: r -> A_REL 기반(A·B·C), G/down -> A_REG 기반(A: 가격 추세만, B: +공통·지역 RMPI, Bc: +공통 RMPI만, C: 개별 변수)."""
    first = S.per(s["timing"]["official_first_decision"])
    sp = spec[~spec["ID"].isin(exclude_ids)]
    dcols = [f"D|{c}" for c in sp.loc[sp.block == "regional", "입력"] if f"D|{c}" in pf.columns]
    ccols = [f"C|{c}" for c in sp.loc[sp.block.isin(["regional", "common"]) & M._in_common(sp), "입력"] if f"C|{c}" in pf.columns]
    xcols = [f"D|{c}" for c in sp.loc[sp.block == "extra", "입력"] if f"D|{c}" in pf.columns] if extra else None   # v9 별도 입력
    frame = pf.copy()
    frame[F.SIGN_R] = frame[f"r{h}"]
    frame[F.SIGN_C] = frame[f"Gnat{h}"]
    ycol = f"{target}{h}"
    if target == "r":
        A = list(F.A_REL)
        if info == "A":
            ct = M.features(s, sp, A)
        elif info == "B":
            ct = M.features(s, sp, A, regional_cols=dcols, prefix_d="D|", extra_cols=xcols)
        elif info == "C":
            ct = M.features(s, sp, A, c_regional=dcols)
        else:
            raise ValueError(info)
    else:
        A = list(F.A_REG)
        if info == "A":
            ct = M.features(s, sp, A)
        elif info == "B":
            ct = M.features(s, sp, A, common_cols=ccols, regional_cols=dcols, prefix_c="C|", prefix_d="D|", extra_cols=xcols)
        elif info == "Bc":
            ct = M.features(s, sp, A, common_cols=ccols, prefix_c="C|")
        elif info == "C":
            ct = M.features(s, sp, A, c_common=ccols, c_regional=dcols)
        else:
            raise ValueError(info)
    if target == "down":
        pipe = M.logistic(s, ct) if model == "ridge" else M.extra_trees(s, ct, "panel", h, "clf")
    else:
        pipe = M.ridge(s, ct, "panel") if model == "ridge" else M.extra_trees(s, ct, "panel", h, "reg")
    preds, comps, diag = [], [], []
    fitted, cur_cuts = None, {}
    n_reg = int(frame["region"].nunique())
    for T in origins:
        tr = frame[(frame.P >= first) & (frame.P <= T - h) & frame[ycol].notna() & frame[A].notna().all(axis=1)]
        te = frame[(frame.P == T) & frame[A].notna().all(axis=1)]
        if te.empty or tr.empty:
            continue
        Xtr, ytr = tr.drop(columns=[ycol]), tr[ycol]
        if placebo_rng is not None and ccols:
            Xtr = _placebo(Xtr, ccols, placebo_rng)
        if target == "down" and ytr.nunique() < 2:
            continue
        if fitted is None or refit_months is None or T.month in refit_months:
            fitted = clone(pipe).fit(Xtr, ytr)
            if target == "down" and alert_rates:     # v9 ④: 훈련기간 예측확률 분포에서 '사건비율 x 배수' 상위 비율의 분위를 컷오프로
                p_tr = fitted.predict_proba(Xtr)[:, 1]
                e = float(ytr.mean())
                cur_cuts = {k: float(np.quantile(p_tr, max(0.0, 1.0 - k * e))) for k in alert_rates}
            if T == origins[0] or T == origins[-1]:
                c = M.compositions(fitted)
                if len(c):
                    c.insert(0, "시점", str(T))
                    comps.append(c)
                if model == "et":
                    d = M.leaf_month_diagnostic(fitted, Xtr, Xtr["P"].astype(str).values)
                    if d:
                        diag.append({"시점": str(T), "h": h, "타깃": target, **d})
        Xte = te.drop(columns=[ycol])
        if target == "down":
            yhat = fitted.predict_proba(Xte)[:, 1]
            base_rate = float(ytr.mean())
        else:
            yhat = fitted.predict(Xte)
            base_rate = np.nan
        rec = pd.DataFrame({"P": te["P"].values, "region": te["region"].values, "h": h, "타깃": target, "정보군": info,
                            "모형": label or f"{model}_{info}", "y": te[ycol].values, "yhat": np.asarray(yhat, dtype=float),
                            "훈련행수": int(len(tr)), "훈련사건비율": base_rate})
        if target == "r":                                   # 같은 달 지역 예측 평균을 빼서 보정 (N=17 완전패널)
            rec["yhat_raw"] = rec["yhat"]
            rec["yhat"] = rec["yhat"] - rec["yhat"].mean() if len(rec) == n_reg else np.nan   # 완전 횡단면에서만 보정
        if prior_state and target == "down":
            rec["기존급락상태"] = te[f"state_down{h}"].values
        if target == "down" and alert_rates:
            for k, v in cur_cuts.items():
                rec[f"컷오프_{k}x"] = v
        preds.append(rec)
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()), comps, pd.DataFrame(diag)


# ============================================================ 단순 기준
def naive_run(s, nf, h, origins, target="G"):
    """전국 표의 단순 기준 예측과 시점별 선택형 단순기준. 반환: (예측 long DataFrame, 선택 기록)"""
    first = S.per(s["timing"]["official_first_decision"])
    nv = s["naive"]
    cand = B.rules(nf, h)
    G = nf[f"{target}{h}"]
    cand["train_median"] = B.train_median(G, nf.index, h, first, nv["median_min_labels"])
    preds, sel = [], []
    for T in origins:
        if T not in nf.index or np.isnan(G.loc[T]):
            continue
        name, maes, n = B.select_naive(G, cand, T, h, first, nv["window_months"], nv["min_window_months"],
                                       tuple(nv["tie_order"]), nv["default_rule"])
        for c in cand.columns:
            preds.append({"P": T, "h": h, "정보군": "-", "모형": f"naive_{c}", "y": float(G.loc[T]), "yhat": float(cand.loc[T, c])})
        preds.append({"P": T, "h": h, "정보군": "-", "모형": "naive_selected", "y": float(G.loc[T]), "yhat": float(cand.loc[T, name])})
        sel.append({"P": T, "h": h, "선택": name, "평가창": n, **{f"MAE_{k}": v for k, v in maes.items()}})
    return pd.DataFrame(preds), pd.DataFrame(sel)


def panel_naive(s, pf, h, origins, target="G"):
    """패널 단순 기준: zero / mom1 / mom3 / lasth (지역별). target 'r' 이면 zero 만 의미가 있다."""
    rows = []
    cand = B.rules(pf, h, by="region")
    for c in cand.columns:
        d = pf[["P", "region", f"{target}{h}"]].copy()
        d["yhat"] = cand[c].values if target == "G" else 0.0
        d = d[d.P.isin(origins) & d[f"{target}{h}"].notna()]
        rows.append(pd.DataFrame({"P": d.P.values, "region": d.region.values, "h": h, "타깃": target, "정보군": "-",
                                  "모형": f"naive_{c}", "y": d[f"{target}{h}"].values, "yhat": d["yhat"].values}))
        if target == "r":
            break
    return pd.concat(rows, ignore_index=True)


# ============================================================ v9: 전진 교차검증 최소 alpha 선택 (2단계 격자)
def _cv_scores(pipe, X, y, cv, grid, loss="mae", param="alpha"):
    """폴드마다 전처리(특징 변환)를 훈련 행으로 적합한 뒤 격자의 모형만 바꿔 손실을 잰다. 반환 (len(grid), len(cv))"""
    scores = np.full((len(grid), len(cv)), np.nan)
    for j, (tr, te) in enumerate(cv):
        base = clone(pipe)
        base.fit(X.iloc[tr], y.iloc[tr])
        Z_tr = base[:-1].transform(X.iloc[tr])
        Z_te = base[:-1].transform(X.iloc[te])
        for i, a in enumerate(grid):
            m = clone(pipe.named_steps["model"]).set_params(**{param: a}).fit(Z_tr, y.iloc[tr])
            if loss == "brier":
                pr = m.predict_proba(Z_te)[:, 1]
                scores[i, j] = np.mean((pr - y.iloc[te].values) ** 2)
            else:
                scores[i, j] = np.abs(m.predict(Z_te) - y.iloc[te].values).mean()
    return scores


def choose_alpha_cv(pipe, X, y, months, h, s, rule="cv_min", loss="mae", param="alpha", cfg=None):
    """v9 규칙(설계안 2.1): 1단계 격자(10^-2~10^4, 25점) → 최적점 ±0.5 로그단위 0.125 간격 → 평균 손실 최소.
    최소점이 격자 끝이면 한 자릿수 한 번 연장. 최소의 within_pct 안에 여럿이면 로그 거리로 prefer 에 가장 가까운 것.
    폴드가 min_folds 미만이면 fallback. rule='one_se' 는 8차 규칙(1 SE 안의 가장 큰 값, 1단계 격자만).
    로지스틱 C 는 param='C', loss='brier' 로 쓴다(격자는 1/alpha). 반환 (값, 기록 dict)."""
    cfg = cfg or s["models"]["ridge_national"]["nested"]
    n_val = cfg["validation"]["n_val"]
    cv = list(ForwardMonthSplit(h, n_val=n_val).split(X, y, groups=months))
    if loss == "brier":     # 단일 계급 폴드 제외
        cv = [(tr, te) for tr, te in cv if y.iloc[tr].nunique() > 1 and len(te)]
    fb = cfg["fallback_alpha"] if param == "alpha" else 1.0 / cfg["fallback_alpha"]
    if len(cv) < cfg["min_folds"]:
        return float(fb), {"규칙": "fallback", "폴드": len(cv)}
    g1 = cfg["stage1"]
    grid = np.logspace(np.log10(g1["min"]), np.log10(g1["max"]), g1["n"])
    if param == "C":
        grid = 1.0 / grid
    sc = _cv_scores(pipe, X, y, cv, grid, loss, param)
    mean = sc.mean(axis=1)
    k = int(np.nanargmin(mean))
    if rule == "one_se":
        se = sc.std(axis=1, ddof=1) / np.sqrt(sc.shape[1])
        ok = np.where(mean <= mean[k] + se[k])[0]
        pick = int(ok.max()) if param == "alpha" else int(ok.min())
        return float(grid[pick]), {"규칙": "one_se", "폴드": len(cv), "최소": float(grid[k]), "선택": float(grid[pick])}
    # 격자 끝이면 한 자릿수 연장 (한 번)
    if cfg.get("edge_extend_once") and k in (0, len(grid) - 1):
        step = np.log10(grid[1]) - np.log10(grid[0])
        if k == 0:
            ext = grid[0] * 10 ** (np.arange(-1, 0, abs(step)))
        else:
            ext = grid[-1] * 10 ** (np.arange(abs(step), 1 + 1e-9, abs(step)))
        sc_e = _cv_scores(pipe, X, y, cv, ext, loss, param)
        grid = np.concatenate([grid, ext]); sc = np.concatenate([sc, sc_e]); mean = sc.mean(axis=1)
        k = int(np.nanargmin(mean))
    # 2단계: 최적점 주변 세밀 격자 (기존 점 제외)
    g2 = cfg["stage2"]
    offs = np.arange(-g2["half_width_log10"], g2["half_width_log10"] + 1e-9, g2["step_log10"])
    fine = grid[k] * 10 ** offs
    fine = np.array([a for a in fine if not np.any(np.isclose(np.log10(a), np.log10(grid), atol=1e-6))])
    if len(fine):
        sc_f = _cv_scores(pipe, X, y, cv, fine, loss, param)
        grid = np.concatenate([grid, fine]); sc = np.concatenate([sc, sc_f]); mean = sc.mean(axis=1)
    k = int(np.nanargmin(mean))
    # 평평한 곡선: 최소의 within_pct 안이면 prefer 에 로그 거리로 가장 가까운 것
    ft = cfg.get("flat_tie", {})
    prefer = ft.get("prefer_log_nearest", 10.0)
    if param == "C":
        prefer = 1.0 / prefer
    within = np.where(mean <= mean[k] * (1 + ft.get("within_pct", 0) / 100.0))[0]
    pick = int(within[np.argmin(np.abs(np.log10(grid[within]) - np.log10(prefer)))]) if len(within) else k
    return float(grid[pick]), {"규칙": rule, "폴드": len(cv), "최소": float(grid[k]), "최소손실": float(mean[k]),
                               "선택": float(grid[pick]), "선택손실": float(mean[pick]), "평평후보수": int(len(within)), "격자점수": int(len(grid))}


# ============================================================ v9: 보정 구조 (M0 · M1 · M1′ · M2 · M3)
CORR_A_NAT = ["V002|6개월%", "V026|6개월%"]            # M2 입력 중 가격 추세(과거 월세 변화 past1·3·6 제외)


def _corr_inputs(s, sp, model, nf_or_pf, panel=False, extra=False):
    """보정모형 입력 변환기. panel=False: 전국(A 가격추세 + 공통 RMPI). panel=True: + 지역 RMPI (+ M3: 별도 입력)."""
    A = list(CORR_A_NAT) + (["past3", "past6"] if model.endswith("+past36") else [])
    base = model.split("+")[0]
    if not panel:
        ccols = [c for c in sp.loc[sp.block.isin(["regional", "common"]) & M._in_common(sp), "입력"] if c in nf_or_pf.columns]
        return A, M.features(s, sp, A, common_cols=ccols)
    dcols = [f"D|{c}" for c in sp.loc[sp.block == "regional", "입력"] if f"D|{c}" in nf_or_pf.columns]
    ccols = [f"C|{c}" for c in sp.loc[sp.block.isin(["regional", "common"]) & M._in_common(sp), "입력"] if f"C|{c}" in nf_or_pf.columns]
    xcols = [f"D|{c}" for c in sp.loc[sp.block == "extra", "입력"] if f"D|{c}" in nf_or_pf.columns] if (base == "M3" or extra) else None
    return A, M.features(s, sp, A, common_cols=ccols, regional_cols=dcols, prefix_c="C|", prefix_d="D|", extra_cols=xcols)


def correction_run(s, frame_in, spec, h, origins, model="M2", panel=False, alpha_rule="cv_min", label=None, exclude_ids=(), alpha_value=None):
    """학습 목표 = G − mom1 (mom1 = h × past1), 최종 예측 = mom1 + 보정값.
    model: 'M0'(mom1), 'M1'(절편만), 'M1s'(mom1 계수 학습), 'M2'(가격추세+RMPI), 'M2+past36', 'M3'(패널: +별도 입력).
    전국: frame_in = 결정월 index 의 nf.  패널: frame_in = pf(region, P 열), 통합 패널 G, 지역별 mom1.
    alpha 는 매년 1월(또는 첫 시점) 전진 교차검증 최소로 고른다(2.1 규칙). 반환 (예측 DataFrame, 구성 명세 목록, alpha 기록)."""
    first = S.per(s["timing"]["official_first_decision"])
    sp = spec[~spec["ID"].isin(exclude_ids)]
    ycol = f"G{h}"
    f = frame_in.copy()
    if not panel:
        f = f.copy(); f["P"] = f.index
    f["mom1"] = h * f["past1"]
    f["resid"] = f[ycol] - f["mom1"]
    f[F.SIGN_C] = f[f"Gnat{h}"] if panel else f[ycol]
    if panel:
        f[F.SIGN_R] = f[f"r{h}"]
    base = model.split("+")[0]
    A, ct = _corr_inputs(s, sp, model, f, panel=panel) if base in ("M2", "M3") else (list(CORR_A_NAT), None)
    pipe = M.ridge(s, ct, "national") if ct is not None else None
    refit_month = s["models"]["ridge_national"]["nested"]["refit_month"]
    preds, comps, alog = [], [], []
    fitted, cur_alpha, m1 = None, None, None
    for T in origins:
        tr = f[(f["P"] >= first) & (f["P"] <= T - h) & f[ycol].notna() & f["mom1"].notna() & f[A].notna().all(axis=1)]
        te = f[(f["P"] == T) & f["mom1"].notna() & f[A].notna().all(axis=1)]
        if te.empty or tr.empty:
            continue
        if base == "M0":
            adj = np.zeros(len(te))
        elif base == "M1":
            adj = np.full(len(te), float(tr["resid"].mean()))
        elif base == "M1s":
            Xo = np.c_[np.ones(len(tr)), tr["mom1"].values]
            beta = np.linalg.lstsq(Xo, tr["resid"].values, rcond=None)[0]
            adj = beta[0] + beta[1] * te["mom1"].values
            m1 = beta
        else:
            Xtr, ytr = tr.drop(columns=["resid", ycol]), tr["resid"]
            if fitted is None or T.month == refit_month:
                p = clone(pipe)
                if alpha_rule in ("cv_min", "one_se"):
                    cur_alpha, info = choose_alpha_cv(p, Xtr, ytr, Xtr["P"], h, s, rule=alpha_rule)
                    alog.append({"시점": str(T), "h": h, "모형": model, **info})
                elif alpha_rule == "fixed" and alpha_value is not None:
                    cur_alpha = float(alpha_value)
                else:
                    cur_alpha = float(s["models"]["ridge_national"]["nested"]["fallback_alpha"])
                p.set_params(model__alpha=cur_alpha)
                fitted = p.fit(Xtr, ytr)
                if T == origins[0] or T == origins[-1]:
                    c = M.compositions(fitted)
                    if len(c):
                        c.insert(0, "시점", str(T)); c.insert(1, "모형", model); comps.append(c)
            adj = fitted.predict(te.drop(columns=["resid", ycol]))
        rec = pd.DataFrame({"P": te["P"].values, "h": h, "모형": label or f"corr_{model}", "y": te[ycol].values,
                            "mom1": te["mom1"].values, "보정": np.asarray(adj, dtype=float),
                            "yhat": te["mom1"].values + np.asarray(adj, dtype=float), "훈련행수": int(len(tr)),
                            "alpha": (cur_alpha if base in ("M2", "M3") else np.nan), "label_month": str(T + h)})
        if panel:
            rec.insert(1, "region", te["region"].values)
        if base == "M1s" and m1 is not None:
            rec["m1s_절편"], rec["m1s_기울기"] = m1[0], m1[1]
        preds.append(rec)
    return (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()), comps, pd.DataFrame(alog)
