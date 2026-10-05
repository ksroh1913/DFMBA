# -*- coding: utf-8 -*-
"""변수 변환·성분 분해 (연구계획 8차 확정본 2장 처리 순서 ①~④).

  ① 기준기간 원자료에서 미래정보 보완 셀 마스킹: 취합본에서 빈칸이고 1차 시트에서 채워진 셀을
     설명서의 처리 방법(역산_지수변화·대용_금리차)으로 찾아 지운다.
  ② 공표 시차 반영: 2차 시트의 이동을 그대로 쓰되(마스킹 표시도 같은 시차로 이동),
     V060 은 3개월, V072 는 1개월 더 민다.
  ③ 변화율·12개월 합계·차분: 계산 창 안에 결측이 하나라도 있으면 결측.
  ④ 기준 지역 평균·월내 편차 분해: 기준 지역은 최초 훈련기간(2016.01~2017.12)에서 한 번 정해 고정.
①~④는 과거와 같은 달의 값만 쓰는 고정 계산이라 분할 전에 한 번 한다. ⑤ 이후는 rmpi/index.py.

입력 열 이름: '<ID>|<변환>' 예) 'V003|수준', 'V003|Δ12', 'V021|12개월합_천세대당', 'V046|로그Δ12', 'V002|6개월%'
"""

import sys

import numpy as np
import pandas as pd

from . import settings as S

sys.path.insert(0, S.BASE)
from variables import RELEASE_MONTHLY  # noqa: E402

ID_COLS = ["panel", "region", "region_code", "month"]
SUFFIX = {"level": ["수준"], "d12": ["Δ12"], "log12": ["로그Δ12"], "level_and_d12": ["수준", "Δ12"],
          "flow_per_1000hh": ["12개월합_천세대당"], "stock_per_1000hh_log1p": ["잔량_천세대당_log1p"],
          "netmig_per_1000pop": ["12개월합_천명당"], "pct6": ["6개월%"]}


def _per(df):
    return pd.PeriodIndex([f"{m // 100}-{m % 100:02d}" for m in df["month"].astype(int)], freq="M")


def load_sheets(s, panel="sido"):
    """(ml 2차, ref 1차, raw 취합본) 을 같은 행 순서(region, month)로 돌려준다."""
    sh = s["inputs"]["sheets"][panel]
    ml = pd.read_excel(S.path(s, "preprocessed_xlsx"), sheet_name=sh["ml"])
    ref = pd.read_excel(S.path(s, "preprocessed_xlsx"), sheet_name=sh["ref"])
    raw = pd.read_excel(S.path(s, "merged_xlsx"), sheet_name=sh["raw"])
    out = []
    for d in (ml, ref, raw):
        d = d.sort_values(["region", "month"]).reset_index(drop=True)
        d["P"] = _per(d)
        out.append(d)
    ml, ref, raw = out
    assert (ml[["region", "month"]].values == ref[["region", "month"]].values).all()
    assert (ml[["region", "month"]].values == raw[["region", "month"]].values).all()
    return ml, ref, raw


def colname(s, vid):
    return s["variables"][vid]["col"]


# ============================================================ ① 보완 셀 마스킹
def future_mask(s, ml, ref, raw, perturb_rng=None):
    """미래정보 보완 셀을 기준기간에서 찾아 공표 시차만큼 민 뒤 2차 값에서 지운다.
    perturb_rng 가 있으면 지우기 전에 그 셀을 난수로 바꾼다(교란 시험용: 결과가 같아야 한다).
    반환: (ml, 기록 DataFrame)"""
    ml = ml.copy()
    rec = []
    for vid in s["masking"]["columns"]:
        c = colname(s, vid)
        filled = ref[c].notna() & raw[c].isna()                     # 보완 셀 (기준기간)
        lag = RELEASE_MONTHLY[vid]["lag"]
        m2 = filled.groupby(ref["region"]).shift(lag, fill_value=False).astype(bool)   # 2차 행
        if perturb_rng is not None:
            ml.loc[m2, c] = ml.loc[m2, c] + perturb_rng.normal(0, 1, int(m2.sum()))
        ml.loc[m2, c] = np.nan
        rec.append({"변수": vid, "열": c, "보완셀(기준기간)": int(filled.sum()), "마스킹행(2차)": int(m2.sum()),
                    "공표시차": lag, "기준기간": _span(ref.loc[filled, "P"]), "결정월": _span(ml.loc[m2, "P"])})
    return ml, pd.DataFrame(rec)


def _span(ps):
    ps = sorted(set(ps))
    if not ps:
        return "없음"
    runs, a, b = [], ps[0], ps[0]
    for p in ps[1:]:
        if p == b + 1:
            b = p
        else:
            runs.append((a, b))
            a = b = p
    runs.append((a, b))
    return ", ".join(str(a) if a == b else f"{a}~{b}" for a, b in runs)


# ============================================================ ② 공표 시차 보정
def lag_adjust(s, ml, extra=False):
    """V060·V072 를 설정만큼 더 민다. extra=True 면 '익월 말' 공표 변수도 1개월 더 민다(강건성)."""
    ml = ml.copy()
    g = ml.groupby("region")
    for vid, k in s["lag_adjustments"].items():
        c = colname(s, vid)
        if c in ml.columns:
            ml[c] = g[c].shift(k)
    if extra:
        for vid in s["robustness_extra_lag"]["columns"]:
            c = colname(s, vid)
            if c in ml.columns:
                ml[c] = ml.groupby("region")[c].shift(1)
        if s["robustness_extra_lag"].get("rt_raw_columns") and "rt_columns" in s:   # v9: 실거래 원열
            for c in s["rt_columns"].values():
                if c in ml.columns:
                    ml[c] = ml.groupby("region")[c].shift(1)
            s["masking"]["rt_extra_lag_rows"] = 1      # 경계 마스킹도 1행 뒤로
    return ml


# ============================================================ ③ 변수 변환
def _roll12(x, g):
    return x.groupby(g).transform(lambda v: v.rolling(12, min_periods=12).sum())


def transform(s, ml, panel="sido"):
    """2차 값(마스킹·시차 보정 뒤) -> 입력표 X (region, P, 입력열...) 와 입력 명세표.
    V074 는 연결 경계를 가로지르는 12개월 변화를 결측으로 둔다."""
    ml = ml.copy()
    g = ml["region"]
    hh = ml[s["inputs"]["households_col"]].astype(float)
    pop = ml[s["inputs"]["population_col"]].astype(float)
    # 지역 유형별 CSI 한 열
    csi = pd.Series(np.nan, index=ml.index)
    for k, regs in s["csi_region_type"].items():
        csi = csi.mask(ml["region"].isin(regs), ml[s["csi_columns"][k]].astype(float))
    if panel == "gu":                # 서울 구는 모두 서울 유형
        csi = ml[s["csi_columns"]["V061"]].astype(float)
    ml["CSI_지역유형"] = csi
    X = ml[["region", "P"]].copy()
    spec = []
    for vid, v in s["variables"].items():
        c = v["col"]
        if c not in ml.columns:      # 서울 구 시트에 없는 시도 변수(V005·V013·V021~23·V031·V032·V037·V050·V072·V015·V017)
            continue
        x = ml[c].astype(float)
        tr = v["transform"]
        outs = {}
        if tr == "level":
            outs["수준"] = x
        elif tr == "d12":
            outs["Δ12"] = x - x.groupby(g).shift(12)
        elif tr == "log12":
            lx = np.log(x.where(x > 0))
            outs["로그Δ12"] = 100 * (lx - lx.groupby(g).shift(12))
        elif tr == "level_and_d12":
            outs["수준"] = x
            outs["Δ12"] = x - x.groupby(g).shift(12)
        elif tr == "flow_per_1000hh":
            outs["12개월합_천세대당"] = 1000 * _roll12(x, g) / hh
        elif tr == "stock_per_1000hh_log1p":
            outs["잔량_천세대당_log1p"] = np.log1p((1000 * x.clip(lower=0) / hh))
        elif tr == "netmig_per_1000pop":
            outs["12개월합_천명당"] = 1000 * _roll12(x, g) / pop
        elif tr == "pct6":
            outs["6개월%"] = 100 * (x / x.groupby(g).shift(6) - 1)
        else:
            raise ValueError(tr)
        if vid == "V074":   # 연결 경계(구계열 -> 신계열)를 가로지르는 12개월 변화
            first_new = S.per(s["masking"]["v074_link_first_new_row"])
            cross = (ml["P"] >= first_new) & (ml["P"] < first_new + 12)
            for k in outs:
                outs[k] = outs[k].mask(cross)
        for suf, val in outs.items():
            name = f"{vid}|{suf}"
            X[name] = val.values
            # v9: 출력 단위 공통 블록 포함 여부(in_common / common_outputs), 묶음 안 동일 개념(concept)
            in_common = bool(v.get("in_common", True))
            co = v.get("common_outputs")
            if co is not None and suf not in co:
                in_common = False
            spec.append({"입력": name, "ID": vid, "원열": c, "변환": tr, "block": v["block"], "동인": v["driver"],
                         "하위묶음": v["group"], "예상부호": v["sign"], "in_common": in_common,
                         "concept": v.get("concept", vid)})
    spec = pd.DataFrame(spec)
    if "rt_columns" in s and "extra_inputs" in s:
        Xr, spr = rt_inputs(s, ml, panel)
        X = pd.concat([X, Xr.drop(columns=["region", "P"])], axis=1)
        spec = pd.concat([spec, spr], ignore_index=True)
    return X, spec


# ============================================================ v9 실거래 별도 입력
def _roll(x, g, k):
    return x.groupby(g).transform(lambda v: v.rolling(k, min_periods=k).sum())


def rt_inputs(s, ml, panel="sido"):
    """실거래 두 입력(RMPI 밖 별도 입력, 지역 블록 처리). 2차 값(1개월 밀림) 기준.
    RT_mix = 100 x Δ12[ log(월세k / 전세k) ],  RT_act = 100 x Δ12[ log(1000 x 전체k / 세대수) ],  k = 창(3 또는 12)개월 합.
    전세 = 전체 − 월세 (2차 시트에 전세 열이 없음). 신고제 경계(rt_break_first_row)를 가로지르는 창은 결측.
    data_asof 기준 최신 rt_truncate_last_months 개월은 결측(사후 확인·전향 운용용. 2018~2025 평가에는 효과 없음)."""
    rc = s["rt_columns"]
    ei = s["extra_inputs"]
    win = 3 if str(ei.get("RT_mix", {}).get("window", "sum3")) == "sum3" else 12
    g = ml["region"]
    rent = ml[rc["월세"]].astype(float)
    allc = ml[rc["전체"]].astype(float)
    hh = ml[s["inputs"]["households_col"]].astype(float)
    asof = S.per(s["inputs"]["data_asof"]) if s["inputs"].get("data_asof") else None
    k_trunc = int(s["masking"].get("rt_truncate_last_months", 0) or 0)
    if asof is not None and k_trunc > 0:
        cut = ml["P"] >= asof - (k_trunc - 1)
        rent = rent.mask(cut)
        allc = allc.mask(cut)
    jeon = (allc - rent).where(allc > rent)
    rk, jk, ak = _roll(rent, g, win), _roll(jeon, g, win), _roll(allc, g, win)
    lmix = np.log(rk.where(rk > 0)) - np.log(jk.where(jk > 0))
    lact = np.log((1000 * ak / hh).where(ak > 0))
    mix = 100 * (lmix - lmix.groupby(g).shift(12))
    act = 100 * (lact - lact.groupby(g).shift(12))
    # 경계 마스킹: 비교 창 t-(win-1)-12 ~ t 가 첫 신규 행을 포함하는 행 = first .. first + win + 12 - 2
    first = S.per(s["masking"]["rt_break_first_row"]) + int(s["masking"].get("rt_extra_lag_rows", 0) or 0)
    n_mask = win + 12 - 1
    cross = (ml["P"] >= first) & (ml["P"] < first + n_mask)
    mix = mix.mask(cross)
    act = act.mask(cross)
    X = ml[["region", "P"]].copy()
    X["RT_mix|Δ12"] = mix.values
    X["RT_act|Δ12"] = act.values
    spec = pd.DataFrame([
        {"입력": "RT_mix|Δ12", "ID": "RT_mix", "원열": f"{rc['월세']} / {rc['전체']}", "변환": f"sum{win}_logit_d12", "block": "extra",
         "동인": 4, "하위묶음": "거래구성", "예상부호": "±", "in_common": False, "concept": "RT_mix"},
        {"입력": "RT_act|Δ12", "ID": "RT_act", "원열": rc["전체"], "변환": f"sum{win}_log_per1000hh_d12", "block": "extra",
         "동인": 4, "하위묶음": "거래활동", "예상부호": "±", "in_common": False, "concept": "RT_act"},
    ])
    return X, spec


# ============================================================ ④ 기준 지역 분해
def reference_regions(s, X, spec):
    """입력 열별 기준 지역: 최초 훈련기간(결정월)에서 그 입력이 한 지역이라도 관측된 달에 빠짐없이 관측된 시도.
    모든 지역이 함께 빠진 달은 선정에 영향을 주지 않는다."""
    lo, hi = S.per(s["timing"]["official_first_decision"]), S.per(s["timing"]["first_train_end"])
    sub = X[(X["P"] >= lo) & (X["P"] <= hi)]
    out = {}
    for name in spec.loc[spec["block"].isin(["regional", "extra"]), "입력"]:
        obs = sub.pivot(index="P", columns="region", values=name).notna()
        any_m = obs.any(axis=1)
        full = obs[any_m].all(axis=0)
        order = {r: i for i, r in enumerate(pd.unique(X["region"]))}
        out[name] = sorted(full.index[full].tolist(), key=order.get)
    return out


def configured_reference_regions(s, spec):
    regs = list(s["inputs"]["regions"])
    exc = s["rmpi"]["reference_regions"]["exceptions"]
    out = {}
    for _, r in spec[spec["block"].isin(["regional", "extra"])].iterrows():
        ex = exc.get(r["ID"], {}).get("exclude", [])
        out[r["입력"]] = [g for g in regs if g not in ex]
    return out


def decompose(s, X, spec, ref_regions):
    """지역 입력 -> (C: 월별 기준 지역 평균, D: 월내 편차 패널).
    기준 지역 중 하나라도 빠진 달은 평균과 그 평균을 쓰는 모든 편차를 결측으로 둔다.
    기준 지역 밖의 시도는 관측된 달에만 편차를 계산한다."""
    months = pd.PeriodIndex(sorted(X["P"].unique()), freq="M")
    C = pd.DataFrame(index=months)
    D = X[["region", "P"]].copy()
    for name in spec.loc[spec["block"].isin(["regional", "extra"]), "입력"]:
        wide = X.pivot(index="P", columns="region", values=name).reindex(months)
        refs = ref_regions[name]
        if not refs:
            C[name] = np.nan
            D[name] = np.nan
            continue
        mean = wide[refs].mean(axis=1).where(wide[refs].notna().all(axis=1))
        C[name] = mean
        dev = wide.sub(mean, axis=0)
        D[name] = dev.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([X["P"], X["region"]])).values
    for name in spec.loc[spec["block"] == "common", "입력"]:   # 전국 공통: 지역에 관계없이 같은 값
        wide = X.pivot(index="P", columns="region", values=name).reindex(months)
        C[name] = wide.bfill(axis=1).iloc[:, 0]
    return C, D


def price_trend(s, X, spec):
    """A 전용 가격 추세(V002·V026 6개월 변화율): 시도별 값과 월내 편차(17개 시도 평균 대비)."""
    cols = spec.loc[spec["block"] == "price_trend", "입력"].tolist()
    out = X[["region", "P"] + cols].copy()
    for c in cols:
        wide = X.pivot(index="P", columns="region", values=c)
        full = wide.notna().all(axis=1)
        mean = wide.mean(axis=1).where(full)
        dev = wide.sub(mean, axis=0)
        out[c + "|월내편차"] = dev.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([X["P"], X["region"]])).values
    return out


def build(s, panel="sido", perturb_rng=None, extra_lag=False):
    """①~④를 한 번에: 반환 dict(ml, mask_log, X, spec, ref_regions, C, D, A_price)"""
    ml, ref, raw = load_sheets(s, panel)
    ml, mask_log = future_mask(s, ml, ref, raw, perturb_rng=perturb_rng)
    ml = lag_adjust(s, ml, extra=extra_lag)
    X, spec = transform(s, ml, panel)
    refs = configured_reference_regions(s, spec) if panel == "sido" else reference_regions(s, X, spec)
    C, D = decompose(s, X, spec, refs)
    A = price_trend(s, X, spec)
    return dict(ml=ml, mask_log=mask_log, X=X, spec=spec, ref_regions=refs, C=C, D=D, A_price=A)
