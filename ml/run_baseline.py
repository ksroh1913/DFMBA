# -*- coding: utf-8 -*-
"""
[ML 1차 실험] 아파트 월세 6개월 방향예측 + 급등락 조기경보 (연구설계 v2 기준선)

입력: 데이터취합_전처리_*.xlsx 의 2차_공표시점반영(ML용) 시트 (가장 최근 파일)
출력: ml/output/결과_요약.csv, ml/output/결과_fold별.csv, ml/output/임계값.csv

시점 정의 (공표 시차 반영)
  - 2차 시트의 t월 행 = t월 말에 알 수 있던 정보. 이때 공표된 마지막 월세지수는 R(t-1).
  - 그래서 예측 기준점을 R(t-1)로 두고 그로부터 6개월 경로를 본다.
      g = 100 x (R(t+5)/R(t-1) - 1)                      방향 (상승/보합/하락)
      U = max_h 100 x (R(t-1+h)/R(t-1) - 1), h=1..6        급등: U > δ↑
      D = min_h ...                                        급락: D < -δ↓
  - 목표값은 공식 지수 구간(Y_평가사용가능 = 1, 2015.06~)만 쓴다. 연결·역산값은 쓰지 않는다.

임계값 (학습 구간에서만 정하고 시험 구간에는 고정)
  ε  = 학습 구간 |g| 의 1/3 분위       -> 보합이 대략 1/3
  δ↑ = 학습 구간 U 의 80% 분위         -> 급등 = 상위 20%
  δ↓ = 학습 구간 D 의 20% 분위의 부호 반전 -> 급락 = 하위 20%

변수 묶음
  M1 가격 관성 : 월세 6개월 변화, 월세 가속도, 매매가격 6개월 변화
  M2 구조요인 : 2차 시트의 나머지 변수(가격 이력 V002·V026 제외)를 12개월 변화 등으로 가공
  M3 통합     : M1 + M2
  실거래 건수(V006·V007)는 전국 수집이 끝나지 않아 이번 실험에서 뺀다.

검증: 시간순 확장 창. 시험연도 Y 의 학습 = 결정월 t <= (Y-1)년 7월 (6개월 목표가 시험 구간과 겹치지 않게)
모형: 부분통합 로짓(지역 고정효과 + L2) / Extra-Trees (한국은행 조기경보모형 비교용) / 기준(학습 빈도)
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "ml", "output")

PANELS = {
    "17시도": ("17시도_2차_공표시점반영(ML용)", "Y_V001_월세통합가격지수(구지수연결)"),
    "서울25구": ("서울25구_2차_공표시점반영(ML용)", "Y_V001_월세통합가격지수(권역역산_학습용)"),
}
H = 6
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]   # 연구설계는 2021~2023. 2024·2025는 추가 확인
EXCLUDE = ("V006_", "V007_")                  # 실거래 (수집 중)
PRICE_HISTORY = ("V002_", "V026_")            # M2 에서 빼는 가격 이력
MAX_MISSING = 0.30                            # 표본 기간 결측이 이보다 많은 변수는 뺀다
RATE_TOKENS = ("%", "지수", "동향", "CSI", "÷", "천세대당", "전국대비", "비중", "순환변동치", "심리", "=100",
               "고용률", "태도")
FLOW_IDS = ("V021", "V022", "V023", "V035", "V036")


def latest_xlsx():
    files = sorted(glob.glob(os.path.join(BASE, "데이터취합_전처리_*.xlsx")))
    return files[-1]


def vid_of(col):
    base = col[3:] if col.startswith("서울_") else col
    return base[:4]


def engineer(df, ycol):
    """지역별 시계열에서 목표값과 M1·M2 변수를 만든다. 반환: (표, M1 열, M2 열)"""
    df = df.sort_values(["region", "month"]).copy()
    g = df.groupby("region", group_keys=False)
    R = df[ycol].where(df["Y_평가사용가능"] == 1)
    Rg = R.groupby(df["region"])
    base = Rg.shift(1)
    path = pd.concat([100 * (Rg.shift(1 - h) / base - 1) for h in range(1, H + 1)], axis=1)
    out = df[["region", "month"]].copy()
    out["g"] = path.iloc[:, -1]
    out["U"] = path.max(axis=1, skipna=False)
    out["D"] = path.min(axis=1, skipna=False)

    # M1 가격 관성 (R 은 t-1 까지만 사용)
    out["M1_월세_6개월변화"] = 100 * (Rg.shift(1) / Rg.shift(7) - 1)
    out["M1_월세_1개월변화"] = 100 * (Rg.shift(1) / Rg.shift(2) - 1)
    out["M1_월세_가속도"] = 100 * (Rg.shift(1) / Rg.shift(4) - Rg.shift(4) / Rg.shift(7))
    v002 = [c for c in df.columns if c.startswith("V002_")][0]
    out["M1_매매_6개월변화"] = 100 * (df[v002] / g[v002].shift(6) - 1)   # 2차 시트 값은 이미 시차 반영
    m1 = [c for c in out.columns if c.startswith("M1_")]

    # M2 구조요인
    pop = [c for c in df.columns if c.startswith("V009_")]
    m2 = []
    skip = ("panel", "region", "region_code", "month", "Y_평가사용가능", ycol)
    for c in df.columns:
        if c in skip or c.startswith(("Y_", "D_")) or c.startswith(EXCLUDE) or c.startswith(PRICE_HISTORY):
            continue
        x = df[c].astype(float)
        xg = x.groupby(df["region"])
        v = vid_of(c)
        if v == "V012" and pop:   # 순이동: 인구 천명당 12개월 누적
            rate = 1000 * xg.transform(lambda s: s.rolling(12, min_periods=12).sum()) / df[pop[0]]
            out[f"{c}|천명당12개월"] = rate
            m2.append(f"{c}|천명당12개월")
        elif v in FLOW_IDS:       # 월 흐름: 12개월 누적의 로그와 그 12개월 변화
            s12 = np.log1p(xg.transform(lambda s: s.rolling(12, min_periods=12).sum()))
            out[f"{c}|12개월누적log"] = s12
            out[f"{c}|12개월누적log_12m차"] = s12 - s12.groupby(df["region"]).shift(12)
            m2 += [f"{c}|12개월누적log", f"{c}|12개월누적log_12m차"]
        elif v == "V038":         # 미분양 재고: 0 이 많아 log1p
            lx = np.log1p(x.clip(lower=0))
            out[f"{c}|log"] = lx
            out[f"{c}|log_12m차"] = lx - lx.groupby(df["region"]).shift(12)
            m2 += [f"{c}|log", f"{c}|log_12m차"]
        elif any(t in c for t in RATE_TOKENS):   # 비율·지수·금리: 수준과 12개월 차
            out[f"{c}|수준"] = x
            out[f"{c}|12m차"] = x - xg.shift(12)
            m2 += [f"{c}|수준", f"{c}|12m차"]
        else:                     # 금액·인구 등 저량: 12개월 로그 변화율
            lx = np.log(x.where(x > 0))
            out[f"{c}|12m로그변화"] = 100 * (lx - lx.groupby(df["region"]).shift(12))
            m2.append(f"{c}|12m로그변화")
    return out, m1, m2


def thresholds(tr):
    return {"eps": tr["g"].abs().quantile(1 / 3), "up": tr["U"].quantile(0.8), "dn": -tr["D"].quantile(0.2)}


def labels(d, th):
    direction = np.where(d["g"] > th["eps"], "상승", np.where(d["g"] < -th["eps"], "하락", "보합"))
    return direction, (d["U"] > th["up"]).astype(int).values, (d["D"] < -th["dn"]).astype(int).values


def models():
    return {
        "로짓": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     LogisticRegression(C=0.1, max_iter=3000)),
        "ET": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    ExtraTreesClassifier(n_estimators=400, min_samples_leaf=20, max_features="sqrt",
                                                         n_jobs=4, random_state=0)),
    }


TARGET_CLASS = {"방향": "상승", "급등": 1, "급락": 1}   # SHAP 으로 설명하는 확률
FEATURES = {}
_DRIVERS = None


def driver_of(feature):
    """변수 -> 데이터사전 1차 동인. 가격 관성(M1)은 해당 원변수의 동인, 지역 더미는 별도 묶음"""
    global _DRIVERS
    if _DRIVERS is None:
        sys.path.insert(0, BASE)
        from variables import drivers
        _DRIVERS = {k: v[0] for k, v in drivers().items()}
    if feature.startswith("지역_"):
        return "지역 고정효과"
    if feature.startswith("M1_월세"):
        return _DRIVERS.get("V001", "")
    if feature.startswith("M1_매매"):
        return _DRIVERS.get("V002", "")
    return _DRIVERS.get(vid_of(feature.split("|")[0]), "기타")


def explain(model, mname, Xtr, Xte, fnames, cls, task, te, panel, yr):
    """표본외 SHAP. ET 는 확률 단위, 로짓은 로그오즈 단위."""
    import shap
    k = cls.index(TARGET_CLASS[task])
    steps = model.named_steps
    imp = steps["simpleimputer"]
    if mname == "ET":
        A = imp.transform(Xte)
        sv = shap.TreeExplainer(steps["extratreesclassifier"]).shap_values(A, check_additivity=False)
    else:
        sc, lr = steps["standardscaler"], steps["logisticregression"]
        B = sc.transform(imp.transform(Xtr))
        A = sc.transform(imp.transform(Xte))
        sv = shap.LinearExplainer(lr, B).shap_values(A)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, k]
    elif len(cls) == 2 and k == 0:
        sv = -sv
    S = pd.DataFrame(sv, columns=fnames)
    meta = dict(패널=panel, 모형=mname, 과제=task, 시험연도=yr)
    feat = pd.DataFrame({**meta, "변수": fnames, "평균|SHAP|": np.abs(sv).mean(0), "평균SHAP": sv.mean(0)})
    grp = S.T.groupby([driver_of(f) for f in fnames]).sum().T
    grp.insert(0, "month", te.month.values)
    grp.insert(0, "region", te.region.values)
    for c, v in reversed(list(meta.items())):
        grp.insert(0, c, v)
    return feat, grp


def run_panel(name, sheet, ycol, xlsx, shap_rows):
    print(f"\n===== {name} =====", flush=True)
    raw = pd.read_excel(xlsx, sheet_name=sheet)
    d, m1, m2 = engineer(raw, ycol)
    d = d[d[["g", "U", "D"]].notna().all(axis=1) & d[m1].notna().all(axis=1)].reset_index(drop=True)
    # 표본 기간 결측이 많거나 값이 하나뿐인 변수 제외
    miss = d[m2].isna().mean()
    m2 = [c for c in m2 if miss[c] <= MAX_MISSING and d[c].nunique() > 1]
    dropped = sorted(set(miss.index) - set(m2))
    print(f"표본 {len(d)}행 (결정월 {d.month.min()}~{d.month.max()}), M1 {len(m1)}개, M2 {len(m2)}개 변수"
          f" (결측·상수로 제외 {len(dropped)}개)")
    regions = pd.get_dummies(d["region"], prefix="지역").astype(float)
    sets = {"M1": m1, "M2": m2, "M3": m1 + m2}
    FEATURES[name] = m1 + m2

    rows, preds, ths = [], [], []
    for yr in TEST_YEARS:
        tr = d[d.month <= (yr - 1) * 100 + 7]
        te = d[(d.month >= yr * 100 + 1) & (d.month <= yr * 100 + 12)]
        if te.empty:
            continue
        th = thresholds(tr)
        ths.append({"패널": name, "시험연도": yr, **{k: round(v, 3) for k, v in th.items()}})
        ytr, utr, dtr = labels(tr, th)
        yte, ute, dte = labels(te, th)
        base_p = {"방향": pd.Series(ytr).value_counts(normalize=True), "급등": utr.mean(), "급락": dtr.mean()}
        for setname, cols in sets.items():
            fnames = cols + list(regions.columns)
            Xtr = pd.concat([tr[cols], regions.loc[tr.index]], axis=1).values
            Xte = pd.concat([te[cols], regions.loc[te.index]], axis=1).values
            for mname, mk in models().items():
                for task, (a, b) in {"방향": (ytr, yte), "급등": (utr, ute), "급락": (dtr, dte)}.items():
                    if len(np.unique(a)) < 2:
                        continue
                    m = mk().fit(Xtr, a)
                    p = m.predict_proba(Xte)
                    cls = list(m.classes_)
                    if setname == "M3":   # 표본외 SHAP: 이 시험연도 모형으로 시험 행을 설명
                        shap_rows.append(explain(m, mname, Xtr, Xte, fnames, cls, task, te, name, yr))
                    rec = pd.DataFrame({"패널": name, "시험연도": yr, "변수": setname, "모형": mname, "과제": task,
                                        "region": te.region.values, "month": te.month.values, "정답": b})
                    for i, c in enumerate(cls):
                        rec[f"p_{c}"] = p[:, i]
                    preds.append(rec)
        # 기준(학습 빈도) 예측
        for task, b in {"방향": yte, "급등": ute, "급락": dte}.items():
            rec = pd.DataFrame({"패널": name, "시험연도": yr, "변수": "-", "모형": "기준(학습빈도)", "과제": task,
                                "region": te.region.values, "month": te.month.values, "정답": b})
            if task == "방향":
                for c in ("보합", "상승", "하락"):
                    rec[f"p_{c}"] = base_p["방향"].get(c, 0)
            else:
                rec["p_0"], rec["p_1"] = 1 - base_p[task], base_p[task]
            preds.append(rec)
    return pd.concat(preds, ignore_index=True), ths, dropped


def score(g):  # noqa: C901
    task = g["과제"].iloc[0]
    y = g["정답"].values
    out = {"시험행수": len(g)}
    if task == "방향":
        cls = ["보합", "상승", "하락"]
        P = g[[f"p_{c}" for c in cls]].fillna(0).values
        P = P / P.sum(axis=1, keepdims=True)
        out["정답비율(상승/보합/하락)"] = "/".join(f"{(y == c).mean():.2f}" for c in ("상승", "보합", "하락"))
        out["정확도"] = accuracy_score(y, np.array(cls)[P.argmax(1)])
        out["로그손실"] = log_loss(y, P, labels=cls)
        present = [c for c in cls if (y == c).any()]
        out["AUC"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=cls) if len(present) == 3 else np.nan
    else:
        y = y.astype(int)
        p = g["p_1"].values.astype(float)
        out["사건비율"] = y.mean()
        out["AUC"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
        out["PR-AUC"] = average_precision_score(y, p) if y.sum() > 0 else np.nan
        out["로그손실"] = log_loss(y, np.c_[1 - p, p], labels=[0, 1])
    return pd.Series(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    xlsx = latest_xlsx()
    print(f"입력: {os.path.basename(xlsx)}")
    allp, allth, notes, shap_rows = [], [], [], []
    for name, (sheet, ycol) in PANELS.items():
        p, th, dropped = run_panel(name, sheet, ycol, xlsx, shap_rows)
        allp.append(p)
        allth += th
        notes.append({"패널": name, "제외변수": "; ".join(dropped)})
    P = pd.concat(allp, ignore_index=True)
    P.to_csv(os.path.join(OUT, "예측값.csv"), index=False, encoding="utf-8-sig")   # 채점 전에 먼저 저장
    summarize(P)
    pd.DataFrame(allth).to_csv(os.path.join(OUT, "임계값.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(notes).to_csv(os.path.join(OUT, "제외변수.csv"), index=False, encoding="utf-8-sig")
    pd.concat([f for f, _ in shap_rows]).to_csv(os.path.join(OUT, "SHAP_변수별.csv"), index=False, encoding="utf-8-sig")
    pd.concat([g for _, g in shap_rows]).to_csv(os.path.join(OUT, "SHAP_동인별_행.csv"), index=False,
                                                encoding="utf-8-sig")
    # 실험에 쓴 변수와 가공 방식
    pd.DataFrame([{"패널": n, "변수": c, "묶음": "M1" if c.startswith("M1_") else "M2", "동인": driver_of(c)}
                  for n, cols in FEATURES.items() for c in cols]).to_csv(
        os.path.join(OUT, "사용변수.csv"), index=False, encoding="utf-8-sig")
    print("\n임계값:\n" + pd.DataFrame(allth).to_string(index=False))


def summarize(P):
    """시험연도별 점수와 요약. 요약의 AUC 는 연도별 AUC 의 평균(주 지표).
    연도를 합쳐 계산한 AUC(통합AUC)는 해마다 기준 비율이 달라 기준 모형도 0.5 가 아니게 되므로 참고용."""
    keys = ["패널", "과제", "변수", "모형"]
    fold = pd.DataFrame([{**dict(zip(keys + ["시험연도"], k)), **score(g).to_dict()}
                         for k, g in P.groupby(keys + ["시험연도"])])
    rows = []
    for span, lo, hi in (("2021~2023(설계안)", 2021, 2023), ("2021~2025", 2021, 2025)):
        sub = P[(P["시험연도"] >= lo) & (P["시험연도"] <= hi)]
        f = fold[(fold["시험연도"] >= lo) & (fold["시험연도"] <= hi)]
        for k, g in sub.groupby(keys):
            ff = f
            for kk, vv in zip(keys, k):
                ff = ff[ff[kk] == vv]
            pooled = score(g)
            rec = {**dict(zip(keys, k)), "시험구간": span, "시험행수": len(g),
                   "AUC": ff["AUC"].mean(), "AUC_연도최저": ff["AUC"].min(), "AUC_연도최고": ff["AUC"].max(),
                   "통합AUC": pooled["AUC"], "로그손실": ff["로그손실"].mean()}
            if k[1] == "방향":
                rec["정확도"] = ff["정확도"].mean()
            else:
                rec["PR-AUC"] = ff["PR-AUC"].mean()
                rec["사건비율"] = pooled["사건비율"]
            rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "결과_요약.csv"), index=False, encoding="utf-8-sig")
    fold.to_csv(os.path.join(OUT, "결과_fold별.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for task in ("방향", "급등", "급락"):
        cols = keys + ["시험구간", "AUC", "AUC_연도최저", "AUC_연도최고", "통합AUC"] + (
            ["정확도"] if task == "방향" else ["PR-AUC", "사건비율"])
        print(f"\n[{task}]")
        print(summ[summ["과제"] == task][cols].round(3).to_string(index=False))
    return summ


if __name__ == "__main__":
    if "--score-only" in sys.argv:   # 저장된 예측값으로 점수만 다시 계산
        summarize(pd.read_csv(os.path.join(OUT, "예측값.csv")))
    else:
        main()
