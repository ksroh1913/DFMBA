# -*- coding: utf-8 -*-
"""
[리뷰] experiment_TH 브랜치 TH_v19 실험의 최종 입력 변수·정상성·단순 기준 비교. docs/리뷰_TH_v19_모형입력_학습방법.md 의 근거 산출물.

TH_v19 의 특징 생성 코드(features_TH_v19.py)를 그 실험의 입력(TH_v11/input_TH_v11/*.csv)에 그대로 돌려,
학습 시점별로 실제 선택된 변수(관측률 >= 0.7, 비상수)를 뽑고, 시도별 ADF·KPSS 정상성 검정과 mom1·0 예측 기준 대비 MAE 를 계산한다.
TH 브랜치 파일은 --th-dir(experiment_TH 체크아웃 경로)에서 읽거나, 없으면 GitHub raw 에서 고정 커밋으로 내려받는다.

출력 (analysis/output/)
  review_TH_v19_입력변수.xlsx
    1_최종변수_시도   variant(base=V18 / short=V19a / short_nolevel=V19b) x 학습시점(2018-01·2022-01·2025-10) 별 선택 여부·관측률
    2_최종변수_서울   같은 표 (서울 25개 구)
    3_입력열_원자료   입력 CSV 의 열 (원변수 ID·열 이름·관측 범위)
    4_정상성_시도     V19b 2025-10 선택 특징 + 모멘텀 + 타깃(g·U·D) 의 시도별 ADF·KPSS 요약 (2016-01~2025-12)
    5_기준비교        Y1 예측(V19a·V19b 네 모형) vs mom1·0·last6 기준의 MAE·방향 일치 (같은 행)
    6_요약
사용: PYTHONUTF8=1 python analysis/review_th_v19_inputs.py [--th-dir <experiment_TH 경로>]
"""

import argparse
import os
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "analysis"))
from review_model_inputs import test_series  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(BASE, "analysis", "output")
XLSX = os.path.join(OUT, "review_TH_v19_입력변수.xlsx")
TH_COMMIT = "e8c294a64403300ea44903da851952852dc0ca8a"      # experiment_TH head 2026-10-08 (리뷰 시점에 고정)
RAW = f"https://raw.githubusercontent.com/ksroh1913/DFMBA/{TH_COMMIT}/ml/experiments_TH_v1"
FILES = {"features_TH_v19.py": "TH_v19/ml/features_TH_v19.py",
         "provinces_aligned_candidates_TH_v11.csv": "TH_v11/input_TH_v11/provinces_aligned_candidates_TH_v11.csv",
         "seoul_aligned_candidates_TH_v11.csv": "TH_v11/input_TH_v11/seoul_aligned_candidates_TH_v11.csv",
         "pred_short.csv": "TH_v19/analysis/output_TH_v19_short/predictions_TH_v19.csv",
         "pred_short_nolevel.csv": "TH_v19/analysis/output_TH_v19_short_nolevel/predictions_TH_v19.csv"}
VARIANTS = {"base": "V18(기준)", "short": "V19a(+3·6개월 변화)", "short_nolevel": "V19b(+3·6개월, 금리 수준 제외)"}
ORIGINS = [201801, 202201, 202510]


def fetch(th_dir):
    """TH 파일 5개의 로컬 경로 dict. th_dir 가 있으면 거기서, 없으면 내려받아 analysis/output/_th_v19_cache/ 에 둔다."""
    paths = {}
    cache = os.path.join(OUT, "_th_v19_cache")
    for name, rel in FILES.items():
        if th_dir:
            paths[name] = os.path.join(th_dir, "ml", "experiments_TH_v1", rel)
        else:
            os.makedirs(cache, exist_ok=True)
            p = os.path.join(cache, name)
            if not os.path.exists(p):
                urllib.request.urlretrieve(f"{RAW}/{rel}", p)
            paths[name] = p
    sys.path.insert(0, os.path.dirname(paths["features_TH_v19.py"]))
    return paths


def feature_tables(F, paths):
    tabs, raws, engineered = {}, {}, {}
    for panel in ("provinces", "seoul"):
        raw = pd.read_csv(paths[f"{panel}_aligned_candidates_TH_v11.csv"], float_precision="round_trip", low_memory=False)
        raws[panel] = raw
        rows = {}
        for variant in VARIANTS:
            d, econ, mom = F.engineer(raw, panel, variant)
            d = d[d[["g", "U", "D"]].notna().all(axis=1)].reset_index(drop=True)
            engineered[(panel, variant)] = (d, econ, mom)
            for origin in ORIGINS:
                tr = F.history(d, origin)
                for kind, cands in (("경제", econ), ("모멘텀", mom)):
                    _, notes = F.select(tr, cands)
                    for n in notes:
                        key = n["feature"]
                        r = rows.setdefault(key, {"특징": key, "종류": kind, "원열": key.split("|")[0], "원변수ID": key.split("|")[0][:4] if key.startswith("V") else ("V001" if key.startswith("M10") else key.split("_")[1] if key.startswith("P15") else ""),
                                                  "변환": key.split("|", 1)[1] if "|" in key else key})
                        r[f"{VARIANTS[variant]}|{origin}"] = ("선택" if n["selected"] else f"탈락(관측률 {n['coverage']:.2f}, 고유값 {n['unique']})")
        tab = pd.DataFrame(rows.values())
        for variant in VARIANTS:
            cols = [c for c in tab.columns if c.startswith(VARIANTS[variant])]
            tab[f"{VARIANTS[variant]}|후보"] = tab[cols].notna().any(axis=1)
        tabs[panel] = tab
    return tabs, raws, engineered


def raw_columns(raws):
    rows = []
    for panel, raw in raws.items():
        for c in raw.columns:
            x = pd.to_numeric(raw[c], errors="coerce") if c not in ("panel", "region") else None
            rows.append({"표": panel, "열": c, "원변수ID": c[3:7] if c.startswith("서울_") else c[:4], "첫관측월": int(raw.loc[x.notna(), "month"].min()) if x is not None and x.notna().any() else "",
                         "결측률": round(float(x.isna().mean()), 3) if x is not None else ""})
    return pd.DataFrame(rows)


def stationarity(F, engineered):
    d, econ, mom = engineered[("provinces", "short_nolevel")]
    tr = F.history(d, 202510)
    cols, _ = F.select(tr, econ)
    mcols, _ = F.select(tr, mom)
    win = tr[(tr.month >= 201601) & (tr.month <= 202512)]
    rows = []
    for c in cols + mcols + ["g", "U", "D"]:
        wide = win.pivot(index="month", columns="region", values=c)
        wide.index = pd.PeriodIndex([f"{m // 100}-{m % 100:02d}" for m in wide.index], freq="M")
        res = [test_series(wide[r], za=False) for r in wide.columns]
        tested = [r for r in res if r["판정"] not in ("", "검정 생략")]
        rows.append({"특징": c, "종류": "경제" if c in cols else ("모멘텀" if c in mcols else "타깃"), "검정시도수": len(tested),
                     "ADF기각비율": round(float(np.mean([r["ADF_c_p"] < 0.05 for r in tested])), 2) if tested else np.nan,
                     "KPSS기각비율": round(float(np.mean([r["KPSS_c_기각"] for r in tested])), 2) if tested else np.nan,
                     "다수판정": pd.Series([r["판정"] for r in tested]).mode().iloc[0] if tested else "검정 생략",
                     "생략사유": "; ".join(sorted({r["처리"] for r in res if r["판정"] in ("", "검정 생략")})),
                     "AR1_중앙값": round(float(np.nanmedian([r["AR1"] for r in res])), 3)})
    return pd.DataFrame(rows)


def baseline_compare(F, engineered, paths):
    rows = []
    for panel in ("provinces", "seoul"):
        d, _, _ = engineered[(panel, "base")]
        base = d[["region", "month", "g", "mom1", "M10_rent_growth_6m"]].rename(columns={"M10_rent_growth_6m": "last6"})
        for tag in ("short", "short_nolevel"):
            p = pd.read_csv(paths[f"pred_{tag}.csv"])
            p = p[(p.panel == panel) & (p.task == "Y1 Growth")].merge(base, on=["region", "month"])
            assert np.allclose(p.actual, p.g)
            for per, (a, b) in {"2021~2025": (2021, 2025), "2018~2025": (2018, 2025)}.items():
                q = p[(p.year >= a) & (p.year <= b)]
                for m, g in q.groupby("model"):
                    rows.append({"표": panel, "variant": VARIANTS[tag], "기간": per, "모형": m, "행수": len(g),
                                 "MAE_모형": (g.actual - g.prediction).abs().mean(), "MAE_mom1": (g.actual - g.mom1).abs().mean(),
                                 "MAE_zero": g.actual.abs().mean(), "MAE_last6": (g.actual - g.last6).abs().mean(),
                                 "방향일치_모형": ((g.actual > 0) == (g.prediction > 0)).mean(), "방향일치_mom1": ((g.actual > 0) == (g.mom1 > 0)).mean()})
    r = pd.DataFrame(rows)
    r["모형/mom1"] = r["MAE_모형"] / r["MAE_mom1"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th-dir", default=None)
    args = ap.parse_args()
    paths = fetch(args.th_dir)
    import features_TH_v19 as F
    tabs, raws, engineered = feature_tables(F, paths)
    rc = raw_columns(raws)
    st = stationarity(F, engineered)
    bc = baseline_compare(F, engineered, paths)
    summ = []
    for panel, tab in tabs.items():
        for variant in VARIANTS:
            for origin in ORIGINS:
                col = f"{VARIANTS[variant]}|{origin}"
                sel = tab[col].eq("선택")
                summ.append({"표": panel, "variant": VARIANTS[variant], "학습시점": origin, "후보": int(tab[col].notna().sum()), "선택": int(sel.sum()),
                             "경제 선택": int((sel & (tab["종류"] == "경제")).sum()), "모멘텀 선택": int((sel & (tab["종류"] == "모멘텀")).sum()),
                             "지역더미": raws[panel].region.nunique(), "총 입력": int(sel.sum()) + raws[panel].region.nunique()})
    summ = pd.DataFrame(summ)
    econ_st = st[st["종류"] == "경제"]
    verdict_cnt = econ_st["다수판정"].value_counts().rename_axis("다수판정").reset_index(name="특징 수")
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        tabs["provinces"].to_excel(w, sheet_name="1_최종변수_시도", index=False)
        tabs["seoul"].to_excel(w, sheet_name="2_최종변수_서울", index=False)
        rc.to_excel(w, sheet_name="3_입력열_원자료", index=False)
        st.to_excel(w, sheet_name="4_정상성_시도", index=False)
        bc.round(4).to_excel(w, sheet_name="5_기준비교", index=False)
        summ.to_excel(w, sheet_name="6_요약", index=False)
        verdict_cnt.to_excel(w, sheet_name="6_요약", index=False, startrow=len(summ) + 3)
    print(f"저장: {XLSX}")
    print(summ.to_string(index=False))
    print(verdict_cnt.to_string(index=False))
    print(bc[bc["기간"] == "2021~2025"][["표", "variant", "모형", "MAE_모형", "MAE_mom1", "모형/mom1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
