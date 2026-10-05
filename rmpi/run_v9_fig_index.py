# -*- coding: utf-8 -*-
"""[그림] 실제 월세지수 흐름과 대표 모형의 예측 지수 흐름. 예측 변화율 Ĝ(T,h) 를 결정월 직전 지수에 적용해 예측 수준 R̂(T−1+h) = R(T−1)·(1+Ĝ/100) 으로 바꿔
실제 지수와 같은 축에 그린다. 전국 h=1·6 (mom1, B cv_min, 보정 M2), 서울·세종 h=6 (mom1=M0, M2).
출력: rmpi/output_v9/v9_fig_지수흐름.csv (긴 형식), fig6_6_지수흐름.png. RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D, settings as S, targets as T, viz as V  # noqa: E402


def levels(pred, R, h):
    """pred: P, yhat (G, %). R: Period -> index level. 반환 Series(month = P-1+h -> R̂)."""
    out = {}
    for _, r in pred.iterrows():
        P = pd.Period(str(r["P"]), freq="M"); base = R.get(P - 1)
        if base is not None and pd.notna(base) and pd.notna(r["yhat"]):
            out[P - 1 + h] = base * (1 + r["yhat"] / 100)
    return pd.Series(out).sort_index()


def main():
    s = S.load(); out = os.path.join(BASE, s["output_dir"])
    b = D.build(s, "sido"); ml = b["ml"]
    Rn = T.rone(S.path(s, "raw_rent"), "전국").astype(float)      # 전국 공식 월세지수(월별 수준)
    Rn.index = pd.PeriodIndex(Rn.index, freq="M")
    ycol = s["inputs"]["target"]["sido"]
    Rr = ml.set_index(["region", "P"])[ycol].astype(float)
    pn = pd.read_csv(os.path.join(out, "v9_s1_예측값_전국.csv")); pc = pd.read_csv(os.path.join(out, "v9_s2_예측값.csv"))
    rows = []
    start, end = pd.Period("2017-07", "M"), pd.Period("2026-09", "M")
    for mth, val in Rn[(Rn.index >= start) & (Rn.index <= end)].items():
        rows.append({"패널": "전국", "h": 0, "계열": "실제", "month": str(mth), "value": val})
    for h in (1, 6):
        series = {"mom1": pn[(pn["모형"] == "naive_mom1") & (pn["h"] == h)], "B": pn[(pn["모형"] == "ridge_B_cvmin") & (pn["h"] == h)],
                  "M2": pc[(pc["단위"] == "전국") & (pc["모형"] == "corr_M2") & (pc["h"] == h)]}
        for nm, df in series.items():
            for mth, val in levels(df, Rn.to_dict(), h).items():
                rows.append({"패널": "전국", "h": h, "계열": nm, "month": str(mth), "value": val})
    for reg in ("서울", "세종"):
        R = Rr.loc[reg]; R.index = pd.PeriodIndex(R.index, freq="M")
        for mth, val in R[(R.index >= start) & (R.index <= end)].items():
            rows.append({"패널": reg, "h": 0, "계열": "실제", "month": str(mth), "value": val})
        for nm, model in (("mom1", "corr_M0"), ("M2", "corr_M2")):
            df = pc[(pc["단위"] == "패널") & (pc["region"] == reg) & (pc["모형"] == model) & (pc["h"] == 6)]
            for mth, val in levels(df, R.to_dict(), 6).items():
                rows.append({"패널": reg, "h": 6, "계열": nm, "month": str(mth), "value": val})
    long = pd.DataFrame(rows); long.to_csv(os.path.join(out, "v9_fig_지수흐름.csv"), index=False, encoding="utf-8-sig")
    # 그림
    V.setup()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    panels = [("전국", 1, "전국, h=1 (한 달 앞)"), ("전국", 6, "전국, h=6 (여섯 달 앞)"), ("서울", 6, "서울, h=6"), ("세종", 6, "세종, h=6")]
    colors = {"mom1": V.GRAY, "B": V.SERIES[0], "M2": V.SERIES[1]}
    for ax, (pan, h, title) in zip(axes.ravel(), panels):
        d = long[long["패널"] == pan]
        act = d[d["계열"] == "실제"].set_index("month")["value"]; act.index = pd.PeriodIndex(act.index, freq="M")
        ax.plot(act.index.to_timestamp(), act.values, color=V.INK, lw=2, label="실제 지수")
        for nm in ("mom1", "B", "M2"):
            q = d[(d["계열"] == nm) & (d["h"] == h)]
            if q.empty:
                continue
            q = q.set_index("month")["value"]; q.index = pd.PeriodIndex(q.index, freq="M")
            ax.plot(q.index.to_timestamp(), q.values, color=colors[nm], lw=1.4, label={"mom1": "단순 기준 mom1", "B": "B (추세+RMPI)", "M2": "보정 M2"}[nm])
        ax.set_title(title, loc="left", fontsize=10.5); ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("실제 월세지수와 대표 모형의 예측 지수 (예측 변화율을 결정월 직전 지수에 적용한 수준, 평가 구간 2018.01~2025.12 + 2026 사후 확인)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(out, "fig6_6_지수흐름.png"), dpi=150); plt.close(fig)
    print("wrote", len(long), "rows")


if __name__ == "__main__":
    main()
