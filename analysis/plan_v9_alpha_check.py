# -*- coding: utf-8 -*-
"""8차 전국 Ridge 의 alpha 민감도(사후 진단). 9차 설계안 2장의 근거.
재실행 주의: 8차 설정의 입력(데이터취합_전처리_20261004.xlsx sha d4cb52e7a2b7c982, 데이터취합_20261004.xlsx 8e65ad87deeab22c)은 이전 라운드 자료 정리 때
저장소에서 지웠다. analysis/output/plan_v9_alpha_check.csv 는 그 실행의 동결 결과이며, 다시 돌리려면 git 이력(커밋 de16c0d)에서 두 파일을 복원한다.

8차 설정(config/plan_v8_settings.yaml, 20261004 입력)으로 정보군 A·B 의 전국 Ridge 를 alpha 격자마다 고정해 2018.01~2025.12 를
전진 평가하고, MAE 와 단순기준(mom1·선택형) 대비 비율을 적는다. 8차의 1-SE 재튜닝은 교차검증 최소점보다 큰 alpha 만 고르므로
10 보다 작은 alpha 의 효과를 말해 주지 못한다. 이 표는 평가 구간 결과를 보고 만든 사후 진단이며 판정에 쓰지 않는다.

출력: analysis/output/plan_v9_alpha_check.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import settings as S, data as D, targets as T, frames as F, engine as E  # noqa: E402

OUT = os.path.join(BASE, "analysis", "output")
GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]


def main():
    t0 = time.time()
    s = S.load("config/plan_v8_settings.yaml")   # 8차 설정 고정 (아래 재실행 주의)
    hs = s["timing"]["horizons"]
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    rows = []
    for h in hs:
        pn, _ = E.naive_run(s, nf, h, origins)
        pn["err"] = (pn["y"] - pn["yhat"]).abs()
        base = pn.groupby("모형")["err"].mean()
        for info in ("A", "B"):
            for a in GRID:
                p, _ = E.national_run(s, nf, spec, h, origins, info=info, alpha=a, label=f"ridge_{info}_a{a}")
                p = p[(p["P"] >= origins[0]) & (p["P"] <= origins[-1])]
                mae = float((p["y"] - p["yhat"]).abs().mean())
                yr = p.assign(e=(p["y"] - p["yhat"]).abs(), y_=p["P"].astype(str).str[:4]).groupby("y_")["e"].mean()
                rows.append({"h": h, "정보군": info, "alpha": a, "MAE": round(mae, 4), "MAE/mom1": round(mae / base["naive_mom1"], 3),
                             "MAE/선택형": round(mae / base["naive_selected"], 3), "n": len(p),
                             **{f"MAE_{y}": round(v, 4) for y, v in yr.items()}})
                print(f"h={h} {info} alpha={a}: MAE {mae:.4f} (mom1 {base['naive_mom1']:.4f}, 선택형 {base['naive_selected']:.4f}) {time.time() - t0:.0f}s", flush=True)
        rows.append({"h": h, "정보군": "naive", "alpha": np.nan, "MAE": round(float(base["naive_mom1"]), 4), "MAE/mom1": 1.0,
                     "MAE/선택형": round(float(base["naive_mom1"] / base["naive_selected"]), 3), "n": int((pn["모형"] == "naive_mom1").sum())})
    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "plan_v9_alpha_check.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.pivot_table(index=["h", "정보군"], columns="alpha", values="MAE").round(4).to_string())
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
