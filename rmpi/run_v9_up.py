# -*- coding: utf-8 -*-
"""[급등 경보 보조, 사후 추가] 급락과 대칭인 급등(연율 +3% 이상; h=1·3·6 경계 0.247·0.742·1.489%) 사건의 확률 예측과 경보 실용성.
설계 문서에서는 개발 구간의 양성이 적어 보조 과제로 두었던 것을, 2018~2025 전체 구간에서 급락 ④ 와 같은 틀로 평가한다.
규칙: events.up_min_positive_train_rows(30) 이상의 학습 양성이 쌓인 결정월부터 평가(설정 그대로). 모형 A·B·B+실거래, 통합 패널 로지스틱(alpha 170 고정).
출력 (rmpi/output_v9/): v9_s5_급등_예측값_패널.csv, v9_s5_급등_비교.csv(Brier), v9_s5_급등_경보지표.csv, v9_s5_급등_국면시작_선행.csv, v9_s5_급등_사례표_h6.csv,
v9_s5_급등_양성수.csv, v9_s5_급등_구성명세_h6.csv, v9_s5_급등_manifest.csv.  RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행."""

import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import engine as E  # noqa: E402
from rmpi import evaluate as EV  # noqa: E402
from rmpi import frames as F  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402

t0 = time.time()
STATE = "기존급등상태"
MODELS = (("A", False, "panel_up_A"), ("B", False, "panel_up_B"), ("B", True, "panel_up_B+RT"))


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def monthly_brier(p):
    q = p.dropna(subset=["y", "yhat"]).copy()
    q["err"] = (q["y"] - q["yhat"]) ** 2
    q = q.groupby("P")["err"].mean()
    q.index = pd.PeriodIndex(q.index, freq="M")
    return q.sort_index()


def compare_brier(pa, pb, name, h, level=0.95):
    """d = Brier_b − Brier_a (음수면 b 가 낫다). 8차 t구간은 참고."""
    la, lb = monthly_brier(pa), monthly_brier(pb)
    idx = la.index.intersection(lb.index)
    d = lb.loc[idx] - la.loc[idx]
    ti = EV.t_intervals(d, level)
    yt = EV.yearly_table(d)
    return {"비교": name, "h": h, "월수": int(len(idx)), "Brier_기준": float(la.loc[idx].mean()), "Brier_비교": float(lb.loc[idx].mean()),
            "비율": float(lb.loc[idx].mean() / la.loc[idx].mean()), "평균개선율(%)": float(100 * (1 - lb.loc[idx].mean() / la.loc[idx].mean())),
            "d평균": ti["mean"], "연평균_하한": ti["annual_lo"], "연평균_상한": ti["annual_hi"], "2년_하한": ti["biennial_lo"], "2년_상한": ti["biennial_hi"],
            "우세연도": f"{ti['years_better']}/{ti['years_total']}", "8차규칙판정(참고)": EV.verdict(ti),
            **{f"d_{y}": round(v, 5) for y, v in yt["연평균 d"].items()}}


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    hs = s["timing"]["horizons"]
    first = S.per(s["timing"]["official_first_decision"])
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    min_pos = int(s["events"]["up_min_positive_train_rows"])
    rates = s["events"]["alert_comparison"]["target_rates_x_event_rate"]
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    pf = F.panel_frame(s, b, pt, nat)
    log(f"자료 준비 완료: pf {pf.shape}")
    preds, cmp_rows, met_rows, leads, pos_rows, comps = [], [], [], [], [], []
    for h in hs:
        up = pf.copy()
        up[f"down{h}"] = up[f"up{h}"]                 # 사건 열을 급등으로 바꿔 급락과 같은 코드 경로를 쓴다
        up[f"state_down{h}"] = up[f"state_up{h}"]    # 결정월에 이미 급등 상태
        for Tm in origins:                            # 학습 양성 수(결정월별): 평가 시작 규칙의 근거
            m = (up["P"] >= first) & (up["P"] <= Tm - h) & (up[f"down{h}"] == 1)
            pos_rows.append({"h": h, "시점": str(Tm), "훈련양성수": int(m.sum()), "훈련행수": int(((up["P"] >= first) & (up["P"] <= Tm - h) & up[f"down{h}"].notna()).sum())})
        start = None
        for info, extra, lab in MODELS:
            p, c, _ = E.panel_run(s, up, spec, h, origins, target="down", info=info, label=lab, prior_state=True, extra=extra, alert_rates=rates)
            if not len(p):
                continue
            p["타깃"] = "up"
            p = p.rename(columns={"기존급락상태": STATE})
            p["훈련양성수"] = (p["훈련사건비율"] * p["훈련행수"]).round().astype(int)
            p["평가대상"] = p["훈련양성수"] >= min_pos
            preds.append(p)
            if h == 6 and c:
                comps.append(pd.concat(c).assign(모형=lab))
            if start is None and p["평가대상"].any():
                start = p.loc[p["평가대상"], "P"].min()
            log(f"급등 h={h} {lab}: 행 {len(p)}, 평가대상 {int(p['평가대상'].sum())} (시작 {start})")
        allp = pd.concat([q for q in preds if q["h"].iloc[0] == h], ignore_index=True)
        ev = allp[allp["평가대상"]].copy()
        ev["P"] = pd.PeriodIndex(ev["P"].astype(str), freq="M")
        if ev.empty:
            continue
        pa = ev[ev["모형"] == "panel_up_A"]; pb = ev[ev["모형"] == "panel_up_B"]; pr = ev[ev["모형"] == "panel_up_B+RT"]
        cmp_rows.append(compare_brier(pa, pb, "급등 Brier B vs A", h))
        cmp_rows.append(compare_brier(pb, pr, "급등 Brier B+실거래 vs B", h))
        hist = pf[pf["P"] < ev["P"].min()][["region", "P", f"up{h}"]].rename(columns={f"up{h}": "y"}).dropna()
        for m, g in ev.groupby("모형"):
            for k in rates:
                met, lead = EV.alert_metrics(g, f"컷오프_{k}x", state_col=STATE, history=hist)
                met_rows.append({"모형": m, "h": h, "목표빈도(사건비율배수)": k, "평가시작": str(ev["P"].min()), **met})
                leads.append(lead.assign(모형=m, h=h, 배수=k))
        log(f"급등 h={h} 지표 완료")
    allp = pd.concat(preds, ignore_index=True)
    allp.to_csv(os.path.join(out, "v9_s5_급등_예측값_패널.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "v9_s5_급등_비교.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(met_rows).rename(columns={"급락포착률": "급등포착률"}).to_csv(os.path.join(out, "v9_s5_급등_경보지표.csv"), index=False, encoding="utf-8-sig")
    pd.concat(leads, ignore_index=True).to_csv(os.path.join(out, "v9_s5_급등_국면시작_선행.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pos_rows).to_csv(os.path.join(out, "v9_s5_급등_양성수.csv"), index=False, encoding="utf-8-sig")
    if comps:
        pd.concat(comps, ignore_index=True).to_csv(os.path.join(out, "v9_s5_급등_구성명세_h6.csv"), index=False, encoding="utf-8-sig")
    # 사례표 (h=6, 1배 컷오프, 평가대상·비급등 상태 행)
    h = 6
    cases = allp[(allp["h"] == h) & allp["평가대상"] & (allp[STATE] == 0)].copy()
    cases["경보_1x"] = cases["yhat"] >= cases["컷오프_1x"]
    cases = cases[cases["경보_1x"] | (cases["y"] == 1)]
    cases["결과"] = np.where(cases["경보_1x"] & (cases["y"] == 1), "적중", np.where(cases["경보_1x"], "오경보", "놓침"))
    cases[["모형", "h", "P", "region", "y", "yhat", "컷오프_1x", "결과"]].to_csv(os.path.join(out, "v9_s5_급등_사례표_h6.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"항목": "settings", "값": f"{s['_path']} {s['_sha']} {s['meta']['settings_version']}"},
                  {"항목": "input", "값": f"{s['inputs']['preprocessed_xlsx']} {S.sha256(S.path(s, 'preprocessed_xlsx'))}"},
                  {"항목": "commit", "값": S.git_commit()}, {"항목": "min_positive_train_rows", "값": min_pos},
                  {"항목": "seconds", "값": f"{time.time() - t0:.0f}"}]).to_csv(os.path.join(out, "v9_s5_급등_manifest.csv"), index=False, encoding="utf-8-sig")
    log("저장 완료")


if __name__ == "__main__":
    main()
