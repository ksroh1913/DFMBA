# -*- coding: utf-8 -*-
"""[개별 변수 모형 C, 사후 추가] 변수를 RMPI 로 묶지 않고 개별 변환 변수를 그대로 Ridge 에 넣는 정보군 C 를 9차 구성·입력으로 실행해 B 와 비교한다.
전국: C 를 ① 과 같은 세 튜닝(cv_min, 고정 10, 1-SE)으로 돌려 v9_s1_예측값_전국.csv 의 B·A·mom1 과 비교. 패널: C 를 r·down 목표로 돌려(고정 alpha 170 / C) 패널 B 와 비교.
출력 (rmpi/output_v9/): v9_s6_C_예측값_전국.csv, v9_s6_C_예측값_패널.csv, v9_s6_C_비교.csv, v9_s6_C_DM참고.csv, v9_s6_C_alpha선택.csv, v9_s6_C_입력수.csv, v9_s6_C_경보지표.csv, v9_s6_C_manifest.csv.
RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행."""

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
from rmpi.run_v9_stage6 import compare, dm_test, monthly_loss  # noqa: E402

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def monthly_brier(p):
    q = p.dropna(subset=["y", "yhat"]).copy(); q["err"] = (q["y"] - q["yhat"]) ** 2
    q = q.groupby("P")["err"].mean(); q.index = pd.PeriodIndex(q.index, freq="M"); return q.sort_index()


def compare_brier(pa, pb, name, h):
    la, lb = monthly_brier(pa), monthly_brier(pb); idx = la.index.intersection(lb.index); d = lb.loc[idx] - la.loc[idx]
    ti = EV.t_intervals(d, 0.95)
    return {"비교": name, "h": h, "월수": int(len(idx)), "지표": "Brier", "MAE_기준": float(la.loc[idx].mean()), "MAE_비교": float(lb.loc[idx].mean()),
            "MAE비율": float(lb.loc[idx].mean() / la.loc[idx].mean()), "평균개선율(%)": float(100 * (1 - lb.loc[idx].mean() / la.loc[idx].mean())),
            "연평균_하한": ti["annual_lo"], "연평균_상한": ti["annual_hi"], "우세연도": f"{ti['years_better']}/{ti['years_total']}", "8차규칙판정(참고)": EV.verdict(ti)}


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    hs = s["timing"]["horizons"]
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    b = D.build(s, "sido"); spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido"); nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    pf = F.panel_frame(s, b, pt, nat)
    log(f"자료 준비 완료: nf {nf.shape}, pf {pf.shape}")
    base_n = pd.read_csv(os.path.join(out, "v9_s1_예측값_전국.csv")); base_n["P"] = pd.PeriodIndex(base_n["P"], freq="M")
    base_p = pd.read_csv(os.path.join(out, "v9_s1_예측값_패널.csv")); base_p["P"] = pd.PeriodIndex(base_p["P"], freq="M")
    ev = lambda p: p[p.P.isin(origins)]
    preds, alog, cmp_rows, dm_rows, ninp, pp, met_rows = [], [], [], [], [], [], []
    for h in hs:
        runs = {}
        p, c = E.national_run(s, nf, spec, h, origins, info="C", alpha_log=alog, alpha_rule="cv_min", label="ridge_C_cvmin"); runs["cvmin"] = p; preds.append(p)
        if c:
            cc = pd.concat(c); ninp.append({"h": h, "단위": "전국", "시점": str(cc["시점"].iloc[0]) if "시점" in cc.columns else "", "입력수": int(cc["유지"].sum()) if "유지" in cc.columns else len(cc)})
        p10, _ = E.national_run(s, nf, spec, h, origins, info="C", alpha=10.0, label="ridge_C_fixed10"); runs["fixed10"] = p10; preds.append(p10)
        p1se, _ = E.national_run(s, nf, spec, h, origins, info="C", retune=True, alpha_log=alog, label="ridge_C_1se"); runs["1se"] = p1se; preds.append(p1se)
        log(f"전국 h={h} C: cv_min {monthly_loss(ev(p)).mean():.4f}, fixed10 {monthly_loss(p10).mean():.4f}, 1se {monthly_loss(p1se).mean():.4f}")
        nm1 = base_n[(base_n["모형"] == "naive_mom1") & (base_n["h"] == h)]
        for tag in ("cvmin", "fixed10", "1se"):
            B_ = ev(base_n[(base_n["모형"] == f"ridge_B_{tag}") & (base_n["h"] == h)]); A_ = ev(base_n[(base_n["모형"] == f"ridge_A_{tag}") & (base_n["h"] == h)]); C_ = ev(runs[tag])
            cmp_rows += [compare(B_, C_, f"C vs B ({tag})", h), compare(A_, C_, f"C vs A ({tag})", h), compare(ev(nm1), C_, f"C vs mom1 ({tag})", h)]
            st, pv = dm_test(B_, C_, h); dm_rows.append({"비교": f"C vs B ({tag})", "h": h, "DM_HLN": st, "p": pv})
        for target in ("r", "down"):
            q, c, _ = E.panel_run(s, pf, spec, h, origins, target=target, info="C", prior_state=(target == "down"),
                                  alert_rates=s["events"]["alert_comparison"]["target_rates_x_event_rate"], label=f"panel_{target}_C")
            pp.append(q)
            if c and target == "r":
                cc = pd.concat(c); ninp.append({"h": h, "단위": "패널", "시점": str(cc["시점"].iloc[0]) if "시점" in cc.columns else "", "입력수": int(cc["유지"].sum()) if "유지" in cc.columns else len(cc)})
            Bq = base_p[(base_p["모형"] == f"panel_{target}_B") & (base_p["h"] == h)]
            if target == "r":
                cmp_rows.append({**compare(Bq, q, "패널 r: C vs B", h), "지표": "MAE"})
            else:
                cmp_rows.append(compare_brier(Bq, q, "패널 down: C vs B", h))
                for k in s["events"]["alert_comparison"]["target_rates_x_event_rate"]:
                    hist = pf[pf["P"] < q["P"].min()][["region", "P", f"down{h}"]].rename(columns={f"down{h}": "y"}).dropna()
                    met, _ = EV.alert_metrics(q, f"컷오프_{k}x", history=hist)
                    met_rows.append({"모형": "panel_down_C", "h": h, "목표빈도(사건비율배수)": k, **met})
            log(f"패널 h={h} {target} C: 행 {len(q)}")
    pd.concat(preds, ignore_index=True).to_csv(os.path.join(out, "v9_s6_C_예측값_전국.csv"), index=False, encoding="utf-8-sig")
    pd.concat(pp, ignore_index=True).to_csv(os.path.join(out, "v9_s6_C_예측값_패널.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "v9_s6_C_비교.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(dm_rows).to_csv(os.path.join(out, "v9_s6_C_DM참고.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(alog).to_csv(os.path.join(out, "v9_s6_C_alpha선택.csv"), index=False, encoding="utf-8-sig")
    import rmpi.models as M
    ccols = spec.loc[spec.block.isin(["regional", "common"]) & M._in_common(spec), "입력"]; dcols = spec.loc[spec.block == "regional", "입력"]
    pd.DataFrame([{"단위": "전국", "개별변수_공통": len(ccols), "A입력": len(F.A_REG), "합계": len(ccols) + len(F.A_REG)},
                  {"단위": "패널", "개별변수_공통": len(ccols), "개별변수_지역": len(dcols), "A입력": len(F.A_REG), "합계": len(ccols) + len(dcols) + len(F.A_REG)}]).to_csv(
        os.path.join(out, "v9_s6_C_입력수.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(met_rows).to_csv(os.path.join(out, "v9_s6_C_경보지표.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"항목": "settings", "값": f"{s['_path']} {s['_sha']} {s['meta']['settings_version']}"},
                  {"항목": "input", "값": f"{s['inputs']['preprocessed_xlsx']} {S.sha256(S.path(s, 'preprocessed_xlsx'))}"},
                  {"항목": "commit", "값": S.git_commit()}, {"항목": "seconds", "값": f"{time.time() - t0:.0f}"}]).to_csv(os.path.join(out, "v9_s6_C_manifest.csv"), index=False, encoding="utf-8-sig")
    log("저장 완료")


if __name__ == "__main__":
    main()
