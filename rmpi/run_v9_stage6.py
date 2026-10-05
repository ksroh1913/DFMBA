# -*- coding: utf-8 -*-
"""
[9차 6단계] 네 단계 비교 (docs/연구계획_9차_보완설계안.md 2장). RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행.

  ① 기준 재정리   전국 A·B 를 전진 교차검증 최소 alpha(매년 1월)로 다시 평가. 민감도: 고정 10, 1-SE. 단순기준 병기.
                 패널(RQ2 r, RQ3 down)의 주 사양(고정 170 / C 1/170)을 v9 입력·선별 규칙으로 재실행.
  ② 보정 구조     전국 G_전국: M0 · M1 · M1′ · M2 · M2+past36.  통합 패널 G: 같은 모형 + M3.
  ③ 실거래 추가   패널 M3(위에 포함), RQ2 B+실거래, RQ3 B+실거래. 민감도: 12개월합 Δ12, 실거래 원열 1개월 추가 시차.
  ④ 급락 경보     비급락 상태 지역에서 A·B(·B+실거래)를 훈련 사건비율 1배·2배 분위 컷오프로 비교.
사용법: python rmpi/run_v9_stage6.py [--steps 1,2,3,4]   (단계별로 결과를 저장하고 다음 단계로)
출력 (rmpi/output_v9/): v9_s1_*.csv, v9_s2_*.csv, v9_s3_*.csv, v9_s4_*.csv, v9_stage6_manifest.csv
"""

import argparse
import copy
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


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def monthly_loss(p, by_region=False):
    """예측표 -> 월별 MAE (패널은 같은 달 지역 평균 먼저)"""
    q = p.dropna(subset=["y", "yhat"]).copy()
    q["err"] = (q["y"] - q["yhat"]).abs()
    if by_region:
        return q.groupby(["region"])["err"].mean()
    if "region" in q.columns:
        q = q.groupby("P")["err"].mean()
    else:
        q = q.set_index("P")["err"]
    q.index = pd.PeriodIndex(q.index, freq="M")
    return q.sort_index()


def compare(pa, pb, name, h, level=0.95):
    """d = L_b − L_a (음수면 b 가 낫다). 8차 t구간을 계산해 보고하되 판정은 '후속 탐색' 표기."""
    la, lb = monthly_loss(pa), monthly_loss(pb)
    idx = la.index.intersection(lb.index)
    d = (lb.loc[idx] - la.loc[idx])
    ti = EV.t_intervals(d, level)
    yt = EV.yearly_table(d)
    return {"비교": name, "h": h, "월수": int(len(idx)), "MAE_기준": float(la.loc[idx].mean()), "MAE_비교": float(lb.loc[idx].mean()),
            "MAE비율": float(lb.loc[idx].mean() / la.loc[idx].mean()), "평균개선율(%)": float(100 * (1 - lb.loc[idx].mean() / la.loc[idx].mean())),
            "d평균": ti["mean"], "연평균_하한": ti["annual_lo"], "연평균_상한": ti["annual_hi"], "2년_하한": ti["biennial_lo"], "2년_상한": ti["biennial_hi"],
            "우세연도": f"{ti['years_better']}/{ti['years_total']}", "8차규칙판정(참고)": EV.verdict(ti),
            **{f"d_{y}": round(v, 4) for y, v in yt["연평균 d"].items()}}


def dm_test(pa, pb, h):
    """Diebold-Mariano (참고): 월별 손실 차 d 에 Newey-West(시차 h−1) 분산, HLN 소표본 보정, t(T−1) 양측 p."""
    la, lb = monthly_loss(pa), monthly_loss(pb)
    idx = la.index.intersection(lb.index)
    d = (lb.loc[idx] - la.loc[idx]).values
    Tn = len(d)
    if Tn < 10:
        return np.nan, np.nan
    dm = d - d.mean()
    L = max(h - 1, 0)
    var = float(np.sum(dm ** 2)) / Tn
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        var += 2 * w * float(np.sum(dm[k:] * dm[:-k])) / Tn
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var / Tn)
    hln = stat * np.sqrt((Tn + 1 - 2 * (L + 1) + (L + 1) * L / Tn) / Tn)
    from scipy import stats as st
    p = 2 * (1 - st.t.cdf(abs(hln), Tn - 1))
    return float(hln), float(p)


def region_table(p, name, h):
    r = monthly_loss(p, by_region=True)
    return pd.DataFrame({"모형": name, "h": h, "region": r.index, "MAE": r.values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1,2,3,4")
    args = ap.parse_args()
    steps = {int(x) for x in args.steps.split(",")}
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    os.makedirs(out, exist_ok=True)
    hs = s["timing"]["horizons"]
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    post = pd.period_range(S.per(s["timing"]["eval_end"]) + 1, S.per(s["timing"]["posthoc_label_end"]), freq="M")
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    pf = F.panel_frame(s, b, pt, nat)
    log(f"자료 준비 완료: nf {nf.shape}, pf {pf.shape}, 설정 {s['meta']['settings_version']}")
    manifest = [{"항목": "settings", "값": f"{s['_path']} {s['_sha']} {s['meta']['settings_version']}"},
                {"항목": "input", "값": f"{s['inputs']['preprocessed_xlsx']} {S.sha256(S.path(s, 'preprocessed_xlsx'))}"},
                {"항목": "commit", "값": S.git_commit()}]

    # ================================================================ ① 기준 재정리
    if 1 in steps:
        preds, alog, cmp_rows, dm_rows = [], [], [], []
        for h in hs:
            pn, sel = E.naive_run(s, nf, h, origins.union(post))
            preds.append(pn)
            sel.to_csv(os.path.join(out, f"v9_s1_선택형기준_h{h}.csv"), index=False, encoding="utf-8-sig")
            runs = {}
            for info in ("A", "B"):
                p, c = E.national_run(s, nf, spec, h, origins.union(post), info=info, alpha_log=alog, label=f"ridge_{info}_cvmin")
                runs[f"{info}_cvmin"] = p; preds.append(p)
                if c:
                    pd.concat(c).assign(h=h, 정보군=info).to_csv(os.path.join(out, f"v9_s1_구성명세_{info}_h{h}.csv"), index=False, encoding="utf-8-sig")
                p10, _ = E.national_run(s, nf, spec, h, origins, info=info, alpha=10.0, label=f"ridge_{info}_fixed10")
                runs[f"{info}_fixed10"] = p10; preds.append(p10)
                p1se, _ = E.national_run(s, nf, spec, h, origins, info=info, retune=True, alpha_log=alog, label=f"ridge_{info}_1se")
                runs[f"{info}_1se"] = p1se; preds.append(p1se)
                log(f"① h={h} {info}: cv_min MAE {monthly_loss(p[p.P.isin(origins)]).mean():.4f}, fixed10 {monthly_loss(p10).mean():.4f}, 1se {monthly_loss(p1se).mean():.4f}")
            nsel = pn[pn["모형"] == "naive_selected"]; nm1 = pn[pn["모형"] == "naive_mom1"]
            ev = lambda p: p[p.P.isin(origins)]
            for tag in ("cvmin", "fixed10", "1se"):
                A_, B_ = ev(runs[f"A_{tag}"]), ev(runs[f"B_{tag}"])
                cmp_rows += [compare(A_, B_, f"B vs A ({tag})", h), compare(ev(nsel), B_, f"B vs 선택형 단순기준 ({tag})", h),
                             compare(ev(nsel), A_, f"A vs 선택형 단순기준 ({tag})", h), compare(ev(nm1), B_, f"B vs mom1 ({tag})", h)]
                for nm, (x, y) in {"B vs A": (A_, B_), "B vs mom1": (ev(nm1), B_)}.items():
                    st, pv = dm_test(x, y, h); dm_rows.append({"비교": f"{nm} ({tag})", "h": h, "DM_HLN": st, "p": pv})
        pd.concat(preds, ignore_index=True).to_csv(os.path.join(out, "v9_s1_예측값_전국.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(alog).to_csv(os.path.join(out, "v9_s1_alpha선택.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "v9_s1_비교.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(dm_rows).to_csv(os.path.join(out, "v9_s1_DM참고.csv"), index=False, encoding="utf-8-sig")
        # 패널 주 사양 재실행 (RQ2 r: A·B, RQ3 down: A·B) — 고정 alpha/C, v9 입력·선별
        pp = []
        for h in hs:
            for target in ("r", "down"):
                for info in ("A", "B"):
                    p, c, _ = E.panel_run(s, pf, spec, h, origins, target=target, info=info, prior_state=(target == "down"),
                                          alert_rates=s["events"]["alert_comparison"]["target_rates_x_event_rate"], label=f"panel_{target}_{info}")
                    pp.append(p)
                    log(f"① 패널 h={h} {target} {info}: 행 {len(p)}")
            pn = E.panel_naive(s, pf, h, origins, target="r"); pp.append(pn)
        pd.concat(pp, ignore_index=True).to_csv(os.path.join(out, "v9_s1_예측값_패널.csv"), index=False, encoding="utf-8-sig")
        log("① 저장 완료")

    # ================================================================ ② 보정 구조 (전국·통합 패널)
    if 2 in steps:
        preds, alog, cmp_rows, reg_rows, dm_rows = [], [], [], [], []
        models_nat = ["M0", "M1", "M1s", "M2", "M2+past36"]
        models_pan = ["M0", "M1", "M1s", "M2", "M2+past36", "M3"]
        for h in hs:
            res = {}
            for m in models_nat:
                p, c, al = E.correction_run(s, nf, spec, h, origins.union(post), model=m)
                res[m] = p[p.P.isin(origins)]; preds.append(p.assign(단위="전국")); alog.append(al.assign(단위="전국"))
                if c:
                    pd.concat(c).assign(h=h, 단위="전국").to_csv(os.path.join(out, f"v9_s2_구성명세_전국_{m.replace('+', '_')}_h{h}.csv"), index=False, encoding="utf-8-sig")
                log(f"② 전국 h={h} {m}: MAE {monthly_loss(res[m]).mean():.4f}")
            for a_, b_ in [("M0", "M1"), ("M1", "M2"), ("M0", "M2"), ("M1s", "M2"), ("M0", "M1s"), ("M2", "M2+past36")]:
                cmp_rows.append({**compare(res[a_], res[b_], f"전국 {b_} vs {a_}", h), "단위": "전국"})
                st, pv = dm_test(res[a_], res[b_], h); dm_rows.append({"비교": f"전국 {b_} vs {a_}", "h": h, "DM_HLN": st, "p": pv})
            resp = {}
            for m in models_pan:
                p, c, al = E.correction_run(s, pf, spec, h, origins, model=m, panel=True)
                resp[m] = p; preds.append(p.assign(단위="패널")); alog.append(al.assign(단위="패널"))
                if c:
                    pd.concat(c).assign(h=h, 단위="패널").to_csv(os.path.join(out, f"v9_s2_구성명세_패널_{m.replace('+', '_')}_h{h}.csv"), index=False, encoding="utf-8-sig")
                reg_rows.append(region_table(p, m, h))
                log(f"② 패널 h={h} {m}: MAE {monthly_loss(p).mean():.4f}")
            for a_, b_ in [("M0", "M1"), ("M1", "M2"), ("M0", "M2"), ("M1s", "M2"), ("M2", "M3"), ("M0", "M3"), ("M2", "M2+past36")]:
                cmp_rows.append({**compare(resp[a_], resp[b_], f"패널 {b_} vs {a_}", h), "단위": "패널"})
                st, pv = dm_test(resp[a_], resp[b_], h); dm_rows.append({"비교": f"패널 {b_} vs {a_}", "h": h, "DM_HLN": st, "p": pv})
        pd.concat(preds, ignore_index=True).to_csv(os.path.join(out, "v9_s2_예측값.csv"), index=False, encoding="utf-8-sig")
        pd.concat(alog, ignore_index=True).to_csv(os.path.join(out, "v9_s2_alpha선택.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "v9_s2_비교.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(dm_rows).to_csv(os.path.join(out, "v9_s2_DM참고.csv"), index=False, encoding="utf-8-sig")
        pd.concat(reg_rows, ignore_index=True).to_csv(os.path.join(out, "v9_s2_지역별MAE.csv"), index=False, encoding="utf-8-sig")
        log("② 저장 완료")

    # ================================================================ ③ 실거래 추가 (RQ2·RQ3 B+실거래, 민감도)
    if 3 in steps:
        pp, cmp_rows, sens_rows = [], [], []
        base = pd.read_csv(os.path.join(out, "v9_s1_예측값_패널.csv"))
        base["P"] = pd.PeriodIndex(base["P"], freq="M")
        for h in hs:
            for target in ("r", "down"):
                p, c, _ = E.panel_run(s, pf, spec, h, origins, target=target, info="B", extra=True, prior_state=(target == "down"),
                                      alert_rates=s["events"]["alert_comparison"]["target_rates_x_event_rate"], label=f"panel_{target}_B+RT")
                pp.append(p)
                if c:
                    pd.concat(c).assign(h=h).to_csv(os.path.join(out, f"v9_s3_구성명세_{target}_h{h}.csv"), index=False, encoding="utf-8-sig")
                pb = base[(base["모형"] == f"panel_{target}_B") & (base["h"] == h)]
                if target == "r":
                    cmp_rows.append({**compare(pb, p, "RQ2 B+실거래 vs B", h), "지표": "MAE"})
                else:
                    lb = lambda q: q.assign(y=q["y"], yhat=q["yhat"])  # Brier = (p−y)^2 의 평균 → 월별 손실로 쓰려면 err 를 제곱으로
                    def brier_monthly(q):
                        z = q.dropna(subset=["y", "yhat"]).copy(); z["err"] = (z["y"] - z["yhat"]) ** 2
                        m = z.groupby("P")["err"].mean(); m.index = pd.PeriodIndex(m.index, freq="M"); return m
                    la, lb_ = brier_monthly(pb), brier_monthly(p)
                    idx = la.index.intersection(lb_.index); d = lb_.loc[idx] - la.loc[idx]; ti = EV.t_intervals(d)
                    cmp_rows.append({"비교": "RQ3 B+실거래 vs B", "h": h, "지표": "Brier", "월수": int(len(idx)), "Brier_기준": float(la.loc[idx].mean()),
                                     "Brier_비교": float(lb_.loc[idx].mean()), "비율": float(lb_.loc[idx].mean() / la.loc[idx].mean()),
                                     "평균개선율(%)": float(100 * (1 - lb_.loc[idx].mean() / la.loc[idx].mean())), "d평균": ti["mean"],
                                     "연평균_하한": ti["annual_lo"], "연평균_상한": ti["annual_hi"], "2년_하한": ti["biennial_lo"], "2년_상한": ti["biennial_hi"],
                                     "우세연도": f"{ti['years_better']}/{ti['years_total']}", "8차규칙판정(참고)": EV.verdict(ti)})
                log(f"③ h={h} {target} B+실거래: 행 {len(p)}")
        pd.concat(pp, ignore_index=True).to_csv(os.path.join(out, "v9_s3_예측값_패널.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "v9_s3_비교.csv"), index=False, encoding="utf-8-sig")
        # 민감도 (h=6 만): 12개월합 Δ12, 실거래 원열 1개월 추가 시차 — 패널 M3 와 RQ2 B+실거래
        h = 6
        s2 = copy.deepcopy(s); s2["extra_inputs"]["RT_mix"]["window"] = "sum12"; s2["extra_inputs"]["RT_act"]["window"] = "sum12"
        b2 = D.build(s2, "sido"); pf2 = F.panel_frame(s2, b2, T.regional(s2, b2["ml"], "sido"), nat)
        s3 = copy.deepcopy(s)
        b3 = D.build(s3, "sido", extra_lag="rt"); pf3 = F.panel_frame(s3, b3, T.regional(s3, b3["ml"], "sido"), nat)
        m3_main = pd.read_csv(os.path.join(out, "v9_s2_예측값.csv")) if os.path.exists(os.path.join(out, "v9_s2_예측값.csv")) else None
        for tag, s_, pf_, sp_ in (("12개월합Δ12", s2, pf2, b2["spec"]), ("원열+1개월시차", s3, pf3, b3["spec"])):
            p, _, _ = E.correction_run(s_, pf_, sp_, h, origins, model="M3", panel=True)
            pm2 = E.correction_run(s_, pf_, sp_, h, origins, model="M2", panel=True)[0]
            sens_rows.append({**compare(pm2, p, f"패널 M3 vs M2 ({tag})", h), "민감도": tag})
            q, _, _ = E.panel_run(s_, pf_, sp_, h, origins, target="r", info="B", extra=True, label=f"panel_r_B+RT_{tag}")
            pb = base[(base["모형"] == "panel_r_B") & (base["h"] == h)]
            sens_rows.append({**compare(pb, q, f"RQ2 B+실거래 vs B ({tag})", h), "민감도": tag})
            log(f"③ 민감도 {tag} 완료")
        pd.DataFrame(sens_rows).to_csv(os.path.join(out, "v9_s3_민감도.csv"), index=False, encoding="utf-8-sig")
        log("③ 저장 완료")

    # ================================================================ ④ 급락 경보 실용성
    if 4 in steps:
        base = pd.read_csv(os.path.join(out, "v9_s1_예측값_패널.csv")); base["P"] = pd.PeriodIndex(base["P"], freq="M")
        ext = pd.read_csv(os.path.join(out, "v9_s3_예측값_패널.csv")); ext["P"] = pd.PeriodIndex(ext["P"], freq="M")
        allp = pd.concat([base[base["타깃"] == "down"], ext[ext["타깃"] == "down"]], ignore_index=True)
        rows, leads = [], []
        for (m, h), g in allp.groupby(["모형", "h"]):
            for k in s["events"]["alert_comparison"]["target_rates_x_event_rate"]:
                met, lead = EV.alert_metrics(g, f"컷오프_{k}x")
                rows.append({"모형": m, "h": h, "목표빈도(사건비율배수)": k, **met})
                leads.append(lead.assign(모형=m, h=h, 배수=k))
        pd.DataFrame(rows).to_csv(os.path.join(out, "v9_s4_경보지표.csv"), index=False, encoding="utf-8-sig")
        pd.concat(leads, ignore_index=True).to_csv(os.path.join(out, "v9_s4_국면시작_선행.csv"), index=False, encoding="utf-8-sig")
        # 사례표: 비급락 상태에서 1배 컷오프 경보가 켜진 행과 결과 (h=6)
        h = 6
        cases = allp[(allp["h"] == h) & (allp["기존급락상태"] == 0)].copy()
        cases["경보_1x"] = cases["yhat"] >= cases["컷오프_1x"]
        cases = cases[cases["경보_1x"] | (cases["y"] == 1)]
        cases["결과"] = np.where(cases["경보_1x"] & (cases["y"] == 1), "적중", np.where(cases["경보_1x"], "오경보", "놓침"))
        cases[["모형", "h", "P", "region", "y", "yhat", "컷오프_1x", "결과"]].to_csv(os.path.join(out, "v9_s4_사례표_h6.csv"), index=False, encoding="utf-8-sig")
        log("④ 저장 완료")

    pd.DataFrame(manifest + [{"항목": "steps", "값": str(sorted(steps))}, {"항목": "seconds", "값": f"{time.time() - t0:.0f}"}]).to_csv(
        os.path.join(out, "v9_stage6_manifest.csv"), index=False, encoding="utf-8-sig")
    log("완료")


if __name__ == "__main__":
    main()
