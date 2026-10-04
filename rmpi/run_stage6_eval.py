# -*- coding: utf-8 -*-
"""
[6단계 예측 평가 · 분석 2~4] 2018.01~2025.12 월별 전진 확장창 평가 (연구계획 8차 확정본 5~9장).

사전 지정 주 분석
  RQ1  전국 Ridge: B_공통 대 A, B_공통 대 선택형 단순기준 (MAE 차이). 같은 h 에서 둘 다 '개선'일 때만 그 h 를 개선으로 센다
  RQ2  17개 시도 상대 변화 Ridge: B_지역 대 A_지역 (월내 상대 MAE 차이)
  RQ3  17개 시도 급락 L2 로지스틱: B(A+공통·지역 RMPI) 대 A (Brier 차이)
보조·탐색: C 정보군, Extra Trees, 분기 재학습, 재튜닝, 2019 시작, 개정 위험 변수 제외, g 타깃, 시차 +1, 위약 시험,
  대용계열 확장 실험, 결합 예측(Ḡ̂ + r̃)과 통합 패널, 급락 층별·국면 시작 사례, 급등, 경계 민감도.
판정: 1장의 두 t구간 규칙(evaluate.verdict). 블록 부트스트랩은 참고. 예측값은 채점 전에 저장한다.

실행: python rmpi/run_stage6_eval.py [--quick]   (--quick: ET·위약·민감도 생략)
출력: rmpi/output/stage6_*.csv, fig6_*.png
"""

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
from rmpi import viz as V  # noqa: E402

plt = V.plt
QUICK = "--quick" in sys.argv
FROM_SAVED = "--from-saved" in sys.argv   # 저장된 예측값으로 채점·그림만 다시
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def monthly_loss(df, kind="abs"):
    """예측표 -> 월별 손실 Series(PeriodIndex). 패널은 같은 달 지역 평균."""
    d = df.dropna(subset=["y", "yhat"]).copy()
    d["loss"] = (d["yhat"] - d["y"]) ** 2 if kind == "sq" else (d["yhat"] - d["y"]).abs()
    out = d.groupby("P")["loss"].mean()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.sort_index()


def judge_pair(name, h, L_new, L_ref, s, boot=True):
    d = (L_new - L_ref).dropna()
    ti = EV.t_intervals(d, s["inference"]["level"])
    rec = {"비교": name, "h": h, **ti, "판정": EV.verdict(ti), "MAE비율(평균의 비)": float(L_new.mean() / L_ref.mean()),
           "손실_추가정보": float(L_new.mean()), "손실_기준": float(L_ref.mean())}
    if boot:
        for L in s["inference"]["bootstrap"]["block_lengths"]:
            lo, hi, pneg = EV.circular_block_bootstrap(d, L, s["inference"]["bootstrap"]["n_boot"], s["meta"]["random_seed"])
            rec[f"부트_블록{L}_하한"], rec[f"부트_블록{L}_상한"], rec[f"부트_블록{L}_음수비율"] = lo, hi, pneg
    return rec


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    os.makedirs(out, exist_ok=True)
    V.setup()
    hs = s["timing"]["horizons"]
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    origins_post = pd.period_range(S.per(s["timing"]["eval_end"]) + 1, S.per(s["timing"]["posthoc_label_end"]), freq="M")
    qm = s["timing"]["quarterly_refit_months"]
    rev = s["revision_risk"]["columns"]
    seed = s["meta"]["random_seed"]

    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    pf = F.panel_frame(s, b, pt, nat)
    log("자료 준비 완료")

    # ======================================================== 분석 2: 전국 모형 (RQ1)
    nat_preds, comps, sel_all = [], [], []
    for h in ([] if FROM_SAVED else hs):
        for info in ("A", "B", "C"):
            p, c = E.national_run(s, nf, spec, h, origins.union(origins_post), info=info)
            nat_preds.append(p)
            comps += c
        pn, sel = E.naive_run(s, nf, h, origins.union(origins_post))
        nat_preds.append(pn)
        sel_all.append(sel)
        # Ḡ 타깃 (결합 예측용)
        for info in ("A", "B"):
            p, _ = E.national_run(s, nf, spec, h, origins, info=info, target="Gbar", label=f"ridge_{info}_Gbar")
            nat_preds.append(p)
        # 분기 재학습
        for info in ("A", "B"):
            p, _ = E.national_run(s, nf, spec, h, origins, info=info, refit_months=qm, label=f"ridge_{info}_분기재학습")
            nat_preds.append(p)
        log(f"전국 h={h} 주 모형·단순기준·Ḡ·분기 완료")
        if not QUICK:
            p, _ = E.national_run(s, nf, spec, h, origins, info="B", model="et", label="et_B")
            nat_preds.append(p)
            p, _ = E.national_run(s, nf, spec, h, origins, info="A", model="et", label="et_A")
            nat_preds.append(p)
            for info in ("A", "B"):
                p, _ = E.national_run(s, nf, spec, h, origins, info=info, retune=True, label=f"ridge_{info}_재튜닝")
                nat_preds.append(p)
            p, _ = E.national_run(s, nf, spec, h, origins, info="B", exclude_ids=rev, label="ridge_B_개정위험제외")
            nat_preds.append(p)
            for info in ("A", "B"):
                p, _ = E.national_run(s, nf, spec, h, origins, info=info, target="g", label=f"ridge_{info}_g타깃")
                nat_preds.append(p)
            # 확장 실험
            for which in ("main", "sensitivity"):
                ef = T.extension_frame(s, h, which)
                for w in s["timing"]["extension_weights"]:
                    for info in ("A", "B"):
                        p, _ = E.national_run(s, nf, spec, h, origins, info=info, extension=ef, ext_weight=w,
                                              label=f"ridge_{info}_확장_{which}_w{w}")
                        nat_preds.append(p)
            # 위약 시험
            for rep in range(s["models"]["placebo"]["n_rep"]):
                p, _ = E.national_run(s, nf, spec, h, origins, info="B", placebo_rng=np.random.default_rng(seed + rep),
                                      label=f"ridge_B_위약_{rep:02d}")
                nat_preds.append(p)
            log(f"전국 h={h} 보조·민감도 완료")
    if FROM_SAVED:
        NAT = pd.read_csv(os.path.join(out, "stage6_예측값_전국.csv"))
        NAT["P"] = pd.PeriodIndex(NAT["P"], freq="M")
        lagp = os.path.join(out, "stage6_예측값_전국_시차+1.csv")
        LAG = pd.read_csv(lagp) if os.path.exists(lagp) else pd.DataFrame()
        if len(LAG):
            LAG["P"] = pd.PeriodIndex(LAG["P"], freq="M")
    else:
        NAT = pd.concat(nat_preds, ignore_index=True)
        NAT.to_csv(os.path.join(out, "stage6_예측값_전국.csv"), index=False, encoding="utf-8-sig")   # 채점 전 저장
        pd.concat(sel_all).to_csv(os.path.join(out, "stage6_선택형기준.csv"), index=False, encoding="utf-8-sig")
        if comps:
            pd.concat(comps).to_csv(os.path.join(out, "stage6_구성명세_전국.csv"), index=False, encoding="utf-8-sig")

    # 시차 +1 강건성 (자료를 다시 만든다)
    if not QUICK and not FROM_SAVED:
        b1 = D.build(s, "sido", extra_lag=True)
        nf1 = F.national_frame(s, b1, nat)
        lag_preds = []
        for h in hs:
            for info in ("A", "B"):
                p, _ = E.national_run(s, nf1, b1["spec"], h, origins, info=info, label=f"ridge_{info}_시차+1")
                lag_preds.append(p)
        LAG = pd.concat(lag_preds)
        LAG.to_csv(os.path.join(out, "stage6_예측값_전국_시차+1.csv"), index=False, encoding="utf-8-sig")
        log("시차 +1 완료")

    # ======================================================== 분석 3·4: 시도 패널 (RQ2·RQ3·결합)
    pan_preds, comps_p, diags = [], [], []
    for h in ([] if FROM_SAVED else hs):
        for info in ("A", "B", "C"):
            p, c, _ = E.panel_run(s, pf, spec, h, origins, target="r", info=info)
            pan_preds.append(p)
            comps_p += c
        pan_preds.append(E.panel_naive(s, pf, h, origins, target="r"))
        for info in ("A", "B", "Bc"):
            p, c, _ = E.panel_run(s, pf, spec, h, origins, target="down", info=info, prior_state=True)
            pan_preds.append(p)
        for info in ("A", "B"):
            p, _, _ = E.panel_run(s, pf, spec, h, origins, target="G", info=info)
            pan_preds.append(p)
        pan_preds.append(E.panel_naive(s, pf, h, origins, target="G"))
        for info in ("A", "B"):
            p, _, _ = E.panel_run(s, pf, spec, h, origins, target="r", info=info, refit_months=qm, label=f"ridge_{info}_분기재학습")
            pan_preds.append(p)
        log(f"패널 h={h} 주 모형 완료")
        if not QUICK:
            for tgt in ("r", "down"):
                p, _, dg = E.panel_run(s, pf, spec, h, origins, target=tgt, info="B", model="et", label="et_B")
                pan_preds.append(p)
                diags.append(dg)
            p, _, _ = E.panel_run(s, pf, spec, h, origins, target="r", info="B", exclude_ids=rev, label="ridge_B_개정위험제외")
            pan_preds.append(p)
            p, _, _ = E.panel_run(s, pf, spec, h, origins, target="down", info="B", exclude_ids=rev, label="ridge_B_개정위험제외")
            pan_preds.append(p)
            # 급등(보조): 학습 양성 30행 이상인 시점부터
            up = pf.copy()
            for hh in hs:
                up[f"down{hh}"] = up[f"up{hh}"]
            for info in ("A", "B"):
                p, _, _ = E.panel_run(s, up, spec, h, origins, target="down", info=info, label=f"logit_{info}_급등")
                if len(p):
                    p = p[p["훈련사건비율"] * p["훈련행수"] >= s["events"]["up_min_positive_train_rows"]]
                    p["타깃"] = "up"
                    pan_preds.append(p)
            # 경계 민감도 (0.75·1.25배)
            for mult in s["events"]["sensitivity_multipliers"]:
                thm = T.thresholds(s, mult)
                pm = pf.copy()
                for hh in hs:
                    pm[f"down{hh}"] = np.where(pm[f"G{hh}"].notna(), (pm[f"G{hh}"] <= thm[hh][0]).astype(float), np.nan)
                for info in ("A", "B"):
                    p, _, _ = E.panel_run(s, pm, spec, h, origins, target="down", info=info, label=f"logit_{info}_경계x{mult}")
                    if len(p):
                        p["타깃"] = f"down_x{mult}"
                        pan_preds.append(p)
            log(f"패널 h={h} 보조·민감도 완료")
    if FROM_SAVED:
        PAN = pd.read_csv(os.path.join(out, "stage6_예측값_패널.csv"))
        PAN["P"] = pd.PeriodIndex(PAN["P"], freq="M")
    else:
        PAN = pd.concat(pan_preds, ignore_index=True)
        PAN.to_csv(os.path.join(out, "stage6_예측값_패널.csv"), index=False, encoding="utf-8-sig")
    if comps_p:
        pd.concat(comps_p).to_csv(os.path.join(out, "stage6_구성명세_패널.csv"), index=False, encoding="utf-8-sig")
    if diags:
        pd.concat(diags).to_csv(os.path.join(out, "stage6_ET잎진단.csv"), index=False, encoding="utf-8-sig")
    log("예측값 저장 완료. 채점 시작")

    # ======================================================== 채점
    def nat_loss(model, h, kind="abs", po=origins):
        d = NAT[(NAT["모형"] == model) & (NAT.h == h) & NAT.P.isin(po)]
        return monthly_loss(d, kind)

    def pan_loss(model, h, target, kind="abs", po=origins, extra=None):
        d = PAN[(PAN["모형"] == model) & (PAN.h == h) & (PAN["타깃"] == target) & PAN.P.isin(po)]
        if extra is not None:
            d = extra(d)
        return monthly_loss(d, kind)

    verdict_rows, rq_rows = [], []
    # RQ1
    for h in hs:
        r1 = judge_pair("RQ1 B_공통 대 A", h, nat_loss("ridge_B", h), nat_loss("ridge_A", h), s)
        r2 = judge_pair("RQ1 B_공통 대 선택형 단순기준", h, nat_loss("ridge_B", h), nat_loss("naive_selected", h), s)
        verdict_rows += [r1, r2]
        if r1["판정"] == "개선" and r2["판정"] == "개선":
            v1 = "개선"
        elif "악화" in (r1["판정"], r2["판정"]):
            v1 = "악화"
        else:
            v1 = f"{r1['판정']}(대 A) / {r2['판정']}(대 단순기준)"
        rq_rows.append({"RQ": "RQ1", "h": h, "판정": v1})
    # RQ2
    for h in hs:
        r = judge_pair("RQ2 B_지역 대 A_지역", h, pan_loss("ridge_B", h, "r"), pan_loss("ridge_A", h, "r"), s)
        verdict_rows.append(r)
        rq_rows.append({"RQ": "RQ2", "h": h, "판정": r["판정"]})
    # RQ3
    for h in hs:
        r = judge_pair("RQ3 B(공통+지역 RMPI) 대 A", h, pan_loss("ridge_B", h, "down", "sq"), pan_loss("ridge_A", h, "down", "sq"), s)
        verdict_rows.append(r)
        rq_rows.append({"RQ": "RQ3", "h": h, "판정": r["판정"]})
    # 보조 비교 (판정 없음에 준하는 참고: 같은 규칙으로 계산만)
    aux_pairs = []
    for h in hs:
        aux_pairs += [
            ("보조 RQ1 A 대 선택형 단순기준", h, nat_loss("ridge_A", h), nat_loss("naive_selected", h)),
            ("보조 RQ1 C_공통 대 A", h, nat_loss("ridge_C", h), nat_loss("ridge_A", h)),
            ("보조 RQ1 C_공통 대 B_공통", h, nat_loss("ridge_C", h), nat_loss("ridge_B", h)),
            ("보조 RQ1 B 분기재학습 대 A 분기재학습", h, nat_loss("ridge_B_분기재학습", h), nat_loss("ridge_A_분기재학습", h)),
            ("보조 RQ2 A_지역 대 차이 0", h, pan_loss("ridge_A", h, "r"), pan_loss("naive_zero", h, "r")),
            ("보조 RQ2 B_지역 대 차이 0", h, pan_loss("ridge_B", h, "r"), pan_loss("naive_zero", h, "r")),
            ("보조 RQ2 C_지역 대 A_지역", h, pan_loss("ridge_C", h, "r"), pan_loss("ridge_A", h, "r")),
            ("보조 RQ3 B 대 A+공통 RMPI(Bc)", h, pan_loss("ridge_B", h, "down", "sq"), pan_loss("ridge_Bc", h, "down", "sq")),
            ("보조 RQ3 A+공통 RMPI(Bc) 대 A", h, pan_loss("ridge_Bc", h, "down", "sq"), pan_loss("ridge_A", h, "down", "sq")),
        ]
        if not QUICK and len(LAG):
            aux_pairs += [
                ("보조 RQ1 ET_B 대 Ridge B", h, nat_loss("et_B", h), nat_loss("ridge_B", h)),
                ("보조 RQ1 ET_B 대 ET_A", h, nat_loss("et_B", h), nat_loss("et_A", h)),
                ("민감도 RQ1 B 재튜닝 대 A 재튜닝", h, nat_loss("ridge_B_재튜닝", h), nat_loss("ridge_A_재튜닝", h)),
                ("민감도 RQ1 B 개정위험제외 대 A", h, nat_loss("ridge_B_개정위험제외", h), nat_loss("ridge_A", h)),
                ("민감도 RQ1 B g타깃 대 A g타깃", h, nat_loss("ridge_B_g타깃", h), nat_loss("ridge_A_g타깃", h)),
                ("민감도 RQ1 B 시차+1 대 A 시차+1", h, monthly_loss(LAG[(LAG["모형"] == "ridge_B_시차+1") & (LAG.h == h)]),
                 monthly_loss(LAG[(LAG["모형"] == "ridge_A_시차+1") & (LAG.h == h)])),
                ("보조 RQ2 ET_B 대 Ridge B_지역", h, pan_loss("et_B", h, "r"), pan_loss("ridge_B", h, "r")),
                ("민감도 RQ2 B 개정위험제외 대 A_지역", h, pan_loss("ridge_B_개정위험제외", h, "r"), pan_loss("ridge_A", h, "r")),
                ("보조 RQ3 ET_B 대 로지스틱 B", h, pan_loss("et_B", h, "down", "sq"), pan_loss("ridge_B", h, "down", "sq")),
                ("민감도 RQ3 B 개정위험제외 대 A", h, pan_loss("ridge_B_개정위험제외", h, "down", "sq"), pan_loss("ridge_A", h, "down", "sq")),
            ]
            for mult in s["events"]["sensitivity_multipliers"]:
                aux_pairs.append((f"민감도 RQ3 경계x{mult} B 대 A", h, pan_loss(f"logit_B_경계x{mult}", h, f"down_x{mult}", "sq"),
                                  pan_loss(f"logit_A_경계x{mult}", h, f"down_x{mult}", "sq")))
        # 2019 시작
        o19 = origins[origins >= S.per(s["timing"]["sensitivity_eval_start"])]
        aux_pairs += [("민감도 RQ1 B 대 A (2019.01~)", h, nat_loss("ridge_B", h, po=o19), nat_loss("ridge_A", h, po=o19)),
                      ("민감도 RQ2 B 대 A (2019.01~)", h, pan_loss("ridge_B", h, "r", po=o19), pan_loss("ridge_A", h, "r", po=o19)),
                      ("민감도 RQ3 B 대 A (2019.01~)", h, pan_loss("ridge_B", h, "down", "sq", po=o19), pan_loss("ridge_A", h, "down", "sq", po=o19))]
    for name, h, a, c in aux_pairs:
        if len(a) and len(c):
            verdict_rows.append(judge_pair(name, h, a, c, s, boot=False))
    VER = pd.DataFrame(verdict_rows)
    VER.to_csv(os.path.join(out, "stage6_판정.csv"), index=False, encoding="utf-8-sig")
    RQ = pd.DataFrame(rq_rows)
    summ = []
    for rq in ("RQ1", "RQ2", "RQ3"):
        v = {r["h"]: r["판정"] for r in rq_rows if r["RQ"] == rq}
        summ.append({"RQ": rq, **{f"h={h}": v[h] for h in hs}, "요약": EV.summarize_question(v)})
    pd.DataFrame(summ).to_csv(os.path.join(out, "stage6_요약.csv"), index=False, encoding="utf-8-sig")

    # ---- 연도별 MAE / Brier 표
    yrows = []
    for h in hs:
        for m in sorted(NAT["모형"].unique()):
            if m.startswith("ridge_B_위약"):
                continue
            L = nat_loss(m, h)
            if len(L):
                yr = L.groupby(L.index.year).mean()
                yrows.append({"패널": "전국", "타깃": "G", "모형": m, "h": h, "전체": float(L.mean()), **{str(y): float(v) for y, v in yr.items()},
                              "2018~2020": float(L[L.index.year <= 2020].mean()), "2021~2025": float(L[L.index.year >= 2021].mean())})
        for tgt, kind in (("r", "abs"), ("down", "sq"), ("G", "abs")):
            for m in sorted(PAN.loc[PAN["타깃"] == tgt, "모형"].unique()):
                L = pan_loss(m, h, tgt, kind)
                if len(L):
                    yr = L.groupby(L.index.year).mean()
                    yrows.append({"패널": "17시도", "타깃": tgt, "모형": m, "h": h, "전체": float(L.mean()), **{str(y): float(v) for y, v in yr.items()},
                                  "2018~2020": float(L[L.index.year <= 2020].mean()), "2021~2025": float(L[L.index.year >= 2021].mean())})
    pd.DataFrame(yrows).to_csv(os.path.join(out, "stage6_연도별손실.csv"), index=False, encoding="utf-8-sig")

    # ---- 전국: 2026 사후 확인, A 대비 표본외 R², 위약 분포, 확장 실험
    extra = []
    for h in hs:
        for m in ("ridge_A", "ridge_B", "ridge_C", "naive_selected", "naive_zero"):
            Lp = nat_loss(m, h, po=origins_post)
            if len(Lp):
                extra.append({"항목": "2026 사후 확인 MAE", "모형": m, "h": h, "값": float(Lp.mean()), "월수": int(len(Lp))})
        dA = NAT[(NAT["모형"] == "ridge_A") & (NAT.h == h) & NAT.P.isin(origins)].set_index("P")
        dB = NAT[(NAT["모형"] == "ridge_B") & (NAT.h == h) & NAT.P.isin(origins)].set_index("P")
        both = dA[["y", "yhat"]].join(dB[["yhat"]], rsuffix="_B").dropna()
        r2 = 1 - ((both["yhat_B"] - both["y"]) ** 2).sum() / ((both["yhat"] - both["y"]) ** 2).sum()
        extra.append({"항목": "A 대비 표본외 R² (1 - SSE_B/SSE_A)", "모형": "ridge_B", "h": h, "값": float(r2), "월수": int(len(both))})
        have = set(NAT["모형"].unique())
        if any(m.startswith("ridge_B_위약") for m in have):
            pl = [nat_loss(f"ridge_B_위약_{rep:02d}", h).mean() for rep in range(s["models"]["placebo"]["n_rep"]) if f"ridge_B_위약_{rep:02d}" in have]
            mB, mA = nat_loss("ridge_B", h).mean(), nat_loss("ridge_A", h).mean()
            extra.append({"항목": "위약 시험: 위약 B MAE 중앙값", "모형": "ridge_B_위약", "h": h, "값": float(np.median(pl)), "월수": len(pl)})
            extra.append({"항목": "위약 시험: 실제 B MAE 가 위약 분포에서 차지하는 분위", "모형": "ridge_B", "h": h,
                          "값": float(np.mean(np.array(pl) <= mB)), "월수": len(pl)})
            extra.append({"항목": "위약 시험: 위약 B 가 A 보다 나은 비율", "모형": "ridge_B_위약", "h": h, "값": float(np.mean(np.array(pl) < mA)), "월수": len(pl)})
        for which in ("main", "sensitivity"):
            for w in s["timing"]["extension_weights"]:
                for info in ("A", "B"):
                    m = f"ridge_{info}_확장_{which}_w{w}"
                    if m in have:
                        L = nat_loss(m, h)
                        base = nat_loss(f"ridge_{info}", h)
                        d = (L - base).dropna()
                        ti = EV.t_intervals(d)
                        extra.append({"항목": f"확장 실험 MAE 차이(확장 − 주 결과) {info} {which} w={w}", "모형": m, "h": h, "값": float(ti["mean"]),
                                      "월수": int(len(d)), "연평균 t구간": f"[{ti['annual_lo']:.3f}, {ti['annual_hi']:.3f}]",
                                      "옛 체계 훈련월(첫 시점)": int(NAT[(NAT['모형'] == m) & (NAT.h == h)]['훈련월_옛체계'].iloc[0])})
    pd.DataFrame(extra).to_csv(os.path.join(out, "stage6_전국_보조.csv"), index=False, encoding="utf-8-sig")

    # ---- 결합 예측 (분석 4): Ĝ = Ḡ̂ + r̃
    comb_rows = []
    for h in hs:
        rB = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == h) & (PAN["타깃"] == "r")]
        rA = PAN[(PAN["모형"] == "ridge_A") & (PAN.h == h) & (PAN["타깃"] == "r")]
        gB = NAT[(NAT["모형"] == "ridge_B_Gbar") & (NAT.h == h)].set_index("P")["yhat"]
        gA = NAT[(NAT["모형"] == "ridge_A_Gbar") & (NAT.h == h)].set_index("P")["yhat"]
        G = pf[["region", "P", f"G{h}", f"Gbar{h}"]].rename(columns={f"G{h}": "G", f"Gbar{h}": "Gbar"})
        combos = {"결합 B(Ḡ̂_B + r̃_B)": (rB, gB), "결합 A(Ḡ̂_A + r̃_A)": (rA, gA), "결합 혼합(Ḡ̂_B + r̃_A)": (rA, gB)}
        for name, (rr, gg) in combos.items():
            d = rr[["region", "P", "yhat"]].merge(G, on=["region", "P"])
            d["Ghat"] = d["P"].map(gg) + d["yhat"]
            d = d.dropna(subset=["Ghat", "G"])
            L = d.assign(loss=(d.Ghat - d.G).abs()).groupby("P")["loss"].mean()
            Lc = (d.groupby("P")["Ghat"].mean() - d.groupby("P")["Gbar"].first()).abs()
            Lw = d.assign(w=((d.Ghat - d.groupby("P")["Ghat"].transform("mean")) - (d.G - d.Gbar)).abs()).groupby("P")["w"].mean()
            comb_rows.append({"모형": name, "h": h, "시도별 MAE": float(L.mean()), "공통 MAE": float(Lc.mean()), "월내 상대 MAE": float(Lw.mean()),
                              "월수": int(len(L))})
        for m in ("ridge_A", "ridge_B", "naive_zero", "naive_mom1", "naive_mom3", "naive_lasth"):
            d = PAN[(PAN["모형"] == m) & (PAN.h == h) & (PAN["타깃"] == "G")].merge(G[["region", "P", "Gbar"]], on=["region", "P"])
            if d.empty:
                continue
            L = d.assign(loss=(d.yhat - d.y).abs()).groupby("P")["loss"].mean()
            Lc = (d.groupby("P")["yhat"].mean() - d.groupby("P")["Gbar"].first()).abs()
            Lw = d.assign(w=((d.yhat - d.groupby("P")["yhat"].transform("mean")) - (d.y - d.Gbar)).abs()).groupby("P")["w"].mean()
            comb_rows.append({"모형": ("통합 패널 " + m) if m.startswith("ridge") else ("단순 " + m), "h": h, "시도별 MAE": float(L.mean()),
                              "공통 MAE": float(Lc.mean()), "월내 상대 MAE": float(Lw.mean()), "월수": int(len(L))})
    COMB = pd.DataFrame(comb_rows)
    COMB.to_csv(os.path.join(out, "stage6_결합예측.csv"), index=False, encoding="utf-8-sig")
    # 결합 B 대 통합 패널 B 판정(보조)
    for h in hs:
        rB = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == h) & (PAN["타깃"] == "r")]
        gB = NAT[(NAT["모형"] == "ridge_B_Gbar") & (NAT.h == h)].set_index("P")["yhat"]
        G = pf[["region", "P", f"G{h}"]].rename(columns={f"G{h}": "G"})
        d = rB[["region", "P", "yhat"]].merge(G, on=["region", "P"])
        d["yhat"] = d["P"].map(gB) + d["yhat"]
        d = d.rename(columns={"G": "y"}).dropna()
        Lcomb = monthly_loss(d)
        verdict_rows.append(judge_pair("보조 결합 B 대 통합 패널 B", h, Lcomb, pan_loss("ridge_B", h, "G"), s, boot=False))
        verdict_rows.append(judge_pair("보조 결합 B 대 단순 lasth(지역별)", h, Lcomb, pan_loss("naive_lasth", h, "G"), s, boot=False))
    pd.DataFrame(verdict_rows).to_csv(os.path.join(out, "stage6_판정.csv"), index=False, encoding="utf-8-sig")

    # ---- 급락: 연도별 지표, 층별, 국면 시작, 급등
    ev_rows, strata_rows, ep_rows = [], [], []
    for h in hs:
        for m in sorted(PAN.loc[PAN["타깃"] == "down", "모형"].unique()):
            d = PAN[(PAN["모형"] == m) & (PAN.h == h) & (PAN["타깃"] == "down") & PAN.P.isin(origins)]
            if d.empty:
                continue
            cut = d["훈련사건비율"].values
            for yr, g in list(d.groupby(d.P.dt.year)) + [("전체", d)]:
                em = EV.event_metrics(g.y, g.yhat, g["훈련사건비율"].values, p_ref=g["훈련사건비율"].values)
                ev_rows.append({"모형": m, "h": h, "연도": yr, **em})
            if "기존급락상태" in d.columns:
                for st, g in d.groupby("기존급락상태"):
                    name = "기존 급락 상태" if st == 1 else "비급락 상태"
                    em = EV.event_metrics(g.y, g.yhat, g["훈련사건비율"].values, p_ref=g["훈련사건비율"].values)
                    strata_rows.append({"모형": m, "h": h, "층": name, "유효월수": int(g.P.nunique()), **em})
        # 층별 B−A 손실 차이
        dB = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == h) & (PAN["타깃"] == "down")].set_index(["region", "P"])
        dA = PAN[(PAN["모형"] == "ridge_A") & (PAN.h == h) & (PAN["타깃"] == "down")].set_index(["region", "P"])
        j = dB[["y", "yhat", "기존급락상태"]].join(dA[["yhat"]], rsuffix="_A").dropna()
        for st, g in j.groupby("기존급락상태"):
            dd = (g.yhat - g.y) ** 2 - (g.yhat_A - g.y) ** 2
            strata_rows.append({"모형": "B−A Brier 차이", "h": h, "층": "기존 급락 상태" if st == 1 else "비급락 상태", "행수": int(len(g)),
                                "유효월수": int(g.index.get_level_values("P").nunique()), "Brier": float(dd.mean()),
                                **{f"{y}": float(v) for y, v in dd.groupby(g.index.get_level_values("P").year).mean().items()}})
        # 국면 시작 사례 (B 모형 확률, 시작 전 1~6개월)
        quiet = s["events"]["episode_quiet_months"]
        probs = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == h) & (PAN["타깃"] == "down")].set_index(["region", "P"])
        for r in s["inputs"]["regions"]:
            y = pt[pt.region == r].set_index("P")[f"down{h}"]
            starts = T.episode_starts(y, quiet)
            for t in y.index[starts.values]:
                if t < origins[0] or t > origins[-1]:
                    continue
                pre = [(t - k) for k in range(1, 7)]
                pr = [probs["yhat"].get((r, p_), np.nan) for p_ in pre]
                cut = [probs["훈련사건비율"].get((r, p_), np.nan) for p_ in pre]
                hits = [(p_ >= c_) if np.isfinite(p_) and np.isfinite(c_) else False for p_, c_ in zip(pr, cut)]
                lead = next((k + 1 for k, hh in enumerate(hits) if hh), None)
                ep_rows.append({"h": h, "시도": r, "국면 시작 결정월": str(t), **{f"p(t-{k + 1})": (round(p_, 3) if np.isfinite(p_) else np.nan) for k, p_ in enumerate(pr)},
                                "컷오프(훈련 사건비율)": round(float(np.nanmean(cut)), 3) if np.isfinite(np.nanmean(cut)) else np.nan,
                                "시작 전 6개월 내 경보": any(hits), "가장 가까운 경보 선행개월": lead})
        # 정상기 경보 빈도 (비사건 행에서 p >= 컷오프 비율)
        d = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == h) & (PAN["타깃"] == "down") & PAN.P.isin(origins)]
        fa = float(((d.y == 0) & (d.yhat >= d["훈련사건비율"])).sum() / max(1, (d.y == 0).sum()))
        ep_rows.append({"h": h, "시도": "(전체)", "국면 시작 결정월": "정상기 경보 빈도(비사건 행 중 p>=컷오프)", "컷오프(훈련 사건비율)": fa})
    pd.DataFrame(ev_rows).to_csv(os.path.join(out, "stage6_급락_지표.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(strata_rows).to_csv(os.path.join(out, "stage6_급락_층별.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(ep_rows).to_csv(os.path.join(out, "stage6_급락_국면시작.csv"), index=False, encoding="utf-8-sig")
    if not QUICK:
        up_rows = []
        for h in hs:
            for m in ("logit_A_급등", "logit_B_급등"):
                d = PAN[(PAN["모형"] == m) & (PAN.h == h) & (PAN["타깃"] == "up")]
                if d.empty:
                    continue
                em = EV.event_metrics(d.y, d.yhat, d["훈련사건비율"].values, p_ref=d["훈련사건비율"].values)
                up_rows.append({"모형": m, "h": h, "평가 시작": str(d.P.min()), "월수": int(d.P.nunique()), **em})
        pd.DataFrame(up_rows).to_csv(os.path.join(out, "stage6_급등_보조.csv"), index=False, encoding="utf-8-sig")

    # ======================================================== 그림
    REG = [(n, pd.Period(a, "M"), pd.Period(b_, "M")) for n, a, b_ in
           (("2016~2019 보합·지방 하락", "2018-01", "2019-12"), ("2020~2021 상승", "2020-01", "2021-12"),
            ("2022 하반기~2023 하락", "2022-07", "2023-12"), ("2024~2026 재상승", "2024-01", "2025-12"))]
    # 6-1 전국 예측 대 실제 (h=3, h=6)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
    for ax, h in zip(axes, (3, 6)):
        y = NAT[(NAT["모형"] == "ridge_A") & (NAT.h == h) & NAT.P.isin(origins)].set_index("P")["y"]
        ax.plot(y.index.to_timestamp(), y.values, color=V.INK, lw=1.4, label="실제 G")
        for i, (m, lab) in enumerate((("ridge_B", "B_공통"), ("ridge_A", "A"), ("naive_selected", "선택형 단순기준"))):
            p = NAT[(NAT["모형"] == m) & (NAT.h == h) & NAT.P.isin(origins)].set_index("P")["yhat"]
            ax.plot(p.index.to_timestamp(), p.values, color=V.SERIES[i], lw=1.6, label=lab)
        ax.axhline(0, color=V.BASE, lw=0.8)
        ax.set_title(f"전국 월세 다음 {h}회 공표분 누적 변화 G(t,{h}) (%): 실제와 예측, 결정월 2018.01~2025.12")
        ax.legend(loc="lower left", ncol=4)
        V.regime_bands(ax, REG, label=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_1_전국예측.png"))
    plt.close(fig)
    # 6-2 손실 차이 누적합 (RQ1·RQ2·RQ3, h=1·3·6)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (title, fn) in zip(axes, (("RQ1 |e_B| - |e_A| (전국)", lambda h: nat_loss("ridge_B", h) - nat_loss("ridge_A", h)),
                                      ("RQ2 월내 상대 MAE: B_지역 - A_지역", lambda h: pan_loss("ridge_B", h, "r") - pan_loss("ridge_A", h, "r")),
                                      ("RQ3 Brier: B - A", lambda h: pan_loss("ridge_B", h, "down", "sq") - pan_loss("ridge_A", h, "down", "sq")))):
        for i, h in enumerate(hs):
            d = fn(h).dropna().cumsum()
            ax.plot(d.index.to_timestamp(), d.values, color=V.SERIES[i], label=f"h={h}")
        ax.axhline(0, color=V.BASE, lw=0.8)
        ax.set_title(title + "  누적합(음수=추가정보 우세)", fontsize=9.5)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_2_손실차이누적.png"))
    plt.close(fig)
    # 6-3 연도별 MAE (RQ1, h=3) 와 급락 확률 사례 (2022H2~2023, h=3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    h = 3
    yrs = sorted(set(origins.year))
    xs = np.arange(len(yrs))
    w = 0.26
    for i, (m, lab) in enumerate((("ridge_B", "B_공통"), ("ridge_A", "A"), ("naive_selected", "선택형 단순기준"))):
        L = nat_loss(m, h)
        vals = [L[L.index.year == y_].mean() for y_ in yrs]
        ax.bar(xs + (i - 1) * (w + 0.01), vals, width=w, color=V.SERIES[i], label=lab)
    ax.set_xticks(xs, [str(y_) for y_ in yrs])
    ax.set_title("RQ1 연도별 MAE (h=3, %p)", fontsize=9.5)
    ax.legend(loc="upper left")
    ax = axes[1]
    d = PAN[(PAN["모형"] == "ridge_B") & (PAN.h == 3) & (PAN["타깃"] == "down")]
    for i, r in enumerate(("대구", "세종", "서울")):
        g = d[d.region == r].set_index("P").sort_index()
        ax.plot(g.index.to_timestamp(), g.yhat.values, color=V.SERIES[i], label=f"{r} 급락 확률(B)")
        ev = g[g.y == 1]
        ax.scatter(ev.index.to_timestamp(), np.full(len(ev), -0.03 - 0.03 * i), s=9, color=V.SERIES[i], marker="|")
    cut = d.groupby("P")["훈련사건비율"].first()
    ax.plot(cut.index.to_timestamp(), cut.values, color=V.GRAY, lw=1.2, label="컷오프(훈련 사건비율)")
    ax.set_ylim(-0.12, 1.0)
    ax.set_title("급락 확률 경로 (h=3, 로지스틱 B). 아래 눈금은 실제 급락 결정월", fontsize=9.5)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    V.regime_bands(ax, REG, label=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_3_연도별_급락사례.png"))
    plt.close(fig)

    S.manifest(s, {"stage": 6, "quick": QUICK, "origins": f"{origins[0]}~{origins[-1]}", "n_pred_nat": len(NAT), "n_pred_panel": len(PAN)}).to_csv(
        os.path.join(out, "stage6_manifest.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(summ).to_string(index=False))
    cols = ["비교", "h", "mean", "annual_lo", "annual_hi", "biennial_lo", "biennial_hi", "years_better", "판정", "MAE비율(평균의 비)"]
    print(VER[cols].to_string(index=False))
    print(COMB.to_string(index=False))
    print(pd.DataFrame(extra).to_string(index=False))
    log("완료")


if __name__ == "__main__":
    main()
