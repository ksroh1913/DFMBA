# -*- coding: utf-8 -*-
"""
[9차 7단계] 해석·보조 분석 (docs/연구계획_9차_보완설계안.md 7장). RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행.

  A. 동인 제거: 보정모형 M2(전국·패널)에서 동인 ①~⑥ 을 하나씩 빼고 재적합(alpha 는 주 실행에서 고른 값의 중앙값으로 고정).
     실거래 제거는 6단계의 M3 vs M2 가 담당.
  B. 신고제 이후 수준형 비교(사후·기술 통계): 2022.07 이후 로짓 월세 비중·로그 전월세 회전율(3개월합)의 월내 편차와 r_h 의
     풀링 Spearman, 월내 순위상관 평균, 3분위별 후속 r6 평균.
  C. 서울 25개 구 보조: 목표 r(구 상대 변화), A·B·B+실거래, 고정 alpha 170. 월내 상대 MAE 와 월내 Spearman.
  D. TH V11 같은 행 비교: 17개 시도 h=6, 2021~2025, (region, month) 키. TH 네 모형·기준선 vs 우리 패널 M0·M2·M3, 8차 통합 패널 B.
출력 (rmpi/output_v9/): v9_s7_동인제거.csv, v9_s7_신고제이후_수준형.csv, v9_s7_서울구.csv, v9_s7_서울구_예측값.csv,
                        v9_s7_TH비교.csv, v9_s7_TH비교_연도별.csv, v9_stage7_manifest.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import engine as E  # noqa: E402
from rmpi import frames as F  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402
from rmpi.run_v9_stage6 import monthly_loss  # noqa: E402

TH_PRED = "/home/user/ksroh1913/dfmba_TH/ml/experiments_TH_v1/TH_v11/analysis/output_TH_v11/predictions_TH_v11.csv"
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    hs = s["timing"]["horizons"]
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_end"]), freq="M")
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    pf = F.panel_frame(s, b, pt, nat)
    log("자료 준비 완료")

    # ============================================================ A. 동인 제거 (M2, alpha 고정 = 주 실행 중앙값)
    alog = pd.read_csv(os.path.join(out, "v9_s2_alpha선택.csv"))
    pred2 = pd.read_csv(os.path.join(out, "v9_s2_예측값.csv")); pred2["P"] = pd.PeriodIndex(pred2["P"], freq="M")
    rows = []
    for unit, panel in (("전국", False), ("패널", True)):
        for h in hs:
            a_med = float(alog[(alog["단위"] == unit) & (alog["모형"] == "M2") & (alog["h"] == h)]["선택"].median())
            base = pred2[(pred2["단위"] == unit) & (pred2["h"] == h) & (pred2["모형"] == "corr_M2") & pred2["P"].isin(origins)]
            L0 = monthly_loss(base).mean()
            frame = pf if panel else nf
            full, _, _ = E.correction_run(s, frame, spec, h, origins, model="M2", panel=panel, alpha_rule="fixed", alpha_value=a_med, label="M2_fixed")
            Lf = monthly_loss(full).mean()
            rows.append({"단위": unit, "h": h, "뺀 동인": "(없음, alpha 고정 재실행)", "alpha": a_med, "MAE": Lf, "MAE_주실행": L0, "손실변화(%)": 100 * (Lf / L0 - 1)})
            for d in sorted({int(x) for x in spec.loc[spec.block.isin(["regional", "common"]), "동인"]}):
                ids = spec.loc[(spec["동인"] == d) & spec.block.isin(["regional", "common"]), "ID"].unique().tolist()
                p, _, _ = E.correction_run(s, frame, spec, h, origins, model="M2", panel=panel, alpha_rule="fixed", alpha_value=a_med, exclude_ids=ids, label=f"M2_drop{d}")
                L = monthly_loss(p).mean()
                rows.append({"단위": unit, "h": h, "뺀 동인": s["driver_names"][d], "alpha": a_med, "MAE": L, "MAE_주실행": L0, "손실변화(%)": 100 * (L / Lf - 1)})
            log(f"A 동인 제거 {unit} h={h}")
    dr = pd.DataFrame(rows)
    dr["손실변화_기준"] = np.where(dr["뺀 동인"].astype(str).str.startswith("(없음"), "주 실행(cv_min alpha) 대비", "alpha 고정 재실행 대비")
    dr.to_csv(os.path.join(out, "v9_s7_동인제거.csv"), index=False, encoding="utf-8-sig")

    # ============================================================ B. 신고제 이후 수준형 (사후·기술 통계)
    rc = s["rt_columns"]
    g = ml["region"]
    hh = ml[s["inputs"]["households_col"]].astype(float)
    rent = ml[rc["월세"]].astype(float).groupby(g).transform(lambda v: v.rolling(3, min_periods=3).sum())
    tot = ml[rc["전체"]].astype(float).groupby(g).transform(lambda v: v.rolling(3, min_periods=3).sum())
    lv = pd.DataFrame({"region": g, "P": ml["P"], "logit_share": np.log(rent) - np.log((tot - rent).where(tot > rent)),
                       "log_turnover": np.log(1000 * tot / hh.groupby(g).transform(lambda v: v.rolling(3, min_periods=3).mean()))})
    start = S.per(s["extra_inputs"]["sensitivities"]["post_break_levels"]["from"])
    asof = S.per(s["inputs"]["data_asof"])
    lv = lv[(lv.P >= start) & (lv.P < asof - 1)]
    for c in ("logit_share", "log_turnover"):
        lv[c + "_dev"] = lv[c] - lv.groupby("P")[c].transform("mean")
    rows = []
    for h in hs:
        m = lv.merge(pt[["region", "P", f"r{h}"]], on=["region", "P"])
        m = m[m[f"r{h}"].notna()]
        for c in ("logit_share_dev", "log_turnover_dev"):
            rho_pool = spearmanr(m[c], m[f"r{h}"]).statistic if len(m) > 20 else np.nan
            wm = [spearmanr(gg[c], gg[f"r{h}"]).statistic for _, gg in m.groupby("P") if len(gg) >= 10]
            terc = m.copy()
            terc["분위"] = terc.groupby("P")[c].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["하", "중", "상"]))
            tm = terc.groupby("분위", observed=True)[f"r{h}"].mean()
            rows.append({"h": h, "수준형": c, "기간": f"{m.P.min()}~{m.P.max()}", "행수": len(m), "풀링 Spearman": rho_pool,
                         "월내 순위상관 평균": float(np.nanmean(wm)) if wm else np.nan, "월내 SD": float(np.nanstd(wm, ddof=1)) if len(wm) > 1 else np.nan,
                         "유효월": len(wm), "후속 r 평균_하": tm.get("하", np.nan), "후속 r 평균_중": tm.get("중", np.nan), "후속 r 평균_상": tm.get("상", np.nan)})
    pd.DataFrame(rows).to_csv(os.path.join(out, "v9_s7_신고제이후_수준형.csv"), index=False, encoding="utf-8-sig")
    log("B 신고제 이후 수준형 완료")

    # ============================================================ C. 서울 25개 구 보조
    bg = D.build(s, "gu")
    spec_g, ml_g = bg["spec"], bg["ml"]
    pg = T.regional(s, ml_g, "gu")
    pfg = F.panel_frame(s, bg, pg, nat)
    gu_rows, gu_preds = [], []
    for h in hs:
        res = {}
        for info, extra in (("A", False), ("B", False), ("B+RT", True)):
            p, c, _ = E.panel_run(s, pfg, spec_g, h, origins, target="r", info=("B" if extra else info), extra=extra, label=f"gu_{info}")
            res[info] = p; gu_preds.append(p)
        res["zero"] = E.panel_naive(s, pfg, h, origins, target="r")
        for k, p in res.items():
            L = monthly_loss(p).mean()
            sp_ = [spearmanr(gg.yhat, gg.y).statistic for _, gg in p.dropna(subset=["y", "yhat"]).groupby("P") if len(gg) >= 10 and gg.yhat.nunique() > 1]
            gu_rows.append({"h": h, "모형": k, "월내 상대 MAE": L, "월내 Spearman(평균)": float(np.nanmean(sp_)) if sp_ else np.nan, "Spearman 유효월": len(sp_)})
        log(f"C 서울 구 h={h}")
    pd.DataFrame(gu_rows).to_csv(os.path.join(out, "v9_s7_서울구.csv"), index=False, encoding="utf-8-sig")
    pd.concat(gu_preds, ignore_index=True).to_csv(os.path.join(out, "v9_s7_서울구_예측값.csv"), index=False, encoding="utf-8-sig")

    # ============================================================ D. TH V11 같은 행 비교 (h=6, 2021~2025)
    rows, yrows = [], []
    if os.path.exists(TH_PRED):
        th = pd.read_csv(TH_PRED)
        th = th[(th["panel"] == "provinces") & (th["task"] == "Y1 Growth")].copy()
        th["P"] = pd.PeriodIndex([f"{m // 100}-{m % 100:02d}" for m in th["month"].astype(int)], freq="M")
        th = th[(th.P >= "2021-01") & (th.P <= "2025-12")]
        ours = pred2[(pred2["단위"] == "패널") & (pred2["h"] == 6) & pred2["모형"].isin(["corr_M0", "corr_M1", "corr_M2", "corr_M3"])].copy()
        old = pd.read_csv(os.path.join(BASE, "rmpi/output/stage6_예측값_패널.csv"))
        old = old[(old["타깃"] == "G") & (old["h"] == 6) & (old["모형"] == "ridge_B")].copy() if "타깃" in old.columns else pd.DataFrame()
        if len(old):
            old["P"] = pd.PeriodIndex(old["P"], freq="M"); old["모형"] = "8차_통합패널_ridge_B"
        cand = pd.concat([ours[["region", "P", "모형", "y", "yhat"]], old[["region", "P", "모형", "y", "yhat"]] if len(old) else None], ignore_index=True)
        keys = th[["region", "P"]].drop_duplicates()
        chk = th[th["model"] == "Linear"].merge(cand[cand["모형"] == "corr_M0"][["region", "P", "y"]], on=["region", "P"])
        rows.append({"모형": "(정답 일치 점검)", "출처": "점검", "행수": len(chk), "MAE": float((chk["actual"] - chk["y"]).abs().max()), "비고": "TH actual − 우리 y 최대 절대 차"})
        common = keys.merge(cand[cand["모형"] == "corr_M0"][["region", "P"]], on=["region", "P"])
        for mdl, gg in th.groupby("model"):
            q = gg.merge(common, on=["region", "P"])
            rows.append({"모형": mdl, "출처": "TH V11", "행수": len(q), "MAE": float((q["actual"] - q["prediction"]).abs().mean()), "비고": "분기 재학습, 훈련 2015.07~"})
            q = q.assign(year=q.P.dt.year, err=(q["actual"] - q["prediction"]).abs())
            for y_, v in q.groupby("year")["err"].mean().items():
                yrows.append({"모형": mdl, "출처": "TH V11", "연도": y_, "MAE": v})
        for mdl, gg in cand.groupby("모형"):
            q = gg.merge(common, on=["region", "P"]).dropna(subset=["y", "yhat"])
            rows.append({"모형": mdl, "출처": "우리(9차 통합 패널)" if mdl.startswith("corr") else "우리(8차)", "행수": len(q), "MAE": float((q["y"] - q["yhat"]).abs().mean()),
                         "비고": "월별 재학습, 훈련 2016.01~"})
            q = q.assign(year=q.P.dt.year, err=(q["y"] - q["yhat"]).abs())
            for y_, v in q.groupby("year")["err"].mean().items():
                yrows.append({"모형": mdl, "출처": "우리", "연도": y_, "MAE": v})
        log("D TH 비교 완료")
    else:
        rows.append({"모형": "", "출처": "", "행수": 0, "MAE": np.nan, "비고": f"TH 예측값 파일 없음: {TH_PRED}"})
    pd.DataFrame(rows).to_csv(os.path.join(out, "v9_s7_TH비교.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(yrows).to_csv(os.path.join(out, "v9_s7_TH비교_연도별.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"항목": "settings", "값": f"{s['_path']} {s['_sha']} {s['meta']['settings_version']}"}, {"항목": "commit", "값": S.git_commit()},
                  {"항목": "TH_pred", "값": TH_PRED}, {"항목": "seconds", "값": f"{time.time() - t0:.0f}"}]).to_csv(
        os.path.join(out, "v9_stage7_manifest.csv"), index=False, encoding="utf-8-sig")
    log("완료")


if __name__ == "__main__":
    main()
